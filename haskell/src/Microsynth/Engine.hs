{-# LANGUAGE BangPatterns #-}

-- | The offline render engine.
--
-- Port of the block loop in Rust @src/engine.rs@ (@render@ / @render_offline@).
-- Everything runs in a single 'ST' region over pre-allocated mutable blocks —
-- the direct analogue of Rust's no-allocation render path.
--
-- Each node's inputs and output are bound once at instantiation, so the block
-- loop is just: for each node in topological order, run its captured action;
-- then copy the sink's block into a single preallocated output buffer.
module Microsynth.Engine
  ( renderOffline
  ) where

import Control.Monad (forM_)
import Control.Monad.ST (runST)
import Data.Map.Strict (Map)
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (maxBlockSize, newBlock)
import Microsynth.Context (Context (..))
import Microsynth.Graph (topoSort)
import Microsynth.Node (Node (..))
import Microsynth.SynthDef (NodeDef (..), SynthDef (..))
import Microsynth.Types
  ( BlockSize (..), NodeId (..), ParamName, Sample, SampleCount (..)
  , SampleOffset (..), SampleRate )
import Microsynth.UGen (instantiate)

-- | Render a synth offline to a list of per-channel sample vectors.
--
-- @renderOffline def sampleRate numSamples overrides@. The scaffold's UGens
-- are mono, so the result is a single-element list. Rendering is done in whole
-- 128-sample blocks and then trimmed to exactly @numSamples@.
renderOffline
  :: SynthDef -> SampleRate -> SampleCount -> Map ParamName Sample
  -> [VU.Vector Sample]
renderOffline sdef sr numSamples overrides = runST $ do
  let bszN      = maxBlockSize
      !bsz      = unBlockSize bszN
      !ns       = unSampleCount numSamples
      defs      = V.fromList (sdNodes sdef)
      nnodes    = V.length defs
      ctx       = Context sr bszN (SampleOffset 0)
      edges     = [ (unNodeId j, i)
                  | (i, nd) <- zip [0 ..] (sdNodes sdef)
                  , j <- ndInputs nd
                  ]
      order     = topoSort nnodes edges
      numBlocks = (ns + bsz - 1) `div` bsz

  -- One output block per node, allocated once and reused every block.
  outs <- V.replicateM nnodes (newBlock bszN)

  -- Instantiate each node with its inputs/output already bound.
  nodes <- V.generateM nnodes $ \i -> do
    let nd  = defs V.! i
        ins = [ outs V.! unNodeId j | j <- ndInputs nd ]
    instantiate ctx overrides (ndKind nd) ins (outs V.! i)

  -- Precompute the actual per-block work sequence in topological order.
  let steps = [ runBlock (nodes V.! i) | i <- order ]
      sink  = outs V.! unNodeId (sdOutput sdef)

  big <- VUM.new (numBlocks * bsz)
  forM_ [0 .. numBlocks - 1] $ \ !b -> do
    sequence_ steps
    VUM.copy (VUM.slice (b * bsz) bsz big) sink

  frozen <- VU.freeze big
  pure [VU.take ns frozen]
