{-# LANGUAGE BangPatterns #-}

-- | The offline render engine.
--
-- Port of the block loop in Rust @src/engine.rs@ (@render@ / @render_offline@).
-- Everything runs in a single 'ST' region over pre-allocated mutable blocks —
-- the direct analogue of Rust's no-allocation render path. Node outputs are
-- stored in a per-node vector and read back by consumers in topological order.
module Microsynth.Engine
  ( renderOffline
  ) where

import Control.Monad (forM_)
import Control.Monad.ST (runST)
import Data.Map.Strict (Map)
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as VU

import Microsynth.Buffer (maxBlockSize, newBlock)
import Microsynth.Context (Context (..))
import Microsynth.Graph (topoSort)
import Microsynth.Node (Node (..))
import Microsynth.SynthDef (NodeDef (..), SynthDef (..))
import Microsynth.UGen (instantiate)

-- | Render a synth offline to a list of per-channel sample vectors.
--
-- @renderOffline def sampleRate numSamples overrides@. The scaffold's UGens
-- are mono, so the result is a single-element list. Rendering is done in whole
-- 128-sample blocks and then trimmed to exactly @numSamples@.
renderOffline :: SynthDef -> Float -> Int -> Map String Float -> [VU.Vector Float]
renderOffline sdef sr numSamples overrides = runST $ do
  let !bsz      = maxBlockSize
      defs      = V.fromList (sdNodes sdef)
      nnodes    = V.length defs
      ctx       = Context sr bsz 0
      -- dependency edges: an input node must run before its consumer
      edges     = [ (j, i)
                  | (i, nd) <- zip [0 ..] (sdNodes sdef)
                  , j <- ndInputs nd
                  ]
      order     = topoSort nnodes edges
      numBlocks = (numSamples + bsz - 1) `div` bsz

  nodes <- V.generateM nnodes (\i -> instantiate ctx overrides (ndKind (defs V.! i)))
  outs  <- V.replicateM nnodes (newBlock bsz)

  let renderBlock = do
        forM_ order $ \i -> do
          let ins = [ outs V.! j | j <- ndInputs (defs V.! i) ]
          runNode (nodes V.! i) ctx ins (outs V.! i)
        VU.freeze (outs V.! sdOutput sdef)

  chunks <- mapM (const renderBlock) [1 .. numBlocks]
  let full = VU.take numSamples (VU.concat chunks)
  pure [full]
