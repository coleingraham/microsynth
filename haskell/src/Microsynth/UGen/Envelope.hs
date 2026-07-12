{-# LANGUAGE BangPatterns #-}

-- | Envelope UGens.
--
-- Port of the gateless @Perc@ envelope from Rust @src/ugens/envelopes.rs@:
-- a linear attack ramp to 1.0 then a linear release to 0.0. Level and stage
-- (0 = attack, 1 = release, 2 = done) are read once per block, threaded
-- through the loop, and written back once.
module Microsynth.UGen.Envelope
  ( mkPerc
  ) where

import Control.Monad.ST (ST)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (MBlock)
import Microsynth.Node (Node (..), readInput)
import Microsynth.UGen.Common (bindPort, scanBlockFI)
import Microsynth.UGen.Spec (UGenTag (..))

-- | Percussive envelope. Inputs: attack (s), release (s). Output: @[0, 1]@.
-- Level and stage live in unboxed cells so the loop's threaded accumulators
-- stay unboxed (see the note in "Microsynth.UGen.Filter"). The written output
-- sample is the /post-update/ level, matching the state threaded on.
mkPerc :: Float -> [MBlock s] -> MBlock s -> ST s (Node s)
mkPerc sr ins out = do
  levelV <- VUM.replicate 1 (0 :: Float)  -- unboxed level
  stageV <- VUM.replicate 1 (0 :: Int)    -- unboxed stage (0=atk,1=rel,2=done)
  let (atkIn, dAtk) = bindPort ins TPerc 0
      (relIn, dRel) = bindPort ins TPerc 1
      !n            = VUM.length out
  pure $ Node $ scanBlockFI levelV stageV n $ \i lvl stage -> do
    at <- max 0.0001 <$> readInput atkIn i dAtk
    rt <- max 0.0001 <$> readInput relIn i dRel
    let (lvl', stage') = case stage of
          0 -> let l = lvl + 1 / (at * sr)
               in if l >= 1 then (1, 1) else (l, 0)
          1 -> let l = lvl - 1 / (rt * sr)
               in if l <= 0 then (0, 2) else (l, 1)
          _ -> (0, 2)
    VUM.unsafeWrite out i lvl'
    pure (lvl', stage')
