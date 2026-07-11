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
import Data.STRef (newSTRef, readSTRef, writeSTRef)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (MBlock)
import Microsynth.Node (Node (..), bindInput, readInput)

-- | Percussive envelope. Inputs: attack (s), release (s). Output: @[0, 1]@.
mkPerc :: Float -> [MBlock s] -> MBlock s -> ST s (Node s)
mkPerc sr ins out = do
  levelRef <- newSTRef 0
  stageRef <- newSTRef (0 :: Int)
  let atkIn = bindInput ins 0
      relIn = bindInput ins 1
      !n    = VUM.length out
  pure $ Node $ do
    l0 <- readSTRef levelRef
    g0 <- readSTRef stageRef
    let step !i !lvl !stage = VUM.unsafeWrite out i lvl >> go (i + 1) lvl stage
        go !i !lvl !stage
          | i >= n    = writeSTRef levelRef lvl >> writeSTRef stageRef stage
          | otherwise = do
              at <- max 0.0001 <$> readInput atkIn i 0.001
              rt <- max 0.0001 <$> readInput relIn i 0.1
              case stage of
                0 -> let l = lvl + 1 / (at * sr)
                     in if l >= 1 then step i 1 1 else step i l 0
                1 -> let l = lvl - 1 / (rt * sr)
                     in if l <= 0 then step i 0 2 else step i l 1
                _ -> step i 0 2
    go 0 l0 g0
