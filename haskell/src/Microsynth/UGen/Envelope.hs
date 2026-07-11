{-# LANGUAGE BangPatterns #-}

-- | Envelope UGens.
--
-- Port of the gateless @Perc@ envelope from Rust @src/ugens/envelopes.rs@:
-- a linear attack ramp to 1.0 followed by a linear release to 0.0. Stage is
-- encoded as 0 = attack, 1 = release, 2 = done.
module Microsynth.UGen.Envelope
  ( mkPerc
  ) where

import Control.Monad.ST (ST)
import Data.STRef (newSTRef, readSTRef, writeSTRef)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Node (Node (..), sampleAt)

-- | Percussive envelope. Inputs: attack (s), release (s). Output: @[0, 1]@.
mkPerc :: Float -> ST s (Node s)
mkPerc sr = do
  levelRef <- newSTRef 0
  stageRef <- newSTRef (0 :: Int)
  pure $ Node $ \_ ins out -> do
    let !n = VUM.length out
        go !i
          | i >= n    = pure ()
          | otherwise = do
              at <- max 0.0001 <$> sampleAt ins 0 i 0.001
              rt <- max 0.0001 <$> sampleAt ins 1 i 0.1
              lvl <- readSTRef levelRef
              st  <- readSTRef stageRef
              let (lvl', st') = case st of
                    0 -> let l = lvl + 1 / (at * sr)
                         in if l >= 1 then (1, 1) else (l, 0)
                    1 -> let l = lvl - 1 / (rt * sr)
                         in if l <= 0 then (0, 2) else (l, 1)
                    _ -> (0, 2)
              writeSTRef levelRef lvl'
              writeSTRef stageRef st'
              VUM.write out i lvl'
              go (i + 1)
    go 0
