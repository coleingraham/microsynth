{-# LANGUAGE BangPatterns #-}

-- | Oscillator UGens.
--
-- Direct port of the phase-accumulator oscillators in Rust
-- @src/ugens/oscillators.rs@: phase is kept in @[0, 1)@ and advanced by
-- @freq / sampleRate@ per sample, with per-sample frequency modulation.
module Microsynth.UGen.Oscillator
  ( mkSinOsc
  , mkSaw
  ) where

import Control.Monad.ST (ST)
import Data.STRef (newSTRef, readSTRef, writeSTRef)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Node (Node (..), sampleAt)

tau :: Float
tau = 2 * pi

-- | Wrap a phase back into @[0, 1)@ (Rust: @phase -= phase.floor()@).
wrap01 :: Float -> Float
wrap01 p = p - fromIntegral (floor p :: Int)
{-# INLINE wrap01 #-}

-- | Sine oscillator. Inputs: freq (Hz), phase offset (radians).
-- Output: @sin(2*pi*phase + phaseOffset)@ in @[-1, 1]@.
mkSinOsc :: Float -> ST s (Node s)
mkSinOsc sr = do
  phaseRef <- newSTRef 0
  let !invSr = 1 / sr
  pure $ Node $ \_ ins out -> do
    let !n = VUM.length out
    p0 <- readSTRef phaseRef
    let go !i !p
          | i >= n    = writeSTRef phaseRef p
          | otherwise = do
              f  <- sampleAt ins 0 i 440
              ph <- sampleAt ins 1 i 0
              VUM.write out i (sin (p * tau + ph))
              go (i + 1) (wrap01 (p + f * invSr))
    go 0 p0

-- | Naive sawtooth. Input: freq (Hz). Output: @2*phase - 1@ in @[-1, 1)@.
mkSaw :: Float -> ST s (Node s)
mkSaw sr = do
  phaseRef <- newSTRef 0
  let !invSr = 1 / sr
  pure $ Node $ \_ ins out -> do
    let !n = VUM.length out
    p0 <- readSTRef phaseRef
    let go !i !p
          | i >= n    = writeSTRef phaseRef p
          | otherwise = do
              f <- sampleAt ins 0 i 440
              VUM.write out i (2 * p - 1)
              go (i + 1) (wrap01 (p + f * invSr))
    go 0 p0
