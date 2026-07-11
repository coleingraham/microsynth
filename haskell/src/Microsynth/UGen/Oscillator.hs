{-# LANGUAGE BangPatterns #-}

-- | Oscillator UGens.
--
-- Direct port of the phase-accumulator oscillators in Rust
-- @src/ugens/oscillators.rs@: phase is kept in @[0, 1)@ and advanced by
-- @freq / sampleRate@ per sample. Phase is read from its 'STRef' once per
-- block, threaded through the inner loop as an unboxed argument, and written
-- back once — so there is no per-sample reference traffic.
module Microsynth.UGen.Oscillator
  ( mkSinOsc
  , mkSaw
  ) where

import Control.Monad.ST (ST)
import Data.STRef (newSTRef, readSTRef, writeSTRef)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (MBlock)
import Microsynth.Node (Node (..), bindInput, readInput)

tau :: Float
tau = 2 * pi

-- | Wrap a phase back into @[0, 1)@. Rust uses @phase -= phase.floor()@, but
-- GHC's @floor :: Float -> Int@ is not a single machine instruction. Since the
-- accumulator is always in @[0, 1)@ and audio-rate increments are @< 1@, the
-- phase before wrapping is in @[0, 2)@, so a single compare-and-subtract is
-- both correct and much faster on the hot path.
wrap01 :: Float -> Float
wrap01 p = if p >= 1 then p - 1 else p
{-# INLINE wrap01 #-}

-- | Sine oscillator. Inputs: freq (Hz), phase offset (radians).
mkSinOsc :: Float -> [MBlock s] -> MBlock s -> ST s (Node s)
mkSinOsc sr ins out = do
  phaseRef <- newSTRef 0
  let !invSr = 1 / sr
      freqIn = bindInput ins 0
      phIn   = bindInput ins 1
      !n     = VUM.length out
  pure $ Node $ do
    p0 <- readSTRef phaseRef
    let go !i !p
          | i >= n    = writeSTRef phaseRef p
          | otherwise = do
              f  <- readInput freqIn i 440
              ph <- readInput phIn i 0
              VUM.unsafeWrite out i (sin (p * tau + ph))
              go (i + 1) (wrap01 (p + f * invSr))
    go 0 p0

-- | Naive sawtooth. Input: freq (Hz). Output: @2*phase - 1@ in @[-1, 1)@.
mkSaw :: Float -> [MBlock s] -> MBlock s -> ST s (Node s)
mkSaw sr ins out = do
  phaseRef <- newSTRef 0
  let !invSr = 1 / sr
      freqIn = bindInput ins 0
      !n     = VUM.length out
  pure $ Node $ do
    p0 <- readSTRef phaseRef
    let go !i !p
          | i >= n    = writeSTRef phaseRef p
          | otherwise = do
              f <- readInput freqIn i 440
              VUM.unsafeWrite out i (2 * p - 1)
              go (i + 1) (wrap01 (p + f * invSr))
    go 0 p0
