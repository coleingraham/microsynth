{-# LANGUAGE BangPatterns #-}

-- | Oscillator UGens.
--
-- Direct port of the phase-accumulator oscillators in Rust
-- @src/ugens/oscillators.rs@: phase is kept in @[0, 1)@ and advanced by
-- @freq / sampleRate@ per sample. Phase lives in a single unboxed cell, read
-- once per block and threaded through the loop as a raw @Float#@; keeping it
-- unboxed (rather than a boxed @STRef Float@) is what lets GHC avoid boxing the
-- accumulator every sample.
module Microsynth.UGen.Oscillator
  ( mkSinOsc
  , mkSaw
  ) where

import Control.Monad.ST (ST)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (MBlock)
import Microsynth.Node (Node (..), bindInput, readInput)
import Microsynth.Numerics (tau)
import Microsynth.UGen.Common (phasorStep, scanBlock1F)

-- | Sine oscillator. Inputs: freq (Hz), phase offset (radians).
mkSinOsc :: Float -> [MBlock s] -> MBlock s -> ST s (Node s)
mkSinOsc sr ins out = do
  phase <- VUM.replicate 1 0  -- unboxed phase accumulator
  let !invSr = 1 / sr
      freqIn = bindInput ins 0
      phIn   = bindInput ins 1
      !n     = VUM.length out
  pure $ Node $ scanBlock1F phase n $ \i p -> do
    f  <- readInput freqIn i 440
    ph <- readInput phIn i 0
    VUM.unsafeWrite out i (sin (p * tau + ph))
    pure (phasorStep invSr f p)

-- | Naive sawtooth. Input: freq (Hz). Output: @2*phase - 1@ in @[-1, 1)@.
mkSaw :: Float -> [MBlock s] -> MBlock s -> ST s (Node s)
mkSaw sr ins out = do
  phase <- VUM.replicate 1 0  -- unboxed phase accumulator
  let !invSr = 1 / sr
      freqIn = bindInput ins 0
      !n     = VUM.length out
  pure $ Node $ scanBlock1F phase n $ \i p -> do
    f <- readInput freqIn i 440
    VUM.unsafeWrite out i (2 * p - 1)
    pure (phasorStep invSr f p)
