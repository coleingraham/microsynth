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
import Microsynth.Node (Node (..), readInput)
import Microsynth.Numerics (tau)
import Microsynth.Types (Sample (..), SampleRate (..))
import Microsynth.UGen.Common (bindPort, phasorStep, scanBlock1F)
import Microsynth.UGen.Spec (UGenTag (..))

-- | Sine oscillator. Inputs: freq (Hz), phase offset (radians).
mkSinOsc :: SampleRate -> [MBlock s] -> MBlock s -> ST s (Node s)
mkSinOsc sr ins out = do
  phase <- VUM.replicate 1 0  -- unboxed phase accumulator
  let !invSr        = Sample (1 / unSampleRate sr)
      (freqIn, dF)  = bindPort ins TSinOsc 0
      (phIn,   dP)  = bindPort ins TSinOsc 1
      !n            = VUM.length out
  pure $ Node $ scanBlock1F phase n $ \i p -> do
    f  <- readInput freqIn i dF
    ph <- readInput phIn i dP
    VUM.unsafeWrite out i (sin (p * tau + ph))
    pure (phasorStep invSr f p)

-- | Naive sawtooth. Input: freq (Hz). Output: @2*phase - 1@ in @[-1, 1)@.
mkSaw :: SampleRate -> [MBlock s] -> MBlock s -> ST s (Node s)
mkSaw sr ins out = do
  phase <- VUM.replicate 1 0  -- unboxed phase accumulator
  let !invSr       = Sample (1 / unSampleRate sr)
      (freqIn, dF) = bindPort ins TSaw 0
      !n           = VUM.length out
  pure $ Node $ scanBlock1F phase n $ \i p -> do
    f <- readInput freqIn i dF
    VUM.unsafeWrite out i (2 * p - 1)
    pure (phasorStep invSr f p)
