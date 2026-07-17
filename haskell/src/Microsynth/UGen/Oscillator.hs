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
import Microsynth.Node (Node, readInput)
import Microsynth.Numerics (tau)
import Microsynth.Types (SampleRate)
import Microsynth.UGen.Common (bindPort, mkPhasorOsc)
import Microsynth.UGen.Spec (UGenTag (..))

-- | Sine oscillator. Inputs: freq (Hz), phase offset (radians). The phase-offset
-- port is bound here and read per sample; freq, the phase cell and the
-- accumulator loop are 'mkPhasorOsc'.
mkSinOsc :: SampleRate -> [MBlock s] -> MBlock s -> ST s (Node s)
mkSinOsc sr ins out =
  let (phIn, dP) = bindPort ins TSinOsc 1
  in mkPhasorOsc sr ins out TSinOsc $ \i p -> do
       ph <- readInput phIn i dP
       VUM.unsafeWrite out i (sin (p * tau + ph))

-- | Naive sawtooth. Input: freq (Hz). Output: @2*phase - 1@ in @[-1, 1)@ — the
-- raw phase ramp, so the UGen is nothing but its output expression.
mkSaw :: SampleRate -> [MBlock s] -> MBlock s -> ST s (Node s)
mkSaw sr ins out =
  mkPhasorOsc sr ins out TSaw $ \i p ->
    VUM.unsafeWrite out i (2 * p - 1)
