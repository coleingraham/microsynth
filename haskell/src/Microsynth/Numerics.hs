-- | Shared numeric constants and conversions for the DSP kernel.
--
-- A dependency-free leaf module so any UGen can share the same constants
-- instead of redefining them locally (previously @tau@ lived, byte-identical,
-- in both "Microsynth.UGen.Oscillator" and "Microsynth.UGen.Filter").
--
-- 'Microsynth.Types.SampleRate' is deliberately /not/ a 'Sample', so that a rate
-- cannot silently enter the audio path. But every rate-dependent UGen legitimately
-- needs to cross that line exactly once, at build time. 'invSampleRate' and
-- 'srSample' are that crossing, named and in one place, rather than an ad-hoc
-- @Sample (unSampleRate sr)@ open-coded per kernel — the unwrap stays explicit
-- (which is the point of the newtype) while the spelling stays single-sourced.
module Microsynth.Numerics
  ( tau
  , invSampleRate
  , srSample
  ) where

import Microsynth.Types (Sample (..), SampleRate (..))

-- | The full-turn angle, @2 * pi@. Used by every phase- and frequency-domain
-- UGen (oscillator phase, biquad @w0@). A 'Sample' so it drops straight into the
-- audio-domain arithmetic.
tau :: Sample
tau = 2 * pi
{-# INLINE tau #-}

-- | The reciprocal sample rate, @1 / sampleRate@, as a 'Sample' — the per-sample
-- time step. Computed once per node at build time so the hot loop multiplies by
-- it instead of dividing (every phase-accumulator oscillator's increment).
invSampleRate :: SampleRate -> Sample
invSampleRate sr = Sample (1 / unSampleRate sr)
{-# INLINE invSampleRate #-}

-- | The sample rate itself as a 'Sample', for kernels that divide /by/ the rate
-- (a biquad's @w0@) or scale a duration in seconds into a sample count (an
-- envelope's ramp).
srSample :: SampleRate -> Sample
srSample = Sample . unSampleRate
{-# INLINE srSample #-}
