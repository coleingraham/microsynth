-- | Shared numeric constants for the DSP kernel.
--
-- A dependency-free leaf module so any UGen can share the same constants
-- instead of redefining them locally (previously @tau@ lived, byte-identical,
-- in both "Microsynth.UGen.Oscillator" and "Microsynth.UGen.Filter").
module Microsynth.Numerics
  ( tau
  ) where

import Microsynth.Types (Sample)

-- | The full-turn angle, @2 * pi@. Used by every phase- and frequency-domain
-- UGen (oscillator phase, biquad @w0@). A 'Sample' so it drops straight into the
-- audio-domain arithmetic.
tau :: Sample
tau = 2 * pi
{-# INLINE tau #-}
