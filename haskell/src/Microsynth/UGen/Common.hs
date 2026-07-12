{-# LANGUAGE BangPatterns #-}

-- | Shared building blocks for the UGen DSP kernels.
--
-- The per-UGen builders in "Microsynth.UGen.*" previously hand-rolled the same
-- phase-accumulator arithmetic and the same per-block loop skeleton. This module
-- factors those out so each UGen carries only its own DSP, not the ceremony
-- around it. Everything here is @INLINE@d so the abstractions compile back to
-- the original tight loops with no boxing on the hot path (see the note in
-- "Microsynth.UGen.Filter").
module Microsynth.UGen.Common
  ( wrap01
  , phasorStep
  ) where

-- | Wrap a phase accumulator back into @[0, 1)@. Since the accumulator is always
-- in @[0, 1)@ and audio-rate increments are @< 1@, the value before wrapping is
-- in @[0, 2)@, so a single compare-and-subtract is both correct and much faster
-- than @floor@ on the hot path (GHC's @floor :: Float -> Int@ is not one
-- instruction). Mirrors Rust's @phase -= phase.floor()@ under the same invariant.
wrap01 :: Float -> Float
wrap01 p = if p >= 1 then p - 1 else p
{-# INLINE wrap01 #-}

-- | Advance a phase accumulator by one sample at frequency @f@ (Hz), given the
-- reciprocal sample rate @invSr = 1 / sampleRate@, wrapping into @[0, 1)@. This
-- is the shared step of every phase-accumulator oscillator.
phasorStep :: Float -> Float -> Float -> Float
phasorStep invSr f p = wrap01 (p + f * invSr)
{-# INLINE phasorStep #-}
