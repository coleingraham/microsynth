{-# LANGUAGE BangPatterns #-}

-- | Shared building blocks for the UGen DSP kernels.
--
-- The per-UGen builders in "Microsynth.UGen.*" previously hand-rolled the same
-- phase-accumulator arithmetic and the same per-block loop skeleton. This module
-- factors those out so each UGen carries only its own DSP, not the ceremony
-- around it. Everything here is @INLINE@d so the abstractions compile back to
-- the original tight loops with no boxing on the hot path.
--
-- The stateful scanners thread their accumulators as /separate, bang-patterned
-- scalar arguments/ (never a single boxed tuple), which is what lets GHC keep
-- the loop's state as raw @Float#@\/@Int#@ — the hard-won property described in
-- "Microsynth.UGen.Filter". The step functions write the output block
-- themselves (output expressions are UGen-specific) and return only the next
-- state; the tiny result tuples are consumed immediately by the inlined loop, so
-- the simplifier cancels them.
module Microsynth.UGen.Common
  ( wrap01
  , phasorStep
  , mkPhasorOsc
  , bindPort
  , mapBlock
  , scanBlock1F
  , scanBlock2F
  , scanBlockFI
  ) where

import Control.Monad.ST (ST)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (MBlock)
import Microsynth.Node (Input, Node (..), bindInput, readInput)
import Microsynth.Numerics (invSampleRate)
import Microsynth.Types (Sample, SampleRate)
import Microsynth.UGen.Spec (UGenTag, portDefaults)

-- | Wrap a phase accumulator back into @[0, 1)@. Since the accumulator is always
-- in @[0, 1)@ and audio-rate increments are @< 1@, the value before wrapping is
-- in @[0, 2)@, so a single compare-and-subtract is both correct and much faster
-- than @floor@ on the hot path (GHC's @floor :: Float -> Int@ is not one
-- instruction). Mirrors Rust's @phase -= phase.floor()@ under the same invariant.
wrap01 :: Sample -> Sample
wrap01 p = if p >= 1 then p - 1 else p
{-# INLINE wrap01 #-}

-- | Advance a phase accumulator by one sample at frequency @f@ (Hz), given the
-- reciprocal sample rate @invSr = 1 / sampleRate@, wrapping into @[0, 1)@. This
-- is the shared step of every phase-accumulator oscillator.
phasorStep :: Sample -> Sample -> Sample -> Sample
phasorStep invSr f p = wrap01 (p + f * invSr)
{-# INLINE phasorStep #-}

-- | Build a phase-accumulator oscillator node — the whole shared skeleton of the
-- family: an unboxed phase cell, the reciprocal sample rate hoisted out of the
-- loop, the freq port (always port 0 of a phasor UGen) bound from the descriptor
-- registry, and the per-sample loop that reads freq, emits, and advances the
-- phase.
--
-- The caller supplies only @emit i p@, which writes the output sample for phase
-- @p@ (in @[0, 1)@) at index @i@ — the sole thing that distinguishes one phasor
-- oscillator from another. Any /extra/ ports beyond freq are bound by the caller
-- and closed over by @emit@, so a UGen's port count stays its own business.
-- 'INLINE'd, so each call site compiles back to the loop it replaced.
mkPhasorOsc
  :: SampleRate -> [MBlock s] -> MBlock s -> UGenTag
  -> (Int -> Sample -> ST s ())
  -> ST s (Node s)
mkPhasorOsc sr ins out tag emit = do
  phase <- VUM.replicate 1 0  -- unboxed phase accumulator
  let !invSr       = invSampleRate sr
      (freqIn, dF) = bindPort ins tag 0
      !n           = VUM.length out
  pure $ Node $ scanBlock1F phase n $ \i p -> do
    f <- readInput freqIn i dF
    emit i p
    pure (phasorStep invSr f p)
{-# INLINE mkPhasorOsc #-}

-- | Bind input port @p@ of a UGen, pairing the resolved source block with the
-- port's default value taken from the descriptor registry (rather than a literal
-- baked into the builder). Called once per node at build time, so the @!!@ and
-- the descriptor lookup never touch the render path; the returned default is a
-- plain 'Sample' read per sample by 'Microsynth.Node.readInput'.
bindPort :: [MBlock s] -> UGenTag -> Int -> (Input s, Sample)
bindPort ins tag p = (bindInput ins p, portDefaults tag !! p)
{-# INLINE bindPort #-}

-- | Fill an output block from a stateless per-sample function of the index.
-- Replaces the hand-written @go !i@ loop in the stateless arithmetic UGens.
mapBlock :: Int -> MBlock s -> (Int -> ST s Sample) -> ST s ()
mapBlock n out f = go 0
  where
    go !i
      | i >= n    = pure ()
      | otherwise = do
          y <- f i
          VUM.unsafeWrite out i y
          go (i + 1)
{-# INLINE mapBlock #-}

-- | Scan a block threading a single 'Float' state cell. The cell is read once at
-- the start and written back once at the end; @step i s@ receives the sample
-- index and current state, writes its own output sample, and returns the next
-- state. The state stays unboxed (@Float#@) through the loop.
scanBlock1F :: MBlock s -> Int -> (Int -> Sample -> ST s Sample) -> ST s ()
scanBlock1F cell n step = do
  s0 <- VUM.unsafeRead cell 0
  let go !i !s
        | i >= n    = VUM.unsafeWrite cell 0 s
        | otherwise = do
            s' <- step i s
            go (i + 1) s'
  go 0 s0
{-# INLINE scanBlock1F #-}

-- | Scan a block threading two 'Float' state values held in one 2-element cell
-- (@[a, b]@). Both are read once up front and written back once at the end;
-- @step i a b@ writes its own output sample and returns @(a', b')@. The two
-- accumulators are threaded as separate unboxed arguments.
scanBlock2F :: MBlock s -> Int -> (Int -> Sample -> Sample -> ST s (Sample, Sample)) -> ST s ()
scanBlock2F cell n step = do
  a0 <- VUM.unsafeRead cell 0
  b0 <- VUM.unsafeRead cell 1
  let go !i !a !b
        | i >= n    = VUM.unsafeWrite cell 0 a >> VUM.unsafeWrite cell 1 b
        | otherwise = do
            (a', b') <- step i a b
            go (i + 1) a' b'
  go 0 a0 b0
{-# INLINE scanBlock2F #-}

-- | Scan a block threading a 'Float' accumulator and an 'Int' accumulator, held
-- in two separate 1-element cells. Read once up front, written back once at the
-- end; @step i f n@ writes its own output sample and returns @(f', n')@.
scanBlockFI
  :: MBlock s -> VUM.MVector s Int -> Int
  -> (Int -> Sample -> Int -> ST s (Sample, Int)) -> ST s ()
scanBlockFI fCell iCell n step = do
  f0 <- VUM.unsafeRead fCell 0
  g0 <- VUM.unsafeRead iCell 0
  let go !i !f !g
        | i >= n    = VUM.unsafeWrite fCell 0 f >> VUM.unsafeWrite iCell 0 g
        | otherwise = do
            (f', g') <- step i f g
            go (i + 1) f' g'
  go 0 f0 g0
{-# INLINE scanBlockFI #-}
