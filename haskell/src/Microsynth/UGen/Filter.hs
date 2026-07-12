{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE UnboxedTuples #-}

-- | Filter UGens.
--
-- Port of the RBJ biquad low-pass from Rust @src/ugens/filters.rs@:
-- coefficients recomputed per sample (audio-rate cutoff/q), processed with a
-- transposed direct-form II biquad. State (@z1@/@z2@) lives in a single unboxed
-- 2-element cell. Keeping state unboxed matters beyond the cell itself: writing
-- an unboxed cell places an /unboxed/ demand on the loop's threaded @s1@/@s2@
-- accumulators, so GHC passes them as raw @Float#@ and the inner loop allocates
-- nothing. (With a boxed @STRef Float@, the write demands a boxed @Float@,
-- forcing a heap box per sample — ~4.5x more allocation at high polyphony.)
-- The unboxed-tuple coefficient return keeps the recompute allocation-free too.
module Microsynth.UGen.Filter
  ( mkLpf
  ) where

import Control.Monad.ST (ST)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (MBlock)
import Microsynth.Node (Node (..), readInput)
import Microsynth.Numerics (tau)
import Microsynth.UGen.Common (bindPort, scanBlock2F)
import Microsynth.UGen.Spec (UGenTag (..))

-- | RBJ low-pass biquad coefficients: @(# b0, b1, b2, a1, a2 #)@ normalised by
-- @a0@. Mirrors @biquad_lpf_coeffs@ in the Rust source. The unboxed tuple
-- return means no per-sample heap allocation.
lpfCoeffs :: Float -> Float -> Float -> (# Float, Float, Float, Float, Float #)
lpfCoeffs freq q sr =
  let w0    = tau * freq / sr
      sinW0 = sin w0
      cosW0 = cos w0
      alpha = sinW0 / (2 * q)
      b0    = (1 - cosW0) / 2
      b1    = 1 - cosW0
      b2    = b0
      a0    = 1 + alpha
      a1    = -2 * cosW0
      a2    = 1 - alpha
      invA0 = 1 / a0
  in (# b0 * invA0, b1 * invA0, b2 * invA0, a1 * invA0, a2 * invA0 #)
{-# INLINE lpfCoeffs #-}

-- | Low-pass filter. Inputs: signal, cutoff (Hz), q.
mkLpf :: Float -> [MBlock s] -> MBlock s -> ST s (Node s)
mkLpf sr ins out = do
  st <- VUM.replicate 2 0  -- [z1, z2], unboxed
  let (sigIn, dSig) = bindPort ins TLpf 0
      (cutIn, dCut) = bindPort ins TLpf 1
      (qIn,   dQ)   = bindPort ins TLpf 2
      !n            = VUM.length out
  pure $ Node $ scanBlock2F st n $ \i s1 s2 -> do
    x  <- readInput sigIn i dSig
    fc <- readInput cutIn i dCut
    q  <- readInput qIn i dQ
    case lpfCoeffs fc q sr of
      (# b0, b1, b2, a1, a2 #) -> do
        let !y = b0 * x + s1
        VUM.unsafeWrite out i y
        pure (b1 * x - a1 * y + s2, b2 * x - a2 * y)
