{-# LANGUAGE BangPatterns #-}

-- | Filter UGens.
--
-- Port of the RBJ biquad low-pass from Rust @src/ugens/filters.rs@:
-- coefficients recomputed per sample (audio-rate cutoff/q), processed with a
-- transposed direct-form II biquad. The @z1@/@z2@ state is read once per block,
-- threaded through the loop as unboxed arguments, and written back once.
module Microsynth.UGen.Filter
  ( mkLpf
  ) where

import Control.Monad.ST (ST)
import Data.STRef (newSTRef, readSTRef, writeSTRef)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (MBlock)
import Microsynth.Node (Node (..), bindInput, readInput)

tau :: Float
tau = 2 * pi

-- | RBJ low-pass biquad coefficients: @(b0, b1, b2, a1, a2)@ normalised by
-- @a0@. Mirrors @biquad_lpf_coeffs@ in the Rust source.
lpfCoeffs :: Float -> Float -> Float -> (Float, Float, Float, Float, Float)
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
  in (b0 * invA0, b1 * invA0, b2 * invA0, a1 * invA0, a2 * invA0)
{-# INLINE lpfCoeffs #-}

-- | Low-pass filter. Inputs: signal, cutoff (Hz), q.
mkLpf :: Float -> [MBlock s] -> MBlock s -> ST s (Node s)
mkLpf sr ins out = do
  z1Ref <- newSTRef 0
  z2Ref <- newSTRef 0
  let sigIn = bindInput ins 0
      cutIn = bindInput ins 1
      qIn   = bindInput ins 2
      !n    = VUM.length out
  pure $ Node $ do
    s1_0 <- readSTRef z1Ref
    s2_0 <- readSTRef z2Ref
    let go !i !s1 !s2
          | i >= n    = writeSTRef z1Ref s1 >> writeSTRef z2Ref s2
          | otherwise = do
              x  <- readInput sigIn i 0
              fc <- readInput cutIn i 1000
              q  <- readInput qIn i 0.707
              let (b0, b1, b2, a1, a2) = lpfCoeffs fc q sr
                  !y = b0 * x + s1
              VUM.unsafeWrite out i y
              go (i + 1) (b1 * x - a1 * y + s2) (b2 * x - a2 * y)
    go 0 s1_0 s2_0
