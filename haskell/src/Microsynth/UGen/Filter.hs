{-# LANGUAGE BangPatterns #-}

-- | Filter UGens.
--
-- Port of the RBJ biquad low-pass from Rust @src/ugens/filters.rs@:
-- coefficients recomputed per sample (audio-rate cutoff/q), processed with a
-- transposed direct-form II biquad (@z1@/@z2@ state).
module Microsynth.UGen.Filter
  ( mkLpf
  ) where

import Control.Monad.ST (ST)
import Data.STRef (newSTRef, readSTRef, writeSTRef)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Node (Node (..), sampleAt)

tau :: Float
tau = 2 * pi

-- | RBJ low-pass biquad coefficients: returns @(b0, b1, b2, a1, a2)@ already
-- normalised by @a0@. Mirrors @biquad_lpf_coeffs@ in the Rust source.
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
mkLpf :: Float -> ST s (Node s)
mkLpf sr = do
  z1Ref <- newSTRef 0
  z2Ref <- newSTRef 0
  pure $ Node $ \_ ins out -> do
    let !n = VUM.length out
        go !i
          | i >= n    = pure ()
          | otherwise = do
              x  <- sampleAt ins 0 i 0
              fc <- sampleAt ins 1 i 1000
              q  <- sampleAt ins 2 i 0.707
              let (b0, b1, b2, a1, a2) = lpfCoeffs fc q sr
              s1 <- readSTRef z1Ref
              s2 <- readSTRef z2Ref
              let !y = b0 * x + s1
              writeSTRef z1Ref (b1 * x - a1 * y + s2)
              writeSTRef z2Ref (b2 * x - a2 * y)
              VUM.write out i y
              go (i + 1)
    go 0
