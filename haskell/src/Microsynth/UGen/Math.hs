{-# LANGUAGE BangPatterns #-}

-- | Stateless arithmetic UGens: constants, binary operators, negation.
--
-- Port of the @Const@ / @BinOpUGen@ / @NegUGen@ nodes from Rust
-- @src/ugens/math.rs@.
module Microsynth.UGen.Math
  ( constNode
  , mkBinOp
  , mkNeg
  ) where

import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Node (Node (..), sampleAt)
import Microsynth.Signal (BinOp (..))

-- | A node that fills its whole output block with a constant. Used for both
-- numeric literals ('Microsynth.Signal.KConst') and parameter values.
constNode :: Float -> Node s
constNode !v = Node $ \_ _ out -> VUM.set out v

binFun :: BinOp -> (Float -> Float -> Float)
binFun Add = (+)
binFun Sub = (-)
binFun Mul = (*)
binFun Div = (/)

-- | Elementwise binary arithmetic over two input blocks.
mkBinOp :: BinOp -> Node s
mkBinOp op = Node $ \_ ins out -> do
  let !n = VUM.length out
      f  = binFun op
      go !i
        | i >= n    = pure ()
        | otherwise = do
            a <- sampleAt ins 0 i 0
            b <- sampleAt ins 1 i 0
            VUM.write out i (f a b)
            go (i + 1)
  go 0

-- | Unary negation of a single input block.
mkNeg :: Node s
mkNeg = Node $ \_ ins out -> do
  let !n = VUM.length out
      go !i
        | i >= n    = pure ()
        | otherwise = do
            x <- sampleAt ins 0 i 0
            VUM.write out i (negate x)
            go (i + 1)
  go 0
