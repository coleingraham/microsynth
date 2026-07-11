{-# LANGUAGE BangPatterns #-}

-- | Stateless arithmetic UGens: constants, binary operators, negation.
--
-- Port of the @Const@ / @BinOpUGen@ / @NegUGen@ nodes from Rust
-- @src/ugens/math.rs@. Each builder captures its input/output blocks and
-- returns a bare per-block action.
module Microsynth.UGen.Math
  ( constNode
  , mkBinOp
  , mkNeg
  ) where

import Control.Monad.ST (ST)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (MBlock)
import Microsynth.Node (Node (..), bindInput, readInput)
import Microsynth.Signal (BinOp (..))

-- | A constant node. Its output never changes, so we fill the block once at
-- build time and do zero work per block.
constNode :: Float -> MBlock s -> ST s (Node s)
constNode !v out = do
  VUM.set out v
  pure (Node (pure ()))

binFun :: BinOp -> (Float -> Float -> Float)
binFun Add = (+)
binFun Sub = (-)
binFun Mul = (*)
binFun Div = (/)

-- | Elementwise binary arithmetic over two input blocks.
mkBinOp :: BinOp -> [MBlock s] -> MBlock s -> ST s (Node s)
mkBinOp op ins out = do
  let a  = bindInput ins 0
      b  = bindInput ins 1
      !n = VUM.length out
      f  = binFun op
  pure $ Node $
    let go !i
          | i >= n    = pure ()
          | otherwise = do
              x <- readInput a i 0
              y <- readInput b i 0
              VUM.unsafeWrite out i (f x y)
              go (i + 1)
    in go 0

-- | Unary negation of a single input block.
mkNeg :: [MBlock s] -> MBlock s -> ST s (Node s)
mkNeg ins out = do
  let a  = bindInput ins 0
      !n = VUM.length out
  pure $ Node $
    let go !i
          | i >= n    = pure ()
          | otherwise = do
              x <- readInput a i 0
              VUM.unsafeWrite out i (negate x)
              go (i + 1)
    in go 0
