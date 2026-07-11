-- | The @UGen@ abstraction.
--
-- Port of the Rust @trait UGen@ (@src/node.rs@). Rust's @&mut self@ +
-- @process(ctx, inputs, out)@ becomes, at instantiation, a closure that has
-- already captured its input blocks, its output block, and its own mutable
-- state — leaving a bare per-block action. Because the graph pre-allocates
-- every block once, a node's inputs never move, so binding them once (instead
-- of threading an input list every block) removes the hot-path indirection.
module Microsynth.Node
  ( Node (..)
  , Input
  , bindInput
  , readInput
  ) where

import Control.Monad.ST (ST)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (MBlock)

-- | A live node's per-block work: read its captured inputs, write its captured
-- output, mutate its captured state. Everything else is closed over.
newtype Node s = Node { runBlock :: ST s () }

-- | A resolved input port: either a source block, or absent (use a default).
type Input s = Maybe (MBlock s)

-- | Resolve input port @port@ from a node's input block list, once, at build
-- time (the analogue of Rust's @inputs.get(port)@).
bindInput :: [MBlock s] -> Int -> Input s
bindInput ins port = case drop port ins of
  (b : _) -> Just b
  []      -> Nothing

-- | Read sample @i@ from a bound input, or a default if the port is absent.
-- Uses an unchecked read: callers iterate @0 .. blockSize-1@ and every block is
-- the same length, so the index is always in bounds.
readInput :: Input s -> Int -> Float -> ST s Float
readInput Nothing    _ d = pure d
readInput (Just b)   i _ = VUM.unsafeRead b i
{-# INLINE readInput #-}
