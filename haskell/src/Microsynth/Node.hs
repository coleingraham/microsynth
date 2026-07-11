-- | The @UGen@ abstraction.
--
-- Port of the Rust @trait UGen@ (@src/node.rs@). Rust's @&mut self@ +
-- @process(ctx, inputs, out)@ becomes a function that closes over the node's
-- own mutable state (@STRef@s / mutable vectors created at instantiation) and
-- writes its output block in place. That closure /is/ the node, so no
-- existential type is needed — the state is captured, not exposed.
module Microsynth.Node
  ( Node (..)
  , sampleAt
  ) where

import Control.Monad.ST (ST)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Buffer (MBlock)
import Microsynth.Context (Context)

-- | A live, instantiated processing node.
--
-- @runNode ctx inputs output@ reads one sample at a time from each input
-- block and writes @output@ in place, mutating any captured state. The graph
-- guarantees (via topological order) that every input block is already
-- filled before this runs.
newtype Node s = Node
  { runNode :: Context -> [MBlock s] -> MBlock s -> ST s () }

-- | Read sample @i@ from input port @port@, or a default if that input is
-- absent. The analogue of Rust's @inputs.get(port)@ + per-sample indexing
-- (minus the modulo channel wrapping, which the mono scaffold doesn't need).
sampleAt :: [MBlock s] -> Int -> Int -> Float -> ST s Float
sampleAt ins port i def = case drop port ins of
  (b : _) -> VUM.read b i
  []      -> pure def
{-# INLINE sampleAt #-}
