-- | Global engine state passed to every UGen on every block.
--
-- Port of the @ProcessContext@ half of Rust @src/context.rs@.
module Microsynth.Context
  ( Context (..)
  ) where

import Microsynth.Types (BlockSize, SampleOffset, SampleRate)

-- | Immutable per-block context. Mirrors @ProcessContext@.
data Context = Context
  { ctxSampleRate   :: !SampleRate
  , ctxBlockSize    :: !BlockSize
  , ctxSampleOffset :: !SampleOffset
  }
  deriving (Eq, Show)
