-- | Global engine state passed to every UGen on every block.
--
-- Port of Rust @src/context.rs@ (@ProcessContext@ and @Rate@).
module Microsynth.Context
  ( Rate (..)
  , Context (..)
  ) where

import Microsynth.Types (BlockSize, SampleOffset, SampleRate)

-- | Whether a port carries one sample per block (control) or a full
-- block of samples (audio). Kept for design fidelity with the Rust
-- engine; the scaffold treats everything as audio rate.
data Rate = Audio | Control
  deriving (Eq, Show)

-- | Immutable per-block context. Mirrors @ProcessContext@.
data Context = Context
  { ctxSampleRate   :: !SampleRate
  , ctxBlockSize    :: !BlockSize
  , ctxSampleOffset :: !SampleOffset
  }
  deriving (Eq, Show)
