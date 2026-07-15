-- | Audio buffers for the render path.
--
-- Port of Rust @src/buffer.rs@. Rust uses a stack @[f32; 128]@ per channel;
-- here a block is an unboxed, mutable @Float@ vector allocated once and
-- reused every block (Rust's "no allocation on the render path" rule).
--
-- The scaffold's UGens are all mono, so the engine works with single
-- @MBlock@s. A true multichannel @AudioBuffer@ (a boxed vector of blocks,
-- for SuperCollider-style expansion) is where multichannel semantics would
-- live in a full port; it is described in the design doc.
module Microsynth.Buffer
  ( MBlock
  , maxBlockSize
  , newBlock
  ) where

import Control.Monad.ST (ST)
import qualified Data.Vector.Unboxed.Mutable as VUM

import Microsynth.Types (BlockSize (..), Sample)

-- | A single-channel, mutable, unboxed block of samples.
type MBlock s = VUM.MVector s Sample

-- | Maximum block size. 128 matches the Web Audio render quantum, as in Rust.
maxBlockSize :: BlockSize
maxBlockSize = BlockSize 128

-- | Allocate a zeroed block of the given length.
newBlock :: BlockSize -> ST s (MBlock s)
newBlock n = VUM.replicate (unBlockSize n) 0
