{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}

-- | Domain newtypes for the whole engine.
--
-- The graph and render path deal in a handful of distinct quantities that all
-- happen to be represented by @Int@, @Float@ or @String@: node ids, names,
-- audio samples, the sample rate, block/sample counts, and so on. Left naked,
-- they are trivially transposed — a node id where a count is expected, a sample
-- rate summed into a signal — and every such mistake type-checks. This module
-- gives each quantity its own newtype so the compiler rejects the mix-ups.
--
-- The audio value type 'Sample' derives the full numeric stack ('Num',
-- 'Fractional', 'Floating') and an unboxed-vector representation identical to
-- 'Float', so the DSP kernel stays written in ordinary arithmetic and compiles
-- to exactly the same code — the types are erased at runtime. 'SampleRate' is
-- kept distinct from 'Sample' on purpose (you must unwrap to combine a rate with
-- a signal), which is the whole point.
module Microsynth.Types
  ( -- * Identifiers
    NodeId (..)
  , SynthName (..)
  , ParamName (..)
  , PortName (..)
  , KindTag (..)
    -- * Audio quantities
  , Sample (..)
  , SampleRate (..)
    -- * Counts and offsets
  , SampleCount (..)
  , BlockSize (..)
  , SampleOffset (..)
    -- * Interchange
  , IRVersion (..)
  ) where

import Data.Aeson (FromJSON, ToJSON)
import Data.String (IsString)
import Data.Vector.Unboxed.Deriving (derivingUnbox)
import Data.Word (Word64)

-- | A node's position/identity in the flat graph. Also used for edge endpoints
-- and the output sink — never as an arithmetic count.
newtype NodeId = NodeId { unNodeId :: Int }
  deriving stock (Show)
  deriving newtype (Eq, Ord, Enum, ToJSON, FromJSON)

-- | The name of a 'Microsynth.SynthDef.SynthDef'.
newtype SynthName = SynthName { unSynthName :: String }
  deriving stock (Show)
  deriving newtype (Eq, Ord, IsString, ToJSON, FromJSON)

-- | The name of a declared parameter (and the key of a parameter override).
newtype ParamName = ParamName { unParamName :: String }
  deriving stock (Show)
  deriving newtype (Eq, Ord, IsString, ToJSON, FromJSON)

-- | A UGen input-port role name (e.g. @"freq"@, @"cutoff"@).
newtype PortName = PortName { unPortName :: String }
  deriving stock (Show)
  deriving newtype (Eq, Ord, IsString, ToJSON, FromJSON)

-- | A UGen kind's serialization tag (e.g. @"Saw"@, @"Lpf"@) — the wire @kind@.
newtype KindTag = KindTag { unKindTag :: String }
  deriving stock (Show)
  deriving newtype (Eq, Ord, IsString, ToJSON, FromJSON)

-- | An audio (or control) sample value: what flows through buffers, what UGen
-- ports read and write, and what graph constants\/parameters hold. Full numeric
-- instances so the DSP kernel reads as ordinary math.
newtype Sample = Sample { unSample :: Float }
  deriving stock (Show)
  deriving newtype
    (Eq, Ord, Num, Fractional, Floating, Real, RealFrac, ToJSON, FromJSON)

-- | The sample rate in Hz. Deliberately /not/ a 'Sample': combining a rate with
-- a signal requires an explicit unwrap, so it cannot silently enter the audio path.
newtype SampleRate = SampleRate { unSampleRate :: Float }
  deriving stock (Show)
  deriving newtype (Eq, Ord)

-- | A number of samples (e.g. a render length).
newtype SampleCount = SampleCount { unSampleCount :: Int }
  deriving stock (Show)
  deriving newtype (Eq, Ord)

-- | A block length in samples.
newtype BlockSize = BlockSize { unBlockSize :: Int }
  deriving stock (Show)
  deriving newtype (Eq, Ord)

-- | A running sample offset (per-block global clock).
newtype SampleOffset = SampleOffset { unSampleOffset :: Word64 }
  deriving stock (Show)
  deriving newtype (Eq, Ord)

-- | The interchange IR schema version.
newtype IRVersion = IRVersion { unIRVersion :: Int }
  deriving stock (Show)
  deriving newtype (Eq, Ord, ToJSON, FromJSON)

-- Unbox instance for Sample, represented identically to Float, so
-- @MVector s Sample@ / @Vector Sample@ are zero-cost over the raw float storage.
derivingUnbox "Sample"
  [t| Sample -> Float |]
  [| unSample |]
  [| Sample |]
