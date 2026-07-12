-- | The UGen descriptor registry — the single source of truth for UGen
-- metadata.
--
-- Previously each UGen's shape was spread across three or four places: the
-- 'UGenKind' constructor, its smart constructor, the @instantiate@ dispatch
-- arm, and — for port defaults — a literal buried inside each DSP builder
-- (@readInput freqIn i 440@). Port /names/ existed nowhere at all. This module
-- pulls the metadata (a serialization tag plus ordered port names and defaults)
-- into one table, the direct analog of what Rust's @spec()@ / @register_spec@
-- did on the runtime side.
--
-- It is deliberately __pure__ (no @ST@, no @s@): the render layer, the graph
-- introspection, the JSON IR, and — in future — the FHRR encoder and the
-- legal-edit proposer all read from here without dragging in the render
-- machinery. Port names are the encoder's binding roles; the enumerable tag set
-- ('allUGens') is the proposer's edit vocabulary; 'uiSerTag' is the IR kind tag.
-- One producer of the vocabulary, many consumers.
module Microsynth.UGen.Spec
  ( UGenTag (..)
  , PortSpec (..)
  , UGenInfo (..)
  , ugenInfo
  , allUGens
  , tagOf
  , serTag
  , portDefaults
  ) where

import Microsynth.Signal (UGenKind (..))

-- | A payload-free, enumerable handle for each UGen kind. Unlike 'UGenKind'
-- (whose leaves carry values), this is nullary and @Bounded@\/@Enum@, so
-- @[minBound .. maxBound]@ is the entire UGen vocabulary as data.
data UGenTag
  = TConst | TParam | TBinOp | TNeg | TSinOsc | TSaw | TLpf | TPerc
  deriving (Eq, Ord, Show, Enum, Bounded)

-- | One input port: its role name and the value used when the port is left
-- unconnected. Names mirror the Rust @spec()@ port names for cross-engine
-- consistency.
data PortSpec = PortSpec
  { portName    :: !String
  , portDefault :: !Float
  }
  deriving (Eq, Show)

-- | The full metadata for one UGen kind: its enumerable tag, its stable
-- serialization tag (the IR @kind@ string), and its ordered input ports.
data UGenInfo = UGenInfo
  { uiTag    :: !UGenTag
  , uiSerTag :: !String
  , uiPorts  :: ![PortSpec]
  }
  deriving (Eq, Show)

-- | The single source of truth. Every other view (defaults for the builders,
-- kind tags for the IR, binding roles for the encoder, the vocabulary for the
-- proposer) is derived from this one function.
ugenInfo :: UGenTag -> UGenInfo
ugenInfo t = case t of
  TConst  -> UGenInfo t "Const"  []
  TParam  -> UGenInfo t "Param"  []
  TBinOp  -> UGenInfo t "BinOp"  [PortSpec "a" 0, PortSpec "b" 0]
  TNeg    -> UGenInfo t "Neg"    [PortSpec "in" 0]
  TSinOsc -> UGenInfo t "SinOsc" [PortSpec "freq" 440, PortSpec "phase" 0]
  TSaw    -> UGenInfo t "Saw"    [PortSpec "freq" 440]
  TLpf    -> UGenInfo t "Lpf"    [PortSpec "sig" 0, PortSpec "cutoff" 1000, PortSpec "q" 0.707]
  TPerc   -> UGenInfo t "Perc"   [PortSpec "attack" 0.001, PortSpec "release" 0.1]

-- | Every UGen kind's metadata — the proposer's whole edit vocabulary as data.
allUGens :: [UGenInfo]
allUGens = map ugenInfo [minBound .. maxBound]

-- | The one @UGenKind -> UGenTag@ mapping (total). Everything that needs a
-- node's tag (the builder dispatch, the IR serializer, the introspection
-- accessors) goes through here rather than re-casing on the constructor.
tagOf :: UGenKind -> UGenTag
tagOf k = case k of
  KConst   _ -> TConst
  KParam _ _ -> TParam
  KBinOp   _ -> TBinOp
  KNeg       -> TNeg
  KSinOsc    -> TSinOsc
  KSaw       -> TSaw
  KLpf       -> TLpf
  KPerc      -> TPerc

-- | The IR serialization tag for a node's kind.
serTag :: UGenKind -> String
serTag = uiSerTag . ugenInfo . tagOf

-- | The ordered per-port default values for a UGen kind — what the DSP builders
-- fall back to for an unconnected port. Single-sourced here so the builders,
-- the IR, and any future consumer agree by construction.
portDefaults :: UGenTag -> [Float]
portDefaults = map portDefault . uiPorts . ugenInfo
