-- | The embedded signal DSL — the heart of the Haskell version.
--
-- In the Rust engine, SynthDefs are written in a hand-parsed, Haskell-flavoured
-- text DSL (@src/dsl/{lexer,parser,compiler}.rs@). In Haskell that DSL needs no
-- parser: a 'Signal' is a pure description of a sub-graph, and the 'Num' /
-- 'Fractional' instances make @osc * env * amp@ and @freq * 4@ ordinary,
-- type-checked Haskell. This module replaces all three Rust DSL modules.
module Microsynth.Signal
  ( -- * The signal graph description
    Signal (..)
  , BinOp (..)
  , UGenKind (..)
    -- * Leaves
  , constSig
  , paramSig
    -- * Built-in UGen smart constructors
  , sinOsc
  , saw
  , lpf
  , perc
  ) where

import Microsynth.Types (ParamName, Sample)

-- | The primitive kind of a node. 'KConst' and 'KParam' are the zero-input
-- leaves; the rest correspond to Rust UGens. This enum is the bridge the
-- compiler ('Microsynth.UGen.instantiate') dispatches on.
data UGenKind
  = KConst !Sample            -- ^ constant value (numeric literal)
  | KParam !ParamName !Sample -- ^ named parameter with a default
  | KBinOp !BinOp          -- ^ arithmetic binary operator node
  | KNeg                   -- ^ unary negation node
  | KSinOsc                -- ^ sine oscillator (freq, phase)
  | KSaw                   -- ^ naive sawtooth (freq)
  | KLpf                   -- ^ RBJ biquad low-pass (sig, cutoff, q)
  | KPerc                  -- ^ percussive attack/release envelope
  deriving (Eq, Show)

data BinOp = Add | Sub | Mul | Div
  deriving (Eq, Show)

-- | A description of a (sub)graph: a node kind plus its input signals.
-- Leaves ('KConst' / 'KParam') carry an empty input list. This tree is what
-- 'Microsynth.SynthDef.synthdef' flattens into a graph.
data Signal = Signal !UGenKind [Signal]
  deriving (Show)

-- | @fromInteger@ / @fromRational@ turn literals into 'KConst' nodes, so that
-- writing @sinOsc freq 0 * amp@ or @freq * 4@ just works.
instance Num Signal where
  a + b       = Signal (KBinOp Add) [a, b]
  a - b       = Signal (KBinOp Sub) [a, b]
  a * b       = Signal (KBinOp Mul) [a, b]
  negate a    = Signal KNeg [a]
  fromInteger = constSig . fromInteger
  abs    _ = error "Microsynth.Signal: abs is not supported on Signal"
  signum _ = error "Microsynth.Signal: signum is not supported on Signal"

instance Fractional Signal where
  a / b        = Signal (KBinOp Div) [a, b]
  fromRational = constSig . fromRational

-- | A constant-valued signal.
constSig :: Sample -> Signal
constSig v = Signal (KConst v) []

-- | A named, defaulted parameter (the analogue of @synthdef p=default@).
paramSig :: ParamName -> Sample -> Signal
paramSig name def = Signal (KParam name def) []

-- | Sine oscillator: @sinOsc freq phase@.
sinOsc :: Signal -> Signal -> Signal
sinOsc freq phase = Signal KSinOsc [freq, phase]

-- | Naive (non-band-limited) sawtooth: @saw freq@.
saw :: Signal -> Signal
saw freq = Signal KSaw [freq]

-- | RBJ biquad low-pass filter: @lpf sig cutoff q@.
lpf :: Signal -> Signal -> Signal -> Signal
lpf sig cutoff q = Signal KLpf [sig, cutoff, q]

-- | Gateless percussive envelope: @perc attack release@ (seconds).
perc :: Signal -> Signal -> Signal
perc attack release = Signal KPerc [attack, release]
