-- | Instantiation: turn a 'UGenKind' into a live 'Node' with its inputs and
-- output already bound.
--
-- The analogue of Rust's @UGenFactory@ closures / @register_builtins@
-- (@src/ugens/mod.rs@): it maps each node kind to its DSP implementation,
-- threading in the sample rate, parameter overrides, the resolved input
-- blocks, and the node's output block.
module Microsynth.UGen
  ( instantiate
  ) where

import Control.Monad.ST (ST)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map

import Microsynth.Buffer (MBlock)
import Microsynth.Context (Context (..))
import Microsynth.Node (Node)
import Microsynth.Signal (UGenKind (..))
import Microsynth.Types (ParamName, Sample)
import Microsynth.UGen.Envelope (mkPerc)
import Microsynth.UGen.Filter (mkLpf)
import Microsynth.UGen.Math (constNode, mkBinOp, mkNeg)
import Microsynth.UGen.Oscillator (mkSaw, mkSinOsc)

-- | Instantiate one node. @overrides@ lets the CLI/host replace a parameter's
-- default at spawn time (e.g. @--param freq=330@). @ins@ are the resolved
-- input blocks (in port order); @out@ is this node's output block.
instantiate
  :: Context -> Map ParamName Sample -> UGenKind
  -> [MBlock s] -> MBlock s -> ST s (Node s)
instantiate ctx overrides kind ins out = case kind of
  KConst v    -> constNode v out
  KParam nm d -> constNode (Map.findWithDefault d nm overrides) out
  KBinOp op   -> mkBinOp op ins out
  KNeg        -> mkNeg ins out
  KSinOsc     -> mkSinOsc sr ins out
  KSaw        -> mkSaw sr ins out
  KLpf        -> mkLpf sr ins out
  KPerc       -> mkPerc sr ins out
  where
    sr = ctxSampleRate ctx
