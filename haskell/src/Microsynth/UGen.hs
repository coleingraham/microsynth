-- | Instantiation: turn a 'UGenKind' into a live, stateful 'Node'.
--
-- The analogue of Rust's @UGenFactory@ closures / @register_builtins@
-- (@src/ugens/mod.rs@): it maps each node kind to its DSP implementation,
-- threading in the sample rate and any parameter overrides.
module Microsynth.UGen
  ( instantiate
  ) where

import Control.Monad.ST (ST)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map

import Microsynth.Context (Context (..))
import Microsynth.Node (Node)
import Microsynth.Signal (UGenKind (..))
import Microsynth.UGen.Envelope (mkPerc)
import Microsynth.UGen.Filter (mkLpf)
import Microsynth.UGen.Math (constNode, mkBinOp, mkNeg)
import Microsynth.UGen.Oscillator (mkSaw, mkSinOsc)

-- | Instantiate one node. @overrides@ lets the CLI/host replace a parameter's
-- default at spawn time (e.g. @--param freq=330@).
instantiate :: Context -> Map String Float -> UGenKind -> ST s (Node s)
instantiate ctx overrides kind = case kind of
  KConst v    -> pure (constNode v)
  KParam nm d -> pure (constNode (Map.findWithDefault d nm overrides))
  KBinOp op   -> pure (mkBinOp op)
  KNeg        -> pure mkNeg
  KSinOsc     -> mkSinOsc sr
  KSaw        -> mkSaw sr
  KLpf        -> mkLpf sr
  KPerc       -> mkPerc sr
  where
    sr = ctxSampleRate ctx
