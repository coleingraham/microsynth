{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- | Compiled synth templates and the builder that produces them.
--
-- A 'SynthDef' is an immutable value (like Rust's @SynthDef@ in
-- @src/synthdef.rs@): a flat list of node definitions, the output node index,
-- and the declared parameters. 'synthdef' \/ 'param' \/ 'out' are the builder
-- interface that replaces the @synthdef NAME p=default = body@ text syntax.
--
-- Compilation flattens the 'Signal' AST into nodes, interning parameters by
-- name and constants by value so shared leaves become a single node (the
-- observable-sharing problem, solved for the cases that matter).
module Microsynth.SynthDef
  ( SynthDef (..)
  , NodeDef (..)
  , Build
  , synthdef
  , param
  , out
  , mkSynthDef
  , paramsOf
  ) where

import Control.Monad.State.Strict (State, execState, gets, modify', runState)
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Maybe (mapMaybe)

import Microsynth.Signal
import Microsynth.Types (NodeId (..), ParamName (..), Sample, SynthName (..))

-- | One node in a compiled synth: its kind plus the node ids feeding each
-- input port, in order.
data NodeDef = NodeDef
  { ndKind   :: !UGenKind
  , ndInputs :: ![NodeId]
  }
  deriving (Show, Eq)

-- | An immutable, compiled synth template.
data SynthDef = SynthDef
  { sdName   :: SynthName
  , sdNodes  :: [NodeDef]             -- ^ indexed by position (node id)
  , sdOutput :: NodeId                -- ^ id of the sink node
  , sdParams :: [(ParamName, Sample)] -- ^ declared parameters (name, default)
  }
  deriving (Show, Eq)

-- | The SynthDef builder monad. It threads the (single) output expression;
-- parameters are recovered from the compiled node list.
newtype Build a = Build (State (Maybe Signal) a)
  deriving (Functor, Applicative, Monad)

-- | Declare a named parameter with a default and get a 'Signal' for it.
param :: ParamName -> Sample -> Build Signal
param name def = pure (paramSig name def)

-- | Mark a signal as the synth's audio output (its sink).
out :: Signal -> Build ()
out s = Build (modify' (const (Just s)))

-- | Build a named 'SynthDef' from a builder body.
synthdef :: SynthName -> Build () -> SynthDef
synthdef name (Build body) =
  case execState body Nothing of
    Nothing  -> error ("synthdef " ++ unSynthName name ++ ": body never called `out`")
    Just sig -> compile name sig

-- --- Compilation (Signal AST -> flat node list) ---

data CompS = CompS
  { csNext   :: !Int
  , csNodes  :: !(Map Int NodeDef)
  , csIntern :: !(Map String Int)  -- shared leaves: "p:name" / "c:value"
  }

compile :: SynthName -> Signal -> SynthDef
compile name sig =
  let (outId, st) = runState (walk sig) (CompS 0 Map.empty Map.empty)
      nodes       = Map.elems (csNodes st) -- Map Int is ordered by id
  in mkSynthDef name nodes (NodeId outId)

-- | Assemble a 'SynthDef' directly from a flat node list and its output id,
-- recovering the declared parameters from the 'KParam' leaves. This is the
-- entry point for rebuilding an /edited/ graph (e.g. a structural edit proposed
-- over the flat 'NodeDef' list) without round-tripping through the 'Signal'
-- AST, which only the builder DSL can produce.
mkSynthDef :: SynthName -> [NodeDef] -> NodeId -> SynthDef
mkSynthDef name nodes outId = SynthDef name nodes outId (paramsOf nodes)

-- | The declared parameters (name, default) of a flat node list, in node order.
paramsOf :: [NodeDef] -> [(ParamName, Sample)]
paramsOf = mapMaybe declaredParam
  where
    declaredParam (NodeDef (KParam nm d) _) = Just (nm, d)
    declaredParam _                          = Nothing

walk :: Signal -> State CompS Int
walk (Signal kind ins) = do
  childIds <- mapM walk ins
  case kind of
    KParam nm _ -> intern ("p:" ++ unParamName nm) kind
    KConst v    -> intern ("c:" ++ show v) kind
    _           -> fresh kind childIds

-- | Allocate a brand-new node and return its id.
fresh :: UGenKind -> [Int] -> State CompS Int
fresh kind ins = do
  i <- gets csNext
  modify' $ \s -> s
    { csNext  = i + 1
    , csNodes = Map.insert i (NodeDef kind (map NodeId ins)) (csNodes s)
    }
  pure i

-- | Return the existing id for a shared leaf, or allocate one on first sight.
intern :: String -> UGenKind -> State CompS Int
intern key kind = do
  m <- gets csIntern
  case Map.lookup key m of
    Just i  -> pure i
    Nothing -> do
      i <- fresh kind []
      modify' $ \s -> s { csIntern = Map.insert key i (csIntern s) }
      pure i
