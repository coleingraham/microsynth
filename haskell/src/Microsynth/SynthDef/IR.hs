{-# LANGUAGE OverloadedStrings #-}

-- | The SynthDef interchange IR — a versioned JSON encoding of the compiled
-- graph.
--
-- This is the contract from @COEXISTENCE.md@: authoring front-ends (the Haskell
-- EDSL, a GUI, an agent) all produce the /same/ flat graph, and the runtime
-- consumes it. The encoding is the flat, compiled graph — not a surface syntax —
-- so anything that can produce the structure can target it, and it is
-- unambiguous by construction (the right target for schema-constrained AI
-- generation).
--
-- Shape (mirrors the @COEXISTENCE.md@ example):
--
-- > { "version": 1, "name": "demo",
-- >   "params": [ { "name": "freq", "default": 220 }, ... ],
-- >   "nodes":  [ { "id": 0, "kind": "Param", "args": { "name": "freq", "default": 220 } },
-- >              { "id": 1, "kind": "Saw", "inputs": [0] }, ... ],
-- >   "output": 8 }
--
-- Kind tags and (for the payload-carrying leaves) their @args@ come from the
-- descriptor registry, so the IR and the engine agree on names by construction.
module Microsynth.SynthDef.IR
  ( toIR
  , fromIR
  , irVersion
  ) where

import Data.Aeson (Value, object, withObject, (.:), (.:?), (.!=), (.=))
import Data.Aeson.Types (Object, Parser, parseEither)

import Microsynth.Signal (BinOp (..), UGenKind (..))
import Microsynth.SynthDef (NodeDef (..), SynthDef (..), mkSynthDef)
import Microsynth.UGen.Spec (serTag)

-- | The IR schema version emitted by 'toIR'. Bumped when the wire shape changes.
irVersion :: Int
irVersion = 1

-- --- Encoding ---

-- | Encode a compiled 'SynthDef' to the interchange 'Value'.
toIR :: SynthDef -> Value
toIR sd = object
  [ "version" .= irVersion
  , "name"    .= sdName sd
  , "params"  .= [ object ["name" .= n, "default" .= d] | (n, d) <- sdParams sd ]
  , "nodes"   .= zipWith nodeToIR [0 :: Int ..] (sdNodes sd)
  , "output"  .= sdOutput sd
  ]

nodeToIR :: Int -> NodeDef -> Value
nodeToIR i nd = object $
  ["id" .= i, "kind" .= serTag (ndKind nd)]
    ++ argsField (ndKind nd)
    ++ inputsField (ndInputs nd)
  where
    inputsField []  = []
    inputsField xs  = ["inputs" .= xs]
    argsField (KConst v)   = ["args" .= object ["value" .= v]]
    argsField (KParam n d) = ["args" .= object ["name" .= n, "default" .= d]]
    argsField (KBinOp op)  = ["args" .= object ["op" .= binOpTag op]]
    argsField _            = []

binOpTag :: BinOp -> String
binOpTag Add = "Add"
binOpTag Sub = "Sub"
binOpTag Mul = "Mul"
binOpTag Div = "Div"

-- --- Decoding ---

-- | Decode an interchange 'Value' back into a 'SynthDef', or a human-readable
-- error. Parameters are recovered from the node list (via 'mkSynthDef'), so a
-- graph produced by any front-end round-trips without trusting a redundant
-- @params@ field.
fromIR :: Value -> Either String SynthDef
fromIR = parseEither parseSynthDef

parseSynthDef :: Value -> Parser SynthDef
parseSynthDef = withObject "SynthDef" $ \o -> do
  name    <- o .: "name"
  nodesV  <- o .: "nodes"
  outId   <- o .: "output"
  nodes   <- mapM parseNode nodesV
  pure (mkSynthDef name nodes outId)

parseNode :: Value -> Parser NodeDef
parseNode = withObject "node" $ \o -> do
  kindTag <- o .: "kind"
  margs   <- o .:? "args"
  inputs  <- o .:? "inputs" .!= []
  kind    <- parseKind kindTag margs
  pure (NodeDef kind inputs)

parseKind :: String -> Maybe Object -> Parser UGenKind
parseKind t margs = case t of
  "Const"  -> withArgs $ \a -> KConst <$> a .: "value"
  "Param"  -> withArgs $ \a -> KParam <$> a .: "name" <*> a .: "default"
  "BinOp"  -> withArgs $ \a -> KBinOp <$> (a .: "op" >>= parseBinOp)
  "Neg"    -> pure KNeg
  "SinOsc" -> pure KSinOsc
  "Saw"    -> pure KSaw
  "Lpf"    -> pure KLpf
  "Perc"   -> pure KPerc
  _        -> fail ("unknown UGen kind: " ++ t)
  where
    withArgs f = maybe (fail ("kind " ++ t ++ " requires \"args\"")) f margs

parseBinOp :: String -> Parser BinOp
parseBinOp s = case s of
  "Add" -> pure Add
  "Sub" -> pure Sub
  "Mul" -> pure Mul
  "Div" -> pure Div
  _     -> fail ("unknown BinOp: " ++ s)
