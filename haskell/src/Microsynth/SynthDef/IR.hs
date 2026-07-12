{-# LANGUAGE OverloadedStrings #-}

-- | The SynthDef interchange IR — a versioned, well-typed encoding of the
-- compiled graph with standard @aeson@ 'ToJSON'\/'FromJSON' instances.
--
-- This is the contract from @COEXISTENCE.md@: authoring front-ends (the Haskell
-- EDSL, a GUI, an agent) all produce the /same/ flat graph, and the runtime
-- consumes it. The encoding is the flat, compiled graph — not a surface syntax —
-- so anything that can produce the structure can target it, and it is
-- unambiguous by construction (the right target for schema-constrained AI
-- generation).
--
-- The IR is a dedicated data type ('IR' \/ 'IRNode' \/ 'IRParam'), distinct from
-- the internal 'SynthDef': the JSON contract is spelled out in one place as
-- types, and the aeson instances are canonical and non-orphan (they live beside
-- the types they serialize, and 'SynthDef' itself stays serialization-agnostic).
-- 'toIR'\/'fromIR' are the pure structural map between the domain graph and the
-- IR; 'encodeSynthDef'\/'decodeSynthDef' are the byte-level convenience wrappers.
--
-- Wire shape (mirrors the @COEXISTENCE.md@ example):
--
-- > { "version": 1, "name": "demo",
-- >   "params": [ { "name": "freq", "default": 220 }, ... ],
-- >   "nodes":  [ { "id": 0, "kind": "Param", "args": { "name": "freq", "default": 220 } },
-- >              { "id": 1, "kind": "Saw", "inputs": [0] }, ... ],
-- >   "output": 8 }
--
-- Each node carries its @id@ as a field ('irnId'); on decode the id is
-- load-bearing — nodes are placed by id and input references resolve against it,
-- so 'fromIR' validates that the ids form a contiguous @0..n-1@ set. Kind tags
-- and (for the payload-carrying leaves) their @args@ come from the descriptor
-- registry, so the IR and the engine agree on names by construction. The node
-- kind reuses the domain 'UGenKind', keeping payloads type-safe without a second
-- parallel enumeration of the UGen vocabulary.
module Microsynth.SynthDef.IR
  ( -- * The IR types
    IR (..)
  , IRNode (..)
  , IRParam (..)
    -- * Domain <-> IR
  , toIR
  , fromIR
    -- * Byte-level JSON
  , encodeSynthDef
  , decodeSynthDef
    -- * Versioning
  , currentIRVersion
  ) where

import Data.Aeson
  ( FromJSON (..), ToJSON (..), eitherDecode, encode, object, withObject
  , (.!=), (.:), (.:?), (.=) )
import Data.Aeson.Types (Object, Pair, Parser)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Map.Strict as Map

import Microsynth.Signal (BinOp (..), UGenKind (..))
import Microsynth.SynthDef (NodeDef (..), SynthDef (..), mkSynthDef)
import Microsynth.UGen.Spec (serTag)

-- | The IR schema version 'toIR' emits. Bumped when the wire shape changes.
currentIRVersion :: Int
currentIRVersion = 1

-- --- The IR types ---

-- | A declared parameter in the IR.
data IRParam = IRParam
  { irpName    :: !String
  , irpDefault :: !Float
  }
  deriving (Eq, Show)

-- | One node in the IR graph: its id (position in the flat graph), its kind
-- (reusing the domain 'UGenKind', so leaf payloads stay typed), and the ids of
-- the nodes feeding each input port, in order.
data IRNode = IRNode
  { irnId     :: !Int
  , irnKind   :: !UGenKind
  , irnInputs :: ![Int]
  }
  deriving (Eq, Show)

-- | The whole interchange document: a versioned, named, flat node graph plus its
-- declared params and output node id.
data IR = IR
  { irVersion :: !Int
  , irName    :: !String
  , irParams  :: ![IRParam]
  , irNodes   :: ![IRNode]
  , irOutput  :: !Int
  }
  deriving (Eq, Show)

-- --- JSON instances (canonical, non-orphan) ---

instance ToJSON IRParam where
  toJSON (IRParam n d) = object ["name" .= n, "default" .= d]

instance FromJSON IRParam where
  parseJSON = withObject "IRParam" $ \o ->
    IRParam <$> o .: "name" <*> o .: "default"

instance ToJSON IRNode where
  toJSON (IRNode i k ins) = object $
    ["id" .= i, "kind" .= serTag k] ++ kindArgs k ++ inputsField ins
    where
      inputsField [] = []
      inputsField xs = ["inputs" .= xs]

instance FromJSON IRNode where
  parseJSON = withObject "IRNode" $ \o -> do
    i      <- o .: "id"
    tag    <- o .: "kind"
    margs  <- o .:? "args"
    inputs <- o .:? "inputs" .!= []
    k      <- parseKind tag margs
    pure (IRNode i k inputs)

instance ToJSON IR where
  toJSON (IR v n ps ns out) = object
    [ "version" .= v
    , "name"    .= n
    , "params"  .= ps
    , "nodes"   .= ns
    , "output"  .= out
    ]

instance FromJSON IR where
  parseJSON = withObject "IR" $ \o -> IR
    <$> o .:? "version" .!= currentIRVersion
    <*> o .:  "name"
    <*> o .:? "params" .!= []
    <*> o .:  "nodes"
    <*> o .:  "output"

-- | The @args@ object (if any) for a node's kind. Only the payload-carrying
-- leaves emit @args@; pure-DSP UGens carry all their data in edges and ports.
kindArgs :: UGenKind -> [Pair]
kindArgs (KConst v)   = ["args" .= object ["value" .= v]]
kindArgs (KParam n d) = ["args" .= object ["name" .= n, "default" .= d]]
kindArgs (KBinOp op)  = ["args" .= object ["op" .= binOpTag op]]
kindArgs _            = []

binOpTag :: BinOp -> String
binOpTag Add = "Add"
binOpTag Sub = "Sub"
binOpTag Mul = "Mul"
binOpTag Div = "Div"

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

-- --- Domain <-> IR ---

-- | Project a compiled 'SynthDef' into the IR, numbering nodes by position.
toIR :: SynthDef -> IR
toIR sd = IR
  { irVersion = currentIRVersion
  , irName    = sdName sd
  , irParams  = [ IRParam n d | (n, d) <- sdParams sd ]
  , irNodes   = zipWith mkNode [0 ..] (sdNodes sd)
  , irOutput  = sdOutput sd
  }
  where
    mkNode i nd = IRNode i (ndKind nd) (ndInputs nd)

-- | Rebuild a 'SynthDef' from the IR, or a human-readable error. The node ids
-- must form a contiguous @0..n-1@ set (which 'toIR' guarantees); nodes are
-- placed by id so input references resolve correctly regardless of list order.
-- Parameters are recovered from the node list (via 'mkSynthDef'), so the
-- redundant @params@ field is never trusted.
fromIR :: IR -> Either String SynthDef
fromIR ir = do
  checkVersion (irVersion ir)
  nodes <- orderNodesById (irNodes ir)
  pure (mkSynthDef (irName ir) nodes (irOutput ir))

checkVersion :: Int -> Either String ()
checkVersion v
  | v == currentIRVersion = Right ()
  | otherwise             = Left ("unsupported IR version: " ++ show v)

-- | Place IR nodes into the flat, position-indexed node list, validating that
-- their ids are exactly @0..n-1@ with no gaps or duplicates.
orderNodesById :: [IRNode] -> Either String [NodeDef]
orderNodesById ns
  | Map.size byId /= n = Left "IR node ids are not unique"
  | otherwise          = traverse pick [0 .. n - 1]
  where
    n    = length ns
    byId = Map.fromList [ (irnId x, x) | x <- ns ]
    pick i = case Map.lookup i byId of
      Just x  -> Right (NodeDef (irnKind x) (irnInputs x))
      Nothing -> Left ("IR is missing node id " ++ show i)

-- --- Byte-level JSON ---

-- | Encode a 'SynthDef' to its IR JSON bytes (@'encode' . 'toIR'@).
encodeSynthDef :: SynthDef -> BL.ByteString
encodeSynthDef = encode . toIR

-- | Decode IR JSON bytes back into a 'SynthDef', reporting JSON or IR errors.
decodeSynthDef :: BL.ByteString -> Either String SynthDef
decodeSynthDef bs = eitherDecode bs >>= fromIR
