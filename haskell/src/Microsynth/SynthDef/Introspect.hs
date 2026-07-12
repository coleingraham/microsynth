-- | Descriptor-derived views over a compiled 'SynthDef' graph.
--
-- The flat 'SynthDef' (a position-indexed @[NodeDef]@ with an output id) is
-- already the right immutable, enumerable graph — this module adds the pure,
-- named views that a graph consumer wants without changing the data model or
-- touching the render path. Every view is derived from the descriptor registry
-- ("Microsynth.UGen.Spec"), so a node's kind tag and its port role names come
-- from the same single source the builders and IR use.
--
-- These are the accessors the future FHRR encoder (binding roles = port names)
-- and legal-edit proposer (kind tags + arity as data) read; they are deliberately
-- the read half only — mutation/rebuilding is 'Microsynth.SynthDef.mkSynthDef'.
module Microsynth.SynthDef.Introspect
  ( nodeTag
  , nodePorts
  , nodeArity
  ) where

import Microsynth.SynthDef (NodeDef (..))
import Microsynth.UGen.Spec (portName, serTag, tagOf, ugenInfo, uiPorts)

-- | The serialization/IR tag for a node's kind (e.g. @"Saw"@, @"Lpf"@).
nodeTag :: NodeDef -> String
nodeTag = serTag . ndKind

-- | A node's inputs as @(role-name, source-node-id)@ pairs: the descriptor's
-- ordered port names zipped with the graph edges feeding them. This is exactly
-- the binding-role view an encoder needs. Leaves ('KConst'\/'KParam') have no
-- ports and yield @[]@.
nodePorts :: NodeDef -> [(String, Int)]
nodePorts nd = zip names (ndInputs nd)
  where
    names = map portName (uiPorts (ugenInfo (tagOf (ndKind nd))))

-- | The declared input-port count for a node's kind (from the descriptor, not
-- from how many edges happen to be connected).
nodeArity :: NodeDef -> Int
nodeArity = length . uiPorts . ugenInfo . tagOf . ndKind
