//! Routing graph for effects bussing.
//!
//! Manages named audio buses, effect chains, and signal routing.
//! The routing graph is built into the engine's `AudioGraph`, with the
//! main bus as the final output (sink).
//!
//! # Example
//!
//! ```rust,ignore
//! let mut routing = RoutingGraph::new();
//! let drums = routing.add_bus("drums", 2);    // stereo bus
//! let reverb_bus = routing.add_bus("reverb", 2);
//!
//! // drums => drumFx => main
//! routing.add_effect(drums, &drum_fx_def, routing.main_bus());
//! // drums => reverbFx => reverb_bus (fan-out from drums)
//! routing.add_effect(drums, &reverb_fx_def, reverb_bus);
//! // reverb_bus => masterFx => main
//! routing.add_effect(reverb_bus, &master_fx_def, routing.main_bus());
//! ```

use crate::node::NodeId;
use crate::synthdef::{Synth, SynthDef};
use alloc::string::String;
use alloc::vec::Vec;

/// Identifier for a bus in the routing graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BusId(pub(crate) usize);

/// Identifier for an effect slot in the routing graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectId(pub(crate) usize);

/// A named audio bus in the routing graph.
struct NamedBus {
    name: String,
    /// Live NodeId of the Bus UGen in the AudioGraph (set during build).
    node_id: Option<NodeId>,
    /// Number of audio channels (e.g. 2 for stereo).
    channels: usize,
}

/// An effect placed between two buses.
pub(crate) struct EffectSlot {
    /// The SynthDef for this effect (stored for instantiation during build).
    pub(crate) def_name: String,
    /// Bus this effect reads from.
    pub(crate) source_bus: BusId,
    /// Bus this effect writes to.
    pub(crate) target_bus: BusId,
    /// Live Synth handle (set during build).
    pub(crate) synth: Option<Synth>,
}

/// A routing graph managing named buses and effect chains.
///
/// The main bus is always present and serves as the final output.
/// Additional buses can be added and connected via effects to form
/// arbitrary routing topologies.
pub struct RoutingGraph {
    buses: Vec<NamedBus>,
    effects: Vec<EffectSlot>,
    main_bus_id: BusId,
}

impl RoutingGraph {
    /// Create a new routing graph with a default stereo main bus.
    pub fn new() -> Self {
        let main_bus = NamedBus {
            name: String::from("main"),
            node_id: None,
            channels: 2,
        };
        RoutingGraph {
            buses: alloc::vec![main_bus],
            effects: Vec::new(),
            main_bus_id: BusId(0),
        }
    }

    /// Get the main (output) bus ID.
    pub fn main_bus(&self) -> BusId {
        self.main_bus_id
    }

    /// Add a named bus with the given channel count. Returns its BusId.
    pub fn add_bus(&mut self, name: impl Into<String>, channels: usize) -> BusId {
        let id = BusId(self.buses.len());
        self.buses.push(NamedBus {
            name: name.into(),
            node_id: None,
            channels: channels.max(1),
        });
        id
    }

    /// Look up a bus by name. Returns None if not found.
    pub fn bus_by_name(&self, name: &str) -> Option<BusId> {
        self.buses
            .iter()
            .position(|b| b.name == name)
            .map(BusId)
    }

    /// Get the live NodeId of a bus (available after building).
    pub fn bus_node(&self, bus_id: BusId) -> Option<NodeId> {
        self.buses.get(bus_id.0).and_then(|b| b.node_id)
    }

    /// Get the channel count for a bus.
    pub fn bus_channels(&self, bus_id: BusId) -> Option<usize> {
        self.buses.get(bus_id.0).map(|b| b.channels)
    }

    /// Add an effect between two buses. Returns the EffectId.
    ///
    /// The effect SynthDef should contain an `audioIn` node which will be
    /// wired to the source bus's output. The effect's output will be
    /// connected to an input slot on the target bus.
    ///
    /// A single bus can be the source for multiple effects (fan-out),
    /// enabling sidechain compression, parallel processing, and send effects.
    pub fn add_effect(
        &mut self,
        source_bus: BusId,
        def: &SynthDef,
        target_bus: BusId,
    ) -> EffectId {
        let id = EffectId(self.effects.len());
        self.effects.push(EffectSlot {
            def_name: String::from(def.name()),
            source_bus,
            target_bus,
            synth: None,
        });
        id
    }

    /// Number of buses in the routing graph.
    pub fn num_buses(&self) -> usize {
        self.buses.len()
    }

    /// Number of effects in the routing graph.
    pub fn num_effects(&self) -> usize {
        self.effects.len()
    }

    /// Get the effect slot's Synth handle (available after building).
    pub fn effect_synth(&self, effect_id: EffectId) -> Option<&Synth> {
        self.effects.get(effect_id.0).and_then(|e| e.synth.as_ref())
    }

    // -- Internal methods used by Engine::build_routing --

    /// Set the live NodeId for a bus (called during build).
    pub(crate) fn set_bus_node(&mut self, bus_id: BusId, node_id: NodeId) {
        if let Some(bus) = self.buses.get_mut(bus_id.0) {
            bus.node_id = Some(node_id);
        }
    }

    /// Get bus info by index (for iteration during build).
    pub(crate) fn bus_info(&self, bus_id: BusId) -> Option<(& str, usize)> {
        self.buses.get(bus_id.0).map(|b| (b.name.as_str(), b.channels))
    }

    /// Get mutable access to effects (for setting synth handles during build).
    pub(crate) fn effects_mut(&mut self) -> &mut [EffectSlot] {
        &mut self.effects
    }

    /// Get read access to effects.
    pub(crate) fn effects(&self) -> &[EffectSlot] {
        &self.effects
    }

    /// Iterate over all bus IDs.
    pub(crate) fn bus_ids(&self) -> impl Iterator<Item = BusId> {
        (0..self.buses.len()).map(BusId)
    }
}

impl Default for RoutingGraph {
    fn default() -> Self {
        Self::new()
    }
}
