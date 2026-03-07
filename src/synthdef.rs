use crate::node::{NodeId, UGen};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// A connection within a SynthDef.
#[derive(Debug, Clone)]
pub struct SynthDefEdge {
    /// Index into the SynthDef's node list (source).
    pub from: usize,
    /// Index into the SynthDef's node list (destination).
    pub to: usize,
    /// Which input port on the destination node.
    pub to_input: usize,
}

/// Factory that creates a UGen instance.
/// Boxed closure so it can capture state (e.g. constant values from DSL).
pub type UGenFactory = Box<dyn Fn() -> Box<dyn UGen> + Send + Sync>;

/// An immutable template for a synthesis graph.
///
/// Like SuperCollider's SynthDef, this describes a fixed topology of UGens
/// and their connections. Once created, a SynthDef cannot be modified.
/// Multiple Synth instances can be created from the same SynthDef.
///
/// SynthDefs are compiled from a description and then instantiated into
/// the live render graph.
pub struct SynthDef {
    name: String,
    /// Factory closures for each node in the def.
    factories: Vec<UGenFactory>,
    /// Connections between nodes.
    edges: Vec<SynthDefEdge>,
    /// Which node index is the output of this SynthDef.
    output_node: usize,
    /// Named parameters: (name, node_index, input_index).
    param_names: Vec<(String, usize, usize)>,
}

impl SynthDef {
    /// Get the name of this SynthDef.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the parameter names.
    pub fn param_names(&self) -> &[(String, usize, usize)] {
        &self.param_names
    }

    /// Number of nodes in this SynthDef.
    pub fn num_nodes(&self) -> usize {
        self.factories.len()
    }

    /// Get the edges.
    pub fn edges(&self) -> &[SynthDefEdge] {
        &self.edges
    }

    /// Get the output node index.
    pub fn output_node(&self) -> usize {
        self.output_node
    }

    /// Instantiate all UGens for this SynthDef.
    /// Returns freshly created UGen instances.
    pub fn instantiate(&self) -> Vec<Box<dyn UGen>> {
        self.factories.iter().map(|f| f()).collect()
    }
}

/// Builder for creating SynthDefs.
pub struct SynthDefBuilder {
    name: String,
    factories: Vec<UGenFactory>,
    edges: Vec<SynthDefEdge>,
    output_node: Option<usize>,
    param_names: Vec<(String, usize, usize)>,
}

impl SynthDefBuilder {
    /// Start building a new SynthDef with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        SynthDefBuilder {
            name: name.into(),
            factories: Vec::new(),
            edges: Vec::new(),
            output_node: None,
            param_names: Vec::new(),
        }
    }

    /// Add a UGen to the SynthDef. Returns its index.
    pub fn add_node<F>(&mut self, factory: F) -> usize
    where
        F: Fn() -> Box<dyn UGen> + Send + Sync + 'static,
    {
        let idx = self.factories.len();
        self.factories.push(Box::new(factory));
        idx
    }

    /// Connect one node's output to another node's input port.
    pub fn connect(&mut self, from: usize, to: usize, to_input: usize) {
        self.edges.push(SynthDefEdge {
            from,
            to,
            to_input,
        });
    }

    /// Name a parameter: associates a name with a specific node's input port.
    pub fn param(&mut self, name: impl Into<String>, node_index: usize, input_index: usize) {
        self.param_names.push((name.into(), node_index, input_index));
    }

    /// Set which node is the output of this SynthDef.
    pub fn set_output(&mut self, node_index: usize) {
        self.output_node = Some(node_index);
    }

    /// Build the immutable SynthDef.
    ///
    /// # Panics
    /// Panics if no output node has been set.
    pub fn build(self) -> SynthDef {
        SynthDef {
            name: self.name,
            factories: self.factories,
            edges: self.edges,
            output_node: self.output_node.expect("SynthDef must have an output node"),
            param_names: self.param_names,
        }
    }
}

/// A live instance of a SynthDef in the render graph.
///
/// Tracks the mapping from SynthDef node indices to live `NodeId`s
/// in the `AudioGraph`.
pub struct Synth {
    /// Which SynthDef this was instantiated from (by name).
    def_name: String,
    /// Map from SynthDef node index to live graph NodeId.
    node_ids: Vec<NodeId>,
    /// The output NodeId in the live graph.
    output_node: NodeId,
}

impl Synth {
    /// Create a new Synth tracking instance.
    pub(crate) fn new(def_name: String, node_ids: Vec<NodeId>, output_node: NodeId) -> Self {
        Synth {
            def_name,
            node_ids,
            output_node,
        }
    }

    /// The name of the SynthDef this was created from.
    pub fn def_name(&self) -> &str {
        &self.def_name
    }

    /// The NodeIds in the live graph.
    pub fn node_ids(&self) -> &[NodeId] {
        &self.node_ids
    }

    /// The output NodeId.
    pub fn output_node(&self) -> NodeId {
        self.output_node
    }

    /// Create a lightweight clone for bookkeeping.
    pub(crate) fn clone_handle(&self) -> Self {
        Synth {
            def_name: self.def_name.clone(),
            node_ids: self.node_ids.clone(),
            output_node: self.output_node,
        }
    }
}
