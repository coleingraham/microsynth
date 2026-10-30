use crate::graph::AudioGraph;
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
    /// Audio input nodes: (name, node_index). These are AudioIn UGen nodes
    /// that receive external audio (e.g. from a bus) when used in an effect chain.
    audio_inputs: Vec<(String, usize)>,
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

    /// Get the audio input nodes (name, node_index).
    /// These are AudioIn UGen nodes that receive external audio from buses.
    pub fn audio_inputs(&self) -> &[(String, usize)] {
        &self.audio_inputs
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
    audio_inputs: Vec<(String, usize)>,
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
            audio_inputs: Vec::new(),
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
        self.edges.push(SynthDefEdge { from, to, to_input });
    }

    /// Replace the factory for an already-added node, keeping its index (and
    /// therefore every edge/param/audio-input already wired to it) intact.
    ///
    /// Exists for the coefficient-table-bank id-reference mechanism
    /// (`ir::IrSynthDef::compile_with_tables`): a table-bound node is first
    /// added with an ordinary factory (so it participates in edge wiring
    /// like any other node), then patched in place once its table id has
    /// been resolved against a live `CoeffTableBank`. Not used by the plain
    /// DSL/IR compile paths, which never need to revisit a node after
    /// adding it.
    ///
    /// # Panics
    /// Panics if `idx` is out of range.
    pub fn replace_node<F>(&mut self, idx: usize, factory: F)
    where
        F: Fn() -> Box<dyn UGen> + Send + Sync + 'static,
    {
        self.factories[idx] = Box::new(factory);
    }

    /// Name a parameter: associates a name with a specific node's input port.
    pub fn param(&mut self, name: impl Into<String>, node_index: usize, input_index: usize) {
        self.param_names
            .push((name.into(), node_index, input_index));
    }

    /// Mark a node as an audio input (e.g. an AudioIn UGen that receives
    /// external audio from a bus). The routing system uses this to wire
    /// bus outputs into effect SynthDefs.
    pub fn audio_input(&mut self, name: impl Into<String>, node_index: usize) {
        self.audio_inputs.push((name.into(), node_index));
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
            audio_inputs: self.audio_inputs,
        }
    }
}

/// A named parameter on a live synth: maps a name to the NodeId of its
/// Const node in the graph.
#[derive(Debug, Clone)]
pub struct SynthParam {
    /// Parameter name (e.g. "freq", "amp").
    pub name: String,
    /// NodeId of the Const node that holds this parameter's value.
    pub node_id: NodeId,
}

/// A live instance of a SynthDef in the render graph.
///
/// Tracks the mapping from SynthDef node indices to live `NodeId`s
/// in the `AudioGraph`, and the synth's done rule (see
/// [`Engine::free_done_synths`](crate::engine::Engine::free_done_synths)).
pub struct Synth {
    /// Which SynthDef this was instantiated from (by name).
    def_name: String,
    /// Map from SynthDef node index to live graph NodeId.
    node_ids: Vec<NodeId>,
    /// The output NodeId in the live graph.
    output_node: NodeId,
    /// Named parameters with their live NodeIds.
    params: Vec<SynthParam>,
    /// Audio input nodes: (name, live NodeId). Used by the routing system
    /// to wire bus outputs into effect synths.
    audio_input_nodes: Vec<(String, NodeId)>,
    /// How done-reporting nodes silence this synth's output (built once at
    /// instantiation from the def's edges and its UGens' specs).
    done_plan: DonePlan,
}

impl Synth {
    /// Create a new Synth tracking instance.
    pub(crate) fn new(
        def_name: String,
        node_ids: Vec<NodeId>,
        output_node: NodeId,
        params: Vec<SynthParam>,
        audio_input_nodes: Vec<(String, NodeId)>,
        done_plan: DonePlan,
    ) -> Self {
        Synth {
            def_name,
            node_ids,
            output_node,
            params,
            audio_input_nodes,
            done_plan,
        }
    }

    /// Whether this synth is done: its output is silenced by done-reporting
    /// nodes on its output path (see
    /// [`Engine::free_done_synths`](crate::engine::Engine::free_done_synths)
    /// for the rule). A synth with no done-reporting node on that path is
    /// never done.
    pub(crate) fn output_is_done(&mut self, graph: &AudioGraph) -> bool {
        self.done_plan.evaluate(&self.node_ids, graph)
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

    /// Get the named parameters on this synth.
    pub fn params(&self) -> &[SynthParam] {
        &self.params
    }

    /// Find the NodeId for a named parameter. Returns None if not found.
    pub fn param_node(&self, name: &str) -> Option<NodeId> {
        self.params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.node_id)
    }

    /// Get the audio input nodes (name, live NodeId).
    pub fn audio_input_nodes(&self) -> &[(String, NodeId)] {
        &self.audio_input_nodes
    }

    /// Find the NodeId for a named audio input. Returns None if not found.
    pub fn audio_input_node(&self, name: &str) -> Option<NodeId> {
        self.audio_input_nodes
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, id)| *id)
    }

    /// Create a lightweight clone for bookkeeping.
    pub(crate) fn clone_handle(&self) -> Self {
        Synth {
            def_name: self.def_name.clone(),
            node_ids: self.node_ids.clone(),
            output_node: self.output_node,
            params: self.params.clone(),
            audio_input_nodes: self.audio_input_nodes.clone(),
            done_plan: self.done_plan.clone(),
        }
    }
}

// --- Done rule ---

/// UGens whose output falls silent, after a short transient, once their
/// `in` port falls silent: filters without a long recirculating memory,
/// dynamics, waveshapers, panners and short modulation effects. Silence
/// propagates through their `in` port only, never through a parameter port.
///
/// Anything not listed here (and not `Mul`/`Div`/`Neg`/`Add`/`Sub`) stops the
/// propagation: oscillators and physical models (their inputs are
/// parameters), envelopes, and nodes that keep sounding after their input
/// stops (`CombFilter`, `Delay`, `FeedbackDelay`, `PingPongDelay`, `GVerb`,
/// `Convolution`, the spectral processors, `PitchShift`, `SampleAndHold`).
/// A new pass-through UGen is added here deliberately; leaving one out only
/// means a done node upstream of it no longer frees the voice.
const SILENCE_PASS_THROUGH: &[&str] = &[
    "OnePole",
    "BiquadLPF",
    "BiquadHPF",
    "BiquadBPF",
    "BiquadNotch",
    "AllpassFilter",
    "BiquadPeaking",
    "BiquadLowShelf",
    "BiquadHighShelf",
    "ParametricEq3",
    "Compressor",
    "Limiter",
    "Bitcrusher",
    "SoftClip",
    "Overdrive",
    "WaveFolder",
    "FreqShift",
    "Chorus",
    "Flanger",
    "Phaser",
    "WowFlutter",
    "StereoWidth",
    "Pan2",
    "Mix",
    "Clip",
    "Lag",
];

/// How one node's silence follows from its inputs' silence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SilenceRule {
    /// Silent only when the node itself reports `is_done()`.
    Own,
    /// Silent when any listed input is silent (a product: `Mul`, or a
    /// pass-through UGen's single `in` port).
    Any,
    /// Silent when every listed input is silent (a sum: `Add`, `Sub`).
    All,
}

/// One node of a [`DonePlan`], in evaluation order.
#[derive(Debug, Clone)]
struct DoneStep {
    /// SynthDef node index.
    node: usize,
    rule: SilenceRule,
    /// Steps (indices into `DonePlan::steps`, all earlier than this one)
    /// feeding the ports the rule reads. An `All` step with an unconnected
    /// port is never silent through its inputs and lists no inputs.
    inputs: Vec<usize>,
}

/// The precomputed done rule of one synth: which nodes can silence its
/// output and how, in topological order (the output node last).
///
/// A node is silent when it reports `is_done()` itself, or when its inputs
/// make it silent: a `Mul` when either factor is silent (the usual
/// `env * signal` VCA), a `Div`/`Neg` when its numerator/input is, an
/// `Add`/`Sub` when both terms are, a pass-through processor
/// ([`SILENCE_PASS_THROUGH`]) when its `in` port is. The synth is done when
/// its output node is silent. Only the cone of nodes that can reach the
/// output this way is kept; everything else (an excitation burst feeding a
/// delay line, a pitch envelope feeding a `freq` port) is never consulted.
#[derive(Debug, Clone, Default)]
pub(crate) struct DonePlan {
    steps: Vec<DoneStep>,
    /// Per-step scratch, reused across evaluations.
    silent: Vec<bool>,
}

impl DonePlan {
    /// Build the plan for a def from its edges, output node and the specs of
    /// its instantiated UGens (`ugens[i]` is def node `i`).
    pub(crate) fn build(def: &SynthDef, ugens: &[Box<dyn UGen>]) -> Self {
        let n = ugens.len();
        // The live graph keeps one edge per input port, the last one
        // connected; mirror that.
        let mut port_source: Vec<Vec<Option<usize>>> = ugens
            .iter()
            .map(|u| alloc::vec![None; u.spec().inputs.len()])
            .collect();
        for e in def.edges() {
            if let Some(slot) = port_source
                .get_mut(e.to)
                .and_then(|ports| ports.get_mut(e.to_input))
            {
                *slot = Some(e.from);
            }
        }

        // Per node: its rule and the def nodes feeding the ports it reads
        // (`None` = an unconnected port, which is never silent).
        let classify = |i: usize| -> (SilenceRule, Vec<Option<usize>>) {
            let spec = ugens[i].spec();
            let ports = &port_source[i];
            let port_named = |name: &str| -> Vec<Option<usize>> {
                spec.inputs
                    .iter()
                    .position(|p| p.name == name)
                    .map(|k| alloc::vec![ports[k]])
                    .unwrap_or_default()
            };
            match spec.name {
                "Mul" => (SilenceRule::Any, ports.clone()),
                "Div" => (SilenceRule::Any, ports.iter().take(1).copied().collect()),
                "Add" | "Sub" => (SilenceRule::All, ports.clone()),
                "Neg" => (SilenceRule::Any, port_named("in")),
                name if SILENCE_PASS_THROUGH.contains(&name) => {
                    (SilenceRule::Any, port_named("in"))
                }
                _ => (SilenceRule::Own, Vec::new()),
            }
        };
        let rules: Vec<(SilenceRule, Vec<Option<usize>>)> = (0..n).map(classify).collect();

        // Post-order DFS from the output over the ports the rules read gives
        // a topological order of the cone (the def graph is acyclic).
        let mut step_of: Vec<Option<usize>> = alloc::vec![None; n];
        let mut on_stack: Vec<bool> = alloc::vec![false; n];
        let mut steps: Vec<DoneStep> = Vec::new();
        let output = def.output_node();
        if output < n {
            // Explicit stack of (node, next input to visit).
            let mut stack: Vec<(usize, usize)> = alloc::vec![(output, 0)];
            on_stack[output] = true;
            while let Some(&mut (node, ref mut next)) = stack.last_mut() {
                let srcs = &rules[node].1;
                if *next < srcs.len() {
                    let k = *next;
                    *next += 1;
                    if let Some(src) = srcs[k]
                        && src < n
                        && step_of[src].is_none()
                        && !on_stack[src]
                    {
                        on_stack[src] = true;
                        stack.push((src, 0));
                    }
                    continue;
                }
                stack.pop();
                on_stack[node] = false;
                let (rule, srcs) = &rules[node];
                let mut rule = *rule;
                let mut inputs = Vec::with_capacity(srcs.len());
                for src in srcs {
                    match src.and_then(|s| step_of.get(s).copied().flatten()) {
                        Some(step) => inputs.push(step),
                        // An unconnected port (or a cycle edge) never
                        // falls silent: a sum with one is never silent
                        // through its inputs; a product just ignores it.
                        None => {
                            if rule == SilenceRule::All {
                                rule = SilenceRule::Own;
                                inputs.clear();
                                break;
                            }
                        }
                    }
                }
                if rule == SilenceRule::All && inputs.is_empty() {
                    rule = SilenceRule::Own;
                }
                step_of[node] = Some(steps.len());
                steps.push(DoneStep { node, rule, inputs });
            }
        }
        let silent = alloc::vec![false; steps.len()];
        DonePlan { steps, silent }
    }

    /// Evaluate the plan against the live graph. `node_ids` maps def node
    /// indices to live ids. Returns whether the output node is silent.
    fn evaluate(&mut self, node_ids: &[NodeId], graph: &AudioGraph) -> bool {
        let DonePlan { steps, silent } = self;
        for (i, step) in steps.iter().enumerate() {
            let own = node_ids
                .get(step.node)
                .is_some_and(|&id| graph.node_is_done(id));
            silent[i] = own
                || match step.rule {
                    SilenceRule::Own => false,
                    SilenceRule::Any => step.inputs.iter().any(|&k| silent[k]),
                    SilenceRule::All => step.inputs.iter().all(|&k| silent[k]),
                };
        }
        silent.last().copied().unwrap_or(false)
    }
}
