use crate::buffer::AudioBuffer;
use crate::context::ProcessContext;
use crate::graph::AudioGraph;
use crate::node::NodeId;
use crate::synthdef::{Synth, SynthDef};
use alloc::string::String;
use alloc::vec::Vec;

/// Configuration for the synthesis engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Number of samples per audio-rate block.
    pub block_size: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            sample_rate: 44100.0,
            block_size: 64,
        }
    }
}

/// The synthesis engine. Owns the audio graph and drives rendering.
///
/// The engine manages:
/// - A live `AudioGraph` that can be modified at runtime
/// - Instantiation of `SynthDef` templates into the live graph
/// - Block-by-block rendering
pub struct Engine {
    context: ProcessContext,
    graph: AudioGraph,
    synths: Vec<Synth>,
}

impl Engine {
    /// Create a new engine with the given configuration.
    pub fn new(config: EngineConfig) -> Self {
        Engine {
            context: ProcessContext::new(config.sample_rate, config.block_size),
            graph: AudioGraph::new(),
            synths: Vec::new(),
        }
    }

    /// Access the process context.
    pub fn context(&self) -> &ProcessContext {
        &self.context
    }

    /// Access the graph for direct manipulation.
    pub fn graph(&self) -> &AudioGraph {
        &self.graph
    }

    /// Mutably access the graph.
    pub fn graph_mut(&mut self) -> &mut AudioGraph {
        &mut self.graph
    }

    /// Instantiate a SynthDef into the live graph.
    ///
    /// Creates fresh UGen instances from the SynthDef's factories,
    /// adds them to the graph, and wires them up according to the
    /// SynthDef's edge list.
    ///
    /// Returns the `Synth` handle which tracks the live node IDs.
    pub fn instantiate_synthdef(&mut self, def: &SynthDef) -> Synth {
        let ugens = def.instantiate();

        // Add all nodes to the graph, collecting their live NodeIds
        let mut node_ids: Vec<NodeId> = Vec::with_capacity(ugens.len());
        for ugen in ugens {
            let id = self.graph.add_node(ugen);
            node_ids.push(id);
        }

        // Wire up edges using the live NodeIds
        for edge in def.edges() {
            let from = node_ids[edge.from];
            let to = node_ids[edge.to];
            self.graph.connect(from, to, edge.to_input);
        }

        let output_id = node_ids[def.output_node()];
        let synth = Synth::new(
            String::from(def.name()),
            node_ids,
            output_id,
        );

        self.synths.push(synth.clone_handle());
        synth
    }

    /// Remove a synth instance from the graph.
    pub fn remove_synth(&mut self, synth: &Synth) {
        for &id in synth.node_ids() {
            self.graph.remove_node(id);
        }
        self.synths.retain(|s| s.output_node() != synth.output_node());
    }

    /// Prepare the graph for rendering. Must be called after any
    /// structural changes (adding/removing synths) and before `render()`.
    pub fn prepare(&mut self) {
        self.graph.prepare(&self.context);
    }

    /// Render one block of audio. Advances the internal clock.
    ///
    /// Returns a reference to the sink node's output, or `None` if
    /// no sink is set.
    pub fn render(&mut self) -> Option<&AudioBuffer> {
        let result = self.graph.render(&self.context);
        self.context.advance();
        result
    }

    /// Render `num_blocks` blocks of audio into a flat buffer.
    ///
    /// Returns one `Vec<f32>` per output channel.
    /// Useful for offline (non-real-time) rendering.
    #[cfg(feature = "std")]
    pub fn render_offline(&mut self, num_blocks: usize) -> Vec<Vec<f32>> {
        let block_size = self.context.block_size;
        let mut output: Vec<Vec<f32>> = Vec::new();

        for _ in 0..num_blocks {
            // Render the block
            let rendered = self.graph.render(&self.context);
            self.context.advance();

            if let Some(buf) = rendered {
                // Initialize output channels on first block
                if output.is_empty() {
                    for _ in 0..buf.num_channels() {
                        output.push(Vec::with_capacity(num_blocks * block_size));
                    }
                }
                // Copy block data to output
                for ch in 0..buf.num_channels() {
                    output[ch].extend_from_slice(buf.channel(ch).samples());
                }
            }
        }

        output
    }

    /// Current sample offset (monotonic time counter).
    pub fn sample_offset(&self) -> u64 {
        self.context.sample_offset
    }

    /// Current time in seconds.
    pub fn time_secs(&self) -> f64 {
        self.context.time_secs()
    }
}
