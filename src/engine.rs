use crate::buffer::AudioBuffer;
use crate::context::ProcessContext;
use crate::graph::AudioGraph;
use crate::node::NodeId;
use crate::scheduler::{EventAction, Scheduler, VoiceId};
use crate::synthdef::{Synth, SynthDef, SynthParam};
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

/// A voice: a live synth instance with a unique ID.
struct Voice {
    id: VoiceId,
    synth: Synth,
    /// Which bus input slot this voice is connected to, if any.
    bus_input: Option<(NodeId, usize)>,
}

/// The synthesis engine. Owns the audio graph and drives rendering.
///
/// The engine manages:
/// - A live `AudioGraph` that can be modified at runtime
/// - Instantiation of `SynthDef` templates into the live graph
/// - Voice management with unique IDs
/// - Event scheduling for sample-accurate automation
/// - Block-by-block rendering
pub struct Engine {
    context: ProcessContext,
    graph: AudioGraph,
    synths: Vec<Synth>,
    voices: Vec<Voice>,
    scheduler: Scheduler,
}

impl Engine {
    /// Create a new engine with the given configuration.
    pub fn new(config: EngineConfig) -> Self {
        Engine {
            context: ProcessContext::new(config.sample_rate, config.block_size),
            graph: AudioGraph::new(),
            synths: Vec::new(),
            voices: Vec::new(),
            scheduler: Scheduler::new(),
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

    /// Access the scheduler for scheduling events.
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Mutably access the scheduler.
    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.scheduler
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

        // Build parameter mapping: param name → live NodeId
        let params: Vec<SynthParam> = def
            .param_names()
            .iter()
            .map(|(name, node_index, _input_index)| SynthParam {
                name: name.clone(),
                node_id: node_ids[*node_index],
            })
            .collect();

        let output_id = node_ids[def.output_node()];
        let synth = Synth::new(
            String::from(def.name()),
            node_ids,
            output_id,
            params,
        );

        self.synths.push(synth.clone_handle());
        synth
    }

    /// Instantiate a SynthDef as a named voice with a VoiceId.
    /// Returns the VoiceId for later reference (parameter changes, freeing).
    pub fn spawn_voice(&mut self, def: &SynthDef) -> VoiceId {
        let synth = self.instantiate_synthdef(def);
        let id = self.scheduler.alloc_voice_id();
        self.voices.push(Voice {
            id,
            synth: synth.clone_handle(),
            bus_input: None,
        });
        id
    }

    /// Instantiate a SynthDef as a voice and connect its output to a Bus node.
    /// Automatically finds the next available input slot on the bus.
    /// Returns the VoiceId, or None if the bus has no free slots.
    pub fn spawn_voice_on_bus(&mut self, def: &SynthDef, bus_node: NodeId) -> Option<VoiceId> {
        // Find the next free input slot on the bus
        let bus_max = match self.graph.node_spec(bus_node) {
            Some(spec) => spec.inputs.len(),
            None => return None,
        };

        // Find which slots are already used by checking existing edges
        let used_slots: Vec<usize> = self.voices.iter()
            .filter_map(|v| {
                if let Some((bus, slot)) = v.bus_input {
                    if bus == bus_node { Some(slot) } else { None }
                } else {
                    None
                }
            })
            .collect();

        let free_slot = (0..bus_max).find(|slot| !used_slots.contains(slot))?;

        let synth = self.instantiate_synthdef(def);
        let id = self.scheduler.alloc_voice_id();
        self.graph.connect(synth.output_node(), bus_node, free_slot);
        self.voices.push(Voice {
            id,
            synth: synth.clone_handle(),
            bus_input: Some((bus_node, free_slot)),
        });
        Some(id)
    }

    /// Set a named parameter on a voice by VoiceId.
    /// Returns true if the voice and parameter were found.
    pub fn set_voice_param(&mut self, voice_id: VoiceId, name: &str, value: f32) -> bool {
        if let Some(voice) = self.voices.iter().find(|v| v.id == voice_id) {
            if let Some(node_id) = voice.synth.param_node(name) {
                self.graph.set_node_value(node_id, value)
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Remove a voice by VoiceId.
    pub fn free_voice(&mut self, voice_id: VoiceId) {
        if let Some(pos) = self.voices.iter().position(|v| v.id == voice_id) {
            let voice = self.voices.remove(pos);
            self.remove_synth(&voice.synth);
        }
    }

    /// Get the Synth handle for a voice by VoiceId.
    pub fn voice_synth(&self, voice_id: VoiceId) -> Option<&Synth> {
        self.voices.iter().find(|v| v.id == voice_id).map(|v| &v.synth)
    }

    /// Connect a voice's output to another node's input.
    /// Useful for routing voices to a bus/mixer.
    pub fn connect_voice_output(&mut self, voice_id: VoiceId, to: NodeId, to_input: usize) {
        if let Some(voice) = self.voices.iter().find(|v| v.id == voice_id) {
            self.graph.connect(voice.synth.output_node(), to, to_input);
        }
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

    /// Process pending scheduled events for the current block, then render.
    /// Advances the internal clock.
    ///
    /// This is the main render method when using the scheduler.
    /// Events whose time falls within this block are dispatched before rendering.
    ///
    /// Returns a reference to the sink node's output, or `None` if
    /// no sink is set.
    pub fn render(&mut self) -> Option<&AudioBuffer> {
        // Dispatch events for this block
        let deadline = self.context.sample_offset + self.context.block_size as u64;
        let events = self.scheduler.drain_before(deadline);

        let mut needs_prepare = false;
        for event in events {
            match event.action {
                EventAction::SetParam { voice, param, value } => {
                    self.set_voice_param(voice, &param, value);
                }
                EventAction::SetGate { voice, value } => {
                    self.set_voice_param(voice, "gate", value);
                }
                EventAction::SetParamGlide { voice, param, target, glide_secs } => {
                    self.set_voice_param_glide(voice, &param, target, glide_secs);
                }
                EventAction::FreeSynth { voice } => {
                    self.free_voice(voice);
                    needs_prepare = true;
                }
            }
        }

        if needs_prepare {
            self.graph.prepare(&self.context);
        }

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
            // Render the block (dispatches events internally)
            let rendered = self.render();

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

    /// Set a named parameter on a live synth to a new value (instant).
    /// Does NOT require `prepare()` — takes effect on the next render.
    /// Returns true if the parameter was found and set.
    pub fn set_param(&mut self, synth: &Synth, name: &str, value: f32) -> bool {
        if let Some(node_id) = synth.param_node(name) {
            self.graph.set_node_value(node_id, value)
        } else {
            false
        }
    }

    /// Set a named parameter with a smooth glide to the target value.
    /// The parameter will linearly ramp from its current value to `target`
    /// over `glide_secs` seconds. Use for crescendo, diminuendo, pitch bends,
    /// filter sweeps, etc.
    /// Returns true if the parameter was found and accepted the glide.
    pub fn set_param_glide(
        &mut self,
        synth: &Synth,
        name: &str,
        target: f32,
        glide_secs: f32,
    ) -> bool {
        if let Some(node_id) = synth.param_node(name) {
            self.graph.set_node_target(node_id, target, glide_secs)
        } else {
            false
        }
    }

    /// Set a named parameter on a voice with a smooth glide.
    pub fn set_voice_param_glide(
        &mut self,
        voice_id: VoiceId,
        name: &str,
        target: f32,
        glide_secs: f32,
    ) -> bool {
        if let Some(voice) = self.voices.iter().find(|v| v.id == voice_id) {
            if let Some(node_id) = voice.synth.param_node(name) {
                self.graph.set_node_target(node_id, target, glide_secs)
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get a reference to the list of active synths.
    pub fn synths(&self) -> &[Synth] {
        &self.synths
    }

    /// Remove all synths that have any node reporting `is_done()`.
    /// This enables "done actions": when an envelope finishes, the whole
    /// synth is removed from the graph.
    /// Returns the number of synths removed.
    /// After calling this, you must call `prepare()` before rendering.
    pub fn free_done_synths(&mut self) -> usize {
        let mut removed = 0;
        let mut i = 0;
        while i < self.synths.len() {
            let is_done = self.synths[i]
                .node_ids()
                .iter()
                .any(|&id| self.graph.node_is_done(id));
            if is_done {
                let synth = self.synths.remove(i);
                // Also remove from voices list
                self.voices.retain(|v| v.synth.output_node() != synth.output_node());
                for &id in synth.node_ids() {
                    self.graph.remove_node(id);
                }
                removed += 1;
            } else {
                i += 1;
            }
        }
        removed
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
