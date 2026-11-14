use crate::buffer::AudioBuffer;
use crate::context::ProcessContext;
use crate::curve::{GlideShape, GlideSpace};
use crate::graph::AudioGraph;
use crate::node::NodeId;
use crate::routing::{BusId, EffectId, RoutingGraph};
use crate::scheduler::{EventAction, Scheduler, VoiceId};
use crate::synthdef::{DonePlan, Synth, SynthDef, SynthParam};
use crate::ugens;
use crate::voice::{Admission, BudgetScope, StealPolicy, VoiceAllocator};
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
    /// Extra graph nodes inserted to place this voice in the stereo field —
    /// `(Pan2 node, its pan-position Const node)`. `None` for voices spawned
    /// at center pan (or via a pan-unaware spawn method), which connect
    /// straight to the bus with no `Pan2` in the path. Owned by the voice so
    /// `free_voice` can tear them down alongside its synth.
    pan_nodes: Option<(NodeId, NodeId)>,
    /// The engine's sample offset when the voice was spawned: it has rendered
    /// at least one block once the offset has moved past this.
    born: u64,
    /// The bus the voice was spawned onto, if any.
    bus: Option<NodeId>,
    /// Set once the voice has been stolen: the `StealFade` node now between
    /// it and its bus slot, the sample offset the fade began at, and the
    /// index of its [`StealRecord`]. The voice keeps its slot until the fade
    /// reports done.
    fade: Option<(NodeId, u64, usize)>,
}

/// One voice steal, as [`Engine::steals`] records it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StealRecord {
    /// The engine's sample offset when the voice was stolen (the start of
    /// the block about to render).
    pub sample: u64,
    /// The victim's level at that moment ([`Engine::voice_level`]), if it
    /// had rendered a block.
    pub level: Option<f32>,
    /// `true` when the victim was freed at once instead of faded: it had no
    /// bus slot to fade through, or its bus was full and the slot was needed
    /// now.
    pub cut: bool,
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
    /// Monotonic count of synthdef instantiations, used to derive a distinct
    /// deterministic seed for each spawned voice's noise UGens (so repeated
    /// kick/snare/hat hits vary but stay reproducible run-to-run).
    noise_spawn_seq: u32,
    /// Optional voice budget and stealing policy. `None` (the default)
    /// leaves `spawn_voice`/`spawn_voice_on_bus`/`free_voice` exactly as
    /// unconditional as before; attaching one only changes behavior for
    /// callers that opt in through `spawn_voice_managed`/
    /// `spawn_voice_on_bus_managed`.
    voice_alloc: Option<VoiceAllocator>,
    /// Every steal made by the managed spawn paths, in order.
    steals: Vec<StealRecord>,
}

/// The allocator group of voices spawned onto no bus.
const OFF_BUS_GROUP: u64 = u64::MAX;

/// The allocator group of voices spawned onto `bus`: the bus itself.
fn bus_group(bus: NodeId) -> u64 {
    bus.index() as u64
}

/// Derive a deterministic noise seed from the voice spawn order and the UGen's
/// index within its synthdef. Folding in `node_idx` gives two noise nodes in one
/// synthdef distinct seeds; the multiply-scramble decorrelates consecutive hits
/// so a steady drum pattern doesn't expose the underlying LCG. The exact
/// constants are part of the reproducible-output contract: changing them changes
/// the rendered noise for a given spawn order.
fn derive_noise_seed(spawn_seq: u32, node_idx: usize) -> u32 {
    0xDEAD_BEEF ^ spawn_seq.wrapping_mul(0x9E37_79B9) ^ (node_idx as u32).wrapping_mul(0x85EB_CA77)
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
            noise_spawn_seq: 0,
            voice_alloc: None,
            steals: Vec::new(),
        }
    }

    /// Attach a voice budget and stealing policy. Only `spawn_voice_managed`
    /// / `spawn_voice_on_bus_managed` enforce it; the plain `spawn_voice` /
    /// `spawn_voice_on_bus` remain unconditional.
    pub fn set_voice_allocator(&mut self, alloc: VoiceAllocator) {
        self.voice_alloc = Some(alloc);
    }

    /// Give every bus a budget of `budget` sounding voices under `policy`
    /// ([`BudgetScope::PerGroup`] with each bus as its group), enforced by
    /// the managed spawn paths; `budget == 0` detaches the allocator. Live
    /// voices are counted from the start (each against its bus; voices
    /// already fading after a steal are not). A steal fades its victim out
    /// over 5 ms in its slot, so keep `budget` below
    /// [`ugens::MAX_BUS_INPUTS`] to leave fading voices room (see
    /// [`spawn_voice_on_bus_managed`](Self::spawn_voice_on_bus_managed)).
    pub fn set_voice_budget(&mut self, budget: usize, policy: StealPolicy) {
        if budget == 0 {
            self.voice_alloc = None;
            return;
        }
        let mut alloc = VoiceAllocator::new(budget, policy).with_scope(BudgetScope::PerGroup);
        for v in self.voices.iter().filter(|v| v.fade.is_none()) {
            alloc.record_spawn_in(v.id, v.bus.map_or(OFF_BUS_GROUP, bus_group));
        }
        self.voice_alloc = Some(alloc);
    }

    /// Detach the current voice allocator, if any.
    pub fn clear_voice_allocator(&mut self) {
        self.voice_alloc = None;
    }

    /// The currently attached voice allocator, if any.
    pub fn voice_allocator(&self) -> Option<&VoiceAllocator> {
        self.voice_alloc.as_ref()
    }

    /// Mutably access the currently attached voice allocator, if any.
    pub fn voice_allocator_mut(&mut self) -> Option<&mut VoiceAllocator> {
        self.voice_alloc.as_mut()
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
        let done_plan = DonePlan::build(def, &ugens);

        // Reseed this voice's noise UGens deterministically from the spawn order
        // (a no-op for non-noise UGens), then add all nodes to the graph,
        // collecting their live NodeIds.
        let spawn_seq = self.noise_spawn_seq;
        let mut node_ids: Vec<NodeId> = Vec::with_capacity(ugens.len());
        for (i, mut ugen) in ugens.into_iter().enumerate() {
            ugen.reseed_noise(derive_noise_seed(spawn_seq, i));
            let id = self.graph.add_node(ugen);
            node_ids.push(id);
        }
        self.noise_spawn_seq = self.noise_spawn_seq.wrapping_add(1);

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

        // Build audio input mapping: name → live NodeId
        let audio_input_nodes: Vec<(String, NodeId)> = def
            .audio_inputs()
            .iter()
            .map(|(name, node_index)| (name.clone(), node_ids[*node_index]))
            .collect();

        let output_id = node_ids[def.output_node()];
        let synth = Synth::new(
            String::from(def.name()),
            node_ids,
            output_id,
            params,
            audio_input_nodes,
            done_plan,
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
            pan_nodes: None,
            born: self.context.sample_offset,
            bus: None,
            fade: None,
        });
        id
    }

    /// Instantiate a SynthDef as a voice and connect its output to a Bus node.
    /// Automatically finds the next available input slot on the bus.
    /// Returns the VoiceId, or None if the bus has no free slots.
    ///
    /// Equivalent to [`spawn_voice_on_bus_panned`](Self::spawn_voice_on_bus_panned)
    /// with `pan = 0.0` (center) — same direct connection, no `Pan2` inserted.
    pub fn spawn_voice_on_bus(&mut self, def: &SynthDef, bus_node: NodeId) -> Option<VoiceId> {
        self.spawn_voice_on_bus_panned(def, bus_node, 0.0)
    }

    /// Like [`spawn_voice_on_bus`](Self::spawn_voice_on_bus), but places the
    /// voice in the stereo field via a `Pan2` node inserted between the
    /// voice's output and the bus input slot. `pan` is the pan position
    /// (-1.0 = left, 0.0 = center, +1.0 = right); out-of-range values are
    /// clamped by `Pan2` itself.
    ///
    /// At exactly `pan == 0.0` this connects the voice directly to the bus
    /// with no `Pan2` in between — byte-for-byte the same signal path this
    /// engine has always used for center-panned voices, so existing
    /// center-pan callers (including `spawn_voice_on_bus`) are unaffected by
    /// this method's existence: a mono voice's samples land unscaled and
    /// identical on every bus output channel via the bus's own channel-wrap,
    /// exactly as before `Pan2` existed in this path.
    pub fn spawn_voice_on_bus_panned(
        &mut self,
        def: &SynthDef,
        bus_node: NodeId,
        pan: f32,
    ) -> Option<VoiceId> {
        // Find the next free input slot on the bus
        let bus_max = self.graph.node_spec(bus_node)?.inputs.len();

        // Find which slots are already used. Read this from the graph's OWN live edges
        // (`used_input_slots`) rather than scanning `self.voices`' bookkeeping: a routing
        // effect's connection to this same bus (wired directly via `self.graph.connect`,
        // e.g. from `build_routing`) has no entry in `self.voices` at all, so a
        // `self.voices`-only search cannot see it -- the very first voice spawned onto a
        // bus a routing effect also targets would compute the SAME "free" slot the effect
        // already occupies, and `connect`'s last-writer-wins semantics would then silently
        // evict the effect's connection, dropping its entire contribution to that bus for
        // the rest of the render. Reading the graph directly makes voice spawns and
        // routing-effect wiring agree on occupancy by construction, and self-maintains
        // across the render's life: freeing a voice (`free_voice`/`free_done_synths`, both
        // via `remove_node`) already strips its edge from the graph, so its slot reopens
        // here automatically, while a routing effect's edge -- never touched by either free
        // path -- stays occupied for as long as it lives.
        let used_slots: Vec<usize> = self.graph.used_input_slots(bus_node).collect();

        let free_slot = (0..bus_max).find(|slot| !used_slots.contains(slot))?;

        let synth = self.instantiate_synthdef(def);
        let id = self.scheduler.alloc_voice_id();

        let pan_nodes = if pan == 0.0 {
            self.graph.connect(synth.output_node(), bus_node, free_slot);
            None
        } else {
            let const_node = self
                .graph
                .add_node(alloc::boxed::Box::new(ugens::Const::new(pan)));
            let pan2_node = self
                .graph
                .add_node(alloc::boxed::Box::new(ugens::Pan2::new()));
            self.graph.connect(synth.output_node(), pan2_node, 0);
            self.graph.connect(const_node, pan2_node, 1);
            self.graph.connect(pan2_node, bus_node, free_slot);
            Some((pan2_node, const_node))
        };

        self.voices.push(Voice {
            id,
            synth: synth.clone_handle(),
            pan_nodes,
            born: self.context.sample_offset,
            bus: Some(bus_node),
            fade: None,
        });
        Some(id)
    }

    /// Spawn a voice through the attached voice allocator, if any.
    ///
    /// Enforces its budget and stealing policy: may steal an existing voice
    /// first to make room (see [`spawn_voice_on_bus_managed`]
    /// (Self::spawn_voice_on_bus_managed) for what a steal does), or decline
    /// the spawn entirely. Falls back to `spawn_voice`'s unconditional
    /// behavior when no allocator is attached.
    ///
    /// Returns `None` only when an allocator is attached, its budget is
    /// full, and its policy declines to make room (`StealPolicy::Reject`).
    pub fn spawn_voice_managed(&mut self, def: &SynthDef) -> Option<VoiceId> {
        match self.request_admission(OFF_BUS_GROUP) {
            Some(Admission::Reject) => return None,
            Some(Admission::Steal(victim)) => self.steal_voice(victim),
            Some(Admission::Admit) | None => {}
        }
        let id = self.spawn_voice(def);
        if let Some(alloc) = self.voice_alloc.as_mut() {
            alloc.record_spawn_in(id, OFF_BUS_GROUP);
        }
        Some(id)
    }

    /// Like `spawn_voice_managed`, but connects the voice to a bus (see
    /// `spawn_voice_on_bus`).
    ///
    /// **A steal fades, it does not cut.** The victim leaves the allocator's
    /// count at once, but stays in the graph behind a [`ugens::StealFade`]
    /// spliced into its bus slot: its output falls to silence along a raised
    /// cosine over [`ugens::steal_fade_samples`] (221 samples at 44.1 kHz,
    /// 5 ms), and [`free_done_synths`](Self::free_done_synths) frees it once
    /// the fade is done. Until then it keeps its slot. A victim with no bus
    /// (spawned through `spawn_voice_managed`) has nothing to fade through
    /// and is freed at once.
    ///
    /// **A full bus.** If the bus has no free slot, an allocator-managed
    /// spawn makes one rather than fail: it frees the fading voice on this
    /// bus that started fading earliest, else (under any policy but
    /// `Reject`) the voice on this bus the policy picks, freed at once. Both
    /// are recorded as cut steals. Leave the budget below the bus's slot
    /// count so stolen voices have room to finish their fades.
    ///
    /// Returns `None` because the allocator's policy declined the spawn, or
    /// because the bus had no free slot and nothing on it could be freed.
    /// These two `None` cases are not equivalent for side effects: a steal
    /// the budget called for happens before the bus is checked for a free
    /// slot, so a `None` return does not mean nothing happened.
    pub fn spawn_voice_on_bus_managed(
        &mut self,
        def: &SynthDef,
        bus_node: NodeId,
    ) -> Option<VoiceId> {
        self.spawn_voice_on_bus_panned_managed(def, bus_node, 0.0)
    }

    /// [`spawn_voice_on_bus_managed`](Self::spawn_voice_on_bus_managed) at a
    /// stereo position (see [`spawn_voice_on_bus_panned`]
    /// (Self::spawn_voice_on_bus_panned)).
    pub fn spawn_voice_on_bus_panned_managed(
        &mut self,
        def: &SynthDef,
        bus_node: NodeId,
        pan: f32,
    ) -> Option<VoiceId> {
        let group = bus_group(bus_node);
        match self.request_admission(group) {
            Some(Admission::Reject) => return None,
            Some(Admission::Steal(victim)) => self.steal_voice(victim),
            Some(Admission::Admit) | None => {}
        }
        let id = loop {
            if let Some(id) = self.spawn_voice_on_bus_panned(def, bus_node, pan) {
                break id;
            }
            if !self.reclaim_bus_slot(bus_node) {
                return None;
            }
        };
        if let Some(alloc) = self.voice_alloc.as_mut() {
            alloc.record_spawn_in(id, group);
        }
        Some(id)
    }

    /// Steal `victim`: drop it from the allocator's count and fade it out
    /// through its bus slot, or free it at once if it has no slot to fade
    /// through. Records the steal.
    fn steal_voice(&mut self, victim: VoiceId) {
        let level = self.voice_level(victim);
        if let Some(alloc) = self.voice_alloc.as_mut() {
            alloc.record_free(victim);
        }
        let record = self.steals.len();
        let faded = self.begin_fade(victim, record);
        if !faded {
            self.free_voice(victim);
        }
        self.steals.push(StealRecord {
            sample: self.context.sample_offset,
            level,
            cut: !faded,
        });
    }

    /// Splice a `StealFade` between `voice` and its bus slot. Returns false
    /// (and changes nothing) if the voice is unknown, already fading, or not
    /// connected to a bus.
    fn begin_fade(&mut self, voice_id: VoiceId, record: usize) -> bool {
        let Some(voice) = self.voices.iter().find(|v| v.id == voice_id) else {
            return false;
        };
        let Some(bus) = voice.bus else {
            return false;
        };
        if voice.fade.is_some() {
            return false;
        }
        let feeder = match voice.pan_nodes {
            Some((pan2, _)) => pan2,
            None => voice.synth.output_node(),
        };
        let Some(slot) = self.graph.input_slot_from(feeder, bus) else {
            return false;
        };
        let len = ugens::steal_fade_samples(self.context.sample_rate);
        let fade = self
            .graph
            .add_node(alloc::boxed::Box::new(ugens::StealFade::new(len)));
        self.graph.connect(feeder, fade, 0);
        self.graph.connect(fade, bus, slot);
        let started = self.context.sample_offset;
        if let Some(v) = self.voices.iter_mut().find(|v| v.id == voice_id) {
            v.fade = Some((fade, started, record));
        }
        true
    }

    /// Free one voice on a full `bus` so a managed spawn can take its slot:
    /// the earliest-started fading voice, else a live one the allocator's
    /// policy picks. Returns false if there is nothing to free (no allocator,
    /// or neither kind of voice on this bus).
    fn reclaim_bus_slot(&mut self, bus: NodeId) -> bool {
        if self.voice_alloc.is_none() {
            return false;
        }
        let fading = self
            .voices
            .iter()
            .filter(|v| v.bus == Some(bus))
            .filter_map(|v| v.fade.map(|(_, started, record)| (v.id, started, record)))
            .min_by_key(|&(_, started, _)| started);
        if let Some((id, _, record)) = fading {
            // Already recorded as a steal; freeing it before its fade ends
            // turns that steal into a cut.
            self.free_voice(id);
            if let Some(rec) = self.steals.get_mut(record) {
                rec.cut = true;
            }
            return true;
        }
        let on_bus: Vec<(VoiceId, Option<f32>)> = self
            .voices
            .iter()
            .filter(|v| v.bus == Some(bus) && v.fade.is_none())
            .map(|v| (v.id, self.voice_level(v.id)))
            .collect();
        let victim = self.voice_alloc.as_ref().and_then(|a| {
            a.pick_victim(
                |v| on_bus.iter().any(|(id, _)| *id == v),
                |v| on_bus.iter().find(|(id, _)| *id == v).and_then(|(_, l)| *l),
            )
        });
        let Some(id) = victim else {
            return false;
        };
        let level = self.voice_level(id);
        self.free_voice(id);
        self.steals.push(StealRecord {
            sample: self.context.sample_offset,
            level,
            cut: true,
        });
        true
    }

    /// Every steal the managed spawn paths have made, in order.
    pub fn steals(&self) -> &[StealRecord] {
        &self.steals
    }

    /// Ask the attached allocator (if any) whether a new voice may spawn,
    /// giving it each voice's current level when its policy needs one.
    fn request_admission(&mut self, group: u64) -> Option<Admission> {
        let quietest = self
            .voice_alloc
            .as_ref()
            .is_some_and(|a| a.policy() == StealPolicy::Quietest);
        let levels: Vec<(VoiceId, Option<f32>)> = if quietest {
            self.voices
                .iter()
                .map(|v| (v.id, self.voice_level(v.id)))
                .collect()
        } else {
            Vec::new()
        };
        let level = |id: VoiceId| levels.iter().find(|(v, _)| *v == id).and_then(|(_, l)| *l);
        self.voice_alloc
            .as_mut()
            .map(|alloc| alloc.request_in(group, level))
    }

    /// A voice's current output level: the RMS, over every channel, of its
    /// output node's last rendered block (before any pan or bus). `None` if
    /// the voice does not exist or has not rendered a block since it was
    /// spawned (its output buffer holds nothing of its own yet).
    ///
    /// This is the level [`StealPolicy::Quietest`] steals by. It reads the
    /// signal itself rather than an envelope: a sampled note's envelope can
    /// sit at full sustain while the recording under it has decayed away,
    /// and a synth voice has no single envelope to read. It costs nothing
    /// until a steal asks for it.
    pub fn voice_level(&self, voice_id: VoiceId) -> Option<f32> {
        let voice = self.voices.iter().find(|v| v.id == voice_id)?;
        if self.context.sample_offset <= voice.born {
            return None;
        }
        let buf = self.graph.node_output(voice.synth.output_node())?;
        let (mut sum, mut n) = (0.0f64, 0usize);
        for ch in 0..buf.num_channels() {
            for &x in buf.channel(ch).samples() {
                sum += f64::from(x) * f64::from(x);
                n += 1;
            }
        }
        (n > 0).then(|| (sum / n as f64).sqrt() as f32)
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
            if let Some((pan2_node, const_node)) = voice.pan_nodes {
                self.graph.remove_node(pan2_node);
                self.graph.remove_node(const_node);
            }
            if let Some((fade, _, _)) = voice.fade {
                self.graph.remove_node(fade);
            }
        }
        // Keep the allocator's bookkeeping accurate regardless of whether
        // this voice was spawned through the managed path.
        if let Some(alloc) = self.voice_alloc.as_mut() {
            alloc.record_free(voice_id);
        }
    }

    /// Get the Synth handle for a voice by VoiceId.
    pub fn voice_synth(&self, voice_id: VoiceId) -> Option<&Synth> {
        self.voices
            .iter()
            .find(|v| v.id == voice_id)
            .map(|v| &v.synth)
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
        self.synths
            .retain(|s| s.output_node() != synth.output_node());
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
                EventAction::SetParam {
                    voice,
                    param,
                    value,
                } => {
                    self.set_voice_param(voice, &param, value);
                }
                EventAction::SetGate { voice, value } => {
                    self.set_voice_param(voice, "gate", value);
                }
                EventAction::SetParamGlide {
                    voice,
                    param,
                    target,
                    glide_secs,
                    shape,
                    space,
                } => {
                    self.set_voice_param_glide(voice, &param, target, glide_secs, shape, space);
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
                for (ch, out_ch) in output.iter_mut().enumerate() {
                    out_ch.extend_from_slice(buf.channel(ch).samples());
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
        shape: GlideShape,
        space: GlideSpace,
    ) -> bool {
        if let Some(node_id) = synth.param_node(name) {
            self.graph
                .set_node_target(node_id, target, glide_secs, shape, space)
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
        shape: GlideShape,
        space: GlideSpace,
    ) -> bool {
        if let Some(voice) = self.voices.iter().find(|v| v.id == voice_id) {
            if let Some(node_id) = voice.synth.param_node(name) {
                self.graph
                    .set_node_target(node_id, target, glide_secs, shape, space)
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

    /// Remove every synth whose output has been silenced by a done-reporting
    /// node ("done actions": when the amp envelope finishes, the whole synth
    /// is removed from the graph).
    ///
    /// The rule: a synth is done when a node that reports `is_done()` sits on
    /// its *output path* and that node's silence reaches the output. Silence
    /// propagates from a node to the output through
    /// - either factor of a `Mul` (the usual `env * signal` VCA), the
    ///   numerator of a `Div`, the input of a `Neg`;
    /// - an `Add`/`Sub` only when both terms are silent (a sum keeps
    ///   sounding while either term does);
    /// - the `in` port of a pass-through processor (filters, dynamics,
    ///   waveshapers, panners, short modulation effects).
    ///
    /// It does not propagate through parameter ports (`freq`, `cutoff`,
    /// `gate`, …), out of oscillators, physical models or envelopes, or
    /// through nodes that keep sounding after their input stops (delay
    /// lines, comb filters, reverbs, spectral processors). So a one-shot on a
    /// side chain does not free the voice: a `perc` exciting a
    /// `feedbackDelay` string, an `xLine` pitch drop into a `freq` port, a
    /// `pluck` summed with a sustained voice. The output path is computed
    /// once per synth at instantiation; a synth with no done-reporting node
    /// on it is never freed here (free it with `free_voice`).
    ///
    /// Also frees every stolen voice whose fade-out has finished (see
    /// [`spawn_voice_on_bus_managed`](Self::spawn_voice_on_bus_managed)).
    ///
    /// Returns the number of synths removed.
    /// After calling this, you must call `prepare()` before rendering.
    pub fn free_done_synths(&mut self) -> usize {
        let mut removed = 0;
        // Stolen voices whose fade has run out (see
        // `spawn_voice_on_bus_managed`).
        let faded: Vec<VoiceId> = self
            .voices
            .iter()
            .filter(|v| v.fade.is_some_and(|(f, _, _)| self.graph.node_is_done(f)))
            .map(|v| v.id)
            .collect();
        for id in faded {
            self.free_voice(id);
            removed += 1;
        }
        let mut i = 0;
        while i < self.synths.len() {
            let is_done = self.synths[i].output_is_done(&self.graph);
            if is_done {
                let synth = self.synths.remove(i);
                // Also remove from voices list, keeping the allocator's
                // bookkeeping in sync for any voice IDs it was tracking.
                let freed_ids: Vec<VoiceId> = self
                    .voices
                    .iter()
                    .filter(|v| v.synth.output_node() == synth.output_node())
                    .map(|v| v.id)
                    .collect();
                // Also tear down any Pan2/Const nodes a panned voice owns —
                // otherwise they'd leak in the graph once their voice's entry
                // is gone (see `free_voice`, which does the same cleanup on
                // the explicit-free path).
                let freed_pan_nodes: Vec<(NodeId, NodeId)> = self
                    .voices
                    .iter()
                    .filter(|v| v.synth.output_node() == synth.output_node())
                    .filter_map(|v| v.pan_nodes)
                    .collect();
                // And the fade a stolen voice was playing through, if it
                // finished on its own first.
                let freed_fades: Vec<NodeId> = self
                    .voices
                    .iter()
                    .filter(|v| v.synth.output_node() == synth.output_node())
                    .filter_map(|v| v.fade.map(|(f, _, _)| f))
                    .collect();
                self.voices
                    .retain(|v| v.synth.output_node() != synth.output_node());
                if let Some(alloc) = self.voice_alloc.as_mut() {
                    for id in freed_ids {
                        alloc.record_free(id);
                    }
                }
                for &id in synth.node_ids() {
                    self.graph.remove_node(id);
                }
                for (pan2_node, const_node) in freed_pan_nodes {
                    self.graph.remove_node(pan2_node);
                    self.graph.remove_node(const_node);
                }
                for fade in freed_fades {
                    self.graph.remove_node(fade);
                }
                removed += 1;
            } else {
                i += 1;
            }
        }
        removed
    }

    /// Schedule a gate-on at `on_time` and auto gate-off after `duration_samples`.
    ///
    /// Both times are absolute sample offsets. Use `TimeConfig::position_to_samples`
    /// and `TimeConfig::steps_to_samples` to convert from musical time.
    pub fn schedule_note(&mut self, voice: VoiceId, on_time: u64, duration_samples: u64) {
        self.scheduler
            .schedule_note(voice, on_time, on_time + duration_samples);
    }

    /// Schedule a note with attack-aligned pre-trigger.
    ///
    /// `grid_time` is where the attack peak should land (absolute sample offset).
    /// The gate-on is scheduled at `grid_time - attack_secs` so the envelope
    /// reaches full level at the musical grid position.
    /// Gate-off is scheduled at `grid_time + duration_samples`.
    pub fn schedule_note_aligned(
        &mut self,
        voice: VoiceId,
        grid_time: u64,
        attack_secs: f32,
        duration_samples: u64,
    ) {
        let attack_samples = (attack_secs * self.context.sample_rate) as u64;
        let on_time = grid_time.saturating_sub(attack_samples);
        self.scheduler
            .schedule_note(voice, on_time, grid_time + duration_samples);
    }

    /// Current sample offset (monotonic time counter).
    pub fn sample_offset(&self) -> u64 {
        self.context.sample_offset
    }

    /// Current time in seconds.
    pub fn time_secs(&self) -> f64 {
        self.context.time_secs()
    }

    // -- Routing graph integration --

    /// Build a routing graph into the engine's audio graph.
    ///
    /// Creates Bus UGen nodes for each bus, instantiates effect SynthDefs,
    /// wires source bus outputs to effect AudioIn nodes, wires effect outputs
    /// to target bus input slots, and sets the main bus as the graph's sink.
    ///
    /// `effect_defs` is a lookup slice: for each effect, the def whose name
    /// matches is used. Call `prepare()` after this method.
    pub fn build_routing(&mut self, routing: &mut RoutingGraph, effect_defs: &[SynthDef]) {
        // 1. Create Bus UGen nodes for each bus
        for bus_id in routing.bus_ids().collect::<Vec<_>>() {
            let (_, channels) = routing.bus_info(bus_id).unwrap();
            let bus_node = self.graph.add_node(alloc::boxed::Box::new(ugens::Bus::new(
                ugens::ChannelCount::Custom(channels),
            )));
            routing.set_bus_node(bus_id, bus_node);
        }

        // 2. Set main bus as the graph's sink
        if let Some(main_node) = routing.bus_node(routing.main_bus()) {
            self.graph.set_sink(main_node);
        }

        // 3. Instantiate effects and wire them up
        for i in 0..routing.num_effects() {
            let source_bus = routing.effects()[i].source_bus;
            let target_bus = routing.effects()[i].target_bus;
            let def_name = routing.effects()[i].def_name.clone();

            // Find the matching SynthDef
            let def = match effect_defs.iter().find(|d| d.name() == def_name) {
                Some(d) => d,
                None => continue, // skip if def not found
            };

            // Instantiate the effect SynthDef
            let synth = self.instantiate_synthdef(def);

            // Wire source bus output → effect's AudioIn node
            if let Some(source_node) = routing.bus_node(source_bus)
                && let Some(audio_in_node) = synth.audio_input_node("in")
            {
                self.graph.connect(source_node, audio_in_node, 0);
            }

            // Wire effect output → target bus input slot
            if let Some(target_node) = routing.bus_node(target_bus) {
                // Find a free input slot on the target bus
                let bus_max = match self.graph.node_spec(target_node) {
                    Some(spec) => spec.inputs.len(),
                    None => continue,
                };
                let used_slots = self.count_connections_to(target_node);
                if used_slots < bus_max {
                    self.graph
                        .connect(synth.output_node(), target_node, used_slots);
                }
            }

            // Store the synth handle
            routing.effects_mut()[i].synth = Some(synth);
        }
    }

    /// Spawn a voice on a routing bus.
    ///
    /// Delegates to `spawn_voice_on_bus` using the bus's live NodeId.
    pub fn spawn_voice_on_routing_bus(
        &mut self,
        def: &SynthDef,
        routing: &RoutingGraph,
        bus_id: BusId,
    ) -> Option<VoiceId> {
        let bus_node = routing.bus_node(bus_id)?;
        self.spawn_voice_on_bus(def, bus_node)
    }

    /// [`spawn_voice_on_routing_bus`](Self::spawn_voice_on_routing_bus)
    /// through the attached allocator (see
    /// [`spawn_voice_on_bus_managed`](Self::spawn_voice_on_bus_managed)).
    pub fn spawn_voice_on_routing_bus_managed(
        &mut self,
        def: &SynthDef,
        routing: &RoutingGraph,
        bus_id: BusId,
    ) -> Option<VoiceId> {
        let bus_node = routing.bus_node(bus_id)?;
        self.spawn_voice_on_bus_managed(def, bus_node)
    }

    /// Set a named parameter on an effect by EffectId.
    /// Returns true if the effect and parameter were found.
    pub fn set_effect_param(
        &mut self,
        routing: &RoutingGraph,
        effect_id: EffectId,
        name: &str,
        value: f32,
    ) -> bool {
        if let Some(synth) = routing.effect_synth(effect_id)
            && let Some(node_id) = synth.param_node(name)
        {
            return self.graph.set_node_value(node_id, value);
        }
        false
    }

    /// Set a named parameter on an effect with a smooth glide.
    /// Returns true if the effect and parameter were found.
    #[allow(clippy::too_many_arguments)] // Routing target, glide, and shape/space are each meaningful and independent.
    pub fn set_effect_param_glide(
        &mut self,
        routing: &RoutingGraph,
        effect_id: EffectId,
        name: &str,
        target: f32,
        glide_secs: f32,
        shape: GlideShape,
        space: GlideSpace,
    ) -> bool {
        if let Some(synth) = routing.effect_synth(effect_id)
            && let Some(node_id) = synth.param_node(name)
        {
            return self
                .graph
                .set_node_target(node_id, target, glide_secs, shape, space);
        }
        false
    }

    /// Count how many edges connect to a given node (for finding free bus slots).
    fn count_connections_to(&self, node_id: NodeId) -> usize {
        self.graph.edges_to(node_id)
    }
}
