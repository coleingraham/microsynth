use crate::buffer::AudioBuffer;
use crate::context::ProcessContext;
use crate::curve::{GlideShape, GlideSpace};
use crate::node::{NodeId, UGen};
use alloc::boxed::Box;
use alloc::vec::Vec;

/// A connection between two nodes in the graph.
#[derive(Debug, Clone)]
pub struct Edge {
    /// Source node.
    pub from: NodeId,
    /// Destination node.
    pub to: NodeId,
    /// Which input port on the destination node.
    pub to_input: usize,
}

/// Internal storage for a node in the graph.
struct NodeSlot {
    ugen: Box<dyn UGen>,
    /// Pre-allocated output buffer for this node.
    output: AudioBuffer,
    /// The node's output over a whole block, built up piece by piece while
    /// the engine renders a block in pieces (`collect_piece`).
    assembled: AudioBuffer,
    /// Whether `assembled` holds pieces of the current block.
    assembled_valid: bool,
    /// Resolved output channel count after multichannel expansion.
    output_channels: usize,
    /// Whether this node has been processed in the current block.
    processed: bool,
    /// Whether `UGen::init` has been called on this node yet. `prepare()`
    /// calls `init` at most once per node (see `init_nodes`): a node index
    /// is never reused (`add_node` only ever pushes, `remove_node` tombstones
    /// in place), so this flag's lifetime matches the one `UGen` instance
    /// living at this slot and can never leak onto a later, unrelated node.
    initialized: bool,
    /// Whether the node runs. An inactive node is not processed: its output
    /// is zeros of the block's length until it is activated (a voice spawned
    /// deferred, see `Engine::start_voice_at`). Every node starts active.
    active: bool,
}

/// A directed acyclic audio processing graph.
///
/// Nodes are connected via edges. Rendering evaluates nodes in topological
/// order (sources first, sink last) using a pull-based model — only nodes
/// upstream of the sink are evaluated.
///
/// The graph supports runtime modification: nodes and edges can be added
/// or removed between render calls. After modification, call `prepare()`
/// to re-resolve channel counts and re-sort.
pub struct AudioGraph {
    nodes: Vec<Option<NodeSlot>>,
    edges: Vec<Edge>,
    /// Cached topological order (indices into `nodes`).
    topo_order: Vec<NodeId>,
    /// Resolved input wiring, indexed by node index. For each node,
    /// `input_sources[idx][port]` is the source node feeding that input port
    /// (or `None` if unconnected). Rebuilt in `prepare()` so that `render()`
    /// never has to scan the edge list — keeping render O(nodes), not
    /// O(nodes × edges).
    input_sources: Vec<Vec<Option<NodeId>>>,
    /// Reusable scratch for gathering input buffer pointers during render,
    /// so the render path performs no per-node allocation.
    input_ptr_scratch: Vec<*const AudioBuffer>,
    /// The output node we pull from.
    sink: Option<NodeId>,
    /// Whether the graph needs re-sorting.
    dirty: bool,
    /// Reusable scratch space for topological sort.
    topo_scratch: TopoScratch,
}

/// Scratch buffers for topological sort to avoid allocation.
struct TopoScratch {
    in_degree: Vec<u32>,
    queue: Vec<NodeId>,
    /// Out-adjacency in compressed form: the targets of node `i` are
    /// `adj_targets[adj_start[i]..adj_start[i + 1]]`, in edge-list order.
    adj_start: Vec<u32>,
    adj_targets: Vec<NodeId>,
    /// Per-node input channel counts, reused by `resolve_channels`.
    input_channels: Vec<usize>,
}

impl TopoScratch {
    fn new() -> Self {
        TopoScratch {
            in_degree: Vec::new(),
            queue: Vec::new(),
            adj_start: Vec::new(),
            adj_targets: Vec::new(),
            input_channels: Vec::new(),
        }
    }
}

impl Default for AudioGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        AudioGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            topo_order: Vec::new(),
            input_sources: Vec::new(),
            input_ptr_scratch: Vec::new(),
            sink: None,
            dirty: true,
            topo_scratch: TopoScratch::new(),
        }
    }

    /// Add a UGen to the graph. Returns its `NodeId`.
    pub fn add_node(&mut self, ugen: Box<dyn UGen>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Some(NodeSlot {
            ugen,
            output: AudioBuffer::new(1, 1), // placeholder, resized in prepare()
            assembled: AudioBuffer::new(1, 1), // placeholder, resized in prepare()
            assembled_valid: false,
            output_channels: 1,
            processed: false,
            initialized: false,
            active: true,
        }));
        self.dirty = true;
        id
    }

    /// Remove a node from the graph. Also removes all edges to/from it.
    pub fn remove_node(&mut self, id: NodeId) {
        if let Some(slot) = self.nodes.get_mut(id.index()) {
            *slot = None;
        }
        self.edges.retain(|e| e.from != id && e.to != id);
        if self.sink == Some(id) {
            self.sink = None;
        }
        self.dirty = true;
    }

    /// Remove several nodes at once, with one pass over the edge list.
    ///
    /// Leaves the graph exactly as calling [`remove_node`](Self::remove_node)
    /// on each id in turn would (same remaining edges in the same order, same
    /// sink), but costs O(ids + nodes + edges) rather than O(ids × edges).
    /// Freeing a voice removes all of its nodes, so this is the voice path.
    pub fn remove_nodes(&mut self, ids: &[NodeId]) {
        if ids.is_empty() {
            return;
        }
        // Ids past the node list are kept aside rather than sized for, so
        // `NodeId(u32::MAX)` costs nothing; an edge naming one is still
        // dropped, just as `remove_node` would drop it.
        let mut removed = alloc::vec![false; self.nodes.len()];
        let mut beyond: Vec<NodeId> = Vec::new();
        for &id in ids {
            match self.nodes.get_mut(id.index()) {
                Some(slot) => {
                    *slot = None;
                    removed[id.index()] = true;
                }
                None => beyond.push(id),
            }
            if self.sink == Some(id) {
                self.sink = None;
            }
        }
        let gone = |n: NodeId| match removed.get(n.index()) {
            Some(&r) => r,
            None => beyond.contains(&n),
        };
        self.edges.retain(|e| !gone(e.from) && !gone(e.to));
        self.dirty = true;
    }

    /// Connect `from` node's output to `to` node's input port `input_index`.
    pub fn connect(&mut self, from: NodeId, to: NodeId, input_index: usize) {
        // Remove any existing connection to this input
        self.edges
            .retain(|e| !(e.to == to && e.to_input == input_index));

        self.edges.push(Edge {
            from,
            to,
            to_input: input_index,
        });
        self.dirty = true;
    }

    /// The input port of `to` that `from` feeds, if they are connected (the
    /// first such edge when `from` feeds several ports of `to`).
    pub fn input_slot_from(&self, from: NodeId, to: NodeId) -> Option<usize> {
        self.edges
            .iter()
            .find(|e| e.from == from && e.to == to)
            .map(|e| e.to_input)
    }

    /// Disconnect a specific input port.
    pub fn disconnect(&mut self, to: NodeId, input_index: usize) {
        self.edges
            .retain(|e| !(e.to == to && e.to_input == input_index));
        self.dirty = true;
    }

    /// Set the sink (output) node. This is the node whose output is
    /// returned by `render()`.
    pub fn set_sink(&mut self, node: NodeId) {
        self.sink = Some(node);
        self.dirty = true;
    }

    /// Prepare the graph for rendering: topological sort, resolve
    /// multichannel expansion, allocate output buffers, and init nodes.
    ///
    /// Must be called after any structural changes and before `render()`.
    pub fn prepare(&mut self, context: &ProcessContext) {
        self.topological_sort();
        self.resolve_inputs();
        self.resolve_channels(context);
        self.init_nodes(context);
        self.dirty = false;
    }

    /// Resolve, once, which source node feeds each input port of each node.
    /// This moves the per-input edge-list scan out of the per-block render
    /// path and into `prepare()` (called only after structural changes).
    ///
    /// Also the fail-loud checkpoint for required ports: a `required` port
    /// (`InputSpec::required`) with no connected source panics *here*, before
    /// `render()` (and therefore `UGen::process`) ever sees this graph state.
    /// The alternative — degrading a missing required input to a zero/silent
    /// buffer, or leaving it for `process` to discover mid-render — would
    /// trade one silent-misrouting hazard for another; see this crate's
    /// `UGen::process` doc and `InputSpec::required`.
    fn resolve_inputs(&mut self) {
        let n = self.nodes.len();
        self.input_sources.clear();
        self.input_sources.resize(n, Vec::new());

        // Size each live node's port table, then place every edge in one pass
        // over the edge list: O(edges + ports), not O(ports × edges). A bus
        // has hundreds of ports, so a per-port edge scan made every
        // `prepare()` cost ports × edges per bus.
        //
        // `seen` reproduces the per-port scan exactly: the FIRST edge listed
        // for a (node, port) decides it, even when its source has since been
        // removed (the port then stays unconnected). `connect` keeps at most
        // one edge per (node, port), so in practice there is only one.
        let mut seen: Vec<Vec<bool>> = Vec::with_capacity(n);
        for idx in 0..n {
            let ports = self.nodes[idx]
                .as_ref()
                .map_or(0, |slot| slot.ugen.spec().inputs.len());
            self.input_sources[idx].resize(ports, None);
            seen.push(alloc::vec![false; ports]);
        }
        for edge in &self.edges {
            let (to, port) = (edge.to.index(), edge.to_input);
            if to >= n || port >= seen[to].len() || seen[to][port] {
                continue;
            }
            seen[to][port] = true;
            if self.nodes[edge.from.index()].is_some() {
                self.input_sources[to][port] = Some(edge.from);
            }
        }

        for idx in 0..n {
            let spec = match &self.nodes[idx] {
                Some(slot) => slot.ugen.spec(),
                None => continue,
            };
            for (input_idx, input_spec) in spec.inputs.iter().enumerate() {
                if input_spec.required && self.input_sources[idx][input_idx].is_none() {
                    panic!(
                        "AudioGraph::prepare: node {} ({}) has an unconnected required \
                         input port {} ({}) — connect a source to this port before \
                         calling prepare()",
                        idx, spec.name, input_idx, input_spec.name
                    );
                }
            }
        }
    }

    /// Render one block of audio by evaluating nodes in topological order.
    ///
    /// Returns a reference to the sink node's output buffer, or `None` if
    /// no sink is set or the graph is dirty.
    pub fn render(&mut self, context: &ProcessContext) -> Option<&AudioBuffer> {
        if self.dirty || self.sink.is_none() {
            return None;
        }

        // Reset processed flags
        for slot in self.nodes.iter_mut().flatten() {
            slot.processed = false;
        }

        // Process nodes in topological order. Inputs were resolved once in
        // `prepare()` (see `input_sources`), so here we only gather the source
        // output pointers — no per-block edge scan, no per-node allocation.
        for order_idx in 0..self.topo_order.len() {
            let node_id = self.topo_order[order_idx];
            let idx = node_id.index();

            if self.nodes[idx].is_none() {
                continue;
            }

            // Gather source output buffer pointers into reusable scratch, one
            // slot per input port (preserving position: an unconnected port
            // pushes a null pointer rather than being skipped, so a later
            // port's buffer never shifts down into an earlier port's slot —
            // see `resolve_inputs`'s doc and `UGen::process`'s `inputs` doc).
            // SAFETY: topological order guarantees every source node has
            // already been processed this block and will not be mutated again,
            // so these pointers stay valid for this node's `process` call.
            self.input_ptr_scratch.clear();
            for src in self.input_sources[idx].iter() {
                match src {
                    Some(src_id) => {
                        let src_slot = self.nodes[src_id.index()].as_ref().unwrap();
                        self.input_ptr_scratch
                            .push(&src_slot.output as *const AudioBuffer);
                    }
                    None => self.input_ptr_scratch.push(core::ptr::null()),
                }
            }

            // View the gathered `*const AudioBuffer` scratch (null = unconnected)
            // as `&[Option<&AudioBuffer>]`. This is sound by the null-pointer
            // optimization: `Option<&T>` is guaranteed to have the same layout
            // as a nullable `*const T`, with `None` represented by null. This
            // borrows nothing tracked, so the mutable borrow of the destination
            // node below is permitted.
            let input_refs: &[Option<&AudioBuffer>] = unsafe {
                core::slice::from_raw_parts(
                    self.input_ptr_scratch.as_ptr() as *const Option<&AudioBuffer>,
                    self.input_ptr_scratch.len(),
                )
            };

            let slot = self.nodes[idx].as_mut().unwrap();
            if slot.active {
                slot.ugen.process(context, input_refs, &mut slot.output);
            } else {
                slot.output.clear();
            }
            slot.processed = true;
        }

        // Return sink output
        let sink_id = self.sink.unwrap();
        self.nodes[sink_id.index()].as_ref().map(|s| &s.output)
    }

    /// Set the active length of every node's output buffer to the block
    /// length `context` describes, without reallocating.
    ///
    /// `render` writes `context.block_size_for_rate(rate)` samples per node,
    /// so a caller that renders a block in pieces (see `Engine::render`)
    /// retargets the lengths before each piece. `prepare()` does the same
    /// for the whole graph; this one touches only the lengths.
    pub fn set_block_len(&mut self, context: &ProcessContext) {
        for slot in self.nodes.iter_mut().flatten() {
            let rate = slot
                .ugen
                .spec()
                .outputs
                .first()
                .map_or(crate::context::Rate::Audio, |o| o.rate);
            slot.output
                .set_block_size(context.block_size_for_rate(rate));
        }
    }

    /// After a piece of a block has rendered, copy every audio-rate node's
    /// output into its whole-block buffer at `from`, so that once the last
    /// piece is in (`finish_split`) a tap on any node reads the full block,
    /// aligned with the sink's. `full` is the block length.
    pub fn collect_piece(&mut self, from: usize, full: usize) {
        for slot in self.nodes.iter_mut().flatten() {
            let audio = slot
                .ugen
                .spec()
                .outputs
                .first()
                .is_none_or(|o| o.rate == crate::context::Rate::Audio);
            if !audio {
                continue;
            }
            let channels = slot.output.num_channels();
            // Sized by `prepare()`: nothing is allocated here.
            debug_assert_eq!(slot.assembled.num_channels(), channels);
            slot.assembled.set_block_size(full);
            for ch in 0..channels {
                let src = slot.output.channel(ch).samples();
                slot.assembled.channel_mut(ch).samples_mut()[from..from + src.len()]
                    .copy_from_slice(src);
            }
            slot.assembled_valid = true;
        }
    }

    /// End a block rendered in pieces: every audio-rate node's output is its
    /// whole block again.
    pub fn finish_split(&mut self, full: usize) {
        for slot in self.nodes.iter_mut().flatten() {
            if slot.assembled_valid {
                core::mem::swap(&mut slot.output, &mut slot.assembled);
                slot.assembled_valid = false;
            } else if slot.output.block_size() > 1 {
                slot.output.set_block_size(full);
            }
        }
    }

    /// The sink node's output buffer, if a sink is set.
    pub fn sink_output(&self) -> Option<&AudioBuffer> {
        self.sink
            .and_then(|id| self.nodes[id.index()].as_ref().map(|s| &s.output))
    }

    /// Compute topological ordering using Kahn's algorithm, in O(nodes + edges).
    ///
    /// The order is fixed by three rules, and every one of them is part of
    /// what the engine renders (nodes run in this order), so they must not
    /// change:
    /// 1. an edge counts only when both its ends are live nodes; parallel
    ///    edges each count;
    /// 2. the queue is seeded with the live zero-in-degree nodes in index
    ///    order, and is first-in first-out;
    /// 3. when a node is popped, its outgoing edges are visited in edge-list
    ///    order, and a target is queued the moment its count reaches zero.
    ///
    /// Rule 3 used to be a scan of the whole edge list per popped node
    /// (O(nodes × edges)). Here the out-adjacency is built once, by a stable
    /// counting sort of the counted edges on their source, so each node's
    /// targets keep edge-list order and the visits happen in the same
    /// sequence. `topological_sort_reference` (tests only) keeps the scan.
    fn topological_sort(&mut self) {
        let n = self.nodes.len();
        let scratch = &mut self.topo_scratch;
        let nodes = &self.nodes;
        let live = |id: NodeId| nodes[id.index()].is_some();

        scratch.in_degree.clear();
        scratch.in_degree.resize(n, 0);
        scratch.queue.clear();

        // In-degrees and out-degrees of the counted edges (rule 1). The
        // out-degrees go in `adj_start[i + 1]`, to become offsets below.
        scratch.adj_start.clear();
        scratch.adj_start.resize(n + 1, 0);
        let mut counted = 0usize;
        for edge in &self.edges {
            if live(edge.to) && live(edge.from) {
                scratch.in_degree[edge.to.index()] += 1;
                scratch.adj_start[edge.from.index() + 1] += 1;
                counted += 1;
            }
        }
        for i in 0..n {
            scratch.adj_start[i + 1] += scratch.adj_start[i];
        }

        // Fill the targets in edge-list order (stable), using `cursor` as
        // each source's next free position.
        scratch.adj_targets.clear();
        scratch.adj_targets.resize(counted, NodeId(0));
        let mut cursor: Vec<u32> = scratch.adj_start[..n].to_vec();
        for edge in &self.edges {
            if live(edge.to) && live(edge.from) {
                let c = &mut cursor[edge.from.index()];
                scratch.adj_targets[*c as usize] = edge.to;
                *c += 1;
            }
        }

        // Seed with nodes that have zero in-degree (rule 2).
        for (i, slot) in self.nodes.iter().enumerate() {
            if slot.is_some() && scratch.in_degree[i] == 0 {
                scratch.queue.push(NodeId(i as u32));
            }
        }

        self.topo_order.clear();
        let mut head = 0;

        while head < scratch.queue.len() {
            let node = scratch.queue[head];
            head += 1;
            self.topo_order.push(node);

            // Rule 3. A popped node is live, so every edge out of it whose
            // target is live is exactly a counted edge.
            let (from, to) = (
                scratch.adj_start[node.index()] as usize,
                scratch.adj_start[node.index() + 1] as usize,
            );
            for k in from..to {
                let target = scratch.adj_targets[k];
                scratch.in_degree[target.index()] -= 1;
                if scratch.in_degree[target.index()] == 0 {
                    scratch.queue.push(target);
                }
            }
        }
    }

    /// The topological sort as it was before `topological_sort` built an
    /// adjacency list: a scan of the whole edge list per popped node. Kept
    /// as the reference the fast sort is tested against.
    #[cfg(test)]
    fn topological_sort_reference(&self) -> Vec<NodeId> {
        let n = self.nodes.len();
        let mut in_degree: Vec<u32> = alloc::vec![0; n];
        let mut queue: Vec<NodeId> = Vec::new();

        for edge in &self.edges {
            if self.nodes[edge.to.index()].is_some() && self.nodes[edge.from.index()].is_some() {
                in_degree[edge.to.index()] += 1;
            }
        }

        for (i, slot) in self.nodes.iter().enumerate() {
            if slot.is_some() && in_degree[i] == 0 {
                queue.push(NodeId(i as u32));
            }
        }

        let mut order = Vec::new();
        let mut head = 0;

        while head < queue.len() {
            let node = queue[head];
            head += 1;
            order.push(node);

            for edge in &self.edges {
                if edge.from == node && self.nodes[edge.to.index()].is_some() {
                    in_degree[edge.to.index()] -= 1;
                    if in_degree[edge.to.index()] == 0 {
                        queue.push(edge.to);
                    }
                }
            }
        }
        order
    }

    /// Resolve multichannel expansion: propagate channel counts through the
    /// graph in topological order and resize output buffers.
    ///
    /// An input port's channel count is its source's (`input_sources`, which
    /// `resolve_inputs` has just built: the first edge listed for the port
    /// decides), or 1 when the port is unconnected or that first edge's
    /// source has been removed.
    fn resolve_channels(&mut self, context: &ProcessContext) {
        let mut input_channels = core::mem::take(&mut self.topo_scratch.input_channels);
        for order_idx in 0..self.topo_order.len() {
            let node_id = self.topo_order[order_idx];
            let idx = node_id.index();

            if self.nodes[idx].is_none() {
                continue;
            }

            // Gather input channel counts. `input_sources[idx]` has one entry
            // per port of the node's spec, and names only live sources.
            input_channels.clear();
            for src in &self.input_sources[idx] {
                let src_ch = match src {
                    Some(src_id) => self.nodes[src_id.index()]
                        .as_ref()
                        .map_or(1, |s| s.output_channels),
                    None => 1,
                };
                input_channels.push(src_ch);
            }

            let slot = self.nodes[idx].as_mut().unwrap();
            let out_ch = slot.ugen.output_channels(&input_channels);
            slot.output_channels = out_ch;

            // Determine the rate of the first output to set block size
            let rate = slot
                .ugen
                .spec()
                .outputs
                .first()
                .map_or(crate::context::Rate::Audio, |o| o.rate);
            let block_size = context.block_size_for_rate(rate);

            slot.output.set_num_channels(out_ch, block_size);
            slot.output.set_block_size(block_size);
            // The whole-block buffer is sized here too, so a block rendered
            // in pieces never allocates (see `collect_piece`).
            slot.assembled.set_num_channels(out_ch, block_size);
            slot.assembled.set_block_size(block_size);
        }
        self.topo_scratch.input_channels = input_channels;
    }

    /// Call `init` on every node that has not been initialized yet.
    ///
    /// `prepare()` runs on every structural change (a node or edge added or
    /// removed), not just once at startup. Calling `init` unconditionally
    /// here would re-run it on nodes that are already live and rendering --
    /// and for a lot of `UGen` impls, `init` is a full state reset (envelope
    /// stage, oscillator/ramp phase, delay-line write cursor, playback
    /// position, ...), so an unrelated voice spawning or freeing elsewhere in
    /// the graph would restart or glitch every other already-running voice.
    /// Gating on `NodeSlot::initialized` makes `init` run exactly once per
    /// node, at the first `prepare()` after it was added, while still
    /// covering nodes added since the last `prepare()`.
    fn init_nodes(&mut self, context: &ProcessContext) {
        for slot in self.nodes.iter_mut().flatten() {
            if !slot.initialized {
                slot.ugen.init(context);
                slot.initialized = true;
            }
        }
    }

    /// Get a reference to a node's output buffer (e.g. for reading results).
    pub fn node_output(&self, id: NodeId) -> Option<&AudioBuffer> {
        self.nodes[id.index()].as_ref().map(|s| &s.output)
    }

    /// Whether the graph needs `prepare()` called before rendering.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Switch a node on or off. An inactive node is not processed and its
    /// output is silence of the block's length, so everything it feeds sees
    /// zeros. Its state stays as `prepare()` left it: the first block it
    /// processes after activation is its first. Needs no `prepare()`.
    pub fn set_node_active(&mut self, id: NodeId, active: bool) {
        if let Some(Some(slot)) = self.nodes.get_mut(id.index()) {
            slot.active = active;
        }
    }

    /// Set an internal scalar value on a node (e.g. change a Const node's value).
    /// Returns true if the node accepted the value.
    /// This does NOT require `prepare()` — it's safe to call between renders.
    pub fn set_node_value(&mut self, id: NodeId, value: f32) -> bool {
        match self.nodes.get_mut(id.index()) {
            Some(Some(slot)) => slot.ugen.set_value(value),
            _ => false,
        }
    }

    /// Force a node's noise generator(s) to a specific seed, overriding the
    /// per-spawn seed `Engine::instantiate_synthdef` already assigned it (a
    /// no-op for a node with no noise generator -- same pattern as
    /// `set_node_value` on a node with no scalar to set). Reseeding is a
    /// hard reset of the generator's internal LCG state, not a target to
    /// glide toward.
    ///
    /// Exists so a caller that spawns several noise-bearing voices meant to
    /// be perceived as ONE sound source (e.g. several `partialsNoise`
    /// instances crossfaded together) can give them all the SAME seed --
    /// their noise streams become sample-for-sample identical rather than
    /// independent, so a crossfaded/mixed sum of them is coherent (adds
    /// linearly) instead of incoherent (adds as the RMS sum of uncorrelated
    /// sources, `sqrt(sum a_k^2)` rather than `sum a_k`). Without this, two
    /// noise-bearing voices instantiated in the same synthdef always get
    /// distinct seeds (`derive_noise_seed`'s `node_idx` term) -- correct and
    /// desired when they are meant to sound like independent sources (e.g.
    /// two different drum voices), wrong when they are meant to sound like
    /// one.
    pub fn reseed_node_noise(&mut self, id: NodeId, seed: u32) -> bool {
        match self.nodes.get_mut(id.index()) {
            Some(Some(slot)) => {
                slot.ugen.reseed_noise(seed);
                true
            }
            _ => false,
        }
    }

    /// Set a target value with glide on a node (e.g. smooth parameter transition).
    /// Returns true if the node accepted the target.
    /// This does NOT require `prepare()` — it's safe to call between renders.
    pub fn set_node_target(
        &mut self,
        id: NodeId,
        target: f32,
        glide_secs: f32,
        shape: GlideShape,
        space: GlideSpace,
    ) -> bool {
        match self.nodes.get_mut(id.index()) {
            Some(Some(slot)) => slot.ugen.set_target(target, glide_secs, shape, space),
            _ => false,
        }
    }

    /// Check if a node reports that it is done (e.g. envelope finished).
    pub fn node_is_done(&self, id: NodeId) -> bool {
        match self.nodes.get(id.index()) {
            Some(Some(slot)) => slot.ugen.is_done(),
            _ => false,
        }
    }

    /// Get the UGenSpec for a node.
    pub fn node_spec(&self, id: NodeId) -> Option<crate::node::UGenSpec> {
        self.nodes.get(id.index())?.as_ref().map(|s| s.ugen.spec())
    }

    /// Count the number of edges connected to a node's inputs.
    pub fn edges_to(&self, node_id: NodeId) -> usize {
        self.edges.iter().filter(|e| e.to == node_id).count()
    }

    /// Which input slots on `node_id` currently have a connected edge, from
    /// any source. Unlike `edges_to`'s bare count, this names the ACTUAL
    /// occupied indices -- the one place that can answer "is slot N free on
    /// this node" by reading the graph's own edges directly, rather than a
    /// caller keeping a shadow count that can drift from what `connect` (an
    /// (to, to_input) pair overwrite, see its own doc) actually did.
    pub fn used_input_slots(&self, node_id: NodeId) -> impl Iterator<Item = usize> + '_ {
        self.edges
            .iter()
            .filter(move |e| e.to == node_id)
            .map(|e| e.to_input)
    }
}

#[cfg(test)]
mod tests {
    //! The fast `prepare` path against the edge-scanning code it replaced:
    //! the same node order, the same channel counts and the same edge list
    //! after removals, on randomised graphs (cycles, self-loops, parallel
    //! edges, removed nodes, duplicate ports and ports past a node's spec).

    use super::*;
    use crate::context::Rate;
    use crate::node::{InputSpec, OutputSpec, UGenCategory, UGenSpec};

    const fn port(name: &'static str) -> InputSpec {
        InputSpec {
            name,
            rate: Rate::Audio,
            required: false,
        }
    }

    static PORTS: [&[InputSpec]; 5] = [
        &[],
        &[port("a")],
        &[port("a"), port("b")],
        &[port("a"), port("b"), port("c")],
        &[port("a"), port("b"), port("c"), port("d")],
    ];
    static OUTPUTS: &[OutputSpec] = &[OutputSpec {
        name: "out",
        rate: Rate::Audio,
    }];

    /// A node whose output channel count depends on each port's position
    /// and channel count, so a wrong source or order shows up in the counts.
    struct Probe {
        ports: usize,
        salt: usize,
    }

    impl UGen for Probe {
        fn spec(&self) -> UGenSpec {
            UGenSpec {
                name: "Probe",
                category: UGenCategory::Utility,
                inputs: PORTS[self.ports],
                outputs: OUTPUTS,
            }
        }
        fn init(&mut self, _context: &ProcessContext) {}
        fn reset(&mut self) {}
        fn output_channels(&self, input_channels: &[usize]) -> usize {
            assert_eq!(input_channels.len(), self.ports);
            let mix: usize = input_channels
                .iter()
                .enumerate()
                .map(|(i, c)| (i + 1) * c)
                .sum();
            1 + (self.salt + mix) % 4
        }
        fn process(
            &mut self,
            _context: &ProcessContext,
            _inputs: &[Option<&AudioBuffer>],
            output: &mut AudioBuffer,
        ) {
            output.clear();
        }
    }

    /// Small deterministic generator (no dev-dependency needed).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }
        fn chance(&mut self, percent: u64) -> bool {
            self.next() % 100 < percent
        }
    }

    fn add_probe(g: &mut AudioGraph, rng: &mut Lcg) -> NodeId {
        let ports = rng.below(5);
        let salt = rng.below(7);
        g.add_node(Box::new(Probe { ports, salt }))
    }

    /// Add `count` random edges among `0..n`: mostly forward (a DAG), with
    /// back edges and self-loops (cycles), parallel edges, edges pushed
    /// directly so a (node, port) is listed twice, and ports past the spec.
    fn add_edges(g: &mut AudioGraph, rng: &mut Lcg, n: usize, count: usize) {
        for _ in 0..count {
            let a = rng.below(n);
            let b = rng.below(n);
            let (from, to) = if rng.chance(85) {
                (a.min(b), a.max(b).max(a.min(b) + 1).min(n - 1))
            } else {
                (a, b)
            };
            let (from, to) = (NodeId(from as u32), NodeId(to as u32));
            let to_input = rng.below(6);
            if rng.chance(25) {
                g.edges.push(Edge { from, to, to_input });
                g.dirty = true;
            } else {
                g.connect(from, to, to_input);
            }
        }
    }

    /// The channel counts the edge-searching `resolve_channels` produced,
    /// starting from the counts the nodes hold now, in `order`.
    fn channels_reference(g: &AudioGraph, order: &[NodeId]) -> Vec<usize> {
        let mut chans: Vec<usize> = g
            .nodes
            .iter()
            .map(|s| s.as_ref().map_or(0, |s| s.output_channels))
            .collect();
        for &node_id in order {
            let slot = match &g.nodes[node_id.index()] {
                Some(slot) => slot,
                None => continue,
            };
            let num_inputs = slot.ugen.spec().inputs.len();
            let mut input_channels = Vec::new();
            for input_idx in 0..num_inputs {
                let source = g
                    .edges
                    .iter()
                    .find(|e| e.to == node_id && e.to_input == input_idx);
                input_channels.push(match source {
                    Some(edge) => g.nodes[edge.from.index()]
                        .as_ref()
                        .map_or(1, |_| chans[edge.from.index()]),
                    None => 1,
                });
            }
            chans[node_id.index()] = slot.ugen.output_channels(&input_channels);
        }
        chans
    }

    fn channels_now(g: &AudioGraph) -> Vec<usize> {
        g.nodes
            .iter()
            .map(|s| s.as_ref().map_or(0, |s| s.output_channels))
            .collect()
    }

    fn edge_list(g: &AudioGraph) -> Vec<(NodeId, NodeId, usize)> {
        g.edges.iter().map(|e| (e.from, e.to, e.to_input)).collect()
    }

    /// Prepare `g` and check order and channels against the references.
    fn prepare_and_check(g: &mut AudioGraph, context: &ProcessContext, case: u64) {
        let order = g.topological_sort_reference();
        let chans = channels_reference(g, &order);
        g.prepare(context);
        assert_eq!(
            g.topo_order, order,
            "topological order differs, case {case}"
        );
        assert_eq!(channels_now(g), chans, "channel counts differ, case {case}");
    }

    #[test]
    fn fast_prepare_matches_reference_on_random_graphs() {
        let context = ProcessContext::new(48000.0, 64);
        let mut cycles_seen = 0;
        for case in 0..1500u64 {
            let mut rng = Lcg(case.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5);
            let mut g = AudioGraph::new();
            let n = 1 + rng.below(48);
            for _ in 0..n {
                add_probe(&mut g, &mut rng);
            }
            let e = rng.below(3 * n + 1);
            add_edges(&mut g, &mut rng, n, e);
            g.set_sink(NodeId(rng.below(n) as u32));
            prepare_and_check(&mut g, &context, case);
            if g.topo_order.len() < n {
                cycles_seen += 1;
            }

            // A few rounds of structural change, as spawns and frees do.
            for _ in 0..3 {
                let n_now = g.nodes.len();
                let doomed: Vec<NodeId> = (0..n_now)
                    .filter(|_| rng.chance(15))
                    .map(|i| NodeId(i as u32))
                    .collect();
                g.remove_nodes(&doomed);
                let added = rng.below(12);
                for _ in 0..added {
                    add_probe(&mut g, &mut rng);
                }
                let n_now = g.nodes.len();
                let e = rng.below(2 * added + 4);
                add_edges(&mut g, &mut rng, n_now, e);
                // Edges whose ends were removed earlier, pushed directly
                // (the engine never makes one, but the sort must not care).
                if rng.chance(20) && !doomed.is_empty() {
                    let dead = doomed[rng.below(doomed.len())];
                    let live = NodeId(rng.below(n_now) as u32);
                    g.edges.push(Edge {
                        from: dead,
                        to: live,
                        to_input: rng.below(4),
                    });
                    g.edges.push(Edge {
                        from: live,
                        to: dead,
                        to_input: rng.below(4),
                    });
                }
                prepare_and_check(&mut g, &context, case);
            }
        }
        assert!(cycles_seen > 50, "the cases should include cycles");
    }

    #[test]
    fn remove_nodes_matches_remove_node_in_turn() {
        let context = ProcessContext::new(48000.0, 64);
        let mut empty_cases = 0;
        for case in 0..1000u64 {
            let build = || {
                let mut rng = Lcg(case ^ 0x5DEE_CE66);
                let mut g = AudioGraph::new();
                let n = 1 + rng.below(40);
                for _ in 0..n {
                    add_probe(&mut g, &mut rng);
                }
                let e = rng.below(4 * n + 1);
                add_edges(&mut g, &mut rng, n, e);
                g.set_sink(NodeId(rng.below(n) as u32));
                (g, rng, n)
            };
            let (mut batch, mut rng, n) = build();
            let (mut one_by_one, _, _) = build();
            assert_eq!(edge_list(&batch), edge_list(&one_by_one));

            // Ids in any order, repeated, already removed, or past the end
            // (just past, or as far past as an id can be).
            let mut ids: Vec<NodeId> = (0..rng.below(n + 1))
                .map(|_| NodeId(rng.below(n + 2) as u32))
                .collect();
            if !ids.is_empty() && rng.chance(20) {
                ids.push(NodeId(u32::MAX));
            }
            if rng.chance(30) {
                ids.push(NodeId(rng.below(n) as u32));
                one_by_one.remove_node(ids[ids.len() - 1]);
                batch.remove_node(ids[ids.len() - 1]);
            }
            // Prepared, so the dirty flag below says what the removal did
            // (the ghost edges below go in after, as `prepare` would not
            // accept them; pushing them leaves the flag alone).
            batch.prepare(&context);
            one_by_one.prepare(&context);
            assert!(!batch.dirty && !one_by_one.dirty);

            if rng.chance(10) {
                // Edges naming ids past the node list.
                for ghost in [NodeId((n + 1) as u32), NodeId(u32::MAX)] {
                    for g in [&mut batch, &mut one_by_one] {
                        g.edges.push(Edge {
                            from: NodeId(0),
                            to: ghost,
                            to_input: 0,
                        });
                    }
                }
            }

            batch.remove_nodes(&ids);
            for &id in &ids {
                one_by_one.remove_node(id);
            }
            assert_eq!(edge_list(&batch), edge_list(&one_by_one), "case {case}");
            assert_eq!(batch.sink, one_by_one.sink, "case {case}");
            let live = |g: &AudioGraph| g.nodes.iter().map(Option::is_some).collect::<Vec<_>>();
            assert_eq!(live(&batch), live(&one_by_one), "case {case}");
            assert_eq!(batch.dirty, one_by_one.dirty, "case {case}");
            assert_eq!(batch.dirty, !ids.is_empty(), "case {case}");
            if ids.is_empty() {
                empty_cases += 1;
            }
        }
        assert!(empty_cases > 10, "the cases should include empty id lists");
    }
}
