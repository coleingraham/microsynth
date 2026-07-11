use crate::buffer::AudioBuffer;
use crate::context::ProcessContext;
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
    /// Resolved output channel count after multichannel expansion.
    output_channels: usize,
    /// Whether this node has been processed in the current block.
    processed: bool,
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
}

impl TopoScratch {
    fn new() -> Self {
        TopoScratch {
            in_degree: Vec::new(),
            queue: Vec::new(),
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
            output_channels: 1,
            processed: false,
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
    fn resolve_inputs(&mut self) {
        let n = self.nodes.len();
        self.input_sources.clear();
        self.input_sources.resize(n, Vec::new());

        for idx in 0..n {
            let num_inputs = match &self.nodes[idx] {
                Some(slot) => slot.ugen.spec().inputs.len(),
                None => {
                    self.input_sources[idx].clear();
                    continue;
                }
            };

            let node_id = NodeId(idx as u32);
            let sources = &mut self.input_sources[idx];
            sources.clear();
            for input_idx in 0..num_inputs {
                let src = self
                    .edges
                    .iter()
                    .find(|e| e.to == node_id && e.to_input == input_idx)
                    .map(|e| e.from)
                    // Only keep the source if the node still exists.
                    .filter(|from| self.nodes[from.index()].is_some());
                sources.push(src);
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

            // Gather source output buffer pointers into reusable scratch.
            // SAFETY: topological order guarantees every source node has
            // already been processed this block and will not be mutated again,
            // so these pointers stay valid for this node's `process` call.
            self.input_ptr_scratch.clear();
            for src in &self.input_sources[idx] {
                if let Some(src_id) = src {
                    let src_slot = self.nodes[src_id.index()].as_ref().unwrap();
                    self.input_ptr_scratch
                        .push(&src_slot.output as *const AudioBuffer);
                }
            }

            // View the gathered `*const AudioBuffer` scratch as `&[&AudioBuffer]`
            // (identical layout). This borrows nothing tracked, so the mutable
            // borrow of the destination node below is permitted.
            let input_refs: &[&AudioBuffer] = unsafe {
                core::slice::from_raw_parts(
                    self.input_ptr_scratch.as_ptr() as *const &AudioBuffer,
                    self.input_ptr_scratch.len(),
                )
            };

            let slot = self.nodes[idx].as_mut().unwrap();
            slot.ugen.process(context, input_refs, &mut slot.output);
            slot.processed = true;
        }

        // Return sink output
        let sink_id = self.sink.unwrap();
        self.nodes[sink_id.index()].as_ref().map(|s| &s.output)
    }

    /// Compute topological ordering using Kahn's algorithm.
    fn topological_sort(&mut self) {
        let n = self.nodes.len();
        let scratch = &mut self.topo_scratch;

        scratch.in_degree.clear();
        scratch.in_degree.resize(n, 0);
        scratch.queue.clear();

        // Compute in-degrees
        for edge in &self.edges {
            if self.nodes[edge.to.index()].is_some() && self.nodes[edge.from.index()].is_some() {
                scratch.in_degree[edge.to.index()] += 1;
            }
        }

        // Seed with nodes that have zero in-degree
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

            for edge in &self.edges {
                if edge.from == node && self.nodes[edge.to.index()].is_some() {
                    scratch.in_degree[edge.to.index()] -= 1;
                    if scratch.in_degree[edge.to.index()] == 0 {
                        scratch.queue.push(edge.to);
                    }
                }
            }
        }
    }

    /// Resolve multichannel expansion: propagate channel counts through the
    /// graph in topological order and resize output buffers.
    fn resolve_channels(&mut self, context: &ProcessContext) {
        for order_idx in 0..self.topo_order.len() {
            let node_id = self.topo_order[order_idx];
            let idx = node_id.index();

            let num_inputs = match &self.nodes[idx] {
                Some(slot) => slot.ugen.spec().inputs.len(),
                None => continue,
            };

            // Gather input channel counts
            let mut input_channels = Vec::new();
            for input_idx in 0..num_inputs {
                let source = self
                    .edges
                    .iter()
                    .find(|e| e.to == node_id && e.to_input == input_idx);

                match source {
                    Some(edge) => {
                        let src_ch = self.nodes[edge.from.index()]
                            .as_ref()
                            .map_or(1, |s| s.output_channels);
                        input_channels.push(src_ch);
                    }
                    None => {
                        input_channels.push(1);
                    }
                }
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
        }
    }

    /// Call init on all nodes.
    fn init_nodes(&mut self, context: &ProcessContext) {
        for slot in self.nodes.iter_mut().flatten() {
            slot.ugen.init(context);
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

    /// Set an internal scalar value on a node (e.g. change a Const node's value).
    /// Returns true if the node accepted the value.
    /// This does NOT require `prepare()` — it's safe to call between renders.
    pub fn set_node_value(&mut self, id: NodeId, value: f32) -> bool {
        match self.nodes.get_mut(id.index()) {
            Some(Some(slot)) => slot.ugen.set_value(value),
            _ => false,
        }
    }

    /// Set a target value with glide on a node (e.g. smooth parameter transition).
    /// Returns true if the node accepted the target.
    /// This does NOT require `prepare()` — it's safe to call between renders.
    pub fn set_node_target(&mut self, id: NodeId, target: f32, glide_secs: f32) -> bool {
        match self.nodes.get_mut(id.index()) {
            Some(Some(slot)) => slot.ugen.set_target(target, glide_secs),
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
}
