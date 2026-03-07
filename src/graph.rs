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

impl AudioGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        AudioGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            topo_order: Vec::new(),
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
        self.resolve_channels(context);
        self.init_nodes(context);
        self.dirty = false;
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

        // Process nodes in topological order.
        // We need to work around the borrow checker: gather input buffer
        // pointers before calling process on each node.
        for order_idx in 0..self.topo_order.len() {
            let node_id = self.topo_order[order_idx];
            let idx = node_id.index();

            // Gather input info: for each input port, which node feeds it?
            let num_inputs = match &self.nodes[idx] {
                Some(slot) => slot.ugen.spec().inputs.len(),
                None => continue,
            };

            // Build a list of source node indices for each input port.
            // We'll read their output buffers as raw pointers to work around
            // the borrow checker (safe because topo order guarantees sources
            // are already processed and won't be mutated again).
            let mut input_ptrs: Vec<*const AudioBuffer> = Vec::new();
            for input_idx in 0..num_inputs {
                let source = self
                    .edges
                    .iter()
                    .find(|e| e.to == node_id && e.to_input == input_idx)
                    .map(|e| e.from);

                match source {
                    Some(src_id) => {
                        let src_slot = self.nodes[src_id.index()].as_ref().unwrap();
                        input_ptrs.push(&src_slot.output as *const AudioBuffer);
                    }
                    None => {
                        // No connection to this input — will skip
                        input_ptrs.push(core::ptr::null());
                    }
                }
            }

            // Now process the node
            let slot = self.nodes[idx].as_mut().unwrap();

            // Build input references from pointers
            // SAFETY: topo order guarantees all source nodes are already processed.
            // Source output buffers won't be mutated again this block.
            let input_refs: Vec<&AudioBuffer> = input_ptrs
                .iter()
                .filter(|p| !p.is_null())
                .map(|p| unsafe { &**p })
                .collect();

            slot.ugen.process(context, &input_refs, &mut slot.output);
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
            if self.nodes[edge.to.index()].is_some()
                && self.nodes[edge.from.index()].is_some()
            {
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
            let rate = slot.ugen.spec().outputs.first().map_or(
                crate::context::Rate::Audio,
                |o| o.rate,
            );
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
}
