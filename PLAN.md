# microsynth - Core Engine Implementation Plan

## Architecture Overview

A pull-based, block-processing audio graph engine in Rust with SuperCollider-style
multichannel expansion. Compiles to WebAssembly for web, native for desktop/mobile.

## Module Structure

```
microsynth/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API, re-exports
│   ├── engine.rs           # Engine: owns graph, drives rendering
│   ├── graph.rs            # AudioGraph: compiled, topologically-sorted node graph
│   ├── node.rs             # Node trait + NodeId, Port types
│   ├── buffer.rs           # Buffer/BufferPool: pre-allocated block buffers
│   ├── context.rs          # ProcessContext: sample_rate, block_size, current_time
│   └── multichannel.rs     # Multichannel expansion logic
```

## Core Types

### `ProcessContext`
Global state passed to every node during processing:
- `sample_rate: f64`
- `block_size: usize`
- `sample_offset: u64` (monotonic sample counter)

### `Buffer`
A single channel of audio for one block. Fixed-size `Vec<f32>` (pre-allocated).

### `Signal`
Multichannel wrapper: `Vec<Buffer>` — 1 channel = mono, 2 = stereo, N = N-channel.
This is the unit of data flowing between nodes.

### `Node` trait
```rust
trait Node: Send {
    /// Number of input ports
    fn num_inputs(&self) -> usize;

    /// Number of output channels this node produces,
    /// given the channel counts of its inputs.
    /// This is where multichannel expansion is resolved.
    fn num_output_channels(&self, input_channels: &[usize]) -> usize;

    /// Process one block. Reads from inputs, writes to outputs.
    fn process(&mut self, context: &ProcessContext, inputs: &[&Signal], output: &mut Signal);

    /// Reset internal state (e.g. phase, filter memory)
    fn reset(&mut self);
}
```

### Multichannel Expansion
Like SuperCollider: if a node expects mono inputs but receives N channels on
any input, it expands — the node's processing runs N times (once per channel),
and produces N output channels. The expansion factor is `max(input_channels...)`.
Inputs with fewer channels wrap around (modulo).

This is handled at the graph level during `process`: for each node, we determine
the expansion factor from input channel counts, then either:
- Call `process` once if all inputs match or the node handles multi-channel natively
- Slice inputs channel-by-channel and call `process` N times (expansion)

### `AudioGraph`
- Stores nodes in topologically sorted order
- Owns all `Signal` buffers (pre-allocated in a pool)
- `render(&mut self, ctx: &ProcessContext) -> &Signal` pulls from the output,
  evaluating nodes in order

### `Engine`
- Owns an `AudioGraph` and `ProcessContext`
- `render_block(&mut self) -> &Signal` — advances time, calls graph render
- `render_to_buffer(&mut self, num_samples: usize) -> Vec<Vec<f32>>` — offline rendering

## Implementation Order

1. **Buffer & Signal types** — pre-allocated audio buffers
2. **ProcessContext** — global render state
3. **Node trait** — the unit generator interface
4. **AudioGraph** — node storage, connections, topological sort, pull-based render
5. **Multichannel expansion** — in the graph's render loop
6. **Engine** — top-level API wrapping graph + context
7. **Basic tests** — sine osc → verify output, multichannel expansion test

## Design Decisions

- **f32 samples**: Standard for real-time audio. f64 can be added later if needed.
- **No allocation in render path**: All buffers pre-allocated. BufferPool recycles.
- **`no_std` compatible core**: The DSP core should avoid std where possible
  for embedded/wasm targets. Engine layer can use std.
- **Send but not Sync on Node**: Nodes are processed single-threaded within a graph.
  `Send` allows moving between threads (e.g. from builder thread to audio thread).
