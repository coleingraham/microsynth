# microsynth - Core Engine Architecture

## Overview

A pull-based, block-processing audio graph engine in Rust with SuperCollider-style
multichannel expansion. `no_std` compatible core. Targets WebAssembly (web),
native (desktop/mobile).

## Design Decisions

- **Immutable SynthDefs**: Like SuperCollider, SynthDefs are compiled templates
  that cannot be modified after creation. The live render graph (AudioGraph)
  supports runtime modification (add/remove nodes and edges).
- **Audio + Control rate**: Nodes declare their output rate. Audio-rate nodes
  produce `block_size` samples per block; control-rate nodes produce 1 sample.
- **`no_std` core**: The DSP core uses `#![no_std]` with `extern crate alloc`.
  `std`-dependent features (like offline rendering) are behind `feature = "std"`.
- **f32 samples**: Standard for real-time audio.
- **No allocation on render path**: All buffers pre-allocated during `prepare()`.
- **Pull-based via topological sort**: Nodes are evaluated in topological order
  (Kahn's algorithm), guaranteeing all inputs are ready before processing.
- **Multichannel expansion**: The expansion factor is `max(input_channels...)`.
  Inputs with fewer channels wrap via modulo (SuperCollider semantics).

## Module Structure

```
src/
├── lib.rs          # no_std, public re-exports
├── buffer.rs       # Block (stack-allocated [f32; 64]) and AudioBuffer (multi-channel)
├── context.rs      # ProcessContext (sample_rate, block_size, time) and Rate enum
├── node.rs         # UGen trait, NodeId, InputSpec, OutputSpec, UGenSpec
├── graph.rs        # AudioGraph: DAG, topo sort, pull render, runtime modification
├── synthdef.rs     # SynthDef (immutable template), SynthDefBuilder, Synth (live instance)
└── engine.rs       # Engine: owns graph + context, drives rendering
```

## Core Types

- `Block` — `[f32; MAX_BLOCK_SIZE]` on the stack, single channel
- `AudioBuffer` — `Vec<Block>`, multi-channel, pre-allocated
- `ProcessContext` — sample_rate, block_size, sample_offset
- `Rate` — `Audio` | `Control`
- `UGen` trait — `spec()`, `init()`, `reset()`, `process()`, `output_channels()`
- `AudioGraph` — nodes + edges, topo sort, pull-based `render()`
- `SynthDef` — immutable template with `UGenFactory` functions
- `SynthDefBuilder` — builds SynthDefs
- `Synth` — tracks live NodeIds for a SynthDef instance
- `Engine` — top-level API, owns graph + context
