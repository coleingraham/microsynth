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
├── engine.rs       # Engine: owns graph + context, drives rendering
├── ugens.rs        # Built-in UGens: Const, BinOpUGen, NegUGen
└── dsl/
    ├── mod.rs      # Public API, compile() entry point, DslError type
    ├── lexer.rs    # Tokenizer (keywords, numbers, operators, comments)
    ├── ast.rs      # AST: Program, SynthDefDecl, Expr, Binding, BinOp
    ├── parser.rs   # Recursive descent parser (Haskell-inspired syntax)
    └── compiler.rs # AST → SynthDef compilation, UGenRegistry
```

## Core Types

- `Block` — `[f32; MAX_BLOCK_SIZE]` on the stack, single channel
- `AudioBuffer` — `Vec<Block>`, multi-channel, pre-allocated
- `ProcessContext` — sample_rate, block_size, sample_offset
- `Rate` — `Audio` | `Control`
- `UGen` trait — `spec()`, `init()`, `reset()`, `process()`, `output_channels()`
- `AudioGraph` — nodes + edges, topo sort, pull-based `render()`
- `SynthDef` — immutable template with `UGenFactory` closures
- `SynthDefBuilder` — builds SynthDefs
- `Synth` — tracks live NodeIds for a SynthDef instance
- `Engine` — top-level API, owns graph + context

## DSL

Haskell-inspired text-based DSL for defining synthesis graphs. Compiles
to `SynthDef` templates via: tokenize → parse → compile.

### Syntax

```haskell
-- Parameters with defaults, function application by juxtaposition
synthdef pad freq=440.0 amp=0.5 =
  let osc = sinOsc freq 0.0
  let env = envGen 0.01 1.0
  osc * env * amp

-- Inline let...in variant
synthdef simple x=1.0 = let y = x * 2.0 in y + 1.0

-- Arithmetic operators: + - * / with standard precedence
-- Function application binds tighter than operators:
--   sinOsc freq * amp  →  (sinOsc freq) * amp
-- Comments: -- to end of line
```

### Compilation Pipeline

1. **Lexer** — source text → tokens (keywords, idents, numbers, operators)
2. **Parser** — tokens → AST (recursive descent, operator precedence)
3. **Compiler** — AST → SynthDef using a `UGenRegistry` that maps names to factories

### UGenRegistry

Maps DSL identifiers to UGen factories + input/output specs. Users register
their own UGens; the compiler uses built-in `Const`, `BinOpUGen`, `NegUGen`
for literals and arithmetic.

### Design Decisions

- **No external parser dependencies** — hand-written lexer and recursive descent
  parser, keeping the zero-dependency policy.
- **`UGenFactory` is `Box<dyn Fn>` not `fn()`** — closures can capture parsed
  values (e.g. constant defaults).
- **Parameters become Const nodes** — each DSL parameter creates a `Const` UGen
  outputting its default value. Runtime parameter modification is future work.
- **Positional arguments** — `sinOsc freq 0.0` maps arguments to inputs in
  declaration order per the UGen's `InputSpec` list.
