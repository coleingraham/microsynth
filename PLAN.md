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
- **Spectral UGens are self-contained**: Each spectral effect UGen internally
  manages its own STFT/ISTFT (input ring buffer, windowed FFT, overlap-add
  output). No "PV chain" buffer type or spectral rate — they appear as
  ordinary audio-rate-in/audio-rate-out nodes. FFT is pure Rust radix-2
  Cooley-Tukey (no external crate), `no_std` compatible.

## Module Structure

```
src/
├── lib.rs          # no_std, public re-exports
├── buffer.rs       # Block (stack-allocated [f32; MAX_BLOCK_SIZE]), AudioBuffer
│                   #   (multi-channel), read_input/channel_wrapped input helpers
├── context.rs      # ProcessContext (sample_rate, block_size, time) and Rate enum
├── node.rs         # UGen trait, NodeId, InputSpec, OutputSpec, UGenSpec, UGenCategory
├── graph.rs        # AudioGraph: DAG, topo sort, pull render, runtime modification
├── synthdef.rs     # SynthDef (immutable template), SynthDefBuilder, Synth (live instance)
├── engine.rs       # Engine: owns graph + context, drives rendering
├── scheduler.rs    # Scheduler: timed events, VoiceId, EventAction, voice lifecycle
├── routing.rs      # RoutingGraph: audio buses (BusId) and effect chains (EffectId)
├── musical_time.rs # MusicalPosition / TimeConfig: bars, steps, tick offsets, BPM
├── tuning.rs       # TuningTable, midi<->hz, cents — incl. non-12TET scales
├── sample.rs       # Sample, SampleBank, SampleId — loaded audio for playback UGens
├── web.rs          # #[cfg(wasm32)] WASM/AudioWorklet backend (feature = "web")
├── bin/
│   └── microsynth-cli.rs  # CLI: render a .synth file to WAV (feature = "std")
├── ir/             # Versioned, serializable SynthDef IR (feature = "ir")
│   ├── mod.rs      # IrSynthDef/IrNode/IrEdge/IrParam, from_decl, compile, validate
│   ├── serialize.rs# Binary codec (canonical) + JSON, content_hash
│   └── render.rs   # RenderSpec + render_ir offline conveniences (feature = "std")
├── ugens/          # Built-in UGens (one file per category)
│   ├── mod.rs      # Re-exports + register_builtins (DSL registration table)
│   ├── macros.rs   # ugen_spec! (and other shared authoring macros)
│   ├── delayline.rs# pub(crate) DelayLine — shared interpolating delay primitive
│   ├── math.rs     # Const, Param, BinOpUGen, NegUGen
│   ├── bus.rs      # Bus: dynamic-arity voice summing
│   ├── rng.rs      # pub(crate) Rng — deterministic PRNG for the noise UGens
│   ├── oscillators.rs, filters.rs, envelopes.rs, ...  # the DSP library
│   └── ...
├── spectral/       # Frequency-domain processing primitives
│   ├── mod.rs      # Module root
│   ├── complex.rs  # Complex number type for FFT operations
│   ├── fft.rs      # Radix-2 Cooley-Tukey FFT/IFFT (in-place, power-of-2)
│   ├── window.rs   # Window functions (Hann, Hamming, Blackman, BlackmanHarris)
│   ├── stft.rs     # StftProcessor: overlap-add STFT bridging block-size to FFT-size
│   └── griffin_lim.rs  # Offline phase reconstruction from magnitude spectrograms
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
- `UGenCategory` — the taxonomy tag on every `UGenSpec` (`Oscillator`, `Filter`,
  `Envelope`, `Effect`, `Math`, `Utility`, …); downstream tooling keys on it
- `UGen` trait — `spec()`, `init()`, `reset()`, `process()`, `output_channels()`,
  plus `set_value`/`set_target` (runtime params), `reseed_noise`, `is_done`
- `AudioGraph` — nodes + edges, topo sort, pull-based `render()`
- `SynthDef` — immutable template with `UGenFactory` closures
- `SynthDefBuilder` — builds SynthDefs
- `SynthParam` — a declared parameter: name, default, and the node/input it feeds
- `Synth` — tracks live NodeIds for a SynthDef instance
- `Engine` — top-level API, owns graph + context
- `Scheduler` / `VoiceId` — timed events and voice lifecycle
- `RoutingGraph` / `BusId` / `EffectId` — buses and effect chains
- `SampleBank`, `TuningTable`, `MusicalPosition` — sample, tuning, and time data
- `IrSynthDef` — the inspectable/serializable form of a compiled graph (`feature = "ir"`)

## DSL

Haskell-inspired text-based DSL for defining synthesis graphs. Compiles
to `SynthDef` templates via: tokenize → parse → compile.

### Syntax

```haskell
-- Parameters with defaults, function application by juxtaposition
synthdef pad freq=440.0 amp=0.5 =
  let osc = sinOsc freq 0.0
  let env = perc 0.01 1.0
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

Maps DSL identifiers to UGen factories. Users register their own UGens; the
compiler uses built-in `Const`, `BinOpUGen`, `NegUGen` for literals and
arithmetic.

Register with `register_spec(name, factory)`: it builds one probe instance and
reads the port names straight from the UGen's own `spec()`, so the ports have a
**single source of truth** (the UGen definition) and are never restated at the
registration site. The lower-level `register(name, factory, inputs, outputs)`
remains for callers that supply explicit specs (e.g. test-only UGens). Note the
DSL name is kept explicit because it differs from the UGen's internal
`spec().name` (camelCase `lpf` vs PascalCase `BiquadLPF`), and several DSL names
may map to one type (`sinTable`/`sawTable`/… → `WaveTable`).

### Design Decisions

- **No external parser dependencies** — hand-written lexer and recursive descent
  parser, keeping the zero-dependency policy.
- **`UGenFactory` is `Box<dyn Fn>` not `fn()`** — closures can capture parsed
  values (e.g. constant defaults).
- **Parameters become Param nodes** — each DSL parameter creates a `Param` UGen
  seeded with its declared default. Unlike a `Const`, a `Param` can be driven at
  runtime: `Engine::set_param` sets it instantly and `set_param_glide` ramps it
  (there are `set_voice_param`/`set_effect_param` variants for a single voice or
  an effect slot), backed by `UGen::set_value`/`set_target`.
- **Positional arguments** — `sinOsc freq 0.0` maps arguments to inputs in
  declaration order per the UGen's `InputSpec` list.

## Authoring a new UGen

The UGen layer was deduplicated (see the `claude/rust-refactor-review` history,
and a later sweep that extended the same idea): port specs, the per-sample input
read, the block-slice read, the registration table, the interpolating delay
line, and whole families of near-identical UGens each had one copy per UGen.
Shared abstractions now hold that logic in one place — **use them** when adding a
UGen so the duplication does not creep back:

1. **Define** the struct plus `new()`/`Default` in the right `src/ugens/*.rs`
   file (grouped by category).

2. **Generate `spec()` with the `ugen_spec!` macro** (in `ugens/macros.rs`),
   not hand-written `static INPUT/OUTPUT` arrays:

   ```rust
   impl UGen for MyOsc {
       ugen_spec!(
           "MyOsc",
           category = Oscillator,
           inputs = ["freq", "amp"],
           outputs = ["out"]
       );
       fn init(&mut self, ctx: &ProcessContext) { /* ... */ }
       fn reset(&mut self) { /* ... */ }
       fn process(&mut self, /* ... */) { /* ... */ }
   }
   ```

   **Always pass `category`.** It is the `UGenCategory` tag on the spec that
   downstream tooling keys on. The macro has a no-category arm, but it silently
   defaults to `UGenCategory::Utility` — so omitting it does not fail loudly, it
   just files your UGen in the wrong drawer. Pick the variant that matches what
   the UGen *does* (`Oscillator`, `Filter`, `Envelope`, `Effect`, `Math`,
   `Utility`, …), and add a case to `tests/categories.rs`.

   Only hand-write `spec()` if the UGen needs `Rate::Control` ports, or computes
   its name/ports at runtime (see `BinOpUGen`, which names itself from its
   `BinOpKind`, and `Bus`, whose port count is dynamic).

3. **Read modulatable inputs with `read_input`** (in `buffer.rs`), not the
   `buf.map(|b| b.channel(ch % b.num_channels()).samples()[i]).unwrap_or(d)`
   idiom. Trailing `.clamp()/.max()` stay at the call site:

   ```rust
   let freq = read_input(freq_buf, ch, i, 440.0).clamp(20.0, nyquist);
   ```

   When you need the *whole* channel slice for a block (not one sample), use its
   block-level counterpart `channel_wrapped(buf, ch)` (also in `buffer.rs`)
   rather than re-spelling `buf.channel(ch % buf.num_channels()).samples()`.

4. **Build any delay-based effect on `DelayLine`** (`ugens/delayline.rs`), never
   a hand-rolled `Vec<f32>` + write cursor. It owns the circular buffer, the
   wrap arithmetic, and fractional (linearly-interpolated) reads via
   `read_interp` — the piece that is easy to get subtly wrong (an off-by-one in
   the interpolation index silently quantizes the delay to whole samples). Comb
   and feedback topologies read-then-write with `write_and_advance`; plain taps
   write-then-read with `write` + `advance`. Every echo, chorus, flanger,
   ping-pong, reverb comb, and the Haas widener share this one type.

5. **Register for the DSL with `register_spec`** in `register_builtins`
   (`ugens/mod.rs`) — pass only the DSL name and a factory. Ports are derived
   from the UGen's `spec()`; never restate `InputSpec`/`OutputSpec` here:

   ```rust
   reg.register_spec("myOsc", || Box::new(MyOsc::new()));
   ```

6. **For a family of variants** that differ only in a small formula (like the
   biquad LPF/HPF/BPF/Notch/Allpass, which differ only in a coefficient function
   and a default), write a small local `macro_rules!` that stamps each concrete
   named type — a macro is preferred over a generic type so the registry and
   `pub use` re-exports keep referring to each variant by name. Existing
   templates to copy: `biquad_ugen!` (`filters.rs`), `phase_osc!`
   (`oscillators.rs`) and `bl_osc!` (`bl_oscillators.rs`) for phase-accumulator
   oscillators, and `ramp_ugen!` / `perc_ugen!` (`envelopes.rs`) for envelopes.
