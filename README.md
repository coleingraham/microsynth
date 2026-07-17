# microsynth

A small real-time and non-real-time audio synthesis engine, written in Rust with
**zero runtime dependencies** and a `no_std`-compatible DSP core.

It is a pull-based, block-processing audio graph with SuperCollider-style
immutable SynthDefs, multichannel expansion, and a Haskell-inspired text DSL for
describing synthesis graphs. It runs natively, offline to a WAV file, and in the
browser via WebAssembly + AudioWorklet.

```haskell
synthdef tone freq=440.0 amp=0.3 gate=1.0 =
  let env = asr gate 0.01 0.3
  let sig = sinOsc freq 0.0
  sig * amp * env
```

## What's in it

- **~60 built-in UGens** — oscillators (naive and band-limited), filters
  (biquad family, one-pole, comb/allpass, reverb, compressor), envelopes,
  delays, modulation, distortion, physical models, noise, and stereo.
- **A text DSL** — hand-written lexer, recursive-descent parser, and compiler
  (`src/dsl/`). No parser dependencies.
- **Spectral processing** — a hand-rolled radix-2 Cooley-Tukey FFT, STFT with
  overlap-add, windows, and Griffin-Lim phase reconstruction (`src/spectral/`).
  Spectral UGens are self-contained: each manages its own STFT internally and
  appears as an ordinary audio-in/audio-out node.
- **A scheduler and routing graph** — timed events with voice lifecycle, plus
  audio buses and effect chains.
- **Musical primitives** — bar/step/tick musical time, and tuning tables
  including non-12TET scales.
- **A versioned SynthDef IR** — the inspectable, serializable form of a compiled
  graph, with a binary codec and JSON (`src/ir/`, `feature = "ir"`).
- **A WASM/AudioWorklet backend** and a browser editor (`web/`).

## Design in one breath

Immutable SynthDef templates compile to a mutable `AudioGraph`. Nodes are
evaluated in topological order (Kahn), so every input is ready before a node
processes. Audio-rate nodes emit `block_size` samples per block; control-rate
nodes emit one. **Nothing allocates on the render path** — all buffers are
pre-allocated in `prepare()`, and each node's input sources are resolved once
there rather than per block.

See [`PLAN.md`](PLAN.md) for the full architecture, the module map, and the
guide to authoring a new UGen.

## Build

```sh
cargo build --release
cargo test --all-features
```

### Features

| Feature | Default | What it enables |
|---|:--:|---|
| `std` | ✅ | Offline rendering and the CLI (pulls in `clap`) |
| `ir` | ✅ | The versioned, serializable SynthDef IR (`alloc` + `core` only) |
| `web` | — | The WASM/`wasm-bindgen` backend |

The DSP core is `#![no_std]` + `extern crate alloc`; everything requiring `std`
sits behind `feature = "std"`.

## CLI

Compiles a DSL patch on stdin and renders it offline:

```sh
cargo run --bin microsynth-cli -- render \
    --synthdef tone --duration 2 --sample-rate 44100 \
    --format wav --output tone.wav < patch.synth
```

`--param name=value` (repeatable) overrides a declared parameter.

## Web

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli
cd web && ./build.sh
python3 -m http.server 8080   # then open http://localhost:8080
```

This builds two WASM outputs — a raw module for the AudioWorklet and a
wasm-bindgen module for the main thread — and serves a browser editor with a
live synthdef editor, a step sequencer, and an oscilloscope.

## Haskell

[`haskell/`](haskell/) holds an idiomatic-Haskell implementation of the same
core architecture. It is not a full port: it implements the pipeline end-to-end
(build a graph → compile → render offline → write a WAV) with a representative
subset of UGens, as a design study and a performance comparison. In Haskell the
DSL needs no parser — a `Signal` is a pure description of a sub-graph, and
`Num`/`Fractional` instances make `osc * env * amp` real, type-checked Haskell.

- [`haskell/README.md`](haskell/README.md) — what's implemented, plus benchmarks
  against the Rust engine.
- [`haskell/DESIGN.md`](haskell/DESIGN.md) — the Rust → Haskell mapping.
- [`haskell/COEXISTENCE.md`](haskell/COEXISTENCE.md) — the proposed
  authoring-in-Haskell / deploying-in-Rust architecture joined by the SynthDef IR.

## License

MIT — see [`LICENSE`](LICENSE).
