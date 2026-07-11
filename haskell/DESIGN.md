# A Haskell version of microsynth — design

This document describes how the Rust `microsynth` engine maps onto idiomatic
Haskell. The `haskell/` directory implements the shaded ("scaffold") rows
end-to-end; the rest is design-only.

## Philosophy

Rust splits the world into **immutable `SynthDef` templates** and a **mutable
live graph**. Haskell keeps that split but sharpens each half:

1. **Build layer — pure, lazy, idiomatic.** A `Signal` is a pure value
   describing a sub-graph. Because the Rust DSL was *already* designed to look
   like Haskell, we make the DSL the host language: `Num`/`Fractional`
   instances on `Signal` mean `osc * env * amp` and `freq * 6` are ordinary,
   type-checked Haskell. This replaces the entire `src/dsl/` pipeline
   (lexer + parser + compiler) with a `Num` instance and a small builder monad.

2. **Render layer — strict, mutable, fast.** Rendering runs in a single `ST`
   region over pre-allocated unboxed `MVector` blocks. This is the direct
   analogue of Rust's "no allocation on the render path" rule; it is also where
   we pay off the performance debt that laziness would otherwise incur in DSP
   inner loops.

## Module mapping (Rust → Haskell)

| Rust | Haskell module | Status | Notes |
|---|---|---|---|
| `buffer.rs` (`Block`, `AudioBuffer`) | `Microsynth.Buffer` | ✅ scaffold | `MVector s Float` block, allocated once, reused |
| `context.rs` (`ProcessContext`, `Rate`) | `Microsynth.Context` | ✅ scaffold | plain strict record + `Rate` |
| `node.rs` (`trait UGen`) | `Microsynth.Node` | ✅ scaffold | `newtype Node s = Node (Context -> [MBlock s] -> MBlock s -> ST s ())`; state captured in the closure |
| `dsl/{lexer,parser,compiler}.rs` | `Microsynth.Signal` | ✅ scaffold | replaced by an EDSL: `Signal` + `Num`/`Fractional` |
| `synthdef.rs` (`SynthDef`, builder) | `Microsynth.SynthDef` | ✅ scaffold | pure `SynthDef` value; `synthdef`/`param`/`out` builder monad; AST→graph with leaf interning |
| `graph.rs` (Kahn topo sort, pull render) | `Microsynth.Graph` | ✅ scaffold | `topoSort`; render lives in `Engine` |
| `engine.rs` (`render`, `render_offline`) | `Microsynth.Engine` | ✅ scaffold | block loop in `runST`; per-node output vector read in topo order |
| `bin/microsynth-cli.rs` (`clap`, WAV) | `app/Main.hs` + `Microsynth.Wav` | ✅ scaffold | `optparse-applicative`; hand-written RIFF writer |
| `ugens/{math,oscillators,filters,envelopes}.rs` | `Microsynth.UGen.*` | ✅ subset | const/binop/neg, sinOsc, saw, RBJ lpf, perc |
| `ugens/*` (≈40 more) | `Microsynth.UGen.*` | ⬜ deferred | band-limited oscs, noise, ADSR/ASR, FM, physical models, distortion, modulation, reverb |
| `spectral/*` (FFT/STFT/Griffin-Lim) | `Microsynth.Spectral.*` | ⬜ deferred | recommend the `fft`/`vector-fft` package unless zero-dep is required |
| `scheduler.rs` | `Microsynth.Scheduler` | ⬜ deferred | time-ordered event queue (`Data.Heap`) |
| `routing.rs`, `musical_time.rs`, `tuning.rs`, `sample.rs` | matching modules | ⬜ deferred | pure data + maps |
| `web.rs` (WASM/WebAudio) | `Microsynth.Web` | ⬜ deferred | GHC WASM backend; the real-time AudioWorklet path is the genuinely hard part |

## Key type designs

### The UGen abstraction (`Node`)

Rust's `trait UGen` with `&mut self` + `process(ctx, inputs, out)` becomes a
function that closes over the node's own mutable state and writes its output in
place. The closure *is* the node — state is captured, not exposed — so no
existential type is needed:

```haskell
newtype Node s = Node
  { runNode :: Context -> [MBlock s] -> MBlock s -> ST s () }
```

A stateful UGen (oscillator phase, biquad `z1`/`z2`, envelope level/stage) holds
its state in `STRef`s created at instantiation, exactly mirroring the fields of
the Rust struct.

### The Signal EDSL (replaces `dsl/`)

```haskell
data Signal = Signal !UGenKind [Signal]     -- node kind + input signals

instance Num Signal where
  a + b       = Signal (KBinOp Add) [a, b]
  a * b       = Signal (KBinOp Mul) [a, b]
  fromInteger = constSig . fromInteger      -- literals → Const nodes
  ...
instance Fractional Signal where
  a / b        = Signal (KBinOp Div) [a, b]
  fromRational = constSig . fromRational
```

Compilation (`SynthDef.compile`) walks the `Signal` tree in post-order,
assigning node ids and **interning** parameters by name and constants by value,
so shared leaves collapse to a single node (a pragmatic answer to the
observable-sharing problem for the cases that actually matter).

### The render loop (`Engine`)

```haskell
renderOffline :: SynthDef -> Float -> Int -> Map String Float -> [VU.Vector Float]
```

Instantiate each node once, allocate one output block per node, then for each
block walk the topological order calling `runNode`. Rust's `unsafe`
input-pointer gathering (sound because of topo order) becomes *safe* indexing of
a boxed `Vector (MBlock s)` here — same invariant, no `unsafe`.

## Performance

See [`README.md`](README.md#performance-vs-rust). Summary: the unoptimized
scaffold is ~2.4–3.0× slower than Rust for identical DSP and still renders
~210× faster than real time. The gap is scaffold overhead (per-`STRef` state,
boxed output indirection), not a language ceiling.

## Why Haskell is a good fit here

- The "Haskell-inspired DSL" stops being a parser project and becomes a `Num`
  instance — hundreds of lines of lexer/parser/compiler deleted.
- SynthDefs are first-class values: compose, parameterize, and generate them
  with ordinary functions; type errors catch malformed graphs at compile time.
- `ST` + unboxed vectors give a no-alloc render path that reads almost exactly
  like the Rust one, so the performance-critical core stays honest.

## Where Haskell has to work harder

- **Laziness in DSP is a hazard**; the render path must be deliberately strict
  (`BangPatterns`, unboxed vectors, strict fields, `-O2`, and ideally `-fllvm`).
- **Real-time/WASM** is the weakest story: GHC's WASM backend exists, but a
  glitch-free AudioWorklet with bounded GC pauses is real research, unlike
  Rust's straightforward `wasm32` + `no_std` target.
