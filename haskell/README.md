# microsynth — Haskell version (scaffold)

An idiomatic-Haskell proof-of-concept for the Rust `microsynth` engine. It is
**not** a full port — it implements the core architecture end-to-end (build a
synth graph → compile → render offline → write a WAV) with a small but
representative set of UGens, so the design can be evaluated and benchmarked
against the Rust original. See [`DESIGN.md`](DESIGN.md) for the full
Rust → Haskell mapping and rationale, and [`COEXISTENCE.md`](COEXISTENCE.md) for
a proposed architecture where Haskell authors instruments and Rust deploys them
via a shared SynthDef IR — aimed at human and AI authoring workflows. (The IR
itself is now half-built on both sides but **not yet shared**; COEXISTENCE.md
tracks what is real versus intended.)

## The idea

In Rust, SynthDefs are written in a hand-parsed, Haskell-flavoured text DSL
(`src/dsl/{lexer,parser,compiler}.rs`). In Haskell that DSL needs **no parser**:
a `Signal` is a pure description of a sub-graph, and `Num`/`Fractional`
instances make `osc * env * amp` and `freq * 6` real, type-checked Haskell.

```haskell
demo :: SynthDef
demo = synthdef "demo" $ do
  freq <- param "freq" 220
  amp  <- param "amp"  0.4
  let osc = saw freq
      env = perc 0.01 0.6
  out (lpf osc (freq * 6) 1.5 * env * amp)
```

Two layers, mirroring Rust's "immutable SynthDef vs. mutable graph" split:

- **Build layer** — pure, lazy: the `Signal` EDSL + a small builder monad
  (`Microsynth.Signal`, `Microsynth.SynthDef`).
- **Render layer** — strict, mutable: everything runs in one `ST` region over
  pre-allocated unboxed `MVector` blocks — the direct analogue of Rust's
  no-allocation render path (`Microsynth.Engine`, `Microsynth.Node`).

## What's implemented

| Area | Modules | Ported from |
|---|---|---|
| Domain newtypes (`Sample`, `SampleRate`, `NodeId`, …) | `Types` | — (Haskell-side type safety) |
| Buffers (unboxed mutable blocks) | `Buffer` | `buffer.rs` |
| UGen abstraction (state-in-closure) | `Node` | `node.rs` (`trait UGen`) |
| Signal EDSL (`Num`/`Fractional`) | `Signal` | replaces `dsl/*` |
| SynthDef + builder + compile | `SynthDef` | `synthdef.rs`, `dsl/compiler.rs` |
| Kahn topological sort | `Graph` | `graph.rs` |
| Offline block render (in `ST`) | `Engine` | `engine.rs` |
| 16-bit PCM WAV writer | `Wav` | `bin/microsynth-cli.rs` |
| UGen descriptor registry (kind tags, port names + defaults) | `UGen.Spec` | `node.rs` (`spec()`), `ugens/mod.rs` (`register_spec`) |
| Shared DSP constants + block combinators | `Numerics`, `UGen.Common` | `ugens/macros.rs`, `buffer.rs` (`read_input`) |
| UGens: const/binop/neg, sinOsc, saw, lpf (RBJ biquad), perc | `UGen.*` | `ugens/*` |
| Graph introspection (named ports, kind tags, arity) | `SynthDef.Introspect` | — (no Rust equivalent) |
| Versioned JSON SynthDef IR | `SynthDef.IR` | `ir/` — **but not the same format**, see `COEXISTENCE.md` |
| CLI (`optparse-applicative`) | `app/Main.hs` | `bin/microsynth-cli.rs` (`clap`) |

Deferred (design-only): the remaining ~50 UGens, FFT/spectral, scheduler,
routing, musical-time/tuning, the optional text-DSL front end (megaparsec), and
the WASM/WebAudio backend. See `DESIGN.md`.

> **Caveat on the IR.** Both engines ship a "version 1" SynthDef IR and the two
> are mutually unparseable — different node model, different field names,
> different version policy. Unifying them is the open blocker on the coexistence
> plan; see [`COEXISTENCE.md`](COEXISTENCE.md#current-state--two-irs-one-version-number).

## Build & run

```sh
cd haskell
cabal build           # library + microsynth-cli
cabal test            # hspec sanity checks
cabal run microsynth-cli -- --duration 2 --sample-rate 44100 -o out.wav
cabal run microsynth-cli -- --synthdef pad --param freq=330 -o pad.wav
```

## Performance vs. Rust

Same patch (filtered percussive saw), same sample rate, both rendering offline
to a mono 16-bit WAV. Pure DSP throughput is isolated from fixed startup +
WAV-write overhead by measuring at two durations and taking the slope;
"× realtime" is seconds of audio rendered per wall-clock second of DSP.

> **Measurement caveat.** This runs on a shared cloud VM. Absolute throughput
> drifts between sessions — the memory-touching Haskell more than the
> register-tight Rust — so **only compare numbers captured back-to-back within
> one run**. Every table here was measured in a single session; the *ratios*
> are stable, the absolutes are not.

**Single voice** (the least representative case — see the polyphony sweep below):

| Build | DSP / audio-second | ≈ realtime | vs. Haskell |
|---|---|---|---|
| Rust, `opt-level=3` + LTO | 0.00158 s | ~633× | **2.4× faster** |
| Rust, `opt-level="s"` (project default) | 0.00201 s | ~497× | **1.9× faster** |
| Haskell, `-O2` | 0.00373 s | ~268× | 1.0× (baseline) |

The Haskell render path was optimized in two rounds:

- **Round 1** (~1.7× over the first, naïve cut): bind each node's inputs/output
  once at instantiation so the block loop is a bare `ST s ()` per node (no
  per-block input list, no `drop` per sample); `unsafeRead`/`unsafeWrite` inner
  loops; thread state through the loop; constants fill their block once; render
  into one preallocated buffer; compare-and-subtract phase wrap instead of
  `floor`.
- **Round 2** — the two "remaining levers": return the biquad coefficients in an
  **unboxed tuple** and keep per-UGen state in **unboxed cells**. Unboxing the
  state is what matters most: a boxed `STRef Float` write demands a boxed
  `Float`, which forces GHC to box the loop's threaded accumulators *every
  sample*; unboxed cells let them stay `Float#`. Net: **4.5× less allocation**
  (1.8 GB → 401 MB at 64 voices / 30 s). This is a deliberate **trade** — ~5%
  faster at high polyphony with far lower GC pressure, but ~25% *slower* at a
  single voice, where the working set is cache-hot and bump-allocation is nearly
  free. We keep it because real synthesis is polyphonic and a real-time path
  wants predictable allocation — and single voice is exactly the benchmark that
  misleads.

`-fllvm` gave no measurable gain (GHC 9.4 predates the box's LLVM 18). The two
Rust rows show the project's size-optimized default profile costs ~20%
throughput vs. a speed-optimized build.

Reproduce: `bash bench.sh 60 5` (end-to-end) or the two-duration slope method.

### Polyphony — the scaling test single-voice benchmarks hide

A single-voice benchmark is easy to over-trust, so here is the same patch scaled
to *N* independent voices (each its own filtered-saw + envelope at its own
frequency, all summed). `s/as` is DSP seconds per audio second; `per-voice` is
that divided by *N*.

| voices | rust `opt=s` (per-voice) | rust `opt=3` (per-voice) | haskell (per-voice) |
|---:|---|---|---|
| 1  | 0.00193 (0.001932) | 0.00155 (0.001549) | 0.00339 (0.003389) |
| 8  | 0.01179 (0.001473) | 0.01003 (0.001254) | 0.01342 (0.001677) |
| 16 | 0.02310 (0.001443) | 0.01998 (0.001248) | 0.02565 (0.001603) |
| 32 | 0.04582 (0.001432) | 0.04000 (0.001249) | 0.04971 (0.001553) |
| 64 | 0.09723 (0.001519) | 0.08320 (0.001299) | 0.10603 (0.001656) |

All three engines now scale **linearly** — per-voice cost is flat across the
sweep. Rust leads at every voice count: at 64 voices Haskell is ~1.27× behind
speed-optimized Rust and ~1.09× behind the project-default build (Haskell still
renders ~9× faster than real time at 64 voices). Haskell's higher *1-voice*
figure is fixed per-render cost (topo sort, instantiation, the output-buffer
freeze) amortising out by ~8 voices.

**This table is the fixed version of an earlier, more dramatic one.** The first
measurement showed Rust's per-voice cost *rising* with polyphony (+23% from 1 to
64 voices) while Haskell's stayed flat, so Haskell *overtook* Rust past ~24
voices. That was real, and it was verifiable in the source: the Rust render loop
re-resolved wiring **every block** — for every node, every input port, a linear
scan of the whole edge list plus two `Vec` allocations per node — which is
~O(N²) in voice count. It was also a *fixable* engine bug, not a language limit.
The fix (`src/graph.rs`): resolve each node's input sources **once** in
`prepare()` into an `input_sources` cache and gather them through a reusable
scratch buffer, so `render()` is O(nodes) with no per-block allocation. That
restored Rust's expected constant-factor lead — the honest conclusion being that
**equally optimized, Rust holds a ~1.1–1.3× edge at high polyphony** (more at one
voice, where Haskell's fixed setup dominates). The episode is the whole point,
though: the single-voice number predicted *neither* the buggy nor the fixed
polyphonic curve.

**Does Haskell blow up on GC at scale?** No. At 64 voices (`+RTS -s`), after the
unboxed-state optimization:

| render | allocated | productivity | max residency | total memory |
|---|---|---|---|---|
| 30 s | 401 MB | 99.8% | 5.7 MB | 19 MiB |

Allocation (down 4.5× from 1.8 GB once the loop accumulators were unboxed) is
short-lived nursery churn that dies immediately; GC stays under 0.5% of runtime
and never pauses meaningfully. Max residency just tracks the output buffer
(numSamples × 4 bytes), not the voice count — no leak, no superlinear growth.

Reproduce: `bash gen_rust_poly.sh N > poly_N.synth` for the Rust side and
`microsynth-cli --synthdef poly --voices N` for Haskell.
