# microsynth — Haskell version (scaffold)

An idiomatic-Haskell proof-of-concept for the Rust `microsynth` engine. It is
**not** a full port — it implements the core architecture end-to-end (build a
synth graph → compile → render offline → write a WAV) with a small but
representative set of UGens, so the design can be evaluated and benchmarked
against the Rust original. See [`DESIGN.md`](DESIGN.md) for the full
Rust → Haskell mapping and rationale.

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
| Buffers (unboxed mutable blocks) | `Buffer` | `buffer.rs` |
| UGen abstraction (state-in-closure) | `Node` | `node.rs` (`trait UGen`) |
| Signal EDSL (`Num`/`Fractional`) | `Signal` | replaces `dsl/*` |
| SynthDef + builder + compile | `SynthDef` | `synthdef.rs`, `dsl/compiler.rs` |
| Kahn topological sort | `Graph` | `graph.rs` |
| Offline block render (in `ST`) | `Engine` | `engine.rs` |
| 16-bit PCM WAV writer | `Wav` | `bin/microsynth-cli.rs` |
| UGens: const/binop/neg, sinOsc, saw, lpf (RBJ biquad), perc | `UGen.*` | `ugens/*` |
| CLI (`optparse-applicative`) | `app/Main.hs` | `bin/microsynth-cli.rs` (`clap`) |

Deferred (design-only): the remaining ~40 UGens, FFT/spectral, scheduler,
routing, musical-time/tuning, the optional text-DSL front end (megaparsec), and
the WASM/WebAudio backend. See `DESIGN.md`.

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
to a mono 16-bit WAV on the same machine. Pure DSP throughput is isolated from
fixed process-startup + WAV-write overhead by measuring at two durations (6 s
and 600 s) and taking the slope; "× realtime" is how many seconds of audio each
renders per wall-clock second of DSP.

| Build | DSP time / audio-second | ≈ realtime factor | vs. Haskell |
|---|---|---|---|
| Rust, `opt-level=3` + LTO | 0.00157 s | ~638× | **1.75× faster** |
| Rust, `opt-level="s"` (project default) | 0.00195 s | ~512× | **1.40× faster** |
| Haskell, `-O2` (optimized) | 0.00274 s | ~365× | 1.0× (baseline) |
| Haskell, `-O2` (first cut) | 0.00476 s | ~210× | 0.58× |

Takeaways:

- The optimized Haskell is **~1.4–1.75× slower** than Rust for identical DSP,
  and renders this patch **~365× faster than real time** — comfortably
  real-time-capable.
- Getting there was **~1.74× faster than the first cut** (0.00476 → 0.00274)
  and closed most of the original 2.4–3.0× gap. The optimizations, all in the
  render path:
  - **Bind each node's inputs/output once at instantiation** so the block loop
    is a bare `ST s ()` per node — no per-block input-list passing, no `drop`
    per sample.
  - **`unsafeRead`/`unsafeWrite`** in the inner loops (indices are in-bounds by
    construction).
  - **Thread filter/envelope state through the loop** — one `STRef` read/write
    per block instead of per sample. (Oscillator phase already did this.)
  - **Constants fill their block once** and then do zero work per block.
  - **Render into one preallocated buffer** (a per-block `copy`) instead of
    freezing + concatenating per block.
  - **Faster phase wrap**: a compare-and-subtract instead of `floor`, which
    GHC does not lower to a single instruction.
- The `-fllvm` backend gave no measurable gain here (GHC 9.4 predates the
  available LLVM 18). Remaining headroom: pack per-UGen state into a single
  unboxed record and specialize the `sin`/`cos` in the per-sample biquad
  coefficients (the dominant cost, and equally expensive in both engines).
- The two Rust bars show the project's size-optimized default profile costs
  ~20% throughput vs. a speed-optimized build.

Reproduce: `bash bench.sh 60 5` (end-to-end) or the two-duration slope method
in the commit description. Numbers vary with hardware; ratios are stable.

### Polyphony — the scaling test single-voice benchmarks hide

A single-voice benchmark is easy to over-trust, so here is the same patch
scaled to *N* independent voices (each its own filtered-saw + envelope at its
own frequency, all summed). Same slope method; `s/as` is DSP seconds per audio
second, `per-voice` is that divided by *N*.

| voices | rust `opt=s` s/as (per-voice) | rust `opt=3` s/as (per-voice) | haskell s/as (per-voice) |
|---:|---|---|---|
| 1  | 0.00180 (0.001804) | 0.00139 (0.001391) | 0.00267 (0.002671) |
| 8  | 0.01278 (0.001597) | 0.01032 (0.001289) | 0.01191 (0.001489) |
| 16 | 0.02684 (0.001677) | 0.02193 (0.001370) | 0.02267 (0.001417) |
| 32 | 0.05905 (0.001845) | 0.04941 (0.001543) | 0.04379 (**0.001368**) |
| 64 | 0.13801 (0.002156) | 0.10913 (0.001705) | 0.09155 (**0.001430**) |

The single-voice number **undersold** the Haskell version — and it turns out the
scaling curves point opposite ways:

- **Haskell's per-voice cost is flat** (~0.00143 from 8 voices up; the higher
  1-voice figure is just fixed per-render cost — topo sort, instantiation, the
  one output-buffer freeze — amortising out).
- **Rust's per-voice cost rises** with polyphony (0.00139 → 0.00171 for
  `opt=3`, +23% from 1 to 64 voices).
- They **cross over around 24–32 voices**. By 64 voices Haskell renders
  **~1.2× faster than speed-optimized Rust** and **~1.5× faster than the
  project-default Rust** (~11× vs ~9× vs ~7× real time).

Why the crossover? It is not a language effect — it is an algorithm difference
in *this* engine, and it is verifiable in the source. The Rust render loop
re-resolves wiring **every block**: for every node, for every input port, it
does a linear scan of the whole edge list
([`src/graph.rs`](../src/graph.rs) `self.edges.iter().find(...)`), plus two
`Vec` allocations per node per block. With *N* voices that is ~O(N²) per block.
This Haskell version binds each node's inputs **once at instantiation**, so the
render path is strictly O(total samples) with no per-block lookup or allocation
— hence the flat per-voice cost. A modestly optimized Rust engine that
pre-resolved its wire buffers (as SuperCollider does) would likely reclaim the
per-voice lead it shows at one voice; the point is only that *these two
implementations*, as written, scale differently, and the single-voice figure
does not predict the polyphonic one.

**Does Haskell blow up on GC at scale?** No. At 64 voices (`+RTS -s`):

| render | productivity | GC time | max residency | total memory |
|---|---|---|---|---|
| 30 s  | 99.7% | <0.5% | 5.7 MB | 19 MiB |
| 120 s | 99.8% | <0.5% | 21.6 MB | 49 MiB |

Allocation is short-lived nursery churn (~60 MB per audio-second — mostly the
per-sample biquad-coefficient tuple) that dies immediately; GC stays under 0.5%
of runtime and never pauses meaningfully. Max residency just tracks the output
buffer (numSamples × 4 bytes: 5.3 MB at 30 s, 21 MB at 120 s), not the voice
count — no leak, no superlinear growth. Killing that per-sample tuple would cut
the nursery traffic and buy a little more speed.

Reproduce: `bash gen_rust_poly.sh N > poly_N.synth` for the Rust side and
`microsynth-cli --synthdef poly --voices N` for Haskell.
