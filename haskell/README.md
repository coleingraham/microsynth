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
