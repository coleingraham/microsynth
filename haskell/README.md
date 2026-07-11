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
| Rust, `opt-level=3` + LTO | 0.00157 s | ~637× | **3.0× faster** |
| Rust, `opt-level="s"` (project default) | 0.00198 s | ~504× | **2.4× faster** |
| Haskell, `-O2` (this scaffold) | 0.00476 s | ~210× | 1.0× (baseline) |

Takeaways:

- The Haskell scaffold is **~2.4–3.0× slower** than Rust for identical DSP, and
  still renders this patch **~210× faster than real time** — comfortably
  real-time-capable.
- The gap is expected and mostly *unoptimized-scaffold* overhead, not a
  language ceiling: node state lives in per-`STRef` cells read/written per
  sample, and node outputs are reached through a boxed vector indirection.
  Known levers to close it (not applied here): pack per-UGen state into a
  single unboxed record, specialize the inner loops, drop the per-node output
  indirection, and try the `-fllvm` backend.
- The two Rust bars show the project's size-optimized default profile costs
  ~25% throughput vs. a speed-optimized build.

Reproduce: `bash bench.sh 60 5` (end-to-end) or see the two-duration slope
method in the commit description. Numbers vary with hardware; ratios are stable.
