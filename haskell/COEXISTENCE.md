# Haskell + Rust coexistence — authoring in one, deploying in the other

## Why this document exists

The benchmarks in [`README.md`](README.md) settled the performance question: for
the *runtime*, Rust wins — it is faster (≈1.1–1.3× at polyphony, ≈2.4× at a
single voice), has no GC to pause a real-time thread, runs `no_std`, and targets
`wasm32` and embedded cleanly. The Haskell version's advantages, meanwhile, all
live in the *authoring* layer: the DSL becomes the host language, SynthDefs are
first-class composable values, graphs are type-checked at build time, and the
DSP kernel needs no `unsafe`.

Those are not competing answers to the same question — they are the answers to
**two different questions**: "how do I *write* an instrument?" and "how do I
*run* one?" This document describes an architecture where each language does the
job it is best at, joined by a single well-defined artifact. It is aimed
squarely at building **instrument-authoring tooling for both human and AI
workflows**, not just at shipping the engine.

## The seam: a SynthDef is the handoff

A compiled SynthDef is already the natural boundary between "authoring" and
"running." This is not a novel idea — it is exactly how SuperCollider is built:
`sclang` (the language) compiles patches to a binary `.scsyndef` format, and
`scsynth` (a separate C++ server) loads and runs it. The authoring language and
the runtime are different languages, joined by a compiled graph.

> **Status (2026-07).** This document was written before any of it was built.
> Step 2 of the [pragmatic path](#a-pragmatic-path) is now **done on both sides**
> — but done *twice, incompatibly*. See
> [Current state](#current-state--two-irs-one-version-number) before treating any
> of the following as descriptive. The architecture below is still the intent;
> it is not yet the reality.

microsynth already has the same shape internally, and each side now *exposes* it
as an interchange format (`src/ir/` in Rust, `Microsynth.SynthDef.IR` in
Haskell):

- Rust's `SynthDef` (`src/synthdef.rs`) is an immutable template: node
  factories + edges + a parameter map + an output node.
- Haskell's `SynthDef` (`src/Microsynth/SynthDef.hs`) is the same thing as a
  pure value: `sdNodes :: [NodeDef]`, `sdOutput :: Int`, `sdParams :: [(String,
  Float)]`.

Both are first-order data. Serializing one and deserializing it in the other is
cheap and lossless.

## Architecture

```
Authoring front-ends            Interchange              Runtime
────────────────────            ───────────              ───────
Haskell EDSL   ─┐
Rust text DSL  ─┼──►   SynthDef IR (versioned)   ──►   Rust engine
GUI / patcher  ─┤      JSON (dev) / binary (prod)       native + WASM
LLM / agent    ─┘
```

- **Front-ends** express intent. Humans use the Haskell EDSL or the existing
  text DSL; a GUI patcher or an AI agent can target the same contract.
- **The IR** is the compiled graph — the contract. One format, many producers,
  one consumer.
- **The runtime** is the Rust engine (native + the WASM/AudioWorklet backend),
  which already knows how to instantiate and render a graph.

The point of the middle column is **decoupling**: *how you author* is
independent of *what you deploy*. Add a front-end without touching the runtime;
optimize the runtime without touching a front-end.

## The IR

The interchange is the flat, compiled graph — not a surface syntax. The original
sketch, mirroring the structure both engines already build:

```jsonc
{
  "version": 1,
  "name": "demo",
  "params": [ { "name": "freq", "default": 220 },
              { "name": "amp",  "default": 0.4 } ],
  "nodes": [
    { "id": 0, "kind": "Param", "args": { "name": "freq", "default": 220 } },
    { "id": 1, "kind": "Saw",   "inputs": [0] },
    { "id": 2, "kind": "Const", "args": { "value": 6 } },
    { "id": 3, "kind": "BinOp", "args": { "op": "Mul" }, "inputs": [0, 2] },
    { "id": 4, "kind": "Const", "args": { "value": 1.5 } },
    { "id": 5, "kind": "Lpf",   "inputs": [1, 3, 4] }
    /* … perc, muls … */
  ],
  "output": 8
}
```

Ship it as JSON in development (diffable, git-friendly, human-inspectable) and,
if size or load time ever matters, a compact binary encoding for production —
same schema, two serializations.

## Current state — two IRs, one version number

**The sketch above is what Haskell implements. Rust implements something else.**
Both shipped independently, and both call themselves format version 1, so the
version number currently carries no information. Neither can read the other:
`IrSynthDef::from_json` rejects Haskell's output at the first field.

| | Haskell (`Microsynth.SynthDef.IR`) | Rust (`src/ir/`) |
|---|---|---|
| version field | `version` | `format_version` |
| edges | inline `inputs: [id]` per node | separate `edges: [{from,to,to_input}]` |
| node shape | `{id, kind, args, inputs}` | externally-tagged `{"UGen": {kind, consts}}` |
| node identity | explicit `id`, validated `0..n-1` | positional index |
| params | `{name, default}` — discarded on decode | `{name, node, input, default}` — load-bearing |
| output | `output` | `output_node` |
| class / channels | absent | `class`, `output_channels` required |
| canonical form | JSON | binary (magic `MICROSYNTH-IR`); JSON is dev-only |
| inline consts | none — always separate `Const` nodes | `UGen { consts: [(port, value)] }` |
| kind tags | `SinOsc`, `Saw`, `Lpf`, `Perc` (spec names) | `sinOsc`, `saw`, `lpf`, `perc` (registry names) |
| arithmetic | one `BinOp` kind + `args.op` | four kinds: `Add`/`Sub`/`Mul`/`Div` |
| version policy | exact match — reject anything else | accept any `<= FORMAT_VERSION` |

The last three rows are the deep ones: the two sides disagree on the *node
model*, not just on spelling. Unifying them is a real design decision (which
format wins, and what the version bump costs), not a rename.

Until that lands, read "one format, many producers, one consumer" as the goal —
not as a description of the code.

## On "just emit the Rust text DSL"

Emitting the text DSL (`synthdef name p=d = body`) from Haskell's `Signal` is a
genuinely good **bootstrap**: the Rust parser already exists, so it is the
zero-new-Rust-code path to running Haskell-authored patches *today*. Do that
first to prove the loop end-to-end.

But evolve the boundary to the structured IR, for one decisive reason: **the
text DSL is the narrowest common denominator.** Serializing to a surface syntax
caps expressiveness at whatever the grammar supports, and — critically for AI
workflows — free text can fail to parse. The graph IR is the *widest* common
denominator: anything that can produce the structure can target it, and it is
unambiguous by construction.

## Human + AI workflows — where coexistence stops being a compromise

This is the actual payoff, and it is why the split is worth the cost.

- **The IR is the correct target for an LLM, not the surface syntax.** Give the
  IR a JSON Schema and a model can be constrained to it via structured
  output / tool-calling, so every generation is *well-formed by construction* —
  no "the DSL didn't parse" failure mode. A schema'd graph is a far more
  reliable agent loop than generating whitespace-sensitive text.

- **Haskell becomes the correctness oracle for machine-authored patches.** An
  agent proposes an instrument (as IR, or as EDSL); the Haskell layer
  type-checks arity and structure and can run static analysis the runtime can't
  cheaply do — unbounded feedback loops, DC offset, provably-silent graphs —
  *before* anything reaches the audio thread. Types + purity make that static
  reasoning tractable. "Haskell as the linter/compiler for AI-generated
  instruments" is an honest, strong fit.

- **Higher-order UGens are the vocabulary agents should speak.** A model does
  far better with `supersaw`, `pluckedEnsemble`, `fmBell` than with raw node
  graphs. Haskell is the natural home for combinators that *expand* to IR: the
  agent targets a small, rich vocabulary; Haskell lowers it to the flat graph
  the runtime executes.

- **Property-based generation for exploration and eval data.** Haskell +
  QuickCheck can generate *valid* random SynthDefs, render them, measure
  features, and shrink counterexamples — ideal for an agent exploring a design
  space, or for building training/evaluation corpora for the AI side.

In short: the same schema'd IR that a GUI and a human author against is the
schema an AI is *constrained* to, and Haskell is the shared brain that checks,
lowers, and generates on both paths.

## Two engines as a feature, not a tax

Maintaining a second engine only pays if it earns its keep. It does — through
**differential testing**. Render the same SynthDef through the Haskell engine
and the Rust engine and assert sample-level agreement. That would make the
Haskell engine an **executable specification / reference oracle** for the Rust
one: any divergence is a bug in exactly one place, found automatically. The
"duplicate implementation" flips into a cross-check that keeps the fast engine
honest.

> **Not yet true.** No cross-engine check runs anywhere. Each engine has its own
> tests; CI runs both, but nothing compares one to the other. An earlier draft
> cited "the scaffold produced a byte-identical WAV against the Rust engine for
> the demo patch" as a seed — that was a one-off manual observation, never
> automated, and two things since have made it a shakier foundation than it
> sounded:
>
> - **Bit-exactness across engines is not a reachable bar for every patch.** The
>   Haskell suite's own golden for `tone` (a raw sine) was pinned on one machine
>   and never reproduced on another: libm's `sin` differs by an ulp across
>   platforms, so a patch evaluating `sin` at many distinct arguments is not
>   bit-reproducible even between two runs of the *same* engine. It is now
>   checked against an analytic reference instead. Differential testing should
>   assert agreement within a tolerance, not byte equality.
> - **The two engines' `poly` patches are not actually the same graph.** The
>   frequencies are re-typed by hand in awk (`gen_rust_poly.sh`) and rounded to
>   4 decimal places, so the benchmark comparison measures two subtly different
>   patches. See "The honest caveat" below — this is exactly the drift it warns
>   about, already happening.

## Division of labor

| Concern | Home | Why |
|---|---|---|
| Patch authoring (human) | Haskell EDSL / text DSL | expressive, type-checked, composable |
| Macros / higher-order UGens | Haskell | metaprogramming lowers to IR |
| Static checks & analysis | Haskell | types + purity make it tractable |
| Generative / property-based search | Haskell + QuickCheck | valid-by-construction, shrinkable |
| AI-facing contract | **SynthDef IR (schema'd)** | constrainable, unambiguous |
| Reference/oracle rendering | Haskell engine | cross-checks the runtime |
| Real-time rendering (native + WASM) | Rust | fast, no GC, `no_std`, wasm32 |
| Deployment artifact | Rust engine + IR loader | one runtime, many front-ends |

## The honest caveat

Two full engines is a real maintenance cost, and it only pays under discipline:

- **Single-source the DSP *algorithms* conceptually** (same math, cross-checked
  by differential tests). Do not let two independent 40-UGen libraries drift.
- **Let the languages diverge only where each is strong** — Haskell *above* the
  graph (authoring, macros, checking, generation), Rust *below* it (the kernel
  and deployment). The IR is the line between them.
- If instead both grow full, independent feature sets, you have simply doubled
  the work for little gain.

Kept to that discipline, the model is: **Haskell is the compiler and the brain,
Rust is the substrate, and a versioned SynthDef IR is the contract between
them** — with that IR doubling as the schema both human tooling and AI agents
author against.

## A pragmatic path

Status as of 2026-07. Step 1 was skipped rather than done; step 2 was done twice,
incompatibly (see [Current state](#current-state--two-irs-one-version-number)),
which is why step 0 now exists.

| | Step | Status |
|---|---|---|
| 0 | **Unify the two IRs.** Pick one node model and one wire format; make the other side conform; bump the version once, deliberately. Prerequisite for 3 and 4 — a contract that two implementations read differently is not a contract. | ⬜ **not started, now the blocker** |
| 1 | **Prove the loop (bootstrap).** Pretty-print Haskell `SynthDef` → text DSL; run it through the existing Rust CLI. No new Rust code. | ⬜ **skipped** — the project went straight to step 2. No pretty-printer exists. Retained here only because it is still the cheapest way to smoke-test a Haskell-authored patch through the Rust runtime. |
| 2 | **Define the IR.** Add a JSON (de)serializer to Haskell's `SynthDef` and a loader on the Rust side (deserialize graph → instantiate). Version it. | ✅ **done on both sides** — `Microsynth.SynthDef.IR` (JSON, aeson) and `src/ir/` (binary + JSON). ❌ **but not as one format.** |
| 3 | **Lock the contract with differential tests.** Render the same IR through both engines in CI; assert sample-level agreement on a corpus of patches. | ⬜ not started. Blocked on 0. Note the assertion must be *within tolerance*, not byte-exact — see the caveat under "Two engines as a feature". |
| 4 | **Publish the schema.** JSON Schema for the IR becomes the target for GUIs, agents (structured output), and validators. | ⬜ not started. Blocked on 0 — there is no single schema to publish yet. |
| 5 | **Grow the authoring brain in Haskell.** Higher-order UGens, static analyses, and QuickCheck generators — all lowering to the same IR the runtime already consumes. | ⬜ not started. Haskell implements 8 UGen kinds; Rust has ~60. |
