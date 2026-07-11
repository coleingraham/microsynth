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

microsynth already has the same shape internally — it is just not yet *exposed*
as an interchange format:

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

The interchange is the flat, compiled graph — not a surface syntax. A first cut,
mirroring the structure both engines already build:

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
and the Rust engine and assert sample-level agreement. (The seed already exists:
the scaffold produced a byte-identical WAV against the Rust engine for the demo
patch.) That makes the Haskell engine an **executable specification / reference
oracle** for the Rust one: any divergence is a bug in exactly one place, found
automatically. The "duplicate implementation" flips into a cross-check that
keeps the fast engine honest.

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

1. **Prove the loop (bootstrap).** Pretty-print Haskell `SynthDef` → text DSL;
   run it through the existing Rust CLI. No new Rust code.
2. **Define the IR.** Add a JSON (de)serializer to Haskell's `SynthDef` and a
   loader on the Rust side (deserialize graph → instantiate). Version it.
3. **Lock the contract with differential tests.** Render the same IR through
   both engines in CI; assert sample-level agreement on a corpus of patches.
4. **Publish the schema.** JSON Schema for the IR becomes the target for GUIs,
   agents (structured output), and validators.
5. **Grow the authoring brain in Haskell.** Higher-order UGens, static analyses,
   and QuickCheck generators — all lowering to the same IR the runtime already
   consumes.
