# Coefficient-table bank: wire format and id-reference mechanism

This document is the layout contract for microsynth's runtime coefficient-table
bank (`src/coeff_table.rs`): the byte format a table upload carries, the bank's
id/name semantics, and how a compiled graph node references a table by id. Any
producer that uploads tables (an exporter, a host application) and any
consumer that reads them (a table-bound ugen) builds against this document.

## Why this exists

Microsynth ships as generic, source-only infrastructure: no per-pitch
coefficient content is ever compiled into this crate. A ugen that needs
per-pitch coefficient data (partial frequencies plus weighted coefficient
sets) gets it exclusively by resolving a runtime-uploaded table, identified
by an id, the same way SuperCollider's `Buffer`/bufnum model lets a UGen
reference sample data that was never part of the SynthDef definition itself.
This document describes the resulting mechanism's structure only — what the
bytes look like and how a graph node binds to them — not why any particular
caller uploads what it uploads.

## Part 1 — `CoeffTable` byte format

Implemented by `CoeffTable::to_bytes`/`CoeffTable::from_bytes`
(`src/coeff_table.rs`). All multi-byte integers are little-endian. Strings
are a `u32` byte length followed by that many UTF-8 bytes (no NUL
terminator).

```
magic:          4 bytes, literal "MSCT"
format_version: u16                  (current: 1)
name:           string               (may be empty)
entry_count:    u32
entries:        entry_count * PitchEntry
```

Each `PitchEntry`:

```
f0_hz:                 f32
inharmonicity_stretch:  f32
partial_freq_count:    u32
partial_freqs:         partial_freq_count * f32   (Hz, at this entry's f0)
k_channels:             u32
j_noise:                u32
coefficient_count:      u32
coefficients:           coefficient_count * f32
metadata_count:         u32
metadata:               metadata_count * (string key, f32 value)
```

### Coefficient layout

`coefficients` is row-major by channel: `k_channels` rows, each
`partial_freq_count + j_noise` columns wide. Element
`coefficients[k * (partial_freq_count + j_noise) + c]` is channel `k`'s
weight on column `c` — partial columns first (indices
`0..partial_freq_count`), then noise-band columns
(`partial_freq_count..partial_freq_count + j_noise`).

`coefficient_count` must equal `k_channels * (partial_freq_count + j_noise)`;
a decoder rejects an entry whose stored count doesn't match this product
(`CoeffTableCodecError::MalformedEntry`) rather than accepting a payload a
consumer could only misinterpret.

### Why both explicit partial frequencies and a stretch factor

`partial_freqs` is the fixed, explicit frequency list this entry was built
against (Hz, at `f0_hz`). `inharmonicity_stretch` is carried alongside it,
not derived from it, because the two serve different consumers: a decoder
that only needs *this* entry's frequencies reads `partial_freqs` directly;
one that needs to re-derive frequencies for a pitch *between* two entries
(interpolating toward a neighboring entry's f0) needs the stretch factor as
an independent, continuously-varying quantity, not something back-solved
from a fixed list. Storing both avoids forcing every consumer through the
inverse computation.

### Metadata

`metadata` is an open `(string, f32)` list, order-preserving. It exists so a
future field can travel in the format without a version bump: a decoder that
doesn't recognize a key simply doesn't look for it; an encoder adds a key
only when it has a value for it. No metadata keys are reserved by this
document today.

### Versioning

`format_version` gates the decoder (`CoeffTable::from_bytes` rejects a
stored version newer than the build's own `FORMAT_VERSION` constant in
`src/coeff_table.rs`, mirroring `ir::FORMAT_VERSION`'s same back-compat
rule). Bump it, and extend `to_bytes`/`from_bytes` together, on any change to
this byte layout.

## Part 2 — the bank: register, replace, free

`CoeffTableBank` (`src/coeff_table.rs`) holds tables by `TableId(u32)`,
optionally indexed by name.

- **`register(table) -> TableId`** issues a fresh id (starting at 1 — `0` is
  reserved as a "no table" sentinel, never issued) and stores `table` under
  it.
- **`replace(id, table) -> Result<(), CoeffTableBankError>`** overwrites the
  content at an *already-registered* id, in place. The id is unchanged, so
  every existing reference to it resolves to the new content on next
  resolution. Errors (`NotFound`) rather than creating a new entry if `id`
  isn't currently registered — `replace` never doubles as `register`.
- **`free(id)`** removes a table. A no-op if `id` isn't registered.
- **`get`/`get_by_name`/`id_for_name`** are read-only lookups.

## Part 3 — the runtime upload ABI

Raw C exports in `src/web.rs` (see that file's "Coefficient-table bank
exports" section for the exact signatures), following the same
pointer-plus-length-into-`ms_alloc`'d-memory pattern every other byte/string
upload in this crate's ABI uses:

| Export | Effect | Return |
| --- | --- | --- |
| `ms_coeff_table_register(data_ptr, data_len)` | decode + `register` | new id (`> 0`), or `0` on failure |
| `ms_coeff_table_replace(id, data_ptr, data_len)` | decode + `replace` | `0` on success, `1` on failure |
| `ms_coeff_table_free(id)` | `free` | — |
| `ms_coeff_table_id_for_name(name_ptr, name_len)` | `id_for_name` | id (`> 0`), or `0` if unresolved |

A session's bank starts empty and is reset by every session-start export
(`ms_init`, `ms_init_with_bus`, `ms_routing_init`) — the same lifecycle as
the other per-session registries in that file.

## Part 4 — the id-reference mechanism (graph nodes)

A DSL/IR-compiled synthesis graph is built from **kinds** registered in a
`UGenRegistry` (`src/dsl/compiler.rs`). Every kind registered the ordinary
way (`register`/`register_spec`) is built from a bare `fn() -> Box<dyn UGen>`
factory — no per-instance data, by construction. A kind that needs a
resolved coefficient table at construction time cannot use that path: a bare
`fn` item cannot close over a value resolved from a bank lookup. This
document's mechanism adds a second, parallel registration path for exactly
that case:

- **`UGenRegistry::register_table_bound(name, factory, category, inputs,
  outputs)`** registers a kind under a disjoint map, keyed by the same kind
  of `name` string, whose factory type is `Arc<dyn Fn(Arc<CoeffTable>) ->
  Box<dyn UGen> + Send + Sync>` — a closure, not a bare pointer, so it can
  capture the resolved table.
- **`IrTableBinding { node, table_id }`** (`src/ir/mod.rs`) is a side table
  on `IrSynthDef` (`table_bindings: Vec<IrTableBinding>`), parallel to
  `IrParam`: it names which node index should be built via the table-bound
  path, and which table id to resolve for it. It is not a new `IrNode`
  variant, because a table reference is not a value flowing through the
  audio graph the way a `Const`/`Param` output is — it identifies what a
  node's *construction* should be bound to, not a signal any edge could
  carry.
- **`IrSynthDef::compile_with_tables(reg, bank)`** resolves every
  `table_bindings` entry against `bank` (erroring `TableNotFound` if an id
  isn't registered), clones the resolved `Arc<CoeffTable>` into that node's
  factory closure, and otherwise builds the graph exactly as the plain
  `compile(reg)` path does. `compile(reg)` itself is unchanged: a document
  whose `table_bindings` names a table-bound-only kind fails there with
  `UnknownKind`, since that kind has no bare factory — a deliberate refusal
  rather than silently building an unfilled node.
- **`IrSynthDef::validate(reg)`** additionally checks that every node whose
  kind is registered *only* as table-bound has a `table_bindings` entry
  naming it (`IrError::MissingTableBinding` if not) — the check that turns
  "forgot to bind a table" into a caught error instead of a node that can
  never be constructed.

### Resolution is a snapshot, not a live handle

`compile_with_tables` clones the table's content into the compiled graph's
factory closure at compile time. A synth built this way does **not**
observe a later `CoeffTableBank::replace`/`free` on the same id — only a
*subsequent* `compile_with_tables` call sees updated or removed content.
Proving replace/free "work" means proving a later compile sees the change,
not that an already-running synth updates live. A consumer that wants live,
per-block re-resolution (analogous to how some SuperCollider UGens re-read a
buffer's current content every block via its bufnum, rather than a
snapshot) would need a bank handle reachable from the audio-render path
itself — this mechanism does not provide that, and no such reachability
change is made here.

### Format-version interaction

`IrSynthDef::table_bindings` is a version-3+ field
(`ir::FORMAT_VERSION`). A version-1 or version-2 document has no trailing
section in the binary form and no `table_bindings` key in the JSON form;
both decoders default it to an empty list rather than erroring, so
pre-existing documents keep decoding unchanged.

### What this mechanism does not (yet) provide

- **No DSL text syntax.** The DSL lexer has no string/id-literal tokens, and
  none were added here. A table-bound kind is reachable only by
  constructing (or programmatically producing) an `IrSynthDef` with a
  `table_bindings` entry — there is no way to write a table reference in
  DSL source today. Extending the DSL surface, if a future consumer wants a
  text-authorable reference, is that consumer's decision, not assumed by
  this document.
- **No wasm-bundle graph-compile reachability.** The wasm build profiles
  compile with the `ir` crate feature off, so `compile_with_tables` is not
  reachable from the `ms_compile`/`ms_register_def`/`ms_compile_def`
  raw-ABI entry points, which go through the plain DSL compiler. The
  upload/replace/free/lookup ABI (Part 3) works in every build regardless —
  only the "bind a table id onto a specific compiled node" step is
  `ir`-feature-gated (native/offline tooling) today.

## Consumers of this document

A ugen that reads coefficient data registers itself via
`UGenRegistry::register_table_bound` and reads its bound `Arc<CoeffTable>`
per the layout in Part 1. An external producer that uploads tables encodes
to the Part 1 byte format and drives the Part 3 ABI. Both build against this
document as the shared contract; neither needs to read the other's source.
