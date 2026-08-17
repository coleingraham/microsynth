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

**No padding slots.** This is a normative statement of what that equality
means, not just a decode-time check: each entry's coefficient row is
*exactly* `partial_freq_count + j_noise` values wide, at that entry's own
width — never a wider, uniform width shared across a batch of differently-
sized entries. A producer that fits internally at such a wider width (e.g. a
fixed `C_max` across a batch of pitches whose partial counts vary) must mask
or slice each entry down to its own `partial_freq_count + j_noise` before
encoding — not merely truncate a wider vector and ship the prefix. Once
sliced, each channel's row must itself be a full simplex — sum to ~1 over
that entry's own `partial_freq_count + j_noise` slots — because **the ugen
does not renormalize at read time**: a stored row that isn't already a
simplex at its own width decodes and is used exactly as stored, silently.

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
only when it has a value for it.

Four keys are reserved today, all consumed by the direct-synthesis ugen
(`microsynth::ugens::partials::PartialsNoise`, MOT-636/MOT-641) as the
governing RFC's "deterministic bridge" — column mass -> sinusoid amplitude
and bump mass -> noise-generator gain, plus (MOT-641) the noise-band span a
table's noise basis was fit against — all per-entry scalars an analysis-side
producer computes once from its own window/FFT/basis configuration:

- **`mainlobe_gain`** (f32): the analysis window's mainlobe-gain scalar —
  multiplies every partial coefficient in the entry to convert its L1 mass
  into a time-domain sinusoid amplitude. Absent: defaults to `1.0` (the
  consuming ugen treats coefficients as amplitudes directly).
- **`noise_gain`** (f32): the expected-white-noise-magnitude scalar for the
  entry's noise bands — multiplies every noise-band coefficient to convert
  its bump mass into a noise-generator gain. Absent: defaults to `1.0`, same
  treatment. Derived assuming a unit-variance noise source and an ideal
  (unity in-band gain) bandpass on the consuming side; `PartialsNoise`
  (MOT-641) rescales its noise source and applies a per-band power-gain
  compensation specifically so those two assumptions hold at render time —
  see that ugen's module doc's "deterministic bridge" section.
- **`noise_band_min_hz`** / **`noise_band_max_hz`** (f32, MOT-641): the Hz
  span `[min, max]` the entry's noise-band columns were fit against (e.g.
  `[0, Nyquist]` at the analysis sample rate — `motif-soundmatch`'s
  `channels.py::noise_basis` mel-spans exactly this). Either absent: the
  consuming ugen falls back to its own hardcoded default span (`[80,
  12000]` Hz in `PartialsNoise` today — see `NOISE_BAND_MIN_HZ`/
  `NOISE_BAND_MAX_HZ` in `partials.rs`). Present, a well-formed table writes
  the *same* span into every entry (not varying entry-by-entry) — see the
  next paragraph for why.

Absent-key defaults are safe only for deliberately-unscaled synthetic tables
(e.g. microsynth's own Rust-side test tables); a producer exporting real
fitted dictionaries MUST write `mainlobe_gain`/`noise_gain` explicitly, since
a fitted table rendered at the 1.0 defaults plays its partials on the order
of 1700x too hot (at `n_fft=1024`, the real `mainlobe_gain` scalar is
~5.8e-4, not 1.0). `motif-soundmatch`'s `channel_export.py` enforces this at
encode time (`encode_coeff_table`'s `allow_unscaled` guard, MOT-649 F10);
any other producer of this format must enforce an equivalent guard of its
own. `noise_band_min_hz`/`noise_band_max_hz` are not held to this same
enforced-at-encode-time bar: their absence degrades to a documented fallback
span rather than an order-of-magnitude amplitude error, so a producer may
omit them (e.g. a synthetic table with no real analysis-side basis to
report).

All four are per-entry, not per-column: each relationship is fixed by a
table's analysis-time window/FFT/basis configuration, which today is
constant across an entry's own partial/noise columns rather than varying
column-by-column. None of the four encodes which of an entry's `k_channels`
a consumer should render — that selection is a construction-time parameter
of the consuming ugen (`PartialsNoise`'s `with_channel`), not carried in the
wire format.

`noise_band_min_hz`/`noise_band_max_hz` are additionally per-*table*, not
truly per-entry, despite living in each entry's metadata slot: the noise
basis a table's coefficients were fit against is one fixed thing for the
whole table (RFC: "`N` — a **fixed** bins x J smooth noise basis"), so band
identity (what center frequency "band index j" means) must not depend on
which entry of the table is currently active — a glide that changed a
band's frequency mid-note between two bracketing entries would be audible
and wrong. A well-formed table's exporter therefore writes the identical
span into every entry, and a consumer (`PartialsNoise`) reads only one
entry's worth (its first) rather than interpolating or varying it per
entry — see `partials.rs`'s `noise_band_span` for the read side of this
convention.

This is enforced, not just documented (MOT-641 QA F8): `CoeffTable::from_bytes`
rejects a table whose entries disagree on these two keys — one entry carrying
only one of the pair, or two entries carrying different values — with
`CoeffTableCodecError::InconsistentNoiseBandSpan` (`coeff_table.rs`). Every
entry having neither key is a valid agreement (the whole table falls back
together); a table with 0 or 1 entries trivially satisfies the rule. This
check only runs at decode time, the same as `MalformedEntry` — a hand-built
`CoeffTable` that bypasses `from_bytes` is not protected by it, same
carve-out as every other decode-time check in this format.
`motif-soundmatch`'s `channel_export.py::encode_coeff_table` makes the
mirrored assertion at encode time, so a producer bug is caught before the
bytes are even written, not only when microsynth later decodes them.

Before MOT-641, the noise-band scalar bridged between the fixed
smooth-noise basis a table's coefficients were fit against — defined by the
analysis-side channel model (`motif-soundmatch`'s `channels.py`, MOT-633) as
mel-spaced/cepstral smooth bumps — and whatever fixed noise-band realization
a consuming ugen happened to render with, with no mechanism for the two to
describe the same band structure. `noise_band_min_hz`/`noise_band_max_hz`
close that gap: a table that supplies its true fit span lets the consuming
ugen's band-center formula (which MOT-641 also changed to match
`channels.py::noise_basis`'s own mel-edges construction exactly, not just
its span) place bands at the same center frequencies the analysis side fit
against.

No other metadata keys are reserved today.

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

### Wasm-bundle graph-compile reachability (MOT-640)

**Provided from this crate's own build scripts.** Both `web/build.sh` wasm
profiles (the raw AudioWorklet build and the wasm-bindgen main-thread build)
compile with the `ir` crate feature on — it is pure `alloc`+`core` (see
`Cargo.toml`'s `ir` feature comment), so this adds no new dependency, only
the IR codec + `compile_with_tables` code itself. A new raw export,
`ms_compile_ir_with_tables(ir_ptr, ir_len)` (`src/web.rs`), takes a
serialized `IrSynthDef` (the `ir::serialize` binary wire format —
`IrSynthDef::to_bytes`/`from_bytes`) instead of DSL text, bypassing the DSL
surface entirely (the lexer gap below is unchanged and not what this
closes), and resolves the document's `table_bindings` against the session's
coefficient-table bank via `IrSynthDef::compile_with_tables` before loading
the result as the render sink. `ms_compile`/`ms_compile_def` (DSL text,
table-unaware) are unchanged and remain reachable exactly as before — this
is a new, additive entry point, not a replacement.
`register_table_bound_builtins` is now called alongside `register_builtins`
at every `UGenRegistry` construction site in `src/web.rs`, so
`partialsNoise` (and any future table-bound kind) is resolvable by name
wherever a document's `table_bindings` names it. Round-trip proof: upload a
table via `ms_coeff_table_register`, build a `partialsNoise` `IrSynthDef`
with a matching `table_bindings` entry, pass its serialized bytes to
`ms_compile_ir_with_tables`, then render via `ms_render` — non-silence
(`tests/wasm_ir_table_reachability.rs`). `tests/wasm_abi_stability.rs`
tracks `ms_compile_ir_with_tables` as an addition, not a replacement, of the
pre-existing raw-export surface.

**Not provided by an external build that doesn't opt into `ir`.** This
crate's own `Cargo.toml` `std` feature does not itself enable `ir`
(`default = ["std", "ir"]` enables both only via the default set; a build
invoked with `--features std --no-default-features` — as an external
build script may do — gets `std` alone, without `ir`, and without this
export). A wasm build produced that way does not contain
`ms_compile_ir_with_tables` regardless of what this section says about
`web/build.sh`; that gap belongs to whichever build script made that
feature choice, not to this document's mechanism.

### What this mechanism does not (yet) provide

- **No DSL text syntax.** The DSL lexer has no string/id-literal tokens, and
  none were added here. A table-bound kind is reachable only by
  constructing (or programmatically producing) an `IrSynthDef` with a
  `table_bindings` entry — there is no way to write a table reference in
  DSL source today. Extending the DSL surface, if a future consumer wants a
  text-authorable reference, is that consumer's decision, not assumed by
  this document.

## Consumers of this document

A ugen that reads coefficient data registers itself via
`UGenRegistry::register_table_bound` and reads its bound `Arc<CoeffTable>`
per the layout in Part 1. An external producer that uploads tables encodes
to the Part 1 byte format and drives the Part 3 ABI. Both build against this
document as the shared contract; neither needs to read the other's source.
