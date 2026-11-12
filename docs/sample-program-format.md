# Sample programs: wire format, binding and shared state

A sample program is one playable articulation of one sampled instrument at
one controller preset: the regions of a compiled instrument spec (an SFZ
program compiled to canonical JSON, every modulation evaluated at the preset)
packed into bytes a host uploads. A `sampleVoice` node is bound to a program
by id when its def is compiled, and plays one note of it. This document is the
contract between a host that compiles and uploads programs and the engine
(`src/sampler/program.rs`, `src/sampler/bank.rs`, `src/ugens/sample_voice.rs`).
The samples themselves travel separately (`docs/sample-bank-format.md`).

## Part 1 — byte format

Little-endian; strings are a `u32` length plus UTF-8 bytes; `NONE` is
`0xFFFFFFFF` where a `u32` field is optional.

```
magic:        4 bytes, "MSPG"
version:      u16                    (current: 1)
name:         string                 the spec's program id
spec_sha256:  32 bytes               sha256 of the canonical spec bytes
sample_count: u32
samples:      sample_count * 32 bytes   content hashes, the sample table
region_count: u32
regions:      region_count * Region
```

Each `Region`, in order:

```
sample          u32   index into the sample table; 0xFFFFFFFF = the silence generator
lokey hikey lovel hivel pitch_keycenter   5 * u8   inclusive MIDI numbers
pitch_keytrack  f32   cents per key
tune_cents      f32
transpose       i32   semitones
offset_frames   u32
end_frames      u32   last frame played (inclusive); NONE = the sample's last frame
loop_mode       u8    0 none, 1 one_shot, 2 continuous, 3 sustain
loop_source     u8    0 none, 1 sfz, 2 file, 3 whole (recorded only)
loop_start      u32   inclusive; NONE together with loop_end when there is no loop
loop_end        u32
volume_db amp_random_db pan width amp_veltrack     5 * f32  (pan, width, veltrack in percent)
velcurve_count  u8, then velcurve_count * (u8 velocity, f32 gain)
ampeg           7 * f32   delay attack hold decay sustain(%) release start(%)
seq_length      u32   0 = no sequence
seq_position    u32   1-based
has_rand        u8,  rand_lo f32,  rand_hi f32
group           i32
has_off_by      u8,  off_by i32
off_mode        u8    0 fast, 1 normal, 2 time
off_time        f32
trigger         u8    0 attack, 1 release, 2 release_key, 3 first, 4 legato
rt_decay        f32   dB per second held
delay_s         f32
note_polyphony  u32   0 = none
polyphony       u32   0 = none
bend_up_cents bend_down_cents   2 * f32
has_xf_vel      u8,  in_lo in_hi out_lo out_hi  4 * f32,  curve u8 (0 power, 1 gain)
filter_count    u8, then per filter:
                  type u8, cutoff f32, resonance f32 (dB), keytrack f32 (cents/key),
                  keycenter u8, veltrack f32 (cents), gain f32 (dB)
eq_count        u8, then per band:
                  type u8, freq f32, bw f32 (octaves), gain f32, vel2freq f32, vel2gain f32
```

Filter types: 0 `lpf_1p`, 1 `hpf_1p`, 2 `lpf_2p`, 3 `hpf_2p`, 4 `bpf_2p`,
5 `brf_2p`, 6 `lpf_4p`, 7 `hpf_4p`, 8 `lpf_6p`, 9 `hpf_6p`, 10 `peq`,
11 `lsh`, 12 `hsh`. EQ types: 0 `peak`, 1 `lshelf`, 2 `hshelf`. A voice runs
at most six filter stages (filters, then EQ bands) per region.

The decoder rejects a wrong magic, a newer version, truncated or trailing
bytes, an out-of-range tag, and a region that cannot play as written: an
empty key or velocity window, a value above 127, a sample index outside the
table, an inverted loop, a sequence position outside `1..=length`, a
non-finite number.

## Part 2 — from the instrument spec

The host compiles each spec region field by field; the units are the
spec's. What a program does not carry, and why:

| spec field | in the program |
| --- | --- |
| `offset_random_frames` | never: random start offsets are not honoured (rendering stays sample-deterministic); the field may be present or absent and is ignored |
| `cc_conditions` | no: a region in the spec already met them at the preset |
| `preset`, `articulation`, `deviations`, `reading`, `library`, `source` | no: they describe the compile; `spec_sha256` names it |
| `choke_groups` | no: derivable from the regions' `group`/`off_by` |
| program-level `bend_up_cents`/`bend_down_cents` | no: each region carries its own |
| `amp_random_db` | yes, drawn deterministically from the voice's `seed` |
| `sample.generator = "silence"` | `sample = 0xFFFFFFFF` |
| filter types `*_sv`, `pink`, `bpf_1p`, `brf_1p`, `apf_1p`, `bpk_2p` | refused by the host compiler (none in the catalog today) |

## Part 3 — registering and binding

`ms_sample_program_register(ptr, len)` decodes the bytes and binds the
program to the session's samples: every hash in its sample table must
already be registered (`ms_sample_register`), or registration fails. The
bound program (`BoundProgram`) keeps its own reference to each sample and
precomputes what voices read per note (the regions per key, each region's
128-point velocity curve, the sinc table). It returns the program id.

A def binds a `sampleVoice` node to a program through the IR's existing
side table, `table_bindings` (`IrTableBinding { node, table_id }`): no IR
format change. The id is resolved against the coefficient-table bank for a
table-bound kind and against the sample-program bank for a program-bound kind
(`UGenRegistry::register_program_bound`; `sampleVoice` is the only one).
`IrSynthDef::compile_with_banks(reg, tables, Some(programs))` does both;
`compile_with_tables` has no program bank and fails such a binding with
`IrError::ProgramNotFound`. Both raw IR exports (`ms_register_def_ir`,
`ms_compile_ir_with_tables`) pass the session's program bank. Validation
treats a program-bound kind like a table-bound one: a node of that kind with
no binding is `MissingTableBinding`.

Unlike a table, which is cloned into each compiled def, a program is shared:
every def compiled against it and every voice spawned from those defs hold
the same `Arc<BoundProgram>`. A host that wants independent round-robin and
choke state for two parts registers the program twice (two ids).

## Part 4 — shared state

Some region behaviour depends on more than one note, so the voices of one
program share a session (`Session`, behind a spin lock that is never
contended when the engine renders from one thread):

- **Sequence counters**, one per region, advanced at every note-on whose key
  the region's key window holds (sfizz's `sequenceCounter`); a region plays
  when `counter % seq_length == seq_position - 1`. A release region consults
  the flag its counter set at the last such note-on.
- **Held notes**, for `first` (no other note held) and `legato` (another held).
- **The event serial**, counting note-ons and note-offs; a note's `rand`
  draw is a hash of the voice's `seed` and this serial.
- **The sounding regions** of every voice (voice, region, group, `off_by`,
  note, velocity, start serial, whether it was turned off). A starting region
  turns off what it chokes and what its polyphony limits steal by marking
  those records with the sample time; each voice applies its marks at the
  start of its next block, at the marked sample if that falls in the block.
  An off therefore lands on time when the target voice renders after the
  choking one in a block, and at most one engine block late otherwise.

Voices render in spawn order within a block, so a session's history is a
function of the score alone: two renders of one score are identical.

## Part 5 — lazy loading and memory

A host uploads only what a render can reach. A region can sound only if its
key window holds a note the part plays and its velocity window holds a
velocity the part plays (release regions use the note-on velocity), so a
program compiled from the reachable regions only, with only their samples,
plays every note of that part exactly as the full program would: sequence
counters and chokes belong to regions, and an unreachable region never
advances, plays or chokes. (`first`/`legato` count notes, not regions.)

Memory: a program holds its samples (`ms_sample_program_bytes`); a voice holds
`ms_sample_voice_bytes()` more (its struct, 32 region players with their
filter states, block scratch), allocated when it is spawned. Rendering
allocates nothing, except that a program's table of sounding regions (Part 4)
grows past its first 256 entries if that many regions sound at once.

## Versioning

`version` gates the decoder; bump it, and extend `to_bytes`/`from_bytes`
together, on any change to Part 1.
