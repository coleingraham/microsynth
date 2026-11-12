# Sample bank: wire format, ids and memory

This document is the contract for the audio a sample program plays: the bytes
a host uploads for one sample, how the session's sample bank keys and counts
them, and the exports that carry them. The programs that reference these
samples are described in `docs/sample-program-format.md`; the node that plays
them is `sampleVoice` (`src/ugens/sample_voice.rs`).

The engine never decodes an audio file. The host reads the file, decodes it
to 32-bit float samples at the file's own rate and uploads that. Everything
here is implemented in `src/sampler/pcm.rs`.

## Part 1 — the upload byte format

All integers are little-endian; strings are a `u32` byte length followed by
that many UTF-8 bytes.

```
magic:        4 bytes, "MSSP"
version:      u16                (current: 1)
name:         string             (may be empty; e.g. the sample's path)
sha256:       32 bytes           (content hash of the source file's bytes)
sample_rate:  f32                (Hz, the file's own rate, > 0)
channels:     u16                (1 or 2)
frames:       u32                (frames per channel, > 0)
data:         channels * frames * f32, planar: channel 0's frames, then channel 1's
```

The decoder rejects a wrong magic, a newer version, a shape outside the
limits above, a payload shorter or longer than `channels * frames * 4` bytes,
and any NaN or infinite value.

`sha256` is the key every program uses to name a sample: the hash of the
file's bytes exactly as the instrument spec records it (`sample.sha256`), not
a hash of the decoded audio. The engine does not check it against the data;
the host does (Part 4).

## Part 2 — the bank

`SamplePcmBank` holds the session's samples by `PcmId` (issued from 1; `0` is
never issued), by content hash and by name.

- **Register** stores a sample and returns its id. A sample whose hash is
  already registered is not stored again: the existing id comes back and the
  first upload is kept. Two uploads of one file therefore cost its memory
  once.
- **Free** drops the bank's reference. A program registered with that
  sample keeps its own (`Arc`) reference, so its audio lives until the
  program is freed too.
- **Lookup** by id, by hash (`id_for_hash`), by name.

Stored layout: each channel is kept with `PCM_PAD` (42) zero frames before
and after its audio, so the interpolation kernels read any position from the
first frame to one past the last without a bounds check. A sample costs

```
bytes = 4 * channels * (frames + 2 * PCM_PAD)
```

which is what the bank counts: `bytes()` per sample, the total per session in
`SamplePcmBank::bytes()` / `ms_sample_bytes_total()`, and the bytes a program
holds (each of its samples once) in `ms_sample_program_bytes(id)`.

## Part 3 — exports

Raw C exports in `src/web.rs`, same pointer-plus-length pattern as the other
uploads (bytes copied into `ms_alloc`'d memory first):

| export | effect | returns |
| --- | --- | --- |
| `ms_sample_register(ptr, len)` | decode Part 1, register | id (`> 0`; the existing id for a known hash), `0` on failure |
| `ms_sample_free(id)` | drop the bank's reference | — |
| `ms_sample_id_for_hash(ptr, len)` | look up by hash: 32 raw bytes or 64 hex digits | id or `0` |
| `ms_sample_bytes_total()` | bytes held by the session's samples | f64 (exact past 2 GiB) |
| `ms_sample_program_register(ptr, len)` | see `sample-program-format.md` | id or `0` |
| `ms_sample_program_free(id)` | | — |
| `ms_sample_program_id_for_name(ptr, len)` | | id or `0` |
| `ms_sample_program_bytes(id)` | bytes of sample audio the program holds | f64, `-1` if unknown |
| `ms_sample_voice_bytes()` | bytes one `sampleVoice` node holds besides the samples | u32 |

Every session-start export (`ms_init`, `ms_init_with_bus`, `ms_routing_init`)
starts empty sample and program banks, as it does the coefficient-table bank.

## Part 4 — what the host does

The engine takes decoded floats only, so decoding, verification and the
choice of what to upload are the host's:

- **Decode** to f32 at the file's own rate and channel count (1 or 2). The
  reference host (`render_score.mjs`, motif-core) decodes RIFF/WAVE itself:
  16-, 24- and 32-bit integer PCM and 32/64-bit float, including
  `WAVE_FORMAT_EXTENSIBLE`; integers are scaled by `1 / 2^(bits-1)`. It
  decodes FLAC through `ffmpeg` when one is on the path. MP3 and Ogg Vorbis
  samples are refused with a named error: their decoders add or drop frames
  at the edges (encoder delay and padding), so the frame counts, offsets and
  loop points the spec took from the source file would not line up.
- **Verify** the file's sha256 against the spec's `sample.sha256`, and the
  decoded frame count, rate and channels against the spec's; a mismatch is a
  loud error, never a substitute.
- **Upload only what can sound**: the samples of the regions whose key and
  velocity windows the score can reach (`sample-program-format.md`, "Lazy
  loading").
