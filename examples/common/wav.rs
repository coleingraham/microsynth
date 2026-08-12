//! Shared WAV-writer helper for microsynth's `examples/` targets: interleave
//! N channels and write a canonical little-endian RIFF/WAVE file, in either
//! 16-bit PCM or 32-bit IEEE-float form.
//!
//! Included via `#[path = "common/wav.rs"] mod wav;` (examples have no
//! shared lib target to hang a real module off of) rather than copied, so
//! `render_partials_demo.rs` and `measure_limiter.rs` no longer each
//! hand-roll their own RIFF header. `src/bin/microsynth-cli.rs` has a third,
//! N-channel/16-bit copy of its own in a different build target (the
//! `std`-only CLI binary); it is left as-is here and could migrate to this
//! helper later.
//!
//! `#[path]`-included separately into each example binary, so each compiles
//! its own copy of this module and only exercises the entry point(s) it
//! calls — `write_f32` looks dead from `render_partials_demo`'s copy (it
//! only calls `write_pcm16`) and vice versa from `measure_limiter`'s. That's
//! a property of how examples share code, not real dead code, hence the
//! blanket allow below rather than pruning an entry point a sibling example
//! needs.
#![allow(dead_code)]

use std::fs;
use std::path::Path;

/// Interleave `channels` and write a 16-bit PCM WAV file, clamping each
/// sample to `[-1.0, 1.0]` before quantizing.
pub fn write_pcm16<C: AsRef<[f32]>>(channels: &[C], sample_rate: f32, path: impl AsRef<Path>) {
    let bytes = encode(channels, sample_rate, 16, 1, |buf, s| {
        let pcm = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&pcm.to_le_bytes());
    });
    fs::write(path, &bytes).expect("failed to write WAV file");
}

/// Interleave `channels` and write a 32-bit IEEE-float WAV file (full
/// precision, no clamping).
pub fn write_f32<C: AsRef<[f32]>>(channels: &[C], sample_rate: f32, path: impl AsRef<Path>) {
    let bytes = encode(channels, sample_rate, 32, 3, |buf, s| {
        buf.extend_from_slice(&s.to_le_bytes());
    });
    fs::write(path, &bytes).expect("failed to write WAV file");
}

/// `format_tag`: the WAVE `fmt ` chunk's format code (`1` = PCM, `3` = IEEE
/// float). `write_sample` appends one sample's encoded bytes to the buffer.
fn encode<C: AsRef<[f32]>>(
    channels: &[C],
    sample_rate: f32,
    bits_per_sample: u16,
    format_tag: u16,
    mut write_sample: impl FnMut(&mut Vec<u8>, f32),
) -> Vec<u8> {
    let num_channels = channels.len() as u16;
    let num_samples = channels.first().map_or(0, |c| c.as_ref().len());
    let byte_rate = sample_rate as u32 * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_size = num_samples as u32 * num_channels as u32 * (bits_per_sample as u32 / 8);
    let file_size = 36 + data_size;

    let mut buf: Vec<u8> = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&format_tag.to_le_bytes());
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for i in 0..num_samples {
        for ch in channels {
            let sample = ch.as_ref().get(i).copied().unwrap_or(0.0);
            write_sample(&mut buf, sample);
        }
    }

    buf
}
