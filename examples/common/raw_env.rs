//! Shared raw-envelope reader for microsynth's `examples/` targets: decode a
//! frame-rate-to-audio-rate-upsampled f32 envelope (gain, freq, or any other
//! per-block control curve) from a raw little-endian f32 file — the format
//! the Python driver side (`synth_worldmodel.channel_export.upsample_gain`)
//! writes with a plain `.tofile()`.
//!
//! Included via `#[path = "common/raw_env.rs"] mod raw_env;` (examples have
//! no shared lib target to hang a real module off of), the same pattern
//! `wav.rs` already uses in this directory. Extracted from
//! `render_dictionary_ab.rs`'s own (formerly private, verbatim-duplicated)
//! `read_f32_raw` once `render_channels_ab.rs` needed the identical function.

use std::fs;
use std::path::Path;

/// Read a raw little-endian f32 envelope file (one sample per audio sample).
/// Panics on a read failure or a length that isn't a multiple of 4 bytes —
/// both indicate a mismatched/corrupt input, not a recoverable condition for
/// a manual staging example.
pub fn read_f32_raw(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    assert!(
        bytes.len().is_multiple_of(4),
        "{path:?} length {} is not a multiple of 4",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
