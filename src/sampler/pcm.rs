//! Registered sample audio: decoded f32 PCM, one or two channels, keyed by
//! the content hash of the file it was decoded from.
//!
//! The engine never decodes an audio file. A host decodes (WAV, FLAC, ...)
//! and uploads planar f32 frames in the byte format below
//! (`docs/sample-bank-format.md`, Part 1); the bank stores them with zero
//! padding on both sides so the interpolation kernels in
//! [`super::interp`] read a fixed window around any position without a
//! bounds check.

use crate::wire::{Reader, WireError, put_f32, put_str, put_u16, put_u32};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Wire-format magic prefix of an uploaded sample.
pub const PCM_MAGIC: &[u8] = b"MSSP";
/// Wire-format version of an uploaded sample. Bump on any layout change.
pub const PCM_FORMAT_VERSION: u16 = 1;
/// Zero frames stored before and after every channel's audio. The widest
/// interpolation window ([`super::interp::SINC_MAX_HALF_TAPS`] taps on each
/// side of the read position, plus one) stays inside the padding when the
/// read position is anywhere from the first frame to one past the last.
pub const PCM_PAD: usize = super::interp::SINC_MAX_HALF_TAPS + 2;
/// Most channels a sample may carry.
pub const PCM_MAX_CHANNELS: usize = 2;

/// A content hash: the sha256 of the source file's bytes, as the compiled
/// instrument spec records it (`sample.sha256`).
pub type ContentHash = [u8; 32];

/// Errors from decoding an uploaded sample.
#[derive(Debug, Clone, PartialEq)]
pub enum PcmCodecError {
    /// The magic prefix did not match.
    BadMagic,
    /// The format version is newer than this build understands.
    UnsupportedVersion(u16),
    /// The input ended before the declared frames were read.
    UnexpectedEof,
    /// The name was not valid UTF-8.
    BadUtf8,
    /// Channels not 1 or 2, a zero frame count or a non-positive rate.
    BadShape,
    /// Bytes left over after the declared frames.
    TrailingBytes,
    /// A sample value was NaN or infinite.
    NonFinite { channel: usize, frame: usize },
}

impl From<WireError> for PcmCodecError {
    fn from(e: WireError) -> Self {
        match e {
            WireError::UnexpectedEof => PcmCodecError::UnexpectedEof,
            WireError::BadUtf8 => PcmCodecError::BadUtf8,
        }
    }
}

/// One registered sample: planar f32 frames at the file's own rate, each
/// channel stored as `PCM_PAD` zeros, the frames, `PCM_PAD` zeros.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplePcm {
    name: String,
    hash: ContentHash,
    sample_rate: f32,
    channels: usize,
    frames: usize,
    data: Vec<f32>,
}

impl SamplePcm {
    /// Build from per-channel frames (1 or 2 channels of equal length).
    /// Returns `None` for any other shape, a non-positive rate or a
    /// non-finite value.
    pub fn new(
        name: impl Into<String>,
        hash: ContentHash,
        sample_rate: f32,
        channels: &[&[f32]],
    ) -> Option<Self> {
        let n = channels.len();
        if n == 0 || n > PCM_MAX_CHANNELS || sample_rate <= 0.0 || !sample_rate.is_finite() {
            return None;
        }
        let frames = channels[0].len();
        if frames == 0 || channels.iter().any(|c| c.len() != frames) {
            return None;
        }
        if channels.iter().any(|c| c.iter().any(|v| !v.is_finite())) {
            return None;
        }
        let stride = frames + 2 * PCM_PAD;
        let mut data = alloc::vec![0.0f32; stride * n];
        for (ch, src) in channels.iter().enumerate() {
            data[ch * stride + PCM_PAD..ch * stride + PCM_PAD + frames].copy_from_slice(src);
        }
        Some(SamplePcm {
            name: name.into(),
            hash,
            sample_rate,
            channels: n,
            frames,
            data,
        })
    }

    /// The name it was uploaded under (may be empty).
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Its content hash.
    pub fn hash(&self) -> &ContentHash {
        &self.hash
    }
    /// The file's sample rate in Hz.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
    /// 1 or 2.
    pub fn channels(&self) -> usize {
        self.channels
    }
    /// Frames per channel (without the padding).
    pub fn frames(&self) -> usize {
        self.frames
    }
    /// Bytes held, padding included: what registering it costs.
    pub fn bytes(&self) -> usize {
        self.data.len() * core::mem::size_of::<f32>()
    }

    /// Channel `ch` (wrapped, so a mono sample feeds both sides) including its
    /// padding: frame `f` is at index `f + PCM_PAD`.
    #[inline]
    pub(crate) fn padded(&self, ch: usize) -> &[f32] {
        let ch = ch % self.channels;
        let stride = self.frames + 2 * PCM_PAD;
        &self.data[ch * stride..(ch + 1) * stride]
    }

    /// Frame `f` of channel `ch`, 0.0 outside `0..frames`.
    pub fn frame(&self, ch: usize, f: isize) -> f32 {
        if f < 0 || f as usize >= self.frames {
            0.0
        } else {
            self.padded(ch)[f as usize + PCM_PAD]
        }
    }

    /// Encode to the upload byte format (`docs/sample-bank-format.md`, Part 1).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.frames * self.channels * 4);
        out.extend_from_slice(PCM_MAGIC);
        put_u16(&mut out, PCM_FORMAT_VERSION);
        put_str(&mut out, &self.name);
        out.extend_from_slice(&self.hash);
        put_f32(&mut out, self.sample_rate);
        put_u16(&mut out, self.channels as u16);
        put_u32(&mut out, self.frames as u32);
        for ch in 0..self.channels {
            for &v in &self.padded(ch)[PCM_PAD..PCM_PAD + self.frames] {
                put_f32(&mut out, v);
            }
        }
        out
    }

    /// Decode the upload byte format. Rejects a newer version, a bad shape,
    /// truncated or trailing bytes, and NaN/infinite values.
    pub fn from_bytes(bytes: &[u8]) -> Result<SamplePcm, PcmCodecError> {
        let mut r = Reader::new(bytes);
        if r.take(PCM_MAGIC.len())? != PCM_MAGIC {
            return Err(PcmCodecError::BadMagic);
        }
        let version = r.u16()?;
        if version > PCM_FORMAT_VERSION {
            return Err(PcmCodecError::UnsupportedVersion(version));
        }
        let name = r.string()?;
        let mut hash = [0u8; 32];
        hash.copy_from_slice(r.take(32)?);
        let sample_rate = r.f32()?;
        let channels = r.u16()? as usize;
        let frames = r.usize32()?;
        if channels == 0
            || channels > PCM_MAX_CHANNELS
            || frames == 0
            || sample_rate <= 0.0
            || !sample_rate.is_finite()
        {
            return Err(PcmCodecError::BadShape);
        }
        let payload = r.take(
            frames
                .checked_mul(channels * 4)
                .ok_or(PcmCodecError::BadShape)?,
        )?;
        if r.pos != bytes.len() {
            return Err(PcmCodecError::TrailingBytes);
        }
        let stride = frames + 2 * PCM_PAD;
        let mut data = alloc::vec![0.0f32; stride * channels];
        for ch in 0..channels {
            let base = ch * frames * 4;
            for f in 0..frames {
                let o = base + f * 4;
                let v = f32::from_le_bytes([
                    payload[o],
                    payload[o + 1],
                    payload[o + 2],
                    payload[o + 3],
                ]);
                if !v.is_finite() {
                    return Err(PcmCodecError::NonFinite {
                        channel: ch,
                        frame: f,
                    });
                }
                data[ch * stride + PCM_PAD + f] = v;
            }
        }
        Ok(SamplePcm {
            name,
            hash,
            sample_rate,
            channels,
            frames,
            data,
        })
    }
}

/// Identifier of a registered sample. `0` is never issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PcmId(pub u32);

/// The session's registered samples, by id, content hash and name.
///
/// A sample is stored once per content hash: registering bytes whose hash is
/// already present returns the existing id and keeps the first upload (the
/// hash names the file, so two uploads with one hash are the same audio).
pub struct SamplePcmBank {
    samples: BTreeMap<PcmId, Arc<SamplePcm>>,
    by_hash: BTreeMap<ContentHash, PcmId>,
    by_name: BTreeMap<String, PcmId>,
    next_id: u32,
    bytes: usize,
}

impl SamplePcmBank {
    /// An empty bank; the first id issued is 1.
    pub fn new() -> Self {
        SamplePcmBank {
            samples: BTreeMap::new(),
            by_hash: BTreeMap::new(),
            by_name: BTreeMap::new(),
            next_id: 1,
            bytes: 0,
        }
    }

    /// Store `pcm`, or return the id already holding its hash.
    pub fn register(&mut self, pcm: SamplePcm) -> PcmId {
        if let Some(&id) = self.by_hash.get(&pcm.hash) {
            return id;
        }
        let id = PcmId(self.next_id);
        self.next_id += 1;
        self.bytes += pcm.bytes();
        self.by_hash.insert(pcm.hash, id);
        if !pcm.name.is_empty() {
            self.by_name.insert(pcm.name.clone(), id);
        }
        self.samples.insert(id, Arc::new(pcm));
        id
    }

    /// Drop the bank's reference. A program already bound to the sample
    /// keeps its own reference until the program is freed.
    pub fn free(&mut self, id: PcmId) {
        if let Some(pcm) = self.samples.remove(&id) {
            self.bytes -= pcm.bytes();
            self.by_hash.remove(&pcm.hash);
            if self.by_name.get(&pcm.name) == Some(&id) {
                self.by_name.remove(&pcm.name);
            }
        }
    }

    /// Look a sample up by id.
    pub fn get(&self, id: PcmId) -> Option<&Arc<SamplePcm>> {
        self.samples.get(&id)
    }
    /// The id holding content hash `hash`.
    pub fn id_for_hash(&self, hash: &ContentHash) -> Option<PcmId> {
        self.by_hash.get(hash).copied()
    }
    /// The id registered under `name`.
    pub fn id_for_name(&self, name: &str) -> Option<PcmId> {
        self.by_name.get(name).copied()
    }
    /// The sample holding content hash `hash`.
    pub fn get_by_hash(&self, hash: &ContentHash) -> Option<&Arc<SamplePcm>> {
        self.id_for_hash(hash).and_then(|id| self.samples.get(&id))
    }
    /// Bytes held by every registered sample, padding included.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Number of registered samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl Default for SamplePcmBank {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse 64 hex digits into a content hash (`None` on anything else).
pub fn hash_from_hex(hex: &str) -> Option<ContentHash> {
    let b = hex.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let nib = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = (nib(b[2 * i])? << 4) | nib(b[2 * i + 1])?;
    }
    Some(out)
}

/// The 64-digit lowercase hex form of a content hash.
pub fn hash_to_hex(hash: &ContentHash) -> String {
    const D: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for &b in hash {
        s.push(D[(b >> 4) as usize] as char);
        s.push(D[(b & 15) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(n: u8) -> ContentHash {
        [n; 32]
    }

    #[test]
    fn round_trips_mono_and_stereo() {
        let l = [0.5, -0.25, 0.125];
        let r = [0.0, 1.0, -1.0];
        for chans in [&[&l[..]][..], &[&l[..], &r[..]][..]] {
            let pcm = SamplePcm::new("x", hash(7), 48_000.0, chans).unwrap();
            let back = SamplePcm::from_bytes(&pcm.to_bytes()).unwrap();
            assert_eq!(back, pcm);
            assert_eq!(back.frame(0, 1), -0.25);
            assert_eq!(back.frame(0, -1), 0.0);
            assert_eq!(back.frame(0, 3), 0.0);
        }
    }

    #[test]
    fn rejects_bad_shapes_and_values() {
        let pcm = SamplePcm::new("", hash(1), 44_100.0, &[&[0.1, 0.2][..]]).unwrap();
        let good = pcm.to_bytes();
        assert_eq!(
            SamplePcm::from_bytes(&good[..good.len() - 1]),
            Err(PcmCodecError::UnexpectedEof)
        );
        let mut trailing = good.clone();
        trailing.push(0);
        assert_eq!(
            SamplePcm::from_bytes(&trailing),
            Err(PcmCodecError::TrailingBytes)
        );
        let mut nan = good.clone();
        let n = nan.len();
        nan[n - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(
            SamplePcm::from_bytes(&nan),
            Err(PcmCodecError::NonFinite {
                channel: 0,
                frame: 1
            })
        );
        let mut bad = good.clone();
        bad[0] = b'X';
        assert_eq!(SamplePcm::from_bytes(&bad), Err(PcmCodecError::BadMagic));
        assert!(SamplePcm::new("", hash(1), 44_100.0, &[&[0.1][..]; 3]).is_none());
        assert!(SamplePcm::new("", hash(1), 0.0, &[&[0.1][..]]).is_none());
    }

    #[test]
    fn bank_dedups_by_hash_and_accounts_bytes() {
        let mut bank = SamplePcmBank::new();
        let a = SamplePcm::new("a", hash(1), 44_100.0, &[&[0.0; 100][..]]).unwrap();
        let bytes = a.bytes();
        assert_eq!(bytes, (100 + 2 * PCM_PAD) * 4);
        let id = bank.register(a.clone());
        assert_eq!(id, PcmId(1));
        assert_eq!(bank.register(a), id, "same hash -> same id");
        assert_eq!(bank.bytes(), bytes);
        assert_eq!(bank.id_for_hash(&hash(1)), Some(id));
        assert_eq!(bank.id_for_name("a"), Some(id));
        bank.free(id);
        assert_eq!(bank.bytes(), 0);
        assert!(bank.is_empty());
        assert_eq!(bank.id_for_hash(&hash(1)), None);
    }

    #[test]
    fn hex_round_trip() {
        let h = hash_from_hex("c97a003184fdcb153d6b6c0bae8d82a90592a1b04b06bd7caaeda8b9776d5284")
            .unwrap();
        assert_eq!(
            hash_to_hex(&h),
            "c97a003184fdcb153d6b6c0bae8d82a90592a1b04b06bd7caaeda8b9776d5284"
        );
        assert!(hash_from_hex("zz").is_none());
    }
}
