//! Shared binary wire-format primitives: little-endian fixed-width integers,
//! IEEE-754 floats, and length-prefixed UTF-8 strings, plus a bounds-checked
//! cursor for reading them back.
//!
//! This is the single home for these primitives across the crate. Two
//! independent wire formats build on them — [`crate::coeff_table`]'s
//! dictionary-table codec (unconditional) and `crate::ir::serialize`'s IR
//! codec (`#[cfg(feature = "ir")]`) — and each used to carry its own private
//! `Writer`/`Reader` pair with identical bodies. This module is never
//! feature-gated, so both call sites can reach it regardless of which
//! features are enabled; `ir::serialize`'s own doc used to claim its
//! `pub(super) Reader` existed so there would never be "a second hand-rolled
//! reader" — this module is now where that claim actually holds, crate-wide.
//!
//! Each format keeps its own format-specific error enum (bad magic,
//! unsupported version, malformed entry, bad tag, ...) — only the two
//! failure modes intrinsic to these primitives themselves (truncated input,
//! invalid UTF-8) are represented here, as [`WireError`]. Each format
//! converts a `WireError` into its own error type via `From<WireError>` at
//! the call site, so `?` composes without the primitives knowing about any
//! downstream error type.

use alloc::string::String;
use alloc::vec::Vec;

/// A failure intrinsic to decoding one of this module's primitives: the
/// input ended before a full record could be read, or a length-prefixed
/// string wasn't valid UTF-8. Format-specific decode errors (bad magic,
/// unknown tag, malformed entry, ...) are not representable here — each
/// format wraps this in its own error enum via `From<WireError>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WireError {
    UnexpectedEof,
    BadUtf8,
}

pub(crate) fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
pub(crate) fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
pub(crate) fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}
pub(crate) fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

/// A cursor reading little-endian records with bounds checks.
pub(crate) struct Reader<'a> {
    pub(crate) buf: &'a [u8],
    pub(crate) pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        let end = self.pos.checked_add(n).ok_or(WireError::UnexpectedEof)?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or(WireError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }
    // Only `ir::serialize`'s node-tag decoding reads a lone byte today;
    // `coeff_table`'s format has no single-byte field. Gated rather than
    // `#[allow(dead_code)]`'d so it stays truthful about which formats use
    // it, and so a real dead-primitive would still be caught.
    #[cfg(feature = "ir")]
    pub(crate) fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }
    pub(crate) fn u16(&mut self) -> Result<u16, WireError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(crate) fn u32(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn usize32(&mut self) -> Result<usize, WireError> {
        Ok(self.u32()? as usize)
    }
    pub(crate) fn f32(&mut self) -> Result<f32, WireError> {
        Ok(f32::from_bits(u32::from_le_bytes(
            self.take(4)?.try_into().unwrap(),
        )))
    }
    pub(crate) fn string(&mut self) -> Result<String, WireError> {
        let len = self.usize32()?;
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes)
            .map(|s| s.into())
            .map_err(|_| WireError::BadUtf8)
    }
}
