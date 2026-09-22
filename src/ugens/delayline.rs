//! A shared interpolating delay line.
//!
//! Almost every time-domain effect in this crate — comb filters, echoes,
//! chorus, flangers, ping-pong delays, reverb components — is built from the
//! same primitive: a circular `Vec<f32>` plus a write cursor, read at some
//! fractional delay and written once per sample. [`DelayLine`] factors that
//! primitive out so each UGen only expresses what makes it distinct (how it
//! computes the delay time, and what it feeds back).
//!
//! The read and write halves are deliberately separate methods, because the
//! order matters and differs between users:
//!
//! - **Read then write** (comb/feedback topologies): the value written depends
//!   on the delayed value just read.
//! - **Write then read** (plain delay/chorus taps): the sample written at the
//!   current cursor is itself readable at a delay of zero.
//!
//! Use [`write_and_advance`](DelayLine::write_and_advance) for the former, and
//! [`write`](DelayLine::write) + [`advance`](DelayLine::advance) — with reads in
//! between — for the latter.

use alloc::vec::Vec;

/// A circular delay line with fractional (linearly interpolated) reads.
pub(crate) struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl DelayLine {
    /// Create an empty delay line. Call [`resize`](DelayLine::resize) in the
    /// UGen's `init` once the sample rate is known.
    pub(crate) fn new() -> Self {
        DelayLine {
            buffer: Vec::new(),
            write_pos: 0,
        }
    }

    /// Create a delay line with a fixed length (at least one sample).
    pub(crate) fn with_len(len: usize) -> Self {
        DelayLine {
            buffer: alloc::vec![0.0; len.max(1)],
            write_pos: 0,
        }
    }

    /// Allocate the backing buffer and rewind. Intended for `UGen::init`.
    pub(crate) fn resize(&mut self, len: usize) {
        self.buffer.resize(len, 0.0);
        self.write_pos = 0;
    }

    /// Zero the buffer and rewind. Intended for `UGen::reset`.
    pub(crate) fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }

    /// Number of samples in the delay line.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the delay line has no backing buffer yet.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Read at an integer delay, in samples, back from the write cursor.
    ///
    /// `delay` must not exceed [`len`](DelayLine::len).
    #[inline]
    pub(crate) fn read(&self, delay: usize) -> f32 {
        let len = self.buffer.len();
        self.buffer[(self.write_pos + len - delay) % len]
    }

    /// Read at a fractional delay, in samples, back from the write cursor,
    /// linearly interpolating between the two neighbouring samples.
    ///
    /// `delay_samples` must be in `0..=len - 1`; callers clamp it to their own
    /// maximum delay first.
    #[inline]
    pub(crate) fn read_interp(&self, delay_samples: f32) -> f32 {
        let len = self.buffer.len();
        let delay_int = delay_samples as usize;
        let frac = delay_samples - delay_int as f32;

        let a = self.buffer[(self.write_pos + len - delay_int) % len];
        let b = self.buffer[(self.write_pos + len - delay_int - 1) % len];
        a + frac * (b - a)
    }

    /// Read at a fractional delay, in samples, back from the write cursor,
    /// with 4-point, 3rd-order Hermite interpolation.
    ///
    /// `delay_samples` must be in `1..=len - 3`; callers clamp it to their own
    /// range first.
    #[inline]
    pub(crate) fn read_hermite(&self, delay_samples: f32) -> f32 {
        let len = self.buffer.len();
        let delay_int = delay_samples as usize;
        let frac = delay_samples - delay_int as f32;
        let at = |delay: usize| self.buffer[(self.write_pos + len - delay) % len];

        let newer = at(delay_int - 1);
        let x0 = at(delay_int);
        let x1 = at(delay_int + 1);
        let older = at(delay_int + 2);
        let c1 = 0.5 * (x1 - newer);
        let c2 = newer - 2.5 * x0 + 2.0 * x1 - 0.5 * older;
        let c3 = 0.5 * (older - newer) + 1.5 * (x0 - x1);
        ((c3 * frac + c2) * frac + c1) * frac + x0
    }

    /// Write a sample at the cursor without advancing.
    ///
    /// Pair with [`advance`](DelayLine::advance) when reads must see the sample
    /// just written (i.e. a delay of zero).
    #[inline]
    pub(crate) fn write(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
    }

    /// Advance the write cursor by one sample, wrapping.
    #[inline]
    pub(crate) fn advance(&mut self) {
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
    }

    /// Write a sample at the cursor and advance. The common case.
    #[inline]
    pub(crate) fn write_and_advance(&mut self, sample: f32) {
        self.write(sample);
        self.advance();
    }
}
