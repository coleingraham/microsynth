//! Short-Time Fourier Transform processor with overlap-add synthesis.
//!
//! Bridges the gap between small audio block sizes (64-128 samples) and
//! large FFT windows (512-4096 samples) by internally accumulating samples
//! in ring buffers.

use super::complex::Complex;
use super::fft::{fft, ifft};
use super::window::{make_window, WindowType};
use alloc::vec;
use alloc::vec::Vec;

/// STFT processor that handles windowing, FFT, and overlap-add synthesis.
///
/// Usage pattern (inside a UGen's `process` method):
/// ```ignore
/// for i in 0..block_size {
///     if stft.push_sample(input[i]) {
///         let spectrum = stft.analyze();
///         // ... modify spectrum in-place ...
///         stft.synthesize();
///     }
///     output[i] = stft.pop_sample();
/// }
/// ```
pub struct StftProcessor {
    fft_size: usize,
    hop_size: usize,
    /// Analysis window.
    window: Vec<f32>,
    /// Circular input accumulation buffer (size = fft_size).
    input_ring: Vec<f32>,
    input_write_pos: usize,
    /// Output overlap-add buffer (size = 2*fft_size for safe overlap-add).
    output_buf: Vec<f32>,
    /// Position in output_buf where the next frame's OLA output starts.
    output_write_pos: usize,
    /// Position in output_buf where we read the next output sample.
    output_read_pos: usize,
    /// FFT workspace (reused across frames).
    fft_buf: Vec<Complex>,
    /// Samples remaining until next hop boundary.
    samples_until_hop: usize,
    /// Total input samples pushed (for tracking absolute position).
    total_input_samples: usize,
    /// COLA normalization factor.
    cola_norm: f32,
}

impl StftProcessor {
    /// Create a new STFT processor.
    ///
    /// - `fft_size`: FFT window size (must be a power of 2).
    /// - `hop_size`: hop between consecutive frames (typically fft_size/4).
    /// - `window_type`: analysis/synthesis window function.
    pub fn new(fft_size: usize, hop_size: usize, window_type: WindowType) -> Self {
        assert!(
            super::fft::is_power_of_two(fft_size),
            "FFT size must be a power of 2"
        );
        assert!(hop_size > 0 && hop_size <= fft_size);

        let window = make_window(window_type, fft_size);

        // Compute COLA norm: sum of squared window at overlapping positions.
        let num_overlaps = (fft_size + hop_size - 1) / hop_size;
        let mut cola = 0.0f32;
        for k in 0..num_overlaps {
            let offset = k * hop_size;
            if offset < fft_size {
                let w = window[offset];
                cola += w * w;
            }
        }
        if cola < 1e-10 {
            cola = 1.0;
        }

        // Output buffer: 2*fft_size gives enough room for overlap-add.
        let out_buf_size = 2 * fft_size;

        Self {
            fft_size,
            hop_size,
            window,
            input_ring: vec![0.0; fft_size],
            input_write_pos: 0,
            output_buf: vec![0.0; out_buf_size],
            output_write_pos: 0,
            output_read_pos: 0,
            fft_buf: vec![Complex::ZERO; fft_size],
            samples_until_hop: fft_size, // Wait for one full window before first frame.
            total_input_samples: 0,
            cola_norm: cola,
        }
    }

    /// Push one input sample. Returns `true` when a hop boundary is reached
    /// and a new FFT frame is ready for analysis.
    #[inline]
    pub fn push_sample(&mut self, sample: f32) -> bool {
        self.input_ring[self.input_write_pos] = sample;
        self.input_write_pos = (self.input_write_pos + 1) % self.fft_size;
        self.total_input_samples += 1;
        self.samples_until_hop -= 1;
        if self.samples_until_hop == 0 {
            self.samples_until_hop = self.hop_size;
            true
        } else {
            false
        }
    }

    /// Perform windowed FFT on the current input frame.
    ///
    /// Returns a mutable slice of the frequency-domain bins. Modify these
    /// in-place before calling [`synthesize`].
    ///
    /// Call only when [`push_sample`] returns `true`.
    pub fn analyze(&mut self) -> &mut [Complex] {
        // The most recent fft_size samples in the ring end at input_write_pos.
        // Read them in order, applying the analysis window.
        for i in 0..self.fft_size {
            let ring_idx = (self.input_write_pos + i) % self.fft_size;
            self.fft_buf[i] = Complex::new(self.input_ring[ring_idx] * self.window[i], 0.0);
        }
        fft(&mut self.fft_buf);
        &mut self.fft_buf
    }

    /// Perform IFFT and overlap-add the result into the output buffer.
    ///
    /// Call after modifying the spectrum returned by [`analyze`].
    pub fn synthesize(&mut self) {
        ifft(&mut self.fft_buf);

        let inv_cola = 1.0 / self.cola_norm;
        let out_len = self.output_buf.len();

        // Overlap-add: write fft_size samples starting at output_write_pos.
        for i in 0..self.fft_size {
            let idx = (self.output_write_pos + i) % out_len;
            self.output_buf[idx] += self.fft_buf[i].re * self.window[i] * inv_cola;
        }

        // Advance write position by hop_size for next frame.
        self.output_write_pos = (self.output_write_pos + self.hop_size) % out_len;
    }

    /// Pop one output sample, advancing the read position.
    ///
    /// During the initial latency period (first fft_size samples), returns 0.0.
    #[inline]
    pub fn pop_sample(&mut self) -> f32 {
        if self.total_input_samples <= self.fft_size {
            return 0.0;
        }
        let out_len = self.output_buf.len();
        let sample = self.output_buf[self.output_read_pos];
        self.output_buf[self.output_read_pos] = 0.0; // Clear for next overlap-add cycle.
        self.output_read_pos = (self.output_read_pos + 1) % out_len;
        sample
    }

    /// The FFT size this processor was configured with.
    #[inline]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// The hop size this processor was configured with.
    #[inline]
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Reset all internal state to zero.
    pub fn reset(&mut self) {
        self.input_ring.fill(0.0);
        self.input_write_pos = 0;
        self.output_buf.fill(0.0);
        self.output_write_pos = 0;
        self.output_read_pos = 0;
        self.fft_buf.fill(Complex::ZERO);
        self.samples_until_hop = self.fft_size;
        self.total_input_samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// STFT→ISTFT round-trip with no spectral modification should recover
    /// the original signal (within overlap-add normalization tolerance).
    #[test]
    fn roundtrip_identity() {
        let fft_size = 256;
        let hop_size = 64;
        let mut stft = StftProcessor::new(fft_size, hop_size, WindowType::Hann);

        // Generate a test signal: sine wave.
        let num_samples = 4096;
        let input: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * core::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let mut output = vec![0.0f32; num_samples];

        for i in 0..num_samples {
            if stft.push_sample(input[i]) {
                let _spectrum = stft.analyze();
                // No modification — identity pass-through.
                stft.synthesize();
            }
            output[i] = stft.pop_sample();
        }

        // After the initial latency (fft_size samples) and settling (another fft_size),
        // the output should match the input with some time offset.
        // Find the best alignment by cross-correlating a short segment.
        let check_start = fft_size * 2;
        let check_end = num_samples - fft_size;
        if check_end <= check_start {
            return;
        }

        // Compute RMS error over the steady-state region.
        // The output should approximately match the input (may have a small time offset).
        let mut min_rms = f32::MAX;
        // Try a few offsets around 0 to find best alignment.
        for offset in 0..=hop_size {
            let mut sum_sq = 0.0f32;
            let mut count = 0;
            for i in check_start..check_end {
                let inp_idx = i.wrapping_sub(offset);
                if inp_idx < num_samples {
                    let err = output[i] - input[inp_idx];
                    sum_sq += err * err;
                    count += 1;
                }
            }
            if count > 0 {
                let rms = (sum_sq / count as f32).sqrt();
                if rms < min_rms {
                    min_rms = rms;
                }
            }
        }

        assert!(
            min_rms < 0.15,
            "Round-trip RMS error too large: {min_rms}"
        );
    }

    #[test]
    fn priming_returns_zero() {
        let fft_size = 512;
        let hop_size = 128;
        let mut stft = StftProcessor::new(fft_size, hop_size, WindowType::Hann);

        // Before fft_size samples, pop_sample should return 0.
        for _ in 0..fft_size {
            stft.push_sample(1.0);
            let out = stft.pop_sample();
            assert_eq!(out, 0.0, "Should return 0 during priming period");
        }
    }
}
