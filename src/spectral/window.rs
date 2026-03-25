//! Standard window functions for spectral analysis/synthesis.

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::PI;

/// Window function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Hann (raised cosine) — good general purpose, COLA-compliant at 50%/75% overlap.
    Hann,
    /// Hamming — slightly less sidelobe leakage than Hann.
    Hamming,
    /// Blackman — lower sidelobes, wider main lobe.
    Blackman,
    /// Blackman-Harris — very low sidelobes (4-term).
    BlackmanHarris,
}

/// Generate a window of the given type and size.
pub fn make_window(window_type: WindowType, size: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; size];
    if size == 0 {
        return w;
    }
    let n = size as f32;
    for i in 0..size {
        let x = i as f32;
        w[i] = match window_type {
            WindowType::Hann => 0.5 * (1.0 - (2.0 * PI * x / n).cos()),
            WindowType::Hamming => 0.54 - 0.46 * (2.0 * PI * x / n).cos(),
            WindowType::Blackman => {
                0.42 - 0.5 * (2.0 * PI * x / n).cos() + 0.08 * (4.0 * PI * x / n).cos()
            }
            WindowType::BlackmanHarris => {
                0.35875 - 0.48829 * (2.0 * PI * x / n).cos()
                    + 0.14128 * (4.0 * PI * x / n).cos()
                    - 0.01168 * (6.0 * PI * x / n).cos()
            }
        };
    }
    w
}

/// Compute the COLA (Constant Overlap-Add) normalization factor for a given
/// window type, FFT size, and hop size.
///
/// When using overlap-add synthesis, dividing the output by this factor
/// ensures unity gain for signals that pass through STFT→ISTFT unchanged.
pub fn cola_norm(window_type: WindowType, fft_size: usize, hop_size: usize) -> f32 {
    let w = make_window(window_type, fft_size);
    // Sum the squared window values at each hop offset.
    // For COLA, the sum of overlapping windows should be constant.
    // We compute the average overlap sum at position 0.
    let num_overlaps = (fft_size + hop_size - 1) / hop_size;
    let mut sum = 0.0f32;
    for k in 0..num_overlaps {
        let offset = k * hop_size;
        if offset < fft_size {
            let val = w[offset];
            sum += val * val;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_symmetry() {
        // Periodic window: w[i] == w[N-i] for i in 1..N-1
        // (w[0] == 0 and there's no w[N], so check inner pairs).
        let n = 64;
        let w = make_window(WindowType::Hann, n);
        for i in 1..n / 2 {
            let diff = (w[i] - w[n - i]).abs();
            assert!(diff < 1e-6, "Hann window not symmetric at {i}");
        }
    }

    #[test]
    fn hann_endpoints() {
        let w = make_window(WindowType::Hann, 1024);
        // Hann window starts and ends near zero (periodic form).
        assert!(w[0].abs() < 1e-6);
        // Middle should be close to 1.
        assert!((w[512] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn cola_hann_50_percent() {
        // Hann window with 50% overlap should have COLA norm ~1.0.
        let norm = cola_norm(WindowType::Hann, 1024, 512);
        assert!(
            (norm - 1.0).abs() < 0.1,
            "COLA norm for Hann 50% overlap: {norm}"
        );
    }

    #[test]
    fn window_non_negative() {
        for wt in [
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
            WindowType::BlackmanHarris,
        ] {
            let w = make_window(wt, 256);
            for (i, &val) in w.iter().enumerate() {
                assert!(val >= -1e-6, "{wt:?} window negative at {i}: {val}");
            }
        }
    }
}
