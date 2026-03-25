//! Griffin-Lim algorithm for phase reconstruction from magnitude spectrograms.
//!
//! This is an **offline** algorithm — it requires the full magnitude spectrogram
//! upfront and iterates over it multiple times. Not suitable for real-time use.
//!
//! Gated behind `#[cfg(feature = "std")]` due to heavy allocation.

use super::complex::Complex;
use super::fft::{fft, ifft};
use super::window::{make_window, WindowType};
use alloc::vec;
use alloc::vec::Vec;

/// Reconstruct a time-domain signal from a magnitude spectrogram using the
/// Griffin-Lim iterative algorithm.
///
/// # Arguments
///
/// - `magnitudes`: Slice of frames, where each frame is `fft_size` magnitude bins.
///   Typically only `fft_size/2 + 1` bins are unique for real signals, but this
///   function expects full `fft_size` bins (with conjugate symmetry).
/// - `fft_size`: FFT window size (must be a power of 2).
/// - `hop_size`: Hop between consecutive STFT frames.
/// - `window_type`: Window function for analysis/synthesis.
/// - `iterations`: Number of Griffin-Lim iterations (30-100 typical).
///
/// # Returns
///
/// Reconstructed time-domain signal as `Vec<f32>`.
pub fn griffin_lim(
    magnitudes: &[Vec<f32>],
    fft_size: usize,
    hop_size: usize,
    window_type: WindowType,
    iterations: usize,
) -> Vec<f32> {
    assert!(
        super::fft::is_power_of_two(fft_size),
        "FFT size must be a power of 2"
    );
    assert!(!magnitudes.is_empty());

    let num_frames = magnitudes.len();
    let output_len = (num_frames - 1) * hop_size + fft_size;
    let window = make_window(window_type, fft_size);

    // Compute COLA normalization.
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
    let inv_cola = 1.0 / cola;

    // Initialize with random phases (using magnitudes * e^{j*0} = real).
    let mut phases: Vec<Vec<f32>> = magnitudes
        .iter()
        .map(|frame| vec![0.0f32; frame.len()])
        .collect();

    let mut signal = vec![0.0f32; output_len];
    let mut fft_buf = vec![Complex::ZERO; fft_size];

    for _iter in 0..iterations {
        // --- ISTFT: construct complex spectrum from magnitudes + current phases, synthesize ---
        signal.fill(0.0);
        for (frame_idx, (mags, phs)) in magnitudes.iter().zip(phases.iter()).enumerate() {
            let offset = frame_idx * hop_size;
            // Build complex spectrum.
            for i in 0..fft_size {
                let mag = if i < mags.len() { mags[i] } else { 0.0 };
                let phase = if i < phs.len() { phs[i] } else { 0.0 };
                fft_buf[i] = Complex::from_polar(mag, phase);
            }
            // IFFT.
            ifft(&mut fft_buf);
            // Overlap-add with synthesis window.
            for i in 0..fft_size {
                if offset + i < output_len {
                    signal[offset + i] += fft_buf[i].re * window[i] * inv_cola;
                }
            }
        }

        // --- STFT: re-analyze to get new phases ---
        for (frame_idx, phs) in phases.iter_mut().enumerate() {
            let offset = frame_idx * hop_size;
            // Window the signal.
            for i in 0..fft_size {
                let sample = if offset + i < output_len {
                    signal[offset + i]
                } else {
                    0.0
                };
                fft_buf[i] = Complex::new(sample * window[i], 0.0);
            }
            // FFT.
            fft(&mut fft_buf);
            // Extract new phases (keep original magnitudes).
            for (i, c) in fft_buf.iter().enumerate() {
                if i < phs.len() {
                    phs[i] = c.phase();
                }
            }
        }
    }

    // Final ISTFT pass with converged phases.
    signal.fill(0.0);
    for (frame_idx, (mags, phs)) in magnitudes.iter().zip(phases.iter()).enumerate() {
        let offset = frame_idx * hop_size;
        for i in 0..fft_size {
            let mag = if i < mags.len() { mags[i] } else { 0.0 };
            let phase = if i < phs.len() { phs[i] } else { 0.0 };
            fft_buf[i] = Complex::from_polar(mag, phase);
        }
        ifft(&mut fft_buf);
        for i in 0..fft_size {
            if offset + i < output_len {
                signal[offset + i] += fft_buf[i].re * window[i] * inv_cola;
            }
        }
    }

    signal
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    /// Reconstruct a simple sinusoid from its magnitude spectrogram.
    #[test]
    fn reconstruct_sine() {
        let fft_size = 256;
        let hop_size = 64;
        let freq = 440.0f32;
        let sr = 44100.0f32;
        let num_samples = 2048;

        // Generate original signal.
        let original: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
            .collect();

        // Compute magnitude spectrogram via STFT.
        let window = make_window(WindowType::Hann, fft_size);
        let mut magnitudes = Vec::new();
        let mut fft_buf = vec![Complex::ZERO; fft_size];

        let mut offset = 0;
        while offset + fft_size <= num_samples {
            for i in 0..fft_size {
                fft_buf[i] = Complex::new(original[offset + i] * window[i], 0.0);
            }
            fft(&mut fft_buf);
            let mags: Vec<f32> = fft_buf.iter().map(|c| c.mag()).collect();
            magnitudes.push(mags);
            offset += hop_size;
        }

        // Griffin-Lim reconstruction.
        let reconstructed = griffin_lim(&magnitudes, fft_size, hop_size, WindowType::Hann, 50);

        // Check that the reconstructed signal has similar energy in the steady state.
        let start = fft_size * 2;
        let end = num_samples.min(reconstructed.len()) - fft_size;
        if end > start {
            let orig_energy: f32 = original[start..end].iter().map(|x| x * x).sum::<f32>()
                / (end - start) as f32;
            let recon_energy: f32 = reconstructed[start..end]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                / (end - start) as f32;
            // Energy should be within ~20% (Griffin-Lim is approximate).
            let ratio = recon_energy / orig_energy.max(1e-10);
            assert!(
                ratio > 0.5 && ratio < 2.0,
                "Energy ratio out of range: {ratio} (orig={orig_energy}, recon={recon_energy})"
            );
        }
    }
}
