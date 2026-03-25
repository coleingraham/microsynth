//! In-place radix-2 Cooley-Tukey FFT and IFFT.
//!
//! Supports power-of-2 sizes only. Twiddle factors are computed on the fly
//! using LLVM-intrinsic `sin`/`cos` (no lookup table).
//!
//! `no_std` compatible.

use super::complex::Complex;
use core::f32::consts::PI;

/// Returns true if `n` is a power of two and > 0.
#[inline]
pub fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// In-place forward FFT (decimation-in-time).
///
/// `buf.len()` must be a power of 2.
///
/// # Panics
/// Panics if `buf.len()` is not a power of 2.
pub fn fft(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(is_power_of_two(n), "FFT size must be a power of 2");
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    bit_reverse_permute(buf);

    // Butterfly stages.
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle_step = -2.0 * PI / size as f32;
        for k in 0..half {
            let angle = angle_step * k as f32;
            let twiddle = Complex::new(angle.cos(), angle.sin());
            let mut j = k;
            while j < n {
                let t = buf[j + half] * twiddle;
                buf[j + half] = buf[j] - t;
                buf[j] = buf[j] + t;
                j += size;
            }
        }
        size <<= 1;
    }
}

/// In-place inverse FFT.
///
/// Conjugate → FFT → conjugate → scale by 1/N.
///
/// # Panics
/// Panics if `buf.len()` is not a power of 2.
pub fn ifft(buf: &mut [Complex]) {
    let n = buf.len();
    // Conjugate.
    for c in buf.iter_mut() {
        *c = c.conj();
    }
    // Forward FFT.
    fft(buf);
    // Conjugate and scale.
    let scale = 1.0 / n as f32;
    for c in buf.iter_mut() {
        *c = c.conj().scale(scale);
    }
}

/// Bit-reversal permutation for in-place FFT.
fn bit_reverse_permute(buf: &mut [Complex]) {
    let n = buf.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn roundtrip_identity() {
        // FFT then IFFT should recover the original signal.
        let original: Vec<Complex> = (0..64)
            .map(|i| Complex::new((i as f32 * 0.1).sin(), 0.0))
            .collect();
        let mut buf = original.clone();
        fft(&mut buf);
        ifft(&mut buf);
        for (i, (a, b)) in original.iter().zip(buf.iter()).enumerate() {
            assert!(
                approx_eq(a.re, b.re, EPS) && approx_eq(a.im, b.im, EPS),
                "Mismatch at {i}: expected ({}, {}), got ({}, {})",
                a.re,
                a.im,
                b.re,
                b.im
            );
        }
    }

    #[test]
    fn dc_signal() {
        // All ones → energy in bin 0 only.
        let n = 16;
        let mut buf = vec![Complex::new(1.0, 0.0); n];
        fft(&mut buf);
        assert!(approx_eq(buf[0].re, n as f32, EPS));
        assert!(approx_eq(buf[0].im, 0.0, EPS));
        for k in 1..n {
            assert!(
                buf[k].mag() < EPS,
                "Bin {k} should be zero, got {}",
                buf[k].mag()
            );
        }
    }

    #[test]
    fn single_frequency() {
        // Cosine at bin frequency k should have energy in bins k and N-k.
        let n = 64;
        let k = 5usize;
        let mut buf: Vec<Complex> = (0..n)
            .map(|i| {
                let phase = 2.0 * PI * k as f32 * i as f32 / n as f32;
                Complex::new(phase.cos(), 0.0)
            })
            .collect();
        fft(&mut buf);
        // Bins k and N-k should have magnitude N/2.
        let expected = n as f32 / 2.0;
        assert!(
            approx_eq(buf[k].mag(), expected, 0.1),
            "Bin {k}: expected {expected}, got {}",
            buf[k].mag()
        );
        assert!(
            approx_eq(buf[n - k].mag(), expected, 0.1),
            "Bin {}: expected {expected}, got {}",
            n - k,
            buf[n - k].mag()
        );
        // Other bins should be near zero.
        for i in 0..n {
            if i == k || i == n - k {
                continue;
            }
            assert!(
                buf[i].mag() < 1.0,
                "Bin {i} should be near zero, got {}",
                buf[i].mag()
            );
        }
    }

    #[test]
    fn parseval_theorem() {
        // Time-domain energy should equal frequency-domain energy / N.
        let n = 128;
        let original: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i as f32 * 0.3).sin() + 0.5, 0.0))
            .collect();
        let time_energy: f32 = original.iter().map(|c| c.norm_sq()).sum();
        let mut buf = original.clone();
        fft(&mut buf);
        let freq_energy: f32 = buf.iter().map(|c| c.norm_sq()).sum();
        let freq_energy_normalized = freq_energy / n as f32;
        assert!(
            approx_eq(time_energy, freq_energy_normalized, 0.5),
            "Parseval: time={time_energy}, freq/N={freq_energy_normalized}"
        );
    }

    #[test]
    #[should_panic]
    fn non_power_of_two_panics() {
        let mut buf = vec![Complex::ZERO; 13];
        fft(&mut buf);
    }

    #[test]
    fn size_one() {
        let mut buf = vec![Complex::new(42.0, 0.0)];
        fft(&mut buf);
        assert!(approx_eq(buf[0].re, 42.0, EPS));
    }
}
