//! Resampling kernels: read a sample at a fractional frame position.
//!
//! Three kernels, measured against each other in `tests/sample_voice_resampler.rs`
//! (aliasing and pitch error at -12, -7, 0, +7 and +12 semitones on a
//! synthetic sample and on a catalog sample; the table is in the
//! `sampleVoice` doc and the sampler report):
//!
//! - [`Interp::Linear`]: two points.
//! - [`Interp::Hermite`]: four-point cubic Hermite (Catmull-Rom), the kernel
//!   sfizz uses at its default quality.
//! - [`Interp::Sinc`]: a Kaiser-windowed sinc whose cutoff follows the
//!   playback step: `fc = SINC_CUTOFF * min(1, 1 / step)`, so a sample pitched
//!   up is low-passed below the engine's Nyquist before it is read (the
//!   kernel widens by `1 / fc`, up to [`SINC_MAX_HALF_TAPS`] taps per side).
//!   Weights are normalised to sum to 1, so DC passes exactly.
//!
//! Every kernel reads a channel stored with [`super::pcm::PCM_PAD`] zero
//! frames on each side ([`super::pcm::SamplePcm::padded`]).

use super::pcm::PCM_PAD;
use alloc::vec::Vec;

/// Zero crossings of the sinc on each side at unit cutoff.
pub const SINC_ZERO_CROSSINGS: usize = 16;
/// Taps per side at most: enough for the full window an octave up
/// (`ceil(16 / 0.44) = 37`); beyond that the cutoff still falls but the
/// window covers fewer zero crossings.
pub const SINC_MAX_HALF_TAPS: usize = 40;
/// Cutoff at unit step, as a fraction of the source Nyquist frequency.
pub const SINC_CUTOFF: f32 = 0.88;
/// Kaiser window shape.
pub const SINC_KAISER_BETA: f64 = 8.6;
/// Table points per zero crossing.
const TABLE_RES: usize = 512;

/// Which kernel a voice reads its samples with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    /// Two-point linear interpolation.
    Linear,
    /// Four-point cubic Hermite (Catmull-Rom).
    Hermite,
    /// Step-tracking Kaiser-windowed sinc.
    Sinc,
}

impl Interp {
    /// The kernel a `quality` port value selects: 0 linear, 1 Hermite,
    /// 2 sinc; anything else (including NaN) gives `default`.
    pub fn from_quality(q: f32, default: Interp) -> Interp {
        if q == 0.0 {
            Interp::Linear
        } else if q == 1.0 {
            Interp::Hermite
        } else if q == 2.0 {
            Interp::Sinc
        } else {
            default
        }
    }
}

/// The windowed sinc `sinc(u) * kaiser(u / ZC)` for `u` in `0..=ZC`, sampled
/// `TABLE_RES` times per zero crossing (plus one guard point).
pub struct SincTable {
    table: Vec<f32>,
}

fn bessel_i0(x: f64) -> f64 {
    // Power series; converges quickly for the window's range (x <= beta).
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..64 {
        term *= q / ((k * k) as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

impl SincTable {
    /// Compute the table (about 4k floats).
    pub fn new() -> Self {
        let n = SINC_ZERO_CROSSINGS * TABLE_RES + 2;
        let i0b = bessel_i0(SINC_KAISER_BETA);
        let mut table = Vec::with_capacity(n);
        for i in 0..n {
            let u = i as f64 / TABLE_RES as f64;
            let s = if i == 0 {
                1.0
            } else {
                let px = core::f64::consts::PI * u;
                px.sin() / px
            };
            let r = u / SINC_ZERO_CROSSINGS as f64;
            let w = if r >= 1.0 {
                0.0
            } else {
                bessel_i0(SINC_KAISER_BETA * (1.0 - r * r).sqrt()) / i0b
            };
            table.push((s * w) as f32);
        }
        SincTable { table }
    }

    /// The window value at `|u| < ZC` (linear between table points).
    #[inline]
    fn at(&self, u: f32) -> f32 {
        let x = u * TABLE_RES as f32;
        let i = x as usize;
        let f = x - i as f32;
        let a = self.table[i];
        a + f * (self.table[i + 1] - a)
    }
}

impl Default for SincTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Catmull-Rom weight at distance `x` (sfizz's `hermite3`).
#[inline]
pub fn hermite3(x: f32) -> f32 {
    let x = x.abs();
    let x2 = x * x;
    let x3 = x2 * x;
    let q = 2.5 * x2;
    if x < 1.0 {
        1.0 - q + 1.5 * x3
    } else if x < 2.0 {
        2.0 - 4.0 * x + q - 0.5 * x3
    } else {
        0.0
    }
}

/// Precomputed sinc read parameters for one playback step: the cutoff and
/// the taps per side.
#[derive(Debug, Clone, Copy)]
pub struct SincSpan {
    fc: f32,
    half: usize,
}

impl SincSpan {
    /// The span for playback step `step` (source frames per output sample).
    pub fn for_step(step: f64) -> Self {
        let stretch = if step > 1.0 { step as f32 } else { 1.0 };
        let fc = SINC_CUTOFF / stretch;
        let half = ((SINC_ZERO_CROSSINGS as f32 / fc).ceil() as usize).min(SINC_MAX_HALF_TAPS);
        SincSpan { fc, half }
    }
}

/// Read a padded channel at fractional frame `pos` (frame 0 is the first
/// audio frame). `pos` must lie in `-1.0 ..= frames as f64` so every kernel's
/// window stays inside the padding.
#[inline]
pub fn read(kind: Interp, padded: &[f32], pos: f64, sinc: &SincTable, span: SincSpan) -> f32 {
    let fl = pos.floor();
    let i = fl as isize + PCM_PAD as isize;
    let f = (pos - fl) as f32;
    let i = i as usize;
    match kind {
        Interp::Linear => {
            let a = padded[i];
            a + f * (padded[i + 1] - a)
        }
        Interp::Hermite => {
            hermite3(-1.0 - f) * padded[i - 1]
                + hermite3(-f) * padded[i]
                + hermite3(1.0 - f) * padded[i + 1]
                + hermite3(2.0 - f) * padded[i + 2]
        }
        Interp::Sinc => {
            let zc = SINC_ZERO_CROSSINGS as f32;
            let mut acc = 0.0f32;
            let mut wsum = 0.0f32;
            // taps n = i - half + 1 ..= i + half; distance x = pos - n
            let half = span.half;
            for k in 0..2 * half {
                let n = i + 1 + k - half;
                let x = f + (half as f32 - 1.0 - k as f32);
                let u = (x * span.fc).abs();
                if u < zc {
                    let w = sinc.at(u);
                    acc += w * padded[n];
                    wsum += w;
                }
            }
            if wsum != 0.0 { acc / wsum } else { 0.0 }
        }
    }
}

/// [`read`] on one or two channels at once (the second is `None` for a mono
/// sample); the kernel weights are computed once for both.
#[inline]
pub fn read_pair(
    kind: Interp,
    a: &[f32],
    b: Option<&[f32]>,
    pos: f64,
    sinc: &SincTable,
    span: SincSpan,
) -> (f32, f32) {
    let Some(b) = b else {
        return (read(kind, a, pos, sinc, span), 0.0);
    };
    if kind != Interp::Sinc {
        return (
            read(kind, a, pos, sinc, span),
            read(kind, b, pos, sinc, span),
        );
    }
    let fl = pos.floor();
    let i = (fl as isize + PCM_PAD as isize) as usize;
    let f = (pos - fl) as f32;
    let zc = SINC_ZERO_CROSSINGS as f32;
    let half = span.half;
    let (mut acc_a, mut acc_b, mut wsum) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..2 * half {
        let n = i + 1 + k - half;
        let x = f + (half as f32 - 1.0 - k as f32);
        let u = (x * span.fc).abs();
        if u < zc {
            let w = sinc.at(u);
            acc_a += w * a[n];
            acc_b += w * b[n];
            wsum += w;
        }
    }
    if wsum != 0.0 {
        (acc_a / wsum, acc_b / wsum)
    } else {
        (0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_pair_matches_read_per_channel() {
        let table = SincTable::new();
        let a: Vec<f32> = (0..80).map(|i| (i as f32 * 0.21).sin()).collect();
        let b: Vec<f32> = (0..80).map(|i| (i as f32 * 0.05).cos()).collect();
        let (pa, pb) = (padded(&a), padded(&b));
        for kind in [Interp::Linear, Interp::Hermite, Interp::Sinc] {
            for step in [0.7, 1.0, 1.9] {
                let span = SincSpan::for_step(step);
                for pos in [3.25, 40.5, 77.9] {
                    let (x, y) = read_pair(kind, &pa, Some(&pb), pos, &table, span);
                    assert!((x - read(kind, &pa, pos, &table, span)).abs() < 1e-6);
                    assert!((y - read(kind, &pb, pos, &table, span)).abs() < 1e-6);
                }
            }
        }
    }

    fn padded(v: &[f32]) -> Vec<f32> {
        let mut out = alloc::vec![0.0; PCM_PAD];
        out.extend_from_slice(v);
        out.extend(core::iter::repeat_n(0.0, PCM_PAD));
        out
    }

    #[test]
    fn every_kernel_is_exact_at_integer_positions_and_passes_dc() {
        let table = SincTable::new();
        let ramp: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let p = padded(&ramp);
        for kind in [Interp::Linear, Interp::Hermite] {
            for (k, &v) in ramp.iter().enumerate() {
                let got = read(kind, &p, k as f64, &table, SincSpan::for_step(1.0));
                assert!((got - v).abs() < 1e-6, "{kind:?} at {k}: {got} vs {v}");
            }
        }
        // Sinc at fc < 1 is a lowpass, not an interpolator through the points;
        // on a constant it returns the constant.
        let dc = padded(&[0.5; 200]);
        for step in [0.5, 1.0, 1.5, 2.0, 4.0, 8.0] {
            let span = SincSpan::for_step(step);
            let got = read(Interp::Sinc, &dc, 100.3, &table, span);
            assert!((got - 0.5).abs() < 1e-5, "step {step}: {got}");
        }
    }

    #[test]
    fn the_sinc_widens_with_the_step_and_caps() {
        // 16 zero crossings at cutoff 0.88: ceil(16 / 0.88) taps per side
        assert_eq!(SincSpan::for_step(1.0).half, 19);
        assert_eq!(SincSpan::for_step(0.5).half, 19);
        assert_eq!(SincSpan::for_step(2.0).half, 37);
        assert_eq!(SincSpan::for_step(16.0).half, SINC_MAX_HALF_TAPS);
    }

    #[test]
    fn from_quality_maps_and_defaults() {
        assert_eq!(Interp::from_quality(0.0, Interp::Sinc), Interp::Linear);
        assert_eq!(Interp::from_quality(1.0, Interp::Sinc), Interp::Hermite);
        assert_eq!(Interp::from_quality(2.0, Interp::Linear), Interp::Sinc);
        assert_eq!(
            Interp::from_quality(f32::NAN, Interp::Hermite),
            Interp::Hermite
        );
        assert_eq!(Interp::from_quality(-1.0, Interp::Hermite), Interp::Hermite);
    }
}
