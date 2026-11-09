//! Formant synthesis: the phase-aligned formant (PAF) generator.
//!
//! - [`Paf`]: one formant, a band of harmonics of `freq` centred on `center`
//!   with bandwidth `bw`, both continuous controls. A vowel is four or five
//!   `paf` nodes on the same `freq`, summed.

use crate::buffer::{AudioBuffer, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use core::f64::consts::TAU;

/// Largest envelope ratio `r` between neighbouring harmonics (see [`Paf`]).
/// It bounds the output's peak at `amp · (1 + r) / (1 − r)` = 199 · `amp`
/// and is reached once `bw` exceeds about 138 · `freq`.
pub const PAF_MAX_RATIO: f64 = 0.99;

/// `freq` is clamped to this range (Hz) before use.
const F0_MIN: f64 = 0.01;

/// `center` and `freq` are clamped below this fraction of the sample rate.
const NYQUIST_GUARD: f64 = 0.49;

/// Phase-aligned formant generator (Puckette 1995): one formant of a
/// harmonic tone, with the formant's centre and bandwidth as continuous,
/// audio-rate controls and no filter in the signal path.
///
/// With `θ` the phase of the fundamental (`freq`), `c = center / freq`,
/// `k = ⌊c⌋`, `α = c − k`:
///
/// ```text
/// out = amp · [ P_r(θ) · ((1 − α)·cos kθ + α·cos (k+1)θ) − DC ]
/// P_r(θ) = (1 − r²) / (1 − 2r·cos θ + r²) = 1 + 2 Σ_{n≥1} rⁿ cos nθ
/// r = 2^(−2·freq / bw)
/// ```
///
/// `P_r` is the *modulator*: a pulse with one peak per period whose
/// harmonics fall geometrically, by `r` per harmonic. It is Puckette's
/// waveshaped half-frequency sinusoid, `1 / (1 + b² sin²(θ/2))` up to a
/// constant, written in closed form (the Cauchy waveshaper
/// `f(x) = 1/(1 + x²)`; `b² = 4r / (1 − r)²`). Multiplying it by a cosine at
/// harmonic `k` moves that comb of decaying harmonics so it is centred on
/// harmonic `k`: line `k ± n` has amplitude `rⁿ`. The two carriers are
/// harmonics of the *same* phase `θ`, so every component of the output is an
/// exact harmonic of `freq` wherever `center` lies: a centre between two
/// harmonics is a crossfade of the two neighbouring combs, not a component
/// at `center` (which would beat against the harmonics). As `center` crosses
/// harmonic `k + 1`, `α` reaches 1 and the output is `P_r · cos (k+1)θ`,
/// which is exactly where it restarts with the new `k` at `α = 0`: the
/// sweep is continuous in value and in spectrum.
///
/// **Levels.** The line at the centre harmonic has amplitude `amp` (when
/// `center` is a harmonic), so `amp` is the formant's spectral peak, and the
/// spectral envelope is `amp · r^(|f − center| / freq)`: −6 dB at
/// `center ± bw/2`. `bw` is therefore the **−6 dB** bandwidth. (A two-pole
/// resonance of −3 dB bandwidth `B` is −6 dB wide at `√3 · B`; matching the
/// two at −6 dB, this envelope falls faster than the resonance's 6 dB/octave
/// skirts further out.) With `center` halfway between harmonics both lines
/// are `amp · (1 + r)/2`, and the formant's power dips by
/// `(1 + r)² / (2(1 + r²))`: under 1 dB once `bw` ≥ 1.25 · `freq`, but a
/// formant much narrower than the harmonic spacing loses up to 3 dB between
/// harmonics, as a real narrow resonance does (it is why sopranos tune F1
/// to a harmonic). The peak of the output is at most
/// `amp · (1 + r) / (1 − r)`, at most 199 · `amp` ([`PAF_MAX_RATIO`]); its
/// RMS is about `amp · sqrt((1 + r²) / (2(1 − r²)))`.
///
/// **Fold-over at 0 Hz.** Lines below harmonic 0 (`n > k`) fold onto
/// positive harmonics with the same sign (cos is even), so a formant whose
/// envelope reaches 0 Hz (`center` within a few `bw` of 0) is fuller on its
/// low side than the formula says, as in any PAF or FOF. The 0 Hz line
/// itself, `(1 − α)·r^k + α·r^(k+1)`, is subtracted: the output has no DC.
///
/// **Aliasing.** The lines do not stop at Nyquist; the first one past it is
/// about `20·log10(r) · (fs/2 − center) / freq` dB re the peak, i.e.
/// `−12 dB · (fs/2 − center) / bw`. For speech-like formants (`bw` up to a
/// few hundred Hz, `center` ≤ 8 kHz at 44.1/48 kHz) that is below −150 dB;
/// it reaches −40 dB only for bandwidths of about a third of the distance
/// from `center` to Nyquist. Folded lines are harmonics of `freq` folded at
/// Nyquist, so they are inharmonic unless `fs / freq` is an integer.
///
/// Inputs (all optional, all audio-rate):
/// - `freq`: the fundamental in Hz (default 220), clamped to
///   0.01 Hz … 0.49 · fs. At `freq ≤ 0` the output is 0 and the phase
///   holds, rather than a full-bandwidth tone at `center`.
/// - `center`: formant centre in Hz (default 800), clamped to 0 … 0.49 · fs;
///   audio-rate, so a vowel can morph or a formant sweep.
/// - `bw`: −6 dB bandwidth in Hz (default 100); `bw ≤ 0` leaves the bare
///   carriers (a sine at a harmonic `center`, two harmonics between).
/// - `amp`: the formant's peak level (default 1). Meant for an envelope or a
///   per-formant level; a non-finite `amp` reads as 0.
/// - `phase_reset`: on a rising edge (previous sample `<= 0`, this one
///   `> 0`) the fundamental's phase restarts at 0.
///
/// Other non-finite inputs read as their defaults.
pub struct Paf {
    phase: f64,
    prev_reset: f32,
    sample_rate: f32,
}

impl Default for Paf {
    fn default() -> Self {
        Self::new()
    }
}

impl Paf {
    pub fn new() -> Self {
        Paf {
            phase: 0.0,
            prev_reset: 0.0,
            sample_rate: 44100.0,
        }
    }
}

/// The geometric ratio between neighbouring lines for a −6 dB bandwidth `bw`
/// at fundamental `f0` (both Hz): `2^(−2·f0/bw)`, capped at [`PAF_MAX_RATIO`].
#[inline]
pub fn paf_ratio(f0: f64, bw: f64) -> f64 {
    if bw > 0.0 && f0 > 0.0 {
        (-2.0 * f0 / bw).exp2().min(PAF_MAX_RATIO)
    } else {
        0.0
    }
}

#[inline]
fn finite_or(x: f32, default: f32) -> f64 {
    if x.is_finite() {
        x as f64
    } else {
        default as f64
    }
}

impl UGen for Paf {
    ugen_spec!(
        "Paf",
        category = Oscillator,
        inputs = [],
        optional_inputs = ["freq", "center", "bw", "amp", "phase_reset"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.prev_reset = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied().flatten();
        let center_buf = inputs.get(1).copied().flatten();
        let bw_buf = inputs.get(2).copied().flatten();
        let amp_buf = inputs.get(3).copied().flatten();
        let reset_buf = inputs.get(4).copied().flatten();
        let sr = self.sample_rate as f64;
        let inv_sr = 1.0 / sr;
        let top = NYQUIST_GUARD * sr;

        // Every channel starts from the same block-start state (see FmOsc).
        let (phase0, reset0) = (self.phase, self.prev_reset);

        for ch in 0..output.num_channels() {
            let (mut phase, mut prev_reset) = (phase0, reset0);
            // the envelope ratio only changes when freq or bw do
            let (mut last_f0, mut last_bw, mut r) = (f64::NAN, f64::NAN, 0.0f64);
            // and r^k only when r or k do
            let (mut last_k, mut last_r, mut rk) = (f64::NAN, f64::NAN, 0.0f64);
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let f0 = finite_or(read_input(freq_buf, ch, i, 220.0), 220.0);
                let center = finite_or(read_input(center_buf, ch, i, 800.0), 800.0).clamp(0.0, top);
                let bw = finite_or(read_input(bw_buf, ch, i, 100.0), 100.0);
                // a VCA port: a non-finite level silences rather than opens
                let amp = finite_or(read_input(amp_buf, ch, i, 1.0), 0.0);
                let reset = read_input(reset_buf, ch, i, 0.0);

                if reset > 0.0 && prev_reset <= 0.0 {
                    phase = 0.0;
                }
                prev_reset = reset;

                if f0 <= 0.0 {
                    *out_sample = 0.0;
                    continue;
                }
                let f0 = f0.clamp(F0_MIN, top);

                if f0 != last_f0 || bw != last_bw {
                    r = paf_ratio(f0, bw);
                    last_f0 = f0;
                    last_bw = bw;
                }

                let c = center / f0;
                let k = c.floor();
                let alpha = c - k;
                let theta = TAU * phase;
                let (s1, c1) = theta.sin_cos();
                // cos kθ and cos (k+1)θ from one sin_cos of kθ (k·phase is
                // reduced to one cycle first, for precision at high k)
                let kp = k * phase;
                let (sk, ck) = (TAU * (kp - kp.floor())).sin_cos();
                let ck1 = ck * c1 - sk * s1;
                let carrier = (1.0 - alpha) * ck + alpha * ck1;
                let modulator = (1.0 - r * r) / (1.0 - 2.0 * r * c1 + r * r);
                // the 0 Hz line: the mean of P_r · cos nθ is rⁿ
                if k != last_k || r != last_r {
                    rk = if k == 0.0 { 1.0 } else { r.powf(k) };
                    last_k = k;
                    last_r = r;
                }
                let dc = (1.0 - alpha) * rk + alpha * rk * r;
                let y = amp * (modulator * carrier - dc);
                *out_sample = if y.is_finite() { y as f32 } else { 0.0 };

                phase += f0 * inv_sr;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
                self.prev_reset = prev_reset;
            }
        }
    }
}
