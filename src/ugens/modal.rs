//! Modal synthesis: a bank of damped two-pole resonators.
//!
//! - [`ModalBank`]: up to [`MODAL_MAX_MODES`] resonators, each tuned to a
//!   ratio of a keytracked fundamental, driven by one excitation signal.

use crate::buffer::{AudioBuffer, read_input, require_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use core::f64::consts::PI;

/// The most modes one [`ModalBank`] carries.
pub const MODAL_MAX_MODES: usize = 32;

/// The pitch a mode table's decay times refer to: C4, 261.6256 Hz.
pub const MODAL_F_REF: f32 = 261.625_56;

/// Number of ports before the per-mode ports (`in`, `freq`, `damp`, `pos`,
/// `bright`, `alpha`).
pub const MODAL_FIXED_PORTS: usize = 6;

/// The port index of mode `mode`'s `field` (0 = ratio `r`, 1 = gain `g`,
/// 2 = decay `d`), e.g. `modal_port(0, 0)` is `r0`.
pub const fn modal_port(mode: usize, field: usize) -> usize {
    MODAL_FIXED_PORTS + 3 * mode + field
}

/// Coefficients are recomputed every this many samples (0.36 ms at 44.1 kHz),
/// from the inputs at the first sample of each sub-block.
const SUB_BLOCK: usize = 16;

/// Bounds on a mode's decay (T60) after `damp` and keytracking: 0.5 ms .. 120 s.
const T60_MIN: f64 = 0.0005;
const T60_MAX: f64 = 120.0;

/// `ln 1000`: an exponential decay reaches -60 dB after `T60` when its time
/// constant is `T60 / ln 1000`.
const LN_1000: f64 = 6.907_755_278_982_137;

/// Modes at or above this fraction of the sample rate are dropped.
const NYQUIST_GUARD: f64 = 0.49;

/// The bank is done once excited, with a silent input block, and every
/// resonator's state below this magnitude (-120 dBFS).
const DONE_LEVEL: f64 = 1e-6;

/// A bank of damped two-pole resonators for modal synthesis (bars, bells,
/// membranes, plates, tines).
///
/// Each mode `k` is a resonator with poles at `ρ e^{±jω}`, written in the
/// coupled ("rotation") form, which keeps its amplitude when `ω` moves (a
/// pitch glide) and stays accurate at low frequencies in finite precision:
///
/// ```text
/// z_k[n] = ρ_k e^{jω_k} z_k[n-1] + w_k x[n]          y[n] = Σ_k Im z_k[n]
/// ω_k = 2π r_k f0 / fs      ρ_k = exp(-ln(1000) / (T60_k fs))
/// T60_k = d_k · damp · (f0 / C4)^(-alpha)
/// w_k = g_k · sin(π · pos · (k+1)) · 1 / sqrt(1 + (r_k f0 / fc)^4)
/// fc = 100 Hz · 2^(8 · bright)
/// ```
///
/// So an impulse of height 1 makes mode `k` ring as `w_k ρ_k^n sin(ω_k n)`: a
/// sine at `r_k · freq` that starts from zero, peaks near `w_k` and falls 60 dB
/// in `T60_k` seconds. Output is bounded by `Σ |w_k|` times the input's
/// peak for impulsive input; a sustained input at a resonance builds up to
/// `1 / (2(1 - ρ))` times the input there, as any resonator does.
///
/// # The mode table
///
/// The table travels as ordinary input ports, three per mode, so it lives in
/// the IR as inline consts and needs no table bank: `r0 g0 d0 r1 g1 d1 ...`
/// up to `r31 g31 d31` ([`modal_port`] gives the index):
///
/// - `rK`: mode `K`'s frequency as a ratio of `freq` (default 0 = mode
///   unused). Only modes with `rK > 0` and `rK · freq` under 0.49 fs ring;
///   a mode pushed past that (a pitch rise) is silenced.
/// - `gK`: its gain (default 1): the peak of its impulse response.
/// - `dK`: its decay, **T60 in seconds at f0 = C4** (default 1).
///
/// A mode table is a modelling choice made once per instrument, so these
/// ports, like `pos`, `bright` and `alpha`, are read once per 16-sample
/// sub-block, as are `freq` and `damp`; only `in` is read every sample.
///
/// # Inputs
///
/// - `in` (required): the excitation, e.g. an impulse, a noise burst under
///   a short envelope, or a lowpassed click. Its spectrum is what the modes
///   pick out.
/// - `freq`: the fundamental in Hz (default 440). Keytrack it from the def's
///   `freq` param, or hold it constant for a fixed-pitch drum; drive it with a
///   falling envelope for a membrane's pitch glide.
/// - `damp`: a global multiplier on every decay (default 1). Lowering it
///   while the modes ring chokes them (a hi-hat closing, a hand on a bar).
/// - `pos`: strike position, 0..1 (default 0 = no position weighting). For
///   `pos > 0` mode slot `K` is weighted by `sin(π · pos · (K + 1))`, the
///   shape of a string's or an ideal bar's `K+1`-th mode at the strike
///   point: `pos = 0.5` (the middle) silences slots 1, 3, 5, ... (the even
///   modes 2, 4, 6 ...). Order the table so slot `K` is the `K+1`-th mode of
///   the shape you mean, or leave `pos` at 0 and fold your own position
///   weights into the gains.
/// - `bright`: excitation bandwidth, the mallet hardness, 0..1 (default 1).
///   Each mode's input weight is a second-order lowpass at
///   `fc = 100 Hz · 2^(8 · bright)` evaluated at the mode's frequency: 100 Hz
///   (felt, a soft hand) at 0, 1.6 kHz at 0.5, 25.6 kHz (flat, a hard click)
///   at 1. It shapes how each mode is struck, not modes already ringing.
/// - `alpha`: keytracked damping exponent (default 0): every decay scales by
///   `(freq / C4)^(-alpha)`, so the bank keeps `T60 ∝ f^(-alpha)` along the
///   keyboard (1 to 2 for wood, 0.3 to 1 for metal). Decay *across* modes is
///   the table's `dK`.
///
/// Decays are clamped to 0.5 ms .. 120 s; non-finite inputs read as their
/// defaults (`in` as 0).
///
/// # Done
///
/// `is_done` turns true once the bank has been excited, its input has been
/// silent for a whole block, and every mode has fallen below -120 dBFS. It
/// keeps sounding after its input stops, like a delay or a reverb, so it
/// passes no silence from its input to its output.
pub struct ModalBank {
    sample_rate: f32,
    /// Resonator state, one per channel (left, right).
    re: [[f64; MODAL_MAX_MODES]; 2],
    im: [[f64; MODAL_MAX_MODES]; 2],
    excited: bool,
    done: bool,
}

impl Default for ModalBank {
    fn default() -> Self {
        Self::new()
    }
}

impl ModalBank {
    pub fn new() -> Self {
        ModalBank {
            sample_rate: 44100.0,
            re: [[0.0; MODAL_MAX_MODES]; 2],
            im: [[0.0; MODAL_MAX_MODES]; 2],
            excited: false,
            done: false,
        }
    }
}

/// `v` if finite, else `default`.
#[inline]
fn finite_or(v: f32, default: f32) -> f32 {
    if v.is_finite() { v } else { default }
}

/// One mode's coefficients for a sub-block: `ρ cos ω`, `ρ sin ω`, input weight.
#[derive(Clone, Copy, Default)]
struct ModeCoef {
    c: f64,
    s: f64,
    w: f64,
    live: bool,
}

impl UGen for ModalBank {
    ugen_spec!(
        "ModalBank",
        category = Physical,
        inputs = ["in"],
        optional_inputs = [
            "freq", "damp", "pos", "bright", "alpha", "r0", "g0", "d0", "r1", "g1", "d1", "r2",
            "g2", "d2", "r3", "g3", "d3", "r4", "g4", "d4", "r5", "g5", "d5", "r6", "g6", "d6",
            "r7", "g7", "d7", "r8", "g8", "d8", "r9", "g9", "d9", "r10", "g10", "d10", "r11",
            "g11", "d11", "r12", "g12", "d12", "r13", "g13", "d13", "r14", "g14", "d14", "r15",
            "g15", "d15", "r16", "g16", "d16", "r17", "g17", "d17", "r18", "g18", "d18", "r19",
            "g19", "d19", "r20", "g20", "d20", "r21", "g21", "d21", "r22", "g22", "d22", "r23",
            "g23", "d23", "r24", "g24", "d24", "r25", "g25", "d25", "r26", "g26", "d26", "r27",
            "g27", "d27", "r28", "g28", "d28", "r29", "g29", "d29", "r30", "g30", "d30", "r31",
            "g31", "d31",
        ],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        self.reset();
    }

    fn reset(&mut self) {
        self.re = [[0.0; MODAL_MAX_MODES]; 2];
        self.im = [[0.0; MODAL_MAX_MODES]; 2];
        self.excited = false;
        self.done = false;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let in_buf = require_input(inputs, 0, "ModalBank", "in");
        let port = |k: usize| inputs.get(k).copied().flatten();
        let (freq_buf, damp_buf, pos_buf, bright_buf, alpha_buf) =
            (port(1), port(2), port(3), port(4), port(5));
        let sr = self.sample_rate as f64;

        // One resonator state per channel, so a channel rings only from its
        // own input; the done flag covers every channel.
        let mut excited = self.excited;
        let mut heard_input = false;
        let mut level = 0.0f64;

        for ch in 0..output.num_channels() {
            let slot = ch.min(1);
            let (mut re, mut im) = (self.re[slot], self.im[slot]);
            let mut coef = [ModeCoef::default(); MODAL_MAX_MODES];
            let out = output.channel_mut(ch).samples_mut();
            let len = out.len();

            let mut start = 0;
            while start < len {
                let end = (start + SUB_BLOCK).min(len);
                let at = |buf: Option<&AudioBuffer>, d: f32| -> f64 {
                    finite_or(read_input(buf, ch, start, d), d) as f64
                };
                let f0 = at(freq_buf, 440.0);
                let damp = at(damp_buf, 1.0).max(0.0);
                let pos = at(pos_buf, 0.0).clamp(0.0, 1.0);
                let bright = at(bright_buf, 1.0).clamp(0.0, 1.0);
                let alpha = at(alpha_buf, 0.0).clamp(-4.0, 4.0);
                let key = if f0 > 0.0 {
                    (f0 / MODAL_F_REF as f64).powf(-alpha)
                } else {
                    1.0
                };
                let fc = 100.0 * (8.0 * bright).exp2();
                for (k, cf) in coef.iter_mut().enumerate() {
                    let ratio = at(port(modal_port(k, 0)), 0.0);
                    let f = ratio * f0;
                    if ratio <= 0.0 || f <= 0.0 || f >= NYQUIST_GUARD * sr {
                        if cf.live || re[k] != 0.0 || im[k] != 0.0 {
                            re[k] = 0.0;
                            im[k] = 0.0;
                        }
                        *cf = ModeCoef::default();
                        continue;
                    }
                    let gain = at(port(modal_port(k, 1)), 1.0);
                    let decay = at(port(modal_port(k, 2)), 1.0);
                    let t60 = (decay * damp * key).clamp(T60_MIN, T60_MAX);
                    let rho = (-LN_1000 / (t60 * sr)).exp();
                    let omega = 2.0 * PI * f / sr;
                    let pos_w = if pos > 0.0 {
                        (PI * pos * (k as f64 + 1.0)).sin()
                    } else {
                        1.0
                    };
                    let tilt = 1.0 / (1.0 + (f / fc).powi(4)).sqrt();
                    *cf = ModeCoef {
                        c: rho * omega.cos(),
                        s: rho * omega.sin(),
                        w: gain * pos_w * tilt,
                        live: true,
                    };
                }

                for (i, o) in out[start..end].iter_mut().enumerate() {
                    let x = finite_or(read_input(Some(in_buf), ch, start + i, 0.0), 0.0) as f64;
                    if x != 0.0 {
                        heard_input = true;
                    }
                    let mut y = 0.0f64;
                    for k in 0..MODAL_MAX_MODES {
                        let cf = coef[k];
                        if !cf.live {
                            continue;
                        }
                        let (a, b) = (re[k], im[k]);
                        re[k] = cf.c * a - cf.s * b + cf.w * x;
                        im[k] = cf.s * a + cf.c * b;
                        y += im[k];
                    }
                    *o = y as f32;
                }
                start = end;
            }

            // Flush denormal-range tails and anything non-finite.
            for k in 0..MODAL_MAX_MODES {
                if !(re[k].is_finite() && im[k].is_finite()) {
                    re[k] = 0.0;
                    im[k] = 0.0;
                }
                let m = re[k].abs().max(im[k].abs());
                if m < 1e-30 {
                    re[k] = 0.0;
                    im[k] = 0.0;
                }
                level = level.max(m);
            }
            if let Some(r) = self.re.get_mut(ch) {
                *r = re;
            }
            if let Some(i) = self.im.get_mut(ch) {
                *i = im;
            }
        }
        excited |= heard_input;
        self.excited = excited;
        self.done = excited && !heard_input && level < DONE_LEVEL;
    }

    fn is_done(&self) -> bool {
        self.done
    }
}
