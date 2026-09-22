//! Filter UGens: OnePole, BiquadLPF, BiquadHPF, BiquadBPF, CombFilter, GVerb.
//!
//! Biquad filters use the standard transposed direct form II implementation.
//! Coefficients are recalculated per-sample to support audio-rate modulation
//! of cutoff frequency and Q.

use crate::buffer::{AudioBuffer, channel_wrapped, read_input, require_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use crate::ugens::delayline::DelayLine;
use core::f32::consts::TAU;

// --- OnePole ---

/// Simple one-pole lowpass/highpass filter.
///
/// Inputs: in (signal), coeff (filter coefficient in (-1, 1)).
///   coeff > 0: lowpass (higher = more smoothing)
///   coeff < 0: highpass
///
/// y[n] = (1 - |coeff|) * x[n] + coeff * y[n-1]
pub struct OnePole {
    y1: [f32; 2],
}

impl Default for OnePole {
    fn default() -> Self {
        Self::new()
    }
}

impl OnePole {
    pub fn new() -> Self {
        OnePole { y1: [0.0; 2] }
    }
}

impl UGen for OnePole {
    ugen_spec!(
        "OnePole",
        category = Filter,
        inputs = ["in"],
        optional_inputs = ["coeff"],
        outputs = ["out"]
    );

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        self.y1 = [0.0; 2];
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let in_buf = require_input(inputs, 0, self.spec().name, "in");
        let coeff_buf = inputs.get(1).copied().flatten();

        // One state per channel. Filter memory shared across channels made
        // channel 1 restart every block from channel 0's memory, which is a
        // block-rate buzz on the right whenever the two inputs differ (any
        // stage after a stereo one). Every stateful UGen follows this shape:
        // read the channel's own block-start state, write it back to the
        // channel's own slot. A third or later channel reads channel 1's
        // block-start state and does not write back, as `Compressor` does.
        // Sources with no per-channel input keep one state instead and
        // snapshot it before the channel loop, so every channel derives
        // the same block rather than continuing from the previous channel.
        for ch in 0..output.num_channels() {
            let mut y1 = self.y1[ch.min(1)];
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let coeff = read_input(coeff_buf, ch, i, 0.5);
                let abs_coeff = coeff.abs().min(0.9999);
                let x = in_ch[i];
                y1 = (1.0 - abs_coeff) * x + coeff * y1;
                out[i] = y1;
            }

            if let Some(slot) = self.y1.get_mut(ch) {
                *slot = y1;
            }
        }
    }
}

// --- Biquad state ---

/// Biquad filter state (transposed direct form II), one per channel.
///
/// `pub(crate)`: reused as-is by `ugens::partials`'s shaped-noise band
/// filters (MOT-636) rather than duplicating the recurrence a second time.
#[derive(Clone, Copy)]
pub(crate) struct BiquadState {
    z1: f32,
    z2: f32,
}

impl BiquadState {
    pub(crate) fn new() -> Self {
        BiquadState { z1: 0.0, z2: 0.0 }
    }

    /// Process one sample through the biquad.
    #[inline]
    pub(crate) fn tick(&mut self, x: f32, b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> f32 {
        let y = b0 * x + self.z1;
        self.z1 = b1 * x - a1 * y + self.z2;
        self.z2 = b2 * x - a2 * y;
        y
    }
}

/// The shared front half of every RBJ biquad coefficient formula.
///
/// All five filter shapes below derive their coefficients from the same three
/// intermediates — `sin(w0)`, `cos(w0)`, and `alpha` — and differ only in how
/// they combine them into `b0`/`b1`/`b2`.
#[inline]
fn biquad_params(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32) {
    let w0 = TAU * freq / sample_rate;
    let (sin_w0, cos_w0) = (w0.sin(), w0.cos());
    let alpha = sin_w0 / (2.0 * q);
    (sin_w0, cos_w0, alpha)
}

/// The shared back half: normalize all coefficients by `a0`.
///
/// `a0`/`a1`/`a2` are identical across every shape except allpass, but are
/// taken as parameters so each formula stays self-contained and readable.
#[inline]
fn normalize(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> (f32, f32, f32, f32, f32) {
    let inv_a0 = 1.0 / a0;
    (
        b0 * inv_a0,
        b1 * inv_a0,
        b2 * inv_a0,
        a1 * inv_a0,
        a2 * inv_a0,
    )
}

/// Compute biquad lowpass coefficients from freq, q, and sample_rate.
#[inline]
fn biquad_lpf_coeffs(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
    let (_sin_w0, cos_w0, alpha) = biquad_params(freq, q, sample_rate);

    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = b0;
    normalize(b0, b1, b2, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
}

/// Compute biquad highpass coefficients.
#[inline]
fn biquad_hpf_coeffs(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
    let (_sin_w0, cos_w0, alpha) = biquad_params(freq, q, sample_rate);

    let b0 = (1.0 + cos_w0) / 2.0;
    let b1 = -(1.0 + cos_w0);
    let b2 = b0;
    normalize(b0, b1, b2, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
}

/// Compute biquad bandpass coefficients (constant-peak-gain).
///
/// `pub(crate)`: reused by `ugens::partials` for its fixed shaped-noise band
/// filters (MOT-636).
#[inline]
pub(crate) fn biquad_bpf_coeffs(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
    let (_sin_w0, cos_w0, alpha) = biquad_params(freq, q, sample_rate);

    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    normalize(b0, b1, b2, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
}

/// Compute biquad notch (band-reject) coefficients.
#[inline]
fn biquad_notch_coeffs(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
    let (_sin_w0, cos_w0, alpha) = biquad_params(freq, q, sample_rate);

    let b0 = 1.0;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0;
    normalize(b0, b1, b2, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
}

/// Compute biquad allpass coefficients.
#[inline]
fn biquad_allpass_coeffs(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
    let (_sin_w0, cos_w0, alpha) = biquad_params(freq, q, sample_rate);

    let b0 = 1.0 - alpha;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 + alpha;
    normalize(b0, b1, b2, 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha)
}

/// Compute biquad peaking-EQ coefficients (boost/cut a band around `freq`).
///
/// `gain_db` is the boost (positive) or cut (negative) at the center
/// frequency; `q` controls the bandwidth of the affected region, same as the
/// other biquad shapes above.
#[inline]
fn biquad_peaking_coeffs(
    freq: f32,
    q: f32,
    gain_db: f32,
    sample_rate: f32,
) -> (f32, f32, f32, f32, f32) {
    let (_sin_w0, cos_w0, alpha) = biquad_params(freq, q, sample_rate);
    let a = 10f32.powf(gain_db / 40.0);

    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 - alpha * a;
    normalize(b0, b1, b2, 1.0 + alpha / a, -2.0 * cos_w0, 1.0 - alpha / a)
}

/// Compute biquad low-shelf coefficients (boost/cut everything below `freq`).
///
/// RBJ cookbook shelf formula, parameterized by `q` (rather than shelf slope
/// `S`) for consistency with the other biquad shapes' `freq`/`q` inputs.
#[inline]
fn biquad_low_shelf_coeffs(
    freq: f32,
    q: f32,
    gain_db: f32,
    sample_rate: f32,
) -> (f32, f32, f32, f32, f32) {
    let (_sin_w0, cos_w0, alpha) = biquad_params(freq, q, sample_rate);
    let a = 10f32.powf(gain_db / 40.0);
    let sqrt_a = a.sqrt();
    let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;

    let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
    normalize(b0, b1, b2, a0, a1, a2)
}

/// Compute biquad high-shelf coefficients (boost/cut everything above `freq`).
#[inline]
fn biquad_high_shelf_coeffs(
    freq: f32,
    q: f32,
    gain_db: f32,
    sample_rate: f32,
) -> (f32, f32, f32, f32, f32) {
    let (_sin_w0, cos_w0, alpha) = biquad_params(freq, q, sample_rate);
    let a = 10f32.powf(gain_db / 40.0);
    let sqrt_a = a.sqrt();
    let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;

    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;
    normalize(b0, b1, b2, a0, a1, a2)
}

// --- Biquad filters (LPF / HPF / BPF / Notch / Allpass) ---
//
// These five second-order filters share an identical struct, lifecycle, port
// specs, and per-sample processing loop; they differ only in their coefficient
// formula and default Q. The `biquad_ugen!` macro stamps each as a concrete
// named type so the DSL registry and `pub use filters::*` re-exports keep
// referencing them by name.

/// Generate a second-order biquad filter UGen.
///
/// Every biquad filter shares the same struct, `Default`/`new`, port specs,
/// lifecycle, and per-sample processing loop; they differ only in their
/// coefficient function (`coeffs`) and default Q (`q_default`). Coefficients
/// are recomputed per sample to support audio-rate modulation of cutoff and Q.
macro_rules! biquad_ugen {
    (
        $(#[$meta:meta])*
        $ty:ident, $name:literal, coeffs = $coeffs:path, q_default = $q_default:expr $(,)?
    ) => {
        $(#[$meta])*
        pub struct $ty {
            state: [BiquadState; 2],
            sample_rate: f32,
        }

        impl Default for $ty {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $ty {
            pub fn new() -> Self {
                $ty {
                    state: [BiquadState::new(); 2],
                    sample_rate: 44100.0,
                }
            }
        }

        impl UGen for $ty {
            ugen_spec!(
                $name,
                category = Filter,
                inputs = ["in"],
                optional_inputs = ["freq", "q"],
                outputs = ["out"]
            );

            fn init(&mut self, context: &ProcessContext) {
                self.sample_rate = context.sample_rate;
            }

            fn reset(&mut self) {
                self.state = [BiquadState::new(); 2];
            }

            fn process(
                &mut self,
                _context: &ProcessContext,
                inputs: &[Option<&AudioBuffer>],
                output: &mut AudioBuffer,
            ) {
                let in_buf = require_input(inputs, 0, self.spec().name, "in");
                let freq_buf = inputs.get(1).copied().flatten();
                let q_buf = inputs.get(2).copied().flatten();
                let sr = self.sample_rate;
                let nyquist = sr * 0.5;

                // One state per channel: see OnePole's process().
                for ch in 0..output.num_channels() {
                    let mut state = self.state[ch.min(1)];
                    let in_ch = channel_wrapped(in_buf, ch);
                    let out = output.channel_mut(ch).samples_mut();

                    for i in 0..out.len() {
                        let freq = read_input(freq_buf, ch, i, 1000.0)
                            .clamp(20.0, nyquist - 1.0);
                        let q = read_input(q_buf, ch, i, $q_default)
                            .max(0.01);

                        let (b0, b1, b2, a1, a2) = $coeffs(freq, q, sr);
                        out[i] = state.tick(in_ch[i], b0, b1, b2, a1, a2);
                    }

                    if let Some(slot) = self.state.get_mut(ch) {
                        *slot = state;
                    }
                }
            }
        }
    };
}

biquad_ugen! {
    /// Second-order Butterworth-style lowpass filter.
    ///
    /// Inputs: in (signal), freq (cutoff Hz), q (resonance, default 0.707).
    BiquadLPF, "BiquadLPF", coeffs = biquad_lpf_coeffs, q_default = 0.707
}

biquad_ugen! {
    /// Second-order highpass filter.
    ///
    /// Inputs: in (signal), freq (cutoff Hz), q (resonance, default 0.707).
    BiquadHPF, "BiquadHPF", coeffs = biquad_hpf_coeffs, q_default = 0.707
}

biquad_ugen! {
    /// Second-order bandpass filter.
    ///
    /// Inputs: in (signal), freq (center Hz), q (bandwidth).
    BiquadBPF, "BiquadBPF", coeffs = biquad_bpf_coeffs, q_default = 1.0
}

biquad_ugen! {
    /// Second-order notch (band-reject) filter.
    ///
    /// Attenuates a narrow band around the center frequency while passing
    /// all other frequencies. The width of the notch is controlled by Q.
    ///
    /// Inputs: in (signal), freq (center Hz), q (notch width, default 1.0).
    BiquadNotch, "BiquadNotch", coeffs = biquad_notch_coeffs, q_default = 1.0
}

biquad_ugen! {
    /// Second-order allpass filter.
    ///
    /// Passes all frequencies at unity gain but shifts the phase. The phase
    /// shift is frequency-dependent and centered around the specified frequency.
    /// Useful for building phasers, diffusion networks, and custom reverbs.
    ///
    /// Inputs: in (signal), freq (center Hz), q (bandwidth, default 0.707).
    AllpassFilter, "AllpassFilter", coeffs = biquad_allpass_coeffs, q_default = 0.707
}

// --- Parametric EQ shapes (Peaking / Low-shelf / High-shelf) ---
//
// Same struct/lifecycle/process shape as `biquad_ugen!` above, plus a `gain`
// (dB) input the five filters above don't need.

/// Generate a second-order biquad EQ UGen with a `gain` (dB) input, alongside
/// the existing `freq`/`q` inputs. See `biquad_ugen!` for the shared shape;
/// this differs only by the extra input and the coefficient function's extra
/// `gain_db` parameter.
macro_rules! biquad_gain_ugen {
    (
        $(#[$meta:meta])*
        $ty:ident, $name:literal, coeffs = $coeffs:path, q_default = $q_default:expr, gain_default = $gain_default:expr $(,)?
    ) => {
        $(#[$meta])*
        pub struct $ty {
            state: [BiquadState; 2],
            sample_rate: f32,
        }

        impl Default for $ty {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $ty {
            pub fn new() -> Self {
                $ty {
                    state: [BiquadState::new(); 2],
                    sample_rate: 44100.0,
                }
            }
        }

        impl UGen for $ty {
            ugen_spec!(
                $name,
                category = Filter,
                inputs = ["in"],
                optional_inputs = ["freq", "q", "gain"],
                outputs = ["out"]
            );

            fn init(&mut self, context: &ProcessContext) {
                self.sample_rate = context.sample_rate;
            }

            fn reset(&mut self) {
                self.state = [BiquadState::new(); 2];
            }

            fn process(
                &mut self,
                _context: &ProcessContext,
                inputs: &[Option<&AudioBuffer>],
                output: &mut AudioBuffer,
            ) {
                let in_buf = require_input(inputs, 0, self.spec().name, "in");
                let freq_buf = inputs.get(1).copied().flatten();
                let q_buf = inputs.get(2).copied().flatten();
                let gain_buf = inputs.get(3).copied().flatten();
                let sr = self.sample_rate;
                let nyquist = sr * 0.5;

                // One state per channel: see OnePole's process().
                for ch in 0..output.num_channels() {
                    let mut state = self.state[ch.min(1)];
                    let in_ch = channel_wrapped(in_buf, ch);
                    let out = output.channel_mut(ch).samples_mut();

                    for i in 0..out.len() {
                        let freq = read_input(freq_buf, ch, i, 1000.0)
                            .clamp(20.0, nyquist - 1.0);
                        let q = read_input(q_buf, ch, i, $q_default).max(0.01);
                        let gain_db = read_input(gain_buf, ch, i, $gain_default);

                        let (b0, b1, b2, a1, a2) = $coeffs(freq, q, gain_db, sr);
                        out[i] = state.tick(in_ch[i], b0, b1, b2, a1, a2);
                    }

                    if let Some(slot) = self.state.get_mut(ch) {
                        *slot = state;
                    }
                }
            }
        }
    };
}

biquad_gain_ugen! {
    /// Peaking EQ: boosts or cuts a band centered on `freq`.
    ///
    /// Inputs: in (signal), freq (center Hz, default 1000), q (bandwidth,
    /// default 1.0), gain (dB boost/cut, default 0.0).
    BiquadPeaking, "BiquadPeaking", coeffs = biquad_peaking_coeffs, q_default = 1.0, gain_default = 0.0
}

biquad_gain_ugen! {
    /// Low shelf: boosts or cuts everything below `freq`.
    ///
    /// Inputs: in (signal), freq (corner Hz, default 1000), q (transition
    /// steepness, default 0.707), gain (dB boost/cut, default 0.0).
    BiquadLowShelf, "BiquadLowShelf", coeffs = biquad_low_shelf_coeffs, q_default = 0.707, gain_default = 0.0
}

biquad_gain_ugen! {
    /// High shelf: boosts or cuts everything above `freq`.
    ///
    /// Inputs: in (signal), freq (corner Hz, default 1000), q (transition
    /// steepness, default 0.707), gain (dB boost/cut, default 0.0).
    BiquadHighShelf, "BiquadHighShelf", coeffs = biquad_high_shelf_coeffs, q_default = 0.707, gain_default = 0.0
}

// --- ParametricEq3 ---

/// Three-band parametric EQ: low shelf → peaking → high shelf in series.
///
/// A convenience wrapper over the three shapes above so a chain spec can
/// author a full tone-shaping EQ as one stage instead of three. Each band
/// keeps the same dB/Q/freq semantics as its standalone UGen.
///
/// Inputs:
/// - `in`: audio signal
/// - `lowFreq`/`lowGain`/`lowQ`: low-shelf corner (Hz, default 200), gain (dB,
///   default 0), and Q (default 0.707)
/// - `midFreq`/`midGain`/`midQ`: peaking band center (Hz, default 1000), gain
///   (dB, default 0), and Q (default 1.0)
/// - `highFreq`/`highGain`/`highQ`: high-shelf corner (Hz, default 5000),
///   gain (dB, default 0), and Q (default 0.707)
pub struct ParametricEq3 {
    /// Per channel: low shelf, peaking, high shelf.
    bands: [[BiquadState; 3]; 2],
    sample_rate: f32,
}

impl Default for ParametricEq3 {
    fn default() -> Self {
        Self::new()
    }
}

impl ParametricEq3 {
    pub fn new() -> Self {
        ParametricEq3 {
            bands: [[BiquadState::new(); 3]; 2],
            sample_rate: 44100.0,
        }
    }
}

impl UGen for ParametricEq3 {
    ugen_spec!(
        "ParametricEq3",
        category = Filter,
        inputs = ["in"],
        optional_inputs = [
            "lowFreq", "lowGain", "lowQ", "midFreq", "midGain", "midQ", "highFreq", "highGain",
            "highQ"
        ],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.bands = [[BiquadState::new(); 3]; 2];
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let in_buf = require_input(inputs, 0, self.spec().name, "in");
        let low_freq_buf = inputs.get(1).copied().flatten();
        let low_gain_buf = inputs.get(2).copied().flatten();
        let low_q_buf = inputs.get(3).copied().flatten();
        let mid_freq_buf = inputs.get(4).copied().flatten();
        let mid_gain_buf = inputs.get(5).copied().flatten();
        let mid_q_buf = inputs.get(6).copied().flatten();
        let high_freq_buf = inputs.get(7).copied().flatten();
        let high_gain_buf = inputs.get(8).copied().flatten();
        let high_q_buf = inputs.get(9).copied().flatten();
        let sr = self.sample_rate;
        let nyquist = sr * 0.5;

        // One state per channel: see OnePole's process().
        for ch in 0..output.num_channels() {
            let [mut low, mut mid, mut high] = self.bands[ch.min(1)];
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let low_freq = read_input(low_freq_buf, ch, i, 200.0).clamp(20.0, nyquist - 1.0);
                let low_gain = read_input(low_gain_buf, ch, i, 0.0);
                let low_q = read_input(low_q_buf, ch, i, 0.707).max(0.01);
                let mid_freq = read_input(mid_freq_buf, ch, i, 1000.0).clamp(20.0, nyquist - 1.0);
                let mid_gain = read_input(mid_gain_buf, ch, i, 0.0);
                let mid_q = read_input(mid_q_buf, ch, i, 1.0).max(0.01);
                let high_freq = read_input(high_freq_buf, ch, i, 5000.0).clamp(20.0, nyquist - 1.0);
                let high_gain = read_input(high_gain_buf, ch, i, 0.0);
                let high_q = read_input(high_q_buf, ch, i, 0.707).max(0.01);

                let (b0, b1, b2, a1, a2) = biquad_low_shelf_coeffs(low_freq, low_q, low_gain, sr);
                let x1 = low.tick(in_ch[i], b0, b1, b2, a1, a2);

                let (b0, b1, b2, a1, a2) = biquad_peaking_coeffs(mid_freq, mid_q, mid_gain, sr);
                let x2 = mid.tick(x1, b0, b1, b2, a1, a2);

                let (b0, b1, b2, a1, a2) =
                    biquad_high_shelf_coeffs(high_freq, high_q, high_gain, sr);
                out[i] = high.tick(x2, b0, b1, b2, a1, a2);
            }

            if let Some(slot) = self.bands.get_mut(ch) {
                *slot = [low, mid, high];
            }
        }
    }
}

// --- CombFilter ---

/// Maximum comb filter delay time in seconds.
const MAX_COMB_DELAY_SECS: f32 = 1.0;

/// Feedback comb filter (IIR).
///
/// y[n] = x[n] + feedback * y[n - delay]
///
/// Inputs: in (signal), delay (delay time in seconds), feedback (0.0 to ~0.99).
/// Useful for Karplus-Strong synthesis, flanging, and as a building block for reverbs.
pub struct CombFilter {
    lines: [DelayLine; 2],
    sample_rate: f32,
}

impl Default for CombFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl CombFilter {
    pub fn new() -> Self {
        CombFilter {
            lines: [DelayLine::new(), DelayLine::new()],
            sample_rate: 44100.0,
        }
    }
}

impl UGen for CombFilter {
    ugen_spec!(
        "CombFilter",
        category = Filter,
        inputs = ["in"],
        optional_inputs = ["delay", "feedback"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (MAX_COMB_DELAY_SECS * context.sample_rate) as usize + 1;
        for line in &mut self.lines {
            line.resize(max_samples);
        }
    }

    fn reset(&mut self) {
        for line in &mut self.lines {
            line.clear();
        }
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let in_buf = require_input(inputs, 0, self.spec().name, "in");
        let delay_buf = inputs.get(1).copied().flatten();
        let fb_buf = inputs.get(2).copied().flatten();
        if self.lines[0].is_empty() {
            return;
        }
        let max_delay = (self.lines[0].len() - 1) as f32;
        let sample_rate = self.sample_rate;

        // One delay line per channel: see OnePole's process().
        for ch in 0..output.num_channels() {
            let line = &mut self.lines[ch.min(1)];
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let delay_time = read_input(delay_buf, ch, i, 0.01).max(0.0);
                let feedback = read_input(fb_buf, ch, i, 0.5).clamp(-0.999, 0.999);

                let delay_samples = (delay_time * sample_rate).min(max_delay).max(1.0);

                // IIR comb: output = input + feedback * delayed_output
                let delayed = line.read_interp(delay_samples);
                let y = in_ch[i] + feedback * delayed;

                line.write_and_advance(y);
                out[i] = y;
            }
        }
    }
}

// --- GVerb ---

/// A damped comb filter for use inside the reverb.
struct ReverbComb {
    delay: DelayLine,
    filter_state: f32,
    delay_samples: usize,
}

impl ReverbComb {
    fn new(delay_samples: usize) -> Self {
        ReverbComb {
            delay: DelayLine::with_len(delay_samples + 1),
            filter_state: 0.0,
            delay_samples,
        }
    }

    fn clear(&mut self) {
        self.delay.clear();
        self.filter_state = 0.0;
    }

    /// Process one sample through the damped comb filter.
    #[inline]
    fn tick(&mut self, input: f32, feedback: f32, damping: f32) -> f32 {
        let delayed = self.delay.read(self.delay_samples);
        // One-pole lowpass on feedback path for damping
        self.filter_state = delayed * (1.0 - damping) + self.filter_state * damping;
        let y = input + self.filter_state * feedback;
        self.delay.write_and_advance(y);
        delayed
    }
}

/// An allpass filter for use inside the reverb.
struct ReverbAllpass {
    delay: DelayLine,
    delay_samples: usize,
}

impl ReverbAllpass {
    fn new(delay_samples: usize) -> Self {
        ReverbAllpass {
            delay: DelayLine::with_len(delay_samples + 1),
            delay_samples,
        }
    }

    fn clear(&mut self) {
        self.delay.clear();
    }

    /// Process one sample through the allpass.
    #[inline]
    fn tick(&mut self, input: f32, feedback: f32) -> f32 {
        let delayed = self.delay.read(self.delay_samples);
        let y = -input + delayed;
        self.delay.write_and_advance(input + delayed * feedback);
        y
    }
}

/// Schroeder-style reverb (similar to FreeVerb/GVerb).
///
/// Architecture: 8 parallel damped comb filters → 4 series allpass filters.
/// Produces stereo output from mono input via slightly different delay taps
/// for left and right channels.
///
/// Inputs:
/// - in: audio signal
/// - roomsize: room size factor (0.0 to 1.0, scales feedback)
/// - damping: high frequency damping (0.0 to 1.0)
/// - wet: wet signal level (0.0 to 1.0)
/// - dry: dry signal level (0.0 to 1.0)
///
/// The wet path is level-normalized: pink noise at `wet = 1.0, dry = 0.0` comes out at
/// its input RMS level for any `roomsize` and `damping`, so those two parameters change
/// the decay and tone of the reverb but not its loudness.
pub struct GVerb {
    combs_l: [ReverbComb; 8],
    combs_r: [ReverbComb; 8],
    allpasses_l: [ReverbAllpass; 4],
    allpasses_r: [ReverbAllpass; 4],
}

// Comb filter delay lengths in samples at 44100 Hz (prime-ish numbers for diffusion).
const COMB_DELAYS_L: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
// Stereo spread offset for right channel decorrelation.
const STEREO_SPREAD: usize = 23;
const ALLPASS_DELAYS_L: [usize; 4] = [556, 441, 341, 225];

/// Damping values at which `GVERB_WET_LOG_GAIN` is sampled. Spacing tightens toward
/// 1.0, where the wet path's gain falls steeply.
const GVERB_DAMPING_NODES: [f32; 29] = [
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 1.0,
];

/// Natural log of the uncompensated wet path's pink-noise RMS gain, averaged over both
/// channels, indexed `[roomsize step][damping node]` with roomsize sampled at 0.0, 0.05,
/// ..., 1.0 and damping at `GVERB_DAMPING_NODES`.
#[rustfmt::skip]
const GVERB_WET_LOG_GAIN: [[f32; 29]; 21] = [
    [3.14637, 3.14216, 3.13869, 3.13565, 3.13288, 3.13029, 3.12780, 3.12535, 3.12291, 3.12044, 3.11787, 3.11516, 3.11225, 3.10906, 3.10546, 3.10130, 3.09630, 3.09000, 3.08148, 3.06790, 3.05341, 3.04816, 3.04502, 3.03109, 3.01066, 2.97700, 2.91358, 2.87634, 2.85298],
    [3.16712, 3.16261, 3.15893, 3.15573, 3.15282, 3.15010, 3.14750, 3.14495, 3.14241, 3.13983, 3.13716, 3.13434, 3.13131, 3.12798, 3.12423, 3.11989, 3.11468, 3.10808, 3.09916, 3.08490, 3.06953, 3.06382, 3.06023, 3.04476, 3.02196, 2.98453, 2.91599, 2.87704, 2.85298],
    [3.18940, 3.18456, 3.18066, 3.17728, 3.17424, 3.17140, 3.16868, 3.16602, 3.16338, 3.16069, 3.15792, 3.15499, 3.15184, 3.14837, 3.14447, 3.13995, 3.13451, 3.12762, 3.11827, 3.10333, 3.08702, 3.08082, 3.07669, 3.05950, 3.03404, 2.99245, 2.91846, 2.87777, 2.85298],
    [3.21333, 3.20814, 3.20401, 3.20046, 3.19727, 3.19430, 3.19146, 3.18870, 3.18595, 3.18316, 3.18028, 3.17724, 3.17397, 3.17037, 3.16631, 3.16161, 3.15594, 3.14874, 3.13898, 3.12332, 3.10602, 3.09928, 3.09454, 3.07539, 3.04695, 3.00075, 2.92099, 2.87853, 2.85298],
    [3.23909, 3.23352, 3.22913, 3.22540, 3.22206, 3.21896, 3.21601, 3.21314, 3.21028, 3.20739, 3.20440, 3.20125, 3.19786, 3.19413, 3.18991, 3.18503, 3.17912, 3.17161, 3.16142, 3.14503, 3.12668, 3.11935, 3.11389, 3.09251, 3.06072, 3.00945, 2.92358, 2.87932, 2.85298],
    [3.26684, 3.26085, 3.25621, 3.25229, 3.24879, 3.24556, 3.24249, 3.23951, 3.23655, 3.23355, 3.23046, 3.22719, 3.22368, 3.21981, 3.21544, 3.21037, 3.20423, 3.19640, 3.18577, 3.16863, 3.14915, 3.14117, 3.13487, 3.11097, 3.07540, 3.01855, 2.92624, 2.88017, 2.85298],
    [3.29678, 3.29033, 3.28542, 3.28130, 3.27765, 3.27429, 3.27109, 3.26800, 3.26493, 3.26183, 3.25863, 3.25526, 3.25162, 3.24762, 3.24309, 3.23783, 3.23145, 3.22331, 3.21223, 3.19430, 3.17362, 3.16490, 3.15762, 3.13085, 3.09103, 3.02804, 2.92898, 2.88108, 2.85298],
    [3.32914, 3.32218, 3.31697, 3.31265, 3.30884, 3.30534, 3.30203, 3.29882, 3.29564, 3.29244, 3.28913, 3.28565, 3.28189, 3.27775, 3.27307, 3.26762, 3.26101, 3.25253, 3.24099, 3.22225, 3.20026, 3.19072, 3.18229, 3.15223, 3.10763, 3.03793, 2.93182, 2.88207, 2.85298],
    [3.36414, 3.35663, 3.35111, 3.34658, 3.34260, 3.33896, 3.33553, 3.33220, 3.32892, 3.32561, 3.32220, 3.31860, 3.31472, 3.31044, 3.30560, 3.29997, 3.29311, 3.28430, 3.27229, 3.25268, 3.22926, 3.21880, 3.20901, 3.17520, 3.12521, 3.04820, 2.93476, 2.88317, 2.85298],
    [3.40209, 3.39394, 3.38809, 3.38334, 3.37919, 3.37541, 3.37184, 3.36840, 3.36501, 3.36159, 3.35807, 3.35437, 3.35037, 3.34595, 3.34095, 3.33513, 3.32803, 3.31887, 3.30637, 3.28584, 3.26083, 3.24929, 3.23793, 3.19981, 3.14379, 3.05884, 2.93786, 2.88440, 2.85298],
    [3.44328, 3.43442, 3.42822, 3.42323, 3.41891, 3.41497, 3.41127, 3.40771, 3.40420, 3.40068, 3.39705, 3.39323, 3.38911, 3.38454, 3.37938, 3.37337, 3.36602, 3.35650, 3.34349, 3.32196, 3.29515, 3.28238, 3.26916, 3.22610, 3.16333, 3.06985, 2.94115, 2.88581, 2.85298],
    [3.48808, 3.47840, 3.47181, 3.46658, 3.46206, 3.45797, 3.45413, 3.45044, 3.44681, 3.44317, 3.43944, 3.43550, 3.43125, 3.42653, 3.42120, 3.41499, 3.40738, 3.39749, 3.38392, 3.36128, 3.33240, 3.31818, 3.30279, 3.25408, 3.18381, 3.08123, 2.94471, 2.88745, 2.85298],
    [3.53692, 3.52628, 3.51926, 3.51376, 3.50904, 3.50478, 3.50078, 3.49695, 3.49319, 3.48943, 3.48558, 3.48152, 3.47712, 3.47225, 3.46674, 3.46032, 3.45244, 3.44214, 3.42796, 3.40404, 3.37274, 3.35679, 3.33886, 3.28373, 3.20518, 3.09302, 2.94866, 2.88942, 2.85298],
    [3.59031, 3.57852, 3.57102, 3.56522, 3.56027, 3.55581, 3.55165, 3.54766, 3.54375, 3.53985, 3.53587, 3.53167, 3.52712, 3.52208, 3.51637, 3.50972, 3.50155, 3.49079, 3.47591, 3.45049, 3.41627, 3.39824, 3.37736, 3.31495, 3.22739, 3.10528, 2.95317, 2.89182, 2.85298],
    [3.64892, 3.63571, 3.62765, 3.62151, 3.61630, 3.61161, 3.60724, 3.60306, 3.59899, 3.59493, 3.59079, 3.58643, 3.58171, 3.57645, 3.57052, 3.56363, 3.55511, 3.54384, 3.52810, 3.50085, 3.46302, 3.44246, 3.41817, 3.34762, 3.25041, 3.11819, 2.95854, 2.89483, 2.85298],
    [3.71370, 3.69868, 3.68995, 3.68340, 3.67787, 3.67291, 3.66828, 3.66387, 3.65957, 3.65531, 3.65098, 3.64644, 3.64149, 3.63598, 3.62979, 3.62261, 3.61368, 3.60178, 3.58496, 3.55534, 3.51295, 3.48926, 3.46109, 3.38161, 3.27430, 3.13213, 2.96524, 2.89867, 2.85298],
    [3.78618, 3.76873, 3.75918, 3.75211, 3.74618, 3.74085, 3.73589, 3.73116, 3.72657, 3.72205, 3.71748, 3.71267, 3.70743, 3.70160, 3.69509, 3.68756, 3.67811, 3.66540, 3.64714, 3.61433, 3.56596, 3.53836, 3.50586, 3.41687, 3.29941, 3.14787, 2.97406, 2.90371, 2.85298],
    [3.86917, 3.84830, 3.83768, 3.82995, 3.82346, 3.81762, 3.81218, 3.80701, 3.80201, 3.79712, 3.79220, 3.78703, 3.78139, 3.77512, 3.76821, 3.76022, 3.75005, 3.73628, 3.71598, 3.67860, 3.62211, 3.58957, 3.55239, 3.45381, 3.32682, 3.16714, 2.98639, 2.91046, 2.85298],
    [3.96865, 3.94261, 3.93054, 3.92190, 3.91461, 3.90799, 3.90182, 3.89598, 3.89038, 3.88494, 3.87948, 3.87375, 3.86749, 3.86061, 3.85324, 3.84464, 3.83340, 3.81815, 3.79494, 3.75041, 3.68237, 3.64348, 3.60161, 3.49434, 3.35959, 3.19380, 3.00471, 2.91975, 2.85298],
    [4.09944, 4.06487, 4.05076, 4.04079, 4.03221, 4.02425, 4.01682, 4.00992, 4.00342, 3.99711, 3.99070, 3.98394, 3.97661, 3.96884, 3.96105, 3.95162, 3.93859, 3.92137, 3.89435, 3.83712, 3.75172, 3.70461, 3.65879, 3.54575, 3.40686, 3.23717, 3.03365, 2.93280, 2.85298],
    [4.30712, 4.25669, 4.23962, 4.22751, 4.21654, 4.20588, 4.19598, 4.18723, 4.17931, 4.17161, 4.16321, 4.15398, 4.14420, 4.13492, 4.12737, 4.11631, 4.09964, 4.07987, 4.05064, 3.96652, 3.85426, 3.79726, 3.74998, 3.63725, 3.49848, 3.32135, 3.08191, 2.95154, 2.85298],
];

impl Default for GVerb {
    fn default() -> Self {
        Self::new()
    }
}

impl GVerb {
    /// The factor applied to the wet path so that pink noise at `wet = 1.0, dry = 0.0`
    /// comes out at its input RMS level. Both arguments are clamped to 0.0..=1.0.
    pub fn wet_gain_compensation(roomsize: f32, damping: f32) -> f32 {
        let roomsize = roomsize.clamp(0.0, 1.0);
        let damping = damping.clamp(0.0, 1.0);
        let last_row = GVERB_WET_LOG_GAIN.len() - 1;
        let x = roomsize * last_row as f32;
        let i = (x as usize).min(last_row - 1);
        let fx = x - i as f32;
        let last_cell = GVERB_DAMPING_NODES.len() - 2;
        let j = GVERB_DAMPING_NODES[1..=last_cell]
            .iter()
            .take_while(|&&node| node <= damping)
            .count();
        let (d0, d1) = (GVERB_DAMPING_NODES[j], GVERB_DAMPING_NODES[j + 1]);
        let fy = (damping - d0) / (d1 - d0);
        let g = &GVERB_WET_LOG_GAIN;
        let log_gain = g[i][j] * (1.0 - fx) * (1.0 - fy)
            + g[i + 1][j] * fx * (1.0 - fy)
            + g[i][j + 1] * (1.0 - fx) * fy
            + g[i + 1][j + 1] * fx * fy;
        (-log_gain).exp()
    }

    pub fn new() -> Self {
        GVerb {
            combs_l: core::array::from_fn(|i| ReverbComb::new(COMB_DELAYS_L[i])),
            combs_r: core::array::from_fn(|i| ReverbComb::new(COMB_DELAYS_L[i] + STEREO_SPREAD)),
            allpasses_l: core::array::from_fn(|i| ReverbAllpass::new(ALLPASS_DELAYS_L[i])),
            allpasses_r: core::array::from_fn(|i| {
                ReverbAllpass::new(ALLPASS_DELAYS_L[i] + STEREO_SPREAD)
            }),
        }
    }
}

/// The per-sample reverb parameters, shared by both stereo sides.
#[derive(Clone, Copy)]
struct GVerbParams<'a> {
    roomsize: Option<&'a AudioBuffer>,
    damping: Option<&'a AudioBuffer>,
    wet: Option<&'a AudioBuffer>,
    dry: Option<&'a AudioBuffer>,
}

impl GVerb {
    /// Render one stereo side: mono input through that side's parallel comb
    /// bank, then its series allpass chain, mixed against the dry signal.
    ///
    /// The two sides are identical but for their delay taps (see `STEREO_SPREAD`),
    /// so both go through here with their own comb/allpass banks.
    fn render_side(
        combs: &mut [ReverbComb; 8],
        allpasses: &mut [ReverbAllpass; 4],
        in_ch: &[f32],
        out: &mut [f32],
        params: GVerbParams<'_>,
    ) {
        for (i, out_sample) in out.iter_mut().enumerate() {
            let input = in_ch[i];
            let roomsize = read_input(params.roomsize, 0, i, 0.5).clamp(0.0, 1.0);
            let damping = read_input(params.damping, 0, i, 0.5).clamp(0.0, 1.0);
            let wet = read_input(params.wet, 0, i, 0.3).clamp(0.0, 1.0);
            let dry = read_input(params.dry, 0, i, 0.7).clamp(0.0, 1.0);

            // Scale roomsize to feedback (0.0 → 0.7, 1.0 → 0.98)
            let feedback = 0.7 + roomsize * 0.28;

            // Sum of parallel comb filters
            let mut comb_sum = 0.0;
            for comb in combs.iter_mut() {
                comb_sum += comb.tick(input, feedback, damping);
            }

            // Series allpass filters
            let mut signal = comb_sum;
            for ap in allpasses.iter_mut() {
                signal = ap.tick(signal, 0.5);
            }

            let wet_gain = wet * Self::wet_gain_compensation(roomsize, damping);
            *out_sample = input * dry + signal * wet_gain;
        }
    }
}

impl UGen for GVerb {
    ugen_spec!(
        "GVerb",
        category = Filter,
        inputs = ["in"],
        optional_inputs = ["roomsize", "damping", "wet", "dry"],
        outputs = ["out"]
    );

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        for c in &mut self.combs_l {
            c.clear();
        }
        for c in &mut self.combs_r {
            c.clear();
        }
        for a in &mut self.allpasses_l {
            a.clear();
        }
        for a in &mut self.allpasses_r {
            a.clear();
        }
    }

    fn output_channels(&self, _input_channels: &[usize]) -> usize {
        2 // always stereo output
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let params = GVerbParams {
            roomsize: inputs.get(1).copied().flatten(),
            damping: inputs.get(2).copied().flatten(),
            wet: inputs.get(3).copied().flatten(),
            dry: inputs.get(4).copied().flatten(),
        };

        let in_ch = require_input(inputs, 0, self.spec().name, "in")
            .channel(0)
            .samples();

        Self::render_side(
            &mut self.combs_l,
            &mut self.allpasses_l,
            in_ch,
            output.channel_mut(0).samples_mut(),
            params,
        );

        if output.num_channels() >= 2 {
            Self::render_side(
                &mut self.combs_r,
                &mut self.allpasses_r,
                in_ch,
                output.channel_mut(1).samples_mut(),
                params,
            );
        }
    }
}

// --- Compressor ---

/// Feed-forward compressor with sidechain support.
///
/// Reduces dynamic range by attenuating signals above a threshold.
/// Uses a log-domain envelope follower with separate attack and release times.
///
/// Inputs:
/// - `in`: signal to compress
/// - `sidechain`: signal used for level detection (use `audioIn` for external sidechain,
///   or connect the same signal as `in` for self-sidechaining)
/// - `threshold`: level in decibels above which compression begins (e.g. -10.0)
/// - `ratio`: compression ratio (e.g. 4.0 means 4:1 — for every 4 dB above threshold,
///   output increases by 1 dB)
/// - `attack`: attack time in seconds (how fast the compressor reacts to increases)
/// - `release`: release time in seconds (how fast the compressor recovers)
/// - `makeup`: makeup gain in decibels added after compression
pub struct Compressor {
    /// Envelope follower state per channel (in dB).
    env_db: [f32; 2],
    sample_rate: f32,
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor {
    pub fn new() -> Self {
        Compressor {
            env_db: [-120.0; 2],
            sample_rate: 44100.0,
        }
    }
}

/// Fast log2 approximation using IEEE 754 float bit tricks (no_std compatible).
/// Accurate to ~0.09 dB for audio signals.
#[inline]
fn fast_log2(x: f32) -> f32 {
    let bits = x.to_bits() as f32;
    // IEEE 754: bits = mantissa + exponent * 2^23
    // log2(x) ≈ bits / 2^23 - 127 (with correction)
    bits * (1.0 / 8388608.0) - 127.0
}

/// Convert linear amplitude to decibels using fast log2.
/// 20*log10(x) = 20 * log2(x) / log2(10) ≈ 6.0206 * log2(x)
#[inline]
fn fast_lin_to_db(x: f32) -> f32 {
    let abs = x.abs().max(1e-6);
    6.0206 * fast_log2(abs)
}

/// Convert decibels to linear gain.
/// 10^(db/20) = 2^(db / 6.0206)
#[inline]
fn fast_db_to_lin(db: f32) -> f32 {
    // 2^x via exp: 2^x = e^(x * ln2)
    (db * (1.0 / 6.0206) * core::f32::consts::LN_2).exp()
}

impl UGen for Compressor {
    ugen_spec!(
        "Compressor",
        category = Filter,
        inputs = ["in"],
        optional_inputs = [
            "sidechain",
            "threshold",
            "ratio",
            "attack",
            "release",
            "makeup"
        ],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        self.env_db = [-120.0; 2];
    }

    fn reset(&mut self) {
        self.env_db = [-120.0; 2];
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let in_buf = require_input(inputs, 0, self.spec().name, "in");
        let sc_buf = inputs.get(1).copied().flatten().unwrap_or(in_buf);
        let thresh_buf = inputs.get(2).copied().flatten();
        let ratio_buf = inputs.get(3).copied().flatten();
        let attack_buf = inputs.get(4).copied().flatten();
        let release_buf = inputs.get(5).copied().flatten();
        let makeup_buf = inputs.get(6).copied().flatten();

        for ch in 0..output.num_channels() {
            let in_ch = channel_wrapped(in_buf, ch);
            let sc_ch = channel_wrapped(sc_buf, ch);
            let out = output.channel_mut(ch).samples_mut();
            let env_idx = ch.min(1);
            let mut env_db = self.env_db[env_idx];

            for i in 0..out.len() {
                let threshold = read_input(thresh_buf, ch, i, -10.0);
                let ratio = read_input(ratio_buf, ch, i, 4.0).max(1.0);
                let attack_time = read_input(attack_buf, ch, i, 0.01).max(0.0001);
                let release_time = read_input(release_buf, ch, i, 0.1).max(0.0001);
                let makeup = read_input(makeup_buf, ch, i, 0.0);

                // Sidechain level detection (peak, in dB)
                let sc_db = fast_lin_to_db(sc_ch[i]);

                // Smooth envelope follower (separate attack/release)
                let coeff = if sc_db > env_db {
                    // Attack: fast rise
                    (-1.0 / (attack_time * self.sample_rate)).exp()
                } else {
                    // Release: slow decay
                    (-1.0 / (release_time * self.sample_rate)).exp()
                };
                env_db = coeff * env_db + (1.0 - coeff) * sc_db;

                // Gain computation
                let over_db = env_db - threshold;
                let gain_db = if over_db > 0.0 {
                    // Compress: reduce by (1 - 1/ratio) * overshoot
                    -(over_db * (1.0 - 1.0 / ratio))
                } else {
                    0.0
                };

                let gain = fast_db_to_lin(gain_db + makeup);
                out[i] = in_ch[i] * gain;
            }

            if ch <= 1 {
                self.env_db[env_idx] = env_db;
            }
        }
    }
}

// --- Limiter ---

/// Fixed look-ahead window for the limiter's gain computation.
///
/// Sized so the fast-attack gain smoothing below (a quarter of this) settles
/// well before a sample that triggered a gain reduction reaches the output
/// tap. Not exposed as a parameter — see the module doc for why a fixed,
/// small look-ahead was chosen over a tunable one.
const LIMITER_LOOKAHEAD_SECS: f32 = 0.0015;

/// Small fixed safety margin subtracted from `ceiling` internally.
///
/// Re-derived from characterization data, not assumed. The gain applied to
/// a given output sample is a single scalar derived from its own local peak
/// estimate, but neighboring samples carry very slightly different gain
/// (the envelope is smoothed, not frozen), so the *actual* reconstructed
/// inter-sample curve isn't quite the same as "peak estimate times one
/// gain." With the windowed-sinc estimator below (replacing a 4-point
/// Catmull-Rom spline measured optimistic by several dB on realistic
/// material -- see that estimator's own doc comment),
/// `examples/measure_limiter.rs --characterize` measured the *residual* gap
/// between the estimator and an independent `ffmpeg` true-peak measurement
/// at a roughly constant +0.34 dB (sparse test content) / +0.68-0.76 dB
/// (dense, adversarial multi-tone-near-Nyquist content) across every gain
/// and ceiling swept -- i.e. still a structural bias from finite kernel
/// width and the single-rate gain architecture (see that module's doc
/// comment on the fix), just a few dB smaller than before and no longer
/// scaling with limiting depth. A 1.0 dB margin (i.e. the worst-case 0.757
/// dB residual plus modest headroom) was tried first and still left the
/// adversarial dense fixture ~0.1 dB over ceiling -- the smoothed causal
/// component doesn't reach a newly-lowered target instantaneously, so
/// re-deriving purely from the *steady-state* residual undercounted the
/// attack envelope's own settling slop. 1.5 dB held the ceiling against
/// `ffmpeg` on every case in that sweep, but only with ~0.04 dB to spare on
/// the adversarial dense case against `tests/ugens.rs`'s permanent
/// regression check (a differently-windowed, differently-sized reference
/// than this file's, calibrated to track `ffmpeg` slightly on the strict
/// side) -- too tight to be a stable, non-flaky gate. 2.0 dB gives that
/// regression test real headroom while still holding every
/// `ffmpeg`-verified case comfortably. A separate, larger, whole-piece
/// guard band applied by callers upstream of this UGen is a different
/// mechanism with its own disposition, not derived from this number.
const LIMITER_SAFETY_MARGIN_DB: f32 = 2.0;

/// True-peak (not just sample-peak) brick-wall limiter with look-ahead.
///
/// A plain `Compressor` + `SoftClip` chain was tried first, per the RFC's own
/// suggestion, and measured against a hot test signal using an oversampled
/// (cubic-interpolated) true-peak estimate — the same technique used below.
/// It did not hold a true-peak ceiling reliably: a compressor's envelope
/// follower reacts to the *sample* peak, and soft clipping's added harmonics
/// can produce inter-sample peaks a purely reactive, no-lookahead chain
/// cannot see coming. This UGen exists because that was demonstrated, not
/// assumed.
///
/// Design: a short internal look-ahead delay line (~1.5 ms, not user-facing)
/// gives the gain envelope time to react *before* a loud sample reaches the
/// output. At every input sample, a cheap 4x-oversampled cubic (Catmull-Rom)
/// interpolation of the two most recent samples estimates the true peak
/// arriving right now; the gain needed to keep that peak under `ceiling` is
/// computed and smoothed toward with a fast, fixed attack (bounded by the
/// look-ahead) and a slower, parametric release. Because the gain and the
/// look-ahead delay share the same sample clock, by the time a given sample
/// reaches the output tap the gain has already had a full look-ahead window
/// to settle to the value that sample needs.
///
/// This is a true-peak-aware limiter, not a true-peak-*exact* one: the
/// interpolation is a cheap local estimate (see `true_peak` in the test
/// suite for the independent oversampled check used to validate it), not a
/// full oversampled signal path. It is, however, measurably better than the
/// reactive preset it replaces — see `tests/ugens.rs`'s limiter tests for the
/// measurement.
///
/// Inputs:
/// - `in`: signal to limit
/// - `ceiling`: true-peak ceiling in dBTP (default -1.0)
/// - `release`: gain recovery time in seconds after a peak passes (default 0.05)
pub struct Limiter {
    /// Independent per-channel look-ahead buffers. A limiter's ceiling is
    /// a hard numeric guarantee: one buffer shared across channels would
    /// let channel 1's look-ahead window read back channel 0's stale
    /// samples for most of every block (the block is shorter than the
    /// look-ahead), silently swapping in a different signal's peak
    /// estimate. A hot stereo test signal exposed exactly that (see
    /// `tests/ugens.rs`'s stereo cross-channel test); every channel must
    /// see only its own history. Every effect that processes its channels
    /// separately keeps one line per channel the same way.
    delays: [DelayLine; 2],
    lookahead_samples: usize,
    /// Per-channel gain state (capped at 2 channels, same as `delays` and
    /// the same convention as `Compressor::env_db`). A 3rd+ channel would
    /// share channel 1's delay/gain state; not a concern for a limiter,
    /// which is essentially always mono or stereo.
    gain_db: [f32; 2],
    attack_coeff: f32,
    sample_rate: f32,
    /// Precomputed `sinc * Blackman-Harris` kernel, one row per oversample
    /// phase (`LIMITER_OVERSAMPLE - 1` rows, phase 0 needs no interpolation)
    /// times `2*SINC_HALF_TAPS` taps per row, flattened. `sin`/`cos` are not
    /// `const fn`-able on stable Rust and this UGen's `process` runs at
    /// audio rate (potentially in a WASM AudioWorklet), so the kernel is
    /// computed once here -- in `init`, off the audio thread's steady-state
    /// path -- rather than recomputing trig functions per sample per tap.
    sinc_kernel: alloc::vec::Vec<f32>,
}

impl Default for Limiter {
    fn default() -> Self {
        Self::new()
    }
}

impl Limiter {
    pub fn new() -> Self {
        Limiter {
            delays: [DelayLine::new(), DelayLine::new()],
            lookahead_samples: 1,
            gain_db: [0.0; 2],
            attack_coeff: 0.0,
            sample_rate: 44100.0,
            sinc_kernel: alloc::vec::Vec::new(),
        }
    }
}

/// Builds the flattened `[phase][tap]` windowed-sinc kernel table described
/// on [`Limiter::sinc_kernel`]. `phase` is 1-based in the returned table's
/// row order (row 0 corresponds to oversample phase 1, i.e. `t = 1 /
/// LIMITER_OVERSAMPLE`) since phase 0 (`t = 0`) is exactly the tap-`k=-1`
/// sample and needs no interpolation.
fn build_sinc_kernel() -> alloc::vec::Vec<f32> {
    let taps = 2 * SINC_HALF_TAPS;
    let taps_f = taps as f32;
    let mut table = alloc::vec::Vec::with_capacity((LIMITER_OVERSAMPLE - 1) * taps);
    for phase in 1..LIMITER_OVERSAMPLE {
        let t = phase as f32 / LIMITER_OVERSAMPLE as f32;
        for tap_idx in 0..taps {
            let k = tap_idx as isize - SINC_HALF_TAPS as isize;
            let dist = k as f32 - t + 1.0;
            let window = blackman_harris(k as f32 + SINC_HALF_TAPS as f32, taps_f);
            table.push(sinc(dist) * window);
        }
    }
    table
}

/// Precise linear-to-dB conversion for the limiter's peak/gain math.
///
/// `Compressor` above uses `fast_lin_to_db`/`fast_db_to_lin` (~0.09 dB
/// error) because it's a continuously-modulated envelope where a fraction of
/// a dB of ripple is inaudible. The limiter is instead judged against a hard
/// ceiling, where that same error could be the difference between holding it
/// and not — so it pays the exact `log10`/`powf` here instead.
#[inline]
fn precise_lin_to_db(x: f32) -> f32 {
    20.0 * x.abs().max(1e-9).log10()
}

#[inline]
fn precise_db_to_lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// A local 4-point Catmull-Rom spline (the technique this replaced) is a
/// curve-fitting convenience, not a model of a bandlimited signal's
/// continuous-time reconstruction — it has no defined stopband, so it
/// systematically misses inter-sample energy that a real oversampling
/// reconstruction filter would show. This crate's own characterization
/// (`examples/measure_limiter.rs --characterize`) measured that blind spot
/// at a *constant* +3.35 dB (sparse content) / +3.84 dB (dense,
/// multi-tone content) against an independent `ffmpeg` measurement, present
/// at every gain and every ceiling tested once the limiter was engaged at
/// all — i.e. not a "one big spike," a structural bias in the interpolation
/// technique itself.
///
/// The replacement below is a windowed-sinc lowpass reconstruction filter —
/// the same family ITU-R BS.1770-4 Annex 2 specifies for true-peak
/// measurement (a proper oversampling filter, not a local polynomial fit).
/// A `SINC_HALF_TAPS`-sample-radius kernel, windowed with Blackman-Harris to
/// control stopband ripple, is evaluated at `LIMITER_OVERSAMPLE` phases
/// between the two nearest real samples; the interpolation converges toward
/// the actual continuous-time peak of the bandlimited signal as the kernel
/// widens, unlike the cubic spline's fixed (and biased) 4-point fit.
const SINC_HALF_TAPS: usize = 32;

/// Oversample factor (phases evaluated between each pair of real samples)
/// for the true-peak interpolation below.
const LIMITER_OVERSAMPLE: usize = 8;

/// Normalized sinc: `sin(pi*x) / (pi*x)`, with the removable singularity at
/// `x == 0` handled explicitly.
#[inline]
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-7 {
        1.0
    } else {
        let px = core::f32::consts::PI * x;
        px.sin() / px
    }
}

/// Blackman-Harris window, `n` in `0..=taps` (inclusive), for a kernel of
/// `taps` samples total width. Low sidelobes (~-92 dB) keep the windowed
/// sinc's stopband well below the sub-dB precision this estimate needs.
#[inline]
fn blackman_harris(n: f32, taps: f32) -> f32 {
    const A0: f32 = 0.358_75;
    const A1: f32 = 0.488_29;
    const A2: f32 = 0.141_28;
    const A3: f32 = 0.011_68;
    let x = core::f32::consts::TAU * n / taps;
    A0 - A1 * x.cos() + A2 * (2.0 * x).cos() - A3 * (3.0 * x).cos()
}

/// Windowed-sinc interpolation at oversample `phase` (`1..LIMITER_OVERSAMPLE`,
/// i.e. `t = phase / LIMITER_OVERSAMPLE`) of a `2*SINC_HALF_TAPS`-sample
/// kernel, via `get(k)` for `k` in `-SINC_HALF_TAPS..SINC_HALF_TAPS`. By
/// convention `get(-1)` is the sample at `t=0` and `get(0)` is the sample at
/// `t=1`; more negative/positive `k` extend the kernel further from the two
/// samples being interpolated between, giving the sinc kernel the wider
/// context a bandlimited reconstruction needs beyond just its two nearest
/// neighbors. `kernel` is [`Limiter::sinc_kernel`] (or an equivalent table
/// from [`build_sinc_kernel`]) -- the `sinc * window` coefficients, already
/// computed, so this inner loop is pure multiply-accumulate.
#[inline]
fn sinc_interp(kernel: &[f32], phase: usize, get: impl Fn(isize) -> f32) -> f32 {
    let taps = 2 * SINC_HALF_TAPS;
    let row = &kernel[(phase - 1) * taps..phase * taps];
    let mut acc = 0.0f32;
    for (tap_idx, &coeff) in row.iter().enumerate() {
        let k = tap_idx as isize - SINC_HALF_TAPS as isize;
        acc += get(k) * coeff;
    }
    acc
}

/// Causal oversampled true-peak estimate for the sample just written at
/// `delay(0)`, reconstructing the segment from `delay(1)` (`t=0`) to
/// `delay(0)` (`t=1`) -- the freshest fully-real segment available, exactly
/// as the old estimator did, just with a wider sinc kernel instead of a
/// 4-point spline. Taps beyond `delay(0)` are genuine future samples that
/// don't exist yet, so they repeat `delay(0)` (the closest real value) --
/// this only feeds the anticipatory envelope, not the hard ceiling
/// guarantee, so a slight approximation here is fine (see module doc).
#[inline]
fn causal_true_peak(line: &DelayLine, kernel: &[f32]) -> f32 {
    let newest = line.read(0);
    let get = |k: isize| -> f32 {
        if k <= 0 {
            line.read((-k) as usize)
        } else {
            newest
        }
    };
    let mut peak = newest.abs();
    for phase in 1..LIMITER_OVERSAMPLE {
        peak = peak.max(sinc_interp(kernel, phase, get).abs());
    }
    peak
}

/// Exact, symmetric oversampled true-peak estimate for the sample about to
/// be output, at `delay(center)`. Unlike [`causal_true_peak`], every tap in
/// both reconstructed segments is real data already sitting in the
/// look-ahead buffer (written on earlier iterations), so no repeated-sample
/// approximation is needed anywhere in the kernel. This is what gives the
/// ceiling its hard guarantee; [`causal_true_peak`]'s smoothed envelope only
/// gives it a musical shape.
#[inline]
fn centered_true_peak(line: &DelayLine, center: usize, kernel: &[f32]) -> f32 {
    let at_center = line.read(center);
    // Segment "before center": delay(center+1) at t=0 to delay(center) at
    // t=1, i.e. get_before(-1) = delay(center+1), get_before(0) = delay(center)
    // => get_before(k) = delay(center - k).
    let get_before = |k: isize| -> f32 {
        let delay = center as isize - k;
        line.read(delay.max(0) as usize)
    };
    // Segment "after center": delay(center) at t=0 to delay(center-1) at
    // t=1, i.e. get_after(-1) = delay(center), get_after(0) = delay(center-1)
    // => get_after(k) = delay(center - 1 - k).
    let get_after = |k: isize| -> f32 {
        let delay = center as isize - 1 - k;
        line.read(delay.max(0) as usize)
    };
    let mut peak = at_center.abs();
    for phase in 1..LIMITER_OVERSAMPLE {
        peak = peak.max(sinc_interp(kernel, phase, get_before).abs());
        peak = peak.max(sinc_interp(kernel, phase, get_after).abs());
    }
    peak
}

impl UGen for Limiter {
    ugen_spec!(
        "Limiter",
        category = Filter,
        inputs = ["in"],
        optional_inputs = ["ceiling", "release"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        // Floor raised from 4 to SINC_HALF_TAPS+2: centered_true_peak's wider
        // sinc kernel reads up to `center - SINC_HALF_TAPS - 1`, which
        // underflows a `center` smaller than that at degenerate sample rates.
        self.lookahead_samples =
            ((LIMITER_LOOKAHEAD_SECS * context.sample_rate) as usize).max(SINC_HALF_TAPS + 2);
        for d in &mut self.delays {
            // Must cover the widest kernel read: centered_true_peak's
            // "before" segment reads up to delay(center + SINC_HALF_TAPS).
            d.resize(self.lookahead_samples + SINC_HALF_TAPS + 2);
        }
        self.gain_db = [0.0; 2];
        let attack_time = LIMITER_LOOKAHEAD_SECS / 4.0;
        self.attack_coeff = (-1.0 / (attack_time * context.sample_rate)).exp();
        if self.sinc_kernel.is_empty() {
            self.sinc_kernel = build_sinc_kernel();
        }
    }

    fn reset(&mut self) {
        for d in &mut self.delays {
            d.clear();
        }
        self.gain_db = [0.0; 2];
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let in_buf = require_input(inputs, 0, self.spec().name, "in");
        let ceiling_buf = inputs.get(1).copied().flatten();
        let release_buf = inputs.get(2).copied().flatten();
        if self.delays[0].is_empty() {
            return;
        }
        let lookahead = self.lookahead_samples;
        let kernel = self.sinc_kernel.as_slice();

        for ch in 0..output.num_channels() {
            let delay_idx = ch.min(1);
            let delay = &mut self.delays[delay_idx];
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();
            let gain_idx = ch.min(1);
            let mut gain_db = self.gain_db[gain_idx];

            for i in 0..out.len() {
                let ceiling_db = read_input(ceiling_buf, ch, i, -1.0) - LIMITER_SAFETY_MARGIN_DB;
                let release_time = read_input(release_buf, ch, i, 0.05).max(0.0001);
                let release_coeff = (-1.0 / (release_time * self.sample_rate)).exp();

                delay.write(in_ch[i]);

                // Anticipatory component: starts the envelope moving as soon
                // as a loud sample is *written*, `lookahead` samples before
                // it will be read back out.
                let causal_peak_db = precise_lin_to_db(causal_true_peak(delay, kernel));
                let causal_target_db = (ceiling_db - causal_peak_db).min(0.0);
                let coeff = if causal_target_db < gain_db {
                    self.attack_coeff // needs MORE reduction: fast, fixed
                } else {
                    release_coeff // recovering: slower, parametric
                };
                gain_db = coeff * gain_db + (1.0 - coeff) * causal_target_db;

                // Exact component: the sample about to be read back out
                // already has real neighbors on both sides in the buffer, so
                // this is computed the same way the independent measurement
                // checks it — no approximation. Whichever component wants
                // *more* reduction wins; the exact one is the hard guarantee,
                // the smoothed one is what keeps it from sounding like a
                // sample-and-hold gate.
                let exact_peak_db = precise_lin_to_db(centered_true_peak(delay, lookahead, kernel));
                let exact_target_db = (ceiling_db - exact_peak_db).min(0.0);
                let applied_gain_db = gain_db.min(exact_target_db);

                let delayed = delay.read(lookahead);
                out[i] = delayed * precise_db_to_lin(applied_gain_db);

                delay.advance();
            }

            if ch <= 1 {
                self.gain_db[gain_idx] = gain_db;
            }
        }
    }
}
