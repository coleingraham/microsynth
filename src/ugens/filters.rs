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
    y1: f32,
}

impl Default for OnePole {
    fn default() -> Self {
        Self::new()
    }
}

impl OnePole {
    pub fn new() -> Self {
        OnePole { y1: 0.0 }
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
        self.y1 = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let in_buf = require_input(inputs, 0, self.spec().name, "in");
        let coeff_buf = inputs.get(1).copied().flatten();

        // Snapshot once, before the channel loop: every channel must start
        // from the same block-start state, not from whatever a prior
        // channel's iteration already wrote back (see MOT multichannel
        // state-writeback fix — read-back-inside-loop made channel 1 start
        // from channel 0's END-of-block state, a full block ahead).
        let y1_start = self.y1;

        for ch in 0..output.num_channels() {
            let mut y1 = y1_start;
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let coeff = read_input(coeff_buf, ch, i, 0.5);
                let abs_coeff = coeff.abs().min(0.9999);
                let x = in_ch[i];
                y1 = (1.0 - abs_coeff) * x + coeff * y1;
                out[i] = y1;
            }

            if ch == 0 {
                self.y1 = y1;
            }
        }
    }
}

// --- Biquad state ---

/// Per-channel biquad filter state (transposed direct form II).
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
            state: BiquadState,
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
                    state: BiquadState::new(),
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
                self.state = BiquadState::new();
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

                // Snapshot once, before the channel loop: see OnePole's
                // process() comment for why (read-back-inside-loop bug).
                let state_start = self.state;

                for ch in 0..output.num_channels() {
                    let mut state = state_start;
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

                    if ch == 0 {
                        self.state = state;
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
            state: BiquadState,
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
                    state: BiquadState::new(),
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
                self.state = BiquadState::new();
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

                // Snapshot once, before the channel loop: see OnePole's
                // process() comment for why (read-back-inside-loop bug).
                let state_start = self.state;

                for ch in 0..output.num_channels() {
                    let mut state = state_start;
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

                    if ch == 0 {
                        self.state = state;
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
    low: BiquadState,
    mid: BiquadState,
    high: BiquadState,
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
            low: BiquadState::new(),
            mid: BiquadState::new(),
            high: BiquadState::new(),
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
        self.low = BiquadState::new();
        self.mid = BiquadState::new();
        self.high = BiquadState::new();
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

        // Snapshot once, before the channel loop: see OnePole's process()
        // comment for why (read-back-inside-loop bug).
        let low_start = self.low;
        let mid_start = self.mid;
        let high_start = self.high;

        for ch in 0..output.num_channels() {
            let mut low = low_start;
            let mut mid = mid_start;
            let mut high = high_start;
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

            if ch == 0 {
                self.low = low;
                self.mid = mid;
                self.high = high;
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
    line: DelayLine,
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
            line: DelayLine::new(),
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
        self.line.resize(max_samples);
    }

    fn reset(&mut self) {
        self.line.clear();
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
        if self.line.is_empty() {
            return;
        }
        let max_delay = (self.line.len() - 1) as f32;

        // Every channel replays the shared delay line from the same cursor.
        let start_pos = self.line.write_pos();

        for ch in 0..output.num_channels() {
            self.line.set_write_pos(start_pos);
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let delay_time = read_input(delay_buf, ch, i, 0.01).max(0.0);
                let feedback = read_input(fb_buf, ch, i, 0.5).clamp(-0.999, 0.999);

                let delay_samples = (delay_time * self.sample_rate).min(max_delay).max(1.0);

                // IIR comb: output = input + feedback * delayed_output
                let delayed = self.line.read_interp(delay_samples);
                let y = in_ch[i] + feedback * delayed;

                self.line.write_and_advance(y);
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

impl Default for GVerb {
    fn default() -> Self {
        Self::new()
    }
}

impl GVerb {
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

            *out_sample = input * dry + signal * wet;
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
    /// **Independent per-channel look-ahead buffers** — deliberately NOT the
    /// single-shared-`DelayLine`-replayed-per-channel convention used by
    /// `Flanger`/`CombFilter`/`FeedbackDelay` elsewhere in this file. That
    /// convention is documented (`delayline.rs`) as an accepted compromise
    /// for effects like chorus where a little cross-channel bleed is
    /// inaudible. A limiter's ceiling is a hard numeric guarantee, not a
    /// vibe: sharing one buffer across channels means channel 1's look-ahead
    /// window reads back channel 0's *stale* samples for most of every
    /// block (the block size is smaller than the look-ahead, so channel 1
    /// never catches up to overwriting what channel 0 just wrote), silently
    /// swapping in a completely different signal's peak estimate. This was
    /// exactly the real defect a hot **stereo** test signal exposed that no
    /// mono synthetic signal could have (see `tests/ugens.rs`'s stereo
    /// cross-channel test and its own real-material-derived measurement
    /// note) — every channel needs to see only its own history.
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
