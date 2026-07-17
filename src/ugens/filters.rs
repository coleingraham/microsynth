//! Filter UGens: OnePole, BiquadLPF, BiquadHPF, BiquadBPF, CombFilter, GVerb.
//!
//! Biquad filters use the standard transposed direct form II implementation.
//! Coefficients are recalculated per-sample to support audio-rate modulation
//! of cutoff frequency and Q.

use crate::buffer::{AudioBuffer, channel_wrapped, read_input};
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
        inputs = ["in", "coeff"],
        outputs = ["out"]
    );

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        self.y1 = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let coeff_buf = inputs.get(1).copied();

        for ch in 0..output.num_channels() {
            let mut y1 = self.y1;
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
#[derive(Clone, Copy)]
struct BiquadState {
    z1: f32,
    z2: f32,
}

impl BiquadState {
    fn new() -> Self {
        BiquadState { z1: 0.0, z2: 0.0 }
    }

    /// Process one sample through the biquad.
    #[inline]
    fn tick(&mut self, x: f32, b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> f32 {
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
#[inline]
fn biquad_bpf_coeffs(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
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
                inputs = ["in", "freq", "q"],
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
                inputs: &[&AudioBuffer],
                output: &mut AudioBuffer,
            ) {
                let in_buf = inputs[0];
                let freq_buf = inputs.get(1).copied();
                let q_buf = inputs.get(2).copied();
                let sr = self.sample_rate;
                let nyquist = sr * 0.5;

                for ch in 0..output.num_channels() {
                    let mut state = self.state;
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
        inputs = ["in", "delay", "feedback"],
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
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let delay_buf = inputs.get(1).copied();
        let fb_buf = inputs.get(2).copied();
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
        inputs = ["in", "roomsize", "damping", "wet", "dry"],
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
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let params = GVerbParams {
            roomsize: inputs.get(1).copied(),
            damping: inputs.get(2).copied(),
            wet: inputs.get(3).copied(),
            dry: inputs.get(4).copied(),
        };

        let in_ch = inputs[0].channel(0).samples();

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
        inputs = [
            "in",
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
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let sc_buf = inputs.get(1).copied().unwrap_or(in_buf);
        let thresh_buf = inputs.get(2).copied();
        let ratio_buf = inputs.get(3).copied();
        let attack_buf = inputs.get(4).copied();
        let release_buf = inputs.get(5).copied();
        let makeup_buf = inputs.get(6).copied();

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
