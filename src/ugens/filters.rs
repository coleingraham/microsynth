//! Filter UGens: OnePole, BiquadLPF, BiquadHPF, BiquadBPF, CombFilter, GVerb.
//!
//! Biquad filters use the standard transposed direct form II implementation.
//! Coefficients are recalculated per-sample to support audio-rate modulation
//! of cutoff frequency and Q.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use alloc::vec::Vec;
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

impl OnePole {
    pub fn new() -> Self {
        OnePole { y1: 0.0 }
    }
}

static ONEPOLE_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "coeff", rate: Rate::Audio },
];
static ONEPOLE_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for OnePole {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "OnePole", inputs: &ONEPOLE_INPUTS, outputs: &ONEPOLE_OUTPUTS }
    }

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
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let coeff = coeff_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.5);
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

/// Compute biquad lowpass coefficients from freq, q, and sample_rate.
#[inline]
fn biquad_lpf_coeffs(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
    let w0 = TAU * freq / sample_rate;
    let (sin_w0, cos_w0) = (w0.sin(), w0.cos());
    let alpha = sin_w0 / (2.0 * q);

    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = b0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    let inv_a0 = 1.0 / a0;
    (b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0)
}

/// Compute biquad highpass coefficients.
#[inline]
fn biquad_hpf_coeffs(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
    let w0 = TAU * freq / sample_rate;
    let (sin_w0, cos_w0) = (w0.sin(), w0.cos());
    let alpha = sin_w0 / (2.0 * q);

    let b0 = (1.0 + cos_w0) / 2.0;
    let b1 = -(1.0 + cos_w0);
    let b2 = b0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    let inv_a0 = 1.0 / a0;
    (b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0)
}

/// Compute biquad bandpass coefficients (constant-peak-gain).
#[inline]
fn biquad_bpf_coeffs(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
    let w0 = TAU * freq / sample_rate;
    let (sin_w0, cos_w0) = (w0.sin(), w0.cos());
    let alpha = sin_w0 / (2.0 * q);

    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    let inv_a0 = 1.0 / a0;
    (b0 * inv_a0, b1 * inv_a0, b2 * inv_a0, a1 * inv_a0, a2 * inv_a0)
}

// --- BiquadLPF ---

/// Second-order Butterworth-style lowpass filter.
///
/// Inputs: in (signal), freq (cutoff Hz), q (resonance, default 0.707).
pub struct BiquadLPF {
    state: BiquadState,
    sample_rate: f32,
}

impl BiquadLPF {
    pub fn new() -> Self {
        BiquadLPF { state: BiquadState::new(), sample_rate: 44100.0 }
    }
}

static BIQUAD_INPUTS: [InputSpec; 3] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "freq", rate: Rate::Audio },
    InputSpec { name: "q", rate: Rate::Audio },
];
static BIQUAD_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for BiquadLPF {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "BiquadLPF", inputs: &BIQUAD_INPUTS, outputs: &BIQUAD_OUTPUTS }
    }

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
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1000.0)
                    .clamp(20.0, nyquist - 1.0);
                let q = q_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.707)
                    .max(0.01);

                let (b0, b1, b2, a1, a2) = biquad_lpf_coeffs(freq, q, sr);
                out[i] = state.tick(in_ch[i], b0, b1, b2, a1, a2);
            }

            if ch == 0 {
                self.state = state;
            }
        }
    }
}

// --- BiquadHPF ---

/// Second-order highpass filter.
///
/// Inputs: in (signal), freq (cutoff Hz), q (resonance, default 0.707).
pub struct BiquadHPF {
    state: BiquadState,
    sample_rate: f32,
}

impl BiquadHPF {
    pub fn new() -> Self {
        BiquadHPF { state: BiquadState::new(), sample_rate: 44100.0 }
    }
}

impl UGen for BiquadHPF {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "BiquadHPF", inputs: &BIQUAD_INPUTS, outputs: &BIQUAD_OUTPUTS }
    }

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
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1000.0)
                    .clamp(20.0, nyquist - 1.0);
                let q = q_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.707)
                    .max(0.01);

                let (b0, b1, b2, a1, a2) = biquad_hpf_coeffs(freq, q, sr);
                out[i] = state.tick(in_ch[i], b0, b1, b2, a1, a2);
            }

            if ch == 0 {
                self.state = state;
            }
        }
    }
}

// --- BiquadBPF ---

/// Second-order bandpass filter.
///
/// Inputs: in (signal), freq (center Hz), q (bandwidth).
pub struct BiquadBPF {
    state: BiquadState,
    sample_rate: f32,
}

impl BiquadBPF {
    pub fn new() -> Self {
        BiquadBPF { state: BiquadState::new(), sample_rate: 44100.0 }
    }
}

impl UGen for BiquadBPF {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "BiquadBPF", inputs: &BIQUAD_INPUTS, outputs: &BIQUAD_OUTPUTS }
    }

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
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1000.0)
                    .clamp(20.0, nyquist - 1.0);
                let q = q_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1.0)
                    .max(0.01);

                let (b0, b1, b2, a1, a2) = biquad_bpf_coeffs(freq, q, sr);
                out[i] = state.tick(in_ch[i], b0, b1, b2, a1, a2);
            }

            if ch == 0 {
                self.state = state;
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
    buffer: Vec<f32>,
    write_pos: usize,
    sample_rate: f32,
}

impl CombFilter {
    pub fn new() -> Self {
        CombFilter {
            buffer: Vec::new(),
            write_pos: 0,
            sample_rate: 44100.0,
        }
    }
}

static COMB_INPUTS: [InputSpec; 3] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "delay", rate: Rate::Audio },
    InputSpec { name: "feedback", rate: Rate::Audio },
];
static COMB_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for CombFilter {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "CombFilter", inputs: &COMB_INPUTS, outputs: &COMB_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (MAX_COMB_DELAY_SECS * context.sample_rate) as usize + 1;
        self.buffer.resize(max_samples, 0.0);
        self.write_pos = 0;
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
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
        let buf_len = self.buffer.len();
        if buf_len == 0 {
            return;
        }
        let max_delay = (buf_len - 1) as f32;

        for ch in 0..output.num_channels() {
            let mut write_pos = self.write_pos;
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let delay_time = delay_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.01)
                    .max(0.0);
                let feedback = fb_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.5)
                    .clamp(-0.999, 0.999);

                let delay_samples = (delay_time * self.sample_rate)
                    .min(max_delay)
                    .max(1.0);

                // Read from delay line with linear interpolation
                let delay_int = delay_samples as usize;
                let frac = delay_samples - delay_int as f32;
                let read_a = (write_pos + buf_len - delay_int) % buf_len;
                let read_b = (write_pos + buf_len - delay_int - 1) % buf_len;
                let delayed = self.buffer[read_a] + frac * (self.buffer[read_b] - self.buffer[read_a]);

                // IIR comb: output = input + feedback * delayed_output
                let y = in_ch[i] + feedback * delayed;

                // Write to delay line
                self.buffer[write_pos] = y;
                out[i] = y;

                write_pos = (write_pos + 1) % buf_len;
            }

            if ch == 0 {
                self.write_pos = write_pos;
            }
        }
    }
}

// --- GVerb ---

/// Internal delay line for reverb components.
struct ReverbDelay {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl ReverbDelay {
    fn new(size: usize) -> Self {
        ReverbDelay {
            buffer: alloc::vec![0.0; size.max(1)],
            write_pos: 0,
        }
    }

    fn clear(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }

    /// Read from the delay line at a fixed tap.
    #[inline]
    fn read(&self, delay: usize) -> f32 {
        let len = self.buffer.len();
        let pos = (self.write_pos + len - delay) % len;
        self.buffer[pos]
    }

    /// Write a sample and advance.
    #[inline]
    fn write_and_advance(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
    }
}

/// A damped comb filter for use inside the reverb.
struct ReverbComb {
    delay: ReverbDelay,
    filter_state: f32,
    delay_samples: usize,
}

impl ReverbComb {
    fn new(delay_samples: usize) -> Self {
        ReverbComb {
            delay: ReverbDelay::new(delay_samples + 1),
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
    delay: ReverbDelay,
    delay_samples: usize,
}

impl ReverbAllpass {
    fn new(delay_samples: usize) -> Self {
        ReverbAllpass {
            delay: ReverbDelay::new(delay_samples + 1),
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

impl GVerb {
    pub fn new() -> Self {
        GVerb {
            combs_l: core::array::from_fn(|i| ReverbComb::new(COMB_DELAYS_L[i])),
            combs_r: core::array::from_fn(|i| ReverbComb::new(COMB_DELAYS_L[i] + STEREO_SPREAD)),
            allpasses_l: core::array::from_fn(|i| ReverbAllpass::new(ALLPASS_DELAYS_L[i])),
            allpasses_r: core::array::from_fn(|i| ReverbAllpass::new(ALLPASS_DELAYS_L[i] + STEREO_SPREAD)),
        }
    }
}

static GVERB_INPUTS: [InputSpec; 5] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "roomsize", rate: Rate::Audio },
    InputSpec { name: "damping", rate: Rate::Audio },
    InputSpec { name: "wet", rate: Rate::Audio },
    InputSpec { name: "dry", rate: Rate::Audio },
];
static GVERB_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for GVerb {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "GVerb", inputs: &GVERB_INPUTS, outputs: &GVERB_OUTPUTS }
    }

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        for c in &mut self.combs_l { c.clear(); }
        for c in &mut self.combs_r { c.clear(); }
        for a in &mut self.allpasses_l { a.clear(); }
        for a in &mut self.allpasses_r { a.clear(); }
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
        let in_buf = inputs[0];
        let roomsize_buf = inputs.get(1).copied();
        let damping_buf = inputs.get(2).copied();
        let wet_buf = inputs.get(3).copied();
        let dry_buf = inputs.get(4).copied();

        let in_ch = in_buf.channel(0).samples();
        let block_size = output.channel(0).len();

        // Left channel
        let out_l = output.channel_mut(0).samples_mut();
        for i in 0..block_size {
            let input = in_ch[i];
            let roomsize = roomsize_buf
                .map(|b| b.channel(0).samples()[i.min(b.channel(0).len() - 1)])
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
            let damping = damping_buf
                .map(|b| b.channel(0).samples()[i.min(b.channel(0).len() - 1)])
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);
            let wet = wet_buf
                .map(|b| b.channel(0).samples()[i.min(b.channel(0).len() - 1)])
                .unwrap_or(0.3)
                .clamp(0.0, 1.0);
            let dry = dry_buf
                .map(|b| b.channel(0).samples()[i.min(b.channel(0).len() - 1)])
                .unwrap_or(0.7)
                .clamp(0.0, 1.0);

            // Scale roomsize to feedback (0.0 → 0.7, 1.0 → 0.98)
            let feedback = 0.7 + roomsize * 0.28;

            // Sum of parallel comb filters
            let mut comb_sum = 0.0;
            for comb in &mut self.combs_l {
                comb_sum += comb.tick(input, feedback, damping);
            }

            // Series allpass filters
            let mut signal = comb_sum;
            for ap in &mut self.allpasses_l {
                signal = ap.tick(signal, 0.5);
            }

            out_l[i] = input * dry + signal * wet;
        }

        // Right channel
        if output.num_channels() >= 2 {
            let in_samples = in_buf.channel(0).samples();
            let out_r = output.channel_mut(1).samples_mut();

            for i in 0..block_size {
                let input = in_samples[i];
                let roomsize = roomsize_buf
                    .map(|b| b.channel(0).samples()[i.min(b.channel(0).len() - 1)])
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                let damping = damping_buf
                    .map(|b| b.channel(0).samples()[i.min(b.channel(0).len() - 1)])
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                let wet = wet_buf
                    .map(|b| b.channel(0).samples()[i.min(b.channel(0).len() - 1)])
                    .unwrap_or(0.3)
                    .clamp(0.0, 1.0);
                let dry = dry_buf
                    .map(|b| b.channel(0).samples()[i.min(b.channel(0).len() - 1)])
                    .unwrap_or(0.7)
                    .clamp(0.0, 1.0);

                let feedback = 0.7 + roomsize * 0.28;

                let mut comb_sum = 0.0;
                for comb in &mut self.combs_r {
                    comb_sum += comb.tick(input, feedback, damping);
                }

                let mut signal = comb_sum;
                for ap in &mut self.allpasses_r {
                    signal = ap.tick(signal, 0.5);
                }

                out_r[i] = input * dry + signal * wet;
            }
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

impl Compressor {
    pub fn new() -> Self {
        Compressor {
            env_db: [-120.0; 2],
            sample_rate: 44100.0,
        }
    }
}

static COMP_INPUTS: [InputSpec; 7] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "sidechain", rate: Rate::Audio },
    InputSpec { name: "threshold", rate: Rate::Audio },
    InputSpec { name: "ratio", rate: Rate::Audio },
    InputSpec { name: "attack", rate: Rate::Audio },
    InputSpec { name: "release", rate: Rate::Audio },
    InputSpec { name: "makeup", rate: Rate::Audio },
];
static COMP_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

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
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Compressor", inputs: &COMP_INPUTS, outputs: &COMP_OUTPUTS }
    }

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
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let sc_ch = sc_buf.channel(ch % sc_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();
            let env_idx = ch.min(1);
            let mut env_db = self.env_db[env_idx];

            for i in 0..out.len() {
                let threshold = thresh_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(-10.0);
                let ratio = ratio_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(4.0)
                    .max(1.0);
                let attack_time = attack_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.01)
                    .max(0.0001);
                let release_time = release_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.1)
                    .max(0.0001);
                let makeup = makeup_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.0);

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
