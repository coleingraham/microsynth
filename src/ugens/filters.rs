//! Filter UGens: OnePole, BiquadLPF, BiquadHPF, BiquadBPF.
//!
//! Biquad filters use the standard transposed direct form II implementation.
//! Coefficients are recalculated per-sample to support audio-rate modulation
//! of cutoff frequency and Q.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
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
