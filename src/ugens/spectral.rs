//! Spectral processing UGens.
//!
//! Each UGen internally manages an [`StftProcessor`] for overlap-add
//! STFT/ISTFT. From the audio graph's perspective these are ordinary
//! audio-rate-in/audio-rate-out nodes.
//!
//! - [`SpectralFreeze`]: Capture and hold the current spectrum on trigger.
//! - [`PitchShift`]: Phase vocoder pitch shifting.
//! - [`SpectralFilter`]: Frequency-domain bandpass/notch filter.
//! - [`SpectralGate`]: Zero bins below a magnitude threshold.
//! - [`SpectralBlur`]: Temporal magnitude smoothing.
//! - [`Convolution`]: FFT-based overlap-add convolution.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use crate::spectral::complex::Complex;
use crate::spectral::stft::StftProcessor;
use crate::spectral::window::WindowType;
use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::PI;

// ---------------------------------------------------------------------------
// SpectralFreeze
// ---------------------------------------------------------------------------

/// Captures the current spectrum and holds it when triggered.
///
/// Inputs: `in` (audio signal), `trig` (>0 on positive edge captures a new frame).
pub struct SpectralFreeze {
    stft: Option<StftProcessor>,
    frozen_spectrum: Vec<Complex>,
    is_frozen: bool,
    prev_trig: f32,
}

impl SpectralFreeze {
    pub fn new() -> Self {
        Self {
            stft: None,
            frozen_spectrum: Vec::new(),
            is_frozen: false,
            prev_trig: 0.0,
        }
    }
}

static FREEZE_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "trig", rate: Rate::Audio },
];
static FREEZE_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for SpectralFreeze {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "SpectralFreeze", inputs: &FREEZE_INPUTS, outputs: &FREEZE_OUTPUTS }
    }

    fn init(&mut self, _context: &ProcessContext) {
        let fft_size = 2048;
        let hop_size = 512;
        self.stft = Some(StftProcessor::new(fft_size, hop_size, WindowType::Hann));
        self.frozen_spectrum = vec![Complex::ZERO; fft_size];
    }

    fn reset(&mut self) {
        if let Some(ref mut stft) = self.stft {
            stft.reset();
        }
        self.frozen_spectrum.fill(Complex::ZERO);
        self.is_frozen = false;
        self.prev_trig = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let trig_buf = inputs.get(1).copied();
        let stft = self.stft.as_mut().unwrap();
        let fft_size = stft.fft_size();

        for ch in 0..output.num_channels() {
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                // Detect trigger positive edge.
                let trig_val = trig_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.0);

                if ch == 0 {
                    let is_positive_edge = trig_val > 0.0 && self.prev_trig <= 0.0;
                    if is_positive_edge {
                        self.is_frozen = true;
                    }
                    // Allow unfreezing when trigger goes negative.
                    if trig_val < 0.0 {
                        self.is_frozen = false;
                    }
                    self.prev_trig = trig_val;
                }

                if stft.push_sample(in_ch[i]) {
                    let spectrum = stft.analyze();

                    if self.is_frozen {
                        if self.frozen_spectrum.iter().all(|c| c.norm_sq() < 1e-20) {
                            // First capture — store current spectrum.
                            self.frozen_spectrum[..fft_size]
                                .copy_from_slice(&spectrum[..fft_size]);
                        } else {
                            // Subsequent frames — use frozen spectrum.
                            spectrum[..fft_size]
                                .copy_from_slice(&self.frozen_spectrum[..fft_size]);
                        }
                    } else {
                        // Not frozen — pass through, clear frozen for next trigger.
                        self.frozen_spectrum.fill(Complex::ZERO);
                    }

                    stft.synthesize();
                }
                out[i] = stft.pop_sample();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PitchShift (Phase Vocoder)
// ---------------------------------------------------------------------------

/// Phase vocoder pitch shifter.
///
/// Inputs: `in` (audio), `shift` (pitch ratio: 1.0 = no change, 2.0 = octave up, 0.5 = octave down).
pub struct PitchShift {
    stft: Option<StftProcessor>,
    prev_phase: Vec<f32>,
    synth_phase: Vec<f32>,
    sample_rate: f32,
}

impl PitchShift {
    pub fn new() -> Self {
        Self {
            stft: None,
            prev_phase: Vec::new(),
            synth_phase: Vec::new(),
            sample_rate: 44100.0,
        }
    }
}

static PITCH_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "shift", rate: Rate::Audio },
];
static PITCH_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for PitchShift {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "PitchShift", inputs: &PITCH_INPUTS, outputs: &PITCH_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        let fft_size = 4096;
        let hop_size = 1024;
        self.stft = Some(StftProcessor::new(fft_size, hop_size, WindowType::Hann));
        self.prev_phase = vec![0.0; fft_size];
        self.synth_phase = vec![0.0; fft_size];
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        if let Some(ref mut stft) = self.stft {
            stft.reset();
        }
        self.prev_phase.fill(0.0);
        self.synth_phase.fill(0.0);
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let shift_buf = inputs.get(1).copied();
        let stft = self.stft.as_mut().unwrap();
        let fft_size = stft.fft_size();
        let hop_size = stft.hop_size();
        let expected_phase_advance = 2.0 * PI * hop_size as f32 / fft_size as f32;

        for ch in 0..output.num_channels() {
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            // Read shift from first sample of the block (control-rate-ish).
            let shift = shift_buf
                .map(|b| b.channel(ch % b.num_channels()).samples()[0])
                .unwrap_or(1.0)
                .max(0.125)
                .min(8.0);

            for i in 0..out.len() {
                if stft.push_sample(in_ch[i]) {
                    let spectrum = stft.analyze();
                    let half = fft_size / 2 + 1;

                    // Phase vocoder pitch shifting.
                    // 1. Compute magnitude and instantaneous frequency for each bin.
                    // 2. Shift bins by the pitch ratio.
                    // 3. Accumulate synthesis phases.
                    let mut new_mags = vec![0.0f32; fft_size];
                    let mut new_freqs = vec![0.0f32; fft_size];

                    for k in 0..half {
                        let mag = spectrum[k].mag();
                        let phase = spectrum[k].phase();

                        // Phase difference from previous frame.
                        let mut dp = phase - self.prev_phase[k];
                        self.prev_phase[k] = phase;

                        // Remove expected phase advance.
                        dp -= k as f32 * expected_phase_advance;

                        // Wrap to [-pi, pi].
                        dp = dp - (dp / (2.0 * PI)).round() * 2.0 * PI;

                        // True frequency deviation in bins.
                        let true_freq = k as f32 + dp / expected_phase_advance;

                        // Shift the bin.
                        let new_bin = (true_freq * shift) as usize;
                        if new_bin < half {
                            new_mags[new_bin] += mag;
                            new_freqs[new_bin] = true_freq * shift;
                        }
                    }

                    // Reconstruct spectrum with accumulated synthesis phases.
                    for k in 0..half {
                        let phase_inc =
                            new_freqs[k] * expected_phase_advance;
                        self.synth_phase[k] += phase_inc;
                        spectrum[k] = Complex::from_polar(new_mags[k], self.synth_phase[k]);
                        // Mirror for conjugate symmetry.
                        if k > 0 && k < fft_size / 2 {
                            spectrum[fft_size - k] = spectrum[k].conj();
                        }
                    }

                    stft.synthesize();
                }
                out[i] = stft.pop_sample();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SpectralFilter
// ---------------------------------------------------------------------------

/// Frequency-domain bandpass/notch filter with Gaussian shape.
///
/// Inputs: `in` (audio), `freq` (center Hz), `bandwidth` (Hz), `gain` (linear multiplier).
pub struct SpectralFilter {
    stft: Option<StftProcessor>,
    sample_rate: f32,
}

impl SpectralFilter {
    pub fn new() -> Self {
        Self {
            stft: None,
            sample_rate: 44100.0,
        }
    }
}

static SFILTER_INPUTS: [InputSpec; 4] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "freq", rate: Rate::Audio },
    InputSpec { name: "bandwidth", rate: Rate::Audio },
    InputSpec { name: "gain", rate: Rate::Audio },
];
static SFILTER_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for SpectralFilter {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "SpectralFilter",
            inputs: &SFILTER_INPUTS,
            outputs: &SFILTER_OUTPUTS,
        }
    }

    fn init(&mut self, context: &ProcessContext) {
        let fft_size = 2048;
        let hop_size = 512;
        self.stft = Some(StftProcessor::new(fft_size, hop_size, WindowType::Hann));
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        if let Some(ref mut stft) = self.stft {
            stft.reset();
        }
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let freq_buf = inputs.get(1).copied();
        let bw_buf = inputs.get(2).copied();
        let gain_buf = inputs.get(3).copied();
        let stft = self.stft.as_mut().unwrap();
        let fft_size = stft.fft_size();

        for ch in 0..output.num_channels() {
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            // Read parameters from first sample (control-rate).
            let center_freq = freq_buf
                .map(|b| b.channel(ch % b.num_channels()).samples()[0])
                .unwrap_or(1000.0)
                .max(20.0);
            let bandwidth = bw_buf
                .map(|b| b.channel(ch % b.num_channels()).samples()[0])
                .unwrap_or(500.0)
                .max(1.0);
            let gain = gain_buf
                .map(|b| b.channel(ch % b.num_channels()).samples()[0])
                .unwrap_or(1.0);

            let bin_freq = self.sample_rate / fft_size as f32;
            let center_bin = center_freq / bin_freq;
            let bw_bins = bandwidth / bin_freq;
            // Gaussian sigma: bandwidth covers ~2 sigma.
            let sigma = bw_bins / 2.0;
            let sigma_sq_2 = 2.0 * sigma * sigma;

            for i in 0..out.len() {
                if stft.push_sample(in_ch[i]) {
                    let spectrum = stft.analyze();
                    let half = fft_size / 2 + 1;

                    for k in 0..half {
                        let dist = k as f32 - center_bin;
                        let gaussian = (-dist * dist / sigma_sq_2.max(0.001)).exp();
                        // Interpolate between 1.0 (no change) and gain at the peak.
                        let multiplier = 1.0 + gaussian * (gain - 1.0);
                        spectrum[k] = spectrum[k].scale(multiplier);
                        if k > 0 && k < fft_size / 2 {
                            spectrum[fft_size - k] = spectrum[k].conj();
                        }
                    }

                    stft.synthesize();
                }
                out[i] = stft.pop_sample();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SpectralGate
// ---------------------------------------------------------------------------

/// Frequency-domain noise gate. Zeros bins whose magnitude falls below
/// `threshold × max_magnitude` in each frame.
///
/// Inputs: `in` (audio), `threshold` (0.0–1.0).
pub struct SpectralGate {
    stft: Option<StftProcessor>,
}

impl SpectralGate {
    pub fn new() -> Self {
        Self { stft: None }
    }
}

static SGATE_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "threshold", rate: Rate::Audio },
];
static SGATE_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for SpectralGate {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "SpectralGate", inputs: &SGATE_INPUTS, outputs: &SGATE_OUTPUTS }
    }

    fn init(&mut self, _context: &ProcessContext) {
        let fft_size = 2048;
        let hop_size = 512;
        self.stft = Some(StftProcessor::new(fft_size, hop_size, WindowType::Hann));
    }

    fn reset(&mut self) {
        if let Some(ref mut stft) = self.stft {
            stft.reset();
        }
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let thresh_buf = inputs.get(1).copied();
        let stft = self.stft.as_mut().unwrap();
        let fft_size = stft.fft_size();

        for ch in 0..output.num_channels() {
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            let threshold = thresh_buf
                .map(|b| b.channel(ch % b.num_channels()).samples()[0])
                .unwrap_or(0.1)
                .clamp(0.0, 1.0);

            for i in 0..out.len() {
                if stft.push_sample(in_ch[i]) {
                    let spectrum = stft.analyze();
                    let half = fft_size / 2 + 1;

                    // Find max magnitude.
                    let mut max_mag = 0.0f32;
                    for k in 0..half {
                        let m = spectrum[k].mag();
                        if m > max_mag {
                            max_mag = m;
                        }
                    }

                    let gate_level = threshold * max_mag;

                    for k in 0..half {
                        if spectrum[k].mag() < gate_level {
                            spectrum[k] = Complex::ZERO;
                        }
                        if k > 0 && k < fft_size / 2 {
                            spectrum[fft_size - k] = spectrum[k].conj();
                        }
                    }

                    stft.synthesize();
                }
                out[i] = stft.pop_sample();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SpectralBlur
// ---------------------------------------------------------------------------

/// Temporal magnitude smoothing. Higher `blur` values smear the spectrum
/// over time, creating pad-like sustain from any input.
///
/// Inputs: `in` (audio), `blur` (0.0 = pass-through, 1.0 = infinite hold).
pub struct SpectralBlur {
    stft: Option<StftProcessor>,
    prev_magnitudes: Vec<f32>,
}

impl SpectralBlur {
    pub fn new() -> Self {
        Self {
            stft: None,
            prev_magnitudes: Vec::new(),
        }
    }
}

static SBLUR_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "blur", rate: Rate::Audio },
];
static SBLUR_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for SpectralBlur {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "SpectralBlur", inputs: &SBLUR_INPUTS, outputs: &SBLUR_OUTPUTS }
    }

    fn init(&mut self, _context: &ProcessContext) {
        let fft_size = 2048;
        let hop_size = 512;
        self.stft = Some(StftProcessor::new(fft_size, hop_size, WindowType::Hann));
        self.prev_magnitudes = vec![0.0; fft_size];
    }

    fn reset(&mut self) {
        if let Some(ref mut stft) = self.stft {
            stft.reset();
        }
        self.prev_magnitudes.fill(0.0);
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let blur_buf = inputs.get(1).copied();
        let stft = self.stft.as_mut().unwrap();
        let fft_size = stft.fft_size();

        for ch in 0..output.num_channels() {
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            let blur = blur_buf
                .map(|b| b.channel(ch % b.num_channels()).samples()[0])
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);

            for i in 0..out.len() {
                if stft.push_sample(in_ch[i]) {
                    let spectrum = stft.analyze();
                    let half = fft_size / 2 + 1;

                    for k in 0..half {
                        let current_mag = spectrum[k].mag();
                        let phase = spectrum[k].phase();

                        // Interpolate magnitude with previous frame.
                        let smoothed =
                            blur * self.prev_magnitudes[k] + (1.0 - blur) * current_mag;
                        self.prev_magnitudes[k] = smoothed;

                        // Reconstruct with current phase but smoothed magnitude.
                        spectrum[k] = Complex::from_polar(smoothed, phase);
                        if k > 0 && k < fft_size / 2 {
                            spectrum[fft_size - k] = spectrum[k].conj();
                        }
                    }

                    stft.synthesize();
                }
                out[i] = stft.pop_sample();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convolution (FFT overlap-add)
// ---------------------------------------------------------------------------

/// FFT-based convolution using overlap-add.
///
/// Convolves the input signal with a short impulse response (up to fft_size/2
/// samples). The IR must be loaded via [`Convolution::set_ir`] before use.
///
/// Inputs: `in` (audio), `mix` (dry/wet 0.0–1.0).
pub struct Convolution {
    fft_size: usize,
    /// Pre-computed FFT of the impulse response.
    ir_spectrum: Vec<Complex>,
    /// Input accumulation ring buffer.
    input_ring: Vec<f32>,
    input_write_pos: usize,
    /// Output overlap-add ring buffer.
    output_ring: Vec<f32>,
    output_read_pos: usize,
    /// FFT scratch buffer.
    fft_buf: Vec<Complex>,
    /// Samples until next processing frame.
    samples_until_frame: usize,
    /// Block size for processing (fft_size / 2).
    block_len: usize,
    /// Whether the IR has been set.
    ir_loaded: bool,
}

impl Convolution {
    pub fn new() -> Self {
        Self {
            fft_size: 0,
            ir_spectrum: Vec::new(),
            input_ring: Vec::new(),
            input_write_pos: 0,
            output_ring: Vec::new(),
            output_read_pos: 0,
            fft_buf: Vec::new(),
            samples_until_frame: 0,
            block_len: 0,
            ir_loaded: false,
        }
    }

    /// Load an impulse response. Must be called before or during `init`.
    ///
    /// The IR will be zero-padded to `fft_size` and pre-transformed.
    pub fn set_ir(&mut self, ir: &[f32]) {
        let fft_size = self.fft_size;
        if fft_size == 0 {
            return;
        }
        self.ir_spectrum.resize(fft_size, Complex::ZERO);
        for (i, c) in self.ir_spectrum.iter_mut().enumerate() {
            *c = if i < ir.len() {
                Complex::new(ir[i], 0.0)
            } else {
                Complex::ZERO
            };
        }
        crate::spectral::fft::fft(&mut self.ir_spectrum);
        self.ir_loaded = true;
    }
}

static CONV_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "mix", rate: Rate::Audio },
];
static CONV_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Convolution {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Convolution", inputs: &CONV_INPUTS, outputs: &CONV_OUTPUTS }
    }

    fn init(&mut self, _context: &ProcessContext) {
        self.fft_size = 4096;
        self.block_len = self.fft_size / 2;
        self.input_ring = vec![0.0; self.fft_size];
        self.input_write_pos = 0;
        self.output_ring = vec![0.0; self.fft_size];
        self.output_read_pos = 0;
        self.fft_buf = vec![Complex::ZERO; self.fft_size];
        self.ir_spectrum = vec![Complex::ZERO; self.fft_size];
        self.samples_until_frame = self.block_len;
    }

    fn reset(&mut self) {
        self.input_ring.fill(0.0);
        self.input_write_pos = 0;
        self.output_ring.fill(0.0);
        self.output_read_pos = 0;
        self.fft_buf.fill(Complex::ZERO);
        self.samples_until_frame = self.block_len;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let mix_buf = inputs.get(1).copied();
        let fft_size = self.fft_size;

        if fft_size == 0 || !self.ir_loaded {
            // No IR loaded — pass through dry signal.
            for ch in 0..output.num_channels() {
                let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
                let out = output.channel_mut(ch).samples_mut();
                out[..in_ch.len()].copy_from_slice(in_ch);
            }
            return;
        }

        for ch in 0..output.num_channels() {
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            let mix = mix_buf
                .map(|b| b.channel(ch % b.num_channels()).samples()[0])
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);

            for i in 0..out.len() {
                let dry = in_ch[i];

                // Accumulate input.
                self.input_ring[self.input_write_pos] = dry;
                self.input_write_pos = (self.input_write_pos + 1) % fft_size;
                self.samples_until_frame -= 1;

                if self.samples_until_frame == 0 {
                    self.samples_until_frame = self.block_len;

                    // Copy last fft_size samples (zero-padded block) into FFT buffer.
                    for j in 0..fft_size {
                        let idx = (self.input_write_pos + j) % fft_size;
                        self.fft_buf[j] = Complex::new(self.input_ring[idx], 0.0);
                    }

                    // Forward FFT of input.
                    crate::spectral::fft::fft(&mut self.fft_buf);

                    // Multiply with IR spectrum.
                    for j in 0..fft_size {
                        self.fft_buf[j] = self.fft_buf[j] * self.ir_spectrum[j];
                    }

                    // Inverse FFT.
                    crate::spectral::fft::ifft(&mut self.fft_buf);

                    // Overlap-add the result.
                    let write_start = self.output_read_pos;
                    for j in 0..fft_size {
                        let idx = (write_start + j) % fft_size;
                        self.output_ring[idx] += self.fft_buf[j].re;
                    }
                }

                // Read output.
                let wet = self.output_ring[self.output_read_pos];
                self.output_ring[self.output_read_pos] = 0.0;
                self.output_read_pos = (self.output_read_pos + 1) % fft_size;

                out[i] = dry * (1.0 - mix) + wet * mix;
            }
        }
    }
}
