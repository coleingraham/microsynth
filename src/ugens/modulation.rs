//! Modulation UGens: Chorus, Flanger, Phaser.
//!
//! Time-modulated delay effects for spatial width and movement.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use alloc::vec::Vec;
use core::f32::consts::TAU;

// --- Chorus ---

/// Stereo chorus effect using modulated delay lines.
///
/// Two LFO-modulated delay taps (one per stereo channel) with slightly
/// different phase offsets create width and shimmer. The modulated delay
/// time oscillates around a center delay, producing pitch/time variation
/// that simulates multiple detuned voices.
///
/// Inputs:
/// - `in`: audio signal
/// - `rate`: LFO rate in Hz (default 1.0). Typical range 0.1–5.0 Hz.
/// - `depth`: modulation depth in seconds (default 0.003). Controls how
///   far the delay time swings from center. 0.001–0.010 is typical.
/// - `mix`: dry/wet blend (default 0.5). 0.0 = dry, 1.0 = fully wet.
///
/// Output: 2-channel stereo signal. Left and right taps use LFO phases
/// offset by 90° for stereo decorrelation.
pub struct Chorus {
    buffer: Vec<f32>,
    write_pos: usize,
    lfo_phase: f32,
    sample_rate: f32,
}

impl Chorus {
    pub fn new() -> Self {
        Chorus {
            buffer: Vec::new(),
            write_pos: 0,
            lfo_phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

/// Center delay for chorus in seconds. Short enough to avoid distinct echo.
const CHORUS_CENTER_DELAY: f32 = 0.007;
/// Maximum delay buffer in seconds (center + max depth + margin).
const CHORUS_MAX_DELAY: f32 = 0.040;

static CHORUS_INPUTS: [InputSpec; 4] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "rate", rate: Rate::Audio },
    InputSpec { name: "depth", rate: Rate::Audio },
    InputSpec { name: "mix", rate: Rate::Audio },
];
static CHORUS_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Chorus {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Chorus", inputs: &CHORUS_INPUTS, outputs: &CHORUS_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (CHORUS_MAX_DELAY * context.sample_rate) as usize + 2;
        self.buffer.resize(max_samples, 0.0);
        self.write_pos = 0;
        self.lfo_phase = 0.0;
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.lfo_phase = 0.0;
    }

    /// Chorus always produces 2 output channels (stereo).
    fn output_channels(&self, _input_channels: &[usize]) -> usize {
        2
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let rate_buf = inputs.get(1).copied();
        let depth_buf = inputs.get(2).copied();
        let mix_buf = inputs.get(3).copied();
        let buf_len = self.buffer.len();
        if buf_len == 0 {
            return;
        }
        let max_delay_samples = (buf_len - 2) as f32;
        let inv_sr = 1.0 / self.sample_rate;

        let block_size = output.block_size();
        let mut write_pos = self.write_pos;
        let mut lfo_phase = self.lfo_phase;

        for i in 0..block_size {
            let x = in_buf.channel(0).samples()[i];
            let rate = rate_buf
                .map(|b| b.channel(0).samples()[i])
                .unwrap_or(1.0)
                .max(0.01);
            let depth = depth_buf
                .map(|b| b.channel(0).samples()[i])
                .unwrap_or(0.003)
                .clamp(0.0, 0.020);
            let mix = mix_buf
                .map(|b| b.channel(0).samples()[i])
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);

            // Write input to buffer
            self.buffer[write_pos] = x;

            // Two LFO taps at 90° phase offset for stereo
            let lfo_l = (lfo_phase * TAU).sin();
            let lfo_r = ((lfo_phase + 0.25) * TAU).sin();

            let delay_l = ((CHORUS_CENTER_DELAY + depth * lfo_l) * self.sample_rate)
                .clamp(1.0, max_delay_samples);
            let delay_r = ((CHORUS_CENTER_DELAY + depth * lfo_r) * self.sample_rate)
                .clamp(1.0, max_delay_samples);

            // Read left tap with linear interpolation
            let dl_int = delay_l as usize;
            let dl_frac = delay_l - dl_int as f32;
            let ra_l = (write_pos + buf_len - dl_int) % buf_len;
            let rb_l = (write_pos + buf_len - dl_int - 1) % buf_len;
            let wet_l = self.buffer[ra_l] + dl_frac * (self.buffer[rb_l] - self.buffer[ra_l]);

            // Read right tap with linear interpolation
            let dr_int = delay_r as usize;
            let dr_frac = delay_r - dr_int as f32;
            let ra_r = (write_pos + buf_len - dr_int) % buf_len;
            let rb_r = (write_pos + buf_len - dr_int - 1) % buf_len;
            let wet_r = self.buffer[ra_r] + dr_frac * (self.buffer[rb_r] - self.buffer[ra_r]);

            // Mix
            output.channel_mut(0).samples_mut()[i] = (1.0 - mix) * x + mix * wet_l;
            output.channel_mut(1).samples_mut()[i] = (1.0 - mix) * x + mix * wet_r;

            lfo_phase += rate * inv_sr;
            lfo_phase -= lfo_phase.floor();
            write_pos = (write_pos + 1) % buf_len;
        }

        self.write_pos = write_pos;
        self.lfo_phase = lfo_phase;
    }
}

// --- Flanger ---

/// Flanger effect using a very short modulated delay with feedback.
///
/// Similar to chorus but with shorter delay times (0.1–5ms) and feedback,
/// producing comb-filter sweeps with a metallic, jet-engine character.
///
/// Inputs:
/// - `in`: audio signal
/// - `rate`: LFO rate in Hz (default 0.3)
/// - `depth`: modulation depth in seconds (default 0.002)
/// - `feedback`: feedback amount (default 0.5, range -0.95 to 0.95)
/// - `mix`: dry/wet blend (default 0.5)
pub struct Flanger {
    buffer: Vec<f32>,
    write_pos: usize,
    lfo_phase: f32,
    sample_rate: f32,
}

impl Flanger {
    pub fn new() -> Self {
        Flanger {
            buffer: Vec::new(),
            write_pos: 0,
            lfo_phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

const FLANGER_MAX_DELAY: f32 = 0.020;

static FLANGER_INPUTS: [InputSpec; 5] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "rate", rate: Rate::Audio },
    InputSpec { name: "depth", rate: Rate::Audio },
    InputSpec { name: "feedback", rate: Rate::Audio },
    InputSpec { name: "mix", rate: Rate::Audio },
];
static FLANGER_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Flanger {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Flanger", inputs: &FLANGER_INPUTS, outputs: &FLANGER_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (FLANGER_MAX_DELAY * context.sample_rate) as usize + 2;
        self.buffer.resize(max_samples, 0.0);
        self.write_pos = 0;
        self.lfo_phase = 0.0;
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.lfo_phase = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let rate_buf = inputs.get(1).copied();
        let depth_buf = inputs.get(2).copied();
        let fb_buf = inputs.get(3).copied();
        let mix_buf = inputs.get(4).copied();
        let buf_len = self.buffer.len();
        if buf_len == 0 {
            return;
        }
        let max_delay_samples = (buf_len - 2) as f32;
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut write_pos = self.write_pos;
            let mut lfo_phase = self.lfo_phase;
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let rate = rate_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.3)
                    .max(0.01);
                let depth = depth_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.002)
                    .clamp(0.0, 0.010);
                let feedback = fb_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.5)
                    .clamp(-0.95, 0.95);
                let mix = mix_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);

                // LFO-modulated delay time (unipolar: 0 to depth)
                let lfo = (lfo_phase * TAU).sin() * 0.5 + 0.5;
                let delay_secs = 0.0005 + depth * lfo; // min 0.5ms
                let delay_samples = (delay_secs * self.sample_rate)
                    .clamp(1.0, max_delay_samples);

                // Read with linear interpolation
                let d_int = delay_samples as usize;
                let d_frac = delay_samples - d_int as f32;
                let ra = (write_pos + buf_len - d_int) % buf_len;
                let rb = (write_pos + buf_len - d_int - 1) % buf_len;
                let delayed = self.buffer[ra] + d_frac * (self.buffer[rb] - self.buffer[ra]);

                // Write input + feedback into buffer
                self.buffer[write_pos] = x + feedback * delayed;

                out[i] = (1.0 - mix) * x + mix * delayed;

                lfo_phase += rate * inv_sr;
                lfo_phase -= lfo_phase.floor();
                write_pos = (write_pos + 1) % buf_len;
            }

            if ch == 0 {
                self.write_pos = write_pos;
                self.lfo_phase = lfo_phase;
            }
        }
    }
}

// --- Phaser ---

/// Phaser effect using cascaded allpass filters with LFO-modulated center frequency.
///
/// Uses 4 first-order allpass stages whose corner frequency sweeps via LFO.
/// The interaction of multiple notches in the frequency response creates the
/// characteristic swooshing, swirling sound.
///
/// Inputs:
/// - `in`: audio signal
/// - `rate`: LFO rate in Hz (default 0.4)
/// - `depth`: modulation depth (default 0.7, range 0.0–1.0)
/// - `feedback`: output-to-input feedback (default 0.3, range -0.95 to 0.95)
/// - `mix`: dry/wet blend (default 0.5)
pub struct Phaser {
    /// Allpass filter state for 4 stages (per-channel, but single-channel for simplicity).
    ap_state: [f32; 4],
    lfo_phase: f32,
    sample_rate: f32,
    feedback_sample: f32,
}

impl Phaser {
    pub fn new() -> Self {
        Phaser {
            ap_state: [0.0; 4],
            lfo_phase: 0.0,
            sample_rate: 44100.0,
            feedback_sample: 0.0,
        }
    }
}

static PHASER_INPUTS: [InputSpec; 5] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "rate", rate: Rate::Audio },
    InputSpec { name: "depth", rate: Rate::Audio },
    InputSpec { name: "feedback", rate: Rate::Audio },
    InputSpec { name: "mix", rate: Rate::Audio },
];
static PHASER_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Phaser {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Phaser", inputs: &PHASER_INPUTS, outputs: &PHASER_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.ap_state = [0.0; 4];
        self.lfo_phase = 0.0;
        self.feedback_sample = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let rate_buf = inputs.get(1).copied();
        let depth_buf = inputs.get(2).copied();
        let fb_buf = inputs.get(3).copied();
        let mix_buf = inputs.get(4).copied();
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut ap_state = self.ap_state;
            let mut lfo_phase = self.lfo_phase;
            let mut fb_sample = self.feedback_sample;
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let rate = rate_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.4)
                    .max(0.01);
                let depth = depth_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.7)
                    .clamp(0.0, 1.0);
                let feedback = fb_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.3)
                    .clamp(-0.95, 0.95);
                let mix = mix_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);

                // LFO sweeps allpass corner frequency in log space
                // Range: 200 Hz to 4000 Hz
                let lfo = (lfo_phase * TAU).sin() * 0.5 + 0.5; // 0..1
                let min_freq: f32 = 200.0;
                let max_freq: f32 = 4000.0;
                let sweep_freq = min_freq * (max_freq / min_freq).powf(lfo * depth);

                // First-order allpass coefficient from corner frequency
                // a = (tan(pi*f/sr) - 1) / (tan(pi*f/sr) + 1)
                let t = (core::f32::consts::PI * sweep_freq * inv_sr).tan();
                let coeff = (t - 1.0) / (t + 1.0);

                // Feed input with feedback
                let mut signal = x + feedback * fb_sample;

                // Cascade 4 allpass stages
                for stage in 0..4 {
                    let y = coeff * signal + ap_state[stage];
                    ap_state[stage] = signal - coeff * y;
                    signal = y;
                }

                fb_sample = signal;
                out[i] = (1.0 - mix) * x + mix * signal;

                lfo_phase += rate * inv_sr;
                lfo_phase -= lfo_phase.floor();
            }

            if ch == 0 {
                self.ap_state = ap_state;
                self.lfo_phase = lfo_phase;
                self.feedback_sample = fb_sample;
            }
        }
    }
}
