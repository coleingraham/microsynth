//! Modulation UGens: Chorus, Flanger, Phaser.
//!
//! Time-modulated delay effects for spatial width and movement.

use crate::buffer::{AudioBuffer, channel_wrapped, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use crate::ugens::delayline::DelayLine;
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
    line: DelayLine,
    lfo_phase: f32,
    sample_rate: f32,
}

impl Default for Chorus {
    fn default() -> Self {
        Self::new()
    }
}

impl Chorus {
    pub fn new() -> Self {
        Chorus {
            line: DelayLine::new(),
            lfo_phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

/// Center delay for chorus in seconds. Short enough to avoid distinct echo.
const CHORUS_CENTER_DELAY: f32 = 0.007;
/// Maximum delay buffer in seconds (center + max depth + margin).
const CHORUS_MAX_DELAY: f32 = 0.040;

impl UGen for Chorus {
    ugen_spec!(
        "Chorus",
        category = Effect,
        inputs = ["in", "rate", "depth", "mix"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (CHORUS_MAX_DELAY * context.sample_rate) as usize + 2;
        self.line.resize(max_samples);
        self.lfo_phase = 0.0;
    }

    fn reset(&mut self) {
        self.line.clear();
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
        if self.line.is_empty() {
            return;
        }
        let max_delay_samples = (self.line.len() - 2) as f32;
        let inv_sr = 1.0 / self.sample_rate;

        let block_size = output.block_size();
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

            // Write input at the cursor; the taps below read behind it.
            self.line.write(x);

            // Two LFO taps at 90° phase offset for stereo
            let lfo_l = (lfo_phase * TAU).sin();
            let lfo_r = ((lfo_phase + 0.25) * TAU).sin();

            let delay_l = ((CHORUS_CENTER_DELAY + depth * lfo_l) * self.sample_rate)
                .clamp(1.0, max_delay_samples);
            let delay_r = ((CHORUS_CENTER_DELAY + depth * lfo_r) * self.sample_rate)
                .clamp(1.0, max_delay_samples);

            let wet_l = self.line.read_interp(delay_l);
            let wet_r = self.line.read_interp(delay_r);

            // Mix
            output.channel_mut(0).samples_mut()[i] = (1.0 - mix) * x + mix * wet_l;
            output.channel_mut(1).samples_mut()[i] = (1.0 - mix) * x + mix * wet_r;

            lfo_phase += rate * inv_sr;
            lfo_phase -= lfo_phase.floor();
            self.line.advance();
        }

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
    line: DelayLine,
    lfo_phase: f32,
    sample_rate: f32,
}

impl Default for Flanger {
    fn default() -> Self {
        Self::new()
    }
}

impl Flanger {
    pub fn new() -> Self {
        Flanger {
            line: DelayLine::new(),
            lfo_phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

const FLANGER_MAX_DELAY: f32 = 0.020;

impl UGen for Flanger {
    ugen_spec!(
        "Flanger",
        category = Effect,
        inputs = ["in", "rate", "depth", "feedback", "mix"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (FLANGER_MAX_DELAY * context.sample_rate) as usize + 2;
        self.line.resize(max_samples);
        self.lfo_phase = 0.0;
    }

    fn reset(&mut self) {
        self.line.clear();
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
        if self.line.is_empty() {
            return;
        }
        let max_delay_samples = (self.line.len() - 2) as f32;
        let inv_sr = 1.0 / self.sample_rate;

        // Every channel replays the shared delay line from the same cursor.
        let start_pos = self.line.write_pos();

        for ch in 0..output.num_channels() {
            self.line.set_write_pos(start_pos);
            let mut lfo_phase = self.lfo_phase;
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let rate = read_input(rate_buf, ch, i, 0.3).max(0.01);
                let depth = read_input(depth_buf, ch, i, 0.002).clamp(0.0, 0.010);
                let feedback = read_input(fb_buf, ch, i, 0.5).clamp(-0.95, 0.95);
                let mix = read_input(mix_buf, ch, i, 0.5).clamp(0.0, 1.0);

                // LFO-modulated delay time (unipolar: 0 to depth)
                let lfo = (lfo_phase * TAU).sin() * 0.5 + 0.5;
                let delay_secs = 0.0005 + depth * lfo; // min 0.5ms
                let delay_samples = (delay_secs * self.sample_rate).clamp(1.0, max_delay_samples);

                let delayed = self.line.read_interp(delay_samples);

                // Write input + feedback into buffer
                self.line.write_and_advance(x + feedback * delayed);

                out[i] = (1.0 - mix) * x + mix * delayed;

                lfo_phase += rate * inv_sr;
                lfo_phase -= lfo_phase.floor();
            }

            if ch == 0 {
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

impl Default for Phaser {
    fn default() -> Self {
        Self::new()
    }
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

impl UGen for Phaser {
    ugen_spec!(
        "Phaser",
        category = Effect,
        inputs = ["in", "rate", "depth", "feedback", "mix"],
        outputs = ["out"]
    );

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
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let rate = read_input(rate_buf, ch, i, 0.4).max(0.01);
                let depth = read_input(depth_buf, ch, i, 0.7).clamp(0.0, 1.0);
                let feedback = read_input(fb_buf, ch, i, 0.3).clamp(-0.95, 0.95);
                let mix = read_input(mix_buf, ch, i, 0.5).clamp(0.0, 1.0);

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
                for ap in ap_state.iter_mut() {
                    let y = coeff * signal + *ap;
                    *ap = signal - coeff * y;
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
