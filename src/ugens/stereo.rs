//! Stereo field UGens: StereoWidth, PingPongDelay.
//!
//! Tools for stereo image manipulation and stereo delay effects.

use crate::buffer::{AudioBuffer, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use crate::ugens::delayline::DelayLine;

// --- StereoWidth ---

/// Stereo width / Haas effect processor.
///
/// Takes a mono input and produces a stereo output with controllable width.
/// The left channel is the dry signal; the right channel blends the dry signal
/// with a short, fractionally-interpolated Haas delay of it. The delay time and
/// the blend both scale with `width`, so the stereo image widens continuously.
///
/// Inputs:
/// - `in`: audio signal (mono)
/// - `width`: stereo width (default 0.5, range 0.0–1.0). At 0.0 both channels
///   carry the dry signal (mono); as it rises, the right channel's Haas delay
///   grows from 0 up to 25 ms and mixes in proportionally.
///
/// Output: 2-channel stereo signal.
pub struct StereoWidth {
    /// Short delay line for the Haas effect on the right channel.
    delay: DelayLine,
    sample_rate: f32,
}

impl Default for StereoWidth {
    fn default() -> Self {
        Self::new()
    }
}

impl StereoWidth {
    pub fn new() -> Self {
        StereoWidth {
            delay: DelayLine::new(),
            sample_rate: 44100.0,
        }
    }
}

/// Maximum Haas delay in seconds (perceptual limit before echo).
const HAAS_MAX_DELAY: f32 = 0.030;

impl UGen for StereoWidth {
    ugen_spec!(
        "StereoWidth",
        category = Effect,
        inputs = ["in", "width"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (HAAS_MAX_DELAY * context.sample_rate) as usize + 2;
        self.delay.resize(max_samples);
    }

    fn reset(&mut self) {
        self.delay.clear();
    }

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
        let width_buf = inputs.get(1).copied();
        if self.delay.is_empty() {
            return;
        }
        // read_interp reads at delay_samples and delay_samples + 1, so the
        // largest safe delay is len - 1.
        let max_delay = (self.delay.len() - 1) as f32;

        let block_size = output.block_size();

        for i in 0..block_size {
            let x = read_input(Some(in_buf), 0, i, 0.0);
            let width = read_input(width_buf, 0, i, 0.5).clamp(0.0, 1.0);

            // Write first, so a delay of zero reads back this very sample.
            self.delay.write(x);

            // Haas delay time scales with width (0 ms at width=0, 25 ms at width=1).
            let delay_samples = (width * 0.025 * self.sample_rate).clamp(0.0, max_delay);
            let delayed = self.delay.read_interp(delay_samples);

            // Left channel: dry signal
            // Right channel: delayed signal (Haas effect)
            // At width=0 both channels get the same signal (mono)
            output.channel_mut(0).samples_mut()[i] = x;
            output.channel_mut(1).samples_mut()[i] = x * (1.0 - width) + delayed * width;

            self.delay.advance();
        }
    }
}

// --- PingPongDelay ---

/// Stereo ping-pong delay.
///
/// Alternating left-right delay taps that bounce the signal between channels.
/// Each echo appears in the opposite channel from the previous one.
///
/// Inputs:
/// - `in`: audio signal
/// - `time`: delay time per tap in seconds (default 0.25)
/// - `feedback`: feedback amount (default 0.4, range 0.0–0.95)
/// - `mix`: dry/wet blend (default 0.4)
///
/// Output: 2-channel stereo signal.
pub struct PingPongDelay {
    line_l: DelayLine,
    line_r: DelayLine,
    sample_rate: f32,
}

impl Default for PingPongDelay {
    fn default() -> Self {
        Self::new()
    }
}

impl PingPongDelay {
    pub fn new() -> Self {
        PingPongDelay {
            line_l: DelayLine::new(),
            line_r: DelayLine::new(),
            sample_rate: 44100.0,
        }
    }
}

/// Max delay for ping-pong (same as regular delay).
const PP_MAX_DELAY: f32 = 5.0;

impl UGen for PingPongDelay {
    ugen_spec!(
        "PingPongDelay",
        category = Effect,
        inputs = ["in", "time", "feedback", "mix"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (PP_MAX_DELAY * context.sample_rate) as usize + 1;
        self.line_l.resize(max_samples);
        self.line_r.resize(max_samples);
    }

    fn reset(&mut self) {
        self.line_l.clear();
        self.line_r.clear();
    }

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
        let time_buf = inputs.get(1).copied();
        let fb_buf = inputs.get(2).copied();
        let mix_buf = inputs.get(3).copied();
        if self.line_l.is_empty() {
            return;
        }
        let max_delay = (self.line_l.len() - 1) as f32;

        let block_size = output.block_size();

        for i in 0..block_size {
            let x = in_buf.channel(0).samples()[i];
            let delay_time = time_buf
                .map(|b| b.channel(0).samples()[i])
                .unwrap_or(0.25)
                .max(0.0);
            let feedback = fb_buf
                .map(|b| b.channel(0).samples()[i])
                .unwrap_or(0.4)
                .clamp(0.0, 0.95);
            let mix = mix_buf
                .map(|b| b.channel(0).samples()[i])
                .unwrap_or(0.4)
                .clamp(0.0, 1.0);

            let delay_samples = (delay_time * self.sample_rate).min(max_delay).max(1.0);

            // Read both delays before writing either: the cross-feed below
            // depends on the pre-write values.
            let del_l = self.line_l.read_interp(delay_samples);
            let del_r = self.line_r.read_interp(delay_samples);

            // Cross-feed: input goes to left, left output feeds right, right feeds left
            self.line_l.write_and_advance(x + feedback * del_r);
            self.line_r.write_and_advance(feedback * del_l);

            // Output
            output.channel_mut(0).samples_mut()[i] = (1.0 - mix) * x + mix * del_l;
            output.channel_mut(1).samples_mut()[i] = (1.0 - mix) * x + mix * del_r;
        }
    }
}
