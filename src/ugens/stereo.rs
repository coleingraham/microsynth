//! Stereo field UGens: StereoWidth, PingPongDelay.
//!
//! Tools for stereo image manipulation and stereo delay effects.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use alloc::vec::Vec;

// --- StereoWidth ---

/// Stereo width / Haas effect processor.
///
/// Takes a mono input and produces a stereo output with controllable width.
/// Uses a short delay on one channel (Haas effect) combined with mid-side
/// processing to widen or narrow the stereo image.
///
/// Inputs:
/// - `in`: audio signal (mono)
/// - `width`: stereo width (default 0.5, range 0.0–1.0).
///   0.0 = mono, 0.5 = natural width, 1.0 = maximum width.
///   Values above 0.5 introduce Haas delay for extra-wide imaging.
///
/// Output: 2-channel stereo signal.
pub struct StereoWidth {
    /// Short delay buffer for Haas effect on right channel.
    delay_buf: Vec<f32>,
    write_pos: usize,
    sample_rate: f32,
}

impl StereoWidth {
    pub fn new() -> Self {
        StereoWidth {
            delay_buf: Vec::new(),
            write_pos: 0,
            sample_rate: 44100.0,
        }
    }
}

/// Maximum Haas delay in seconds (perceptual limit before echo).
const HAAS_MAX_DELAY: f32 = 0.030;

static STEREO_WIDTH_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "width", rate: Rate::Audio },
];
static STEREO_WIDTH_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for StereoWidth {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "StereoWidth",
            inputs: &STEREO_WIDTH_INPUTS,
            outputs: &STEREO_WIDTH_OUTPUTS,
        }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (HAAS_MAX_DELAY * context.sample_rate) as usize + 2;
        self.delay_buf.resize(max_samples, 0.0);
        self.write_pos = 0;
    }

    fn reset(&mut self) {
        self.delay_buf.fill(0.0);
        self.write_pos = 0;
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
        let buf_len = self.delay_buf.len();
        if buf_len == 0 {
            return;
        }

        let block_size = output.block_size();
        let mut write_pos = self.write_pos;

        for i in 0..block_size {
            let x = in_buf.channel(0).samples()[i];
            let width = width_buf
                .map(|b| b.channel(0).samples()[i])
                .unwrap_or(0.5)
                .clamp(0.0, 1.0);

            // Write to delay buffer
            self.delay_buf[write_pos] = x;

            // Haas delay time scales with width (0ms at width=0, 25ms at width=1)
            let delay_secs = width * 0.025;
            let delay_samples = (delay_secs * self.sample_rate).min((buf_len - 2) as f32).max(0.0);

            // Read delayed sample with linear interpolation
            let d_int = delay_samples as usize;
            let d_frac = delay_samples - d_int as f32;
            let delayed = if d_int == 0 && d_frac < 0.001 {
                x
            } else {
                let ra = (write_pos + buf_len - d_int) % buf_len;
                let rb = (write_pos + buf_len - d_int.max(1)) % buf_len;
                self.delay_buf[ra] + d_frac * (self.delay_buf[rb] - self.delay_buf[ra])
            };

            // Left channel: dry signal
            // Right channel: delayed signal (Haas effect)
            // At width=0 both channels get the same signal (mono)
            let left = x;
            let right = x * (1.0 - width) + delayed * width;

            output.channel_mut(0).samples_mut()[i] = left;
            output.channel_mut(1).samples_mut()[i] = right;

            write_pos = (write_pos + 1) % buf_len;
        }

        self.write_pos = write_pos;
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
    buf_l: Vec<f32>,
    buf_r: Vec<f32>,
    write_pos_l: usize,
    write_pos_r: usize,
    sample_rate: f32,
}

impl PingPongDelay {
    pub fn new() -> Self {
        PingPongDelay {
            buf_l: Vec::new(),
            buf_r: Vec::new(),
            write_pos_l: 0,
            write_pos_r: 0,
            sample_rate: 44100.0,
        }
    }
}

/// Max delay for ping-pong (same as regular delay).
const PP_MAX_DELAY: f32 = 5.0;

static PP_DELAY_INPUTS: [InputSpec; 4] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "time", rate: Rate::Audio },
    InputSpec { name: "feedback", rate: Rate::Audio },
    InputSpec { name: "mix", rate: Rate::Audio },
];
static PP_DELAY_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for PingPongDelay {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "PingPongDelay",
            inputs: &PP_DELAY_INPUTS,
            outputs: &PP_DELAY_OUTPUTS,
        }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (PP_MAX_DELAY * context.sample_rate) as usize + 1;
        self.buf_l.resize(max_samples, 0.0);
        self.buf_r.resize(max_samples, 0.0);
        self.write_pos_l = 0;
        self.write_pos_r = 0;
    }

    fn reset(&mut self) {
        self.buf_l.fill(0.0);
        self.buf_r.fill(0.0);
        self.write_pos_l = 0;
        self.write_pos_r = 0;
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
        let buf_len = self.buf_l.len();
        if buf_len == 0 {
            return;
        }
        let max_delay = (buf_len - 1) as f32;

        let block_size = output.block_size();
        let mut wp_l = self.write_pos_l;
        let mut wp_r = self.write_pos_r;

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
            let d_int = delay_samples as usize;
            let d_frac = delay_samples - d_int as f32;

            // Read from left delay
            let ra_l = (wp_l + buf_len - d_int) % buf_len;
            let rb_l = (wp_l + buf_len - d_int - 1) % buf_len;
            let del_l = self.buf_l[ra_l] + d_frac * (self.buf_l[rb_l] - self.buf_l[ra_l]);

            // Read from right delay
            let ra_r = (wp_r + buf_len - d_int) % buf_len;
            let rb_r = (wp_r + buf_len - d_int - 1) % buf_len;
            let del_r = self.buf_r[ra_r] + d_frac * (self.buf_r[rb_r] - self.buf_r[ra_r]);

            // Cross-feed: input goes to left, left output feeds right, right feeds left
            self.buf_l[wp_l] = x + feedback * del_r;
            self.buf_r[wp_r] = feedback * del_l;

            // Output
            output.channel_mut(0).samples_mut()[i] = (1.0 - mix) * x + mix * del_l;
            output.channel_mut(1).samples_mut()[i] = (1.0 - mix) * x + mix * del_r;

            wp_l = (wp_l + 1) % buf_len;
            wp_r = (wp_r + 1) % buf_len;
        }

        self.write_pos_l = wp_l;
        self.write_pos_r = wp_r;
    }
}
