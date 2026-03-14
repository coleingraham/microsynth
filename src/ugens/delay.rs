//! Delay UGens with linear interpolation.
//!
//! - [`Delay`]: Simple read-only delay line.
//! - [`FeedbackDelay`]: Delay line with feedback (output feeds back into input).

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use alloc::vec::Vec;

/// Maximum delay time in seconds. Determines buffer size at init.
const MAX_DELAY_SECS: f32 = 5.0;

/// Simple delay line with linear interpolation.
///
/// Inputs: in (signal), time (delay time in seconds, clamped to max).
pub struct Delay {
    buffer: Vec<f32>,
    write_pos: usize,
    sample_rate: f32,
}

impl Delay {
    pub fn new() -> Self {
        Delay {
            buffer: Vec::new(),
            write_pos: 0,
            sample_rate: 44100.0,
        }
    }
}

static DELAY_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "time", rate: Rate::Audio },
];
static DELAY_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Delay {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Delay", inputs: &DELAY_INPUTS, outputs: &DELAY_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (MAX_DELAY_SECS * context.sample_rate) as usize + 1;
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
        let time_buf = inputs.get(1).copied();
        let buf_len = self.buffer.len();
        if buf_len == 0 {
            return;
        }
        let max_delay_samples = (buf_len - 1) as f32;

        for ch in 0..output.num_channels() {
            let mut write_pos = self.write_pos;
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let delay_time = time_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.1)
                    .max(0.0);
                let delay_samples = (delay_time * self.sample_rate)
                    .min(max_delay_samples)
                    .max(0.0);

                // Write current sample
                self.buffer[write_pos] = in_ch[i];

                // Read with linear interpolation
                let delay_int = delay_samples as usize;
                let frac = delay_samples - delay_int as f32;

                let read_pos_a = (write_pos + buf_len - delay_int) % buf_len;
                let read_pos_b = (write_pos + buf_len - delay_int - 1) % buf_len;

                let a = self.buffer[read_pos_a];
                let b = self.buffer[read_pos_b];
                out[i] = a + frac * (b - a);

                write_pos = (write_pos + 1) % buf_len;
            }

            if ch == 0 {
                self.write_pos = write_pos;
            }
        }
    }
}

// --- FeedbackDelay ---

/// Delay line with feedback.
///
/// y[n] = x[n] + feedback * y[n - delay_time]
///
/// Inputs: in (signal), time (delay time in seconds), feedback (−0.999 to 0.999).
/// Like a comb filter but with longer max delay (5 seconds), suitable for
/// echo/delay effects. Use lower feedback values (0.3–0.6) for clean echoes,
/// higher values for dub-style repeats.
pub struct FeedbackDelay {
    buffer: Vec<f32>,
    write_pos: usize,
    sample_rate: f32,
}

impl FeedbackDelay {
    pub fn new() -> Self {
        FeedbackDelay {
            buffer: Vec::new(),
            write_pos: 0,
            sample_rate: 44100.0,
        }
    }
}

static FB_DELAY_INPUTS: [InputSpec; 3] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "time", rate: Rate::Audio },
    InputSpec { name: "feedback", rate: Rate::Audio },
];
static FB_DELAY_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for FeedbackDelay {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "FeedbackDelay", inputs: &FB_DELAY_INPUTS, outputs: &FB_DELAY_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (MAX_DELAY_SECS * context.sample_rate) as usize + 1;
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
        let time_buf = inputs.get(1).copied();
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
                let delay_time = time_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.25)
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
                let delayed = self.buffer[read_a]
                    + frac * (self.buffer[read_b] - self.buffer[read_a]);

                // Output = input + feedback * delayed output
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
