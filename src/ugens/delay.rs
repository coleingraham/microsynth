//! Delay UGens with linear interpolation.
//!
//! - [`Delay`]: Simple read-only delay line.
//! - [`FeedbackDelay`]: Delay line with feedback (output feeds back into input).

use crate::buffer::{AudioBuffer, channel_wrapped, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use crate::ugens::delayline::DelayLine;

/// Maximum delay time in seconds. Determines buffer size at init.
const MAX_DELAY_SECS: f32 = 5.0;

/// Simple delay line with linear interpolation.
///
/// Inputs: in (signal), time (delay time in seconds, clamped to max).
pub struct Delay {
    line: DelayLine,
    sample_rate: f32,
}

impl Default for Delay {
    fn default() -> Self {
        Self::new()
    }
}

impl Delay {
    pub fn new() -> Self {
        Delay {
            line: DelayLine::new(),
            sample_rate: 44100.0,
        }
    }
}

impl UGen for Delay {
    ugen_spec!(
        "Delay",
        category = Effect,
        inputs = ["in", "time"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (MAX_DELAY_SECS * context.sample_rate) as usize + 1;
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
        let time_buf = inputs.get(1).copied();
        if self.line.is_empty() {
            return;
        }
        let max_delay_samples = (self.line.len() - 1) as f32;

        // Every channel replays the shared delay line from the same cursor.
        let start_pos = self.line.write_pos();

        for ch in 0..output.num_channels() {
            self.line.set_write_pos(start_pos);
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let delay_time = read_input(time_buf, ch, i, 0.1).max(0.0);
                let delay_samples = (delay_time * self.sample_rate)
                    .min(max_delay_samples)
                    .max(0.0);

                // Write first: a delay of zero reads back this very sample.
                self.line.write(in_ch[i]);
                out[i] = self.line.read_interp(delay_samples);
                self.line.advance();
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
    line: DelayLine,
    sample_rate: f32,
}

impl Default for FeedbackDelay {
    fn default() -> Self {
        Self::new()
    }
}

impl FeedbackDelay {
    pub fn new() -> Self {
        FeedbackDelay {
            line: DelayLine::new(),
            sample_rate: 44100.0,
        }
    }
}

impl UGen for FeedbackDelay {
    ugen_spec!(
        "FeedbackDelay",
        category = Effect,
        inputs = ["in", "time", "feedback"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (MAX_DELAY_SECS * context.sample_rate) as usize + 1;
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
        let time_buf = inputs.get(1).copied();
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
                let delay_time = read_input(time_buf, ch, i, 0.25).max(0.0);
                let feedback = read_input(fb_buf, ch, i, 0.5).clamp(-0.999, 0.999);

                let delay_samples = (delay_time * self.sample_rate).min(max_delay).max(1.0);

                // Output = input + feedback * delayed output
                let delayed = self.line.read_interp(delay_samples);
                let y = in_ch[i] + feedback * delayed;

                self.line.write_and_advance(y);
                out[i] = y;
            }
        }
    }
}
