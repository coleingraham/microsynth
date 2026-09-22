//! Delay UGens with linear interpolation.
//!
//! - [`Delay`]: Simple read-only delay line.
//! - [`FeedbackDelay`]: Delay line with feedback (output feeds back into input).

use crate::buffer::{AudioBuffer, channel_wrapped, read_input, require_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use crate::ugens::delayline::DelayLine;

/// Maximum delay time in seconds. Determines buffer size at init.
const MAX_DELAY_SECS: f32 = 5.0;

/// Simple delay line with linear interpolation.
///
/// Inputs: in (signal), time (delay time in seconds, clamped to max).
pub struct Delay {
    lines: [DelayLine; 2],
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
            lines: [DelayLine::new(), DelayLine::new()],
            sample_rate: 44100.0,
        }
    }
}

impl UGen for Delay {
    ugen_spec!(
        "Delay",
        category = Effect,
        inputs = ["in"],
        optional_inputs = ["time"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (MAX_DELAY_SECS * context.sample_rate) as usize + 1;
        for line in &mut self.lines {
            line.resize(max_samples);
        }
    }

    fn reset(&mut self) {
        for line in &mut self.lines {
            line.clear();
        }
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let in_buf = require_input(inputs, 0, self.spec().name, "in");
        let time_buf = inputs.get(1).copied().flatten();
        if self.lines[0].is_empty() {
            return;
        }
        let max_delay_samples = (self.lines[0].len() - 1) as f32;
        let sample_rate = self.sample_rate;

        // One delay line per channel: see filters::OnePole's process().
        for ch in 0..output.num_channels() {
            let line = &mut self.lines[ch.min(1)];
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let delay_time = read_input(time_buf, ch, i, 0.1).max(0.0);
                let delay_samples = (delay_time * sample_rate).min(max_delay_samples).max(0.0);

                // Write first: a delay of zero reads back this very sample.
                line.write(in_ch[i]);
                out[i] = line.read_interp(delay_samples);
                line.advance();
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
    lines: [DelayLine; 2],
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
            lines: [DelayLine::new(), DelayLine::new()],
            sample_rate: 44100.0,
        }
    }
}

impl UGen for FeedbackDelay {
    ugen_spec!(
        "FeedbackDelay",
        category = Effect,
        inputs = ["in"],
        optional_inputs = ["time", "feedback"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (MAX_DELAY_SECS * context.sample_rate) as usize + 1;
        for line in &mut self.lines {
            line.resize(max_samples);
        }
    }

    fn reset(&mut self) {
        for line in &mut self.lines {
            line.clear();
        }
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let in_buf = require_input(inputs, 0, self.spec().name, "in");
        let time_buf = inputs.get(1).copied().flatten();
        let fb_buf = inputs.get(2).copied().flatten();
        if self.lines[0].is_empty() {
            return;
        }
        let max_delay = (self.lines[0].len() - 1) as f32;
        let sample_rate = self.sample_rate;

        // One delay line per channel: see filters::OnePole's process().
        for ch in 0..output.num_channels() {
            let line = &mut self.lines[ch.min(1)];
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let delay_time = read_input(time_buf, ch, i, 0.25).max(0.0);
                let feedback = read_input(fb_buf, ch, i, 0.5).clamp(-0.999, 0.999);

                let delay_samples = (delay_time * sample_rate).min(max_delay).max(1.0);

                // Output = input + feedback * delayed output
                let delayed = line.read_interp(delay_samples);
                let y = in_ch[i] + feedback * delayed;

                line.write_and_advance(y);
                out[i] = y;
            }
        }
    }
}
