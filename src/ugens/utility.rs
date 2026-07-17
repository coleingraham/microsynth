//! Utility UGens: Pan2, Mix, SampleAndHold, Impulse, Lag, Clip.

use crate::buffer::{AudioBuffer, channel_wrapped, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;

// --- Pan2 ---

/// Equal-power stereo panner.
///
/// Inputs: in (mono signal), pos (pan position: -1 = left, 0 = center, +1 = right).
/// Outputs: 2-channel stereo signal.
///
/// Uses equal-power panning: left = cos(theta) * in, right = sin(theta) * in,
/// where theta = (pos + 1) * pi/4.
pub struct Pan2;

impl Default for Pan2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Pan2 {
    pub fn new() -> Self {
        Pan2
    }
}

impl UGen for Pan2 {
    ugen_spec!("Pan2", inputs = ["in", "pos"], outputs = ["out"]);

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    /// Pan2 always produces 2 output channels regardless of input channel count.
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
        let pos_buf = inputs.get(1).copied();
        let quarter_pi = core::f32::consts::FRAC_PI_4;

        // Output channel 0 = left, channel 1 = right
        let block_size = output.block_size();
        for i in 0..block_size {
            // Mono input (use channel 0, wrapping if multichannel)
            let x = in_buf.channel(0).samples()[i];
            let pos = pos_buf
                .map(|b| b.channel(0).samples()[i])
                .unwrap_or(0.0)
                .clamp(-1.0, 1.0);

            let theta = (pos + 1.0) * quarter_pi;
            let (sin_t, cos_t) = (theta.sin(), theta.cos());

            output.channel_mut(0).samples_mut()[i] = cos_t * x;
            output.channel_mut(1).samples_mut()[i] = sin_t * x;
        }
    }
}

// --- Mix ---

/// Mixes a multichannel input down to mono by summing all channels.
///
/// Inputs: in (any number of channels).
/// Outputs: 1-channel mono mix (sum of all input channels).
pub struct Mix;

impl Default for Mix {
    fn default() -> Self {
        Self::new()
    }
}

impl Mix {
    pub fn new() -> Self {
        Mix
    }
}

impl UGen for Mix {
    ugen_spec!("Mix", inputs = ["in"], outputs = ["out"]);

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    /// Mix always produces exactly 1 output channel.
    fn output_channels(&self, _input_channels: &[usize]) -> usize {
        1
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let out = output.channel_mut(0).samples_mut();

        // Sum all input channels into the output
        let len = out.len();
        out[..len].fill(0.0);
        for ch in 0..in_buf.num_channels() {
            let ch_samples = in_buf.channel(ch).samples();
            for i in 0..len {
                out[i] += ch_samples[i];
            }
        }
    }
}

// --- SampleAndHold ---

/// Sample and Hold: captures the input value when the trigger crosses from
/// <= 0 to > 0, and holds it until the next trigger.
///
/// Inputs: in (signal to sample), trig (trigger signal).
pub struct SampleAndHold {
    held_value: f32,
    prev_trig: f32,
}

impl Default for SampleAndHold {
    fn default() -> Self {
        Self::new()
    }
}

impl SampleAndHold {
    pub fn new() -> Self {
        SampleAndHold {
            held_value: 0.0,
            prev_trig: 0.0,
        }
    }
}

impl UGen for SampleAndHold {
    ugen_spec!("SampleAndHold", inputs = ["in", "trig"], outputs = ["out"]);

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        self.held_value = 0.0;
        self.prev_trig = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let trig_buf = inputs[1];

        for ch in 0..output.num_channels() {
            let mut held = self.held_value;
            let mut prev_trig = self.prev_trig;
            let in_ch = channel_wrapped(in_buf, ch);
            let trig_ch = channel_wrapped(trig_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let trig = trig_ch[i];
                // Trigger on positive-going zero crossing
                if trig > 0.0 && prev_trig <= 0.0 {
                    held = in_ch[i];
                }
                out[i] = held;
                prev_trig = trig;
            }

            if ch == 0 {
                self.held_value = held;
                self.prev_trig = prev_trig;
            }
        }
    }
}

// --- Impulse ---

/// Periodic impulse generator. Outputs 1.0 once per period, 0.0 otherwise.
///
/// Inputs: freq (Hz — impulses per second).
/// Fires on the very first sample, then at each period boundary.
pub struct Impulse {
    phase: f32,
    sample_rate: f32,
    first: bool,
}

impl Default for Impulse {
    fn default() -> Self {
        Self::new()
    }
}

impl Impulse {
    pub fn new() -> Self {
        Impulse {
            phase: 0.0,
            sample_rate: 44100.0,
            first: true,
        }
    }
}

impl UGen for Impulse {
    ugen_spec!("Impulse", inputs = ["freq"], outputs = ["out"]);

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        self.phase = 0.0;
        self.first = true;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.first = true;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied();
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut phase = self.phase;
            let mut first = self.first;
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let freq = read_input(freq_buf, ch, i, 1.0);

                if first {
                    *out_sample = 1.0;
                    first = false;
                    phase += freq * inv_sr;
                } else {
                    phase += freq * inv_sr;
                    if phase >= 1.0 {
                        phase -= phase.floor();
                        *out_sample = 1.0;
                    } else {
                        *out_sample = 0.0;
                    }
                }
            }

            if ch == 0 {
                self.phase = phase;
                self.first = first;
            }
        }
    }
}

// --- Lag ---

/// Exponential lag (one-pole smoothing filter) for parameter smoothing.
///
/// Inputs: in (signal to smooth), time (lag time in seconds).
/// Smoothly follows the input with the given time constant.
/// Useful for avoiding clicks when changing parameters.
pub struct Lag {
    y1: f32,
    sample_rate: f32,
}

impl Default for Lag {
    fn default() -> Self {
        Self::new()
    }
}

impl Lag {
    pub fn new() -> Self {
        Lag {
            y1: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for Lag {
    ugen_spec!("Lag", inputs = ["in", "time"], outputs = ["out"]);

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

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
        let time_buf = inputs.get(1).copied();

        for ch in 0..output.num_channels() {
            let mut y1 = self.y1;
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let lag_time = read_input(time_buf, ch, i, 0.1).max(0.0);

                if lag_time <= 0.0 {
                    y1 = x;
                } else {
                    // One-pole coefficient from time constant
                    let coeff = (-1.0 / (lag_time * self.sample_rate)).exp();
                    y1 = x + coeff * (y1 - x);
                }
                out[i] = y1;
            }

            if ch == 0 {
                self.y1 = y1;
            }
        }
    }
}

// --- Clip ---

/// Hard clipper: clamps the input signal between lo and hi.
///
/// Inputs: in (signal), lo (minimum), hi (maximum).
pub struct Clip;

impl Default for Clip {
    fn default() -> Self {
        Self::new()
    }
}

impl Clip {
    pub fn new() -> Self {
        Clip
    }
}

impl UGen for Clip {
    ugen_spec!("Clip", inputs = ["in", "lo", "hi"], outputs = ["out"]);

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let lo_buf = inputs.get(1).copied();
        let hi_buf = inputs.get(2).copied();

        for ch in 0..output.num_channels() {
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let lo = read_input(lo_buf, ch, i, -1.0);
                let hi = read_input(hi_buf, ch, i, 1.0);
                out[i] = x.clamp(lo, hi);
            }
        }
    }
}
