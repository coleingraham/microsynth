//! Delay UGen with linear interpolation.

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
