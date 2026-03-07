//! Utility UGens: Pan2, Mix, SampleAndHold.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};

// --- Pan2 ---

/// Equal-power stereo panner.
///
/// Inputs: in (mono signal), pos (pan position: -1 = left, 0 = center, +1 = right).
/// Outputs: 2-channel stereo signal.
///
/// Uses equal-power panning: left = cos(theta) * in, right = sin(theta) * in,
/// where theta = (pos + 1) * pi/4.
pub struct Pan2;

impl Pan2 {
    pub fn new() -> Self {
        Pan2
    }
}

static PAN2_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "pos", rate: Rate::Audio },
];
static PAN2_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Pan2 {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Pan2", inputs: &PAN2_INPUTS, outputs: &PAN2_OUTPUTS }
    }

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

impl Mix {
    pub fn new() -> Self {
        Mix
    }
}

static MIX_INPUTS: [InputSpec; 1] = [InputSpec { name: "in", rate: Rate::Audio }];
static MIX_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Mix {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Mix", inputs: &MIX_INPUTS, outputs: &MIX_OUTPUTS }
    }

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

impl SampleAndHold {
    pub fn new() -> Self {
        SampleAndHold {
            held_value: 0.0,
            prev_trig: 0.0,
        }
    }
}

static SH_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "trig", rate: Rate::Audio },
];
static SH_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for SampleAndHold {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "SampleAndHold", inputs: &SH_INPUTS, outputs: &SH_OUTPUTS }
    }

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
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let trig_ch = trig_buf.channel(ch % trig_buf.num_channels()).samples();
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
