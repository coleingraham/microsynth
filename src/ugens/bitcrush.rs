//! Bitcrusher UGen: sample rate and bit depth reduction.
//!
//! Lo-fi digital degradation effects for retro/vintage textures.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};

// --- Bitcrusher ---

/// Sample rate reducer and bit depth quantizer.
///
/// Produces lo-fi digital artifacts by:
/// 1. Reducing the effective sample rate (sample-and-hold at lower rate)
/// 2. Quantizing amplitude to fewer bits
///
/// Inputs:
/// - `in`: audio signal
/// - `bits`: bit depth (default 16.0, range 1.0–32.0). Lower values
///   produce more audible quantization noise. At 8 bits the sound is
///   distinctly lo-fi; at 4 bits it becomes harsh and distorted.
/// - `downsample`: sample rate reduction factor (default 1.0, range 1.0–64.0).
///   At 1.0 = no reduction. At 4.0 the effective sample rate is 1/4th,
///   producing aliasing and staircase artifacts.
pub struct Bitcrusher {
    hold_sample: f32,
    hold_counter: f32,
}

impl Bitcrusher {
    pub fn new() -> Self {
        Bitcrusher {
            hold_sample: 0.0,
            hold_counter: 0.0,
        }
    }
}

static BITCRUSH_INPUTS: [InputSpec; 3] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "bits", rate: Rate::Audio },
    InputSpec { name: "downsample", rate: Rate::Audio },
];
static BITCRUSH_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Bitcrusher {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Bitcrusher", inputs: &BITCRUSH_INPUTS, outputs: &BITCRUSH_OUTPUTS }
    }

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        self.hold_sample = 0.0;
        self.hold_counter = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let bits_buf = inputs.get(1).copied();
        let ds_buf = inputs.get(2).copied();

        for ch in 0..output.num_channels() {
            let mut hold = self.hold_sample;
            let mut counter = self.hold_counter;
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let bits = bits_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(16.0)
                    .clamp(1.0, 32.0);
                let downsample = ds_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1.0)
                    .max(1.0);

                // Sample rate reduction via sample-and-hold
                counter += 1.0;
                if counter >= downsample {
                    counter -= downsample;

                    // Bit depth reduction: quantize to 2^bits levels
                    // Use fast integer approximation via float math
                    let levels = (1u32 << (bits as u32).min(24)) as f32;
                    let half_levels = levels * 0.5;
                    hold = ((x * half_levels).round()) / half_levels;
                }

                out[i] = hold;
            }

            if ch == 0 {
                self.hold_sample = hold;
                self.hold_counter = counter;
            }
        }
    }
}
