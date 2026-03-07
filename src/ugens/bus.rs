//! Bus UGen for summing multiple voice outputs.
//!
//! A Bus accepts up to `max_voices` inputs and sums them together.
//! Each input slot is a separate port that a voice's output can be connected to.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use alloc::boxed::Box;
use alloc::vec::Vec;

/// Maximum number of simultaneous voices on a single bus.
const MAX_BUS_VOICES: usize = 64;

/// A summing bus that mixes multiple inputs together.
///
/// Has `max_voices` input ports, each representing one voice slot.
/// Outputs the sum of all connected inputs. Unconnected inputs contribute silence.
pub struct Bus {
    /// Leaked static reference to input specs (allocated once, lives for program duration).
    input_specs: &'static [InputSpec],
}

impl Bus {
    /// Create a bus with the given maximum number of voice inputs.
    pub fn new(max_voices: usize) -> Self {
        let max_voices = max_voices.min(MAX_BUS_VOICES);
        let mut specs = Vec::with_capacity(max_voices);
        for _ in 0..max_voices {
            specs.push(InputSpec {
                name: "in",
                rate: Rate::Audio,
            });
        }
        // Leak once — Bus nodes are infrastructure and live for the engine's lifetime
        let input_specs: &'static [InputSpec] = Box::leak(specs.into_boxed_slice());
        Bus { input_specs }
    }

    /// Create a bus with default 32 voice slots.
    pub fn default_bus() -> Self {
        Self::new(32)
    }

    /// Number of voice input slots.
    pub fn max_voices(&self) -> usize {
        self.input_specs.len()
    }
}

static BUS_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Bus {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "Bus",
            inputs: self.input_specs,
            outputs: &BUS_OUTPUTS,
        }
    }

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn output_channels(&self, input_channels: &[usize]) -> usize {
        // Output channel count = max channel count of any connected input, default 2
        input_channels.iter().copied().max().unwrap_or(2).max(1)
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        // Clear output
        output.clear();

        // Sum all connected inputs
        for input in inputs {
            for ch in 0..output.num_channels() {
                let in_ch = ch % input.num_channels();
                let in_samples = input.channel(in_ch).samples();
                let out_samples = output.channel_mut(ch).samples_mut();
                for i in 0..out_samples.len() {
                    out_samples[i] += in_samples[i];
                }
            }
        }
    }
}
