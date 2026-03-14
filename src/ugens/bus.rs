//! Bus UGen for summing multiple voice outputs.
//!
//! A Bus accepts up to `max_voices` inputs and sums them together.
//! Each input slot is a separate port that a voice's output can be connected to.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use alloc::boxed::Box;
use alloc::vec::Vec;

/// Maximum number of simultaneous voice inputs on a single bus.
const MAX_BUS_INPUTS: usize = 64;

/// A summing bus that mixes multiple inputs together.
///
/// Has up to 64 input ports. Each input slot can receive a voice or effect output.
/// Outputs the sum of all connected inputs with a fixed channel count.
/// Unconnected inputs contribute silence.
pub struct Bus {
    /// Leaked static reference to input specs (allocated once, lives for program duration).
    input_specs: &'static [InputSpec],
    /// Declared output channel count (e.g. 2 for stereo).
    channels: usize,
}

impl Bus {
    /// Create a bus with the given output channel count.
    /// Input slot count defaults to 64 (internal detail).
    pub fn new(channels: usize) -> Self {
        let max_inputs = MAX_BUS_INPUTS;
        let mut specs = Vec::with_capacity(max_inputs);
        for _ in 0..max_inputs {
            specs.push(InputSpec {
                name: "in",
                rate: Rate::Audio,
            });
        }
        // Leak once — Bus nodes are infrastructure and live for the engine's lifetime
        let input_specs: &'static [InputSpec] = Box::leak(specs.into_boxed_slice());
        Bus { input_specs, channels: channels.max(1) }
    }

    /// Create a default stereo bus.
    pub fn default_bus() -> Self {
        Self::new(2)
    }

    /// Declared output channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Number of input slots available.
    pub fn max_inputs(&self) -> usize {
        self.input_specs.len()
    }
}

/// A pass-through UGen that copies its audio input to its output.
///
/// Used in effect SynthDefs to mark where external audio (from a bus) enters
/// the processing chain. The routing system wires a bus's output to this
/// node's input when instantiating an effect.
pub struct AudioIn;

static AUDIO_IN_INPUTS: [InputSpec; 1] = [InputSpec { name: "in", rate: Rate::Audio }];
static AUDIO_IN_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for AudioIn {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "AudioIn",
            inputs: &AUDIO_IN_INPUTS,
            outputs: &AUDIO_IN_OUTPUTS,
        }
    }

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn output_channels(&self, input_channels: &[usize]) -> usize {
        // Match input channel count, default to 2 (stereo)
        input_channels.first().copied().unwrap_or(2)
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        if inputs.is_empty() {
            output.clear();
            return;
        }
        let input = inputs[0];
        for ch in 0..output.num_channels() {
            let in_ch = ch % input.num_channels();
            let in_samples = input.channel(in_ch).samples();
            let out_samples = output.channel_mut(ch).samples_mut();
            out_samples[..in_samples.len()].copy_from_slice(in_samples);
        }
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

    fn output_channels(&self, _input_channels: &[usize]) -> usize {
        // Fixed output channel count as declared at construction
        self.channels
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
