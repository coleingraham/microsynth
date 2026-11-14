//! Bus UGen for summing multiple voice outputs.
//!
//! A Bus accepts up to [`MAX_BUS_INPUTS`] inputs and sums them together.
//! Each input slot is a separate port that a voice's output can be connected to.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenCategory, UGenSpec};

/// Maximum number of simultaneous inputs (voices, effect returns) on a single
/// bus.
///
/// 256 is sized for offline rendering of long-tail sampled instruments: a
/// harp or vibraphone whose release outlasts the recording keeps every voice
/// alive to the end of its sample, and a phrase of chords on one of them
/// legitimately holds 70 or more voices at once. A voice that finds no free
/// slot is refused (or, through the managed spawn path with a voice budget,
/// makes room by stealing). An unconnected slot costs one null pointer
/// gathered and skipped per block; the slot specs are one shared static
/// table, so the count adds no per-bus allocation.
pub const MAX_BUS_INPUTS: usize = 256;

/// The input-port table every bus shares: [`MAX_BUS_INPUTS`] identical,
/// optional audio ports. One static array instead of a per-bus leaked
/// allocation — every bus's slots are the same, so they can share it.
static BUS_INPUT_SPECS: [InputSpec; MAX_BUS_INPUTS] = [BUS_SLOT; MAX_BUS_INPUTS];

/// One bus input slot.
const BUS_SLOT: InputSpec = InputSpec {
    name: "in",
    rate: Rate::Audio,
    // Every slot is optional: a bus with only some of its slots wired is the
    // normal case (unused voice/effect capacity), and an unconnected slot
    // contributes silence to the sum — never a construction error.
    required: false,
};

/// A bus's declared *output* channel count, expressed as intent rather than
/// a bare `usize`.
///
/// This exists because a bare integer here is ambiguous in exactly the way
/// that broke production output: `Bus::new(64)` reads as "room for 64
/// voices," but the input-slot count is always [`MAX_BUS_INPUTS`] regardless
/// of what's passed here — the argument is *only* the output channel count,
/// and 64 output channels is not what anyone wanted. Naming the common cases
/// and requiring `Custom` for anything else makes that confusion impossible
/// to write by accident: a call site that wants a voice-capacity number has
/// nowhere to put it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelCount {
    /// Single output channel.
    Mono,
    /// Two output channels (left, right). The production default.
    Stereo,
    /// An explicit, non-standard channel count. Spelling this out (rather
    /// than accepting a bare number) forces any call site that isn't
    /// mono/stereo to say so, so it reads as a deliberate choice in review.
    Custom(usize),
}

impl ChannelCount {
    fn channels(self) -> usize {
        match self {
            ChannelCount::Mono => 1,
            ChannelCount::Stereo => 2,
            ChannelCount::Custom(n) => n.max(1),
        }
    }
}

/// A summing bus that mixes multiple inputs together.
///
/// Has [`MAX_BUS_INPUTS`] input ports. Each input slot can receive a voice or effect output.
/// Outputs the sum of all connected inputs with a fixed channel count.
/// Unconnected inputs contribute silence.
pub struct Bus {
    /// The shared static input-port table ([`BUS_INPUT_SPECS`]).
    input_specs: &'static [InputSpec],
    /// Declared output channel count (e.g. 2 for stereo).
    channels: usize,
}

impl Bus {
    /// Create a bus with the given output channel count.
    /// Input slot count is [`MAX_BUS_INPUTS`] (internal detail) regardless of
    /// the output channel count requested here — see [`ChannelCount`]'s doc
    /// for why the count is typed instead of a bare `usize`.
    pub fn new(channels: ChannelCount) -> Self {
        Bus {
            input_specs: &BUS_INPUT_SPECS,
            channels: channels.channels(),
        }
    }

    /// Create a default stereo bus.
    pub fn default_bus() -> Self {
        Self::new(ChannelCount::Stereo)
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

impl UGen for AudioIn {
    // `in` is optional, not required: `process` already treats an
    // unconnected AudioIn as silence (the routing system normally always
    // wires it when instantiating an effect, but nothing upstream of
    // `AudioGraph::prepare` guarantees that), and this preserves that
    // existing, deliberately-coded tolerance exactly rather than turning it
    // into a new prepare()-time panic.
    ugen_spec!(
        "AudioIn",
        inputs = [],
        optional_inputs = ["in"],
        outputs = ["out"]
    );

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn output_channels(&self, input_channels: &[usize]) -> usize {
        // Match input channel count, default to 2 (stereo)
        input_channels.first().copied().unwrap_or(2)
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let input = match inputs.first().copied().flatten() {
            Some(input) => input,
            None => {
                output.clear();
                return;
            }
        };
        for ch in 0..output.num_channels() {
            let in_ch = ch % input.num_channels();
            let in_samples = input.channel(in_ch).samples();
            let out_samples = output.channel_mut(ch).samples_mut();
            out_samples[..in_samples.len()].copy_from_slice(in_samples);
        }
    }
}

static BUS_OUTPUTS: [OutputSpec; 1] = [OutputSpec {
    name: "out",
    rate: Rate::Audio,
}];

impl UGen for Bus {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "Bus",
            category: UGenCategory::Utility,
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
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        // Clear output
        output.clear();

        // Sum all connected inputs (unconnected slots contribute nothing —
        // summation is order-independent, so skipping `None` slots here
        // produces the same result as the old compacted-slice iteration did).
        for input in inputs.iter().flatten() {
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
