use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};

/// Unique identifier for a node in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) u32);

impl NodeId {
    /// The raw integer index.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Descriptor for a node input port.
#[derive(Debug, Clone)]
pub struct InputSpec {
    /// Human-readable name (e.g. "freq", "amp").
    pub name: &'static str,
    /// The rate this input expects.
    pub rate: Rate,
}

/// Descriptor for a node output port.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    /// Human-readable name (e.g. "out").
    pub name: &'static str,
    /// The rate this output produces.
    pub rate: Rate,
}

/// Static metadata about a UGen type.
#[derive(Debug, Clone)]
pub struct UGenSpec {
    /// Name of this UGen (e.g. "SinOsc", "LPF").
    pub name: &'static str,
    /// Input port descriptors.
    pub inputs: &'static [InputSpec],
    /// Output port descriptors.
    pub outputs: &'static [OutputSpec],
}

/// The core trait for all audio processing nodes.
///
/// Each UGen processes one block at a time. The graph calls `process` in
/// topological order so that all inputs are available when a node runs.
///
/// # Multichannel Expansion
///
/// When inputs have differing channel counts, the graph resolves the
/// "expansion factor" (max channel count across inputs) and sets the
/// output buffer to that many channels. Inside `process`, use modulo
/// indexing on inputs: `inputs[port].channel(ch % inputs[port].num_channels())`
/// to implement SuperCollider-style wrapping.
pub trait UGen: Send {
    /// Return the static specification for this UGen.
    fn spec(&self) -> UGenSpec;

    /// Called once when the graph is prepared for rendering.
    fn init(&mut self, context: &ProcessContext);

    /// Reset internal state (phase accumulators, filter memory, etc.)
    fn reset(&mut self);

    /// Determine how many output channels this node produces given
    /// the channel counts of each input.
    ///
    /// Default implementation: max of all input channel counts, or 1 if no inputs.
    fn output_channels(&self, input_channels: &[usize]) -> usize {
        input_channels.iter().copied().max().unwrap_or(1)
    }

    /// Process one block of audio/control data.
    ///
    /// - `context`: global engine state (sample rate, block size, time)
    /// - `inputs`: one `AudioBuffer` per input port, read-only
    /// - `output`: the output buffer to write into; channel count has been
    ///   pre-set by the graph according to `output_channels`.
    fn process(&mut self, context: &ProcessContext, inputs: &[&AudioBuffer], output: &mut AudioBuffer);

    /// Set an internal scalar value (e.g. the value of a Const or Param node).
    /// Returns true if the node accepted the value.
    fn set_value(&mut self, _value: f32) -> bool {
        false
    }

    /// Set a target value with a glide time in seconds.
    /// The node smoothly transitions from its current value to `target`
    /// over `glide_secs` seconds. Returns true if accepted.
    fn set_target(&mut self, _target: f32, _glide_secs: f32) -> bool {
        false
    }

    /// Query whether this node has finished producing useful output
    /// (e.g. an envelope that reached its end). Used by the engine to
    /// implement done actions (auto-freeing voices).
    fn is_done(&self) -> bool {
        false
    }
}
