//! Arithmetic UGens: Const, Param, BinOp, Neg.

use crate::buffer::{AudioBuffer, require_input};
use crate::context::{ProcessContext, Rate};
use crate::curve::{GlideShape, GlideSpace, glide_fraction};
use crate::node::{InputSpec, OutputSpec, UGen, UGenCategory, UGenSpec};
use crate::tuning::{hz_to_midi_12tet, midi_to_hz_12tet};

/// Reference frequency used to convert a pitch-space glide's endpoints to and
/// from 12-TET note numbers. The choice is arbitrary — any equal-tempered
/// reference cancels out of the resulting frequency ratio — but it must be
/// applied consistently to both endpoints, which this constant guarantees.
const PITCH_SPACE_A4_HZ: f32 = 440.0;

/// Outputs a constant value on all channels and samples.
pub struct Const {
    value: f32,
}

impl Const {
    pub fn new(value: f32) -> Self {
        Const { value }
    }
}

impl UGen for Const {
    ugen_spec!("Const", inputs = [], outputs = ["out"]);

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn process(
        &mut self,
        _context: &ProcessContext,
        _inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        for ch in 0..output.num_channels() {
            output.channel_mut(ch).fill(self.value);
        }
    }

    fn set_value(&mut self, value: f32) -> bool {
        self.value = value;
        true
    }
}

/// A controllable parameter node with optional glide (portamento).
///
/// Like `Const`, outputs a value on all channels and samples. But unlike
/// Const, `Param` supports smooth transitions via
/// `set_target(value, glide_secs, shape, space)`.
///
/// When glide is active, the output ramps from the current value to the
/// target over the specified duration, following one of four interpolation
/// shapes (see [`GlideShape`]) and interpolating either directly in the
/// parameter's own units or in pitch space (see [`GlideSpace`]). This is the
/// backbone of continuous parameter control: crescendo, diminuendo, pitch
/// bends, filter sweeps, etc.
///
/// Used by the DSL compiler for synthdef parameters.
pub struct Param {
    value: f32,
    target: f32,
    /// Per-sample delta for the `Linear` + `Raw` fast path.
    increment: f32,
    glide_shape: GlideShape,
    glide_space: GlideSpace,
    /// Start value of the active glide, in the working space (`Raw`: same
    /// units as `value`/`target`; `Pitch`: 12-TET note number).
    glide_start: f32,
    /// Target value of the active glide, in the working space.
    glide_target: f32,
    samples_remaining: u64,
    total_samples: u64,
    sample_rate: f32,
}

impl Param {
    pub fn new(value: f32) -> Self {
        Param {
            value,
            target: value,
            increment: 0.0,
            glide_shape: GlideShape::default(),
            glide_space: GlideSpace::default(),
            glide_start: value,
            glide_target: value,
            samples_remaining: 0,
            total_samples: 0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for Param {
    ugen_spec!("Param", inputs = [], outputs = ["out"]);

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.target = self.value;
        self.increment = 0.0;
        self.samples_remaining = 0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        _inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        if self.samples_remaining == 0 {
            // No glide active — output flat value
            for ch in 0..output.num_channels() {
                output.channel_mut(ch).fill(self.value);
            }
            return;
        }

        // `Linear` + `Raw` is the glide behavior that predates shapes and
        // spaces; keep it on the original constant-rate-increment path so
        // its output is bit-for-bit unchanged. Every other shape/space
        // combination re-evaluates the blend fraction each sample.
        let is_default_glide =
            self.glide_shape == GlideShape::Linear && self.glide_space == GlideSpace::Raw;

        for ch in 0..output.num_channels() {
            let mut value = self.value;
            let mut remaining = self.samples_remaining;
            let out = output.channel_mut(ch).samples_mut();

            if is_default_glide {
                for sample in out.iter_mut() {
                    *sample = value;
                    if remaining > 0 {
                        value += self.increment;
                        remaining -= 1;
                        if remaining == 0 {
                            value = self.target;
                        }
                    }
                }
            } else {
                let mut elapsed = self.total_samples - remaining;
                for sample in out.iter_mut() {
                    *sample = value;
                    if remaining > 0 {
                        elapsed += 1;
                        remaining -= 1;
                        value = if remaining == 0 {
                            self.target
                        } else {
                            let x = elapsed as f32 / self.total_samples as f32;
                            let frac = glide_fraction(self.glide_shape, x);
                            let interpolated =
                                self.glide_start + frac * (self.glide_target - self.glide_start);
                            match self.glide_space {
                                GlideSpace::Raw => interpolated,
                                GlideSpace::Pitch => {
                                    midi_to_hz_12tet(interpolated, PITCH_SPACE_A4_HZ)
                                }
                            }
                        };
                    }
                }
            }

            if ch == 0 {
                self.value = value;
                self.samples_remaining = remaining;
            }
        }
    }

    fn set_value(&mut self, value: f32) -> bool {
        self.value = value;
        self.target = value;
        self.increment = 0.0;
        self.samples_remaining = 0;
        true
    }

    fn set_target(
        &mut self,
        target: f32,
        glide_secs: f32,
        shape: GlideShape,
        space: GlideSpace,
    ) -> bool {
        if glide_secs <= 0.0 {
            return self.set_value(target);
        }
        let total_samples = (glide_secs * self.sample_rate) as u64;
        if total_samples == 0 {
            return self.set_value(target);
        }
        self.target = target;
        self.glide_shape = shape;
        self.glide_space = space;
        self.total_samples = total_samples;
        self.samples_remaining = total_samples;
        match space {
            GlideSpace::Raw => {
                self.increment = (target - self.value) / total_samples as f32;
                self.glide_start = self.value;
                self.glide_target = target;
            }
            GlideSpace::Pitch => {
                self.increment = 0.0; // Unused outside the Linear+Raw fast path.
                self.glide_start = hz_to_midi_12tet(self.value, PITCH_SPACE_A4_HZ);
                self.glide_target = hz_to_midi_12tet(target, PITCH_SPACE_A4_HZ);
            }
        }
        true
    }
}

/// Which binary operation to perform.
#[derive(Debug, Clone, Copy)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
}

/// Binary operation UGen: applies an operation to two inputs.
pub struct BinOpUGen {
    kind: BinOpKind,
}

impl BinOpUGen {
    pub fn new(kind: BinOpKind) -> Self {
        BinOpUGen { kind }
    }
}

static BINOP_INPUTS: [InputSpec; 2] = [
    InputSpec {
        name: "a",
        rate: Rate::Audio,
        required: true,
    },
    InputSpec {
        name: "b",
        rate: Rate::Audio,
        required: true,
    },
];

static BINOP_OUTPUTS: [OutputSpec; 1] = [OutputSpec {
    name: "out",
    rate: Rate::Audio,
}];

impl UGen for BinOpUGen {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: match self.kind {
                BinOpKind::Add => "Add",
                BinOpKind::Sub => "Sub",
                BinOpKind::Mul => "Mul",
                BinOpKind::Div => "Div",
            },
            category: UGenCategory::Math,
            inputs: &BINOP_INPUTS,
            outputs: &BINOP_OUTPUTS,
        }
    }

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let name = self.spec().name;
        let a = require_input(inputs, 0, name, "a");
        let b = require_input(inputs, 1, name, "b");
        let op: fn(f32, f32) -> f32 = match self.kind {
            BinOpKind::Add => |a, b| a + b,
            BinOpKind::Sub => |a, b| a - b,
            BinOpKind::Mul => |a, b| a * b,
            BinOpKind::Div => |a, b| if b != 0.0 { a / b } else { 0.0 },
        };
        for ch in 0..output.num_channels() {
            let a_ch = ch % a.num_channels();
            let b_ch = ch % b.num_channels();
            let a_samples = a.channel(a_ch).samples();
            let b_samples = b.channel(b_ch).samples();
            let out = output.channel_mut(ch).samples_mut();
            for i in 0..out.len() {
                out[i] = op(a_samples[i], b_samples[i]);
            }
        }
    }
}

/// Negation UGen: outputs -input.
pub struct NegUGen;

impl UGen for NegUGen {
    ugen_spec!("Neg", category = Math, inputs = ["in"], outputs = ["out"]);

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let input = require_input(inputs, 0, self.spec().name, "in");
        for ch in 0..output.num_channels() {
            let in_ch = ch % input.num_channels();
            let in_samples = input.channel(in_ch).samples();
            let out = output.channel_mut(ch).samples_mut();
            for i in 0..out.len() {
                out[i] = -in_samples[i];
            }
        }
    }
}
