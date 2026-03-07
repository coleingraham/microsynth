//! Built-in UGens used by the DSL compiler and available for direct use.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};

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
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "Const",
            inputs: &[],
            outputs: &[OutputSpec {
                name: "out",
                rate: Rate::Audio,
            }],
        }
    }

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn process(
        &mut self,
        _context: &ProcessContext,
        _inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        for ch in 0..output.num_channels() {
            output.channel_mut(ch).fill(self.value);
        }
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
    },
    InputSpec {
        name: "b",
        rate: Rate::Audio,
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
            inputs: &BINOP_INPUTS,
            outputs: &BINOP_OUTPUTS,
        }
    }

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let a = inputs[0];
        let b = inputs[1];
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

static NEG_INPUTS: [InputSpec; 1] = [InputSpec {
    name: "in",
    rate: Rate::Audio,
}];

static NEG_OUTPUTS: [OutputSpec; 1] = [OutputSpec {
    name: "out",
    rate: Rate::Audio,
}];

impl UGen for NegUGen {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "Neg",
            inputs: &NEG_INPUTS,
            outputs: &NEG_OUTPUTS,
        }
    }

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let input = inputs[0];
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
