//! Abstract syntax tree for the microsynth DSL.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// A complete DSL program: SynthDef declarations, bus declarations, route
/// declarations, and voice-mode declarations.
#[derive(Debug, Clone)]
pub struct Program {
    pub defs: Vec<SynthDefDecl>,
    pub buses: Vec<BusDecl>,
    pub routes: Vec<RouteDecl>,
    pub voice_modes: Vec<VoiceModeDecl>,
}

/// A bus declaration: `bus NAME CHANNELS`
#[derive(Debug, Clone)]
pub struct BusDecl {
    pub name: String,
    /// Number of audio channels (e.g. 2 for stereo).
    pub channels: usize,
}

/// A route declaration: `route SOURCE => EFFECT => TARGET`
///
/// The chain is a list of names: [source_bus, effect1, ..., target_bus].
/// The first and last entries are bus names; middle entries are effect SynthDef names.
#[derive(Debug, Clone)]
pub struct RouteDecl {
    pub chain: Vec<String>,
}

/// A SynthDef declaration.
///
/// ```text
/// synthdef pad freq=440.0 amp=0.5 =
///   let osc = sinOsc freq 0.0
///   osc * amp
/// ```
#[derive(Debug, Clone)]
pub struct SynthDefDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub body: Expr,
}

/// A named parameter with a default value.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub default: f32,
}

/// A mono/legato voice-mode declaration:
/// `voice NAME mono legato PITCH_PARAM PORTAMENTO_SECS`
///
/// Declares that instances of the named SynthDef should be played on a
/// single mono/legato track (see [`crate::voice::LegatoVoice`]) rather than
/// as independent polyphonic voices: a note that overlaps or abuts the
/// currently held one glides `pitch_param` to the new value over
/// `portamento_secs` and leaves the envelope's gate open (no re-attack),
/// while a gap between notes retriggers a fresh attack on the same voice.
///
/// This only names *that* a SynthDef is played this way; the words `mono`
/// and `legato` are literal syntax (not currently configurable) that keep
/// the declaration's intent readable and leave room for other voice modes
/// to be added to this position later without a grammar change.
#[derive(Debug, Clone)]
pub struct VoiceModeDecl {
    /// Name of the SynthDef this declaration applies to.
    pub synth_name: String,
    /// Name of the SynthDef parameter driven by note pitch (e.g. `"freq"`).
    pub pitch_param: String,
    /// Glide time used when tying two notes together, in seconds.
    pub portamento_secs: f32,
}

/// A let-binding: `name = expr`.
#[derive(Debug, Clone)]
pub struct Binding {
    pub name: String,
    pub value: Expr,
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinOp {
    /// The engine [`BinOpKind`](crate::ugens::BinOpKind) this operator compiles
    /// to. The single source of truth for the mapping: the DSL compiler and the
    /// IR decompiler both go through here rather than restating it.
    pub fn kind(self) -> crate::ugens::BinOpKind {
        use crate::ugens::BinOpKind;
        match self {
            BinOp::Add => BinOpKind::Add,
            BinOp::Sub => BinOpKind::Sub,
            BinOp::Mul => BinOpKind::Mul,
            BinOp::Div => BinOpKind::Div,
        }
    }
}

/// An expression in the DSL.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A numeric literal: `440.0`, `0.5`.
    Lit(f32),
    /// A variable reference: `freq`, `osc`.
    Var(String),
    /// Function application: `sinOsc freq 0.0`.
    /// The function name and its positional arguments.
    App(String, Vec<Expr>),
    /// Binary operation: `a + b`, `osc * amp`.
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    /// Unary negation: `-x`.
    Neg(Box<Expr>),
    /// Let bindings followed by a body expression.
    Let(Vec<Binding>, Box<Expr>),
}
