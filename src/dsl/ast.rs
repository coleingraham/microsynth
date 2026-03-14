//! Abstract syntax tree for the microsynth DSL.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

/// A complete DSL program: SynthDef declarations, bus declarations, and route declarations.
#[derive(Debug, Clone)]
pub struct Program {
    pub defs: Vec<SynthDefDecl>,
    pub buses: Vec<BusDecl>,
    pub routes: Vec<RouteDecl>,
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
