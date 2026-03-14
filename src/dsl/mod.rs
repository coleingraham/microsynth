//! Text-based DSL for defining synthesis graphs.
//!
//! The DSL has a Haskell-inspired syntax. A synthdef declaration looks like:
//!
//! ```text
//! synthdef pad freq=440.0 amp=0.5 =
//!   let osc = sinOsc freq 0.0
//!   let env = envGen 0.01 1.0
//!   osc * env * amp
//! ```
//!
//! # Syntax
//!
//! - `synthdef NAME PARAM=DEFAULT ... = BODY` — declare a synthesis graph
//! - `let NAME = EXPR` — bind a sub-expression to a name
//! - Function application is by juxtaposition: `sinOsc freq 0.0`
//! - Arithmetic operators: `+`, `-`, `*`, `/`
//! - Parentheses for grouping: `(freq + 1.0) * 2.0`
//! - Comments: `-- this is a comment`
//!
//! # Usage
//!
//! ```rust,ignore
//! use microsynth::dsl::{compile, UGenRegistry};
//!
//! let mut registry = UGenRegistry::new();
//! // register your UGens...
//!
//! let defs = compile("synthdef test x=1.0 = x * 2.0", &registry).unwrap();
//! ```

pub mod ast;
pub mod compiler;
pub mod lexer;
pub mod parser;

pub use compiler::{compile_program, compile_routing, compile_synthdef, CompileError, UGenEntry, UGenRegistry};
pub use lexer::LexError;
pub use parser::ParseError;

use crate::routing::RoutingGraph;
use crate::synthdef::SynthDef;
use alloc::vec::Vec;
use core::fmt;

/// Unified error type for the DSL pipeline.
#[derive(Debug, Clone)]
pub enum DslError {
    Lex(LexError),
    Parse(ParseError),
    Compile(CompileError),
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslError::Lex(e) => write!(f, "{e}"),
            DslError::Parse(e) => write!(f, "{e}"),
            DslError::Compile(e) => write!(f, "{e}"),
        }
    }
}

impl From<LexError> for DslError {
    fn from(e: LexError) -> Self {
        DslError::Lex(e)
    }
}

impl From<ParseError> for DslError {
    fn from(e: ParseError) -> Self {
        DslError::Parse(e)
    }
}

impl From<CompileError> for DslError {
    fn from(e: CompileError) -> Self {
        DslError::Compile(e)
    }
}

/// Parse and compile DSL source into SynthDefs.
///
/// This is the main entry point: tokenize → parse → compile.
/// Bus and route declarations are ignored; use `compile_with_routing`
/// to also produce a routing graph.
pub fn compile(source: &str, registry: &UGenRegistry) -> Result<Vec<SynthDef>, DslError> {
    let tokens = lexer::tokenize(source)?;
    let mut parser = parser::Parser::new(tokens);
    let program = parser.parse_program()?;
    let defs = compile_program(&program, registry)?;
    Ok(defs)
}

/// Parse and compile DSL source into SynthDefs and a RoutingGraph.
///
/// Returns `(synthdefs, routing_graph)`. The routing graph contains
/// bus and route declarations from the source, with effect references
/// resolved against the compiled SynthDefs.
pub fn compile_with_routing(
    source: &str,
    registry: &UGenRegistry,
) -> Result<(Vec<SynthDef>, RoutingGraph), DslError> {
    let tokens = lexer::tokenize(source)?;
    let mut parser = parser::Parser::new(tokens);
    let program = parser.parse_program()?;
    let defs = compile_program(&program, registry)?;
    let routing = compile_routing(&program, &defs)?;
    Ok((defs, routing))
}
