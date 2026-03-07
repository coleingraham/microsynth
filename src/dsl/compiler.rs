//! Compiles DSL AST into SynthDef templates.

use crate::dsl::ast::{BinOp, Expr, SynthDefDecl};
use crate::node::{InputSpec, OutputSpec, UGen};
use crate::synthdef::{SynthDef, SynthDefBuilder};
use crate::ugens;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Metadata about a registered UGen type for the compiler.
#[derive(Clone)]
pub struct UGenEntry {
    /// Factory that creates a fresh instance.
    pub factory: fn() -> Box<dyn UGen>,
    /// Input port names (in order). The compiler maps positional args to these.
    pub input_names: Vec<&'static str>,
    /// Output port names.
    pub output_names: Vec<&'static str>,
}

/// Registry of available UGen types, keyed by name.
pub struct UGenRegistry {
    entries: BTreeMap<String, UGenEntry>,
}

impl UGenRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        UGenRegistry {
            entries: BTreeMap::new(),
        }
    }

    /// Register a UGen type.
    ///
    /// `name` is the identifier used in DSL source (e.g. "sinOsc").
    /// `factory` creates a default instance.
    /// `inputs` and `outputs` describe the port specs.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        factory: fn() -> Box<dyn UGen>,
        inputs: &[InputSpec],
        outputs: &[OutputSpec],
    ) {
        let input_names = inputs.iter().map(|i| i.name).collect();
        let output_names = outputs.iter().map(|o| o.name).collect();
        self.entries.insert(
            name.into(),
            UGenEntry {
                factory,
                input_names,
                output_names,
            },
        );
    }

    fn get(&self, name: &str) -> Option<&UGenEntry> {
        self.entries.get(name)
    }
}

impl Default for UGenRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiler state for a single SynthDef.
struct Compiler<'a> {
    builder: SynthDefBuilder,
    registry: &'a UGenRegistry,
    /// Maps variable names to node indices in the builder.
    scope: BTreeMap<String, usize>,
}

impl<'a> Compiler<'a> {
    fn new(name: &str, registry: &'a UGenRegistry) -> Self {
        Compiler {
            builder: SynthDefBuilder::new(name),
            registry,
            scope: BTreeMap::new(),
        }
    }

    /// Compile a SynthDefDecl into a SynthDef.
    fn compile(mut self, decl: &SynthDefDecl) -> Result<SynthDef, CompileError> {
        // Create Param nodes for each parameter. Param supports both instant
        // set_value and smooth set_target (glide/portamento) for continuous
        // control of parameters like freq, amp, filter cutoff, etc.
        for param in &decl.params {
            let value = param.default;
            let idx = self.builder.add_node(move || Box::new(ugens::Param::new(value)));
            self.builder.param(param.name.clone(), idx, 0);
            self.scope.insert(param.name.clone(), idx);
        }

        // Compile the body expression
        let output_idx = self.compile_expr(&decl.body)?;
        self.builder.set_output(output_idx);

        Ok(self.builder.build())
    }

    /// Compile an expression, returning the node index of its output.
    fn compile_expr(&mut self, expr: &Expr) -> Result<usize, CompileError> {
        match expr {
            Expr::Lit(value) => {
                let v = *value;
                Ok(self.builder.add_node(move || Box::new(ugens::Const::new(v))))
            }

            Expr::Var(name) => {
                if let Some(&idx) = self.scope.get(name) {
                    Ok(idx)
                } else if let Some(entry) = self.registry.get(name) {
                    // Zero-argument UGen (e.g. whiteNoise, pinkNoise)
                    if entry.input_names.is_empty() {
                        let factory = entry.factory;
                        Ok(self.builder.add_node(move || factory()))
                    } else {
                        Err(CompileError {
                            message: alloc::format!(
                                "{name} requires {} arguments",
                                entry.input_names.len()
                            ),
                        })
                    }
                } else {
                    Err(CompileError {
                        message: alloc::format!("undefined variable: {name}"),
                    })
                }
            }

            Expr::App(func_name, args) => {
                let entry = self
                    .registry
                    .get(func_name)
                    .ok_or_else(|| CompileError {
                        message: alloc::format!("unknown UGen: {func_name}"),
                    })?
                    .clone();

                if args.len() > entry.input_names.len() {
                    return Err(CompileError {
                        message: alloc::format!(
                            "{func_name} expects {} arguments, got {}",
                            entry.input_names.len(),
                            args.len()
                        ),
                    });
                }

                let factory = entry.factory;
                let node_idx = self.builder.add_node(move || factory());

                // Connect each argument to the corresponding input
                for (i, arg) in args.iter().enumerate() {
                    let arg_idx = self.compile_expr(arg)?;
                    self.builder.connect(arg_idx, node_idx, i);
                }

                Ok(node_idx)
            }

            Expr::BinOp(op, lhs, rhs) => {
                let lhs_idx = self.compile_expr(lhs)?;
                let rhs_idx = self.compile_expr(rhs)?;

                let kind = match op {
                    BinOp::Add => ugens::BinOpKind::Add,
                    BinOp::Sub => ugens::BinOpKind::Sub,
                    BinOp::Mul => ugens::BinOpKind::Mul,
                    BinOp::Div => ugens::BinOpKind::Div,
                };

                let node_idx = self
                    .builder
                    .add_node(move || Box::new(ugens::BinOpUGen::new(kind)));
                self.builder.connect(lhs_idx, node_idx, 0); // input a
                self.builder.connect(rhs_idx, node_idx, 1); // input b

                Ok(node_idx)
            }

            Expr::Neg(inner) => {
                let inner_idx = self.compile_expr(inner)?;
                let neg_idx = self.builder.add_node(|| Box::new(ugens::NegUGen));
                self.builder.connect(inner_idx, neg_idx, 0);
                Ok(neg_idx)
            }

            Expr::Let(bindings, body) => {
                for binding in bindings {
                    let idx = self.compile_expr(&binding.value)?;
                    self.scope.insert(binding.name.clone(), idx);
                }
                self.compile_expr(body)
            }
        }
    }
}

/// Compile a single SynthDefDecl into a SynthDef.
pub fn compile_synthdef(
    decl: &SynthDefDecl,
    registry: &UGenRegistry,
) -> Result<SynthDef, CompileError> {
    let compiler = Compiler::new(&decl.name, registry);
    compiler.compile(decl)
}

/// Compile all synthdefs in a program.
pub fn compile_program(
    program: &crate::dsl::ast::Program,
    registry: &UGenRegistry,
) -> Result<Vec<SynthDef>, CompileError> {
    program
        .defs
        .iter()
        .map(|decl| compile_synthdef(decl, registry))
        .collect()
}

/// A compilation error.
#[derive(Debug, Clone)]
pub struct CompileError {
    pub message: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "compile error: {}", self.message)
    }
}
