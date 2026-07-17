//! A versioned, serializable SynthDef intermediate representation.
//!
//! A compiled [`SynthDef`](crate::synthdef::SynthDef) hides its node kinds
//! inside factory closures (`synthdef.rs`) — it can render, but it cannot be
//! inspected, serialized, hashed, or programmatically edited. The IR is the
//! inspectable, serializable form of the same graph, and the intended
//! interchange contract described in `haskell/COEXISTENCE.md` ("one format,
//! many producers, one consumer").
//!
//! **Caveat: that goal is not yet met.** The Haskell engine ships its own
//! "version 1" IR (`Microsynth.SynthDef.IR`) with a different node model —
//! inline per-node `inputs` instead of an edge list, spec-style kind tags
//! (`"Saw"`) instead of registry names (`"saw"`), one `BinOp` kind instead of
//! four, and no `class`/`output_channels`. The two are mutually unparseable
//! despite sharing a version number. Unifying them is an open design decision;
//! see COEXISTENCE.md's "Current state — two IRs, one version number".
//!
//! ```text
//!   DSL text  ──parse──►  AST  ──[from_decl]──►  IrSynthDef  ──[compile]──►  SynthDef
//! ```
//!
//! [`from_decl`] decompiles a parsed DSL declaration into an `IrSynthDef` as a
//! faithful 1:1 mirror of the graph the DSL compiler builds (same nodes, same
//! order, same edges). [`IrSynthDef::compile`] rebuilds a `SynthDef` from the
//! IR via the same [`SynthDefBuilder`](crate::synthdef::SynthDefBuilder) path
//! the DSL compiler uses, so `DSL → SynthDef` and `DSL → IR → SynthDef` render
//! byte-identically by construction.

use crate::dsl::ast::{BinOp, Expr, SynthDefDecl};
use crate::dsl::compiler::UGenRegistry;
use crate::synthdef::{SynthDef, SynthDefBuilder};
use crate::ugens::{self, BinOpKind};
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

#[cfg(feature = "std")]
mod render;
mod serialize;

#[cfg(feature = "std")]
pub use render::{RenderSpec, render_ir};
pub use serialize::IrCodecError;

/// Current on-disk / on-wire format version. Bump on any semantic change to the
/// byte layout or node model.
pub const FORMAT_VERSION: u16 = 1;

/// The structural class of a SynthDef — a discriminant carried in the wire
/// format from day one so adding classes later is not a breaking bump. Only
/// the engine-general structural rules are validated here; a consumer may
/// layer its own policy on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthDefClass {
    /// A tone generator: no audio input, mono output. The only class produced
    /// by the DSL today.
    Source,
    /// An insert effect: one or more audio inputs. Reserved — validated
    /// structurally but not otherwise exercised in this crate.
    Effect,
}

/// A node in the IR graph. Index-addressed by position in `IrSynthDef::nodes`.
#[derive(Debug, Clone, PartialEq)]
pub enum IrNode {
    /// A UGen instance. `kind` is the registry name (e.g. `"sinOsc"`) or a core
    /// arithmetic kind (`"Add"`/`"Sub"`/`"Mul"`/`"Div"`/`"Neg"`). `consts` are
    /// literal values baked into the given input ports (empty for
    /// DSL-decompiled IR, where literals are separate `Const` nodes); `compile`
    /// materializes them as `Const` nodes wired to those inputs.
    UGen {
        kind: String,
        consts: Vec<(u32, f32)>,
    },
    /// A constant-value node (0 inputs, 1 output).
    Const(f32),
    /// A named, controllable parameter node (0 inputs, 1 output).
    Param { name: String, default: f32 },
}

/// A connection: `from`'s output feeds `to`'s input port `to_input`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrEdge {
    pub from: usize,
    pub to: usize,
    pub to_input: usize,
}

/// A named parameter binding: `name` addresses input `input` of node `node`.
#[derive(Debug, Clone, PartialEq)]
pub struct IrParam {
    pub name: String,
    pub node: usize,
    pub input: usize,
    pub default: f32,
}

/// The inspectable, serializable form of a synthesis graph.
#[derive(Debug, Clone, PartialEq)]
pub struct IrSynthDef {
    pub format_version: u16,
    pub name: String,
    pub class: SynthDefClass,
    /// Declared output channel count (1 = mono). A shell constraint; `1` for
    /// everything the DSL produces today.
    pub output_channels: u16,
    pub nodes: Vec<IrNode>,
    pub edges: Vec<IrEdge>,
    pub params: Vec<IrParam>,
    /// Audio input nodes `(name, node_index)` — non-empty only for effects.
    pub audio_inputs: Vec<(String, usize)>,
    pub output_node: usize,
}

/// Errors from IR validation.
#[derive(Debug, Clone, PartialEq)]
pub enum IrError {
    /// A UGen node references a kind that is neither a core kind nor registered.
    UnknownKind(String),
    /// An edge or param references a node index outside `nodes`.
    NodeOutOfRange { at: &'static str, index: usize },
    /// An edge (or inline const) targets an input port outside the kind's arity.
    InputOutOfRange {
        node: usize,
        input: usize,
        arity: usize,
    },
    /// The output node index is outside `nodes`.
    OutputOutOfRange(usize),
    /// The graph contains a cycle (not a DAG).
    Cycle,
    /// A param entry does not reference a `Param` node.
    ParamNotAParamNode { param: String, node: usize },
    /// A `Source` has an audio input, or an `Effect` has none.
    ShellViolation(String),
}

impl fmt::Display for IrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrError::UnknownKind(k) => write!(f, "unknown UGen kind: {k}"),
            IrError::NodeOutOfRange { at, index } => {
                write!(f, "{at} references out-of-range node index {index}")
            }
            IrError::InputOutOfRange { node, input, arity } => write!(
                f,
                "node {node}: input port {input} out of range (arity {arity})"
            ),
            IrError::OutputOutOfRange(i) => write!(f, "output node index {i} out of range"),
            IrError::Cycle => write!(f, "graph contains a cycle (not a DAG)"),
            IrError::ParamNotAParamNode { param, node } => {
                write!(
                    f,
                    "param {param:?} references node {node}, which is not a Param"
                )
            }
            IrError::ShellViolation(msg) => write!(f, "shell violation: {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Kind classification
// ---------------------------------------------------------------------------

/// The DSL's core arithmetic kinds: canonical IR kind name paired with the AST
/// [`BinOp`]. Single source of truth for the name↔op mapping the decompiler,
/// compiler, and validator share — so the four names live in exactly one place.
///
/// The op→engine-[`BinOpKind`] half of the mapping is *not* restated here; it
/// lives on [`BinOp::kind`], which the DSL compiler uses too.
const BINOPS: [(&str, BinOp); 4] = [
    ("Add", BinOp::Add),
    ("Sub", BinOp::Sub),
    ("Mul", BinOp::Mul),
    ("Div", BinOp::Div),
];

/// The unary-negation core kind (1 input).
const NEG_KIND: &str = "Neg";
/// The audio-input pass-through core kind.
const AUDIO_IN_KIND: &str = "audioIn";

/// The engine [`BinOpKind`] for a kind name, if it is one of the core binops.
fn binop_kind(name: &str) -> Option<BinOpKind> {
    BINOPS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|&(_, op)| op.kind())
}

/// The canonical kind name for an AST [`BinOp`].
fn binop_name(op: BinOp) -> &'static str {
    BINOPS
        .iter()
        .find(|(_, o)| *o == op)
        .map(|&(n, _)| n)
        .expect("BINOPS covers every BinOp variant")
}

/// Input arity of a core kind (binops = 2, `Neg` = 1), or `None` if not core.
fn core_kind_arity(kind: &str) -> Option<usize> {
    if binop_kind(kind).is_some() {
        Some(2)
    } else if kind == NEG_KIND {
        Some(1)
    } else {
        None
    }
}

/// Input arity of a UGen kind: core arithmetic, or the registry's port count.
fn kind_arity(reg: &UGenRegistry, kind: &str) -> Option<usize> {
    core_kind_arity(kind).or_else(|| reg.entry(kind).map(|e| e.input_names.len()))
}

/// Bounds-check a referenced node index, labelling the referrer for the error.
fn require_node(node_count: usize, index: usize, at: &'static str) -> Result<(), IrError> {
    if index >= node_count {
        Err(IrError::NodeOutOfRange { at, index })
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

impl IrSynthDef {
    /// Structural validation against a registry. Checks: every kind is known;
    /// edge/param node indices and input ports are in range; the output node
    /// exists; the graph is a DAG; params reference `Param` nodes; and the
    /// engine-general half of the class shell (`Source` has no audio input,
    /// `Effect` has at least one).
    pub fn validate(&self, reg: &UGenRegistry) -> Result<(), IrError> {
        let n = self.nodes.len();

        // Node kinds and inline-const ports.
        for (i, node) in self.nodes.iter().enumerate() {
            if let IrNode::UGen { kind, consts } = node {
                let arity =
                    kind_arity(reg, kind).ok_or_else(|| IrError::UnknownKind(kind.clone()))?;
                for &(input, _) in consts {
                    if input as usize >= arity {
                        return Err(IrError::InputOutOfRange {
                            node: i,
                            input: input as usize,
                            arity,
                        });
                    }
                }
            }
        }

        // Edges: endpoints in range, target port within destination arity.
        for e in &self.edges {
            require_node(n, e.from, "edge.from")?;
            require_node(n, e.to, "edge.to")?;
            let arity = self.node_arity(reg, e.to)?;
            if e.to_input >= arity {
                return Err(IrError::InputOutOfRange {
                    node: e.to,
                    input: e.to_input,
                    arity,
                });
            }
        }

        // Output node.
        if self.output_node >= n {
            return Err(IrError::OutputOutOfRange(self.output_node));
        }

        // Params reference Param nodes.
        for p in &self.params {
            require_node(n, p.node, "param.node")?;
            if !matches!(self.nodes[p.node], IrNode::Param { .. }) {
                return Err(IrError::ParamNotAParamNode {
                    param: p.name.clone(),
                    node: p.node,
                });
            }
        }

        // Audio inputs reference nodes in range.
        for (_, node) in &self.audio_inputs {
            require_node(n, *node, "audio_input")?;
        }

        self.check_acyclic()?;
        self.check_shell()?;
        Ok(())
    }

    /// Input arity of node `i` (Const/Param = 0).
    fn node_arity(&self, reg: &UGenRegistry, i: usize) -> Result<usize, IrError> {
        match &self.nodes[i] {
            IrNode::Const(_) | IrNode::Param { .. } => Ok(0),
            IrNode::UGen { kind, .. } => {
                kind_arity(reg, kind).ok_or_else(|| IrError::UnknownKind(kind.clone()))
            }
        }
    }

    /// Kahn's algorithm over the edge dependencies; a leftover node means a cycle.
    fn check_acyclic(&self) -> Result<(), IrError> {
        let n = self.nodes.len();
        let mut in_degree = alloc::vec![0usize; n];
        for e in &self.edges {
            in_degree[e.to] += 1;
        }
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut visited = 0usize;
        while let Some(node) = queue.pop() {
            visited += 1;
            for e in self.edges.iter().filter(|e| e.from == node) {
                in_degree[e.to] -= 1;
                if in_degree[e.to] == 0 {
                    queue.push(e.to);
                }
            }
        }
        if visited == n {
            Ok(())
        } else {
            Err(IrError::Cycle)
        }
    }

    /// The engine-general half of the class shell.
    fn check_shell(&self) -> Result<(), IrError> {
        match self.class {
            SynthDefClass::Source => {
                if !self.audio_inputs.is_empty() {
                    return Err(IrError::ShellViolation(
                        "Source must have no audio inputs".to_string(),
                    ));
                }
            }
            SynthDefClass::Effect => {
                if self.audio_inputs.is_empty() {
                    return Err(IrError::ShellViolation(
                        "Effect must have at least one audio input".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Compile: IR -> SynthDef
// ---------------------------------------------------------------------------

impl IrSynthDef {
    /// Rebuild a renderable [`SynthDef`] from this IR, via the same
    /// [`SynthDefBuilder`] path the DSL compiler uses. The base nodes keep their
    /// IR indices (so edges stay valid); inline `consts` are materialized as
    /// extra `Const` nodes appended afterwards.
    ///
    /// Assumes the IR is valid; call [`validate`](Self::validate) first if the
    /// IR comes from an untrusted source.
    pub fn compile(&self, reg: &UGenRegistry) -> Result<SynthDef, IrError> {
        let mut builder = SynthDefBuilder::new(self.name.clone());

        // Base nodes, in IR index order.
        for node in &self.nodes {
            match node {
                IrNode::Const(v) => {
                    let v = *v;
                    builder.add_node(move || Box::new(ugens::Const::new(v)));
                }
                IrNode::Param { default, .. } => {
                    let v = *default;
                    builder.add_node(move || Box::new(ugens::Param::new(v)));
                }
                IrNode::UGen { kind, .. } => {
                    add_ugen_node(&mut builder, reg, kind)?;
                }
            }
        }

        // Wired edges (reference base-node IR indices).
        for e in &self.edges {
            builder.connect(e.from, e.to, e.to_input);
        }

        // Inline consts → appended Const nodes wired to their input ports.
        for (i, node) in self.nodes.iter().enumerate() {
            if let IrNode::UGen { consts, .. } = node {
                for &(input, value) in consts {
                    let const_idx = builder.add_node(move || Box::new(ugens::Const::new(value)));
                    builder.connect(const_idx, i, input as usize);
                }
            }
        }

        // Params and audio inputs.
        for p in &self.params {
            builder.param(p.name.clone(), p.node, p.input);
        }
        for (name, node) in &self.audio_inputs {
            builder.audio_input(name.clone(), *node);
        }

        builder.set_output(self.output_node);
        Ok(builder.build())
    }
}

/// Add a UGen node for `kind`, resolving core arithmetic kinds directly and
/// everything else through the registry.
fn add_ugen_node(
    builder: &mut SynthDefBuilder,
    reg: &UGenRegistry,
    kind: &str,
) -> Result<usize, IrError> {
    if let Some(bk) = binop_kind(kind) {
        return Ok(builder.add_node(move || Box::new(ugens::BinOpUGen::new(bk))));
    }
    if kind == NEG_KIND {
        return Ok(builder.add_node(|| Box::new(ugens::NegUGen)));
    }
    let factory = reg
        .entry(kind)
        .ok_or_else(|| IrError::UnknownKind(kind.to_string()))?
        .factory;
    Ok(builder.add_node(move || factory()))
}

// ---------------------------------------------------------------------------
// Decompile: AST -> IR
// ---------------------------------------------------------------------------

/// Decompile a parsed DSL declaration into an `IrSynthDef`, mirroring the DSL
/// compiler's node/edge construction exactly (same order, same topology).
///
/// `reg` is consulted only to resolve zero-argument UGen variables (e.g.
/// `whiteNoise` used bare), matching the compiler.
pub fn from_decl(decl: &SynthDefDecl, reg: &UGenRegistry) -> IrSynthDef {
    let mut b = IrBuilder::new(reg);

    // Param nodes first, exactly as the compiler does.
    for param in &decl.params {
        let idx = b.push(IrNode::Param {
            name: param.name.clone(),
            default: param.default,
        });
        b.params.push(IrParam {
            name: param.name.clone(),
            node: idx,
            input: 0,
            default: param.default,
        });
        b.scope.insert(param.name.clone(), idx);
    }

    let output_node = b.compile_expr(&decl.body);

    let class = if b.audio_inputs.is_empty() {
        SynthDefClass::Source
    } else {
        SynthDefClass::Effect
    };

    IrSynthDef {
        format_version: FORMAT_VERSION,
        name: decl.name.clone(),
        class,
        output_channels: 1,
        nodes: b.nodes,
        edges: b.edges,
        params: b.params,
        audio_inputs: b.audio_inputs,
        output_node,
    }
}

/// Mirrors `dsl::compiler::Compiler`, emitting IR nodes instead of factories.
struct IrBuilder<'a> {
    reg: &'a UGenRegistry,
    nodes: Vec<IrNode>,
    edges: Vec<IrEdge>,
    params: Vec<IrParam>,
    audio_inputs: Vec<(String, usize)>,
    scope: BTreeMap<String, usize>,
}

impl<'a> IrBuilder<'a> {
    fn new(reg: &'a UGenRegistry) -> Self {
        IrBuilder {
            reg,
            nodes: Vec::new(),
            edges: Vec::new(),
            params: Vec::new(),
            audio_inputs: Vec::new(),
            scope: BTreeMap::new(),
        }
    }

    fn push(&mut self, node: IrNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    fn connect(&mut self, from: usize, to: usize, to_input: usize) {
        self.edges.push(IrEdge { from, to, to_input });
    }

    /// Push a UGen node with no inline consts (the only shape decompile emits).
    fn push_ugen(&mut self, kind: impl Into<String>) -> usize {
        self.push(IrNode::UGen {
            kind: kind.into(),
            consts: Vec::new(),
        })
    }

    /// Mirror of `Compiler::compile_expr` — the same AST traversal, emitting IR
    /// nodes instead of graph nodes. The two must stay in lockstep; the check
    /// that enforces it is the byte-identical render test in `tests/ir.rs`
    /// (`DSL → SynthDef` vs `DSL → IR → SynthDef`). Touch one, run that test.
    ///
    /// **This duplication is deliberate.** The obvious dedup — routing the DSL
    /// compiler through the IR so this traversal exists once — would make `dsl`
    /// depend on `ir`, and `ir` is an optional feature that both WASM builds
    /// turn off (`web/build.sh` passes `--no-default-features`). Unifying them
    /// would force the IR into every WASM bundle, which the crate's
    /// `opt-level = "s"` + LTO profile exists to avoid. The duplicate traversal
    /// is the cheaper of the two costs.
    fn compile_expr(&mut self, expr: &Expr) -> usize {
        match expr {
            Expr::Lit(v) => self.push(IrNode::Const(*v)),

            Expr::Var(name) => {
                if let Some(&idx) = self.scope.get(name) {
                    idx
                } else if name == AUDIO_IN_KIND {
                    let idx = self.push_ugen(AUDIO_IN_KIND);
                    self.audio_inputs.push(("in".to_string(), idx));
                    idx
                } else {
                    // Zero-argument UGen (e.g. whiteNoise). The declaration
                    // already compiled successfully, so this is registered;
                    // consult the registry to mirror the compiler's check.
                    debug_assert!(
                        self.reg.entry(name).is_some(),
                        "unknown bare identifier in decompile: {name}"
                    );
                    self.push_ugen(name.clone())
                }
            }

            Expr::App(func_name, args) => {
                let node_idx = self.push_ugen(func_name.clone());
                for (i, arg) in args.iter().enumerate() {
                    let arg_idx = self.compile_expr(arg);
                    self.connect(arg_idx, node_idx, i);
                }
                node_idx
            }

            Expr::BinOp(op, lhs, rhs) => {
                let lhs_idx = self.compile_expr(lhs);
                let rhs_idx = self.compile_expr(rhs);
                let node_idx = self.push_ugen(binop_name(*op));
                self.connect(lhs_idx, node_idx, 0);
                self.connect(rhs_idx, node_idx, 1);
                node_idx
            }

            Expr::Neg(inner) => {
                let inner_idx = self.compile_expr(inner);
                let node_idx = self.push_ugen(NEG_KIND);
                self.connect(inner_idx, node_idx, 0);
                node_idx
            }

            Expr::Let(bindings, body) => {
                for binding in bindings {
                    let idx = self.compile_expr(&binding.value);
                    self.scope.insert(binding.name.clone(), idx);
                }
                self.compile_expr(body)
            }
        }
    }
}
