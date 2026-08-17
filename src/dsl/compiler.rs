//! Compiles DSL AST into SynthDef templates.

use crate::coeff_table::CoeffTable;
use crate::dsl::ast::{Expr, SynthDefDecl, VoiceModeDecl};
use crate::node::{InputSpec, OutputSpec, UGen, UGenCategory};
use crate::synthdef::{SynthDef, SynthDefBuilder};
use crate::ugens;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

/// Metadata about a registered UGen type for the compiler.
#[derive(Clone)]
pub struct UGenEntry {
    /// Factory that creates a fresh instance.
    pub factory: fn() -> Box<dyn UGen>,
    /// Coarse category of this UGen, always read from its own `spec()`.
    pub category: UGenCategory,
    /// Input port names (in order). The compiler maps positional args to these.
    pub input_names: Vec<&'static str>,
    /// Whether each input port (same order/length as `input_names`) is
    /// required — mirrors [`InputSpec::required`] for each port. Consulted by
    /// [`crate::ir::IrSynthDef::validate`] to reject a required port with no
    /// connected source (edge or inline const) before it ever reaches
    /// [`crate::graph::AudioGraph::prepare`], which enforces the same
    /// invariant for already-constructed graphs (see that function's doc).
    pub required: Vec<bool>,
    /// Output port names.
    pub output_names: Vec<&'static str>,
}

/// Factory for a UGen kind whose construction needs a runtime-filled
/// coefficient table, resolved by id against a [`crate::coeff_table::CoeffTableBank`]
/// at graph-build time. `Arc`, not a bare `fn` pointer: unlike [`UGenEntry`]'s
/// `factory`, this closure captures the resolved `Arc<CoeffTable>`, so it
/// cannot be a plain function pointer (bare `fn` items cannot close over
/// per-call data). This is the mechanism note in `src/ir/mod.rs`'s "Structural
/// gap" doc comment resolves: a DSL/IR-reachable UGen kind can now carry
/// runtime data, without touching how the ~40 existing bare-`fn`-factory
/// kinds are registered or constructed.
pub type TableUGenFactory = Arc<dyn Fn(Arc<CoeffTable>) -> Box<dyn UGen> + Send + Sync>;

/// Metadata about a registered table-bound UGen type — the [`UGenEntry`]
/// counterpart for kinds registered via
/// [`UGenRegistry::register_table_bound`].
#[derive(Clone)]
pub struct TableUGenEntry {
    /// Factory that creates an instance bound to a specific resolved table.
    pub factory: TableUGenFactory,
    /// Coarse category, as declared at registration (unlike [`UGenEntry`],
    /// there is no construction-time probe instance to read this from: the
    /// factory needs a table to build one).
    pub category: UGenCategory,
    /// Input port names (in order), covering this kind's ordinary
    /// audio/control-rate ports — the bound table is not itself a port.
    pub input_names: Vec<&'static str>,
    /// Whether each input port (same order/length as `input_names`) is
    /// required — see [`UGenEntry::required`]'s doc.
    pub required: Vec<bool>,
    /// Output port names.
    pub output_names: Vec<&'static str>,
}

/// Registry of available UGen types, keyed by name.
pub struct UGenRegistry {
    entries: BTreeMap<String, UGenEntry>,
    /// Table-bound kinds, disjoint from `entries` by name (a kind is either
    /// bare or table-bound, never both). See [`register_table_bound`](Self::register_table_bound).
    table_bound: BTreeMap<String, TableUGenEntry>,
}

impl UGenRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        UGenRegistry {
            entries: BTreeMap::new(),
            table_bound: BTreeMap::new(),
        }
    }

    /// Register a UGen type with explicit port specs.
    ///
    /// `name` is the identifier used in DSL source (e.g. "sinOsc").
    /// `factory` creates a default instance.
    /// `inputs` and `outputs` describe the port specs.
    ///
    /// Prefer [`register_spec`](Self::register_spec), which derives the ports
    /// from the UGen itself. This lower-level form is for callers that must
    /// supply ports that differ from `spec()` (e.g. test-only UGens). The
    /// category is still read from the UGen's own `spec()` — it is never
    /// something the call site can get wrong.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        factory: fn() -> Box<dyn UGen>,
        inputs: &[InputSpec],
        outputs: &[OutputSpec],
    ) {
        let input_names = inputs.iter().map(|i| i.name).collect();
        let required = inputs.iter().map(|i| i.required).collect();
        let output_names = outputs.iter().map(|o| o.name).collect();
        let category = factory().spec().category;
        self.entries.insert(
            name.into(),
            UGenEntry {
                factory,
                category,
                input_names,
                required,
                output_names,
            },
        );
    }

    /// Register a UGen using the port specs from its own `spec()`.
    ///
    /// This avoids re-declaring `InputSpec`/`OutputSpec` at the call site: the
    /// port names are read from a one-time probe instance built by `factory`.
    /// The probe is created off the render path (at registration time), so the
    /// allocation-free render invariant is untouched.
    ///
    /// The DSL `name` stays explicit because it deliberately differs from the
    /// UGen's internal `spec().name` (camelCase DSL identifier vs PascalCase
    /// type name), and several DSL names may map to the same UGen type
    /// (e.g. `sinTable`/`sawTable`/... all build a `WaveTable`).
    pub fn register_spec(&mut self, name: impl Into<String>, factory: fn() -> Box<dyn UGen>) {
        let probe = factory();
        let spec = probe.spec();
        let input_names = spec.inputs.iter().map(|i| i.name).collect();
        let required = spec.inputs.iter().map(|i| i.required).collect();
        let output_names = spec.outputs.iter().map(|o| o.name).collect();
        self.entries.insert(
            name.into(),
            UGenEntry {
                factory,
                category: spec.category,
                input_names,
                required,
                output_names,
            },
        );
    }

    fn get(&self, name: &str) -> Option<&UGenEntry> {
        self.entries.get(name)
    }

    /// Look up a registered UGen by name (its DSL/registry kind). Public so the
    /// [`ir`](crate::ir) layer can resolve kinds to factories and port arities.
    pub fn entry(&self, name: &str) -> Option<&UGenEntry> {
        self.entries.get(name)
    }

    /// Iterate over all registered `(name, entry)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &UGenEntry)> {
        self.entries.iter()
    }

    /// Register a table-bound UGen kind: one whose construction requires a
    /// resolved [`CoeffTable`], supplied at graph-build time (never at
    /// registration time — this factory is called once per resolved
    /// reference, not once here).
    ///
    /// `name` should not collide with a kind already registered via
    /// [`register`](Self::register)/[`register_spec`](Self::register_spec) —
    /// callers should keep the two name spaces disjoint by convention — but
    /// "disjoint by convention" is not enforced, and re-registering the
    /// *same* name in the *same* map (bare-bare or table_bound-table_bound)
    /// is unconditional last-write-wins, exactly like a plain `BTreeMap`
    /// insert. A **cross-namespace** collision (a name registered once via
    /// `register`/`register_spec` and once via `register_table_bound`) is
    /// different: `entries` and `table_bound` are separate maps, so neither
    /// registration overwrites the other — both persist, and the name is
    /// legal to resolve either way. What a colliding name resolves to then
    /// depends on **the node**, not on registration order, and both
    /// `crate::ir`'s private `kind_arity` helper (which validation's
    /// arity/port-range checks go through) and node construction
    /// (`IrSynthDef::compile`/`compile_with_tables`, via `build_base`) agree
    /// on the same rule (MOT-652): whether that specific node has an
    /// [`IrTableBinding`](crate::ir::IrTableBinding) decides it — bound nodes
    /// resolve/construct via the table-bound entry, unbound nodes via the
    /// bare one, regardless of which was registered first or second. Because
    /// both functions key off the same per-node fact, `validate()` never
    /// signs off on an arity that construction then contradicts. See `tests`
    /// at the bottom of this file for both the same-map and cross-namespace
    /// cases pinned explicitly, including the collision-dispatch case.
    pub fn register_table_bound(
        &mut self,
        name: impl Into<String>,
        factory: TableUGenFactory,
        category: UGenCategory,
        inputs: &[InputSpec],
        outputs: &[OutputSpec],
    ) {
        let input_names = inputs.iter().map(|i| i.name).collect();
        let required = inputs.iter().map(|i| i.required).collect();
        let output_names = outputs.iter().map(|o| o.name).collect();
        self.table_bound.insert(
            name.into(),
            TableUGenEntry {
                factory,
                category,
                input_names,
                required,
                output_names,
            },
        );
    }

    /// Look up a registered table-bound kind's metadata by name.
    pub fn table_bound_entry(&self, name: &str) -> Option<&TableUGenEntry> {
        self.table_bound.get(name)
    }

    /// Iterate over all registered table-bound `(name, entry)` pairs —
    /// the [`iter`](Self::iter) counterpart for the table-bound namespace.
    pub fn table_bound_iter(&self) -> impl Iterator<Item = (&String, &TableUGenEntry)> {
        self.table_bound.iter()
    }

    /// Build an instance of table-bound kind `name`, bound to `table`.
    /// Returns `None` if `name` is not registered as table-bound.
    pub fn resolve_table_bound(&self, name: &str, table: Arc<CoeffTable>) -> Option<Box<dyn UGen>> {
        self.table_bound.get(name).map(|e| (e.factory)(table))
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
            let idx = self
                .builder
                .add_node(move || Box::new(ugens::Param::new(value)));
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
                Ok(self
                    .builder
                    .add_node(move || Box::new(ugens::Const::new(v))))
            }

            Expr::Var(name) => {
                if let Some(&idx) = self.scope.get(name) {
                    Ok(idx)
                } else if name == "audioIn" {
                    // Special handling: audioIn creates an AudioIn pass-through node
                    // and marks it as an audio input on the SynthDef
                    let idx = self.builder.add_node(|| Box::new(ugens::AudioIn));
                    self.builder.audio_input("in", idx);
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

                // Positional args wire ports 0..args.len() as a prefix, leaving
                // the rest unconnected (that's how an optional trailing port,
                // e.g. a filter's "q", gets its own default). But a *required*
                // port left in that gap is not a default case — it is a graph
                // AudioGraph::prepare will refuse to render. Catch it here,
                // as a CompileError naming the ugen and the missing port,
                // instead of letting it reach that panic (or, before the
                // required/optional split existed, an out-of-bounds index
                // inside some UGen's own `process`).
                for i in args.len()..entry.input_names.len() {
                    if entry.required[i] {
                        return Err(CompileError {
                            message: alloc::format!(
                                "{func_name}: required argument '{}' (port {i}) not supplied — \
                                 got {} argument(s)",
                                entry.input_names[i],
                                args.len()
                            ),
                        });
                    }
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

                let kind = op.kind();

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

/// Compile a program's bus and route declarations into a RoutingGraph.
///
/// The `defs` parameter should be the SynthDefs compiled from the same program,
/// so that route declarations can reference effect SynthDefs by name.
pub fn compile_routing(
    program: &crate::dsl::ast::Program,
    defs: &[crate::synthdef::SynthDef],
) -> Result<crate::routing::RoutingGraph, CompileError> {
    let mut routing = crate::routing::RoutingGraph::new();

    // Create buses from declarations
    for bus_decl in &program.buses {
        routing.add_bus(bus_decl.name.clone(), bus_decl.channels);
    }

    // Process route declarations
    for route_decl in &program.routes {
        // chain: [source_bus, effect1, ..., effectN, target_bus]
        // Process consecutive triplets: bus => effect => bus
        let chain = &route_decl.chain;
        // Walk the chain in steps of 2: each pair (bus, effect) followed by next bus
        let mut i = 0;
        while i + 2 < chain.len() {
            let source_name = &chain[i];
            let effect_name = &chain[i + 1];
            let target_name = &chain[i + 2];

            let source_bus = routing
                .bus_by_name(source_name)
                .ok_or_else(|| CompileError {
                    message: alloc::format!("unknown bus in route: {source_name}"),
                })?;

            let target_bus = routing
                .bus_by_name(target_name)
                .ok_or_else(|| CompileError {
                    message: alloc::format!("unknown bus in route: {target_name}"),
                })?;

            let def = defs
                .iter()
                .find(|d| d.name() == effect_name)
                .ok_or_else(|| CompileError {
                    message: alloc::format!("unknown effect synthdef in route: {effect_name}"),
                })?;

            routing.add_effect(source_bus, def, target_bus);
            i += 2;
        }
    }

    Ok(routing)
}

/// Validate a program's voice-mode declarations against its compiled
/// SynthDefs.
///
/// Each declaration must name an existing SynthDef and an existing parameter
/// on it — the same kind of cross-reference check `compile_routing` already
/// does for bus/effect names, so a typo in either name is caught at compile
/// time instead of silently doing nothing at spawn time.
pub fn compile_voice_modes(
    program: &crate::dsl::ast::Program,
    defs: &[crate::synthdef::SynthDef],
) -> Result<Vec<VoiceModeDecl>, CompileError> {
    for decl in &program.voice_modes {
        let def = defs
            .iter()
            .find(|d| d.name() == decl.synth_name)
            .ok_or_else(|| CompileError {
                message: alloc::format!(
                    "unknown synthdef in voice declaration: {}",
                    decl.synth_name
                ),
            })?;
        if !def
            .param_names()
            .iter()
            .any(|(name, _, _)| name == &decl.pitch_param)
        {
            return Err(CompileError {
                message: alloc::format!(
                    "voice declaration for {} references unknown parameter: {}",
                    decl.synth_name,
                    decl.pitch_param
                ),
            });
        }
    }
    Ok(program.voice_modes.clone())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Rate;
    use crate::ugens::Const;

    fn bare_factory() -> Box<dyn UGen> {
        Box::new(Const::new(1.0))
    }

    fn out_spec() -> [OutputSpec; 1] {
        [OutputSpec {
            name: "out",
            rate: Rate::Audio,
        }]
    }

    fn in_spec() -> [InputSpec; 1] {
        [InputSpec {
            name: "in",
            rate: Rate::Audio,
            required: true,
        }]
    }

    /// MOT-649 F12: re-registering the same name in the *same* map
    /// (bare-bare) is unconditional last-write-wins, exactly like a plain
    /// `BTreeMap` insert -- made observable via arity, since `register`'s
    /// probe-derived `category` would otherwise hide which registration won.
    #[test]
    fn bare_bare_collision_is_last_write_wins() {
        let mut reg = UGenRegistry::new();
        reg.register("dup", bare_factory, &[], &out_spec());
        reg.register("dup", bare_factory, &in_spec(), &out_spec());
        assert_eq!(
            reg.entry("dup").unwrap().input_names,
            alloc::vec!["in"],
            "the second register() call should win over the first, same map"
        );
    }

    /// MOT-649 F12: same as above, for the table-bound map.
    #[test]
    fn table_bound_table_bound_collision_is_last_write_wins() {
        let mut reg = UGenRegistry::new();
        let factory: TableUGenFactory = Arc::new(|_table| Box::new(Const::new(1.0)));
        reg.register_table_bound(
            "dup",
            factory.clone(),
            UGenCategory::Utility,
            &[],
            &out_spec(),
        );
        reg.register_table_bound(
            "dup",
            factory,
            UGenCategory::Utility,
            &in_spec(),
            &out_spec(),
        );
        assert_eq!(
            reg.table_bound_entry("dup").unwrap().input_names,
            alloc::vec!["in"],
            "the second register_table_bound() call should win over the first, same map"
        );
    }

    /// MOT-649 F12: a *cross-namespace* collision (same name registered once
    /// via `register` and once via `register_table_bound`) does NOT
    /// overwrite either map -- both persist. This is the case the module doc
    /// warns is easy to misread as "last write wins": it isn't, at the
    /// registry level.
    #[test]
    fn cross_namespace_collision_does_not_overwrite_either_map() {
        let mut reg = UGenRegistry::new();
        reg.register("dup", bare_factory, &[], &out_spec());
        let factory: TableUGenFactory = Arc::new(|_table| Box::new(Const::new(2.0)));
        reg.register_table_bound("dup", factory, UGenCategory::Utility, &[], &out_spec());

        assert!(
            reg.entry("dup").is_some(),
            "the bare entry must survive a table-bound registration under the same name"
        );
        assert!(
            reg.table_bound_entry("dup").is_some(),
            "the table-bound entry must survive a bare registration under the same name"
        );
    }

    /// MOT-649 F12: pins what a cross-namespace-colliding name actually
    /// resolves to when compiled -- not registration order (bare was
    /// registered *first* here, table-bound *second*), but whether the
    /// specific IR node carries an `IrTableBinding`. Without one, the bare
    /// factory renders (1.0); with one, the table-bound factory renders
    /// (2.0) -- despite table-bound having been registered second in both
    /// cases, and despite `validate()` never raising `MissingTableBinding`
    /// for this kind (it isn't table-bound-*only*, since a bare entry also
    /// exists).
    #[cfg(feature = "ir")]
    #[test]
    fn colliding_name_dispatch_is_governed_by_table_binding_presence_not_registration_order() {
        use crate::coeff_table::{CoeffTable, CoeffTableBank};
        use crate::ir::{FORMAT_VERSION, IrNode, IrSynthDef, IrTableBinding, SynthDefClass};
        use crate::{Engine, EngineConfig};

        let mut reg = UGenRegistry::new();
        reg.register("dup", bare_factory, &[], &out_spec()); // registered FIRST
        let factory: TableUGenFactory = Arc::new(|_table| Box::new(Const::new(2.0)));
        reg.register_table_bound("dup", factory, UGenCategory::Utility, &[], &out_spec()); // registered SECOND

        let mut bank = CoeffTableBank::new();
        let table = CoeffTable {
            name: "unused".into(),
            entries: alloc::vec![],
        };
        let id = bank.register(table);

        let base_def = |table_bindings| IrSynthDef {
            format_version: FORMAT_VERSION,
            name: "colliding".into(),
            class: SynthDefClass::Source,
            output_channels: 1,
            nodes: alloc::vec![IrNode::UGen {
                kind: "dup".into(),
                consts: alloc::vec![],
            }],
            edges: alloc::vec![],
            params: alloc::vec![],
            audio_inputs: alloc::vec![],
            table_bindings,
            output_node: 0,
        };

        let render_first_sample = |def: &crate::synthdef::SynthDef| -> f32 {
            let mut engine = Engine::new(EngineConfig {
                sample_rate: 44100.0,
                block_size: 128,
            });
            let synth = engine.instantiate_synthdef(def);
            engine.graph_mut().set_sink(synth.output_node());
            engine.prepare();
            let output = engine.render().expect("engine should produce output");
            output.channel(0).samples()[0]
        };

        // No table_bindings entry for node 0: validate() does not demand
        // one (this kind also has a bare entry, so it is not
        // table-bound-only), and compile() uses the bare factory.
        let unbound = base_def(alloc::vec![]);
        unbound
            .validate(&reg)
            .expect("a colliding name with a bare entry needs no table binding");
        let unbound_def = unbound
            .compile(&reg)
            .expect("compiles via the bare factory");
        assert_eq!(
            render_first_sample(&unbound_def),
            1.0,
            "no table binding on the node -> the bare factory (1.0) should render, \
             even though table-bound was registered second"
        );

        // node 0 now carries an explicit table binding: build_base's
        // per-node dispatch uses the table-bound factory instead, still
        // regardless of registration order.
        let bound = base_def(alloc::vec![IrTableBinding {
            node: 0,
            table_id: id.0,
        }]);
        bound.validate(&reg).expect("valid");
        let bound_def = bound
            .compile_with_tables(&reg, &bank)
            .expect("resolves via the table-bound factory");
        assert_eq!(
            render_first_sample(&bound_def),
            2.0,
            "a table binding on the node -> the table-bound factory (2.0) should \
             render, matching the module doc's stated per-node dispatch rule"
        );
    }

    /// MOT-652: a colliding name whose bare and table-bound entries have
    /// *differing* arities is the case where `validate()` and construction
    /// could disagree before this fix (arity checks always preferred the
    /// bare entry, while construction dispatched per node). Same node, same
    /// edge, only the presence of a table binding changes: unbound, the
    /// edge targets an input port the 0-arity bare entry doesn't have, so
    /// `validate()` must reject it; bound, the same port is valid against
    /// the 1-arity table-bound entry, so `validate()` must accept it *and*
    /// `compile_with_tables` must actually build successfully at that
    /// arity -- proving the two layers now agree per node instead of just
    /// per name.
    #[cfg(feature = "ir")]
    #[test]
    fn colliding_name_with_differing_arities_agrees_between_validate_and_build() {
        use crate::coeff_table::{CoeffTable, CoeffTableBank};
        use crate::ir::{
            FORMAT_VERSION, IrEdge, IrError, IrNode, IrSynthDef, IrTableBinding, SynthDefClass,
        };

        let mut reg = UGenRegistry::new();
        // Bare "dup": 0 inputs.
        reg.register("dup", bare_factory, &[], &out_spec());
        // Table-bound "dup": 1 input -- a genuine arity collision.
        let factory: TableUGenFactory = Arc::new(|_table| Box::new(Const::new(2.0)));
        reg.register_table_bound(
            "dup",
            factory,
            UGenCategory::Utility,
            &in_spec(),
            &out_spec(),
        );

        let mut bank = CoeffTableBank::new();
        let table = CoeffTable {
            name: "unused".into(),
            entries: alloc::vec![],
        };
        let id = bank.register(table);

        let base_def = |table_bindings| IrSynthDef {
            format_version: FORMAT_VERSION,
            name: "colliding_arity".into(),
            class: SynthDefClass::Source,
            output_channels: 1,
            nodes: alloc::vec![
                IrNode::Const(0.0),
                IrNode::UGen {
                    kind: "dup".into(),
                    consts: alloc::vec![],
                },
            ],
            edges: alloc::vec![IrEdge {
                from: 0,
                to: 1,
                to_input: 0,
            }],
            params: alloc::vec![],
            audio_inputs: alloc::vec![],
            table_bindings,
            output_node: 1,
        };

        // Unbound: dispatch uses the bare (0-input) entry, so input port 0
        // is out of range -- validate() must reject it, matching that
        // add_ugen_node/build_base could never wire this edge either.
        let unbound = base_def(alloc::vec![]);
        assert_eq!(
            unbound.validate(&reg),
            Err(IrError::InputOutOfRange {
                node: 1,
                input: 0,
                arity: 0,
            }),
            "unbound node dispatches to the 0-arity bare entry, so port 0 is out of range"
        );

        // Bound: dispatch uses the table-bound (1-input) entry, so the same
        // port 0 is in range -- validate() must accept it, and construction
        // must actually succeed at that arity.
        let bound = base_def(alloc::vec![IrTableBinding {
            node: 1,
            table_id: id.0,
        }]);
        bound.validate(&reg).expect(
            "bound node dispatches to the 1-arity table-bound entry, so port 0 is in range",
        );
        bound
            .compile_with_tables(&reg, &bank)
            .expect("construction succeeds at the arity validate() just checked");
    }
}
