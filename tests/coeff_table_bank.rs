//! Round-trip tests for the runtime coefficient-table bank (MOT-634): a
//! synthetic table registered in a [`CoeffTableBank`], resolved by id through
//! a table-bound `UGenRegistry` entry and [`IrSynthDef::compile_with_tables`],
//! and read back via a minimal probe UGen — the IR-level half of the
//! id-reference mechanism. `tests/wasm_coeff_table_abi.rs` covers the
//! wasm-ABI half (upload/replace/free via the raw `ms_coeff_table_*` C
//! exports).
//!
//! The probe UGen defined here (`CoeffSumProbe`) is test-only scaffolding,
//! not a shipped ugen: no consuming ugen exists yet (that is a later
//! ticket's job — see `src/coeff_table.rs`'s module doc and
//! `docs/coeff-table-bank-format.md`). Its only job is to prove the
//! plumbing: construct it from a resolved `Arc<CoeffTable>`, and it reports
//! back a value computed from that table's coefficients, deterministically,
//! so the test can assert the exact expected number.

#![cfg(feature = "ir")]

use microsynth::coeff_table::{CoeffTable, CoeffTableBank, PitchEntry, TableId};
use microsynth::dsl::compiler::{TableUGenFactory, UGenRegistry};
use microsynth::ir::{IrError, IrNode, IrSynthDef, IrTableBinding, SynthDefClass};
use microsynth::{
    AudioBuffer, InputSpec, OutputSpec, ProcessContext, UGen, UGenCategory, UGenSpec,
};
use std::sync::Arc;

/// A table-bound test UGen with no inputs, one control-rate output: the sum
/// of channel 0's coefficients for the table's first pitch entry. Exists
/// only so this test can observe, through the ordinary
/// `SynthDef::instantiate` + `UGen::process` path, that a node built via
/// `compile_with_tables` really is bound to the table the test registered
/// (as opposed to merely not erroring).
struct CoeffSumProbe {
    table: Arc<CoeffTable>,
}

impl UGen for CoeffSumProbe {
    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "CoeffSumProbe",
            category: UGenCategory::Utility,
            inputs: &[],
            outputs: &[OutputSpec {
                name: "out",
                rate: microsynth::Rate::Control,
            }],
        }
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        _inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let sum: f32 = self
            .table
            .entries
            .first()
            .and_then(|e| e.channel(0))
            .map(|c| c.iter().sum())
            .unwrap_or(0.0);
        output.channel_mut(0).samples_mut()[0] = sum;
    }
}

fn synthetic_table(name: &str, f0_hz: f32) -> CoeffTable {
    CoeffTable {
        name: name.into(),
        entries: vec![PitchEntry {
            f0_hz,
            inharmonicity_stretch: 1.0,
            partial_freqs: vec![f0_hz, f0_hz * 2.0, f0_hz * 3.0],
            k_channels: 2,
            j_noise: 1,
            // width = 3 partials + 1 noise = 4; channel 0 sums to 0.9.
            coefficients: vec![0.5, 0.2, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
            metadata: vec![],
        }],
    }
}

fn registry_with_probe() -> UGenRegistry {
    let mut reg = UGenRegistry::new();
    microsynth::register_builtins(&mut reg);
    let factory: TableUGenFactory = Arc::new(|table| Box::new(CoeffSumProbe { table }));
    reg.register_table_bound(
        "coeffSumProbe",
        factory,
        UGenCategory::Utility,
        &[] as &[InputSpec],
        &[OutputSpec {
            name: "out",
            rate: microsynth::Rate::Control,
        }],
    );
    reg
}

/// A single-node IR document: one table-bound `coeffSumProbe` UGen, bound to
/// `table_id`, as its own output.
fn probe_ir(table_id: u32) -> IrSynthDef {
    IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "probe".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![IrNode::UGen {
            kind: "coeffSumProbe".into(),
            consts: vec![],
        }],
        edges: vec![],
        params: vec![],
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding { node: 0, table_id }],
        output_node: 0,
    }
}

/// Render the probe's single control-rate output sample.
fn render_probe_output(def: &microsynth::SynthDef) -> f32 {
    let mut engine = microsynth::Engine::new(microsynth::EngineConfig {
        sample_rate: 44100.0,
        block_size: 64,
    });
    let synth = engine.instantiate_synthdef(def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let output = engine.render().expect("engine should produce output");
    output.channel(0).samples()[0]
}

#[test]
fn register_resolve_and_read_back_coefficients() {
    let reg = registry_with_probe();
    let mut bank = CoeffTableBank::new();
    let id = bank.register(synthetic_table("inst", 220.0));

    let ir = probe_ir(id.0);
    ir.validate(&reg).expect("valid IR");
    let def = ir
        .compile_with_tables(&reg, &bank)
        .expect("resolves against bank");

    let value = render_probe_output(&def);
    assert!(
        (value - 0.9).abs() < 1e-5,
        "expected channel-0 coefficient sum 0.9, got {value}"
    );
}

#[test]
fn replace_is_visible_to_a_later_compile() {
    let reg = registry_with_probe();
    let mut bank = CoeffTableBank::new();
    let id = bank.register(synthetic_table("inst", 220.0));

    let ir = probe_ir(id.0);
    let before = render_probe_output(&ir.compile_with_tables(&reg, &bank).unwrap());
    assert!((before - 0.9).abs() < 1e-5);

    let mut replacement = synthetic_table("inst", 220.0);
    replacement.entries[0].coefficients = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    bank.replace(id, replacement).unwrap();

    let after = render_probe_output(&ir.compile_with_tables(&reg, &bank).unwrap());
    assert!(
        (after - 4.0).abs() < 1e-5,
        "expected updated channel-0 sum 4.0 after replace, got {after}"
    );
    assert_ne!(
        before, after,
        "replace should change what a fresh compile resolves"
    );
}

#[test]
fn free_makes_a_later_compile_fail_with_table_not_found() {
    let reg = registry_with_probe();
    let mut bank = CoeffTableBank::new();
    let id = bank.register(synthetic_table("inst", 220.0));

    // Still resolves before free.
    let ir = probe_ir(id.0);
    assert!(ir.compile_with_tables(&reg, &bank).is_ok());

    bank.free(id);
    match ir.compile_with_tables(&reg, &bank) {
        Err(e) => assert_eq!(e, IrError::TableNotFound(id.0)),
        Ok(_) => panic!("expected TableNotFound after free"),
    }
}

#[test]
fn missing_binding_for_a_table_bound_only_kind_fails_validation() {
    let reg = registry_with_probe();
    // No table_bindings entry for the coeffSumProbe node.
    let ir = IrSynthDef {
        table_bindings: vec![],
        ..probe_ir(0)
    };
    assert_eq!(
        ir.validate(&reg),
        Err(IrError::MissingTableBinding {
            node: 0,
            kind: "coeffSumProbe".into(),
        })
    );
}

#[test]
fn plain_compile_rejects_a_table_bound_only_kind() {
    let reg = registry_with_probe();
    let mut bank = CoeffTableBank::new();
    let id = bank.register(synthetic_table("inst", 220.0));
    let ir = probe_ir(id.0);
    // The bare `compile` path never resolves table bindings; a table-bound-
    // only kind has no bare factory, so it fails loudly rather than
    // silently building an unfilled node.
    match ir.compile(&reg) {
        Err(e) => assert_eq!(e, IrError::UnknownKind("coeffSumProbe".into())),
        Ok(_) => panic!("expected UnknownKind for a table-bound-only kind"),
    }
}

#[test]
fn id_for_name_resolves_a_registered_table() {
    let mut bank = CoeffTableBank::new();
    let id = bank.register(synthetic_table("named-instrument", 220.0));
    assert_eq!(bank.id_for_name("named-instrument"), Some(id));
    assert_eq!(TableId(id.0), id);
}
