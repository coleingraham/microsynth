//! Wasm-ABI-level coverage for `ms_register_def_ir`: register a named
//! SynthDef from its IR JSON text form (not the binary wire format
//! `ms_compile_ir_with_tables` takes, and not DSL source text like
//! `ms_register_def`), then spawn and render it by name through the same
//! `DEF_REGISTRY`/`ms_spawn_voice_named` path a DSL-registered def uses.
//!
//! Mirrors `tests/wasm_ir_table_reachability.rs`'s raw-C-ABI conventions
//! (shared-lock, `ms_alloc`-backed buffers) but exercises registration by
//! name rather than `ms_compile_ir_with_tables`'s whole-engine replacement.

#![cfg(feature = "ir")]

use microsynth::coeff_table::{CoeffTable, PitchEntry};
use microsynth::dsl::ast::SynthDefDecl;
use microsynth::dsl::compiler::UGenRegistry;
use microsynth::dsl::{lexer::tokenize, parser::Parser};
use microsynth::ir::{IrEdge, IrNode, IrSynthDef, IrTableBinding, SynthDefClass, from_decl};
use microsynth::ugens::{register_builtins, register_table_bound_builtins};
use microsynth::web;
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Write `s` into a fresh `ms_alloc`'d buffer, the way a JS caller would
/// before passing it to any `ms_*` export that takes `(ptr, len)`.
fn alloc_str(s: &str) -> (*mut u8, usize) {
    let len = s.len();
    let ptr = web::ms_alloc(len);
    unsafe {
        core::ptr::copy_nonoverlapping(s.as_ptr(), ptr, len);
    }
    (ptr, len)
}

fn free_bytes(ptr: *mut u8, len: usize) {
    unsafe { web::ms_free(ptr, len) };
}

fn parse_one(src: &str) -> SynthDefDecl {
    let tokens = tokenize(src).unwrap_or_else(|e| panic!("lex {src:?}: {e}"));
    let mut parser = Parser::new(tokens);
    let mut program = parser
        .parse_program()
        .unwrap_or_else(|e| panic!("parse {src:?}: {e}"));
    assert_eq!(program.defs.len(), 1, "expected exactly one def in {src:?}");
    program.defs.pop().unwrap()
}

/// The same `UGenRegistry` construction `ms_init`/`ms_init_with_bus` use,
/// for decompiling DSL source into IR outside the wasm ABI.
fn builtin_registry() -> UGenRegistry {
    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);
    register_table_bound_builtins(&mut reg);
    reg
}

/// A pitched def with the conventional `freq`/`amp`/`gate` control signature
/// a host render path drives, exercising the same node kinds (`sinOsc`,
/// `Add`, arithmetic) a synthesized-and-fit SynthDef's IR JSON commonly
/// carries, without depending on any file on disk.
const PITCHED_SRC: &str = "synthdef probe freq=440.0 amp=0.5 gate=1.0 = adsr gate 0.01 0.1 0.7 0.3 * sinOsc freq 0.0 * amp";

fn render_voice(name: &str) -> (Vec<f32>, Vec<f32>) {
    let (nptr, nlen) = alloc_str(name);
    let voice_id = unsafe { web::ms_spawn_voice_named(nptr, nlen) };
    free_bytes(nptr, nlen);
    assert!(voice_id > 0, "spawning {name:?} should succeed");
    web::ms_voice_gate(voice_id, 1.0);

    let mut left = [0.0f32; 128];
    let mut right = [0.0f32; 128];
    unsafe { web::ms_render(left.as_mut_ptr(), right.as_mut_ptr()) };
    (left.to_vec(), right.to_vec())
}

fn assert_bitwise_eq(a: &[f32], b: &[f32], case: &str) {
    assert_eq!(a.len(), b.len(), "{case}: length differs");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{case}: sample {i} differs: {x} vs {y}"
        );
    }
}

/// Registering the IR-JSON form of a def renders non-silence and is
/// bit-for-bit identical to registering the same def's DSL source directly
/// -- the property the doc comment on `ms_register_def_ir` claims
/// (`compile_with_tables` over an empty `table_bindings` is a no-op
/// superset of `compile`, and `IrSynthDef::compile`'s own module doc says
/// `DSL -> SynthDef` and `DSL -> IR -> SynthDef` render byte-identically).
#[test]
fn register_via_json_matches_dsl_registration_bit_for_bit() {
    let _guard = lock();

    // Two separate engine sessions (re-`ms_init_with_bus` resets
    // `DEF_REGISTRY`/`ENGINE`/the bus) so each render reflects exactly one
    // voice's history -- rendering both voices in the same live session
    // would mix them on the shared bus and make the comparison meaningless.
    web::ms_init_with_bus(44100.0);
    let (dptr, dlen) = alloc_str(PITCHED_SRC);
    let (nptr, nlen) = alloc_str("probe");
    let status = unsafe { web::ms_register_def(nptr, nlen, dptr, dlen) };
    free_bytes(dptr, dlen);
    free_bytes(nptr, nlen);
    assert_eq!(status, 0, "DSL registration should succeed");
    let (left_dsl, right_dsl) = render_voice("probe");

    let reg = builtin_registry();
    let decl = parse_one(PITCHED_SRC);
    let ir = from_decl(&decl, &reg);
    let json = ir.to_json();

    web::ms_init_with_bus(44100.0);
    let (jptr, jlen) = alloc_str(&json);
    let (nptr, nlen) = alloc_str("probe");
    let status = unsafe { web::ms_register_def_ir(nptr, nlen, jptr, jlen) };
    free_bytes(jptr, jlen);
    free_bytes(nptr, nlen);
    assert_eq!(status, 0, "IR-JSON registration should succeed");
    let (left_ir, right_ir) = render_voice("probe");

    assert!(
        left_dsl.iter().any(|&s| s != 0.0),
        "DSL-registered voice should render non-silence"
    );
    assert_bitwise_eq(&left_dsl, &left_ir, "left channel");
    assert_bitwise_eq(&right_dsl, &right_ir, "right channel");
}

/// Malformed JSON fails cleanly (1), not a panic.
#[test]
fn malformed_json_fails_cleanly() {
    let _guard = lock();
    web::ms_init_with_bus(44100.0);

    let (jptr, jlen) = alloc_str("not json");
    let (nptr, nlen) = alloc_str("bad_probe");
    let status = unsafe { web::ms_register_def_ir(nptr, nlen, jptr, jlen) };
    free_bytes(jptr, jlen);
    free_bytes(nptr, nlen);
    assert_eq!(status, 1, "malformed JSON should fail, not panic");
}

/// A document naming an unregistered UGen kind fails `validate()` (status
/// 1), not a panic, and nothing is inserted into `DEF_REGISTRY` -- spawning
/// the name afterward still fails.
#[test]
fn unknown_ugen_kind_fails_validate_and_does_not_register() {
    let _guard = lock();
    web::ms_init_with_bus(44100.0);

    let bad = IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "unknown_kind_probe".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![IrNode::UGen {
            kind: "definitelyNotARegisteredUGenKind".into(),
            consts: vec![],
        }],
        edges: vec![],
        params: vec![],
        audio_inputs: vec![],
        table_bindings: vec![],
        output_node: 0,
    };
    let json = bad.to_json();

    let (jptr, jlen) = alloc_str(&json);
    let (nptr, nlen) = alloc_str("unknown_kind_probe");
    let status = unsafe { web::ms_register_def_ir(nptr, nlen, jptr, jlen) };
    free_bytes(jptr, jlen);
    free_bytes(nptr, nlen);
    assert_eq!(
        status, 1,
        "an unknown UGen kind should fail validate(), not panic"
    );

    let (nptr, nlen) = alloc_str("unknown_kind_probe");
    let voice_id = unsafe { web::ms_spawn_voice_named(nptr, nlen) };
    free_bytes(nptr, nlen);
    assert_eq!(
        voice_id, 0,
        "a def that failed to register should not be spawnable"
    );
}

fn synthetic_table() -> CoeffTable {
    CoeffTable {
        name: "ir_def_registration_probe".into(),
        entries: vec![PitchEntry {
            f0_hz: 220.0,
            inharmonicity_stretch: 1.0,
            partial_freqs: vec![220.0, 440.0, 660.0],
            k_channels: 1,
            j_noise: 0,
            coefficients: vec![0.5, 0.3, 0.1],
            metadata: vec![],
        }],
    }
}

/// `table_bindings` are honored through the JSON path too, not just the
/// binary path `ms_compile_ir_with_tables` takes: a `partialsNoise` def
/// registered via `ms_register_def_ir` and spawned by name renders
/// non-silence, proving `compile_with_tables` (not the bare `compile`) ran.
#[test]
fn table_bindings_honored_through_json_registration() {
    let _guard = lock();
    web::ms_init_with_bus(44100.0);

    // synthetic_table().to_bytes() is a binary payload, not UTF-8, so go
    // through ms_alloc/copy_nonoverlapping directly rather than the
    // str-based alloc_str helper.
    let table_bytes = synthetic_table().to_bytes();
    let tptr = web::ms_alloc(table_bytes.len());
    unsafe { core::ptr::copy_nonoverlapping(table_bytes.as_ptr(), tptr, table_bytes.len()) };
    let table_id = unsafe { web::ms_coeff_table_register(tptr, table_bytes.len()) };
    free_bytes(tptr, table_bytes.len());
    assert!(table_id > 0, "table upload should succeed");

    let def = IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "table_bound_probe".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![
            IrNode::Param {
                name: "freq".into(),
                default: 220.0,
            },
            IrNode::Param {
                name: "gain".into(),
                default: 1.0,
            },
            IrNode::UGen {
                kind: "partialsNoise".into(),
                consts: vec![],
            },
        ],
        edges: vec![
            IrEdge {
                from: 0,
                to: 2,
                to_input: 0,
            },
            IrEdge {
                from: 1,
                to: 2,
                to_input: 1,
            },
        ],
        params: vec![],
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding { node: 2, table_id }],
        output_node: 2,
    };
    let json = def.to_json();

    let (jptr, jlen) = alloc_str(&json);
    let (nptr, nlen) = alloc_str("table_bound_probe");
    let status = unsafe { web::ms_register_def_ir(nptr, nlen, jptr, jlen) };
    free_bytes(jptr, jlen);
    free_bytes(nptr, nlen);
    assert_eq!(status, 0, "table-bound IR-JSON registration should succeed");

    let (left, _right) = render_voice("table_bound_probe");
    assert!(
        left.iter().any(|&s| s != 0.0),
        "table-bound partialsNoise registered via JSON should render non-silence"
    );
}
