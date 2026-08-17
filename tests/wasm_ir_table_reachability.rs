//! Wasm-ABI-level round trip for MOT-640: upload a coefficient table via the
//! raw `ms_coeff_table_*` exports, bind it to a `partialsNoise` node via
//! `ms_compile_ir_with_tables` (a serialized `ir::IrSynthDef`, not DSL text),
//! and render non-silence -- entirely from wasm entry points, proving the
//! gap `docs/coeff-table-bank-format.md`'s "What this mechanism does not
//! (yet) provide" used to disclose (no wasm-bundle graph-compile
//! reachability for table-bound nodes) is closed.
//!
//! Mirrors `tests/partials_noise.rs`'s native-API
//! `table_bound_registration_resolves_and_renders`, but drives the raw
//! C-ABI exports the way a wasm host actually would, using the same
//! shared-lock convention as `tests/wasm_raw_abi.rs`'s module doc (the
//! exports are backed by process-global statics).

#![cfg(feature = "ir")]

use microsynth::coeff_table::{CoeffTable, PitchEntry};
use microsynth::ir::{IrEdge, IrNode, IrSynthDef, IrTableBinding, SynthDefClass};
use microsynth::web;
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Write `bytes` into a fresh `ms_alloc`'d buffer, the way a JS caller would
/// before passing it to any `ms_*` export that takes `(ptr, len)`.
fn alloc_bytes(bytes: &[u8]) -> (*mut u8, usize) {
    let len = bytes.len();
    let ptr = web::ms_alloc(len);
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len);
    }
    (ptr, len)
}

fn free_bytes(ptr: *mut u8, len: usize) {
    unsafe { web::ms_free(ptr, len) };
}

fn synthetic_table() -> CoeffTable {
    CoeffTable {
        name: "wasm_reachability".into(),
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

/// A one-node `partialsNoise` graph, freq/gain fed by inline-const `Param`
/// nodes, bound to `table_id` -- same shape as
/// `tests/partials_noise.rs`'s `table_bound_registration_resolves_and_renders`,
/// serialized to the IR's own binary wire format.
fn partials_noise_ir_bytes(table_id: u32) -> Vec<u8> {
    IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "wasm_partials_probe".into(),
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
    }
    .to_bytes()
}

#[test]
fn upload_bind_and_render_partials_noise_through_wasm_abi() {
    let _guard = lock();
    web::ms_init(44100.0);

    let table_bytes = synthetic_table().to_bytes();
    let (tptr, tlen) = alloc_bytes(&table_bytes);
    let table_id = unsafe { web::ms_coeff_table_register(tptr, tlen) };
    free_bytes(tptr, tlen);
    assert!(table_id > 0, "table upload should succeed");

    let ir_bytes = partials_noise_ir_bytes(table_id);
    let (iptr, ilen) = alloc_bytes(&ir_bytes);
    let status = unsafe { web::ms_compile_ir_with_tables(iptr, ilen) };
    free_bytes(iptr, ilen);
    assert_eq!(status, 0, "compiling the table-bound IR should succeed");

    let mut left = [0.0f32; 128];
    let mut right = [0.0f32; 128];
    unsafe { web::ms_render(left.as_mut_ptr(), right.as_mut_ptr()) };

    assert!(
        left.iter().any(|&s| s != 0.0),
        "table-bound partialsNoise compiled via the wasm ABI should render non-silence"
    );
}

/// Same round trip, but through `ms_routing_init`'s `UGenRegistry`
/// construction site instead of `ms_init`'s -- a second of the five
/// `register_table_bound_builtins` call sites in `src/web.rs` (F13:
/// previously only `ms_init`'s site was exercised). `ms_compile_ir_with_tables`
/// replaces the engine and its sink outright, so it renders the compiled
/// table-bound def directly regardless of the routing session
/// `ms_routing_init` set up -- this test is only proving that entry point's
/// `REGISTRY` has `partialsNoise` resolvable by name, the same property the
/// `ms_init` version proves for its own site.
#[test]
fn upload_bind_and_render_partials_noise_through_routing_init() {
    let _guard = lock();
    web::ms_routing_init(44100.0);

    let table_bytes = synthetic_table().to_bytes();
    let (tptr, tlen) = alloc_bytes(&table_bytes);
    let table_id = unsafe { web::ms_coeff_table_register(tptr, tlen) };
    free_bytes(tptr, tlen);
    assert!(table_id > 0, "table upload should succeed");

    let ir_bytes = partials_noise_ir_bytes(table_id);
    let (iptr, ilen) = alloc_bytes(&ir_bytes);
    let status = unsafe { web::ms_compile_ir_with_tables(iptr, ilen) };
    free_bytes(iptr, ilen);
    assert_eq!(
        status, 0,
        "compiling the table-bound IR should succeed through the ms_routing_init registry site"
    );

    let mut left = [0.0f32; 128];
    let mut right = [0.0f32; 128];
    unsafe { web::ms_render(left.as_mut_ptr(), right.as_mut_ptr()) };

    assert!(
        left.iter().any(|&s| s != 0.0),
        "table-bound partialsNoise compiled via ms_routing_init's registry should render non-silence"
    );
}

/// An IR document whose `table_bindings` names a table id that was never
/// registered fails `ms_compile_ir_with_tables` cleanly (1), rather than
/// panicking or silently building an unfilled node.
#[test]
fn unregistered_table_id_fails_cleanly() {
    let _guard = lock();
    web::ms_init(44100.0);

    let ir_bytes = partials_noise_ir_bytes(999);
    let (iptr, ilen) = alloc_bytes(&ir_bytes);
    let status = unsafe { web::ms_compile_ir_with_tables(iptr, ilen) };
    free_bytes(iptr, ilen);
    assert_eq!(status, 1, "an unresolved table id should fail, not panic");
}

/// Malformed IR bytes fail cleanly too.
#[test]
fn malformed_ir_bytes_fail_cleanly() {
    let _guard = lock();
    web::ms_init(44100.0);

    let bad = [0u8; 4];
    let (iptr, ilen) = alloc_bytes(&bad);
    let status = unsafe { web::ms_compile_ir_with_tables(iptr, ilen) };
    free_bytes(iptr, ilen);
    assert_eq!(status, 1, "malformed IR bytes should fail, not panic");
}
