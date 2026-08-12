//! Integration tests for the raw C-ABI coefficient-table bank exports
//! (`ms_coeff_table_*`, `src/web.rs`) — register/replace/free/name-lookup
//! driven exactly the way a JS caller would (writing bytes into an
//! `ms_alloc`'d buffer, passing pointer + length). See
//! `tests/wasm_raw_abi.rs`'s module doc for why these tests take a shared
//! lock: the exports are backed by process-global statics, safe in
//! production (one WASM instance is inherently single-threaded) but not
//! under `cargo test`'s concurrent `#[test]` execution within one binary.
//!
//! Complements `tests/coeff_table_bank.rs`, which covers the IR-level half
//! of the mechanism (a table-bound node resolving a table id via
//! `IrSynthDef::compile_with_tables`); this file covers only the upload/
//! replace/free/lookup ABI surface itself.

use microsynth::coeff_table::{CoeffTable, PitchEntry};
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

fn synthetic_table(name: &str) -> CoeffTable {
    CoeffTable {
        name: name.into(),
        entries: vec![PitchEntry {
            f0_hz: 440.0,
            inharmonicity_stretch: 1.0,
            partial_freqs: vec![440.0, 880.0],
            k_channels: 1,
            j_noise: 1,
            coefficients: vec![0.6, 0.3, 0.1],
            metadata: vec![],
        }],
    }
}

#[test]
fn register_read_back_replace_and_free_round_trip() {
    let _guard = lock();
    web::ms_init(44100.0);

    // Register.
    let bytes = synthetic_table("kalimba").to_bytes();
    let (ptr, len) = alloc_bytes(&bytes);
    let id = unsafe { web::ms_coeff_table_register(ptr, len) };
    free_bytes(ptr, len);
    assert!(id > 0, "register should return a nonzero id");

    // Name resolves to the same id.
    let (name_ptr, name_len) = alloc_bytes(b"kalimba");
    let resolved = unsafe { web::ms_coeff_table_id_for_name(name_ptr, name_len) };
    free_bytes(name_ptr, name_len);
    assert_eq!(resolved, id);

    // Replace at the same id.
    let mut replacement = synthetic_table("kalimba");
    replacement.entries[0].coefficients = vec![0.1, 0.1, 0.8];
    let replacement_bytes = replacement.to_bytes();
    let (rptr, rlen) = alloc_bytes(&replacement_bytes);
    let replace_status = unsafe { web::ms_coeff_table_replace(id, rptr, rlen) };
    free_bytes(rptr, rlen);
    assert_eq!(replace_status, 0, "replace on a live id should succeed");

    // Free.
    web::ms_coeff_table_free(id);

    // A second replace against the now-freed id fails.
    let (rptr2, rlen2) = alloc_bytes(&replacement_bytes);
    let replace_after_free = unsafe { web::ms_coeff_table_replace(id, rptr2, rlen2) };
    free_bytes(rptr2, rlen2);
    assert_eq!(
        replace_after_free, 1,
        "replace against a freed id should fail"
    );

    // Name no longer resolves after free.
    let (name_ptr2, name_len2) = alloc_bytes(b"kalimba");
    let resolved_after_free = unsafe { web::ms_coeff_table_id_for_name(name_ptr2, name_len2) };
    free_bytes(name_ptr2, name_len2);
    assert_eq!(
        resolved_after_free, 0,
        "freed table's name should not resolve"
    );
}

#[test]
fn replace_on_an_id_that_was_never_registered_fails() {
    let _guard = lock();
    web::ms_init(44100.0);

    let bytes = synthetic_table("nope").to_bytes();
    let (ptr, len) = alloc_bytes(&bytes);
    let status = unsafe { web::ms_coeff_table_replace(9999, ptr, len) };
    free_bytes(ptr, len);
    assert_eq!(status, 1);
}

#[test]
fn register_with_malformed_bytes_returns_zero() {
    let _guard = lock();
    web::ms_init(44100.0);

    let bad = [0u8; 4]; // too short to even hold the magic + version.
    let (ptr, len) = alloc_bytes(&bad);
    let id = unsafe { web::ms_coeff_table_register(ptr, len) };
    free_bytes(ptr, len);
    assert_eq!(id, 0);
}

#[test]
fn id_for_name_with_no_match_returns_zero() {
    let _guard = lock();
    web::ms_init(44100.0);

    let (name_ptr, name_len) = alloc_bytes(b"never-registered");
    let resolved = unsafe { web::ms_coeff_table_id_for_name(name_ptr, name_len) };
    free_bytes(name_ptr, name_len);
    assert_eq!(resolved, 0);
}

#[test]
fn free_of_an_unregistered_id_is_a_harmless_no_op() {
    let _guard = lock();
    web::ms_init(44100.0);
    // Must not panic.
    web::ms_coeff_table_free(424242);
}

#[test]
fn bank_is_fresh_after_ms_init_with_bus_and_ms_routing_init_too() {
    let _guard = lock();

    // Register under the plain ms_init session, then re-init via each of
    // the other two session-start exports and confirm the name no longer
    // resolves -- the bank reset is a per-export requirement, not just
    // ms_init's.
    web::ms_init(44100.0);
    let bytes = synthetic_table("stale").to_bytes();
    let (ptr, len) = alloc_bytes(&bytes);
    let id = unsafe { web::ms_coeff_table_register(ptr, len) };
    free_bytes(ptr, len);
    assert!(id > 0);

    web::ms_init_with_bus(44100.0);
    let (name_ptr, name_len) = alloc_bytes(b"stale");
    let resolved = unsafe { web::ms_coeff_table_id_for_name(name_ptr, name_len) };
    free_bytes(name_ptr, name_len);
    assert_eq!(
        resolved, 0,
        "ms_init_with_bus should reset the coeff-table bank"
    );

    let bytes2 = synthetic_table("stale2").to_bytes();
    let (ptr2, len2) = alloc_bytes(&bytes2);
    let id2 = unsafe { web::ms_coeff_table_register(ptr2, len2) };
    free_bytes(ptr2, len2);
    assert!(id2 > 0);

    web::ms_routing_init(44100.0);
    let (name_ptr2, name_len2) = alloc_bytes(b"stale2");
    let resolved2 = unsafe { web::ms_coeff_table_id_for_name(name_ptr2, name_len2) };
    free_bytes(name_ptr2, name_len2);
    assert_eq!(
        resolved2, 0,
        "ms_routing_init should reset the coeff-table bank"
    );
}
