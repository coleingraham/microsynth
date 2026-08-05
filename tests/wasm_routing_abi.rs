//! Integration test for the raw C-ABI multi-bus routing exports
//! (`ms_routing_*`/`ms_register_effect_def`/
//! `ms_spawn_voice_on_routing_bus_named` in `src/web.rs`) — the wasm-ABI
//! entry point onto the same `RoutingGraph`/`Engine::build_routing`
//! pipeline the offline `ir::IrRoutingContainer` path drives (see
//! `tests/ir_routing.rs` and the "Multi-bus routing exports" section doc
//! comment in `src/web.rs` for why this is a parallel, not a duplicate,
//! path).
//!
//! Follows `tests/wasm_raw_abi.rs`'s pattern: calls the `#[no_mangle]
//! extern "C"` exports directly, writing strings into `ms_alloc`'d buffers
//! the way a JS caller would.

use microsynth::web;
use std::sync::Mutex;

// The raw exports are backed by process-global statics (`WasmCell`); cargo
// runs `#[test]` functions in one binary concurrently by default, so each
// test below takes this lock for its whole body — the same pattern
// `tests/wasm_raw_abi.rs` uses, for the same reason (see that file's module
// doc comment).
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Write `s` into a fresh `ms_alloc`'d buffer.
fn alloc_str(s: &str) -> (*mut u8, usize) {
    let len = s.len();
    let ptr = web::ms_alloc(len);
    unsafe {
        core::ptr::copy_nonoverlapping(s.as_ptr(), ptr, len);
    }
    (ptr, len)
}

fn free_str(ptr: *mut u8, len: usize) {
    unsafe { web::ms_free(ptr, len) };
}

#[test]
fn routing_topology_reaches_render_through_raw_abi() {
    let _guard = lock();
    unsafe {
        web::ms_routing_init(44100.0);

        let (bus_ptr, bus_len) = alloc_str("group");
        assert_eq!(web::ms_routing_add_bus(bus_ptr, bus_len, 2), 0);

        let (name_ptr, name_len) = alloc_str("groupFx");
        let (src_ptr, src_len) = alloc_str("synthdef groupFx = audioIn * 0.5");
        assert_eq!(
            web::ms_register_effect_def(name_ptr, name_len, src_ptr, src_len),
            0
        );
        free_str(src_ptr, src_len);

        let (grp_ptr, grp_len) = alloc_str("group");
        let (fx_ptr, fx_len) = alloc_str("groupFx");
        let (main_ptr, main_len) = alloc_str("main");
        assert_eq!(
            web::ms_routing_add_effect(grp_ptr, grp_len, fx_ptr, fx_len, main_ptr, main_len),
            0
        );
        free_str(grp_ptr, grp_len);
        free_str(fx_ptr, fx_len);
        free_str(main_ptr, main_len);

        assert_eq!(web::ms_routing_build(), 0);

        let (vname_ptr, vname_len) = alloc_str("instrument");
        let (vsrc_ptr, vsrc_len) = alloc_str("synthdef instrument = 1.0");
        assert_eq!(
            web::ms_register_def(vname_ptr, vname_len, vsrc_ptr, vsrc_len),
            0
        );
        free_str(vsrc_ptr, vsrc_len);

        let (vname2_ptr, vname2_len) = alloc_str("instrument");
        let (bus2_ptr, bus2_len) = alloc_str("group");
        let voice_id =
            web::ms_spawn_voice_on_routing_bus_named(vname2_ptr, vname2_len, bus2_ptr, bus2_len);
        assert!(voice_id > 0, "voice should spawn onto the named bus");
        free_str(vname2_ptr, vname2_len);
        free_str(bus2_ptr, bus2_len);
        free_str(bus_ptr, bus_len);
        free_str(name_ptr, name_len);
        free_str(vname_ptr, vname_len);

        let mut left = [0f32; 128];
        let mut right = [0f32; 128];
        web::ms_render(left.as_mut_ptr(), right.as_mut_ptr());

        // instrument outputs 1.0 -> group -> groupFx halves it -> main = 0.5
        for &s in &left {
            assert!((s - 0.5).abs() < 1e-6, "expected 0.5, got {s}");
        }
    }
}

#[test]
fn routing_add_bus_rejects_main() {
    let _guard = lock();
    unsafe {
        web::ms_routing_init(44100.0);
        let (ptr, len) = alloc_str("main");
        assert_eq!(
            web::ms_routing_add_bus(ptr, len, 2),
            1,
            "declaring \"main\" explicitly must be rejected"
        );
        free_str(ptr, len);
    }
}

#[test]
fn routing_add_effect_rejects_unknown_bus() {
    let _guard = lock();
    unsafe {
        web::ms_routing_init(44100.0);

        let (name_ptr, name_len) = alloc_str("fx");
        let (src_ptr, src_len) = alloc_str("synthdef fx = audioIn * 0.5");
        assert_eq!(
            web::ms_register_effect_def(name_ptr, name_len, src_ptr, src_len),
            0
        );
        free_str(src_ptr, src_len);

        let (bad_ptr, bad_len) = alloc_str("nonexistent");
        let (fx_ptr, fx_len) = alloc_str("fx");
        let (main_ptr, main_len) = alloc_str("main");
        assert_eq!(
            web::ms_routing_add_effect(bad_ptr, bad_len, fx_ptr, fx_len, main_ptr, main_len),
            1,
            "routing from an undeclared bus must fail"
        );
        free_str(bad_ptr, bad_len);
        free_str(fx_ptr, fx_len);
        free_str(main_ptr, main_len);
        free_str(name_ptr, name_len);
    }
}
