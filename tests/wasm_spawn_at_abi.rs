//! `ms_spawn_voice_on_routing_bus_named_at` and `ms_voice_start_at`, driven
//! through the raw C-ABI the way render_score.mjs drives it (process-global
//! statics, one lock; see `tests/wasm_voice_budget_abi.rs`).

use microsynth::web;
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn with_str<T>(s: &str, f: impl FnOnce(*const u8, usize) -> T) -> T {
    let len = s.len();
    let ptr = web::ms_alloc(len);
    unsafe { core::ptr::copy_nonoverlapping(s.as_ptr(), ptr, len) };
    let out = f(ptr, len);
    unsafe { web::ms_free(ptr, len) };
    out
}

fn session() {
    unsafe {
        web::ms_routing_init(44100.0);
        assert_eq!(web::ms_routing_build(), 0);
        with_str("dc", |np, nl| {
            with_str("synthdef dc amp=0.25 = amp", |sp, sl| {
                assert_eq!(web::ms_register_def(np, nl, sp, sl), 0)
            })
        });
    }
}

fn spawn_at(start: Option<u64>) -> u64 {
    with_str("dc", |dp, dl| {
        with_str("main", |bp, bl| unsafe {
            match start {
                Some(s) => web::ms_spawn_voice_on_routing_bus_named_at(dp, dl, bp, bl, s),
                None => web::ms_spawn_voice_on_routing_bus_named(dp, dl, bp, bl),
            }
        })
    })
}

fn block() -> Vec<f32> {
    let mut left = [0f32; 128];
    let mut right = [0f32; 128];
    unsafe { web::ms_render(left.as_mut_ptr(), right.as_mut_ptr()) };
    left.to_vec()
}

#[test]
fn spawn_at_starts_the_voice_on_its_sample() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    session();
    let id = spawn_at(Some(200));
    assert!(id > 0);
    let first = block();
    assert!(first.iter().all(|&x| x == 0.0), "silent until sample 200");
    let second = block();
    assert!(second[..72].iter().all(|&x| x == 0.0));
    assert!(second[72..].iter().all(|&x| x == 0.25));
    assert!(block().iter().all(|&x| x == 0.25));
}

#[test]
fn a_start_not_after_the_block_is_the_plain_spawn() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    session();
    assert!(spawn_at(Some(0)) > 0);
    assert!(block().iter().all(|&x| x == 0.25));
    assert_eq!(web::ms_voice_start_at(1, 500), 0, "already rendered");
    assert_eq!(web::ms_voice_start_at(999, 500), 0, "no such voice");
}

#[test]
fn start_at_defers_a_voice_spawned_by_any_export() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    session();
    let id = spawn_at(None);
    assert!(id > 0);
    assert_eq!(web::ms_voice_start_at(id, 64), 1);
    let out = block();
    assert!(out[..64].iter().all(|&x| x == 0.0));
    assert!(out[64..].iter().all(|&x| x == 0.25));
}
