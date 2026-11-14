//! The voice-budget exports (`ms_set_voice_budget`, `ms_voice_steals`,
//! `ms_voice_cuts`, `ms_voice_steal_level`, `ms_bus_max_inputs`) and the
//! spawn exports going through the managed path, driven through the raw
//! C-ABI the way render_score.mjs drives it. Follows
//! `tests/wasm_routing_abi.rs`'s pattern (process-global statics, one lock).

use microsynth::web;
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn with_str<T>(s: &str, f: impl FnOnce(*const u8, usize) -> T) -> T {
    let len = s.len();
    let ptr = web::ms_alloc(len);
    unsafe { core::ptr::copy_nonoverlapping(s.as_ptr(), ptr, len) };
    let out = f(ptr, len);
    unsafe { web::ms_free(ptr, len) };
    out
}

/// A routing session with buses "a" and "b" feeding main, and a voice def
/// `name` whose output is the constant `level`.
fn session(defs: &[(&str, &str)]) {
    unsafe {
        web::ms_routing_init(44100.0);
        for bus in ["a", "b"] {
            assert_eq!(with_str(bus, |p, l| web::ms_routing_add_bus(p, l, 2)), 0);
        }
        with_str("thru", |np, nl| {
            with_str("synthdef thru = audioIn * 1.0", |sp, sl| {
                assert_eq!(web::ms_register_effect_def(np, nl, sp, sl), 0)
            })
        });
        for bus in ["a", "b"] {
            with_str(bus, |bp, bl| {
                with_str("thru", |fp, fl| {
                    with_str("main", |mp, ml| {
                        assert_eq!(web::ms_routing_add_effect(bp, bl, fp, fl, mp, ml), 0)
                    })
                })
            });
        }
        assert_eq!(web::ms_routing_build(), 0);
        for (name, src) in defs {
            with_str(name, |np, nl| {
                with_str(src, |sp, sl| {
                    assert_eq!(web::ms_register_def(np, nl, sp, sl), 0)
                })
            });
        }
    }
}

fn spawn(def: &str, bus: &str) -> u64 {
    with_str(def, |dp, dl| {
        with_str(bus, |bp, bl| unsafe {
            web::ms_spawn_voice_on_routing_bus_named(dp, dl, bp, bl)
        })
    })
}

fn block() -> Vec<f32> {
    let mut left = [0f32; 128];
    let mut right = [0f32; 128];
    web::ms_free_done();
    unsafe { web::ms_render(left.as_mut_ptr(), right.as_mut_ptr()) };
    left.to_vec()
}

#[test]
fn bus_capacity_is_exported() {
    assert_eq!(web::ms_bus_max_inputs(), 256);
}

#[test]
fn set_voice_budget_validates_policy_and_session() {
    let _g = lock();
    session(&[]);
    for code in 0..4 {
        assert_eq!(web::ms_set_voice_budget(8, code), 0);
    }
    assert_eq!(web::ms_set_voice_budget(8, 4), 2, "unknown policy");
    assert_eq!(web::ms_set_voice_budget(0, 99), 2);
    assert_eq!(web::ms_set_voice_budget(0, 3), 0, "detach");
    assert_eq!(web::ms_voice_steals(), 0);
    assert!(web::ms_voice_steal_level(0).is_nan());
}

#[test]
fn routing_spawn_steals_per_bus_and_fades() {
    let _g = lock();
    session(&[
        ("dc", "synthdef dc = 0.25"),
        ("quiet", "synthdef quiet = 0.01"),
    ]);
    assert_eq!(web::ms_set_voice_budget(2, 3), 0, "quietest, 2 per bus");
    assert!(spawn("dc", "a") > 0);
    let quiet = spawn("quiet", "a");
    assert!(quiet > 0);
    assert!(spawn("dc", "b") > 0, "bus b has its own budget");
    assert!(spawn("dc", "b") > 0);
    let before = block();
    assert!((before[0] - (0.25 + 0.01 + 0.5)).abs() < 1e-6);
    assert_eq!(web::ms_voice_steals(), 0);

    // bus a is full: the third voice steals the quiet one, which fades
    assert!(spawn("dc", "a") > 0);
    assert_eq!(web::ms_voice_steals(), 1);
    assert_eq!(web::ms_voice_cuts(), 0);
    assert!((web::ms_voice_steal_level(0) - 0.01).abs() < 1e-6);
    let mut out = block();
    out.extend(block());
    out.extend(block());
    let steps = out
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f32::max);
    assert!(
        steps < 0.01,
        "no step beyond the new voice's own entry: {steps}"
    );
    assert!(
        (out[300] - 1.0).abs() < 1e-6,
        "0.25 x 4, the quiet voice gone: {}",
        out[300]
    );
}

#[test]
fn no_budget_keeps_refusing_on_a_full_bus() {
    let _g = lock();
    session(&[("dc", "synthdef dc = 0.001")]);
    for i in 0..web::ms_bus_max_inputs() {
        assert!(spawn("dc", "a") > 0, "voice {i}");
    }
    assert_eq!(spawn("dc", "a"), 0, "a full bus with no budget refuses");
    assert_eq!(web::ms_voice_steals(), 0);
}

#[test]
fn reject_policy_refuses_over_budget() {
    let _g = lock();
    session(&[("dc", "synthdef dc = 0.25")]);
    assert_eq!(web::ms_set_voice_budget(1, 0), 0);
    assert!(spawn("dc", "a") > 0);
    assert_eq!(spawn("dc", "a"), 0);
    assert_eq!(web::ms_voice_steals(), 0);
}

#[test]
fn flat_bus_spawn_exports_use_the_budget_too() {
    let _g = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        with_str("dc", |np, nl| {
            with_str("synthdef dc = 0.25", |sp, sl| {
                assert_eq!(web::ms_register_def(np, nl, sp, sl), 0)
            })
        });
        assert_eq!(web::ms_set_voice_budget(1, 1), 0);
        let a = with_str("dc", |p, l| web::ms_spawn_voice_named(p, l));
        let b = with_str("dc", |p, l| web::ms_spawn_voice_named_panned(p, l, 0.5));
        assert!(a > 0 && b > 0);
        assert_eq!(
            web::ms_voice_steals(),
            1,
            "oldest stolen by the panned spawn"
        );
    }
}

#[test]
fn session_init_clears_the_budget_and_counts() {
    let _g = lock();
    session(&[("dc", "synthdef dc = 0.25")]);
    assert_eq!(web::ms_set_voice_budget(1, 1), 0);
    spawn("dc", "a");
    spawn("dc", "a");
    assert_eq!(web::ms_voice_steals(), 1);
    session(&[("dc", "synthdef dc = 0.25")]);
    assert_eq!(web::ms_voice_steals(), 0);
    spawn("dc", "a");
    assert!(spawn("dc", "a") > 0);
    assert_eq!(web::ms_voice_steals(), 0, "no budget after a fresh init");
}
