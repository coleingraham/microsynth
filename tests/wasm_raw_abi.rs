//! Integration tests for the raw C-ABI (`microsynth::web::ms_*`) — the
//! surface the AudioWorklet render path actually drives (see `src/web.rs`
//! module docs: the wasm-bindgen `WebSynth` class is a separate surface for
//! the main thread). These tests call the `#[no_mangle] extern "C"`
//! functions directly, the same way a JS caller would (writing bytes into
//! an `ms_alloc`'d buffer, passing pointer + length), just without an actual
//! WASM boundary in between.
//!
//! # Why a lock
//!
//! The raw exports are backed by process-global statics (`WasmCell`), matching
//! the AudioWorklet's single logical instance per session. That's safe in
//! production because a WASM module instance is inherently single-threaded,
//! but `cargo test` runs the `#[test]` functions in this file concurrently by
//! default, and they all share the same statics within one test binary. All
//! tests below take `TEST_LOCK` for their whole body to serialize access, so
//! each behaves like its own exclusive session — this is a test-harness
//! concern only, not a production one.

use microsynth::musical_time::{MusicalPosition, TimeConfig};
use microsynth::web;
use microsynth::{GlideShape, glide_fraction};
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

fn free_str(ptr: *mut u8, len: usize) {
    unsafe { web::ms_free(ptr, len) };
}

/// Render one 128-sample block, returning the left channel (mono synthdefs
/// under test broadcast the same value to every bus channel, so left ==
/// right; see `ugens::Bus::process`).
fn render_block() -> [f32; 128] {
    let mut left = [0f32; 128];
    let mut right = [0f32; 128];
    unsafe { web::ms_render(left.as_mut_ptr(), right.as_mut_ptr()) };
    let _ = right;
    left
}

/// Count 0 -> positive transitions in an envelope trace: each one is a
/// fresh attack (mirrors `tests/voice_allocation.rs`'s helper of the same
/// name, applied here to audio captured through the raw C-ABI instead of
/// direct `Engine` access).
fn count_attacks(samples: &[f32]) -> usize {
    let mut count = 0;
    let mut prev = 0.0f32;
    for &s in samples {
        if prev <= 1e-6 && s > 1e-6 {
            count += 1;
        }
        prev = s;
    }
    count
}

// ============================================================================
// Shaped parameter glide (`ms_voice_param_glide`)
// ============================================================================

#[test]
fn test_ms_voice_param_glide_linear_raw_reaches_target() {
    let _guard = lock();
    unsafe {
        web::ms_init(44100.0);
        let (sptr, slen) = alloc_str("synthdef test val=0.0 = val");
        assert_eq!(web::ms_compile_def(sptr, slen), 0);
        free_str(sptr, slen);

        let voice = web::ms_spawn_voice();
        assert!(voice > 0, "expected a voice id");

        let (pptr, plen) = alloc_str("val");
        // Linear (shape_kind=1), Raw space (space_kind=0), glide to 1.0 over 0.01s.
        web::ms_voice_param_glide(voice, pptr, plen, 1.0, 0.01, 1, 0.0, 0);
        free_str(pptr, plen);

        let mut trace = Vec::new();
        for _ in 0..10 {
            trace.extend_from_slice(&render_block());
        }

        let total_samples = (0.01f32 * 44100.0) as usize;
        let mid = total_samples / 2;
        assert!(
            (trace[mid] - 0.5).abs() < 0.02,
            "expected ~0.5 at the linear glide's midpoint, got {}",
            trace[mid]
        );
        let last = *trace.last().unwrap();
        assert!(
            (last - 1.0).abs() < 1e-3,
            "expected the glide to have reached its target, got {last}"
        );
    }
}

#[test]
fn test_ms_voice_param_glide_sine_shape_is_not_linear() {
    let _guard = lock();
    unsafe {
        web::ms_init(44100.0);
        let (sptr, slen) = alloc_str("synthdef test val=0.0 = val");
        assert_eq!(web::ms_compile_def(sptr, slen), 0);
        free_str(sptr, slen);

        let voice = web::ms_spawn_voice();
        assert!(voice > 0);

        let (pptr, plen) = alloc_str("val");
        // Sine (shape_kind=2), Raw space, glide to 1.0 over 0.02s.
        web::ms_voice_param_glide(voice, pptr, plen, 1.0, 0.02, 2, 0.0, 0);
        free_str(pptr, plen);

        let mut trace = Vec::new();
        for _ in 0..20 {
            trace.extend_from_slice(&render_block());
        }

        let total_samples = (0.02f32 * 44100.0) as usize;
        let quarter = total_samples / 4;
        let expected_sine = glide_fraction(GlideShape::Sine, 0.25);
        let expected_linear = 0.25f32;
        assert!(
            (trace[quarter] - expected_sine).abs() < 0.02,
            "expected raised-cosine value ~{expected_sine} at x=0.25, got {}",
            trace[quarter]
        );
        assert!(
            (trace[quarter] - expected_linear).abs() > 0.03,
            "shape_kind=2 should behave differently from Linear at x=0.25 \
             (got {}, which is indistinguishable from linear {expected_linear})",
            trace[quarter]
        );
    }
}

#[test]
fn test_ms_voice_param_glide_pitch_space_uses_geometric_midpoint() {
    let _guard = lock();
    unsafe {
        web::ms_init(44100.0);
        let (sptr, slen) = alloc_str("synthdef test val=220.0 = val");
        assert_eq!(web::ms_compile_def(sptr, slen), 0);
        free_str(sptr, slen);

        let voice = web::ms_spawn_voice();
        assert!(voice > 0);

        let (pptr, plen) = alloc_str("val");
        // Linear shape, Pitch space (space_kind=1): a one-octave glide.
        web::ms_voice_param_glide(voice, pptr, plen, 440.0, 0.02, 1, 0.0, 1);
        free_str(pptr, plen);

        let mut trace = Vec::new();
        for _ in 0..20 {
            trace.extend_from_slice(&render_block());
        }

        let total_samples = (0.02f32 * 44100.0) as usize;
        let mid = total_samples / 2;
        let expected_geometric_mid = (220.0f32 * 440.0).sqrt(); // ~311.13
        let arithmetic_mid = (220.0 + 440.0) / 2.0; // 330.0 — what Raw space would give
        assert!(
            (trace[mid] - expected_geometric_mid).abs() < 2.0,
            "expected pitch-space midpoint ~{expected_geometric_mid} (equal-ratio sweep), got {}",
            trace[mid]
        );
        assert!(
            (trace[mid] - arithmetic_mid).abs() > 5.0,
            "space_kind=1 should not behave like Raw's linear-in-Hz midpoint {arithmetic_mid} \
             (got {})",
            trace[mid]
        );
    }
}

// ============================================================================
// Mono/legato voice mode (`ms_legato_note_on` / `ms_legato_note_off`)
// ============================================================================

/// DSL source for a synthdef whose output *is* its envelope (so the
/// rendered signal can be inspected directly for attacks), declared with a
/// `voice ... mono legato ...` mode.
const LEGATO_DEF_SOURCE: &str =
    "synthdef lead freq=440.0 gate=0.0 = asr gate 0.005 0.02\n\nvoice lead mono legato freq 0.01";

#[test]
fn test_ms_legato_note_on_tie_produces_single_attack() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("lead");
        let (sptr, slen) = alloc_str(LEGATO_DEF_SOURCE);
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        let (nptr2, nlen2) = alloc_str("lead");
        let voice = web::ms_legato_note_on(nptr, nlen, 440.0);
        assert!(voice > 0, "expected a legato voice id");

        let mut trace = Vec::new();
        for _ in 0..15 {
            trace.extend_from_slice(&render_block());
        }

        // Overlapping note: gate is still open (no note_off yet), so this
        // must tie into the same voice rather than retrigger.
        let tied_voice = web::ms_legato_note_on(nptr2, nlen2, 660.0);
        assert_eq!(tied_voice, voice, "legato tie must reuse the same voice");

        for _ in 0..20 {
            trace.extend_from_slice(&render_block());
        }

        web::ms_legato_note_off(nptr, nlen);
        for _ in 0..25 {
            trace.extend_from_slice(&render_block());
        }

        assert_eq!(
            count_attacks(&trace),
            1,
            "a fully tied legato run must produce exactly one envelope attack"
        );

        free_str(nptr, nlen);
        free_str(nptr2, nlen2);
    }
}

#[test]
fn test_ms_legato_note_on_gap_retriggers_same_voice() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("lead");
        let (sptr, slen) = alloc_str(LEGATO_DEF_SOURCE);
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        let voice = web::ms_legato_note_on(nptr, nlen, 440.0);
        assert!(voice > 0);

        let mut trace = Vec::new();
        for _ in 0..15 {
            trace.extend_from_slice(&render_block());
        }

        // Release fully: 0.02s release is well under 25 blocks at 128 samples.
        web::ms_legato_note_off(nptr, nlen);
        for _ in 0..25 {
            trace.extend_from_slice(&render_block());
        }

        // A new note after the gap: must retrigger, but stay on the same voice.
        let second_voice = web::ms_legato_note_on(nptr, nlen, 550.0);
        assert_eq!(
            second_voice, voice,
            "mono mode keeps the same voice across a gap"
        );
        for _ in 0..15 {
            trace.extend_from_slice(&render_block());
        }

        assert_eq!(
            count_attacks(&trace),
            2,
            "a gap between notes must produce a second envelope attack"
        );

        free_str(nptr, nlen);
    }
}

#[test]
fn test_ms_legato_note_on_unregistered_name_returns_zero() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("nonexistent");
        let voice = web::ms_legato_note_on(nptr, nlen, 440.0);
        assert_eq!(voice, 0, "an unregistered name must not produce a voice");
        free_str(nptr, nlen);
    }
}

// ============================================================================
// Musical-time sequenced glide (`ms_schedule_musical_glides`)
// ============================================================================

#[test]
fn test_ms_schedule_musical_glides_lands_at_expected_sample_time() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("test");
        let (sptr, slen) = alloc_str("synthdef test val=0.0 = val");
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        let voice = web::ms_spawn_voice_named(nptr, nlen);
        assert!(voice > 0);
        free_str(nptr, nlen);

        // Same TimeConfig as passed (as scalars) to the raw export, used
        // here only to compute the *expected* sample time/glide length from
        // the already-tested pure conversion (see tests/musical_primitives.rs)
        // — not to reimplement it.
        let config = TimeConfig::new_4_4(120.0, 44100.0);
        let position = MusicalPosition::new(0, 4, 0);
        let expected_time = config.position_to_samples(position);
        let expected_glide_secs = config.steps_to_secs(2.0) as f32;

        let (pptr, plen) = alloc_str("val");
        let bars = [0u32];
        let steps = [4u16];
        let ticks = [0i16];
        let targets = [1.0f32];
        let glide_steps = [2.0f32];
        let shape_kinds = [1u32]; // Linear
        let tensions = [0.0f32];
        let space_kinds = [0u32]; // Raw

        let result = web::ms_schedule_musical_glides(
            voice,
            pptr,
            plen,
            config.bpm,
            config.numerator,
            config.denominator,
            config.grid_steps,
            config.ppqn,
            config.sample_rate,
            bars.as_ptr(),
            steps.as_ptr(),
            ticks.as_ptr(),
            targets.as_ptr(),
            glide_steps.as_ptr(),
            shape_kinds.as_ptr(),
            tensions.as_ptr(),
            space_kinds.as_ptr(),
            1,
        );
        assert_eq!(result, 0);
        free_str(pptr, plen);

        let total_end_sample = expected_time + (expected_glide_secs * 44100.0).round() as u64;
        let mut trace = Vec::new();
        let mut sample_offset: u64 = 0;
        while sample_offset < total_end_sample + 256 {
            trace.extend_from_slice(&render_block());
            sample_offset += 128;
        }

        // Before the scheduled position: untouched.
        assert!(
            (trace[(expected_time.saturating_sub(64)) as usize] - 0.0).abs() < 1e-6,
            "value should be unchanged before the scheduled musical position"
        );
        // After the glide completes: at target.
        let after = (total_end_sample + 64) as usize;
        assert!(
            (trace[after.min(trace.len() - 1)] - 1.0).abs() < 1e-3,
            "expected the musical-time glide to reach its target by sample {total_end_sample}, \
             got {} at {after}",
            trace[after.min(trace.len() - 1)]
        );
    }
}

#[test]
fn test_ms_schedule_musical_glides_empty_segment_list_is_a_safe_no_op() {
    let _guard = lock();
    // count == 0 must be a well-defined no-op even with null array pointers
    // (a real caller with nothing to schedule has no other obviously-valid
    // pointer to pass) — see the doc comment on `ms_schedule_musical_glides`
    // and the early `count == 0` return in its body, added specifically so
    // this doesn't reach `slice::from_raw_parts`, which requires non-null
    // pointers even for a zero-length slice.
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (pptr, plen) = alloc_str("val");
        let result = web::ms_schedule_musical_glides(
            999,
            pptr,
            plen,
            120.0,
            4,
            4,
            16,
            96,
            44100.0,
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            core::ptr::null(),
            0,
        );
        assert_eq!(
            result, 0,
            "an empty (count=0) segment list is a valid no-op"
        );
        free_str(pptr, plen);
    }
}
