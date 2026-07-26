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

#[test]
fn test_ms_init_resets_legato_bookkeeping() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("lead");
        let (sptr, slen) = alloc_str(
            "synthdef lead freq=440.0 gate=0.0 = freq\n\nvoice lead mono legato freq 0.02",
        );
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        // Leave the track in a "tie-capable" state (held + gate open) —
        // exactly the state a surviving stale entry would misuse after a
        // fresh engine replaces the one it was recorded against.
        let voice = web::ms_legato_note_on(nptr, nlen, 220.0);
        assert!(voice > 0);
        for _ in 0..2 {
            render_block();
        }

        // Re-init via the plain path (paired with ms_compile/ms_render, not
        // ms_register_def/ms_legato_note_on) — this replaces the engine
        // with a fresh one whose VoiceIds restart at 1, so any surviving
        // legato bookkeeping from before would alias the new engine's ids.
        web::ms_init(44100.0);

        // DEF_REGISTRY is untouched by ms_init (by design — it belongs to
        // the ms_register_def/ms_spawn_voice_named workflow, not ms_init's),
        // so "lead" is technically still registered. But its *voice mode*
        // must be gone: ms_legato_note_on has nothing left to key a track
        // off of and must fail cleanly (0), not resurrect the stale track
        // and silently glide/gate an unrelated voice under the new engine.
        let second = web::ms_legato_note_on(nptr, nlen, 440.0);
        assert_eq!(
            second, 0,
            "ms_init must clear legato voice-mode state, not leave a stale \
             track whose held VoiceId could alias the new engine's ids"
        );

        free_str(nptr, nlen);
    }
}

#[test]
fn test_ms_init_resets_bus_node() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("test");
        let (sptr, slen) = alloc_str("synthdef test freq=440.0 = sinOsc freq 0.0");
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        // Re-init via the plain path — destroys the engine (and its graph)
        // that BUS_NODE's NodeId pointed into. In `ms_init_with_bus`, the
        // Bus is always the very first node added, so its NodeId's index is
        // 0 — a stale BUS_NODE would look exactly like this after re-init.
        web::ms_init(44100.0);

        // Immediately after `ms_init`, the new engine's graph is empty, so
        // a stale BUS_NODE would fail safe on its own (index 0 wouldn't
        // exist yet) — that's *why* this is latent rather than an
        // immediate crash, and a test stopping here wouldn't distinguish
        // fixed from broken. `ms_compile` is the "later call [that]
        // repopulates the graph" the doc comment warns about: it resets
        // the engine yet again and compiles a synthdef whose first node
        // (index 0, same as the old Bus) is a UGen with real input ports
        // (`sinOsc`, not a zero-input `Param`/`Const`) — index 0 is now a
        // live, unrelated node with slots a stale BUS_NODE could wire into.
        let (dptr, dlen) = alloc_str("synthdef dummy = sinOsc 440.0 0.0");
        assert_eq!(web::ms_compile(dptr, dlen), 0);
        free_str(dptr, dlen);

        // DEF_REGISTRY is untouched by ms_init (by design, see its doc
        // comment), so "test" is still registered — but BUS_NODE must be
        // cleared, or this wires the spawned voice into `dummy`'s sinOsc
        // node (index 0, now valid) instead of failing.
        let voice = web::ms_spawn_voice_named(nptr, nlen);
        assert_eq!(
            voice, 0,
            "ms_init must clear BUS_NODE — got a voice wired into whatever \
             now occupies the stale NodeId's index in the new graph"
        );

        free_str(nptr, nlen);
    }
}

#[test]
fn test_ms_legato_note_on_rewires_bus_after_reap_and_respawn() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);

        // "lead": will be brought to a full release, reaped via
        // ms_free_done, and then respawned by a second note_on.
        let (lead_nptr, lead_nlen) = alloc_str("lead");
        let (lead_sptr, lead_slen) = alloc_str(
            "synthdef lead freq=440.0 gate=0.0 = sinOsc freq 0.0 * asr gate 0.005 0.02\n\n\
             voice lead mono legato freq 0.02",
        );
        assert_eq!(
            web::ms_register_def(lead_nptr, lead_nlen, lead_sptr, lead_slen),
            0
        );
        free_str(lead_sptr, lead_slen);

        let lead_voice_1 = web::ms_legato_note_on(lead_nptr, lead_nlen, 440.0);
        assert!(lead_voice_1 > 0);

        let (slot_ptr, slot_len) = alloc_str("lead");
        let lead_slot_before = web::ms_legato_slot_for(slot_ptr, slot_len);
        free_str(slot_ptr, slot_len);
        assert!(
            lead_slot_before >= 0,
            "expected lead to be wired to a bus slot on its first note"
        );

        // A second, independent legato track, registered and wired to its
        // own slot *before* lead's reap+respawn. Its gate is set then
        // immediately cleared without an intervening render, so its ASR
        // never observes a rendered "on" sample and stays silently Idle
        // forever — it never makes a sound, but it did occupy a slot, which
        // is exactly what makes "recompute the counting-down formula on
        // respawn" (the bug) diverge from "reuse the stored slot" (the fix):
        // recomputing after a second track has registered a slot would
        // collide with that track's slot instead of returning to lead's own.
        let (second_nptr, second_nlen) = alloc_str("second");
        let (second_sptr, second_slen) = alloc_str(
            "synthdef second freq=220.0 gate=0.0 = sinOsc freq 0.0 * asr gate 0.005 0.02\n\n\
             voice second mono legato freq 0.02",
        );
        assert_eq!(
            web::ms_register_def(second_nptr, second_nlen, second_sptr, second_slen),
            0
        );
        free_str(second_sptr, second_slen);
        let second_voice = web::ms_legato_note_on(second_nptr, second_nlen, 220.0);
        assert!(second_voice > 0);
        web::ms_legato_note_off(second_nptr, second_nlen);

        let (slot_ptr2, slot_len2) = alloc_str("second");
        let second_slot = web::ms_legato_slot_for(slot_ptr2, slot_len2);
        free_str(slot_ptr2, slot_len2);
        assert!(
            second_slot >= 0 && second_slot != lead_slot_before,
            "expected second to occupy a distinct slot from lead"
        );

        // Bring lead through attack, release, and full decay to Idle.
        for _ in 0..10 {
            render_block();
        }
        web::ms_legato_note_off(lead_nptr, lead_nlen);
        for _ in 0..15 {
            render_block();
        }

        let freed = web::ms_free_done();
        assert!(
            freed > 0,
            "expected lead's fully-released envelope to be reaped"
        );

        // The respawn: this is the exact call the fix is about.
        let lead_voice_2 = web::ms_legato_note_on(lead_nptr, lead_nlen, 660.0);
        assert!(lead_voice_2 > 0);
        assert_ne!(
            lead_voice_2, lead_voice_1,
            "a reap must produce a genuinely fresh voice id, not reuse the dead one"
        );

        let (slot_ptr3, slot_len3) = alloc_str("lead");
        let lead_slot_after = web::ms_legato_slot_for(slot_ptr3, slot_len3);
        free_str(slot_ptr3, slot_len3);
        assert_eq!(
            lead_slot_after, lead_slot_before,
            "the respawned voice must reuse lead's original bus slot rather \
             than recomputing one that could collide with second's"
        );

        let (slot_ptr4, slot_len4) = alloc_str("second");
        let second_slot_after = web::ms_legato_slot_for(slot_ptr4, slot_len4);
        free_str(slot_ptr4, slot_len4);
        assert_eq!(
            second_slot_after, second_slot,
            "lead's respawn must not disturb second's slot"
        );

        // The decisive check: the respawned voice's audio must actually
        // reach the bus output. A wiring-skip bug leaves this silent even
        // though the voice exists and its envelope is genuinely running —
        // second stays silent throughout (see above), so any energy here
        // can only be lead's.
        let mut trace = Vec::new();
        for _ in 0..10 {
            trace.extend_from_slice(&render_block());
        }
        let peak = trace.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!(
            peak > 0.1,
            "expected the respawned legato voice to be audible at the bus \
             output after reap+respawn, got peak {peak}"
        );

        free_str(lead_nptr, lead_nlen);
        free_str(second_nptr, second_nlen);
    }
}

/// The specific ordering that makes "reuse the stored slot" and "recompute
/// the counting-down formula" diverge: track A's full note/release/reap/
/// respawn cycle happens *before* track B's very first note. If A's
/// respawn recomputed instead of reusing, it would land on the same slot
/// value B's first-ever computation would independently derive right
/// after (both read `slots.len()` as "1 other track registered") — a real
/// bus-slot collision between two unrelated tracks, not just A losing its
/// own slot. Each track's audibility is checked in isolation (A silenced
/// again before B ever starts) so a collision would show up as either a
/// wrong slot value or corrupted/missing audio, not just "some energy
/// somewhere."
#[test]
fn test_ms_legato_note_on_reap_respawn_then_new_track_get_distinct_slots() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);

        // -- Track A: full note / release / reap / respawn cycle --
        let (a_nptr, a_nlen) = alloc_str("trackA");
        let (a_sptr, a_slen) = alloc_str(
            "synthdef trackA freq=440.0 gate=0.0 = sinOsc freq 0.0 * asr gate 0.005 0.02\n\n\
             voice trackA mono legato freq 0.02",
        );
        assert_eq!(web::ms_register_def(a_nptr, a_nlen, a_sptr, a_slen), 0);
        free_str(a_sptr, a_slen);

        let a_voice_1 = web::ms_legato_note_on(a_nptr, a_nlen, 440.0);
        assert!(a_voice_1 > 0);
        let (slot_ptr, slot_len) = alloc_str("trackA");
        let a_slot_before = web::ms_legato_slot_for(slot_ptr, slot_len);
        free_str(slot_ptr, slot_len);
        assert!(a_slot_before >= 0);

        for _ in 0..10 {
            render_block();
        }
        web::ms_legato_note_off(a_nptr, a_nlen);
        for _ in 0..15 {
            render_block();
        }
        assert!(web::ms_free_done() > 0, "expected trackA to be reaped");

        let a_voice_2 = web::ms_legato_note_on(a_nptr, a_nlen, 660.0);
        assert!(a_voice_2 > 0);
        assert_ne!(
            a_voice_2, a_voice_1,
            "respawn must get a genuinely new voice id"
        );

        let (slot_ptr2, slot_len2) = alloc_str("trackA");
        let a_slot_after = web::ms_legato_slot_for(slot_ptr2, slot_len2);
        free_str(slot_ptr2, slot_len2);
        assert_eq!(
            a_slot_after, a_slot_before,
            "trackA's respawn must reuse its original slot"
        );

        // Confirm A's respawn is audible in isolation (trackB doesn't exist
        // yet, so any energy here can only be A's).
        let mut trace_a = Vec::new();
        for _ in 0..10 {
            trace_a.extend_from_slice(&render_block());
        }
        let peak_a = trace_a.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!(
            peak_a > 0.1,
            "expected trackA's respawned voice to be audible, got peak {peak_a}"
        );

        // Silence A again before starting B, so B's audibility check below
        // isn't confounded by A's still-sounding tone.
        web::ms_legato_note_off(a_nptr, a_nlen);
        for _ in 0..15 {
            render_block();
        }

        // -- Track B: registered and given its first-ever note *after* A's
        // full cycle above. This is the exact ordering where a recompute
        // (instead of reuse) on A's respawn would have collided with B. --
        let (b_nptr, b_nlen) = alloc_str("trackB");
        let (b_sptr, b_slen) = alloc_str(
            "synthdef trackB freq=330.0 gate=0.0 = sinOsc freq 0.0 * asr gate 0.005 0.02\n\n\
             voice trackB mono legato freq 0.02",
        );
        assert_eq!(web::ms_register_def(b_nptr, b_nlen, b_sptr, b_slen), 0);
        free_str(b_sptr, b_slen);

        let b_voice = web::ms_legato_note_on(b_nptr, b_nlen, 330.0);
        assert!(b_voice > 0);

        let (slot_ptr3, slot_len3) = alloc_str("trackB");
        let b_slot = web::ms_legato_slot_for(slot_ptr3, slot_len3);
        free_str(slot_ptr3, slot_len3);
        assert!(
            b_slot >= 0 && b_slot != a_slot_after,
            "trackB must get a slot distinct from trackA's (got trackA={a_slot_after}, trackB={b_slot})"
        );

        let mut trace_b = Vec::new();
        for _ in 0..10 {
            trace_b.extend_from_slice(&render_block());
        }
        let peak_b = trace_b.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!(
            peak_b > 0.1,
            "expected trackB's first note to be audible, got peak {peak_b}"
        );

        free_str(a_nptr, a_nlen);
        free_str(b_nptr, b_nlen);
    }
}

// -- The legato tie portamento's shape/space --
//
// `LegatoVoice`'s tie branch used to glide with `GlideShape::default()` /
// `GlideSpace::default()` (Linear/Raw) — a linear-in-Hz sweep on a parameter
// explicitly identified as driving note pitch, the exact defect the shape
// work fixed everywhere else. These three tests prove, through the raw
// C-ABI, that a legato tie now (1) defaults to an equal-ratio (Pitch-space)
// portamento, and (2)/(3) can be overridden to a different space/shape via
// the `voice` declaration's optional trailing terms.
//
// Each uses a synthdef whose output *is* the pitch parameter directly (no
// envelope in the way), so the rendered signal is the parameter's value.

#[test]
fn test_ms_legato_note_on_tie_defaults_to_pitch_space_portamento() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("lead");
        let (sptr, slen) = alloc_str(
            "synthdef lead freq=440.0 gate=0.0 = freq\n\nvoice lead mono legato freq 0.02",
        );
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        // First note: fresh attack, instant set to 220.0 (no glide involved).
        let voice = web::ms_legato_note_on(nptr, nlen, 220.0);
        assert!(voice > 0);
        for _ in 0..2 {
            render_block();
        }

        // Overlapping note: gate is still open, so this ties — using the
        // declaration's default shape/space (Linear/Pitch, since none was
        // written in the source above).
        let (nptr2, nlen2) = alloc_str("lead");
        let tied = web::ms_legato_note_on(nptr2, nlen2, 440.0);
        assert_eq!(tied, voice);
        free_str(nptr2, nlen2);

        let mut trace = Vec::new();
        for _ in 0..10 {
            trace.extend_from_slice(&render_block());
        }

        let total_samples = (0.02f32 * 44100.0) as usize;
        let mid = total_samples / 2;
        let expected_geometric_mid = (220.0f32 * 440.0).sqrt(); // ~311.13
        let arithmetic_mid = (220.0 + 440.0) / 2.0; // 330.0 — what Raw space would give
        assert!(
            (trace[mid] - expected_geometric_mid).abs() < 3.0,
            "default legato tie should be an equal-ratio (Pitch-space) portamento: \
             expected ~{expected_geometric_mid} at the midpoint, got {}",
            trace[mid]
        );
        assert!(
            (trace[mid] - arithmetic_mid).abs() > 5.0,
            "default legato tie should not be the old linear-in-Hz behavior \
             (midpoint {}, which reads as Raw's {arithmetic_mid})",
            trace[mid]
        );
        let last = *trace.last().unwrap();
        assert!(
            (last - 440.0).abs() < 1e-2,
            "expected the tie to have reached its target, got {last}"
        );

        free_str(nptr, nlen);
    }
}

#[test]
fn test_ms_legato_note_on_tie_can_override_to_raw_space() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("lead");
        let (sptr, slen) = alloc_str(
            "synthdef lead freq=440.0 gate=0.0 = freq\n\n\
             voice lead mono legato freq 0.02 linear raw",
        );
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        let voice = web::ms_legato_note_on(nptr, nlen, 220.0);
        assert!(voice > 0);
        for _ in 0..2 {
            render_block();
        }

        let (nptr2, nlen2) = alloc_str("lead");
        let tied = web::ms_legato_note_on(nptr2, nlen2, 440.0);
        assert_eq!(tied, voice);
        free_str(nptr2, nlen2);

        let mut trace = Vec::new();
        for _ in 0..10 {
            trace.extend_from_slice(&render_block());
        }

        let total_samples = (0.02f32 * 44100.0) as usize;
        let mid = total_samples / 2;
        let arithmetic_mid = (220.0f32 + 440.0) / 2.0; // 330.0
        let geometric_mid = (220.0f32 * 440.0).sqrt(); // ~311.13 — what Pitch space would give
        assert!(
            (trace[mid] - arithmetic_mid).abs() < 3.0,
            "explicit 'raw' should override the default to a linear-in-Hz sweep: \
             expected ~{arithmetic_mid} at the midpoint, got {}",
            trace[mid]
        );
        assert!(
            (trace[mid] - geometric_mid).abs() > 5.0,
            "explicit 'raw' should not still read as the Pitch-space default \
             (midpoint {}, which reads as Pitch's {geometric_mid})",
            trace[mid]
        );

        free_str(nptr, nlen);
    }
}

#[test]
fn test_ms_legato_note_on_tie_can_override_shape() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("lead");
        // Raw space isolates the shape check from the space's own curve.
        let (sptr, slen) = alloc_str(
            "synthdef lead freq=440.0 gate=0.0 = freq\n\n\
             voice lead mono legato freq 0.02 sine raw",
        );
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        let voice = web::ms_legato_note_on(nptr, nlen, 220.0);
        assert!(voice > 0);
        for _ in 0..2 {
            render_block();
        }

        let (nptr2, nlen2) = alloc_str("lead");
        let tied = web::ms_legato_note_on(nptr2, nlen2, 440.0);
        assert_eq!(tied, voice);
        free_str(nptr2, nlen2);

        let mut trace = Vec::new();
        for _ in 0..10 {
            trace.extend_from_slice(&render_block());
        }

        let total_samples = (0.02f32 * 44100.0) as usize;
        let quarter = total_samples / 4;
        let sine_frac = glide_fraction(GlideShape::Sine, 0.25);
        let expected_sine_quarter = 220.0 + sine_frac * (440.0 - 220.0);
        let expected_linear_quarter = 220.0 + 0.25 * (440.0 - 220.0); // 275.0
        assert!(
            (trace[quarter] - expected_sine_quarter).abs() < 3.0,
            "explicit 'sine' should override the default Linear shape: \
             expected ~{expected_sine_quarter} at x=0.25, got {}",
            trace[quarter]
        );
        assert!(
            (trace[quarter] - expected_linear_quarter).abs() > 5.0,
            "explicit 'sine' should not read as Linear at x=0.25 \
             (got {}, which is indistinguishable from linear {expected_linear_quarter})",
            trace[quarter]
        );

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

/// `ms_schedule_musical_glides` takes no `sample_rate` argument — it reads
/// the initialized engine's own rate instead (see the export's doc comment).
/// Uses a rate distinct from every other test in this file (44100 Hz) so a
/// regression back to some hardcoded/wrong rate would place the scheduled
/// event at the wrong sample and be caught here rather than passing by
/// coincidence.
#[test]
fn test_ms_schedule_musical_glides_uses_the_engines_own_sample_rate() {
    let _guard = lock();
    unsafe {
        let engine_sample_rate = 22050.0f32;
        web::ms_init_with_bus(engine_sample_rate);
        let (nptr, nlen) = alloc_str("test");
        let (sptr, slen) = alloc_str("synthdef test val=0.0 = val");
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        let voice = web::ms_spawn_voice_named(nptr, nlen);
        assert!(voice > 0);
        free_str(nptr, nlen);

        let config = TimeConfig::new_4_4(120.0, engine_sample_rate);
        let position = MusicalPosition::new(0, 4, 0);
        let expected_time = config.position_to_samples(position);
        let expected_glide_secs = config.steps_to_secs(2.0) as f32;

        let (pptr, plen) = alloc_str("val");
        let bars = [0u32];
        let steps = [4u16];
        let ticks = [0i16];
        let targets = [1.0f32];
        let glide_steps = [2.0f32];
        let shape_kinds = [1u32];
        let tensions = [0.0f32];
        let space_kinds = [0u32];

        let result = web::ms_schedule_musical_glides(
            voice,
            pptr,
            plen,
            config.bpm,
            config.numerator,
            config.denominator,
            config.grid_steps,
            config.ppqn,
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

        let total_end_sample =
            expected_time + (expected_glide_secs * engine_sample_rate).round() as u64;
        let mut trace = Vec::new();
        let mut sample_offset: u64 = 0;
        while sample_offset < total_end_sample + 256 {
            trace.extend_from_slice(&render_block());
            sample_offset += 128;
        }

        assert!(
            (trace[(expected_time.saturating_sub(64)) as usize] - 0.0).abs() < 1e-6,
            "value should be unchanged before the position computed at the \
             engine's actual {engine_sample_rate} Hz rate — a stale/wrong \
             rate would place this either too early or too late"
        );
        let after = (total_end_sample + 64) as usize;
        assert!(
            (trace[after.min(trace.len() - 1)] - 1.0).abs() < 1e-3,
            "expected the glide to reach target by sample {total_end_sample} \
             (computed at the engine's {engine_sample_rate} Hz rate), got {} at {after}",
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

// ============================================================================
// Composite: legato + musical-time scheduling on the same voice
// ============================================================================
//
// Every export above is exercised on its own; this is the one test that
// composes two of them the way an actual caller would — declare a
// mono/legato instrument, start it, and schedule a shaped, musical-time
// pitch sequence directly on the voice id `ms_legato_note_on` returned —
// proving the raw C-ABI surfaces work *together*, not just individually.

// Output is the raw envelope, not an audible tone modulated by it: an
// oscillating `sinOsc * envelope` signal crosses zero on every cycle of the
// tone itself, which would make a 0->positive zero-crossing counter count
// hundreds of "attacks" instead of the one real envelope attack (a mistake
// caught while adapting this test — see the module docs' precedent of using
// a bare envelope output for exactly this reason). "Audible" here means
// "reaches the bus and stays nonzero," which a raw envelope still proves.
const COMPOSITE_LEGATO_SOURCE: &str = "synthdef lead freq=440.0 gate=0.0 = asr gate 0.005 0.5\n\n\
     voice lead mono legato freq 0.01";

#[test]
fn test_legato_voice_accepts_a_scheduled_musical_time_sequence() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("lead");
        let (sptr, slen) = alloc_str(COMPOSITE_LEGATO_SOURCE);
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        // Declare + start the legato instrument through the raw C-ABI.
        let voice = web::ms_legato_note_on(nptr, nlen, 220.0);
        assert!(voice > 0, "legato instrument must start");
        free_str(nptr, nlen);

        // Schedule a shaped, musical-time pitch sequence directly on that
        // voice id: two segments mixing shapes (Sine, then Exponential) and
        // both in pitch space, at 120bpm on a 16th-note grid.
        let (pptr, plen) = alloc_str("freq");
        let bars = [0u32, 0];
        let steps = [4u16, 8];
        let ticks = [0i16, 0];
        let targets = [440.0f32, 330.0];
        let glide_steps = [2.0f32, 2.0];
        let shape_kinds = [2u32, 3]; // Sine, Exponential
        let tensions = [0.0f32, 3.0];
        let space_kinds = [1u32, 1]; // Pitch, Pitch

        let result = web::ms_schedule_musical_glides(
            voice,
            pptr,
            plen,
            120.0,
            4,
            4,
            16,
            96,
            bars.as_ptr(),
            steps.as_ptr(),
            ticks.as_ptr(),
            targets.as_ptr(),
            glide_steps.as_ptr(),
            shape_kinds.as_ptr(),
            tensions.as_ptr(),
            space_kinds.as_ptr(),
            2,
        );
        assert_eq!(result, 0, "scheduling on a legato voice must succeed");
        free_str(pptr, plen);

        // Render across the whole scheduled span (step 8 at 120bpm/16 steps
        // lands at ~44100 samples; its own glide adds ~11025 more) with
        // generous margin, capturing the bus output from the very first
        // render after note_on so the attack itself is included in the trace.
        let mut trace = Vec::new();
        for _ in 0..450 {
            trace.extend_from_slice(&render_block());
        }

        let peak = trace.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
        assert!(
            peak > 0.1,
            "the legato voice must still be sounding across the scheduled span, got peak {peak}"
        );

        assert_eq!(
            count_attacks(&trace),
            1,
            "a musical-time glide sequence scheduled on a legato voice must \
             not re-attack its envelope — this must be a pitch sequence \
             layered on the held note, not a series of new notes"
        );
    }
}

/// A caller can hold a voice id from before a reap (e.g. it scheduled a
/// sequence, then the note fully released and was reaped, then the track
/// respawned under a new id before the schedule's own segments were all
/// due). This confirms that composition stays harmless: scheduling against
/// the retired id still succeeds (matching every other `ms_schedule_*`/
/// `ms_voice_param_glide` export — none of them validate voice liveness at
/// schedule time), and when the event's time arrives it is a silent no-op
/// rather than landing on the *new* voice, because `Engine::spawn_voice`
/// never reuses a `VoiceId` within one engine's lifetime.
#[test]
fn test_scheduling_on_a_reaped_voice_id_does_not_affect_its_replacement() {
    let _guard = lock();
    unsafe {
        web::ms_init_with_bus(44100.0);
        let (nptr, nlen) = alloc_str("lead2");
        // Output is `freq` directly so the respawned voice's pitch can be
        // read straight off the bus; the envelope exists only so the synth
        // can report is_done() and be reaped — multiplying its value by 0.0
        // keeps it out of the observed output.
        let (sptr, slen) = alloc_str(
            "synthdef lead2 freq=440.0 gate=0.0 = freq + asr gate 0.005 0.02 * 0.0\n\n\
             voice lead2 mono legato freq 0.02",
        );
        assert_eq!(web::ms_register_def(nptr, nlen, sptr, slen), 0);
        free_str(sptr, slen);

        let v1 = web::ms_legato_note_on(nptr, nlen, 220.0);
        assert!(v1 > 0);
        for _ in 0..5 {
            render_block();
        }
        web::ms_legato_note_off(nptr, nlen);
        for _ in 0..15 {
            render_block();
        }
        assert!(web::ms_free_done() > 0, "expected lead2 to be reaped");

        let v2 = web::ms_legato_note_on(nptr, nlen, 440.0);
        assert!(v2 > 0);
        assert_ne!(v2, v1, "respawn must get a genuinely new voice id");
        free_str(nptr, nlen);

        // Schedule against the STALE id v1, as a caller still holding it
        // from before the reap would.
        let (pptr, plen) = alloc_str("freq");
        let bars = [0u32];
        let steps = [1u16];
        let ticks = [0i16];
        let targets = [880.0f32];
        let glide_steps = [1.0f32];
        let shape_kinds = [1u32];
        let tensions = [0.0f32];
        let space_kinds = [0u32];
        let result = web::ms_schedule_musical_glides(
            v1,
            pptr,
            plen,
            120.0,
            4,
            4,
            16,
            96,
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
        assert_eq!(
            result, 0,
            "scheduling against a stale voice id still succeeds — the engine \
             has no way to know it's stale until dispatch time"
        );
        free_str(pptr, plen);

        // Render well past the scheduled position and its glide.
        let mut last = 0.0f32;
        for _ in 0..100 {
            last = *render_block().last().unwrap();
        }
        assert!(
            (last - 440.0).abs() < 1e-3,
            "the schedule aimed at the retired voice must not have reached \
             v2's freq (still expected 440.0, got {last})"
        );
    }
}
