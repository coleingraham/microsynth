//! Tests for the voice-allocation layer (`microsynth::voice`):
//! - `VoiceAllocator`: voice budgets and stealing policies, layered over
//!   `Engine::spawn_voice`/`free_voice`.
//! - `LegatoVoice`: a mono/legato voice mode that ties or retriggers a
//!   single held voice, and is independent of any allocation policy.

use microsynth::dsl;
use microsynth::*;

mod common;
use common::builtin_registry;

fn make_engine(block_size: usize) -> Engine {
    Engine::new(EngineConfig {
        sample_rate: 44100.0,
        block_size,
    })
}

/// Count 0 -> positive transitions in an envelope trace: each one is a
/// fresh attack (the envelope only returns to exactly 0.0 when idle or
/// fully released).
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
// VoiceAllocator: budgets and stealing policies
// ============================================================================

#[test]
fn test_allocator_admits_within_budget() {
    let registry = builtin_registry();
    let defs = dsl::compile("synthdef test freq=440.0 = sinOsc freq 0.0", &registry).unwrap();

    let mut engine = make_engine(64);
    engine.set_voice_allocator(VoiceAllocator::new(2, StealPolicy::Reject));

    let v1 = engine.spawn_voice_managed(&defs[0]);
    let v2 = engine.spawn_voice_managed(&defs[0]);

    assert!(v1.is_some());
    assert!(v2.is_some());
    assert_eq!(engine.voice_allocator().unwrap().len(), 2);
    assert_eq!(engine.synths().len(), 2);
}

#[test]
fn test_allocator_reject_policy_declines_over_budget() {
    let registry = builtin_registry();
    let defs = dsl::compile("synthdef test freq=440.0 = sinOsc freq 0.0", &registry).unwrap();

    let mut engine = make_engine(64);
    engine.set_voice_allocator(VoiceAllocator::new(1, StealPolicy::Reject));

    let v1 = engine.spawn_voice_managed(&defs[0]);
    let v2 = engine.spawn_voice_managed(&defs[0]);

    assert!(v1.is_some(), "first spawn should be admitted");
    assert!(v2.is_none(), "second spawn should be declined at budget 1");
    assert_eq!(
        engine.synths().len(),
        1,
        "no synth should leak for the declined spawn"
    );
    assert_eq!(
        engine.voice_allocator().unwrap().active_voices(),
        &[v1.unwrap()]
    );
}

#[test]
fn test_allocator_oldest_policy_steals_fifo() {
    let registry = builtin_registry();
    let defs = dsl::compile("synthdef test freq=440.0 = sinOsc freq 0.0", &registry).unwrap();

    let mut engine = make_engine(64);
    engine.set_voice_allocator(VoiceAllocator::new(1, StealPolicy::Oldest));

    let v1 = engine.spawn_voice_managed(&defs[0]).unwrap();
    let v2 = engine.spawn_voice_managed(&defs[0]).unwrap();

    assert!(
        engine.voice_synth(v1).is_none(),
        "oldest voice should have been stolen"
    );
    assert!(
        engine.voice_synth(v2).is_some(),
        "newest voice should still be alive"
    );
    assert_eq!(engine.voice_allocator().unwrap().active_voices(), &[v2]);
    assert_eq!(engine.synths().len(), 1);
}

#[test]
fn test_allocator_newest_policy_steals_lifo() {
    let registry = builtin_registry();
    let defs = dsl::compile("synthdef test freq=440.0 = sinOsc freq 0.0", &registry).unwrap();

    let mut engine = make_engine(64);
    engine.set_voice_allocator(VoiceAllocator::new(2, StealPolicy::Newest));

    let v1 = engine.spawn_voice_managed(&defs[0]).unwrap();
    let v2 = engine.spawn_voice_managed(&defs[0]).unwrap();
    let v3 = engine.spawn_voice_managed(&defs[0]).unwrap();

    assert!(
        engine.voice_synth(v1).is_some(),
        "oldest voice should survive under Newest"
    );
    assert!(
        engine.voice_synth(v2).is_none(),
        "most recently active voice before the 3rd spawn should be stolen"
    );
    assert!(engine.voice_synth(v3).is_some());
    assert_eq!(engine.voice_allocator().unwrap().active_voices(), &[v1, v3]);
}

#[test]
fn test_spawn_on_bus_managed_steal_then_bus_full_still_frees_the_victim() {
    // A steal and the bus-slot check are two separate decisions: the steal
    // can succeed (a voice really is freed) even though the bus spawn that
    // triggered it still fails for an unrelated reason (no free slot). A
    // caller must not read a `None` return as "nothing happened".
    let registry = builtin_registry();
    let defs = dsl::compile("synthdef test freq=440.0 = sinOsc freq 0.0", &registry).unwrap();

    let mut engine = make_engine(64);
    // A bus's input-slot count is a fixed 64 regardless of the channel count
    // passed to `Bus::new` — fill every slot with voices the allocator never
    // sees, so there is truly no room left.
    let bus = engine.graph_mut().add_node(Box::new(ugens::Bus::new(2)));
    engine.graph_mut().set_sink(bus);
    let bus_voices: Vec<_> = (0..64)
        .map(|_| engine.spawn_voice_on_bus(&defs[0], bus).unwrap())
        .collect();
    engine.prepare();

    // A managed, off-bus voice fills the budget; this is the steal victim.
    // Oldest is a non-Reject policy, so a full budget really does steal.
    engine.set_voice_allocator(VoiceAllocator::new(1, StealPolicy::Oldest));
    let victim = engine.spawn_voice_managed(&defs[0]).unwrap();
    let count_before = engine.synths().len();

    // Budget is full, so this must steal `victim` — but the bus still has no
    // free slot (all 64 are `bus_voices`, untouched by the steal), so the
    // spawn itself fails.
    let result = engine.spawn_voice_on_bus_managed(&defs[0], bus);

    assert!(
        result.is_none(),
        "the bus has no free slot, so this must fail"
    );
    // The observable proof that the steal still happened despite the
    // failure: the live voice count actually dropped by one (the victim),
    // not just "the spawn produced nothing".
    assert_eq!(
        engine.synths().len(),
        count_before - 1,
        "the victim must have been freed even though the spawn returned None"
    );
    assert!(
        engine.voice_synth(victim).is_none(),
        "the steal must have already freed the victim, even though the spawn failed"
    );
    assert_eq!(
        engine.voice_allocator().unwrap().len(),
        0,
        "the allocator's bookkeeping should reflect the freed victim"
    );
    for bus_voice in &bus_voices {
        assert!(
            engine.voice_synth(*bus_voice).is_some(),
            "the unrelated bus occupants must be untouched"
        );
    }
    assert_eq!(
        engine.synths().len(),
        64,
        "only the original bus voices should remain"
    );
}

#[test]
fn test_direct_free_voice_updates_allocator_bookkeeping() {
    let registry = builtin_registry();
    let defs = dsl::compile("synthdef test freq=440.0 = sinOsc freq 0.0", &registry).unwrap();

    let mut engine = make_engine(64);
    engine.set_voice_allocator(VoiceAllocator::new(1, StealPolicy::Reject));

    let v1 = engine.spawn_voice_managed(&defs[0]).unwrap();
    // Free directly, bypassing the managed spawn path entirely.
    engine.free_voice(v1);

    assert_eq!(engine.voice_allocator().unwrap().len(), 0);
    // The budget is free again, so a new spawn should be admitted.
    let v2 = engine.spawn_voice_managed(&defs[0]);
    assert!(v2.is_some());
}

#[test]
fn test_free_done_synths_updates_allocator_bookkeeping() {
    let registry = builtin_registry();
    // A short Perc envelope with no gate: finishes on its own.
    let defs = dsl::compile("synthdef test = perc 0.001 0.005", &registry).unwrap();

    let mut engine = make_engine(64);
    let bus = engine.graph_mut().add_node(Box::new(ugens::Bus::new(4)));
    engine.graph_mut().set_sink(bus);
    engine.set_voice_allocator(VoiceAllocator::new(4, StealPolicy::Reject));

    let v1 = engine
        .spawn_voice_on_bus_managed(&defs[0], bus)
        .expect("bus should have a free slot");
    engine.prepare();

    assert_eq!(engine.voice_allocator().unwrap().len(), 1);

    // Render long enough for the Perc envelope to finish (attack + release
    // is well under a second).
    for _ in 0..40 {
        engine.render();
    }
    let removed = engine.free_done_synths();
    assert!(removed > 0, "expected the Perc envelope to signal done");
    assert_eq!(
        engine.voice_allocator().unwrap().len(),
        0,
        "allocator bookkeeping should follow natural voice death, not just explicit free_voice"
    );
    let _ = v1;
}

// ============================================================================
// LegatoVoice: mono/legato mode
// ============================================================================

/// A synthdef whose output *is* its envelope, so the rendered signal can be
/// inspected directly for attacks instead of guessing from audible pitch.
fn mono_def() -> SynthDef {
    let registry = builtin_registry();
    dsl::compile(
        "synthdef mono freq=440.0 gate=0.0 = asr gate 0.005 0.02",
        &registry,
    )
    .unwrap()
    .remove(0)
}

#[test]
fn test_legato_tie_produces_single_attack_and_updates_pitch() {
    let def = mono_def();
    let mut engine = make_engine(64);
    let mut legato = LegatoVoice::new("freq", 0.01);

    let voice = legato.note_on(&mut engine, &def, 440.0);
    let synth = engine.voice_synth(voice).unwrap();
    let env_node = synth.output_node();
    let freq_node = synth.param_node("freq").unwrap();
    engine.graph_mut().set_sink(env_node);
    engine.prepare();

    let mut trace = Vec::new();
    let render_and_capture = |engine: &mut Engine, trace: &mut Vec<f32>| {
        engine.render();
        if let Some(buf) = engine.graph().node_output(env_node) {
            trace.extend_from_slice(buf.channel(0).samples());
        }
    };

    // Attack + settle into sustain.
    for _ in 0..15 {
        render_and_capture(&mut engine, &mut trace);
    }

    // Overlapping note: gate is still open, so this must tie, not retrigger.
    let tied_voice = legato.note_on(&mut engine, &def, 660.0);
    assert_eq!(tied_voice, voice, "legato tie must reuse the same voice");

    // Let the glide finish and hold in sustain a while longer.
    for _ in 0..20 {
        render_and_capture(&mut engine, &mut trace);
    }

    legato.note_off(&mut engine);
    for _ in 0..25 {
        render_and_capture(&mut engine, &mut trace);
    }

    assert_eq!(
        count_attacks(&trace),
        1,
        "a fully tied legato run must produce exactly one envelope attack"
    );

    // The pitch itself really did change, via a glide (set_target), not a
    // second attack.
    let final_freq = engine
        .graph()
        .node_output(freq_node)
        .unwrap()
        .channel(0)
        .samples()
        .last()
        .copied()
        .unwrap();
    assert!(
        (final_freq - 660.0).abs() < 0.01,
        "expected freq to have glided to 660, got {final_freq}"
    );
}

#[test]
fn test_legato_gap_retriggers_same_voice() {
    let def = mono_def();
    let mut engine = make_engine(64);
    let mut legato = LegatoVoice::new("freq", 0.01);

    let voice = legato.note_on(&mut engine, &def, 440.0);
    let synth = engine.voice_synth(voice).unwrap();
    let env_node = synth.output_node();
    engine.graph_mut().set_sink(env_node);
    engine.prepare();

    let mut trace = Vec::new();
    let render_and_capture = |engine: &mut Engine, trace: &mut Vec<f32>| {
        engine.render();
        if let Some(buf) = engine.graph().node_output(env_node) {
            trace.extend_from_slice(buf.channel(0).samples());
        }
    };

    for _ in 0..15 {
        render_and_capture(&mut engine, &mut trace);
    }

    // Release fully: 0.02s release is well under 20 blocks at 64 samples.
    legato.note_off(&mut engine);
    for _ in 0..25 {
        render_and_capture(&mut engine, &mut trace);
    }
    assert!(!legato.is_gate_open());

    // A new note after the gap: must retrigger, but stay on the same voice.
    let second_voice = legato.note_on(&mut engine, &def, 550.0);
    assert_eq!(
        second_voice, voice,
        "mono mode keeps the same voice across a gap"
    );
    for _ in 0..15 {
        render_and_capture(&mut engine, &mut trace);
    }

    assert_eq!(
        count_attacks(&trace),
        2,
        "a gap between notes must produce a second envelope attack"
    );
    assert_eq!(
        engine.synths().len(),
        1,
        "only ever one synth for the whole mono track"
    );
}

#[test]
fn test_legato_respawns_after_release_and_reap() {
    // A full release can end with the voice being reaped by
    // `Engine::free_done_synths` (e.g. a host polling for finished envelopes)
    // before the next note arrives. `held` must not go on pointing at a voice
    // that no longer exists: the next `note_on` has to notice and start a
    // fresh one, not silently no-op while claiming success.
    let def = mono_def();
    let mut engine = make_engine(64);
    let mut legato = LegatoVoice::new("freq", 0.0);

    let voice1 = legato.note_on(&mut engine, &def, 440.0);
    let env1 = engine.voice_synth(voice1).unwrap().output_node();
    engine.graph_mut().set_sink(env1);
    engine.prepare();

    let mut trace1 = Vec::new();
    for _ in 0..15 {
        engine.render();
        if let Some(buf) = engine.graph().node_output(env1) {
            trace1.extend_from_slice(buf.channel(0).samples());
        }
    }
    assert_eq!(count_attacks(&trace1), 1);

    legato.note_off(&mut engine);
    // 0.02s release is well under 40 blocks at 64 samples: render past it so
    // the envelope actually reports done.
    for _ in 0..40 {
        engine.render();
        if let Some(buf) = engine.graph().node_output(env1) {
            trace1.extend_from_slice(buf.channel(0).samples());
        }
    }

    let removed = engine.free_done_synths();
    assert!(removed > 0, "expected the released voice to be reaped");
    assert!(
        engine.voice_synth(voice1).is_none(),
        "sanity check: the reap should have actually removed the voice"
    );

    // The next note_on must produce a genuinely live, sounding voice — not
    // silently return the id of a voice that's already gone.
    let voice2 = legato.note_on(&mut engine, &def, 660.0);
    assert!(
        engine.voice_synth(voice2).is_some(),
        "note_on after the held voice was reaped must return a live voice, \
         not the stale id of the one that was just freed"
    );

    let synth2 = engine.voice_synth(voice2).unwrap();
    let env2 = synth2.output_node();
    let freq2 = synth2.param_node("freq").unwrap();
    engine.graph_mut().set_sink(env2);
    engine.prepare();

    let mut trace2 = Vec::new();
    for _ in 0..15 {
        engine.render();
        if let Some(buf) = engine.graph().node_output(env2) {
            trace2.extend_from_slice(buf.channel(0).samples());
        }
    }
    assert_eq!(
        count_attacks(&trace2),
        1,
        "note_on after a reap must produce a fresh envelope attack, not silence"
    );

    // Not just *an* attack — the respawned voice must actually carry the
    // pitch that was requested, not a leftover/default value.
    let final_freq = engine
        .graph()
        .node_output(freq2)
        .unwrap()
        .channel(0)
        .samples()
        .last()
        .copied()
        .unwrap();
    assert!(
        (final_freq - 660.0).abs() < 0.01,
        "respawned voice must carry the requested pitch (660), got {final_freq}"
    );
}

#[test]
fn test_legato_independent_of_allocation_policy() {
    let def = mono_def();
    // Fold over the full set of stealing policies rather than picking one:
    // legato must render identically no matter which is attached, because
    // it never asks for more than the one voice its budget already covers.
    let policies = [
        StealPolicy::Reject,
        StealPolicy::Oldest,
        StealPolicy::Newest,
    ];

    let mut recordings: Vec<Vec<f32>> = Vec::new();
    for policy in policies {
        let mut engine = make_engine(64);
        engine.set_voice_allocator(VoiceAllocator::new(1, policy));
        let mut legato = LegatoVoice::new("freq", 0.0);

        let voice = legato.note_on(&mut engine, &def, 440.0);
        let synth = engine.voice_synth(voice).unwrap();
        let env_node = synth.output_node();
        engine.graph_mut().set_sink(env_node);
        engine.prepare();

        let mut trace = Vec::new();
        for i in 0..40 {
            engine.render();
            if let Some(buf) = engine.graph().node_output(env_node) {
                trace.extend_from_slice(buf.channel(0).samples());
            }
            if i == 10 {
                legato.note_on(&mut engine, &def, 660.0);
            }
            if i == 20 {
                legato.note_on(&mut engine, &def, 550.0);
            }
        }
        legato.note_off(&mut engine);
        for _ in 0..20 {
            engine.render();
            if let Some(buf) = engine.graph().node_output(env_node) {
                trace.extend_from_slice(buf.channel(0).samples());
            }
        }

        // The allocator was never touched by legato's single voice: it
        // spawns through the unmanaged `spawn_voice`, not the managed path.
        assert_eq!(engine.voice_allocator().unwrap().len(), 0);
        recordings.push(trace);
    }

    assert_eq!(count_attacks(&recordings[0]), 1);
    for r in &recordings[1..] {
        assert_eq!(
            r, &recordings[0],
            "legato output must be bit-identical regardless of the attached stealing policy"
        );
    }
}

#[test]
fn test_polyphonic_material_unaffected_by_unused_allocator_field() {
    // Regression: plain spawn_voice/spawn_voice_on_bus/free_voice must
    // behave exactly as before when no allocator is attached at all.
    let registry = builtin_registry();
    let defs = dsl::compile(
        "synthdef tone freq=440.0 gate=1.0 = sinOsc freq 0.0 * asr gate 0.01 0.01",
        &registry,
    )
    .unwrap();

    let mut engine = make_engine(64);
    let bus = engine.graph_mut().add_node(Box::new(ugens::Bus::new(8)));
    engine.graph_mut().set_sink(bus);

    let v1 = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    let v2 = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    engine.prepare();

    for _ in 0..10 {
        engine.render();
    }
    assert_eq!(engine.synths().len(), 2);

    engine.free_voice(v1);
    assert_eq!(engine.synths().len(), 1);
    assert!(engine.voice_synth(v2).is_some());
}
