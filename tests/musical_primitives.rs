use microsynth::musical_time::*;
use microsynth::tuning::*;

// ============================================================================
// Tuning tests
// ============================================================================

#[test]
fn test_midi_to_hz_12tet_a4() {
    let hz = midi_to_hz_12tet(69.0, 440.0);
    assert!((hz - 440.0).abs() < 0.01, "A4 should be 440 Hz, got {hz}");
}

#[test]
fn test_midi_to_hz_12tet_middle_c() {
    let hz = midi_to_hz_12tet(60.0, 440.0);
    assert!(
        (hz - 261.626).abs() < 0.01,
        "C4 should be ~261.63 Hz, got {hz}"
    );
}

#[test]
fn test_midi_to_hz_12tet_a5() {
    let hz = midi_to_hz_12tet(81.0, 440.0);
    assert!((hz - 880.0).abs() < 0.01, "A5 should be 880 Hz, got {hz}");
}

#[test]
fn test_midi_to_hz_12tet_custom_a4() {
    // A4 = 432 Hz tuning
    let hz = midi_to_hz_12tet(69.0, 432.0);
    assert!(
        (hz - 432.0).abs() < 0.01,
        "A4 at 432 should be 432 Hz, got {hz}"
    );
}

#[test]
fn test_hz_to_midi_roundtrip() {
    for note in [48.0, 60.0, 69.0, 72.0, 84.0] {
        let hz = midi_to_hz_12tet(note, 440.0);
        let back = hz_to_midi_12tet(hz, 440.0);
        assert!(
            (back - note).abs() < 0.01,
            "roundtrip failed for note {note}: got {back}"
        );
    }
}

#[test]
fn test_apply_cents_semitone_up() {
    let hz = apply_cents(440.0, 100.0);
    let expected = midi_to_hz_12tet(70.0, 440.0);
    assert!(
        (hz - expected).abs() < 0.1,
        "100 cents up from 440 should be ~{expected}, got {hz}"
    );
}

#[test]
fn test_apply_cents_semitone_down() {
    let hz = apply_cents(440.0, -100.0);
    let expected = midi_to_hz_12tet(68.0, 440.0);
    assert!(
        (hz - expected).abs() < 0.1,
        "100 cents down from 440 should be ~{expected}, got {hz}"
    );
}

#[test]
fn test_apply_cents_quarter_tone() {
    let hz = apply_cents(440.0, -50.0);
    // Quarter-tone flat: between A4 and Ab4
    assert!(
        hz > 415.0 && hz < 440.0,
        "quarter-tone flat should be between Ab4 and A4, got {hz}"
    );
}

#[test]
fn test_apply_cents_zero() {
    let hz = apply_cents(440.0, 0.0);
    assert!(
        (hz - 440.0).abs() < 0.01,
        "0 cents should not change frequency, got {hz}"
    );
}

// -- TuningTable tests -------------------------------------------------------

#[test]
fn test_tuning_table_12tet_matches_free_function() {
    let table = TuningTable::equal_temperament_12();
    for note in [48, 60, 69, 72, 84] {
        let table_hz = table.note_to_hz(note as f32);
        let free_hz = midi_to_hz_12tet(note as f32, 440.0);
        assert!(
            (table_hz - free_hz).abs() < 0.01,
            "table and free function disagree for note {note}: {table_hz} vs {free_hz}"
        );
    }
}

#[test]
fn test_tuning_table_24tet() {
    let table = TuningTable::equal_temperament_24();
    assert_eq!(table.divisions(), 24);
    // Anchor should produce 440
    let hz = table.note_to_hz(138.0); // A4 in 24-TET space
    assert!(
        (hz - 440.0).abs() < 0.01,
        "anchor should be 440 Hz, got {hz}"
    );
}

#[test]
fn test_tuning_table_just_intonation_perfect_fifth() {
    let table = TuningTable::just_intonation();
    // In JI, a perfect fifth (7 semitones above anchor) = 3/2 ratio
    let anchor_hz = table.note_to_hz(69.0);
    let fifth_hz = table.note_to_hz(76.0); // A4 + 7 = E5
    let ratio = fifth_hz / anchor_hz;
    assert!(
        (ratio - 1.5).abs() < 0.001,
        "JI perfect fifth should be 3/2, got ratio {ratio}"
    );
}

#[test]
fn test_tuning_table_just_intonation_major_third() {
    let table = TuningTable::just_intonation();
    let anchor_hz = table.note_to_hz(69.0);
    let third_hz = table.note_to_hz(73.0); // 4 semitones = major third = 5/4
    let ratio = third_hz / anchor_hz;
    assert!(
        (ratio - 1.25).abs() < 0.001,
        "JI major third should be 5/4, got ratio {ratio}"
    );
}

#[test]
fn test_tuning_table_from_cents_slendro() {
    // Approximate Javanese Slendro: 5 notes per octave
    let cents = [0.0, 240.0, 480.0, 720.0, 960.0];
    let table = TuningTable::from_cents(&cents, 60, 261.63);
    assert_eq!(table.divisions(), 5);
    // Anchor should produce 261.63
    let hz = table.note_to_hz(60.0);
    assert!(
        (hz - 261.63).abs() < 0.1,
        "anchor should be 261.63, got {hz}"
    );
    // One octave up
    let hz_oct = table.note_to_hz(65.0); // 60 + 5 divisions
    assert!(
        (hz_oct - 523.26).abs() < 0.5,
        "octave should be ~523.26, got {hz_oct}"
    );
}

#[test]
fn test_tuning_table_hz_to_note_roundtrip() {
    let table = TuningTable::equal_temperament_12();
    for note in [48, 60, 69, 72] {
        let hz = table.note_to_hz(note as f32);
        let back = table.hz_to_note(hz);
        assert!(
            (back - note as f32).abs() < 0.5,
            "roundtrip failed for note {note}: got {back}"
        );
    }
}

#[test]
fn test_tuning_table_below_anchor() {
    let table = TuningTable::equal_temperament_12();
    // Note 57 = A3 = 220 Hz
    let hz = table.note_to_hz(57.0);
    assert!((hz - 220.0).abs() < 0.1, "A3 should be 220 Hz, got {hz}");
}

// ============================================================================
// Musical time tests
// ============================================================================

#[test]
fn test_bar_duration_4_4_120bpm() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    let bar_secs = tc.bar_duration_secs();
    assert!(
        (bar_secs - 2.0).abs() < 1e-9,
        "4/4 at 120 BPM bar should be 2.0s, got {bar_secs}"
    );
}

#[test]
fn test_step_duration_16th_at_120bpm() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    let step_secs = tc.step_duration_secs();
    // 16 steps per 2-second bar = 0.125s per step
    assert!(
        (step_secs - 0.125).abs() < 1e-9,
        "16th note at 120 BPM should be 0.125s, got {step_secs}"
    );
}

#[test]
fn test_bar_duration_3_4() {
    let tc = TimeConfig {
        bpm: 120.0,
        numerator: 3,
        denominator: 4,
        grid_steps: 12,
        ppqn: 96,
        sample_rate: 44100.0,
    };
    let bar_secs = tc.bar_duration_secs();
    // 3/4 at 120 BPM: 3 quarter notes * 0.5s = 1.5s
    assert!(
        (bar_secs - 1.5).abs() < 1e-9,
        "3/4 at 120 BPM bar should be 1.5s, got {bar_secs}"
    );
}

#[test]
fn test_bar_duration_7_8() {
    let tc = TimeConfig {
        bpm: 120.0,
        numerator: 7,
        denominator: 8,
        grid_steps: 14,
        ppqn: 96,
        sample_rate: 44100.0,
    };
    let bar_secs = tc.bar_duration_secs();
    // 7/8 at 120 BPM: 7 eighth notes, each 0.25s = 1.75s
    assert!(
        (bar_secs - 1.75).abs() < 1e-9,
        "7/8 at 120 BPM bar should be 1.75s, got {bar_secs}"
    );
}

#[test]
fn test_position_to_samples_origin() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    let pos = MusicalPosition::new(0, 0, 0);
    assert_eq!(tc.position_to_samples(pos), 0);
}

#[test]
fn test_position_to_samples_bar_1() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    let pos = MusicalPosition::new(1, 0, 0);
    let expected = (44100.0 * 2.0) as u64; // 2 seconds at 44100 Hz
    assert_eq!(tc.position_to_samples(pos), expected);
}

#[test]
fn test_position_to_samples_step() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    let pos = MusicalPosition::new(0, 4, 0);
    // Step 4 = beat 2 (one quarter note in). 0.5s * 44100 = 22050
    let samples = tc.position_to_samples(pos);
    assert_eq!(samples, 22050);
}

#[test]
fn test_ticks_per_step() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    // 4/4, grid_steps=16, ppqn=96
    // 4 quarter notes per bar, 16 steps per bar → 4 steps per quarter
    // 96 ppqn / 4 steps per quarter = 24 ticks per step
    assert_eq!(tc.ticks_per_step(), 24);
}

#[test]
fn test_position_with_positive_tick_offset() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    let base = tc.position_to_samples(MusicalPosition::new(0, 0, 0));
    let offset = tc.position_to_samples(MusicalPosition::new(0, 0, 15));
    assert!(
        offset > base,
        "positive tick offset should produce later time"
    );
}

#[test]
fn test_position_with_negative_tick_offset() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    let base = tc.position_to_samples(MusicalPosition::new(0, 4, 0));
    let early = tc.position_to_samples(MusicalPosition::new(0, 4, -10));
    assert!(
        early < base,
        "negative tick offset should produce earlier time"
    );
}

#[test]
fn test_negative_tick_at_bar_zero_clamps() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    // Negative offset at the very start should clamp to 0
    let pos = MusicalPosition::new(0, 0, -100);
    assert_eq!(tc.position_to_samples(pos), 0);
}

#[test]
fn test_steps_to_samples() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    // 4 steps = 1 beat = 0.5s = 22050 samples
    let samples = tc.steps_to_samples(4.0);
    assert_eq!(samples, 22050);
}

#[test]
fn test_steps_to_secs() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    let secs = tc.steps_to_secs(4.0);
    assert!(
        (secs - 0.5).abs() < 1e-9,
        "4 steps should be 0.5s, got {secs}"
    );
}

#[test]
fn test_step_duration_samples() {
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    let step_samples = tc.step_duration_samples();
    // 0.125s * 44100 = 5512.5
    assert!(
        (step_samples - 5512.5).abs() < 0.01,
        "step should be 5512.5 samples, got {step_samples}"
    );
}

// ============================================================================
// Scheduling integration tests
// ============================================================================

#[test]
fn test_schedule_note_creates_gate_events() {
    let mut scheduler = microsynth::Scheduler::new();
    let voice = scheduler.alloc_voice_id();
    scheduler.schedule_note(voice, 1000, 5000);

    // Should have 2 events
    assert_eq!(scheduler.len(), 2);

    // Drain all
    let events = scheduler.drain_before(10000);
    assert_eq!(events.len(), 2);

    // First: gate on at 1000
    assert_eq!(events[0].time, 1000);
    match &events[0].action {
        microsynth::EventAction::SetGate { value, .. } => {
            assert!(*value > 0.0, "first event should be gate on");
        }
        _ => panic!("expected SetGate"),
    }

    // Second: gate off at 5000
    assert_eq!(events[1].time, 5000);
    match &events[1].action {
        microsynth::EventAction::SetGate { value, .. } => {
            assert!(*value == 0.0, "second event should be gate off");
        }
        _ => panic!("expected SetGate"),
    }
}

#[test]
fn test_schedule_note_aligned_pre_trigger() {
    let config = microsynth::EngineConfig {
        sample_rate: 44100.0,
        block_size: 64,
    };
    let mut engine = microsynth::Engine::new(config);

    // We need a SynthDef to spawn a voice — use a minimal one
    let mut registry = microsynth::dsl::UGenRegistry::new();
    microsynth::ugens::register_builtins(&mut registry);
    let defs = microsynth::dsl::compile(
        "synthdef test freq=440.0 amp=0.5 gate=1.0 = let env = asr gate 0.1 0.3\nsinOsc freq 0.0 * amp * env",
        &registry,
    ).unwrap();

    // Set up bus + voice
    let bus = microsynth::ugens::Bus::new(microsynth::ugens::ChannelCount::Stereo);
    let bus_id = engine.graph_mut().add_node(alloc::boxed::Box::new(bus));
    engine.graph_mut().set_sink(bus_id);
    engine.prepare();

    let voice_id = engine.spawn_voice_on_bus(&defs[0], bus_id).unwrap();
    engine.prepare();

    // Schedule with 0.1s attack alignment at grid_time = 44100 (1 second)
    let grid_time: u64 = 44100;
    let attack_secs: f32 = 0.1;
    let duration: u64 = 22050;
    engine.schedule_note_aligned(voice_id, grid_time, attack_secs, duration);

    // The gate-on should be at grid_time - attack_samples
    let attack_samples = (0.1 * 44100.0) as u64; // 4410
    let expected_on = grid_time - attack_samples;
    let expected_off = grid_time + duration;

    // Drain all events
    let events = engine.scheduler_mut().drain_before(100000);
    assert_eq!(events.len(), 2, "should have gate-on and gate-off");
    assert_eq!(
        events[0].time, expected_on,
        "gate-on should be pre-triggered"
    );
    assert_eq!(
        events[1].time, expected_off,
        "gate-off should be at grid_time + duration"
    );
}

// ============================================================================
// Musical-time sequence scheduling tests
// ============================================================================

use microsynth::{GlideShape, GlideSpace, MusicalGlideSegment, schedule_musical_glides};

#[test]
fn test_sequence_to_samples_constructs_no_engine() {
    // The conversion is pure: a TimeConfig and a slice of segments in, a
    // Vec of sample-time events out, no Scheduler or Engine involved.
    let tc = TimeConfig::new_4_4(120.0, 44100.0);
    let segments = [MusicalGlideSegment::new(
        MusicalPosition::new(0, 4, 0),
        880.0,
        2.0,
        GlideShape::Linear,
    )];

    let events = tc.sequence_to_samples(&segments);

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].time, 22050); // 1 beat in at 120 BPM, 44.1kHz
    assert_eq!(events[0].target, 880.0);
    assert!((events[0].glide_secs - 0.25).abs() < 1e-6); // 2 steps = 1/8 note = 0.25s
    assert_eq!(events[0].shape, GlideShape::Linear);
}

#[test]
fn test_sequence_to_samples_is_deterministic() {
    let tc = TimeConfig::new_4_4(133.0, 48000.0);
    let segments = [
        MusicalGlideSegment::new(MusicalPosition::new(0, 0, 0), 220.0, 1.0, GlideShape::Hold),
        MusicalGlideSegment::new(
            MusicalPosition::new(1, 3, 5),
            440.0,
            3.5,
            GlideShape::Exponential(2.0),
        ),
    ];

    let first = tc.sequence_to_samples(&segments);
    let second = tc.sequence_to_samples(&segments);

    assert_eq!(
        first, second,
        "identical inputs must produce identical output"
    );
}

#[test]
fn test_sequence_to_samples_scales_with_tempo() {
    // Glide length is denominated in grid steps, not seconds, so doubling
    // the tempo must halve both the event's position and its glide length
    // in absolute (sample/second) terms.
    let segments = [MusicalGlideSegment::new(
        MusicalPosition::new(0, 4, 0),
        660.0,
        2.0,
        GlideShape::Sine,
    )];

    let base = TimeConfig::new_4_4(120.0, 44100.0);
    let double_tempo = TimeConfig::new_4_4(240.0, 44100.0);

    let base_events = base.sequence_to_samples(&segments);
    let fast_events = double_tempo.sequence_to_samples(&segments);

    assert_eq!(base_events[0].time, fast_events[0].time * 2);
    assert!((base_events[0].glide_secs - fast_events[0].glide_secs * 2.0).abs() < 1e-6);
}

#[test]
fn test_schedule_musical_glides_matches_pure_conversion() {
    let config = TimeConfig::new_4_4(100.0, 48000.0);
    let segments = [
        MusicalGlideSegment::new(MusicalPosition::new(0, 0, 0), 440.0, 4.0, GlideShape::Sine),
        MusicalGlideSegment::new(MusicalPosition::new(0, 8, 0), 880.0, 2.0, GlideShape::Hold)
            .with_space(GlideSpace::Pitch),
    ];
    let expected = config.sequence_to_samples(&segments);

    let mut scheduler = microsynth::Scheduler::new();
    let voice = scheduler.alloc_voice_id();
    schedule_musical_glides(&mut scheduler, &config, voice, "freq", &segments);

    let events = scheduler.drain_before(u64::MAX);
    assert_eq!(events.len(), expected.len());
    for (event, want) in events.iter().zip(expected.iter()) {
        assert_eq!(event.time, want.time);
        match &event.action {
            microsynth::EventAction::SetParamGlide {
                voice: event_voice,
                param,
                target,
                glide_secs,
                shape,
                space,
            } => {
                assert_eq!(*event_voice, voice);
                assert_eq!(param, "freq");
                assert_eq!(*target, want.target);
                assert_eq!(*glide_secs, want.glide_secs);
                assert_eq!(*shape, want.shape);
                assert_eq!(
                    *space, want.space,
                    "each segment's own space must be preserved"
                );
            }
            _ => panic!("expected SetParamGlide"),
        }
    }
}

#[test]
fn test_musical_glide_segment_defaults_to_raw_space() {
    let segment = MusicalGlideSegment::new(
        MusicalPosition::new(0, 0, 0),
        440.0,
        1.0,
        GlideShape::Linear,
    );
    assert_eq!(segment.space, GlideSpace::default());
    assert_eq!(segment.space, GlideSpace::Raw);
}

#[test]
fn test_schedule_musical_glides_preserves_pitch_space() {
    // A pitch segment scheduled in raw space would glide linear-in-Hz
    // instead of as an equal-ratio pitch sweep, so the space a caller sets
    // on a segment must survive both the pure conversion and scheduling.
    let config = TimeConfig::new_4_4(120.0, 44100.0);
    let segments = [MusicalGlideSegment::new(
        MusicalPosition::new(0, 0, 0),
        880.0,
        4.0,
        GlideShape::Exponential(-2.0),
    )
    .with_space(GlideSpace::Pitch)];

    let converted = config.sequence_to_samples(&segments);
    assert_eq!(converted[0].space, GlideSpace::Pitch);

    let mut scheduler = microsynth::Scheduler::new();
    let voice = scheduler.alloc_voice_id();
    schedule_musical_glides(&mut scheduler, &config, voice, "freq", &segments);

    let events = scheduler.drain_before(u64::MAX);
    assert_eq!(events.len(), 1);
    match &events[0].action {
        microsynth::EventAction::SetParamGlide { space, .. } => {
            assert_eq!(
                *space,
                GlideSpace::Pitch,
                "scheduled event must carry the segment's glide space"
            );
        }
        _ => panic!("expected SetParamGlide"),
    }
}

#[test]
fn test_musical_glide_sub_block_boundary_lands_in_correct_block() {
    // 16th-note grid at 120 BPM / 44.1kHz puts step 1 at 5512.5 samples,
    // which rounds to 5513 — strictly inside the block [5504, 5632) for a
    // 128-sample block size, not on either boundary. The scheduler dispatches
    // an event scheduled for sample N at the start of the block containing
    // N (see `crate::scheduler`), which the engine implements as draining
    // with `deadline = block_start + block_size`. Confirm this event only
    // surfaces once that block's deadline is reached, not the prior one.
    const BLOCK_SIZE: u64 = 128;

    let config = TimeConfig::new_4_4(120.0, 44100.0);
    let segments = [MusicalGlideSegment::new(
        MusicalPosition::new(0, 1, 0),
        1.0,
        0.5,
        GlideShape::Linear,
    )];
    let events = config.sequence_to_samples(&segments);
    let time = events[0].time;
    assert_eq!(time, 5513);

    let block_start = (time / BLOCK_SIZE) * BLOCK_SIZE;
    assert_eq!(block_start, 5504);

    let mut scheduler = microsynth::Scheduler::new();
    let voice = scheduler.alloc_voice_id();
    schedule_musical_glides(&mut scheduler, &config, voice, "freq", &segments);

    // Draining up through the start of the containing block must not yet
    // yield the event...
    let too_early = scheduler.drain_before(block_start);
    assert_eq!(
        too_early.len(),
        0,
        "event must not surface before its block"
    );

    // ...but draining through that block's deadline must.
    let mut scheduler = microsynth::Scheduler::new();
    schedule_musical_glides(&mut scheduler, &config, voice, "freq", &segments);
    let on_time = scheduler.drain_before(block_start + BLOCK_SIZE);
    assert_eq!(on_time.len(), 1, "event must surface within its own block");
}

#[test]
fn test_arrive_aligned_glide_completes_at_target_position() {
    // MusicalGlideSegment::position is depart-aligned: it's where the glide
    // starts, not where it ends. A caller wanting arrive-alignment (the
    // glide completing AT a musical position) resolves that itself by
    // building the segment at `arrival_position - glide_steps`.
    let config = TimeConfig::new_4_4(120.0, 44100.0);
    let glide_steps = 4.0;
    let arrival = MusicalPosition::new(0, 8, 0);

    let start = MusicalPosition::new(0, arrival.step - glide_steps as u16, 0);
    let segment = MusicalGlideSegment::new(start, 880.0, glide_steps, GlideShape::Linear);

    let events = config.sequence_to_samples(&[segment]);
    let glide_samples = (events[0].glide_secs as f64 * config.sample_rate as f64).round() as u64;
    let completion_time = events[0].time + glide_samples;

    assert_eq!(
        completion_time,
        config.position_to_samples(arrival),
        "a glide built with arrival - glide_steps as its position must complete at the arrival position"
    );
}

extern crate alloc;
