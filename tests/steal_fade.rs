//! A stolen voice fades out over 5 ms before it is freed: no click when a
//! long tail (a harp, a vibraphone) is stolen mid-note.

use microsynth::dsl;
use microsynth::*;

mod common;
use common::builtin_registry;

const SR: f32 = 44100.0;
const BLOCK: usize = 128;

fn engine_with_bus() -> (Engine, NodeId) {
    let mut engine = Engine::new(EngineConfig {
        sample_rate: SR,
        block_size: BLOCK,
    });
    let bus = engine
        .graph_mut()
        .add_node(Box::new(ugens::Bus::new(ugens::ChannelCount::Stereo)));
    engine.graph_mut().set_sink(bus);
    engine.prepare();
    (engine, bus)
}

/// Render `blocks` blocks the way a host does (free finished voices, then
/// render), returning channel 0.
fn run(engine: &mut Engine, blocks: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for _ in 0..blocks {
        if engine.free_done_synths() > 0 {
            engine.prepare();
        }
        let buf = engine.render().expect("bus is the sink");
        out.extend_from_slice(buf.channel(0).samples());
    }
    out
}

fn max_jump(x: &[f32]) -> f32 {
    x.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f32::max)
}

#[test]
fn fade_length_is_5ms_in_whole_samples() {
    assert_eq!(ugens::steal_fade_samples(44100.0), 221);
    assert_eq!(ugens::steal_fade_samples(48000.0), 240);
    assert_eq!(ugens::steal_fade_samples(96000.0), 480);
    let f = ugens::StealFade::new(221);
    assert_eq!(f.gain(0), 1.0);
    assert!((f.gain(110) - 0.5).abs() < 0.01);
    assert!(f.gain(220) > 0.0 && f.gain(220) < 1e-3);
    assert_eq!(f.gain(221), 0.0);
    // steepest step: pi / (2 * len) of the level
    let steepest = (0..221)
        .map(|k| f.gain(k) - f.gain(k + 1))
        .fold(0.0, f32::max);
    assert!(steepest <= std::f32::consts::PI / (2.0 * 221.0) + 1e-6);
}

#[test]
fn stolen_sustained_sine_fades_without_a_click() {
    // A 220 Hz sine at 0.5 held indefinitely, stolen mid-note by a silent
    // voice. A sine's own largest step is 0.5 * 2pi * 220 / 44100 = 0.0157;
    // a cut drops up to 0.5 in one sample. The fade adds at most
    // 0.5 * pi / 442 = 0.0036.
    let registry = builtin_registry();
    let sine = dsl::compile("synthdef s freq=220.0 = sinOsc freq 0.0 * 0.5", &registry)
        .unwrap()
        .remove(0);
    let silent = dsl::compile("synthdef z = 0.0", &registry)
        .unwrap()
        .remove(0);
    let (mut engine, bus) = engine_with_bus();
    engine.set_voice_allocator(VoiceAllocator::new(1, StealPolicy::Oldest));
    let victim = engine.spawn_voice_on_bus_managed(&sine, bus).unwrap();
    engine.prepare();
    // 10.3 blocks in: the sine is mid-cycle and far from a zero crossing at
    // the block start where the steal lands.
    let mut out = run(&mut engine, 10);
    let thief = engine.spawn_voice_on_bus_managed(&silent, bus).unwrap();
    engine.prepare();
    let at = out.len();
    out.extend(run(&mut engine, 6));

    let natural = 0.5 * 2.0 * std::f32::consts::PI * 220.0 / SR;
    let jump = max_jump(&out);
    assert!(
        jump <= natural + 0.5 * std::f32::consts::PI / 442.0 + 1e-4,
        "max step {jump} (sine alone {natural})"
    );
    // the level at the steal was far from zero, so a cut would have clicked
    assert!(out[at - 1].abs() > 0.1, "steal landed at {}", out[at - 1]);
    // the tail is gone after the fade, and only after it
    assert!(out[at..at + 100].iter().any(|x| x.abs() > 0.05));
    assert!(out[at + 221..].iter().all(|&x| x == 0.0));
    // the victim is freed once the fade is done; the thief plays on
    assert!(engine.voice_synth(victim).is_none());
    assert!(engine.voice_synth(thief).is_some());
    let rec = engine.steals();
    assert_eq!(rec.len(), 1);
    assert!(!rec[0].cut);
    assert_eq!(rec[0].sample, (10 * BLOCK) as u64);
    // RMS of the last 128-sample block: 0.64 of a cycle, so near but not
    // exactly 0.5 / sqrt 2
    let l = rec[0].level.unwrap();
    assert!(l > 0.25 && l < 0.5, "victim RMS {l}");
}

#[test]
fn a_cut_would_click_the_same_steal_unfaded() {
    // The control for the test above: the same sine freed directly.
    let registry = builtin_registry();
    let sine = dsl::compile("synthdef s freq=220.0 = sinOsc freq 0.0 * 0.5", &registry)
        .unwrap()
        .remove(0);
    let (mut engine, bus) = engine_with_bus();
    let v = engine.spawn_voice_on_bus(&sine, bus).unwrap();
    engine.prepare();
    let mut out = run(&mut engine, 10);
    engine.free_voice(v);
    engine.prepare();
    out.extend(run(&mut engine, 2));
    assert!(max_jump(&out) > 0.1);
}

#[test]
fn victim_keeps_its_slot_while_fading_and_frees_it_after() {
    let registry = builtin_registry();
    let dc = dsl::compile("synthdef dc = 0.25", &registry)
        .unwrap()
        .remove(0);
    let (mut engine, bus) = engine_with_bus();
    engine.set_voice_allocator(VoiceAllocator::new(1, StealPolicy::Oldest));
    let a = engine.spawn_voice_on_bus_managed(&dc, bus).unwrap();
    engine.prepare();
    run(&mut engine, 2);
    let b = engine.spawn_voice_on_bus_managed(&dc, bus).unwrap();
    engine.prepare();
    assert!(engine.voice_synth(a).is_some(), "still fading");
    assert_eq!(engine.voice_allocator().unwrap().active_voices(), &[b]);
    assert_eq!(engine.graph().used_input_slots(bus).count(), 2);
    let out = run(&mut engine, 3);
    // 221 samples of fade summed with b's 0.25, then b alone
    assert!((out[0] - 0.5).abs() < 1e-6);
    assert!((out[300] - 0.25).abs() < 1e-6);
    assert!(engine.voice_synth(a).is_none(), "freed after the fade");
    assert_eq!(engine.graph().used_input_slots(bus).count(), 1);
}

#[test]
fn panned_victim_fades_through_its_pan() {
    let registry = builtin_registry();
    let dc = dsl::compile("synthdef dc = 0.25", &registry)
        .unwrap()
        .remove(0);
    let (mut engine, bus) = engine_with_bus();
    engine.set_voice_allocator(VoiceAllocator::new(1, StealPolicy::Oldest));
    let a = engine
        .spawn_voice_on_bus_panned_managed(&dc, bus, -1.0)
        .unwrap();
    engine.prepare();
    let before = run(&mut engine, 2);
    assert!(before[0] > 0.2);
    let _b = engine
        .spawn_voice_on_bus_panned_managed(&dc, bus, 1.0)
        .unwrap();
    engine.prepare();
    let mut left = Vec::new();
    for _ in 0..4 {
        if engine.free_done_synths() > 0 {
            engine.prepare();
        }
        left.extend_from_slice(engine.render().unwrap().channel(0).samples());
    }
    assert!(
        max_jump(&left) < 0.01,
        "left channel fades, max step {}",
        max_jump(&left)
    );
    assert!(left[300].abs() < 1e-6);
    assert!(engine.voice_synth(a).is_none());
}

#[test]
fn full_bus_cuts_the_earliest_fading_voice_first() {
    let registry = builtin_registry();
    let dc = dsl::compile("synthdef dc = 0.001", &registry)
        .unwrap()
        .remove(0);
    let (mut engine, bus) = engine_with_bus();
    let slots = ugens::MAX_BUS_INPUTS;
    // a budget equal to the slot count: every steal finds the bus full
    engine.set_voice_allocator(VoiceAllocator::new(slots, StealPolicy::Oldest));
    let first: Vec<_> = (0..slots)
        .map(|_| engine.spawn_voice_on_bus_managed(&dc, bus).unwrap())
        .collect();
    engine.prepare();
    run(&mut engine, 1);
    // steal: first[0] starts fading and keeps its slot, so the bus is full
    // and the fading voice is cut to make room
    let extra = engine.spawn_voice_on_bus_managed(&dc, bus);
    assert!(extra.is_some(), "a managed spawn makes room on a full bus");
    assert!(engine.voice_synth(first[0]).is_none());
    let rec = engine.steals();
    assert_eq!(rec.len(), 1);
    assert!(rec[0].cut, "freed before its fade ended");
}

#[test]
fn full_bus_of_unmanaged_voices_still_refuses() {
    let registry = builtin_registry();
    let dc = dsl::compile("synthdef dc = 0.001", &registry)
        .unwrap()
        .remove(0);
    let (mut engine, bus) = engine_with_bus();
    for _ in 0..ugens::MAX_BUS_INPUTS {
        engine.spawn_voice_on_bus(&dc, bus).unwrap();
    }
    engine.set_voice_allocator(VoiceAllocator::new(1000, StealPolicy::Oldest));
    assert!(engine.spawn_voice_on_bus_managed(&dc, bus).is_none());
    assert!(engine.steals().is_empty());
}

#[test]
fn reject_never_steals_or_fades() {
    let registry = builtin_registry();
    let dc = dsl::compile("synthdef dc = 0.25", &registry)
        .unwrap()
        .remove(0);
    let (mut engine, bus) = engine_with_bus();
    engine.set_voice_allocator(VoiceAllocator::new(1, StealPolicy::Reject));
    engine.spawn_voice_on_bus_managed(&dc, bus).unwrap();
    assert!(engine.spawn_voice_on_bus_managed(&dc, bus).is_none());
    assert!(engine.steals().is_empty());
}
