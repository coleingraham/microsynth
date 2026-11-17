//! A voice started at a sample (`Engine::start_voice_at`) begins there: a
//! note written off the block grid is the note written on it, shifted.
//!
//! Every expected value is the output of the same def spawned at sample 0 of
//! a fresh engine, never a value taken from the scheduler or the engine's
//! own bookkeeping.

use microsynth::dsl;
use microsynth::*;

mod common;
use common::builtin_registry;

const SR: f32 = 44_100.0;
const BLOCKS: [usize; 2] = [64, 128];

/// Defs whose every part is gate-free or runs from the spawn: an exponential
/// percussion envelope, a noise burst into a feedback delay, free-running
/// oscillators and an LFO.
const DEFS: [&str; 4] = [
    "synthdef perc freq=330.0 gate=1.0 = sinOsc freq 0.0 * expPerc 0.002 0.08 * 0.5",
    "synthdef pluck freq=220.0 gate=1.0 = \
     let burst = whiteNoise * perc 0.001 0.005; feedbackDelay burst (1.0 / freq) 0.95 * 0.5",
    "synthdef pad freq=262.0 gate=1.0 = \
     let env = adsr gate 0.01 0.05 0.6 0.05; \
     saw freq * env * (sinOsc 5.0 0.0 * 0.25 + 0.75) * 0.3",
    "synthdef sweep freq=440.0 gate=1.0 = \
     let f = xLine 2000.0 200.0 0.1; sinOsc f 0.0 * line 1.0 0.0 0.2 * 0.4",
];

fn rig(block: usize) -> (Engine, NodeId) {
    let mut engine = Engine::new(EngineConfig {
        sample_rate: SR,
        block_size: block,
    });
    let bus = engine
        .graph_mut()
        .add_node(Box::new(ugens::Bus::new(ugens::ChannelCount::Stereo)));
    engine.graph_mut().set_sink(bus);
    engine.prepare();
    (engine, bus)
}

fn def(src: &str) -> SynthDef {
    dsl::compile(src, &builtin_registry())
        .expect("def compiles")
        .remove(0)
}

fn run(engine: &mut Engine, blocks: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for _ in 0..blocks {
        let buf = engine.render().expect("bus is the sink");
        out.extend_from_slice(buf.channel(0).samples());
    }
    out
}

/// One note of `src`, gate scheduled at `on`, started at `on` (or not, when
/// `deferred` is false), rendered for `blocks` blocks.
fn note(src: &str, block: usize, on: u64, deferred: bool, blocks: usize) -> Vec<f32> {
    let (mut engine, bus) = rig(block);
    let id = engine.spawn_voice_on_bus(&def(src), bus).unwrap();
    engine.prepare();
    if deferred {
        engine.start_voice_at(id, on);
    }
    engine.schedule_note_span(id, on, on + 3000);
    run(&mut engine, blocks)
}

#[test]
fn off_grid_note_is_the_on_grid_note_shifted() {
    for block in BLOCKS {
        let b = block as u64;
        for src in DEFS {
            let blocks = 80;
            let grid = note(src, block, 0, true, blocks);
            assert!(grid.iter().any(|&x| x != 0.0), "{src} is silent");
            for off in [1, b / 2 + 3, b - 1, b + 17, 3 * b + 94] {
                let moved = note(src, block, off, true, blocks);
                let off = off as usize;
                assert!(
                    moved[..off].iter().all(|&x| x == 0.0),
                    "{src}: sound before the start at offset {off}, block {block}"
                );
                let worst = (0..moved.len() - off)
                    .map(|i| (moved[i + off] - grid[i]).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst <= 1e-6,
                    "{src}: offset {off}, block {block}: residual {worst}"
                );
            }
        }
    }
}

#[test]
fn start_at_or_before_the_block_is_the_plain_spawn() {
    for block in BLOCKS {
        for src in DEFS {
            let plain = note(src, block, 0, false, 40);
            let at_zero = note(src, block, 0, true, 40);
            assert_eq!(plain, at_zero, "{src}: start 0 differs from plain spawn");
        }
    }
}

#[test]
fn a_started_voice_cannot_be_started_again() {
    let (mut engine, bus) = rig(128);
    let id = engine.spawn_voice_on_bus(&def(DEFS[0]), bus).unwrap();
    engine.prepare();
    assert!(!engine.start_voice_at(id, 0), "start in the past");
    assert!(engine.start_voice_at(id, 300));
    assert!(!engine.start_voice_at(id, 400), "already deferred");
    assert!(!engine.start_voice_at(VoiceId(999), 300), "unknown voice");
    run(&mut engine, 1);
    assert!(!engine.start_voice_at(id, 900), "already rendered");
}

#[test]
fn params_set_before_the_start_land_and_later_events_keep_their_sample() {
    let src = "synthdef dc amp=0.0 gate=1.0 = amp";
    let (mut engine, bus) = rig(128);
    let id = engine.spawn_voice_on_bus(&def(src), bus).unwrap();
    engine.prepare();
    assert!(engine.set_voice_param(id, "amp", 0.25));
    assert!(engine.start_voice_at(id, 200));
    engine
        .scheduler_mut()
        .schedule_set_param(300, id, "amp", 0.5);
    let out = run(&mut engine, 4);
    assert!(out[..200].iter().all(|&x| x == 0.0));
    assert!(
        out[200..300].iter().all(|&x| x == 0.25),
        "{:?}",
        &out[195..205]
    );
    assert!(out[300..512].iter().all(|&x| x == 0.5));
}

#[test]
fn a_voice_waiting_for_its_start_is_not_freed_as_done() {
    let (mut engine, bus) = rig(128);
    let id = engine.spawn_voice_on_bus(&def(DEFS[0]), bus).unwrap();
    engine.prepare();
    assert!(engine.start_voice_at(id, 5 * 128 + 9));
    for _ in 0..8 {
        assert_eq!(engine.free_done_synths(), 0);
        engine.prepare();
        run(&mut engine, 1);
    }
    assert!(engine.voice_synth(id).is_some());
}

#[test]
fn two_voices_start_at_their_own_samples() {
    let (mut engine, bus) = rig(128);
    let d = def("synthdef dc amp=0.25 gate=1.0 = amp");
    let a = engine.spawn_voice_on_bus(&d, bus).unwrap();
    let b = engine.spawn_voice_on_bus(&d, bus).unwrap();
    engine.prepare();
    assert!(engine.start_voice_at(a, 10));
    assert!(engine.start_voice_at(b, 200));
    let out = run(&mut engine, 3);
    assert!(out[..10].iter().all(|&x| x == 0.0));
    assert!(out[10..200].iter().all(|&x| x == 0.25));
    assert!(out[200..384].iter().all(|&x| x == 0.5));
}
