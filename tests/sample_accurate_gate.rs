//! A gate, a parameter step and a preroll land on their own sample, whatever
//! the voice and however it was spawned.
//!
//! Every expected value is a literal in this file or the output of the same
//! def played with its gate held open from sample 0 (no scheduling), never
//! a value taken from the scheduler or the engine.

#![cfg(feature = "ir")]

use microsynth::dsl;
use microsynth::ir::{IrEdge, IrNode, IrParam, IrSynthDef, IrTableBinding, SynthDefClass};
use microsynth::sampler::program::{Envelope, Region, SampleProgram};
use microsynth::sampler::{ProgramId, SamplePcm, SamplePcmBank, SampleProgramBank};
use microsynth::*;

mod common;
use common::builtin_registry;

const SR: f32 = 44_100.0;
const BLOCKS: [usize; 2] = [64, 128];
/// A preroll that is a multiple of neither block size.
const PREROLL: u64 = 37;
/// Notated onsets: on the grid, one off it either way, off the grid.
fn onsets(b: usize) -> [u64; 6] {
    let b = b as u64;
    [0, 1, b - 1, b, b + 1, 3 * b + 17]
}

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

/// Render `blocks` blocks, returning channel 0.
fn run(engine: &mut Engine, blocks: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for _ in 0..blocks {
        let buf = engine.render().expect("bus is the sink");
        out.extend_from_slice(buf.channel(0).samples());
    }
    out
}

fn first_nonzero(x: &[f32]) -> Option<usize> {
    x.iter().position(|&v| v != 0.0)
}

// --- the voices -----------------------------------------------------------

fn dc(level: f32, frames: usize) -> SamplePcm {
    SamplePcm::new("dc", [1; 32], SR, &[&vec![level; frames][..]]).unwrap()
}

/// A sample whose first `lead` frames are silent, then one full-scale frame.
fn lead_click(lead: usize) -> SamplePcm {
    let mut x = vec![0.0f32; lead + 1 + 400];
    x[lead] = 1.0;
    SamplePcm::new("click", [2; 32], SR, &[&x[..]]).unwrap()
}

struct Banks {
    _samples: SamplePcmBank,
    programs: SampleProgramBank,
    pid: ProgramId,
}

fn banks(pcm: SamplePcm) -> Banks {
    let prog = SampleProgram {
        name: "test".into(),
        spec_sha256: [0xAB; 32],
        samples: vec![*pcm.hash()],
        regions: vec![Region {
            sample: 0,
            amp_veltrack: 0.0,
            ampeg: Envelope {
                release: 0.001,
                ..Envelope::default()
            },
            ..Region::default()
        }],
    };
    let mut sb = SamplePcmBank::new();
    sb.register(pcm);
    let mut pb = SampleProgramBank::new();
    let pid = pb.register(prog, &sb).expect("program binds");
    Banks {
        _samples: sb,
        programs: pb,
        pid,
    }
}

/// The `sampleVoice` def with `freq`, `gate` (default 0) and `vel` ports.
fn sample_def(b: &Banks, name: &str) -> SynthDef {
    let ports = ["freq", "gate", "vel"];
    let defaults = [261.625_58, 0.0, 1.0];
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut params = Vec::new();
    for (i, (&p, &d)) in ports.iter().zip(&defaults).enumerate() {
        nodes.push(IrNode::Param {
            name: p.into(),
            default: d,
        });
        params.push(IrParam {
            name: p.into(),
            node: i,
            input: 0,
            default: d,
        });
        // sampleVoice's ports are freq, gate, vel, ...: the same order.
        edges.push(IrEdge {
            from: i,
            to: 3,
            to_input: i,
        });
    }
    nodes.push(IrNode::UGen {
        kind: "sampleVoice".into(),
        consts: vec![],
    });
    let ir = IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: name.into(),
        class: SynthDefClass::Source,
        output_channels: 2,
        nodes,
        edges,
        params,
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding {
            node: 3,
            table_id: b.pid.0,
        }],
        output_node: 3,
    };
    let mut reg = dsl::UGenRegistry::new();
    ugens::register_builtins(&mut reg);
    ugens::register_table_bound_builtins(&mut reg);
    ir.validate(&reg).expect("valid IR");
    ir.compile_with_banks(&reg, &CoeffTableBank::new(), Some(&b.programs))
        .expect("compiles")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Group {
    Plain,
    Registered,
    Sample,
}

struct Kind {
    group: Group,
    name: String,
    def: SynthDef,
}

/// The DSL for a registered UGen kind that has a `gate` port, wired so the
/// gate drives the output. A new such kind has no recipe and fails here, so
/// it cannot slip past the loop unchecked.
fn recipe(kind: &str) -> String {
    match kind {
        "adsr" => "synthdef k gate=0.0 = adsr gate 0.0001 0.0001 1.0 0.0001".into(),
        "asr" => "synthdef k gate=0.0 = asr gate 0.0001 0.0001".into(),
        // A rise plucks: the pluck waveform's first sample is the gate's.
        "guitarString" => "synthdef k gate=0.0 = guitarString 220.0 gate".into(),
        other => panic!("registered kind `{other}` has a gate port and no recipe here"),
    }
}

/// Every voice kind: a plain gated level, every registered UGen kind with a
/// `gate` port, and the sample-playback voice.
fn kinds(sample_banks: &Banks) -> Vec<Kind> {
    let reg = builtin_registry();
    let mut out = vec![Kind {
        group: Group::Plain,
        name: "plain".into(),
        def: dsl::compile("synthdef plain gate=0.0 = gate * 0.5", &reg)
            .unwrap()
            .remove(0),
    }];
    let mut names: Vec<&str> = reg
        .iter()
        .filter(|(_, e)| (e.factory)().spec().inputs.iter().any(|i| i.name == "gate"))
        .map(|(n, _)| n.as_str())
        .collect();
    names.sort_unstable();
    for n in names {
        out.push(Kind {
            group: Group::Registered,
            name: n.into(),
            def: dsl::compile(&recipe(n), &reg).unwrap().remove(0),
        });
    }
    out.push(Kind {
        group: Group::Sample,
        name: "sampleVoice".into(),
        def: sample_def(sample_banks, "sv"),
    });
    out
}

fn count(ks: &[Kind], g: Group) -> usize {
    ks.iter().filter(|k| k.group == g).count()
}

// --- how a voice is spawned ----------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Spawn {
    OnBus,
    /// Through `spawn_voice_on_bus_managed` under a budget.
    Managed,
    /// The voice a `LegatoVoice` holds, wired to the bus by hand.
    Legato,
}

const SPAWNS: [Spawn; 3] = [Spawn::OnBus, Spawn::Managed, Spawn::Legato];

fn spawn(engine: &mut Engine, bus: NodeId, def: &SynthDef, how: Spawn) -> VoiceId {
    let v = match how {
        Spawn::OnBus => engine.spawn_voice_on_bus(def, bus).unwrap(),
        Spawn::Managed => {
            engine.set_voice_budget(4, StealPolicy::Oldest);
            engine.spawn_voice_on_bus_managed(def, bus).unwrap()
        }
        Spawn::Legato => {
            let mut leg = LegatoVoice::new("gate", 0.0);
            let v = leg.note_on(engine, def, 1.0);
            leg.note_off(engine);
            engine.connect_voice_output(v, bus, 0);
            v
        }
    };
    engine.prepare();
    v
}

/// The first nonzero sample of `def` with its gate open from sample 0: how
/// late the voice itself answers a gate that is already high.
fn own_latency(def: &SynthDef, block: usize) -> usize {
    let (mut engine, bus) = rig(block);
    let v = spawn(&mut engine, bus, def, Spawn::OnBus);
    engine.set_voice_param(v, "gate", 1.0);
    let out = run(&mut engine, 4);
    first_nonzero(&out).expect("the voice sounds with its gate open")
}

/// Schedule one note at `notated` (preroll `preroll` on the def), render
/// past it, and return the output and the returned onsets.
fn play(
    k: &Kind,
    how: Spawn,
    block: usize,
    preroll: u64,
    notated: u64,
    dur: u64,
) -> (Vec<f32>, NoteOnsets) {
    let (mut engine, bus) = rig(block);
    engine.set_preroll(k.def.name(), preroll);
    let v = spawn(&mut engine, bus, &k.def, how);
    let rec = engine.schedule_note(v, notated, dur);
    let blocks = (notated + dur) as usize / block + 3;
    (run(&mut engine, blocks), rec)
}

// --- B1 -------------------------------------------------------------------

#[test]
fn gate_at_sample_n_opens_at_n() {
    let banks = banks(dc(0.5, 44_100));
    let ks = kinds(&banks);
    let plain = ks.iter().find(|k| k.group == Group::Plain).unwrap();
    for &b in &BLOCKS {
        for n in onsets(b) {
            let (out, _) = play(plain, Spawn::OnBus, b, 0, n, 4000);
            assert_eq!(first_nonzero(&out), Some(n as usize), "block {b}, N={n}");
        }
    }
}

#[test]
fn set_param_step_lands_on_its_sample() {
    let reg = builtin_registry();
    let def = dsl::compile("synthdef p level=0.0 = level", &reg)
        .unwrap()
        .remove(0);
    for &b in &BLOCKS {
        for n in onsets(b) {
            let (mut engine, bus) = rig(b);
            let v = spawn(&mut engine, bus, &def, Spawn::OnBus);
            engine
                .scheduler_mut()
                .schedule_set_param(n, v, "level", 0.7);
            let out = run(&mut engine, n as usize / b + 3);
            assert_eq!(first_nonzero(&out), Some(n as usize), "block {b}, N={n}");
            assert!(out[n as usize..].iter().all(|&x| x == 0.7), "holds 0.7");
        }
    }
}

#[test]
fn splitting_a_block_does_not_change_the_audio() {
    // Events that change nothing (a parameter set to the value it holds),
    // placed inside blocks, must leave a stateful voice bit-identical.
    let reg = builtin_registry();
    let def = dsl::compile(
        "synthdef s freq=220.0 level=0.5 = lpf (saw freq) 2000.0 0.7 * level",
        &reg,
    )
    .unwrap()
    .remove(0);
    for &b in &BLOCKS {
        let render = |events: &[u64]| {
            let (mut engine, bus) = rig(b);
            let v = spawn(&mut engine, bus, &def, Spawn::OnBus);
            for &t in events {
                engine
                    .scheduler_mut()
                    .schedule_set_param(t, v, "level", 0.5);
            }
            run(&mut engine, 6)
        };
        let whole = render(&[]);
        let split = render(&[3, 70, 129, 130, 200, 301, 383]);
        assert!(whole.iter().any(|&x| x.abs() > 0.01), "the voice sounds");
        assert_eq!(whole, split, "block {b}");
    }
}

// --- B2 -------------------------------------------------------------------

#[test]
fn gate_opens_at_n_for_every_voice_kind() {
    let banks = banks(dc(0.5, 44_100));
    let ks = kinds(&banks);
    assert!(count(&ks, Group::Plain) >= 1, "no plain voice visited");
    assert!(count(&ks, Group::Registered) >= 1, "no registered kind");
    assert!(count(&ks, Group::Sample) >= 1, "no sample voice");
    let mut visited = [0usize; 3];
    for k in &ks {
        for &b in &BLOCKS {
            let lat = own_latency(&k.def, b);
            assert!(lat <= 1, "{}: own latency {lat}", k.name);
            for how in SPAWNS {
                for n in onsets(b) {
                    let (out, _) = play(k, how, b, 0, n, 4000);
                    assert_eq!(
                        first_nonzero(&out),
                        Some(n as usize + lat),
                        "{} {how:?} block {b}, N={n}",
                        k.name
                    );
                    visited[how as usize] += 1;
                }
            }
        }
    }
    assert!(visited.iter().all(|&v| v > 0), "a spawn path was skipped");
}

#[test]
fn stolen_voice_starts_at_n_while_the_victim_fades() {
    let reg = builtin_registry();
    let def = dsl::compile("synthdef v gate=1.0 = gate * 0.5", &reg)
        .unwrap()
        .remove(0);
    for &b in &BLOCKS {
        for (n, preroll) in [(301u64, 0u64), (700, 0), (700, PREROLL)] {
            let (mut engine, bus) = rig(b);
            engine.set_voice_budget(1, StealPolicy::Oldest);
            engine.set_preroll("v", preroll);
            engine.spawn_voice_on_bus_managed(&def, bus).unwrap();
            let thief = engine.spawn_voice_on_bus_managed(&def, bus).unwrap();
            engine.set_voice_param(thief, "gate", 0.0);
            engine.prepare();
            assert_eq!(engine.steals().len(), 1, "the second spawn steals");
            engine.schedule_note(thief, n, 3000);
            let out = run(&mut engine, (n as usize + 3000) / b + 3);
            let at = (n - preroll) as usize;
            // The victim is heard, fading, from sample 0.
            assert!(out[..100].iter().any(|&x| x > 0.05), "victim fades");
            // It is gone after 221 samples (5 ms), before the thief starts.
            assert!(221 < at);
            assert!(out[221..at].iter().all(|&x| x == 0.0), "gap before thief");
            assert_eq!(out[at], 0.5, "block {b}, N={n}, preroll {preroll}");
        }
    }
}

// --- B3 -------------------------------------------------------------------

#[test]
fn preroll_starts_voice_early_for_every_kind() {
    let banks = banks(dc(0.5, 44_100));
    let ks = kinds(&banks);
    assert!(count(&ks, Group::Registered) >= 1 && count(&ks, Group::Sample) >= 1);
    let mut ran = 0;
    for k in &ks {
        for &b in &BLOCKS {
            let lat = own_latency(&k.def, b);
            for how in SPAWNS {
                // Preroll 37 at notated 1000 starts at 963; preroll 0 at
                // 1000; notated 20 with preroll 37 starts at 0.
                for (n, p, start) in [
                    (1000u64, PREROLL, 963usize),
                    (1000, 0, 1000),
                    (20, PREROLL, 0),
                ] {
                    let (out, _) = play(k, how, b, p, n, 600);
                    assert_eq!(
                        first_nonzero(&out),
                        Some(start + lat),
                        "{} {how:?} block {b}, N={n}, preroll {p}",
                        k.name
                    );
                    ran += 1;
                }
                if k.group == Group::Plain {
                    // The gate closes at notated + duration, unshifted.
                    let (out, _) = play(k, how, b, PREROLL, 1000, 600);
                    let off = out[1100..].iter().position(|&x| x == 0.0).unwrap() + 1100;
                    assert_eq!(off, 1600, "{how:?} block {b}: gate-off at N + dur");
                }
            }
        }
    }
    assert!(ran >= 3 * 3 * 2 * 3, "the loop ran too little: {ran}");
}

// --- B4 -------------------------------------------------------------------

#[test]
fn preroll_click_lands_on_notated_time() {
    let banks = banks(lead_click(PREROLL as usize));
    let def = sample_def(&banks, "click");
    let k = Kind {
        group: Group::Sample,
        name: "click".into(),
        def,
    };
    for &b in &BLOCKS {
        let (out, _) = play(&k, Spawn::OnBus, b, PREROLL, 1000, 2000);
        let peak =
            out.iter().enumerate().fold(
                (0, 0.0f32),
                |m, (i, &x)| if x.abs() > m.1 { (i, x.abs()) } else { m },
            );
        assert!(peak.1 > 0.5, "the click sounds");
        assert!(
            (999..=1001).contains(&peak.0),
            "block {b}: click at {}",
            peak.0
        );
    }
}

// --- B5 -------------------------------------------------------------------

#[test]
fn scheduled_note_records_both_onsets() {
    let banks = banks(dc(0.5, 44_100));
    let ks = kinds(&banks);
    let k = ks.iter().find(|k| k.group == Group::Plain).unwrap();
    for &b in &BLOCKS {
        let (mut engine, bus) = rig(b);
        engine.set_preroll(k.def.name(), PREROLL);
        let v = spawn(&mut engine, bus, &k.def, Spawn::OnBus);
        engine.set_preroll(k.def.name(), 0);
        let zero = engine.schedule_note(v, 1000, 100);
        engine.set_preroll(k.def.name(), PREROLL);
        let early = engine.schedule_note(v, 1000, 100);
        let clamped = engine.schedule_note(v, 20, 100);
        assert_eq!((early.notated, early.sound), (1000, 963));
        assert_eq!((zero.notated, zero.sound), (1000, 1000));
        assert_eq!((clamped.notated, clamped.sound), (20, 0));
        let kept: Vec<(u64, u64)> = engine
            .note_onsets()
            .iter()
            .map(|o| (o.notated, o.sound))
            .collect();
        assert_eq!(kept, vec![(1000, 1000), (1000, 963), (20, 0)]);
    }
}

#[test]
fn a_bus_tap_equals_the_sink_on_a_split_block() {
    // The stem taps read a node's output after the block. After a block
    // rendered in pieces that output must be the whole block, aligned with
    // the sink's, not the last piece.
    let reg = builtin_registry();
    let def = dsl::compile("synthdef plain gate=0.0 = gate * 0.5", &reg)
        .unwrap()
        .remove(0);
    for &b in &BLOCKS {
        let mut engine = Engine::new(EngineConfig {
            sample_rate: SR,
            block_size: b,
        });
        let stem = engine
            .graph_mut()
            .add_node(Box::new(ugens::Bus::new(ugens::ChannelCount::Stereo)));
        let main = engine
            .graph_mut()
            .add_node(Box::new(ugens::Bus::new(ugens::ChannelCount::Stereo)));
        engine.graph_mut().connect(stem, main, 0);
        engine.graph_mut().set_sink(main);
        let v = engine.spawn_voice_on_bus(&def, stem).unwrap();
        engine.prepare();
        // The gate opens at sample 37 and a second step lands at 50: the
        // first block renders in three pieces.
        let n = 37;
        engine.scheduler_mut().schedule_gate(n, v, 1.0);
        engine.scheduler_mut().schedule_gate(50, v, 0.0);
        let sink: Vec<Vec<f32>> = {
            let buf = engine.render().expect("sink");
            (0..2).map(|c| buf.channel(c).samples().to_vec()).collect()
        };
        let tap = engine.graph().node_output(stem).expect("stem bus");
        assert_eq!(tap.block_size(), b, "the tap is a whole block");
        for (c, want) in sink.iter().enumerate() {
            assert_eq!(tap.channel(c).samples(), &want[..], "block {b} channel {c}");
        }
        assert_eq!(first_nonzero(&sink[0]), Some(n as usize));
    }
}
