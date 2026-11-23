//! `partialsEnsemble` (`microsynth::ugens::partials_ensemble`): the bank at
//! one player matches the blend bank it extends, renders are reproducible
//! from the voice seed and depend on it, a section keeps one player's power,
//! the onset smear, a multichannel control input, the done rule through a
//! def's envelope, and registration.

use microsynth::coeff_table::{CoeffTable, PitchEntry};
use microsynth::ugens::partials_blend::PartialsBlend;
use microsynth::ugens::partials_ensemble::PartialsEnsemble;
use microsynth::ugens::*;
use microsynth::*;
use std::sync::Arc;

const SR: f32 = 44100.0;
const BLOCK: usize = 128;

fn config() -> EngineConfig {
    EngineConfig {
        sample_rate: SR,
        block_size: BLOCK,
    }
}

fn coeff(seed: u32, i: usize) -> f32 {
    let x = (seed as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add((i as u64).wrapping_mul(1442695040888963407));
    ((x >> 40) as f32 / (1u64 << 24) as f32) * 0.9 + 0.1
}

const F0S: [f32; 4] = [180.0, 220.0, 262.0, 330.0];

/// A three-channel table with noise bands, partial counts falling with pitch.
fn table(noise: bool) -> CoeffTable {
    let j = 6;
    CoeffTable {
        name: "ens".into(),
        entries: F0S
            .iter()
            .enumerate()
            .map(|(e, &f0)| {
                let m = 24 - 3 * e;
                let coefficients = (0..3)
                    .flat_map(|k| {
                        let mut r: Vec<f32> = (0..m)
                            .map(|i| coeff(10 + k * 7 + e as u32, i) / (i as f32 + 1.0))
                            .collect();
                        r.extend((0..j).map(|i| if noise { 0.02 * coeff(99 + k, i) } else { 0.0 }));
                        r
                    })
                    .collect();
                PitchEntry {
                    f0_hz: f0,
                    inharmonicity_stretch: 1.0,
                    partial_freqs: (1..=m).map(|k| k as f32 * f0).collect(),
                    k_channels: 3,
                    j_noise: j as u32,
                    coefficients,
                    metadata: vec![
                        ("mainlobe_gain".into(), 0.7),
                        ("noise_gain".into(), 1.0),
                        ("noise_band_min_hz".into(), 0.0),
                        ("noise_band_max_hz".into(), 8000.0),
                    ],
                }
            })
            .collect(),
    }
}

/// Inputs: a vibrato-like freq (240 Hz ± 1.5 % at 5.5 Hz) and three gain
/// lines (an attack decay and its complement split 0.4/0.6).
fn lanes(engine: &mut Engine, secs: f32) -> (NodeId, Vec<NodeId>) {
    let g = engine.graph_mut();
    let rate = g.add_node(Box::new(Const::new(5.5)));
    let lfo = g.add_node(Box::new(SinOsc::new()));
    g.connect(rate, lfo, 0);
    let depth = g.add_node(Box::new(Const::new(240.0 * 0.015)));
    let m = g.add_node(Box::new(BinOpUGen::new(BinOpKind::Mul)));
    g.connect(lfo, m, 0);
    g.connect(depth, m, 1);
    let base = g.add_node(Box::new(Const::new(240.0)));
    let freq = g.add_node(Box::new(BinOpUGen::new(BinOpKind::Add)));
    g.connect(m, freq, 0);
    g.connect(base, freq, 1);
    let mut line = |a: f32, b: f32| {
        let s = g.add_node(Box::new(Const::new(a)));
        let e = g.add_node(Box::new(Const::new(b)));
        let d = g.add_node(Box::new(Const::new(secs)));
        let l = g.add_node(Box::new(Line::new()));
        g.connect(s, l, 0);
        g.connect(e, l, 1);
        g.connect(d, l, 2);
        l
    };
    let gains = vec![line(1.0, 0.0), line(0.0, 0.4), line(0.0, 0.6)];
    (freq, gains)
}

fn konst(engine: &mut Engine, v: f32) -> NodeId {
    engine.graph_mut().add_node(Box::new(Const::new(v)))
}

/// The blend bank.
fn render_blend(t: CoeffTable, secs: f32, seed: u32) -> Vec<f32> {
    let mut engine = Engine::new(config());
    let (freq, gains) = lanes(&mut engine, secs);
    let mut bank = PartialsBlend::new(Arc::new(t));
    bank.reseed_noise(seed);
    let b = engine.graph_mut().add_node(Box::new(bank));
    engine.graph_mut().connect(freq, b, 0);
    for (k, g) in gains.iter().enumerate() {
        engine.graph_mut().connect(*g, b, 1 + k);
    }
    engine.graph_mut().set_sink(b);
    engine.prepare();
    engine
        .render_offline(((secs * SR) as usize).div_ceil(BLOCK))
        .remove(0)
}

/// The ensemble bank with port settings `(port index, value)`; returns (L, R).
fn render_ens(t: CoeffTable, secs: f32, seed: u32, ports: &[(usize, f32)]) -> (Vec<f32>, Vec<f32>) {
    let mut engine = Engine::new(config());
    let (freq, gains) = lanes(&mut engine, secs);
    let mut bank = PartialsEnsemble::new(Arc::new(t));
    bank.reseed_noise(seed);
    let b = engine.graph_mut().add_node(Box::new(bank));
    engine.graph_mut().connect(freq, b, 0);
    for (k, g) in gains.iter().enumerate() {
        engine.graph_mut().connect(*g, b, 1 + k);
    }
    for &(p, v) in ports {
        let c = konst(&mut engine, v);
        engine.graph_mut().connect(c, b, p);
    }
    engine.graph_mut().set_sink(b);
    engine.prepare();
    let mut out = engine.render_offline(((secs * SR) as usize).div_ceil(BLOCK));
    assert_eq!(out.len(), 2, "stereo output");
    let r = out.remove(1);
    (out.remove(0), r)
}

const PLAYERS: usize = 5;
const SPREAD: usize = 6;
const VIB: usize = 7;
const ONSET: usize = 8;
const WIDTH: usize = 9;
const CTRL: usize = 10;
const PHASE: usize = 11;

fn rms_diff_db(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut d, mut e) = (0.0f64, 0.0f64);
    for i in 0..n {
        d += ((a[i] - b[i]) as f64).powi(2);
        e += (a[i] as f64).powi(2);
    }
    10.0 * (d / e).log10()
}

#[test]
fn one_player_at_full_control_rate_matches_the_blend_bank() {
    // phase 0, one player, ctrl 1: both channels are the blend bank's output
    // up to the sine polynomial and summation order
    let a = render_blend(table(true), 0.8, 21);
    let (l, r) = render_ens(table(true), 0.8, 21, &[(PHASE, 0.0), (CTRL, 1.0)]);
    assert_eq!(l, r);
    let db = rms_diff_db(&a, &l);
    assert!(db < -100.0, "ensemble(1 player) vs blend: {db:.1} dB");
    // at the default control period the amplitudes are interpolated
    let (l16, _) = render_ens(table(true), 0.8, 21, &[(PHASE, 0.0)]);
    let db16 = rms_diff_db(&a, &l16);
    assert!(db16 < -50.0, "ctrl 16 vs blend: {db16:.1} dB");
}

#[test]
fn renders_are_reproducible_and_seeded() {
    let set = [
        (PLAYERS, 12.0),
        (SPREAD, 5.0),
        (VIB, 20.0),
        (ONSET, 0.013),
        (WIDTH, 0.7),
    ];
    let a = render_ens(table(true), 0.5, 7, &set);
    let b = render_ens(table(true), 0.5, 7, &set);
    let c = render_ens(table(true), 0.5, 8, &set);
    assert_eq!(a, b);
    assert_ne!(a.0, c.0);
    assert_ne!(a.0, a.1, "width 0.7 decorrelates the channels");
    let w0 = render_ens(table(true), 0.5, 7, &[(PLAYERS, 12.0), (SPREAD, 5.0)]);
    assert_eq!(w0.0, w0.1, "width 0: both channels equal");
}

#[test]
fn ensemble_keeps_the_solo_power() {
    // long-run power of a large section within 0.5 dB of one player's
    // (pooled over seeds: the section's power is itself random)
    let secs = 4.0;
    let p = |x: &[f32]| x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    let (mut solo, mut sec) = (0.0, 0.0);
    for seed in 0..6 {
        solo += p(&render_ens(table(false), secs, seed, &[]).0);
        sec += p(&render_ens(table(false), secs, seed, &[(PLAYERS, 1e6), (SPREAD, 20.0)]).0);
    }
    let db = 10.0 * (sec / solo).log10();
    assert!(db.abs() < 0.5, "section vs solo power {db:.2} dB");
}

#[test]
fn onset_smear_delays_the_attack_burst() {
    // the attack-table gain (gain0) decays from 1: with a 40 ms smear its
    // first samples carry a quarter of it (one tap of four)
    let (a, _) = render_ens(table(false), 0.2, 3, &[(PHASE, 0.0)]);
    let (b, _) = render_ens(table(false), 0.2, 3, &[(PHASE, 0.0), (ONSET, 0.04)]);
    let e = |x: &[f32], lo: usize, hi: usize| x[lo..hi].iter().map(|v| v * v).sum::<f32>();
    let early = e(&b, 0, 400) / e(&a, 0, 400);
    assert!(early < 0.15, "early energy ratio {early}");
}

#[test]
fn registered_as_table_bound() {
    let mut reg = microsynth::dsl::compiler::UGenRegistry::new();
    microsynth::register_builtins(&mut reg);
    microsynth::register_table_bound_builtins(&mut reg);
    let entry = reg
        .table_bound_entry("partialsEnsemble")
        .expect("registered");
    assert_eq!(entry.category, UGenCategory::Oscillator);
    assert_eq!(
        entry.input_names,
        vec![
            "freq", "gain0", "gain1", "gain2", "gain3", "players", "spread", "vib", "onset",
            "width", "ctrl", "phase"
        ]
    );
    assert!(entry.required.iter().all(|r| !r), "every port is optional");
}

#[test]
fn the_output_is_stereo_and_a_multichannel_control_input_reads_its_first_channel() {
    // a stereo freq whose channels are equal must give the mono-input render
    let set = [(PLAYERS, 6.0), (SPREAD, 5.0), (WIDTH, 0.7)];
    let mono = render_ens(table(true), 0.3, 4, &set);
    let mut engine = Engine::new(config());
    let (freq, gains) = lanes(&mut engine, 0.3);
    let g = engine.graph_mut();
    // a stereo zero (pan2 of silence), added to the mono freq: two exactly
    // equal channels
    let zero = g.add_node(Box::new(Const::new(0.0)));
    let pan = g.add_node(Box::new(Pan2::new()));
    g.connect(zero, pan, 0);
    let f2 = g.add_node(Box::new(BinOpUGen::new(BinOpKind::Add)));
    g.connect(freq, f2, 0);
    g.connect(pan, f2, 1);
    let mut bank = PartialsEnsemble::new(Arc::new(table(true)));
    bank.reseed_noise(4);
    let b = g.add_node(Box::new(bank));
    g.connect(f2, b, 0);
    for (i, gn) in gains.iter().enumerate() {
        g.connect(*gn, b, 1 + i);
    }
    for &(p, v) in &set {
        let c = g.add_node(Box::new(Const::new(v)));
        g.connect(c, b, p);
    }
    g.set_sink(b);
    engine.prepare();
    let out = engine.render_offline(((0.3 * SR) as usize).div_ceil(BLOCK));
    assert_eq!(out.len(), 2);
    let n = mono.0.len().min(out[0].len());
    for (ch, (want, got)) in [&mono.0, &mono.1].into_iter().zip(&out).enumerate() {
        assert_eq!(
            &want[..n],
            &got[..n],
            "channel {ch} differs from the mono-input render"
        );
    }
}

/// A def whose bank output passes through a finished envelope is freed (the
/// bank is a source and stops silence like the other banks; the envelope's
/// `Mul` carries the done rule).
#[test]
fn a_finished_envelope_after_the_bank_frees_the_voice() {
    use microsynth::coeff_table::CoeffTableBank;
    use microsynth::ir::{IrEdge, IrNode, IrSynthDef, IrTableBinding, SynthDefClass};
    let mut reg = microsynth::dsl::compiler::UGenRegistry::new();
    microsynth::register_builtins(&mut reg);
    microsynth::register_table_bound_builtins(&mut reg);
    let mut tables = CoeffTableBank::new();
    let id = tables.register(table(true));
    let ir = IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "ens_done".into(),
        class: SynthDefClass::Source,
        output_channels: 2,
        nodes: vec![
            IrNode::Param {
                name: "freq".into(),
                default: 220.0,
            },
            IrNode::UGen {
                kind: "partialsEnsemble".into(),
                consts: vec![(5, 6.0), (6, 5.0), (9, 0.7)],
            },
            IrNode::UGen {
                kind: "expPerc".into(),
                consts: vec![(0, 0.001), (1, 0.02)],
            },
            IrNode::UGen {
                kind: "Mul".into(),
                consts: vec![],
            },
        ],
        edges: vec![
            IrEdge {
                from: 0,
                to: 1,
                to_input: 0,
            },
            IrEdge {
                from: 1,
                to: 3,
                to_input: 0,
            },
            IrEdge {
                from: 2,
                to: 3,
                to_input: 1,
            },
        ],
        params: vec![],
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding {
            node: 1,
            table_id: id.0,
        }],
        output_node: 3,
    };
    ir.validate(&reg).expect("valid IR");
    let def = ir.compile_with_tables(&reg, &tables).expect("resolves");
    let mut engine = Engine::new(config());
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let mut freed_at = None;
    let mut peak = 0.0f32;
    for blk in 0..400 {
        if let Some(buf) = engine.render() {
            for ch in 0..buf.num_channels() {
                for &s in buf.channel(ch).samples() {
                    peak = peak.max(s.abs());
                }
            }
        }
        if engine.free_done_synths() > 0 {
            freed_at = Some(blk);
            break;
        }
    }
    assert!(peak > 0.01, "the def sounded before it ended");
    assert!(freed_at.is_some(), "the voice was never freed");
}
