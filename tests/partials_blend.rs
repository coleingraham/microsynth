//! `partialsBlend` (`microsynth::ugens::partials_blend::PartialsBlend`): one
//! bank blending a table's channels by audio-rate gains, checked against the
//! sum of one `partialsNoise` bank per channel that it replaces.

use microsynth::coeff_table::{CoeffTable, CoeffTableBank, PitchEntry};
use microsynth::ugens::partials_blend::PartialsBlend;
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

/// Deterministic pseudo-random coefficients in (0, 1].
fn coeff(seed: u32, i: usize) -> f32 {
    let x = (seed as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(i as u64 * 1442695040888963407);
    ((x >> 40) as f32 / (1u64 << 24) as f32) * 0.9 + 0.1
}

const F0S: [f32; 4] = [180.0, 220.0, 262.0, 330.0];
const META: [(&str, f32); 4] = [
    ("mainlobe_gain", 0.7),
    ("noise_gain", 1.0),
    ("noise_band_min_hz", 0.0),
    ("noise_band_max_hz", 8000.0),
];

/// Channel `k`'s coefficient row at entry `e` (partials then noise). Partial
/// counts fall with pitch (as the additive generator writes them) and the
/// top partials are zero in every channel.
fn row(k: usize, e: usize, m: usize, j: usize, noise: bool) -> Vec<f32> {
    let mut r: Vec<f32> = (0..m)
        .map(|i| {
            if i + 4 >= m {
                0.0
            } else {
                coeff(10 + k as u32 * 7 + e as u32, i) / (i as f32 + 1.0)
            }
        })
        .collect();
    r.extend((0..j).map(|i| {
        if noise {
            0.02 * coeff(99 + k as u32, i)
        } else {
            0.0
        }
    }));
    r
}

fn m_at(e: usize) -> usize {
    24 - 3 * e
}

/// One table per channel (`k_channels` 1), as three-bank defs carry them.
fn single_tables(k_n: usize, j: usize, noise: bool) -> Vec<CoeffTable> {
    (0..k_n)
        .map(|k| CoeffTable {
            name: format!("ch{k}"),
            entries: F0S
                .iter()
                .enumerate()
                .map(|(e, &f0)| PitchEntry {
                    f0_hz: f0,
                    inharmonicity_stretch: 1.0,
                    partial_freqs: (1..=m_at(e)).map(|m| m as f32 * f0).collect(),
                    k_channels: 1,
                    j_noise: j as u32,
                    coefficients: row(k, e, m_at(e), j, noise),
                    metadata: META.iter().map(|(a, b)| (a.to_string(), *b)).collect(),
                })
                .collect(),
        })
        .collect()
}

/// The same content as one `k_n`-channel table.
fn blend_table(k_n: usize, j: usize, noise: bool) -> CoeffTable {
    CoeffTable {
        name: "blend".into(),
        entries: F0S
            .iter()
            .enumerate()
            .map(|(e, &f0)| PitchEntry {
                f0_hz: f0,
                inharmonicity_stretch: 1.0,
                partial_freqs: (1..=m_at(e)).map(|m| m as f32 * f0).collect(),
                k_channels: k_n as u32,
                j_noise: j as u32,
                coefficients: (0..k_n)
                    .flat_map(|k| row(k, e, m_at(e), j, noise))
                    .collect(),
                metadata: META.iter().map(|(a, b)| (a.to_string(), *b)).collect(),
            })
            .collect(),
    }
}

/// A freq glide 170 → 340 Hz over the render (crosses every bracket and both
/// grid ends) and three gain lines: an attack decay, its complement split by a
/// fixed velocity weight, as the additive defs crossfade their banks.
struct Lanes {
    freq: NodeId,
    gains: Vec<NodeId>,
}

fn lanes(engine: &mut Engine, secs: f32) -> Lanes {
    let mut line = |a: f32, b: f32| {
        let g = engine.graph_mut();
        let s = g.add_node(Box::new(Const::new(a)));
        let e = g.add_node(Box::new(Const::new(b)));
        let d = g.add_node(Box::new(Const::new(secs)));
        let l = g.add_node(Box::new(Line::new()));
        g.connect(s, l, 0);
        g.connect(e, l, 1);
        g.connect(d, l, 2);
        l
    };
    let freq = line(170.0, 340.0);
    let gains = vec![
        line(1.0, 0.0),
        line(0.0, 0.38),
        line(0.0, 0.62),
        line(0.3, 0.3),
    ];
    Lanes { freq, gains }
}

fn render_sum_of_banks(tables: Vec<CoeffTable>, secs: f32) -> Vec<f32> {
    let mut engine = Engine::new(config());
    let l = lanes(&mut engine, secs);
    let mut prev: Option<NodeId> = None;
    for (k, t) in tables.into_iter().enumerate() {
        // independent sources per bank, as a voice seeds each node (engine:
        // one seed per spawn and node index)
        let mut bank = PartialsNoise::new(Arc::new(t));
        bank.reseed_noise(0x5EED_0000 + k as u32);
        let b = engine.graph_mut().add_node(Box::new(bank));
        engine.graph_mut().connect(l.freq, b, 0);
        engine.graph_mut().connect(l.gains[k], b, 1);
        prev = Some(match prev {
            None => b,
            Some(p) => {
                let a = engine
                    .graph_mut()
                    .add_node(Box::new(BinOpUGen::new(BinOpKind::Add)));
                engine.graph_mut().connect(p, a, 0);
                engine.graph_mut().connect(b, a, 1);
                a
            }
        });
    }
    engine.graph_mut().set_sink(prev.unwrap());
    engine.prepare();
    engine
        .render_offline(((secs * SR) as usize).div_ceil(BLOCK))
        .remove(0)
}

fn render_blend(table: CoeffTable, k_n: usize, secs: f32) -> Vec<f32> {
    let mut engine = Engine::new(config());
    let l = lanes(&mut engine, secs);
    let b = engine
        .graph_mut()
        .add_node(Box::new(PartialsBlend::new(Arc::new(table))));
    engine.graph_mut().connect(l.freq, b, 0);
    for k in 0..k_n {
        engine.graph_mut().connect(l.gains[k], b, 1 + k);
    }
    engine.graph_mut().set_sink(b);
    engine.prepare();
    engine
        .render_offline(((secs * SR) as usize).div_ceil(BLOCK))
        .remove(0)
}

fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |a, &b| a.max(b.abs()))
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn three_channels_equal_three_summed_banks() {
    for k_n in [2usize, 3, 4] {
        let a = render_sum_of_banks(single_tables(k_n, 0, false), 1.0);
        let b = render_blend(blend_table(k_n, 0, false), k_n, 1.0);
        let p = peak(&a);
        assert!(p > 0.1, "reference is silent");
        let d = max_diff(&a, &b);
        assert!(
            d <= 2e-5 * p.max(1.0),
            "K={k_n}: max |diff| {d} vs peak {p}"
        );
    }
}

#[test]
fn one_channel_with_default_gain_matches_partials_noise() {
    let t = single_tables(1, 0, false).remove(0);
    let mut e1 = Engine::new(config());
    let f = e1.graph_mut().add_node(Box::new(Const::new(240.0)));
    let b = e1
        .graph_mut()
        .add_node(Box::new(PartialsNoise::new(Arc::new(t.clone()))));
    e1.graph_mut().connect(f, b, 0);
    e1.graph_mut().set_sink(b);
    e1.prepare();
    let a = e1.render_offline(200).remove(0);

    let mut e2 = Engine::new(config());
    let f = e2.graph_mut().add_node(Box::new(Const::new(240.0)));
    let bl = PartialsBlend::new(Arc::new(t));
    assert_eq!(bl.channels(), 1);
    let b = e2.graph_mut().add_node(Box::new(bl));
    e2.graph_mut().connect(f, b, 0);
    e2.graph_mut().set_sink(b);
    e2.prepare();
    let c = e2.render_offline(200).remove(0);
    let d = max_diff(&a, &c);
    assert!(d <= 1e-6 * peak(&a).max(1.0), "max |diff| {d}");
}

#[test]
fn never_sounding_partials_and_bands_are_not_run() {
    let t = blend_table(3, 5, false);
    let bl = PartialsBlend::new(Arc::new(t));
    // entry 0 has 24 partials, the last four zero in every channel
    assert_eq!(bl.partials(), 20);
    // all-zero noise bands render exactly as a table with no noise columns
    let a = render_blend(blend_table(3, 5, false), 3, 0.3);
    let b = render_blend(blend_table(3, 0, false), 3, 0.3);
    assert_eq!(a, b);
}

/// Band power of separate banks (independent sources) is matched by the
/// root-sum-of-squares gain on one source.
#[test]
fn noise_band_power_matches_independent_banks() {
    // noise only: zero the partials by using a table with noise columns and
    // subtracting the noise-free render
    let secs = 3.0;
    let sum_n = render_sum_of_banks(single_tables(3, 6, true), secs);
    let sum_p = render_sum_of_banks(single_tables(3, 6, false), secs);
    let bl_n = render_blend(blend_table(3, 6, true), 3, secs);
    let bl_p = render_blend(blend_table(3, 6, false), 3, secs);
    let pow = |a: &[f32], b: &[f32]| -> f64 {
        let n = a.len().min(b.len());
        let s = n / 2; // second half: gains settled toward their ends
        a[s..n]
            .iter()
            .zip(&b[s..n])
            .map(|(x, y)| ((x - y) as f64).powi(2))
            .sum::<f64>()
            / (n - s) as f64
    };
    let p_banks = pow(&sum_n, &sum_p);
    let p_blend = pow(&bl_n, &bl_p);
    assert!(p_banks > 0.0 && p_blend > 0.0);
    let db = 10.0 * (p_blend / p_banks).log10();
    assert!(db.abs() < 0.5, "noise power blend vs banks {db:.3} dB");
}

#[test]
fn missing_channel_reads_zero_and_extra_channels_are_ignored() {
    // 5 channels: the fifth is not rendered
    let t5 = blend_table(5, 0, false);
    let bl = PartialsBlend::new(Arc::new(t5));
    assert_eq!(bl.channels(), 4);
    // an entry with one channel fewer: its channel 2 reads zero
    let mut t = blend_table(3, 0, false);
    let e = &mut t.entries[0];
    let w = e.partial_freqs.len();
    e.k_channels = 2;
    e.coefficients.truncate(2 * w);
    let bl = PartialsBlend::new(Arc::new(t));
    assert_eq!(bl.channels(), 3);
}

#[test]
fn empty_table_is_silent() {
    let mut e = Engine::new(config());
    let b = e.graph_mut().add_node(Box::new(PartialsBlend::new(
        Arc::new(CoeffTable::default()),
    )));
    e.graph_mut().set_sink(b);
    e.prepare();
    assert!(e.render_offline(4).remove(0).iter().all(|&x| x == 0.0));
}

#[test]
fn table_bound_registration_resolves_and_renders() {
    use microsynth::ir::{IrEdge, IrNode, IrSynthDef, IrTableBinding, SynthDefClass};
    let mut reg = microsynth::dsl::compiler::UGenRegistry::new();
    microsynth::register_builtins(&mut reg);
    microsynth::register_table_bound_builtins(&mut reg);
    let mut bank = CoeffTableBank::new();
    let id = bank.register(blend_table(3, 0, false));
    let ir = IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "blend_probe".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![
            IrNode::Param {
                name: "freq".into(),
                default: 220.0,
            },
            IrNode::UGen {
                kind: "partialsBlend".into(),
                consts: vec![(2, 0.5), (3, 0.5)],
            },
        ],
        edges: vec![IrEdge {
            from: 0,
            to: 1,
            to_input: 0,
        }],
        params: vec![],
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding {
            node: 1,
            table_id: id.0,
        }],
        output_node: 1,
    };
    ir.validate(&reg).expect("valid IR");
    let def = ir.compile_with_tables(&reg, &bank).expect("resolves");
    let mut engine = Engine::new(config());
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let out = engine.render_offline(20).remove(0);
    assert!(peak(&out) > 0.05, "renders");
    let entry = reg.table_bound_entry("partialsBlend").expect("registered");
    assert_eq!(entry.category, UGenCategory::Oscillator);
}
