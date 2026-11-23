//! `tableValue` (`microsynth::ugens::table_value::TableValue`): a per-pitch
//! value from a coefficient table's entry metadata, interpolated at `freq`
//! as the partial banks interpolate, clamped outside the grid, a default for
//! a missing key, one output channel per `freq` channel, and the IR path
//! with the node and a bank bound to one table.

use microsynth::coeff_table::{CoeffTable, CoeffTableBank, PitchEntry};
use microsynth::ugens::table_value::TableValue;
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

/// Entries at 100, 200, 400 Hz carrying `value.3` = 1, 3, 2 (the middle
/// entry also `value.0` = 9), one partial each.
fn table() -> CoeffTable {
    let e = |f0: f32, v3: f32, extra: bool| {
        let mut metadata = vec![("mainlobe_gain".into(), 1.0), ("value.3".into(), v3)];
        if extra {
            metadata.push(("value.0".into(), 9.0));
        }
        PitchEntry {
            f0_hz: f0,
            inharmonicity_stretch: 1.0,
            partial_freqs: vec![f0],
            k_channels: 1,
            j_noise: 0,
            coefficients: vec![0.5],
            metadata,
        }
    };
    // out of order on purpose: the node sorts by f0
    CoeffTable {
        name: "values".into(),
        entries: vec![
            e(400.0, 2.0, false),
            e(100.0, 1.0, false),
            e(200.0, 3.0, true),
        ],
    }
}

/// Render `freq` (a constant per channel: one channel per value) through a
/// tableValue with `slot` and `default`; the first sample of each channel.
fn value(freqs: &[f32], slot: f32, default: Option<f32>) -> Vec<f32> {
    let mut engine = Engine::new(config());
    let g = engine.graph_mut();
    let node = g.add_node(Box::new(TableValue::new(Arc::new(table()))));
    let f = g.add_node(Box::new(Const::new(freqs[0])));
    let freq = if freqs.len() == 2 {
        // a two-channel freq: pan2 hard left/right of two constants, summed
        let a = g.add_node(Box::new(Pan2::new()));
        let b = g.add_node(Box::new(Pan2::new()));
        let l = g.add_node(Box::new(Const::new(-1.0)));
        let r = g.add_node(Box::new(Const::new(1.0)));
        let f1 = g.add_node(Box::new(Const::new(freqs[1])));
        g.connect(f, a, 0);
        g.connect(l, a, 1);
        g.connect(f1, b, 0);
        g.connect(r, b, 1);
        let s = g.add_node(Box::new(BinOpUGen::new(BinOpKind::Add)));
        g.connect(a, s, 0);
        g.connect(b, s, 1);
        s
    } else {
        f
    };
    g.connect(freq, node, 0);
    let s = g.add_node(Box::new(Const::new(slot)));
    g.connect(s, node, 1);
    if let Some(d) = default {
        let c = g.add_node(Box::new(Const::new(d)));
        g.connect(c, node, 2);
    }
    g.set_sink(node);
    engine.prepare();
    let out = engine.render_offline(2);
    out.iter().map(|ch| ch[BLOCK + 5]).collect()
}

#[test]
fn interpolates_linearly_in_f0_and_clamps_outside_the_grid() {
    assert_eq!(value(&[100.0], 3.0, None), vec![1.0]);
    assert_eq!(value(&[150.0], 3.0, None), vec![2.0]);
    assert_eq!(value(&[200.0], 3.0, None), vec![3.0]);
    assert_eq!(value(&[300.0], 3.0, None), vec![2.5]);
    assert_eq!(value(&[50.0], 3.0, None), vec![1.0]);
    assert_eq!(value(&[1000.0], 3.0, None), vec![2.0]);
}

#[test]
fn a_missing_key_reads_the_default() {
    // value.0 only on the 200 Hz entry: 150 Hz is halfway from default 4 to 9
    assert_eq!(value(&[150.0], 0.0, Some(4.0)), vec![6.5]);
    assert_eq!(value(&[150.0], 7.0, Some(-1.0)), vec![-1.0]);
    assert_eq!(value(&[150.0], 7.0, None), vec![0.0]);
}

#[test]
fn each_freq_channel_gets_its_own_value() {
    // pan2 hard left/right gives cos(0)=1 and sin(pi/2)=1 on the open sides
    let v = value(&[150.0, 300.0], 3.0, None);
    assert_eq!(v.len(), 2);
    assert!(
        (v[0] - 2.0).abs() < 1e-5 && (v[1] - 2.5).abs() < 1e-5,
        "{v:?}"
    );
}

#[test]
fn follows_a_moving_freq() {
    let mut engine = Engine::new(config());
    let g = engine.graph_mut();
    let node = g.add_node(Box::new(TableValue::new(Arc::new(table()))));
    let (a, b, d) = (
        g.add_node(Box::new(Const::new(100.0))),
        g.add_node(Box::new(Const::new(200.0))),
        g.add_node(Box::new(Const::new(BLOCK as f32 * 4.0 / SR))),
    );
    let l = g.add_node(Box::new(Line::new()));
    g.connect(a, l, 0);
    g.connect(b, l, 1);
    g.connect(d, l, 2);
    g.connect(l, node, 0);
    let s = g.add_node(Box::new(Const::new(3.0)));
    g.connect(s, node, 1);
    g.set_sink(node);
    engine.prepare();
    let out = engine.render_offline(6).remove(0);
    // the value walks 1 -> 3 with the line, then holds
    assert!(out[1] < out[BLOCK * 2] && out[BLOCK * 2] < out[BLOCK * 4 - 1]);
    assert!((out[BLOCK * 5] - 3.0).abs() < 1e-5);
}

/// The IR path: a bank and a tableValue bound to one table id.
#[test]
fn binds_with_a_bank_to_one_table() {
    use microsynth::ir::{IrEdge, IrNode, IrSynthDef, IrTableBinding, SynthDefClass};
    let mut reg = microsynth::dsl::compiler::UGenRegistry::new();
    microsynth::register_builtins(&mut reg);
    microsynth::register_table_bound_builtins(&mut reg);
    let entry = reg.table_bound_entry("tableValue").expect("registered");
    assert_eq!(entry.category, UGenCategory::Utility);
    assert_eq!(entry.input_names, vec!["freq", "slot", "default"]);
    let mut bank = CoeffTableBank::new();
    let id = bank.register(table());
    let ir = IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "value_probe".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![
            IrNode::Param {
                name: "freq".into(),
                default: 300.0,
            },
            IrNode::UGen {
                kind: "tableValue".into(),
                consts: vec![(1, 3.0)],
            },
            IrNode::UGen {
                kind: "partialsBlend".into(),
                consts: vec![],
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
                from: 0,
                to: 2,
                to_input: 0,
            },
            IrEdge {
                from: 2,
                to: 3,
                to_input: 0,
            },
            IrEdge {
                from: 1,
                to: 3,
                to_input: 1,
            },
        ],
        params: vec![],
        audio_inputs: vec![],
        table_bindings: vec![
            IrTableBinding {
                node: 1,
                table_id: id.0,
            },
            IrTableBinding {
                node: 2,
                table_id: id.0,
            },
        ],
        output_node: 3,
    };
    ir.validate(&reg).expect("valid IR");
    let def = ir.compile_with_tables(&reg, &bank).expect("resolves");
    let mut engine = Engine::new(config());
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let out = engine.render_offline(20).remove(0);
    // the bank alone peaks at 0.5; the value at 300 Hz is 2.5
    let peak = out.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!((peak - 1.25).abs() < 0.01, "peak {peak}");
}
