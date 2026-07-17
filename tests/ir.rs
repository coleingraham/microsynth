//! SynthDef IR: decompile/compile round-trip renders byte-identically to the
//! direct DSL compile, plus validation coverage.

use microsynth::dsl::compiler::compile_synthdef;
use microsynth::dsl::lexer::tokenize;
use microsynth::dsl::parser::Parser;
use microsynth::ir::{IrEdge, IrNode, IrSynthDef, RenderSpec, SynthDefClass, from_decl, render_ir};
use microsynth::{Engine, EngineConfig, SynthDef};

mod common;
use common::builtin_registry as registry;

/// Parse one `synthdef` source into a single declaration.
fn parse_one(src: &str) -> microsynth::dsl::ast::SynthDefDecl {
    let tokens = tokenize(src).unwrap_or_else(|e| panic!("lex {src:?}: {e}"));
    let mut parser = Parser::new(tokens);
    let mut program = parser
        .parse_program()
        .unwrap_or_else(|e| panic!("parse {src:?}: {e}"));
    assert_eq!(program.defs.len(), 1, "expected exactly one def in {src:?}");
    program.defs.pop().unwrap()
}

/// Render a def to raw samples. Mirrors the CLI: auto-gate sustaining synths,
/// fixed config so two renders are directly comparable.
fn render(def: &SynthDef) -> Vec<Vec<f32>> {
    let mut engine = Engine::new(EngineConfig::default());
    let synth = engine.instantiate_synthdef(def);
    engine.graph_mut().set_sink(synth.output_node());
    if def.param_names().iter().any(|(n, _, _)| n == "gate") {
        engine.set_param(&synth, "gate", 1.0);
    }
    engine.prepare();
    engine.render_offline(16)
}

/// Bit-for-bit equality of two renders (compares f32 bit patterns, so NaN
/// payloads and signed zero must match too).
fn assert_render_identical(a: &[Vec<f32>], b: &[Vec<f32>], case: &str) {
    assert_eq!(a.len(), b.len(), "{case}: channel count differs");
    for (ch, (ca, cb)) in a.iter().zip(b).enumerate() {
        assert_eq!(ca.len(), cb.len(), "{case}: ch {ch} length differs");
        for (i, (x, y)) in ca.iter().zip(cb).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{case}: ch {ch} sample {i} differs: {x} vs {y}"
            );
        }
    }
}

/// The corpus: exercises every IrNode kind — literals, all four binops, neg,
/// params, zero-arg UGens, multi-arg UGens (osc/filter/env/physical/pan),
/// nested lets, and precedence/grouping.
const CORPUS: &[&str] = &[
    "synthdef test = 42.0",
    "synthdef test x=1.0 = x * 2.0",
    "synthdef test = 1.0 + 2.0 * 3.0",
    "synthdef test = (1.0 + 2.0) * 3.0",
    "synthdef test = -42.0",
    "synthdef test = -5.0 + 8.0",
    "synthdef test = 10.0 - 3.0",
    "synthdef test = 12.0 / 4.0",
    "synthdef test = sinOsc 440.0 0.0",
    "synthdef test = sinOsc 440.0 * 0.5",
    "synthdef test freq=440.0 amp=0.5 = sinOsc freq 0.0 * amp",
    "synthdef test = whiteNoise * 0.3",
    "synthdef test = saw 220.0",
    "synthdef test = lpf (saw 220.0) 800.0 1.0",
    "synthdef test = perc 0.01 0.5 * sinOsc 440.0 0.0",
    "synthdef test gate=1.0 freq=440.0 = adsr gate 0.01 0.1 0.7 0.3 * sinOsc freq 0.0",
    "synthdef test = pluck 220.0 0.5 1.0",
    "synthdef test = pan2 (sinOsc 440.0 0.0) 0.0",
    "synthdef test x=3.0 =\n  let a = x * 2.0\n  let b = x + 1.0\n  a + b",
    "synthdef test = let x = 3.0; y = 4.0 in x + y",
];

#[test]
fn dsl_ir_compile_renders_byte_identical_to_direct_compile() {
    let reg = registry();
    for src in CORPUS {
        let decl = parse_one(src);

        let direct =
            compile_synthdef(&decl, &reg).unwrap_or_else(|e| panic!("compile {src:?}: {e}"));

        let ir = from_decl(&decl, &reg);
        ir.validate(&reg)
            .unwrap_or_else(|e| panic!("validate {src:?}: {e}"));
        let via_ir = ir
            .compile(&reg)
            .unwrap_or_else(|e| panic!("ir.compile {src:?}: {e}"));

        assert_render_identical(&render(&direct), &render(&via_ir), src);
    }
}

#[test]
fn binary_and_json_round_trip_over_corpus() {
    let reg = registry();
    for src in CORPUS {
        let decl = parse_one(src);
        let ir = from_decl(&decl, &reg);

        // Binary round-trip is exact.
        let bytes = ir.to_bytes();
        let from_bin =
            IrSynthDef::from_bytes(&bytes).unwrap_or_else(|e| panic!("from_bytes {src:?}: {e}"));
        assert_eq!(ir, from_bin, "binary round-trip differs for {src:?}");

        // JSON round-trip is exact.
        let json = ir.to_json();
        let from_json = IrSynthDef::from_json(&json)
            .unwrap_or_else(|e| panic!("from_json {src:?}: {e}\njson: {json}"));
        assert_eq!(ir, from_json, "json round-trip differs for {src:?}");
    }
}

#[test]
fn decoded_ir_renders_byte_identical() {
    // The full seam: DSL -> IR -> bytes -> IR -> SynthDef must match DSL -> SynthDef.
    let reg = registry();
    for src in CORPUS {
        let decl = parse_one(src);
        let direct = compile_synthdef(&decl, &reg).unwrap();

        let ir = from_decl(&decl, &reg);
        let reloaded = IrSynthDef::from_bytes(&ir.to_bytes()).unwrap();
        reloaded
            .validate(&reg)
            .unwrap_or_else(|e| panic!("validate {src:?}: {e}"));
        let via_ir = reloaded.compile(&reg).unwrap();

        assert_render_identical(&render(&direct), &render(&via_ir), src);
    }
}

#[test]
fn content_hash_is_stable_and_value_sensitive() {
    let reg = registry();
    let ir_a = from_decl(
        &parse_one("synthdef test freq=440.0 = sinOsc freq 0.0"),
        &reg,
    );
    let ir_a2 = from_decl(
        &parse_one("synthdef test freq=440.0 = sinOsc freq 0.0"),
        &reg,
    );
    // Only the param default differs (a "param nudge").
    let ir_b = from_decl(
        &parse_one("synthdef test freq=330.0 = sinOsc freq 0.0"),
        &reg,
    );
    // Structural difference (extra gain multiply).
    let ir_c = from_decl(
        &parse_one("synthdef test freq=440.0 = sinOsc freq 0.0 * 0.5"),
        &reg,
    );

    // Deterministic.
    assert_eq!(ir_a.content_hash(true), ir_a2.content_hash(true));
    assert_eq!(ir_a.content_hash(false), ir_a2.content_hash(false));

    // A param nudge changes the full hash but NOT the topology-only hash.
    assert_ne!(ir_a.content_hash(true), ir_b.content_hash(true));
    assert_eq!(ir_a.content_hash(false), ir_b.content_hash(false));

    // A structural edit changes both.
    assert_ne!(ir_a.content_hash(true), ir_c.content_hash(true));
    assert_ne!(ir_a.content_hash(false), ir_c.content_hash(false));
}

#[test]
fn from_bytes_rejects_bad_magic() {
    assert!(IrSynthDef::from_bytes(b"not an ir stream at all").is_err());
    assert!(IrSynthDef::from_bytes(b"").is_err());
}

#[test]
fn json_handles_inline_consts_and_effect_class() {
    // Authored IR exercising inline consts + Effect class + audio input, paths
    // the DSL decompiler does not currently produce.
    let reg = registry();
    let ir = IrSynthDef {
        format_version: 1,
        name: "fx".into(),
        class: SynthDefClass::Effect,
        output_channels: 1,
        nodes: vec![
            IrNode::UGen {
                kind: "audioIn".into(),
                consts: vec![],
            },
            IrNode::UGen {
                kind: "lpf".into(),
                consts: vec![(1, 800.0), (2, 1.0)],
            },
        ],
        edges: vec![IrEdge {
            from: 0,
            to: 1,
            to_input: 0,
        }],
        params: vec![],
        audio_inputs: vec![("in".into(), 0)],
        output_node: 1,
    };
    let round = IrSynthDef::from_json(&ir.to_json()).unwrap();
    assert_eq!(ir, round);
    let round_bin = IrSynthDef::from_bytes(&ir.to_bytes()).unwrap();
    assert_eq!(ir, round_bin);
    // And it validates + compiles.
    ir.validate(&reg).unwrap();
    let _ = ir.compile(&reg).unwrap();
}

#[test]
fn decompile_preserves_structure() {
    let reg = registry();
    // sinOsc freq 0.0 * amp, with params freq, amp.
    let decl = parse_one("synthdef test freq=440.0 amp=0.5 = sinOsc freq 0.0 * amp");
    let ir = from_decl(&decl, &reg);

    assert_eq!(ir.name, "test");
    assert_eq!(ir.class, SynthDefClass::Source);
    assert_eq!(ir.params.len(), 2);
    assert!(
        ir.params
            .iter()
            .any(|p| p.name == "freq" && p.default == 440.0)
    );
    assert!(
        ir.params
            .iter()
            .any(|p| p.name == "amp" && p.default == 0.5)
    );

    // Param nodes come first (compiler order), then the body.
    assert!(matches!(&ir.nodes[0], IrNode::Param { name, .. } if name == "freq"));
    assert!(matches!(&ir.nodes[1], IrNode::Param { name, .. } if name == "amp"));
    // Exactly one Mul (the `* amp`) and one SinOsc kind present.
    let kinds: Vec<&str> = ir
        .nodes
        .iter()
        .filter_map(|n| match n {
            IrNode::UGen { kind, .. } => Some(kind.as_str()),
            _ => None,
        })
        .collect();
    assert!(kinds.contains(&"sinOsc"));
    assert!(kinds.contains(&"Mul"));
}

#[test]
fn render_ir_produces_fixed_length_audible_output() {
    let reg = registry();

    // One-shot: a percussive envelope over a tone. No gate param, so it rings
    // out on its own; the helper trims/pads to exactly duration_secs.
    let one_shot = from_decl(
        &parse_one("synthdef test = perc 0.005 0.15 * sinOsc 440.0 0.0"),
        &reg,
    );
    let spec = RenderSpec {
        sample_rate: 16_000.0,
        block_size: 64,
        params: vec![],
        gate_on_secs: 0.0,
        max_tail_secs: 0.5,
        duration_secs: 0.5,
    };
    let out = render_ir(&one_shot, &reg, &spec).unwrap();
    let target = (0.5 * 16_000.0_f32).round() as usize;
    assert_eq!(
        out[0].len(),
        target,
        "output trimmed/padded to exact length"
    );
    assert!(
        out[0].iter().any(|&s| s.abs() > 1e-3),
        "one-shot should be audible"
    );
    // The attack (0.005 s * 16 kHz ≈ 80 samples) produces energy near the start.
    assert!(out[0][..1600].iter().any(|&s| s.abs() > 1e-3));

    // Sustaining: a gated ADSR. The helper holds gate=1 for gate_on_secs, then
    // releases it; output is still fixed-length.
    let gated = from_decl(
        &parse_one(
            "synthdef test gate=1.0 freq=440.0 = adsr gate 0.01 0.05 0.7 0.1 * sinOsc freq 0.0",
        ),
        &reg,
    );
    let gspec = RenderSpec {
        sample_rate: 16_000.0,
        block_size: 64,
        params: vec![("freq".into(), 330.0)],
        gate_on_secs: 0.2,
        max_tail_secs: 0.3,
        duration_secs: 0.4,
    };
    let gout = render_ir(&gated, &reg, &gspec).unwrap();
    assert_eq!(gout[0].len(), (0.4 * 16_000.0_f32).round() as usize);
    assert!(
        gout[0].iter().any(|&s| s.abs() > 1e-3),
        "gated synth should be audible"
    );
}

#[test]
fn validate_rejects_unknown_kind() {
    let reg = registry();
    let ir = IrSynthDef {
        format_version: 1,
        name: "bad".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![IrNode::UGen {
            kind: "notARealUgen".into(),
            consts: vec![],
        }],
        edges: vec![],
        params: vec![],
        audio_inputs: vec![],
        output_node: 0,
    };
    assert!(ir.validate(&reg).is_err());
}

#[test]
fn validate_rejects_cycle() {
    let reg = registry();
    // Two Neg nodes feeding each other: 0 -> 1 -> 0.
    let ir = IrSynthDef {
        format_version: 1,
        name: "cyc".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![
            IrNode::UGen {
                kind: "Neg".into(),
                consts: vec![],
            },
            IrNode::UGen {
                kind: "Neg".into(),
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
                to: 0,
                to_input: 0,
            },
        ],
        params: vec![],
        audio_inputs: vec![],
        output_node: 1,
    };
    assert!(ir.validate(&reg).is_err());
}

#[test]
fn validate_rejects_input_port_out_of_range() {
    let reg = registry();
    // Neg has arity 1; wiring to input port 1 is out of range.
    let ir = IrSynthDef {
        format_version: 1,
        name: "oob".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![
            IrNode::Const(1.0),
            IrNode::UGen {
                kind: "Neg".into(),
                consts: vec![],
            },
        ],
        edges: vec![IrEdge {
            from: 0,
            to: 1,
            to_input: 1,
        }],
        params: vec![],
        audio_inputs: vec![],
        output_node: 1,
    };
    assert!(ir.validate(&reg).is_err());
}
