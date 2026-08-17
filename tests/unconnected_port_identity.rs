use microsynth::ir::{IrError, IrNode, IrSynthDef, SynthDefClass};
use microsynth::ugens;
use microsynth::*;

mod common;
use common::builtin_registry as registry;

/// Regression test for a defect where `AudioGraph::render` compacted away
/// unconnected input ports before handing them to `UGen::process`, so a
/// later port's buffer silently shifted down into an earlier, unconnected
/// port's slot. The gap is built through `AudioGraph::connect` directly: a
/// two-input oscillator with its first port ("freq") deliberately left
/// unconnected and only its second port ("phase") wired to a real signal.
#[test]
fn unconnected_leading_port_does_not_shift_a_later_port_into_its_slot() {
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 8);

    // A phase offset of 2.5 radians on a freshly-inited oscillator (phase
    // accumulator starts at 0.0) makes the very first output sample
    // sin(2.5) ~= 0.598472 if it lands on the "phase" port as intended. If
    // the 2.5 value is instead misrouted onto the (unconnected) "freq"
    // port -- the bug this test catches -- the first sample stays at
    // sin(0.0) == 0.0: freq's own default (440.0) never gets used because
    // the connected buffer overwrote it, and "phase" falls back to its
    // 0.0 default because it appears unconnected once port 0 was dropped.
    let phase_src = graph.add_node(Box::new(ugens::Const::new(2.5)));
    let osc = graph.add_node(Box::new(ugens::SinOsc::new()));

    // Deliberately do NOT connect port 0 (freq). Connect only port 1 (phase).
    graph.connect(phase_src, osc, 1);
    graph.set_sink(osc);
    graph.prepare(&ctx);

    let output = graph.render(&ctx).expect("graph should render");
    let first_sample = output.channel(0).samples()[0];

    let expected = 2.5f32.sin();
    assert!(
        (first_sample - expected).abs() < 1e-4,
        "expected first sample ~= sin(2.5) = {expected}, got {first_sample} -- \
         this is the port-identity bug if it's 0.0: the phase signal connected \
         to port 1 shifted down into unconnected port 0 (freq) instead"
    );
}

// -- Required-port fail-loud coverage ---------------------------------------
//
// Restoring port identity (above) is not enough on its own: a fix that
// resolved every unconnected port to a shared silent/zero buffer would also
// have "fixed" the shifting, but it would have silently changed the meaning
// of every currently-unconnected *required* port from "construction error"
// to "connected to silence". These two tests pin the fail-loud behavior at
// both of this crate's construction boundaries: `AudioGraph::prepare` (a
// native, already-materialized graph -- panics, since this is a programmer
// error in trusted code) and `IrSynthDef::validate` (an IR document, which
// may come from an untrusted source such as the wasm host -- returns a
// `Result::Err` instead of panicking).

/// `AudioGraph::prepare` must panic, not silently render, when a `required`
/// input port (here: `Neg`'s single "in" port) has no connected source.
#[test]
#[should_panic(expected = "required input port")]
fn prepare_panics_on_an_unconnected_required_port() {
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 8);

    let neg = graph.add_node(Box::new(ugens::NegUGen));
    graph.set_sink(neg);
    // Port 0 ("in") is required and never connected.
    graph.prepare(&ctx);
}

/// The same defect, caught before `AudioGraph::prepare` at the IR
/// (compile-time) boundary: a `Neg` node with no edge and no inline const
/// on its required port fails `validate` with `RequiredInputUnconnected`
/// rather than compiling into a graph that would panic later.
#[test]
fn ir_validate_rejects_an_unconnected_required_port() {
    let reg = registry();
    let ir = IrSynthDef {
        format_version: 1,
        name: "gap".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![IrNode::UGen {
            kind: "Neg".into(),
            consts: vec![],
        }],
        edges: vec![],
        params: vec![],
        audio_inputs: vec![],
        table_bindings: vec![],
        output_node: 0,
    };
    match ir.validate(&reg) {
        Err(IrError::RequiredInputUnconnected { node: 0, input: 0 }) => {}
        other => panic!("expected RequiredInputUnconnected{{node: 0, input: 0}}, got {other:?}"),
    }
}

/// An inline const on a required port satisfies it just as an edge would --
/// `validate` must not confuse "no edge" with "unconnected" (inline consts
/// are materialized as real `Const` node edges by `compile`, so they are
/// unconnected in exactly the same sense as an edge is).
#[test]
fn ir_validate_accepts_a_required_port_satisfied_by_an_inline_const() {
    let reg = registry();
    let ir = IrSynthDef {
        format_version: 1,
        name: "const_satisfied".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![IrNode::UGen {
            kind: "Neg".into(),
            consts: vec![(0, 3.5)],
        }],
        edges: vec![],
        params: vec![],
        audio_inputs: vec![],
        table_bindings: vec![],
        output_node: 0,
    };
    assert!(
        ir.validate(&reg).is_ok(),
        "an inline const on the required port should satisfy it"
    );
}
