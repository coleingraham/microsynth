//! Equivalence and back-compat tests for the IR routing container
//! (`src/ir/container.rs`): buses, routes, and effect SynthDefs as one
//! serializable unit.

use microsynth::dsl::UGenRegistry;
use microsynth::ir::{
    IrBus, IrEdge, IrNode, IrRoute, IrRoutingContainer, IrSynthDef, SynthDefClass,
};
use microsynth::{Engine, RoutingGraph, SynthDefBuilder};
use microsynth::{engine::EngineConfig, ugens};

/// Hand-built IR for a simple gain effect: `audioIn * 0.5`. Structurally
/// identical (same node kinds, same order, same edges) to what
/// `direct_gain_half_def` below builds through the Rust API directly, so
/// `IrSynthDef::compile` and the direct `SynthDefBuilder` calls produce
/// synths that behave identically.
fn ir_gain_half_def() -> IrSynthDef {
    IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "groupFx".to_string(),
        class: SynthDefClass::Effect,
        output_channels: 1,
        nodes: alloc_vec_nodes(),
        edges: vec![
            IrEdge {
                from: 0,
                to: 2,
                to_input: 0,
            },
            IrEdge {
                from: 1,
                to: 2,
                to_input: 1,
            },
        ],
        params: vec![],
        audio_inputs: vec![("in".to_string(), 0)],
        output_node: 2,
    }
}

fn alloc_vec_nodes() -> Vec<IrNode> {
    vec![
        IrNode::UGen {
            kind: "audioIn".to_string(),
            consts: vec![],
        },
        IrNode::Const(0.5),
        IrNode::UGen {
            kind: "Mul".to_string(),
            consts: vec![],
        },
    ]
}

/// Direct-API equivalent of `ir_gain_half_def`: `audioIn * 0.5`, built via
/// `SynthDefBuilder` exactly as `tests/routing.rs`'s effect fixtures are.
fn direct_gain_half_def() -> microsynth::synthdef::SynthDef {
    let mut b = SynthDefBuilder::new("groupFx");
    let audio_in = b.add_node(|| Box::new(ugens::AudioIn));
    b.audio_input("in", audio_in);
    let half = b.add_node(|| Box::new(ugens::Const::new(0.5)));
    let mul = b.add_node(|| Box::new(ugens::BinOpUGen::new(ugens::BinOpKind::Mul)));
    b.connect(audio_in, mul, 0);
    b.connect(half, mul, 1);
    b.set_output(mul);
    b.build()
}

/// A constant-1.0 "instrument" voice, identical in both topologies.
fn voice_def() -> microsynth::synthdef::SynthDef {
    let mut b = SynthDefBuilder::new("instrument");
    let c = b.add_node(|| Box::new(ugens::Const::new(1.0)));
    b.set_output(c);
    b.build()
}

fn render_direct() -> Vec<Vec<f32>> {
    let mut engine = Engine::new(EngineConfig::default());
    let mut routing = RoutingGraph::new();
    let group = routing.add_bus("group", 2);
    let fx = direct_gain_half_def();
    routing.add_effect(group, &fx, routing.main_bus());
    engine.build_routing(&mut routing, &[fx]);

    let voice = voice_def();
    engine
        .spawn_voice_on_routing_bus(&voice, &routing, group)
        .expect("should spawn voice");
    engine.prepare();
    engine.render_offline(4)
}

fn render_via_container() -> Vec<Vec<f32>> {
    let mut reg = UGenRegistry::new();
    ugens::register_builtins(&mut reg);

    let container = IrRoutingContainer {
        format_version: microsynth::ir::FORMAT_VERSION,
        buses: vec![IrBus {
            name: "group".to_string(),
            channels: 2,
        }],
        routes: vec![IrRoute {
            source_bus: "group".to_string(),
            effect_def: "groupFx".to_string(),
            target_bus: "main".to_string(),
        }],
        effects: vec![ir_gain_half_def()],
    };

    // Round-trip through the wire bytes — this is the "IR bytes" leg of the
    // equivalence claim, not just an in-memory container.
    let bytes = container.to_bytes();
    let decoded = IrRoutingContainer::from_bytes(&bytes).expect("container should decode");

    let (mut routing, effect_defs) = decoded
        .to_routing_graph(&reg)
        .expect("container should build a routing graph");
    let mut engine = Engine::new(EngineConfig::default());
    engine.build_routing(&mut routing, &effect_defs);

    let voice = voice_def();
    let group = routing.bus_by_name("group").expect("group bus by name");
    engine
        .spawn_voice_on_routing_bus(&voice, &routing, group)
        .expect("should spawn voice");
    engine.prepare();
    engine.render_offline(4)
}

#[test]
fn container_round_trip_renders_identically_to_direct_construction() {
    let direct = render_direct();
    let via_container = render_via_container();

    assert_eq!(direct.len(), via_container.len(), "channel count differs");
    for (ch, (d, c)) in direct.iter().zip(via_container.iter()).enumerate() {
        assert_eq!(d.len(), c.len(), "channel {ch} sample count differs");
        for (i, (&ds, &cs)) in d.iter().zip(c.iter()).enumerate() {
            assert_eq!(
                ds, cs,
                "channel {ch} sample {i}: direct={ds} container={cs}"
            );
        }
        // Sanity: this isn't a vacuous all-zero comparison. Voice outputs
        // 1.0, groupFx halves it -> 0.5.
        assert!(
            d.iter().any(|&s| (s - 0.5).abs() < 1e-6),
            "expected 0.5 somewhere in channel {ch}, got {d:?}"
        );
    }
}

// -- Back-compat: legacy single-def IR must still decode, and must be
//    cleanly rejected by the container decoder rather than misparsed. -------

#[test]
fn legacy_single_def_ir_still_decodes_unchanged() {
    let legacy = ir_gain_half_def();
    let bytes = legacy.to_bytes();
    let decoded = IrSynthDef::from_bytes(&bytes).expect("legacy single-def IR should decode");
    assert_eq!(decoded, legacy);
}

#[test]
fn container_decoder_rejects_legacy_single_def_bytes() {
    // A version-1-shaped single IrSynthDef stream is not a routing
    // container: it carries `serialize::MAGIC` ("MICROSYNTH-IR"), not
    // `container`'s `CONTAINER_MAGIC` ("MSIR-ROUTE"). Feeding it to the
    // container decoder must fail with BadMagic, not succeed by chance or
    // misparse a truncated/garbage container.
    let legacy_bytes = ir_gain_half_def().to_bytes();
    let err = IrRoutingContainer::from_bytes(&legacy_bytes)
        .expect_err("legacy single-def bytes must not decode as a routing container");
    assert_eq!(err, microsynth::ir::IrCodecError::BadMagic);
}

#[test]
fn container_json_round_trips() {
    let container = IrRoutingContainer {
        format_version: microsynth::ir::FORMAT_VERSION,
        buses: vec![
            IrBus {
                name: "drums".to_string(),
                channels: 2,
            },
            IrBus {
                name: "reverb".to_string(),
                channels: 2,
            },
        ],
        routes: vec![
            IrRoute {
                source_bus: "drums".to_string(),
                effect_def: "groupFx".to_string(),
                target_bus: "reverb".to_string(),
            },
            IrRoute {
                source_bus: "reverb".to_string(),
                effect_def: "groupFx".to_string(),
                target_bus: "main".to_string(),
            },
        ],
        effects: vec![ir_gain_half_def()],
    };

    let json = container.to_json();
    let decoded = IrRoutingContainer::from_json(&json).expect("container JSON should decode");
    assert_eq!(decoded, container);
}

#[test]
fn to_routing_graph_rejects_route_to_unknown_bus() {
    let mut reg = UGenRegistry::new();
    ugens::register_builtins(&mut reg);

    let container = IrRoutingContainer {
        format_version: microsynth::ir::FORMAT_VERSION,
        buses: vec![],
        routes: vec![IrRoute {
            source_bus: "nonexistent".to_string(),
            effect_def: "groupFx".to_string(),
            target_bus: "main".to_string(),
        }],
        effects: vec![ir_gain_half_def()],
    };

    // `RoutingGraph`/`SynthDef` aren't `Debug` (they hold boxed UGen
    // factories), so match directly instead of `.expect_err`.
    match container.to_routing_graph(&reg) {
        Err(microsynth::ir::IrRoutingError::UnknownBus { name, .. }) => {
            assert_eq!(name, "nonexistent");
        }
        Err(other) => panic!("expected UnknownBus, got {other:?}"),
        Ok(_) => panic!("route to an undeclared bus must be rejected"),
    }
}

#[test]
fn to_routing_graph_rejects_bus_named_main() {
    let reg = UGenRegistry::new();
    let container = IrRoutingContainer {
        format_version: microsynth::ir::FORMAT_VERSION,
        buses: vec![IrBus {
            name: "main".to_string(),
            channels: 2,
        }],
        routes: vec![],
        effects: vec![],
    };
    match container.to_routing_graph(&reg) {
        Err(microsynth::ir::IrRoutingError::DuplicateBus(name)) => assert_eq!(name, "main"),
        Err(other) => panic!("expected DuplicateBus, got {other:?}"),
        Ok(_) => panic!("declaring \"main\" explicitly must be rejected"),
    }
}
