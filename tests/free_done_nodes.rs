//! A voice that finishes on its own takes its helper nodes with it:
//! `Engine::free_done_synths` removes a panned voice's `Pan2` and `Const`,
//! and the `StealFade` of a stolen voice whose synth ends before the fade.

use microsynth::dsl;
use microsynth::*;

mod common;
use common::builtin_registry;

const SR: f32 = 44100.0;
const BLOCK: usize = 128;

/// Node ids 0..n. Ids are indices into a graph's node list and the type has
/// no public constructor, so they are taken from a scratch graph.
fn every_id(n: usize) -> Vec<NodeId> {
    let mut scratch = AudioGraph::new();
    (0..n)
        .map(|_| scratch.add_node(Box::new(ugens::Const::new(0.0))))
        .collect()
}

/// The live nodes of the engine's graph, with their UGen names.
fn live_nodes(engine: &Engine) -> Vec<(NodeId, &'static str)> {
    every_id(256)
        .into_iter()
        .filter_map(|id| engine.graph().node_spec(id).map(|s| (id, s.name)))
        .collect()
}

fn ids_named(live: &[(NodeId, &'static str)], name: &str) -> Vec<NodeId> {
    live.iter()
        .filter(|(_, n)| *n == name)
        .map(|(id, _)| *id)
        .collect()
}

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

/// A sine under a 4 ms ramp to zero: done after 176 samples, and the
/// ramp's done reaches the output through the `Mul`.
fn short_voice() -> SynthDef {
    dsl::compile(
        "synthdef s = sinOsc 220.0 0.0 * line 1.0 0.0 0.004",
        &builtin_registry(),
    )
    .unwrap()
    .remove(0)
}

#[test]
fn finished_panned_voice_frees_its_pan2_and_const() {
    let def = short_voice();
    let (mut engine, bus) = engine_with_bus();
    let before = live_nodes(&engine);
    let v = engine.spawn_voice_on_bus_panned(&def, bus, -0.5).unwrap();
    engine.prepare();
    let spawned = live_nodes(&engine);
    let pan2 = ids_named(&spawned, "Pan2");
    let konst: Vec<NodeId> = ids_named(&spawned, "Const")
        .into_iter()
        .filter(|id| !engine.voice_synth(v).unwrap().node_ids().contains(id))
        .collect();
    assert_eq!(pan2.len(), 1, "{spawned:?}");
    assert_eq!(konst.len(), 1, "the pan position, {spawned:?}");

    engine.render().unwrap();
    assert_eq!(engine.free_done_synths(), 0, "still sounding");
    engine.render().unwrap();
    assert_eq!(engine.free_done_synths(), 1, "the ramp has ended");

    assert!(engine.voice_synth(v).is_none());
    assert!(engine.graph().node_spec(pan2[0]).is_none(), "Pan2 freed");
    assert!(engine.graph().node_spec(konst[0]).is_none(), "Const freed");
    assert_eq!(live_nodes(&engine), before);
    assert_eq!(engine.graph().used_input_slots(bus).count(), 0);
}

#[test]
fn stolen_voice_that_finishes_first_frees_its_fade() {
    let def = short_voice();
    let hold = dsl::compile("synthdef dc = 0.25", &builtin_registry())
        .unwrap()
        .remove(0);
    let (mut engine, bus) = engine_with_bus();
    engine.set_voice_allocator(VoiceAllocator::new(1, StealPolicy::Oldest));
    let victim = engine.spawn_voice_on_bus_managed(&def, bus).unwrap();
    engine.prepare();
    engine.render().unwrap();
    assert_eq!(engine.free_done_synths(), 0);

    // The steal starts a 221-sample fade at sample 128; the victim's ramp
    // ends at sample 176, inside the next block, before the fade does.
    let thief = engine.spawn_voice_on_bus_managed(&hold, bus).unwrap();
    engine.prepare();
    let after_steal = live_nodes(&engine);
    let fade = ids_named(&after_steal, "StealFade");
    assert_eq!(fade.len(), 1, "{after_steal:?}");
    let thief_nodes = engine.voice_synth(thief).unwrap().node_ids().to_vec();

    engine.render().unwrap();
    assert!(!engine.graph().node_is_done(fade[0]), "fade still running");
    assert_eq!(engine.free_done_synths(), 1, "the victim's ramp has ended");

    assert!(engine.voice_synth(victim).is_none());
    assert!(engine.graph().node_spec(fade[0]).is_none(), "fade freed");
    // Only the bus and the thief are left.
    let mut left: Vec<NodeId> = live_nodes(&engine).iter().map(|(id, _)| *id).collect();
    left.retain(|id| *id != bus);
    assert_eq!(left, thief_nodes);
    assert_eq!(engine.graph().used_input_slots(bus).count(), 1);
}

#[test]
fn stolen_panned_voice_that_finishes_first_frees_fade_pan2_and_const() {
    let def = short_voice();
    let hold = dsl::compile("synthdef dc = 0.25", &builtin_registry())
        .unwrap()
        .remove(0);
    let (mut engine, bus) = engine_with_bus();
    engine.set_voice_allocator(VoiceAllocator::new(1, StealPolicy::Oldest));
    let before = live_nodes(&engine);
    engine
        .spawn_voice_on_bus_panned_managed(&def, bus, 0.5)
        .unwrap();
    engine.prepare();
    engine.render().unwrap();
    assert_eq!(engine.free_done_synths(), 0);
    let thief = engine.spawn_voice_on_bus_managed(&hold, bus).unwrap();
    engine.prepare();
    let after_steal = live_nodes(&engine);
    let helpers: Vec<NodeId> = ["StealFade", "Pan2"]
        .iter()
        .flat_map(|n| ids_named(&after_steal, n))
        .collect();
    assert_eq!(helpers.len(), 2, "{after_steal:?}");

    engine.render().unwrap();
    assert_eq!(engine.free_done_synths(), 1);
    for id in helpers {
        assert!(engine.graph().node_spec(id).is_none(), "{id:?} freed");
    }
    // Everything the victim added is gone, the pan position's Const too.
    let thief_nodes = engine.voice_synth(thief).unwrap().node_ids().to_vec();
    let mut left = live_nodes(&engine);
    left.retain(|(id, _)| !thief_nodes.contains(id));
    assert_eq!(left, before);
}
