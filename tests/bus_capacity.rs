//! A bus takes 256 inputs: a long-tail sampled instrument (harp, vibraphone)
//! can hold 70+ voices at once, and offline rendering has no reason to refuse
//! the 65th.

use microsynth::dsl;
use microsynth::*;

mod common;
use common::builtin_registry;

fn engine_with_bus() -> (Engine, NodeId) {
    let mut engine = Engine::new(EngineConfig {
        sample_rate: 44100.0,
        block_size: 64,
    });
    let bus = engine
        .graph_mut()
        .add_node(Box::new(ugens::Bus::new(ugens::ChannelCount::Stereo)));
    engine.graph_mut().set_sink(bus);
    engine.prepare();
    (engine, bus)
}

#[test]
fn bus_has_256_input_slots() {
    assert_eq!(ugens::MAX_BUS_INPUTS, 256);
    let bus = ugens::Bus::new(ugens::ChannelCount::Stereo);
    assert_eq!(bus.max_inputs(), 256);
    assert_eq!(bus.spec().inputs.len(), 256);
    assert!(bus.spec().inputs.iter().all(|s| !s.required));
    // The channel count is the output width only, never the slot count.
    assert_eq!(ugens::Bus::new(ugens::ChannelCount::Mono).max_inputs(), 256);
    assert_eq!(
        ugens::Bus::new(ugens::ChannelCount::Custom(8)).max_inputs(),
        256
    );
}

#[test]
fn bus_admits_256_voices_and_refuses_the_257th() {
    let registry = builtin_registry();
    let defs = dsl::compile("synthdef dc = 0.001", &registry).unwrap();
    let (mut engine, bus) = engine_with_bus();

    let ids: Vec<_> = (0..256)
        .map(|i| {
            engine
                .spawn_voice_on_bus(&defs[0], bus)
                .unwrap_or_else(|| panic!("voice {i} refused on a 256-slot bus"))
        })
        .collect();
    assert_eq!(ids.len(), 256);
    assert!(
        engine.spawn_voice_on_bus(&defs[0], bus).is_none(),
        "the 257th voice has no slot"
    );
    engine.prepare();

    // Every one of the 256 reaches the sum.
    let out = engine
        .render()
        .expect("bus is the sink")
        .channel(0)
        .samples()[0];
    assert!(
        (out - 0.256).abs() < 1e-5,
        "256 voices of 0.001 should sum to 0.256, got {out}"
    );

    // Freeing one reopens exactly one slot.
    engine.free_voice(ids[100]);
    assert!(engine.spawn_voice_on_bus(&defs[0], bus).is_some());
    assert!(engine.spawn_voice_on_bus(&defs[0], bus).is_none());
}

#[test]
fn slots_above_64_carry_signal_on_both_channels() {
    // The old 64-slot bus cut off here; a voice in slot 200 must be summed
    // like one in slot 0.
    let registry = builtin_registry();
    let silent = dsl::compile("synthdef z = 0.0", &registry).unwrap();
    let loud = dsl::compile("synthdef one = 0.5", &registry).unwrap();
    let (mut engine, bus) = engine_with_bus();
    for _ in 0..200 {
        engine.spawn_voice_on_bus(&silent[0], bus).unwrap();
    }
    engine.spawn_voice_on_bus(&loud[0], bus).unwrap();
    engine.prepare();
    let out = engine.render().unwrap();
    assert_eq!(out.channel(0).samples()[10], 0.5);
    assert_eq!(out.channel(1).samples()[10], 0.5);
}
