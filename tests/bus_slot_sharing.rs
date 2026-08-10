//! A bus can receive audio two different ways: a voice spawned directly onto
//! it (`Engine::spawn_voice_on_bus`/`spawn_voice_on_routing_bus`), or a
//! routing effect's output wired onto it (`Engine::build_routing`). These
//! tests pin down that a bus receiving BOTH kinds of connection at once sums
//! both contributions -- neither one silently displaces the other.

use microsynth::*;

#[test]
fn test_direct_voice_and_routing_effect_share_a_target_bus() {
    // Build: source => gain effect (x2) => main, PLUS a second voice spawned
    // directly onto main. Both should reach main's output, summed.
    //
    //   source voice: constant 0.25 --[gain x2]--> main   (contributes 0.5)
    //   direct voice: constant 0.3  ------------->  main  (contributes 0.3)
    //   expected main output: 0.8
    let mut engine = Engine::new(EngineConfig::default());

    let mut effect_builder = SynthDefBuilder::new("gain_fx");
    let audio_in_idx = effect_builder.add_node(|| Box::new(ugens::AudioIn));
    effect_builder.audio_input("in", audio_in_idx);
    let gain_const = effect_builder.add_node(|| Box::new(ugens::Const::new(2.0)));
    let mul = effect_builder.add_node(|| Box::new(ugens::BinOpUGen::new(ugens::BinOpKind::Mul)));
    effect_builder.connect(audio_in_idx, mul, 0);
    effect_builder.connect(gain_const, mul, 1);
    effect_builder.set_output(mul);
    let effect_def = effect_builder.build();

    let mut source_voice_builder = SynthDefBuilder::new("source_voice");
    let sc = source_voice_builder.add_node(|| Box::new(ugens::Const::new(0.25)));
    source_voice_builder.set_output(sc);
    let source_voice_def = source_voice_builder.build();

    let mut direct_voice_builder = SynthDefBuilder::new("direct_voice");
    let dc = direct_voice_builder.add_node(|| Box::new(ugens::Const::new(0.3)));
    direct_voice_builder.set_output(dc);
    let direct_voice_def = direct_voice_builder.build();

    let mut routing = RoutingGraph::new();
    let source_bus = routing.add_bus("source", 2);
    // Registered BEFORE any voice spawns -- build_routing wires this straight
    // onto main's first free input slot, exactly as it always has.
    routing.add_effect(source_bus, &effect_def, routing.main_bus());
    engine.build_routing(&mut routing, &[effect_def]);

    let _source_voice_id = engine
        .spawn_voice_on_routing_bus(&source_voice_def, &routing, source_bus)
        .expect("should spawn the source-bus voice");

    // The bug this test pins down: spawning a voice DIRECTLY onto main (the
    // SAME bus the effect above already targets) must not evict the effect's
    // own connection. Order matters for reproducing the historical bug --
    // main has no voices in its own bookkeeping yet, so a slot search that
    // only consults voice bookkeeping (rather than the graph's live edges)
    // computes the effect's own occupied slot as "free".
    let _direct_voice_id = engine
        .spawn_voice_on_routing_bus(&direct_voice_def, &routing, routing.main_bus())
        .expect("should spawn the direct-on-main voice");

    engine.prepare();

    let output = engine.render().expect("should render");
    for &s in output.channel(0).samples() {
        assert!(
            (s - 0.8).abs() < 1e-6,
            "expected 0.8 (0.25*2.0 from the routed effect + 0.3 from the direct voice), got {s} \
             -- one of the two contributions is missing, which is exactly the silent-eviction \
             failure mode this test exists to catch"
        );
    }
}

#[test]
fn test_direct_voice_onto_main_before_any_routing_effect_is_unaffected() {
    // Sanity/negative control: when NOTHING else targets main, a direct voice
    // spawn still works exactly as before -- this bug is specific to a SHARED
    // target bus, not to `spawn_voice_on_routing_bus`(main) in general.
    let mut engine = Engine::new(EngineConfig::default());

    let mut voice_builder = SynthDefBuilder::new("voice");
    let c = voice_builder.add_node(|| Box::new(ugens::Const::new(0.42)));
    voice_builder.set_output(c);
    let voice_def = voice_builder.build();

    let mut routing = RoutingGraph::new();
    engine.build_routing(&mut routing, &[]);

    let _voice_id = engine
        .spawn_voice_on_routing_bus(&voice_def, &routing, routing.main_bus())
        .expect("should spawn");

    engine.prepare();
    let output = engine.render().expect("should render");
    for &s in output.channel(0).samples() {
        assert!((s - 0.42).abs() < 1e-6, "expected 0.42, got {s}");
    }
}
