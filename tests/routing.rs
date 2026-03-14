use microsynth::*;
use microsynth::dsl::{self, UGenRegistry};
use microsynth::ugens;

// -- Test UGens for routing tests -------------------------------------------

/// Constant-value generator for testing.
struct ConstGen {
    value: f32,
}

impl ConstGen {
    fn new(value: f32) -> Self {
        ConstGen { value }
    }
}

impl UGen for ConstGen {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "ConstGen",
            inputs: &[],
            outputs: &[OutputSpec {
                name: "out",
                rate: Rate::Audio,
            }],
        }
    }

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn process(
        &mut self,
        _context: &ProcessContext,
        _inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        for ch in 0..output.num_channels() {
            output.channel_mut(ch).fill(self.value);
        }
    }
}

// -- AudioIn UGen tests -----------------------------------------------------

#[test]
fn test_audio_in_pass_through() {
    // ConstGen(0.5) -> AudioIn -> sink
    // AudioIn should pass audio through unchanged
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 64);

    let src = graph.add_node(Box::new(ConstGen::new(0.5)));
    let audio_in = graph.add_node(Box::new(ugens::AudioIn));

    graph.connect(src, audio_in, 0);
    graph.set_sink(audio_in);
    graph.prepare(&ctx);

    let output = graph.render(&ctx).expect("should produce output");
    for &s in output.channel(0).samples() {
        assert!((s - 0.5).abs() < 1e-6, "expected 0.5, got {s}");
    }
}

#[test]
fn test_audio_in_silence_when_disconnected() {
    // AudioIn with no input should produce silence
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 64);

    let audio_in = graph.add_node(Box::new(ugens::AudioIn));
    graph.set_sink(audio_in);
    graph.prepare(&ctx);

    let output = graph.render(&ctx).expect("should produce output");
    for &s in output.channel(0).samples() {
        assert!((s - 0.0).abs() < 1e-6, "expected silence, got {s}");
    }
}

// -- Bus channel count tests ------------------------------------------------

#[test]
fn test_bus_fixed_channel_count() {
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 64);

    // Create a 4-channel bus
    let bus = graph.add_node(Box::new(ugens::Bus::new(4)));
    graph.set_sink(bus);
    graph.prepare(&ctx);

    let output = graph.render(&ctx).expect("should produce output");
    assert_eq!(output.num_channels(), 4, "bus should have 4 channels");
}

// -- Routing graph tests (programmatic API) ---------------------------------

#[test]
fn test_single_effect_chain() {
    // Build: source bus => gain effect (x2) => main bus
    // Voice outputs 0.25, gain doubles it => main output 0.5
    let mut engine = Engine::new(EngineConfig::default());

    // Create effect SynthDef: audioIn * 2.0
    let mut effect_builder = SynthDefBuilder::new("gain_fx");
    let audio_in_idx = effect_builder.add_node(|| Box::new(ugens::AudioIn));
    effect_builder.audio_input("in", audio_in_idx);
    let gain_const = effect_builder.add_node(|| Box::new(ugens::Const::new(2.0)));
    let mul = effect_builder.add_node(|| Box::new(ugens::BinOpUGen::new(ugens::BinOpKind::Mul)));
    effect_builder.connect(audio_in_idx, mul, 0);
    effect_builder.connect(gain_const, mul, 1);
    effect_builder.set_output(mul);
    let effect_def = effect_builder.build();

    // Create source voice SynthDef: constant 0.25
    let mut voice_builder = SynthDefBuilder::new("voice");
    let c = voice_builder.add_node(|| Box::new(ugens::Const::new(0.25)));
    voice_builder.set_output(c);
    let voice_def = voice_builder.build();

    // Build routing: source_bus => gain_fx => main
    let mut routing = RoutingGraph::new();
    let source_bus = routing.add_bus("source", 2);
    routing.add_effect(source_bus, &effect_def, routing.main_bus());

    engine.build_routing(&mut routing, &[effect_def]);

    // Spawn a voice on the source bus
    let _voice_id = engine.spawn_voice_on_routing_bus(&voice_def, &routing, source_bus)
        .expect("should spawn voice");

    engine.prepare();

    let output = engine.render().expect("should render");
    assert_eq!(output.num_channels(), 2, "main bus should be stereo");
    for &s in output.channel(0).samples() {
        assert!(
            (s - 0.5).abs() < 1e-6,
            "expected 0.5 (0.25 * 2.0), got {s}"
        );
    }
}

#[test]
fn test_multi_effect_chain() {
    // source => effect1 (x2) => intermediate => effect2 (x3) => main
    // Voice outputs 1.0 => 2.0 => 6.0
    let mut engine = Engine::new(EngineConfig::default());

    // Effect 1: audioIn * 2.0
    let mut b1 = SynthDefBuilder::new("fx1");
    let ai1 = b1.add_node(|| Box::new(ugens::AudioIn));
    b1.audio_input("in", ai1);
    let c1 = b1.add_node(|| Box::new(ugens::Const::new(2.0)));
    let m1 = b1.add_node(|| Box::new(ugens::BinOpUGen::new(ugens::BinOpKind::Mul)));
    b1.connect(ai1, m1, 0);
    b1.connect(c1, m1, 1);
    b1.set_output(m1);
    let fx1_def = b1.build();

    // Effect 2: audioIn * 3.0
    let mut b2 = SynthDefBuilder::new("fx2");
    let ai2 = b2.add_node(|| Box::new(ugens::AudioIn));
    b2.audio_input("in", ai2);
    let c2 = b2.add_node(|| Box::new(ugens::Const::new(3.0)));
    let m2 = b2.add_node(|| Box::new(ugens::BinOpUGen::new(ugens::BinOpKind::Mul)));
    b2.connect(ai2, m2, 0);
    b2.connect(c2, m2, 1);
    b2.set_output(m2);
    let fx2_def = b2.build();

    // Voice: constant 1.0
    let mut vb = SynthDefBuilder::new("voice");
    let vc = vb.add_node(|| Box::new(ugens::Const::new(1.0)));
    vb.set_output(vc);
    let voice_def = vb.build();

    // Routing: source => fx1 => mid => fx2 => main
    let mut routing = RoutingGraph::new();
    let source = routing.add_bus("source", 2);
    let mid = routing.add_bus("mid", 2);
    routing.add_effect(source, &fx1_def, mid);
    routing.add_effect(mid, &fx2_def, routing.main_bus());

    engine.build_routing(&mut routing, &[fx1_def, fx2_def]);

    let _voice = engine.spawn_voice_on_routing_bus(&voice_def, &routing, source)
        .expect("should spawn voice");
    engine.prepare();

    let output = engine.render().expect("should render");
    for &s in output.channel(0).samples() {
        assert!(
            (s - 6.0).abs() < 1e-6,
            "expected 6.0 (1.0 * 2.0 * 3.0), got {s}"
        );
    }
}

#[test]
fn test_fan_out_parallel_effects() {
    // source bus feeds two effects, both going to main:
    // source => fx_double (x2) => main
    // source => fx_triple (x3) => main
    // Voice outputs 1.0. Main bus sums both: 2.0 + 3.0 = 5.0
    let mut engine = Engine::new(EngineConfig::default());

    // Effect: audioIn * 2.0
    let mut b1 = SynthDefBuilder::new("fx_double");
    let ai1 = b1.add_node(|| Box::new(ugens::AudioIn));
    b1.audio_input("in", ai1);
    let c1 = b1.add_node(|| Box::new(ugens::Const::new(2.0)));
    let m1 = b1.add_node(|| Box::new(ugens::BinOpUGen::new(ugens::BinOpKind::Mul)));
    b1.connect(ai1, m1, 0);
    b1.connect(c1, m1, 1);
    b1.set_output(m1);
    let fx_double = b1.build();

    // Effect: audioIn * 3.0
    let mut b2 = SynthDefBuilder::new("fx_triple");
    let ai2 = b2.add_node(|| Box::new(ugens::AudioIn));
    b2.audio_input("in", ai2);
    let c2 = b2.add_node(|| Box::new(ugens::Const::new(3.0)));
    let m2 = b2.add_node(|| Box::new(ugens::BinOpUGen::new(ugens::BinOpKind::Mul)));
    b2.connect(ai2, m2, 0);
    b2.connect(c2, m2, 1);
    b2.set_output(m2);
    let fx_triple = b2.build();

    // Voice: constant 1.0
    let mut vb = SynthDefBuilder::new("voice");
    let vc = vb.add_node(|| Box::new(ugens::Const::new(1.0)));
    vb.set_output(vc);
    let voice_def = vb.build();

    // Routing: fan-out from source
    let mut routing = RoutingGraph::new();
    let source = routing.add_bus("source", 2);
    routing.add_effect(source, &fx_double, routing.main_bus());
    routing.add_effect(source, &fx_triple, routing.main_bus());

    engine.build_routing(&mut routing, &[fx_double, fx_triple]);

    let _voice = engine.spawn_voice_on_routing_bus(&voice_def, &routing, source)
        .expect("should spawn voice");
    engine.prepare();

    let output = engine.render().expect("should render");
    for &s in output.channel(0).samples() {
        assert!(
            (s - 5.0).abs() < 1e-6,
            "expected 5.0 (2.0 + 3.0), got {s}"
        );
    }
}

// -- DSL integration tests --------------------------------------------------

#[test]
fn test_dsl_effect_synthdef_with_audio_in() {
    let mut reg = UGenRegistry::new();
    ugens::register_builtins(&mut reg);

    let source = r#"
        synthdef myEffect amp=0.5 =
            let sig = audioIn
            sig * amp
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name(), "myEffect");
    // Should have one audio input
    assert_eq!(defs[0].audio_inputs().len(), 1);
    assert_eq!(defs[0].audio_inputs()[0].0, "in");
}

#[test]
fn test_dsl_compile_with_routing() {
    let mut reg = UGenRegistry::new();
    ugens::register_builtins(&mut reg);

    let source = r#"
        synthdef pad freq=440.0 amp=0.5 =
            sinOsc freq 0.0 * amp

        synthdef myFilter cutoff=2000.0 q=1.0 =
            let sig = audioIn
            lpf sig cutoff q

        bus pads 2

        route pads => myFilter => main
    "#;

    let (defs, routing) = dsl::compile_with_routing(source, &reg).unwrap();
    assert_eq!(defs.len(), 2);
    assert_eq!(routing.num_buses(), 2); // "pads" + "main"
    assert_eq!(routing.num_effects(), 1);
    assert!(routing.bus_by_name("pads").is_some());
    assert!(routing.bus_by_name("main").is_some());
}

#[test]
fn test_dsl_routing_fan_out() {
    let mut reg = UGenRegistry::new();
    ugens::register_builtins(&mut reg);

    let source = r#"
        synthdef fx1 =
            audioIn * 0.5

        synthdef fx2 =
            audioIn * 0.3

        bus source 2

        route source => fx1 => main
        route source => fx2 => main
    "#;

    let (defs, routing) = dsl::compile_with_routing(source, &reg).unwrap();
    assert_eq!(defs.len(), 2);
    assert_eq!(routing.num_effects(), 2); // Two fan-out effects
}

#[test]
fn test_dsl_routing_multi_bus_chain() {
    let mut reg = UGenRegistry::new();
    ugens::register_builtins(&mut reg);

    let source = r#"
        synthdef fx1 =
            audioIn * 0.5

        synthdef fx2 =
            audioIn * 0.3

        bus drums 2
        bus reverb 2

        route drums => fx1 => reverb
        route reverb => fx2 => main
    "#;

    let (_, routing) = dsl::compile_with_routing(source, &reg).unwrap();
    assert_eq!(routing.num_buses(), 3); // drums, reverb, main
    assert_eq!(routing.num_effects(), 2);
}

#[test]
fn test_dsl_route_error_unknown_bus() {
    let mut reg = UGenRegistry::new();
    ugens::register_builtins(&mut reg);

    let source = r#"
        synthdef fx =
            audioIn * 0.5

        route nonexistent => fx => main
    "#;

    match dsl::compile_with_routing(source, &reg) {
        Err(e) => {
            let err = e.to_string();
            assert!(err.contains("unknown bus"), "got: {err}");
        }
        Ok(_) => panic!("expected error for unknown bus"),
    }
}

#[test]
fn test_dsl_route_error_unknown_effect() {
    let mut reg = UGenRegistry::new();
    ugens::register_builtins(&mut reg);

    let source = r#"
        bus source 2

        route source => nonexistent => main
    "#;

    match dsl::compile_with_routing(source, &reg) {
        Err(e) => {
            let err = e.to_string();
            assert!(err.contains("unknown effect"), "got: {err}");
        }
        Ok(_) => panic!("expected error for unknown effect"),
    }
}

// -- Full end-to-end routing test -------------------------------------------

#[test]
fn test_end_to_end_dsl_routing() {
    let mut reg = UGenRegistry::new();
    ugens::register_builtins(&mut reg);

    let source = r#"
        synthdef voice amp=1.0 =
            amp

        synthdef halfGain =
            audioIn * 0.5

        bus voices 2

        route voices => halfGain => main
    "#;

    let (defs, mut routing) = dsl::compile_with_routing(source, &reg).unwrap();

    let mut engine = Engine::new(EngineConfig::default());
    engine.build_routing(&mut routing, &defs);

    // Spawn a voice outputting 1.0
    let voice_def = defs.iter().find(|d| d.name() == "voice").unwrap();
    let voices_bus = routing.bus_by_name("voices").unwrap();
    let _v = engine.spawn_voice_on_routing_bus(voice_def, &routing, voices_bus)
        .expect("should spawn voice");

    engine.prepare();
    let output = engine.render().expect("should render");

    // Voice outputs 1.0, halfGain multiplies by 0.5
    for &s in output.channel(0).samples() {
        assert!(
            (s - 0.5).abs() < 1e-6,
            "expected 0.5, got {s}"
        );
    }
}

#[test]
fn test_effect_param_control() {
    let mut engine = Engine::new(EngineConfig::default());

    // Effect: audioIn * amp (where amp is a parameter)
    let mut eb = SynthDefBuilder::new("gain_fx");
    let ai = eb.add_node(|| Box::new(ugens::AudioIn));
    eb.audio_input("in", ai);
    let amp = eb.add_node(|| Box::new(ugens::Param::new(1.0)));
    eb.param("amp", amp, 0);
    let mul = eb.add_node(|| Box::new(ugens::BinOpUGen::new(ugens::BinOpKind::Mul)));
    eb.connect(ai, mul, 0);
    eb.connect(amp, mul, 1);
    eb.set_output(mul);
    let effect_def = eb.build();

    // Voice: constant 2.0
    let mut vb = SynthDefBuilder::new("voice");
    let vc = vb.add_node(|| Box::new(ugens::Const::new(2.0)));
    vb.set_output(vc);
    let voice_def = vb.build();

    // Routing
    let mut routing = RoutingGraph::new();
    let source = routing.add_bus("source", 2);
    let effect_id = routing.add_effect(source, &effect_def, routing.main_bus());

    engine.build_routing(&mut routing, &[effect_def]);
    let _v = engine.spawn_voice_on_routing_bus(&voice_def, &routing, source).unwrap();
    engine.prepare();

    // Render with amp=1.0 (default)
    let output = engine.render().expect("should render");
    assert!(
        (output.channel(0).samples()[0] - 2.0).abs() < 1e-6,
        "expected 2.0 with amp=1.0"
    );

    // Change effect param to 0.5
    assert!(engine.set_effect_param(&routing, effect_id, "amp", 0.5));

    let output = engine.render().expect("should render");
    assert!(
        (output.channel(0).samples()[0] - 1.0).abs() < 1e-6,
        "expected 1.0 with amp=0.5"
    );
}

// -- Lexer tests for new tokens ---------------------------------------------

#[test]
fn test_lexer_fat_arrow() {
    use microsynth::dsl::lexer::{tokenize, Token};
    let tokens = tokenize("a => b").unwrap();
    let kinds: Vec<_> = tokens.iter().map(|t| &t.token).collect();
    assert!(matches!(kinds[0], Token::Ident(s) if s == "a"));
    assert!(matches!(kinds[1], Token::FatArrow));
    assert!(matches!(kinds[2], Token::Ident(s) if s == "b"));
}

#[test]
fn test_lexer_bus_route_keywords() {
    use microsynth::dsl::lexer::{tokenize, Token};
    let tokens = tokenize("bus drums 2").unwrap();
    assert!(matches!(tokens[0].token, Token::Bus));

    let tokens = tokenize("route a => b => c").unwrap();
    assert!(matches!(tokens[0].token, Token::Route));
}

#[test]
fn test_lexer_eq_vs_fat_arrow() {
    use microsynth::dsl::lexer::{tokenize, Token};
    // Ensure = and => are distinguished correctly
    let tokens = tokenize("x = 1.0 a => b").unwrap();
    let kinds: Vec<_> = tokens.iter().map(|t| &t.token).collect();
    // x, =, 1.0, a, =>, b, Eof
    assert!(matches!(kinds[1], Token::Eq));
    assert!(matches!(kinds[4], Token::FatArrow));
}

// -- Voice spawning on multiple routing buses -------------------------------

#[test]
fn test_multiple_voices_on_routing_bus() {
    let mut engine = Engine::new(EngineConfig::default());

    // Pass-through effect
    let mut eb = SynthDefBuilder::new("pass");
    let ai = eb.add_node(|| Box::new(ugens::AudioIn));
    eb.audio_input("in", ai);
    eb.set_output(ai);
    let effect_def = eb.build();

    // Voice: constant value
    let mut vb = SynthDefBuilder::new("voice");
    let amp = vb.add_node(|| Box::new(ugens::Param::new(1.0)));
    vb.param("amp", amp, 0);
    vb.set_output(amp);
    let voice_def = vb.build();

    // Routing
    let mut routing = RoutingGraph::new();
    let source = routing.add_bus("source", 2);
    routing.add_effect(source, &effect_def, routing.main_bus());

    engine.build_routing(&mut routing, &[effect_def]);

    // Spawn 3 voices with different values
    let v1 = engine.spawn_voice_on_routing_bus(&voice_def, &routing, source).unwrap();
    let v2 = engine.spawn_voice_on_routing_bus(&voice_def, &routing, source).unwrap();
    let v3 = engine.spawn_voice_on_routing_bus(&voice_def, &routing, source).unwrap();

    engine.set_voice_param(v1, "amp", 1.0);
    engine.set_voice_param(v2, "amp", 2.0);
    engine.set_voice_param(v3, "amp", 3.0);
    engine.prepare();

    let output = engine.render().expect("should render");
    // Sum of 1.0 + 2.0 + 3.0 = 6.0
    for &s in output.channel(0).samples() {
        assert!(
            (s - 6.0).abs() < 1e-6,
            "expected 6.0 (sum of 3 voices), got {s}"
        );
    }
}
