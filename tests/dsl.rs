use microsynth::*;
use microsynth::dsl::{self, UGenRegistry};

// -- Test UGens for DSL tests ------------------------------------------------

/// A simple gain UGen: multiplies input by a factor (set at construction).
struct TestGain {
    factor: f32,
}

impl TestGain {
    fn new(factor: f32) -> Self {
        TestGain { factor }
    }
}

impl UGen for TestGain {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "TestGain",
            inputs: &[InputSpec {
                name: "in",
                rate: Rate::Audio,
            }],
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
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let input = inputs[0];
        for ch in 0..output.num_channels() {
            let in_ch = ch % input.num_channels();
            let in_samples = input.channel(in_ch).samples();
            let out = output.channel_mut(ch).samples_mut();
            for i in 0..out.len() {
                out[i] = in_samples[i] * self.factor;
            }
        }
    }
}

/// Two-input mixer: a + b
struct TestMix;

impl UGen for TestMix {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "TestMix",
            inputs: &[
                InputSpec {
                    name: "a",
                    rate: Rate::Audio,
                },
                InputSpec {
                    name: "b",
                    rate: Rate::Audio,
                },
            ],
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
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let a = inputs[0];
        let b = inputs[1];
        for ch in 0..output.num_channels() {
            let a_ch = ch % a.num_channels();
            let b_ch = ch % b.num_channels();
            let a_s = a.channel(a_ch).samples();
            let b_s = b.channel(b_ch).samples();
            let out = output.channel_mut(ch).samples_mut();
            for i in 0..out.len() {
                out[i] = a_s[i] + b_s[i];
            }
        }
    }
}

fn make_registry() -> UGenRegistry {
    let mut reg = UGenRegistry::new();
    reg.register(
        "testGain",
        || Box::new(TestGain::new(2.0)),
        &[InputSpec {
            name: "in",
            rate: Rate::Audio,
        }],
        &[OutputSpec {
            name: "out",
            rate: Rate::Audio,
        }],
    );
    reg.register(
        "testMix",
        || Box::new(TestMix),
        &[
            InputSpec {
                name: "a",
                rate: Rate::Audio,
            },
            InputSpec {
                name: "b",
                rate: Rate::Audio,
            },
        ],
        &[OutputSpec {
            name: "out",
            rate: Rate::Audio,
        }],
    );
    reg
}

fn render_synthdef(def: &SynthDef) -> f32 {
    let mut engine = Engine::new(EngineConfig::default());
    let synth = engine.instantiate_synthdef(def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let output = engine.render().expect("should produce output");
    output.channel(0).samples()[0]
}

// -- Lexer tests -------------------------------------------------------------

#[test]
fn test_lexer_basic() {
    use microsynth::dsl::lexer::{tokenize, Token};
    let tokens = tokenize("synthdef test x=1.0 = x * 2.0").unwrap();
    let kinds: Vec<_> = tokens.iter().map(|t| &t.token).collect();
    assert!(matches!(kinds[0], Token::SynthDef));
    assert!(matches!(kinds[1], Token::Ident(s) if s == "test"));
    assert!(matches!(kinds[2], Token::Ident(s) if s == "x"));
    assert!(matches!(kinds[3], Token::Eq));
    assert!(matches!(kinds[4], Token::Number(v) if (*v - 1.0).abs() < 1e-6));
    assert!(matches!(kinds[5], Token::Eq));
    assert!(matches!(kinds[6], Token::Ident(s) if s == "x"));
    assert!(matches!(kinds[7], Token::Star));
    assert!(matches!(kinds[8], Token::Number(v) if (*v - 2.0).abs() < 1e-6));
    assert!(matches!(kinds[9], Token::Eof));
}

#[test]
fn test_lexer_comments() {
    use microsynth::dsl::lexer::{tokenize, Token};
    let tokens = tokenize("synthdef test = 1.0 -- this is ignored").unwrap();
    // Should have: SynthDef, Ident(test), Eq, Number(1.0), Eof
    let non_newline: Vec<_> = tokens
        .iter()
        .filter(|t| t.token != Token::Newline)
        .collect();
    assert_eq!(non_newline.len(), 5);
    assert!(matches!(non_newline[4].token, Token::Eof));
}

#[test]
fn test_lexer_newlines() {
    use microsynth::dsl::lexer::{tokenize, Token};
    let tokens = tokenize("a\n\n\nb").unwrap();
    // Multiple newlines collapse to one
    let kinds: Vec<_> = tokens.iter().map(|t| &t.token).collect();
    assert!(matches!(kinds[0], Token::Ident(_)));
    assert!(matches!(kinds[1], Token::Newline));
    assert!(matches!(kinds[2], Token::Ident(_)));
    assert!(matches!(kinds[3], Token::Eof));
}

// -- Parser tests ------------------------------------------------------------

#[test]
fn test_parse_simple_literal() {
    use microsynth::dsl::ast::Expr;
    use microsynth::dsl::lexer::tokenize;
    use microsynth::dsl::parser::Parser;

    let tokens = tokenize("synthdef test = 42.0").unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    assert_eq!(program.defs.len(), 1);
    assert_eq!(program.defs[0].name, "test");
    assert!(program.defs[0].params.is_empty());
    assert!(matches!(program.defs[0].body, Expr::Lit(v) if (v - 42.0).abs() < 1e-6));
}

#[test]
fn test_parse_params() {
    use microsynth::dsl::lexer::tokenize;
    use microsynth::dsl::parser::Parser;

    let tokens = tokenize("synthdef pad freq=440.0 amp=0.5 = freq").unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let def = &program.defs[0];
    assert_eq!(def.params.len(), 2);
    assert_eq!(def.params[0].name, "freq");
    assert!((def.params[0].default - 440.0).abs() < 1e-6);
    assert_eq!(def.params[1].name, "amp");
    assert!((def.params[1].default - 0.5).abs() < 1e-6);
}

#[test]
fn test_parse_binop() {
    use microsynth::dsl::ast::{BinOp, Expr};
    use microsynth::dsl::lexer::tokenize;
    use microsynth::dsl::parser::Parser;

    let tokens = tokenize("synthdef test = 1.0 + 2.0 * 3.0").unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    // Should be 1.0 + (2.0 * 3.0) due to precedence
    match &program.defs[0].body {
        Expr::BinOp(BinOp::Add, lhs, rhs) => {
            assert!(matches!(lhs.as_ref(), Expr::Lit(v) if (*v - 1.0).abs() < 1e-6));
            assert!(matches!(rhs.as_ref(), Expr::BinOp(BinOp::Mul, _, _)));
        }
        other => panic!("expected Add, got {:?}", other),
    }
}

#[test]
fn test_parse_function_application() {
    use microsynth::dsl::ast::Expr;
    use microsynth::dsl::lexer::tokenize;
    use microsynth::dsl::parser::Parser;

    let tokens = tokenize("synthdef test = sinOsc 440.0 0.0").unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    match &program.defs[0].body {
        Expr::App(name, args) => {
            assert_eq!(name, "sinOsc");
            assert_eq!(args.len(), 2);
        }
        other => panic!("expected App, got {:?}", other),
    }
}

#[test]
fn test_parse_app_with_operator() {
    use microsynth::dsl::ast::{BinOp, Expr};
    use microsynth::dsl::lexer::tokenize;
    use microsynth::dsl::parser::Parser;

    // sinOsc 440.0 * 0.5  should be (sinOsc 440.0) * 0.5
    let tokens = tokenize("synthdef test = sinOsc 440.0 * 0.5").unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    match &program.defs[0].body {
        Expr::BinOp(BinOp::Mul, lhs, rhs) => {
            assert!(matches!(lhs.as_ref(), Expr::App(name, args) if name == "sinOsc" && args.len() == 1));
            assert!(matches!(rhs.as_ref(), Expr::Lit(v) if (*v - 0.5).abs() < 1e-6));
        }
        other => panic!("expected Mul(App, Lit), got {:?}", other),
    }
}

#[test]
fn test_parse_let_bindings() {
    use microsynth::dsl::ast::Expr;
    use microsynth::dsl::lexer::tokenize;
    use microsynth::dsl::parser::Parser;

    let source = "synthdef test x=1.0 =\n  let a = x * 2.0\n  let b = x * 3.0\n  a + b";
    let tokens = tokenize(source).unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    match &program.defs[0].body {
        Expr::Let(bindings, _body) => {
            assert_eq!(bindings.len(), 2);
            assert_eq!(bindings[0].name, "a");
            assert_eq!(bindings[1].name, "b");
        }
        other => panic!("expected Let, got {:?}", other),
    }
}

#[test]
fn test_parse_negation() {
    use microsynth::dsl::ast::Expr;
    use microsynth::dsl::lexer::tokenize;
    use microsynth::dsl::parser::Parser;

    let tokens = tokenize("synthdef test = -42.0").unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    // Negation of literal is optimized to Lit(-42.0)
    assert!(matches!(program.defs[0].body, Expr::Lit(v) if (v + 42.0).abs() < 1e-6));
}

#[test]
fn test_parse_parentheses() {
    use microsynth::dsl::ast::{BinOp, Expr};
    use microsynth::dsl::lexer::tokenize;
    use microsynth::dsl::parser::Parser;

    // (1.0 + 2.0) * 3.0
    let tokens = tokenize("synthdef test = (1.0 + 2.0) * 3.0").unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    match &program.defs[0].body {
        Expr::BinOp(BinOp::Mul, lhs, _rhs) => {
            assert!(matches!(lhs.as_ref(), Expr::BinOp(BinOp::Add, _, _)));
        }
        other => panic!("expected Mul(Add, _), got {:?}", other),
    }
}

// -- Compiler tests ----------------------------------------------------------

#[test]
fn test_compile_literal() {
    let reg = UGenRegistry::new();
    let defs = dsl::compile("synthdef test = 42.0", &reg).unwrap();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name(), "test");
    let value = render_synthdef(&defs[0]);
    assert!((value - 42.0).abs() < 1e-6, "expected 42.0, got {value}");
}

#[test]
fn test_compile_arithmetic() {
    let reg = UGenRegistry::new();

    // 3.0 + 4.0 = 7.0
    let defs = dsl::compile("synthdef test = 3.0 + 4.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 7.0).abs() < 1e-6, "expected 7.0, got {value}");

    // 3.0 * 4.0 = 12.0
    let defs = dsl::compile("synthdef test = 3.0 * 4.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 12.0).abs() < 1e-6, "expected 12.0, got {value}");

    // 10.0 - 3.0 = 7.0
    let defs = dsl::compile("synthdef test = 10.0 - 3.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 7.0).abs() < 1e-6, "expected 7.0, got {value}");

    // 12.0 / 4.0 = 3.0
    let defs = dsl::compile("synthdef test = 12.0 / 4.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 3.0).abs() < 1e-6, "expected 3.0, got {value}");
}

#[test]
fn test_compile_precedence() {
    let reg = UGenRegistry::new();
    // 1.0 + 2.0 * 3.0 = 1.0 + 6.0 = 7.0
    let defs = dsl::compile("synthdef test = 1.0 + 2.0 * 3.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 7.0).abs() < 1e-6, "expected 7.0, got {value}");

    // (1.0 + 2.0) * 3.0 = 9.0
    let defs = dsl::compile("synthdef test = (1.0 + 2.0) * 3.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 9.0).abs() < 1e-6, "expected 9.0, got {value}");
}

#[test]
fn test_compile_params() {
    let reg = UGenRegistry::new();
    // param x defaults to 5.0; output is x * 2.0 = 10.0
    let defs = dsl::compile("synthdef test x=5.0 = x * 2.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 10.0).abs() < 1e-6, "expected 10.0, got {value}");
}

#[test]
fn test_compile_let_bindings() {
    let reg = UGenRegistry::new();
    let source = "synthdef test x=3.0 =\n  let a = x * 2.0\n  let b = x + 1.0\n  a + b";
    // x=3, a=6, b=4, output=10
    let defs = dsl::compile(source, &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 10.0).abs() < 1e-6, "expected 10.0, got {value}");
}

#[test]
fn test_compile_negation() {
    let reg = UGenRegistry::new();
    let defs = dsl::compile("synthdef test = -5.0 + 8.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 3.0).abs() < 1e-6, "expected 3.0, got {value}");
}

#[test]
fn test_compile_with_ugen() {
    let reg = make_registry();
    // testGain doubles its input; input is 3.0 → output is 6.0
    let defs = dsl::compile("synthdef test = testGain 3.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 6.0).abs() < 1e-6, "expected 6.0, got {value}");
}

#[test]
fn test_compile_ugen_with_operator() {
    let reg = make_registry();
    // testGain 3.0 * 0.5 = (testGain 3.0) * 0.5 = 6.0 * 0.5 = 3.0
    let defs = dsl::compile("synthdef test = testGain 3.0 * 0.5", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 3.0).abs() < 1e-6, "expected 3.0, got {value}");
}

#[test]
fn test_compile_two_input_ugen() {
    let reg = make_registry();
    // testMix 3.0 4.0 = 3.0 + 4.0 = 7.0
    let defs = dsl::compile("synthdef test = testMix 3.0 4.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 7.0).abs() < 1e-6, "expected 7.0, got {value}");
}

#[test]
fn test_compile_complex_expression() {
    let reg = make_registry();
    let source = r#"
        synthdef test freq=100.0 amp=0.5 =
            let signal = testGain freq
            signal * amp
    "#;
    // testGain 100.0 = 200.0; 200.0 * 0.5 = 100.0
    let defs = dsl::compile(source, &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 100.0).abs() < 1e-6, "expected 100.0, got {value}");
}

#[test]
fn test_compile_multiple_synthdefs() {
    let reg = UGenRegistry::new();
    let source = r#"
        synthdef first = 1.0
        synthdef second = 2.0
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs.len(), 2);
    assert_eq!(defs[0].name(), "first");
    assert_eq!(defs[1].name(), "second");
    assert!((render_synthdef(&defs[0]) - 1.0).abs() < 1e-6);
    assert!((render_synthdef(&defs[1]) - 2.0).abs() < 1e-6);
}

#[test]
fn test_compile_inline_let_in() {
    let reg = UGenRegistry::new();
    let source = "synthdef test = let x = 3.0; y = 4.0 in x + y";
    let defs = dsl::compile(source, &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 7.0).abs() < 1e-6, "expected 7.0, got {value}");
}

#[test]
fn test_compile_comments() {
    let reg = UGenRegistry::new();
    let source = r#"
        -- A simple test
        synthdef test = 42.0 -- the answer
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - 42.0).abs() < 1e-6);
}

#[test]
fn test_compile_negative_param() {
    let reg = UGenRegistry::new();
    let defs = dsl::compile("synthdef test x=-1.0 = x * 5.0", &reg).unwrap();
    let value = render_synthdef(&defs[0]);
    assert!((value - -5.0).abs() < 1e-6, "expected -5.0, got {value}");
}

// -- Zero-arg UGen tests -----------------------------------------------------

#[test]
fn test_compile_zero_arg_ugen_in_let() {
    let mut reg = UGenRegistry::new();
    microsynth::ugens::register_builtins(&mut reg);
    // whiteNoise is a zero-arg UGen used as a variable-like binding
    let source = r#"
        synthdef test amp=0.3 =
            let sig = whiteNoise
            sig * amp
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs[0].name(), "test");
}

#[test]
fn test_compile_hihat_preset() {
    let mut reg = UGenRegistry::new();
    microsynth::ugens::register_builtins(&mut reg);
    let source = r#"
        synthdef hihat amp=0.3 =
            let env = perc 0.001 0.08
            let sig = whiteNoise
            let filt = hpf sig 8000.0 1.0
            filt * amp * env
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs[0].name(), "hihat");
}

#[test]
fn test_compile_filtered_noise_preset() {
    let mut reg = UGenRegistry::new();
    microsynth::ugens::register_builtins(&mut reg);
    let source = r#"
        synthdef wind cutoff=600.0 q=5.0 amp=0.2 gate=1.0 =
            let env = asr gate 0.01 0.5
            let n = whiteNoise
            let filt = bpf n cutoff q
            filt * amp * env
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs[0].name(), "wind");
}

// -- XLine + ExpPerc tests ---------------------------------------------------

fn render_synthdef_samples(def: &SynthDef, num_blocks: usize) -> Vec<f32> {
    let mut engine = Engine::new(EngineConfig::default());
    let synth = engine.instantiate_synthdef(def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let mut samples = Vec::new();
    for _ in 0..num_blocks {
        let output = engine.render().expect("should produce output");
        samples.extend_from_slice(output.channel(0).samples());
    }
    samples
}

#[test]
fn test_xline_compiles_and_decreasing() {
    let mut reg = UGenRegistry::new();
    microsynth::ugens::register_builtins(&mut reg);
    let source = "synthdef test = xLine 1000.0 1.0 0.01";
    let defs = dsl::compile(source, &reg).unwrap();
    let samples = render_synthdef_samples(&defs[0], 8);
    // First sample should be near start value
    assert!(samples[0] > 500.0, "first sample should be near 1000, got {}", samples[0]);
    // Should be monotonically decreasing
    for i in 1..samples.len() {
        assert!(
            samples[i] <= samples[i - 1] + 1e-3,
            "xLine should be monotonically decreasing at sample {}: {} > {}",
            i, samples[i], samples[i - 1]
        );
    }
    // Last samples should be near end value
    let last = samples[samples.len() - 1];
    assert!(last < 100.0, "last sample should be near 1.0, got {last}");
}

#[test]
fn test_expperc_compiles_and_concave() {
    let mut reg = UGenRegistry::new();
    microsynth::ugens::register_builtins(&mut reg);
    // Short attack, moderate release
    let source = "synthdef test = expPerc 0.001 0.05";
    let defs = dsl::compile(source, &reg).unwrap();
    let samples = render_synthdef_samples(&defs[0], 40);

    // Find peak (should be ~1.0 after attack)
    let peak_idx = samples.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap().0;
    assert!((samples[peak_idx] - 1.0).abs() < 0.05, "peak should be ~1.0, got {}", samples[peak_idx]);

    // Check concavity: at midpoint of decay, level should be above 0.5 (exponential stays higher)
    let decay_samples = &samples[peak_idx..];
    if decay_samples.len() > 10 {
        let mid = decay_samples.len() / 2;
        let mid_val = decay_samples[mid];
        // Exponential decay should be above what linear would give (0.5 at midpoint)
        // For exp decay, midpoint value is well above linear midpoint
        assert!(
            mid_val > 0.3,
            "expPerc at decay midpoint should be > 0.3 (concave), got {mid_val}"
        );
    }
}

// -- Error tests -------------------------------------------------------------

#[test]
fn test_error_unknown_ugen() {
    let reg = UGenRegistry::new();
    let result = dsl::compile("synthdef test = unknownUgen 440.0", &reg);
    let err = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("unknown UGen"), "got: {err}");
}

#[test]
fn test_error_undefined_variable() {
    let reg = UGenRegistry::new();
    let result = dsl::compile("synthdef test = x * 2.0", &reg);
    let err = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("undefined variable"), "got: {err}");
}

#[test]
fn test_error_too_many_args() {
    let reg = make_registry();
    // testGain expects 1 input, giving it 3
    let result = dsl::compile("synthdef test = testGain 1.0 2.0 3.0", &reg);
    let err = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error"),
    };
    assert!(err.contains("expects"), "got: {err}");
}
