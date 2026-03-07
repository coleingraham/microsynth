use microsynth::*;

// -- Test UGens ----------------------------------------------------------

/// A constant-value generator. Outputs a fixed value on all samples.
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

/// A simple gain node: multiplies input by a constant.
struct Gain {
    amount: f32,
}

impl Gain {
    fn new(amount: f32) -> Self {
        Gain { amount }
    }
}

impl UGen for Gain {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "Gain",
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
            let out_samples = output.channel_mut(ch).samples_mut();
            for i in 0..out_samples.len() {
                out_samples[i] = in_samples[i] * self.amount;
            }
        }
    }
}

/// Adds two signals together.
struct Add;

impl UGen for Add {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "Add",
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
            let a_samples = a.channel(a_ch).samples();
            let b_samples = b.channel(b_ch).samples();
            let out = output.channel_mut(ch).samples_mut();
            for i in 0..out.len() {
                out[i] = a_samples[i] + b_samples[i];
            }
        }
    }
}

/// Multichannel constant: outputs different values per channel.
struct MultiConst {
    values: Vec<f32>,
}

impl MultiConst {
    fn new(values: Vec<f32>) -> Self {
        MultiConst { values }
    }
}

impl UGen for MultiConst {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "MultiConst",
            inputs: &[],
            outputs: &[OutputSpec {
                name: "out",
                rate: Rate::Audio,
            }],
        }
    }

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn output_channels(&self, _input_channels: &[usize]) -> usize {
        self.values.len()
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        _inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        for ch in 0..output.num_channels() {
            output.channel_mut(ch).fill(self.values[ch]);
        }
    }
}

/// Control-rate constant (1 sample per block).
struct ControlConst {
    value: f32,
}

impl ControlConst {
    fn new(value: f32) -> Self {
        ControlConst { value }
    }
}

impl UGen for ControlConst {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "ControlConst",
            inputs: &[],
            outputs: &[OutputSpec {
                name: "out",
                rate: Rate::Control,
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

// -- Tests ---------------------------------------------------------------

#[test]
fn test_block_basics() {
    let mut block = Block::new(64);
    assert_eq!(block.len(), 64);
    assert!(!block.is_empty());

    // Starts zeroed
    for &s in block.samples() {
        assert_eq!(s, 0.0);
    }

    // Fill
    block.fill(1.0);
    for &s in block.samples() {
        assert_eq!(s, 1.0);
    }

    // Index
    block[0] = 42.0;
    assert_eq!(block[0], 42.0);
    assert_eq!(block[1], 1.0);

    // Clear
    block.clear();
    assert_eq!(block[0], 0.0);
}

#[test]
fn test_audio_buffer() {
    let buf = AudioBuffer::new(2, 64);
    assert_eq!(buf.num_channels(), 2);
    assert_eq!(buf.block_size(), 64);
    assert_eq!(buf.channel(0).len(), 64);
    assert_eq!(buf.channel(1).len(), 64);
}

#[test]
fn test_audio_buffer_resize() {
    let mut buf = AudioBuffer::new(1, 64);
    assert_eq!(buf.num_channels(), 1);

    buf.set_num_channels(4, 64);
    assert_eq!(buf.num_channels(), 4);

    buf.set_num_channels(2, 64);
    assert_eq!(buf.num_channels(), 2);
}

#[test]
fn test_process_context() {
    let ctx = ProcessContext::new(44100.0, 64);
    assert_eq!(ctx.sample_rate, 44100.0);
    assert_eq!(ctx.block_size, 64);
    assert_eq!(ctx.sample_offset, 0);
    assert_eq!(ctx.time_secs(), 0.0);

    assert_eq!(ctx.block_size_for_rate(Rate::Audio), 64);
    assert_eq!(ctx.block_size_for_rate(Rate::Control), 1);
}

#[test]
fn test_context_advance() {
    let mut ctx = ProcessContext::new(44100.0, 64);
    ctx.advance();
    assert_eq!(ctx.sample_offset, 64);
    ctx.advance();
    assert_eq!(ctx.sample_offset, 128);
}

#[test]
fn test_simple_graph_render() {
    // ConstGen(0.5) -> Gain(2.0) -> output
    // Expected: 1.0
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 64);

    let const_node = graph.add_node(Box::new(ConstGen::new(0.5)));
    let gain_node = graph.add_node(Box::new(Gain::new(2.0)));

    graph.connect(const_node, gain_node, 0);
    graph.set_sink(gain_node);
    graph.prepare(&ctx);

    let output = graph.render(&ctx).expect("should produce output");
    assert_eq!(output.num_channels(), 1);
    for &s in output.channel(0).samples() {
        assert!((s - 1.0).abs() < 1e-6, "expected 1.0, got {}", s);
    }
}

#[test]
fn test_add_graph() {
    // ConstGen(0.3) --\
    //                  Add -> output
    // ConstGen(0.7) --/
    // Expected: 1.0
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 64);

    let a = graph.add_node(Box::new(ConstGen::new(0.3)));
    let b = graph.add_node(Box::new(ConstGen::new(0.7)));
    let add = graph.add_node(Box::new(Add));

    graph.connect(a, add, 0);
    graph.connect(b, add, 1);
    graph.set_sink(add);
    graph.prepare(&ctx);

    let output = graph.render(&ctx).expect("should produce output");
    for &s in output.channel(0).samples() {
        assert!((s - 1.0).abs() < 1e-6, "expected 1.0, got {}", s);
    }
}

#[test]
fn test_multichannel_expansion() {
    // MultiConst([1.0, 2.0, 3.0]) -> Gain(0.5) -> output
    // Gain should expand to 3 channels: [0.5, 1.0, 1.5]
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 64);

    let mc = graph.add_node(Box::new(MultiConst::new(vec![1.0, 2.0, 3.0])));
    let gain = graph.add_node(Box::new(Gain::new(0.5)));

    graph.connect(mc, gain, 0);
    graph.set_sink(gain);
    graph.prepare(&ctx);

    let output = graph.render(&ctx).expect("should produce output");
    assert_eq!(output.num_channels(), 3, "should expand to 3 channels");

    let expected = [0.5, 1.0, 1.5];
    for ch in 0..3 {
        for &s in output.channel(ch).samples() {
            assert!(
                (s - expected[ch]).abs() < 1e-6,
                "ch {}: expected {}, got {}",
                ch,
                expected[ch],
                s
            );
        }
    }
}

#[test]
fn test_multichannel_wrapping() {
    // MultiConst([1.0, 2.0]) --\
    //                           Add -> output
    // MultiConst([10.0, 20.0, 30.0]) --/
    //
    // Add expands to 3 channels. Input A wraps: ch0=1.0, ch1=2.0, ch2=1.0 (wraps)
    // Expected: [11.0, 22.0, 31.0]
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 64);

    let a = graph.add_node(Box::new(MultiConst::new(vec![1.0, 2.0])));
    let b = graph.add_node(Box::new(MultiConst::new(vec![10.0, 20.0, 30.0])));
    let add = graph.add_node(Box::new(Add));

    graph.connect(a, add, 0);
    graph.connect(b, add, 1);
    graph.set_sink(add);
    graph.prepare(&ctx);

    let output = graph.render(&ctx).expect("should produce output");
    assert_eq!(output.num_channels(), 3);

    let expected = [11.0, 22.0, 31.0];
    for ch in 0..3 {
        for &s in output.channel(ch).samples() {
            assert!(
                (s - expected[ch]).abs() < 1e-6,
                "ch {}: expected {}, got {}",
                ch,
                expected[ch],
                s
            );
        }
    }
}

#[test]
fn test_control_rate_node() {
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 64);

    let kr = graph.add_node(Box::new(ControlConst::new(0.5)));
    graph.set_sink(kr);
    graph.prepare(&ctx);

    let output = graph.render(&ctx).expect("should produce output");
    // Control rate: 1 sample per block
    assert_eq!(output.channel(0).len(), 1);
    assert!((output.channel(0)[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_engine_render() {
    let mut engine = Engine::new(EngineConfig::default());

    let c = engine.graph_mut().add_node(Box::new(ConstGen::new(0.25)));
    let g = engine.graph_mut().add_node(Box::new(Gain::new(4.0)));
    engine.graph_mut().connect(c, g, 0);
    engine.graph_mut().set_sink(g);
    engine.prepare();

    assert_eq!(engine.sample_offset(), 0);

    let output = engine.render().expect("should render");
    for &s in output.channel(0).samples() {
        assert!((s - 1.0).abs() < 1e-6);
    }

    assert_eq!(engine.sample_offset(), 64);
}

#[test]
fn test_engine_offline_render() {
    let mut engine = Engine::new(EngineConfig::default());

    let c = engine.graph_mut().add_node(Box::new(ConstGen::new(1.0)));
    engine.graph_mut().set_sink(c);
    engine.prepare();

    let output = engine.render_offline(10);
    assert_eq!(output.len(), 1); // 1 channel
    assert_eq!(output[0].len(), 640); // 10 blocks * 64 samples
    for &s in &output[0] {
        assert!((s - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_synthdef_instantiation() {
    let mut builder = SynthDefBuilder::new("test");
    let c = builder.add_node(|| Box::new(ConstGen::new(0.5)));
    let g = builder.add_node(|| Box::new(Gain::new(2.0)));
    builder.connect(c, g, 0);
    builder.set_output(g);
    let def = builder.build();

    assert_eq!(def.name(), "test");
    assert_eq!(def.num_nodes(), 2);
}

#[test]
fn test_graph_node_removal() {
    let mut graph = AudioGraph::new();
    let ctx = ProcessContext::new(44100.0, 64);

    let a = graph.add_node(Box::new(ConstGen::new(1.0)));
    let b = graph.add_node(Box::new(ConstGen::new(2.0)));
    let add = graph.add_node(Box::new(Add));

    graph.connect(a, add, 0);
    graph.connect(b, add, 1);
    graph.set_sink(add);
    graph.prepare(&ctx);

    // Render once
    let output = graph.render(&ctx).unwrap();
    assert!((output.channel(0)[0] - 3.0).abs() < 1e-6);

    // Remove node b, replace with different value
    graph.remove_node(b);
    let c = graph.add_node(Box::new(ConstGen::new(5.0)));
    graph.connect(c, add, 1);
    graph.prepare(&ctx);

    let output = graph.render(&ctx).unwrap();
    assert!((output.channel(0)[0] - 6.0).abs() < 1e-6);
}

#[test]
fn test_render_multiple_blocks() {
    let mut engine = Engine::new(EngineConfig::default());

    let c = engine.graph_mut().add_node(Box::new(ConstGen::new(1.0)));
    engine.graph_mut().set_sink(c);
    engine.prepare();

    // Render multiple blocks, verify time advances
    for i in 0..10u64 {
        assert_eq!(engine.sample_offset(), i * 64);
        let _ = engine.render();
    }
    assert_eq!(engine.sample_offset(), 640);
}
