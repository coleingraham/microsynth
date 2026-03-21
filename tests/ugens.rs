//! Integration tests for the built-in audio UGens.

use microsynth::*;
use microsynth::ugens::*;

/// Helper: build a simple graph with a single source UGen as the sink.
fn render_source(ugen: Box<dyn UGen>, num_blocks: usize) -> Vec<Vec<f32>> {
    let mut engine = Engine::new(EngineConfig::default());
    let id = engine.graph_mut().add_node(ugen);
    engine.graph_mut().set_sink(id);
    engine.prepare();
    engine.render_offline(num_blocks)
}

/// Helper: render one block and return the first channel's samples.
fn render_one_block(ugen: Box<dyn UGen>) -> Vec<f32> {
    let output = render_source(ugen, 1);
    output[0].clone()
}

// ============================================================================
// Oscillator tests
// ============================================================================

#[test]
fn test_sinosc_produces_output() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(phase, osc, 1);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render().expect("should produce output");
    let samples = output.channel(0).samples();

    // First sample at phase=0 should be sin(0) = 0
    assert!(samples[0].abs() < 0.01, "first sample should be near 0, got {}", samples[0]);

    // Should have non-zero samples (it's oscillating)
    let max = samples.iter().copied().fold(0.0f32, f32::max);
    assert!(max > 0.5, "sine should reach above 0.5, max was {max}");
}

#[test]
fn test_sinosc_frequency_accuracy() {
    // At 44100 Hz sample rate with freq=44100/64 = 689.0625 Hz,
    // one block of 64 samples = exactly one full cycle.
    let config = EngineConfig { sample_rate: 44100.0, block_size: 64 };
    let freq_val = 44100.0 / 64.0; // exactly 1 cycle per block

    let mut engine = Engine::new(config);
    let freq = engine.graph_mut().add_node(Box::new(Const::new(freq_val)));
    let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(phase, osc, 1);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render().unwrap();
    let samples = output.channel(0).samples();

    // After one full cycle, the last sample should be close to where it started
    // (phase wraps back near 0, so sin should be near 0)
    assert!(
        samples[63].abs() < 0.15,
        "after one cycle, should be near 0, got {}",
        samples[63]
    );
}

#[test]
fn test_saw_range() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let osc = engine.graph_mut().add_node(Box::new(Saw::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render_offline(10);
    for &s in &output[0] {
        assert!(s >= -1.0 && s < 1.0, "saw sample {s} out of range [-1, 1)");
    }
}

#[test]
fn test_saw_first_sample() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let osc = engine.graph_mut().add_node(Box::new(Saw::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render().unwrap();
    // Phase starts at 0, so first sample = 2*0 - 1 = -1
    assert!(
        (output.channel(0).samples()[0] - (-1.0)).abs() < 1e-6,
        "first saw sample should be -1.0"
    );
}

#[test]
fn test_pulse_square_wave() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let width = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let osc = engine.graph_mut().add_node(Box::new(Pulse::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(width, osc, 1);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render_offline(10);
    // All samples should be either +1 or -1
    for &s in &output[0] {
        assert!(
            (s - 1.0).abs() < 1e-6 || (s - (-1.0)).abs() < 1e-6,
            "pulse sample {s} should be +1 or -1"
        );
    }
}

#[test]
fn test_tri_range() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let osc = engine.graph_mut().add_node(Box::new(Tri::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render_offline(10);
    for &s in &output[0] {
        assert!(s >= -1.0 && s <= 1.0, "tri sample {s} out of range [-1, 1]");
    }
}

#[test]
fn test_phasor_range() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let osc = engine.graph_mut().add_node(Box::new(Phasor::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render_offline(10);
    for &s in &output[0] {
        assert!(s >= 0.0 && s < 1.0, "phasor sample {s} out of range [0, 1)");
    }
}

// ============================================================================
// Noise tests
// ============================================================================

#[test]
fn test_white_noise_range() {
    let output = render_source(Box::new(WhiteNoise::new()), 10);
    for &s in &output[0] {
        assert!(s >= -1.0 && s <= 1.0, "white noise sample {s} out of range");
    }
}

#[test]
fn test_white_noise_not_silence() {
    let output = render_source(Box::new(WhiteNoise::new()), 1);
    let nonzero = output[0].iter().filter(|&&s| s.abs() > 0.001).count();
    assert!(nonzero > 10, "white noise should have many non-zero samples");
}

#[test]
fn test_white_noise_deterministic_with_seed() {
    let a = render_one_block(Box::new(WhiteNoise::with_seed(42)));
    let b = render_one_block(Box::new(WhiteNoise::with_seed(42)));
    assert_eq!(a, b, "same seed should produce same output");
}

#[test]
fn test_pink_noise_range() {
    let output = render_source(Box::new(PinkNoise::new()), 10);
    for &s in &output[0] {
        assert!(s >= -2.0 && s <= 2.0, "pink noise sample {s} unexpectedly large");
    }
}

#[test]
fn test_pink_noise_not_silence() {
    let output = render_source(Box::new(PinkNoise::new()), 1);
    let nonzero = output[0].iter().filter(|&&s| s.abs() > 0.001).count();
    assert!(nonzero > 10, "pink noise should have many non-zero samples");
}

// ============================================================================
// Filter tests
// ============================================================================

#[test]
fn test_onepole_smoothing() {
    // Feed a constant 1.0 into OnePole with coeff=0.9 (lowpass).
    // Output should converge toward 1.0 over time.
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let coeff = engine.graph_mut().add_node(Box::new(Const::new(0.9)));
    let filt = engine.graph_mut().add_node(Box::new(OnePole::new()));
    engine.graph_mut().connect(src, filt, 0);
    engine.graph_mut().connect(coeff, filt, 1);
    engine.graph_mut().set_sink(filt);
    engine.prepare();

    let output = engine.render_offline(100);
    let last = *output[0].last().unwrap();
    // After many blocks, should be very close to 1.0
    assert!(
        (last - 1.0).abs() < 0.01,
        "OnePole should converge to 1.0, got {last}"
    );
    // First sample should be much less than 1.0
    assert!(
        output[0][0] < 0.5,
        "first sample should be less than 0.5, got {}",
        output[0][0]
    );
}

#[test]
fn test_biquad_lpf_passes_dc() {
    // A lowpass filter should pass a DC signal (constant value) through unchanged.
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let freq = engine.graph_mut().add_node(Box::new(Const::new(1000.0)));
    let q = engine.graph_mut().add_node(Box::new(Const::new(0.707)));
    let filt = engine.graph_mut().add_node(Box::new(BiquadLPF::new()));
    engine.graph_mut().connect(src, filt, 0);
    engine.graph_mut().connect(freq, filt, 1);
    engine.graph_mut().connect(q, filt, 2);
    engine.graph_mut().set_sink(filt);
    engine.prepare();

    let output = engine.render_offline(100);
    let last = *output[0].last().unwrap();
    assert!(
        (last - 1.0).abs() < 0.01,
        "LPF should pass DC, got {last}"
    );
}

#[test]
fn test_biquad_hpf_blocks_dc() {
    // A highpass filter should block DC (constant signal -> 0).
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let freq = engine.graph_mut().add_node(Box::new(Const::new(1000.0)));
    let q = engine.graph_mut().add_node(Box::new(Const::new(0.707)));
    let filt = engine.graph_mut().add_node(Box::new(BiquadHPF::new()));
    engine.graph_mut().connect(src, filt, 0);
    engine.graph_mut().connect(freq, filt, 1);
    engine.graph_mut().connect(q, filt, 2);
    engine.graph_mut().set_sink(filt);
    engine.prepare();

    let output = engine.render_offline(100);
    let last = *output[0].last().unwrap();
    assert!(
        last.abs() < 0.01,
        "HPF should block DC, got {last}"
    );
}

// ============================================================================
// Envelope tests
// ============================================================================

#[test]
fn test_line_envelope() {
    // Line from 0.0 to 1.0 over a very short duration.
    let mut engine = Engine::new(EngineConfig::default());
    let start = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let end = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let dur = engine.graph_mut().add_node(Box::new(Const::new(0.01))); // 10ms
    let line = engine.graph_mut().add_node(Box::new(Line::new()));
    engine.graph_mut().connect(start, line, 0);
    engine.graph_mut().connect(end, line, 1);
    engine.graph_mut().connect(dur, line, 2);
    engine.graph_mut().set_sink(line);
    engine.prepare();

    let output = engine.render_offline(20);
    // Should start near 0
    assert!(output[0][0].abs() < 0.1, "line should start near 0");
    // Should end at 1.0 after enough time
    let last = *output[0].last().unwrap();
    assert!(
        (last - 1.0).abs() < 0.01,
        "line should reach 1.0, got {last}"
    );
}

#[test]
fn test_line_holds_at_target() {
    // Line from 0.0 to 0.5 over very short time.
    let mut engine = Engine::new(EngineConfig { sample_rate: 44100.0, block_size: 64 });
    let start = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let end = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let dur = engine.graph_mut().add_node(Box::new(Const::new(0.001))); // 1ms
    let line = engine.graph_mut().add_node(Box::new(Line::new()));
    engine.graph_mut().connect(start, line, 0);
    engine.graph_mut().connect(end, line, 1);
    engine.graph_mut().connect(dur, line, 2);
    engine.graph_mut().set_sink(line);
    engine.prepare();

    let output = engine.render_offline(50);
    // After the line completes, all remaining samples should be at 0.5
    let last = *output[0].last().unwrap();
    assert!(
        (last - 0.5).abs() < 0.01,
        "line should hold at 0.5, got {last}"
    );
}

#[test]
fn test_asr_attack_sustain_release() {
    let sr = 44100.0;
    let block_size = 64;
    let mut engine = Engine::new(EngineConfig { sample_rate: sr, block_size });

    // gate: 1.0 for attack+sustain, then 0.0 for release
    // We'll render in two phases: gate on, then gate off
    let gate = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let attack = engine.graph_mut().add_node(Box::new(Const::new(0.005))); // 5ms attack
    let release = engine.graph_mut().add_node(Box::new(Const::new(0.005))); // 5ms release
    let asr = engine.graph_mut().add_node(Box::new(ASR::new()));
    engine.graph_mut().connect(gate, asr, 0);
    engine.graph_mut().connect(attack, asr, 1);
    engine.graph_mut().connect(release, asr, 2);
    engine.graph_mut().set_sink(asr);
    engine.prepare();

    // Render with gate on - should ramp up to 1.0
    let output = engine.render_offline(20);
    let last = *output[0].last().unwrap();
    assert!(
        (last - 1.0).abs() < 0.01,
        "ASR should reach sustain level 1.0, got {last}"
    );

    // First sample should be near 0 (just starting attack)
    assert!(
        output[0][0] < 0.5,
        "ASR should start near 0, got {}",
        output[0][0]
    );
}

// ============================================================================
// Delay tests
// ============================================================================

#[test]
fn test_delay_produces_delayed_signal() {
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let time = engine.graph_mut().add_node(Box::new(Const::new(0.01))); // 10ms delay
    let delay = engine.graph_mut().add_node(Box::new(Delay::new()));
    engine.graph_mut().connect(src, delay, 0);
    engine.graph_mut().connect(time, delay, 1);
    engine.graph_mut().set_sink(delay);
    engine.prepare();

    let output = engine.render_offline(10);
    // First few samples should be 0 (delayed)
    assert!(
        output[0][0].abs() < 0.01,
        "first sample should be ~0 due to delay, got {}",
        output[0][0]
    );
    // Later samples should approach 1.0
    let last = *output[0].last().unwrap();
    assert!(
        (last - 1.0).abs() < 0.01,
        "after delay, should output 1.0, got {last}"
    );
}

// ============================================================================
// Utility tests
// ============================================================================

#[test]
fn test_pan2_center() {
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let pos = engine.graph_mut().add_node(Box::new(Const::new(0.0))); // center
    let pan = engine.graph_mut().add_node(Box::new(Pan2::new()));
    engine.graph_mut().connect(src, pan, 0);
    engine.graph_mut().connect(pos, pan, 1);
    engine.graph_mut().set_sink(pan);
    engine.prepare();

    let output = engine.render().unwrap();
    assert_eq!(output.num_channels(), 2, "Pan2 should produce 2 channels");

    let left = output.channel(0).samples()[0];
    let right = output.channel(1).samples()[0];

    // At center (pos=0), both channels should be equal
    assert!(
        (left - right).abs() < 0.01,
        "center pan: left={left}, right={right} should be equal"
    );
    // Equal power: each should be ~0.707
    assert!(
        (left - core::f32::consts::FRAC_1_SQRT_2).abs() < 0.01,
        "center pan level should be ~0.707, got {left}"
    );
}

#[test]
fn test_pan2_hard_left() {
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let pos = engine.graph_mut().add_node(Box::new(Const::new(-1.0))); // hard left
    let pan = engine.graph_mut().add_node(Box::new(Pan2::new()));
    engine.graph_mut().connect(src, pan, 0);
    engine.graph_mut().connect(pos, pan, 1);
    engine.graph_mut().set_sink(pan);
    engine.prepare();

    let output = engine.render().unwrap();
    let left = output.channel(0).samples()[0];
    let right = output.channel(1).samples()[0];

    assert!(
        (left - 1.0).abs() < 0.01,
        "hard left: left should be ~1.0, got {left}"
    );
    assert!(
        right.abs() < 0.01,
        "hard left: right should be ~0.0, got {right}"
    );
}

#[test]
fn test_pan2_hard_right() {
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let pos = engine.graph_mut().add_node(Box::new(Const::new(1.0))); // hard right
    let pan = engine.graph_mut().add_node(Box::new(Pan2::new()));
    engine.graph_mut().connect(src, pan, 0);
    engine.graph_mut().connect(pos, pan, 1);
    engine.graph_mut().set_sink(pan);
    engine.prepare();

    let output = engine.render().unwrap();
    let left = output.channel(0).samples()[0];
    let right = output.channel(1).samples()[0];

    assert!(
        left.abs() < 0.01,
        "hard right: left should be ~0.0, got {left}"
    );
    assert!(
        (right - 1.0).abs() < 0.01,
        "hard right: right should be ~1.0, got {right}"
    );
}

#[test]
fn test_mix_sums_channels() {
    // Create a 3-channel source (using MultiConst-like approach from graph)
    // and verify Mix sums them to mono.
    // We'll use the direct graph API with a custom multichannel UGen.
    struct ThreeChannel;
    impl UGen for ThreeChannel {
        fn spec(&self) -> UGenSpec {
            UGenSpec {
                name: "ThreeChannel",
                inputs: &[],
                outputs: &[OutputSpec { name: "out", rate: Rate::Audio }],
            }
        }
        fn init(&mut self, _: &ProcessContext) {}
        fn reset(&mut self) {}
        fn output_channels(&self, _: &[usize]) -> usize { 3 }
        fn process(&mut self, _: &ProcessContext, _: &[&AudioBuffer], output: &mut AudioBuffer) {
            output.channel_mut(0).fill(1.0);
            output.channel_mut(1).fill(2.0);
            output.channel_mut(2).fill(3.0);
        }
    }

    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(ThreeChannel));
    let mix = engine.graph_mut().add_node(Box::new(Mix::new()));
    engine.graph_mut().connect(src, mix, 0);
    engine.graph_mut().set_sink(mix);
    engine.prepare();

    let output = engine.render().unwrap();
    assert_eq!(output.num_channels(), 1, "Mix should produce 1 channel");
    let value = output.channel(0).samples()[0];
    assert!(
        (value - 6.0).abs() < 1e-6,
        "Mix of [1, 2, 3] should be 6.0, got {value}"
    );
}

#[test]
fn test_sample_and_hold() {
    // SampleAndHold: when trigger goes high, captures the input.
    // We test by setting up a constant input and toggling trigger.
    struct Ramp { sample_rate: f32 }
    impl UGen for Ramp {
        fn spec(&self) -> UGenSpec {
            UGenSpec {
                name: "Ramp",
                inputs: &[],
                outputs: &[OutputSpec { name: "out", rate: Rate::Audio }],
            }
        }
        fn init(&mut self, ctx: &ProcessContext) { self.sample_rate = ctx.sample_rate; }
        fn reset(&mut self) {}
        fn process(&mut self, ctx: &ProcessContext, _: &[&AudioBuffer], output: &mut AudioBuffer) {
            let out = output.channel_mut(0).samples_mut();
            for (i, sample) in out.iter_mut().enumerate() {
                *sample = (ctx.sample_offset as f32 + i as f32) / self.sample_rate;
            }
        }
    }

    // Use a constant trigger of 1.0 — SH will capture on the first sample
    // (positive-going crossing from 0 initial state) and hold forever.
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Ramp { sample_rate: 44100.0 }));
    let trig = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let sh = engine.graph_mut().add_node(Box::new(SampleAndHold::new()));
    engine.graph_mut().connect(src, sh, 0);
    engine.graph_mut().connect(trig, sh, 1);
    engine.graph_mut().set_sink(sh);
    engine.prepare();

    let output = engine.render_offline(5);
    // All samples should be the same (held value from first trigger)
    let held = output[0][0];
    for &s in &output[0] {
        assert!(
            (s - held).abs() < 1e-6,
            "SH should hold at {held}, got {s}"
        );
    }
}

// ============================================================================
// DSL registry tests
// ============================================================================

#[test]
fn test_register_builtins_and_compile() {
    use microsynth::dsl::{self, UGenRegistry};

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    // sinOsc with a literal freq
    let defs = dsl::compile("synthdef test = sinOsc 440.0 0.0", &reg).unwrap();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name(), "test");
}

#[test]
fn test_dsl_saw_compiles() {
    use microsynth::dsl::{self, UGenRegistry};

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let defs = dsl::compile("synthdef test = saw 440.0", &reg).unwrap();
    assert_eq!(defs[0].name(), "test");
}

#[test]
fn test_dsl_filter_chain() {
    use microsynth::dsl::{self, UGenRegistry};

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let source = r#"
        synthdef filtered freq=440.0 =
            let sig = saw freq
            lpf sig 1000.0 0.707
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs[0].name(), "filtered");

    // Actually render it to verify it doesn't crash
    let mut engine = Engine::new(EngineConfig::default());
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let output = engine.render().expect("filtered synthdef should render");
    assert_eq!(output.num_channels(), 1);
}

#[test]
fn test_dsl_complex_patch() {
    use microsynth::dsl::{self, UGenRegistry};

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let source = r#"
        synthdef pad freq=220.0 amp=0.5 =
            let osc1 = sinOsc freq 0.0
            let osc2 = saw freq
            let mixed = osc1 + osc2
            mixed * amp
    "#;
    let defs = dsl::compile(source, &reg).unwrap();

    let mut engine = Engine::new(EngineConfig::default());
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    let output = engine.render_offline(10);
    assert!(!output.is_empty());
    assert_eq!(output[0].len(), 640);

    // Should have non-trivial output (not all zeros)
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.01, "complex patch should produce non-zero output, max={max}");
}

// ============================================================================
// Multichannel expansion with new UGens
// ============================================================================

#[test]
fn test_oscillator_multichannel_expansion() {
    // A 2-channel frequency source feeding a SinOsc should produce 2 channels.
    struct TwoFreq;
    impl UGen for TwoFreq {
        fn spec(&self) -> UGenSpec {
            UGenSpec {
                name: "TwoFreq",
                inputs: &[],
                outputs: &[OutputSpec { name: "out", rate: Rate::Audio }],
            }
        }
        fn init(&mut self, _: &ProcessContext) {}
        fn reset(&mut self) {}
        fn output_channels(&self, _: &[usize]) -> usize { 2 }
        fn process(&mut self, _: &ProcessContext, _: &[&AudioBuffer], output: &mut AudioBuffer) {
            output.channel_mut(0).fill(440.0);
            output.channel_mut(1).fill(880.0);
        }
    }

    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(TwoFreq));
    let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(phase, osc, 1);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render().unwrap();
    assert_eq!(
        output.num_channels(),
        2,
        "SinOsc should expand to 2 channels with 2-ch freq input"
    );
}

// ============================================================================
// FeedbackDelay tests
// ============================================================================

#[test]
fn test_feedback_delay_produces_echoes() {
    // FeedbackDelay with a short delay and high feedback should produce
    // decaying echoes: output amplitude should decrease over time when
    // input stops after initial impulse.
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let time = engine.graph_mut().add_node(Box::new(Const::new(0.01))); // 10ms
    let fb = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let delay = engine.graph_mut().add_node(Box::new(FeedbackDelay::new()));
    engine.graph_mut().connect(src, delay, 0);
    engine.graph_mut().connect(time, delay, 1);
    engine.graph_mut().connect(fb, delay, 2);
    engine.graph_mut().set_sink(delay);
    engine.prepare();

    let output = engine.render_offline(10);
    // With constant input of 1.0 and feedback 0.5, the output should
    // converge to 1 / (1 - 0.5) = 2.0 after the delay line fills.
    let last = *output[0].last().unwrap();
    assert!(
        last >= 1.5,
        "feedback delay with const input should accumulate to >= 1.5, got {last}"
    );
}

#[test]
fn test_feedback_delay_zero_feedback_matches_delay() {
    // With feedback=0, FeedbackDelay should behave like plain Delay.
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let time = engine.graph_mut().add_node(Box::new(Const::new(0.01)));
    let fb = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let delay = engine.graph_mut().add_node(Box::new(FeedbackDelay::new()));
    engine.graph_mut().connect(src, delay, 0);
    engine.graph_mut().connect(time, delay, 1);
    engine.graph_mut().connect(fb, delay, 2);
    engine.graph_mut().set_sink(delay);
    engine.prepare();

    let output = engine.render_offline(10);
    // With zero feedback and const 1.0 input, output should converge to 1.0
    // (the feedback delay writes input then reads, so first sample = input value)
    let last = *output[0].last().unwrap();
    assert!(
        (last - 1.0).abs() < 0.01,
        "with zero feedback, should output 1.0, got {last}"
    );
}

#[test]
fn test_dsl_feedback_delay_compiles() {
    use microsynth::dsl::{self, UGenRegistry};

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let source = r#"
        synthdef echo freq=440.0 =
            let sig = sinOsc freq 0.0
            feedbackDelay sig 0.25 0.5
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs[0].name(), "echo");
}

// ============================================================================
// Compressor tests
// ============================================================================

#[test]
fn test_compressor_reduces_loud_signal() {
    // A loud signal above threshold should be attenuated.
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0))); // 0 dB
    let sc = engine.graph_mut().add_node(Box::new(Const::new(1.0)));  // sidechain = same
    let thresh = engine.graph_mut().add_node(Box::new(Const::new(-20.0))); // -20 dB threshold
    let ratio = engine.graph_mut().add_node(Box::new(Const::new(4.0)));    // 4:1
    let attack = engine.graph_mut().add_node(Box::new(Const::new(0.001))); // 1ms attack
    let release = engine.graph_mut().add_node(Box::new(Const::new(0.1)));
    let makeup = engine.graph_mut().add_node(Box::new(Const::new(0.0)));   // no makeup
    let comp = engine.graph_mut().add_node(Box::new(Compressor::new()));

    engine.graph_mut().connect(src, comp, 0);
    engine.graph_mut().connect(sc, comp, 1);
    engine.graph_mut().connect(thresh, comp, 2);
    engine.graph_mut().connect(ratio, comp, 3);
    engine.graph_mut().connect(attack, comp, 4);
    engine.graph_mut().connect(release, comp, 5);
    engine.graph_mut().connect(makeup, comp, 6);
    engine.graph_mut().set_sink(comp);
    engine.prepare();

    // Render several blocks so envelope settles
    let output = engine.render_offline(20);
    let last = *output[0].last().unwrap();
    // Input is 1.0 (0 dB), threshold is -20 dB, ratio 4:1
    // Over threshold by 20 dB, gain reduction = 20 * (1 - 1/4) = 15 dB
    // Output should be around -15 dB ≈ 0.178
    assert!(
        last < 0.5,
        "compressor should attenuate loud signal, got {last}"
    );
    assert!(
        last > 0.05,
        "compressor shouldn't silence the signal completely, got {last}"
    );
}

#[test]
fn test_compressor_passes_quiet_signal() {
    // A signal below threshold should pass through uncompressed.
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(0.01))); // very quiet
    let sc = engine.graph_mut().add_node(Box::new(Const::new(0.01)));
    let thresh = engine.graph_mut().add_node(Box::new(Const::new(-6.0)));
    let ratio = engine.graph_mut().add_node(Box::new(Const::new(10.0)));
    let attack = engine.graph_mut().add_node(Box::new(Const::new(0.001)));
    let release = engine.graph_mut().add_node(Box::new(Const::new(0.1)));
    let makeup = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let comp = engine.graph_mut().add_node(Box::new(Compressor::new()));

    engine.graph_mut().connect(src, comp, 0);
    engine.graph_mut().connect(sc, comp, 1);
    engine.graph_mut().connect(thresh, comp, 2);
    engine.graph_mut().connect(ratio, comp, 3);
    engine.graph_mut().connect(attack, comp, 4);
    engine.graph_mut().connect(release, comp, 5);
    engine.graph_mut().connect(makeup, comp, 6);
    engine.graph_mut().set_sink(comp);
    engine.prepare();

    let output = engine.render_offline(20);
    let last = *output[0].last().unwrap();
    // 0.01 is ~-40 dB, well below -6 dB threshold. Should pass through.
    assert!(
        (last - 0.01).abs() < 0.005,
        "quiet signal should pass through uncompressed, got {last}"
    );
}

#[test]
fn test_compressor_makeup_gain() {
    // Makeup gain should boost the output.
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(0.1))); // -20 dB
    let sc = engine.graph_mut().add_node(Box::new(Const::new(0.1)));
    let thresh = engine.graph_mut().add_node(Box::new(Const::new(-6.0)));  // above signal
    let ratio = engine.graph_mut().add_node(Box::new(Const::new(4.0)));
    let attack = engine.graph_mut().add_node(Box::new(Const::new(0.001)));
    let release = engine.graph_mut().add_node(Box::new(Const::new(0.1)));
    let makeup = engine.graph_mut().add_node(Box::new(Const::new(12.0)));  // +12 dB makeup
    let comp = engine.graph_mut().add_node(Box::new(Compressor::new()));

    engine.graph_mut().connect(src, comp, 0);
    engine.graph_mut().connect(sc, comp, 1);
    engine.graph_mut().connect(thresh, comp, 2);
    engine.graph_mut().connect(ratio, comp, 3);
    engine.graph_mut().connect(attack, comp, 4);
    engine.graph_mut().connect(release, comp, 5);
    engine.graph_mut().connect(makeup, comp, 6);
    engine.graph_mut().set_sink(comp);
    engine.prepare();

    let output = engine.render_offline(20);
    let last = *output[0].last().unwrap();
    // Signal is below threshold, so no compression.
    // Makeup of +12 dB ≈ 4x gain. 0.1 * 4 ≈ 0.4
    assert!(
        last > 0.3,
        "makeup gain should boost output, got {last}"
    );
}

#[test]
fn test_dsl_compressor_compiles() {
    use microsynth::dsl::{self, UGenRegistry};

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let source = r#"
        synthdef compressed freq=440.0 =
            let sig = sinOsc freq 0.0
            compressor sig sig (0.0 - 10.0) 4.0 0.01 0.1 6.0
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs[0].name(), "compressed");

    // Render to verify no crash
    let mut engine = Engine::new(EngineConfig::default());
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let output = engine.render_offline(10);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.0, "compressed synth should produce output");
}

// ============================================================================
// Distortion tests
// ============================================================================

#[test]
fn test_softclip_bounds() {
    // A loud constant (10.0) through SoftClip should be bounded to (-1, 1)
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(10.0)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let sc = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    engine.graph_mut().connect(input, sc, 0);
    engine.graph_mut().connect(drive, sc, 1);
    engine.graph_mut().set_sink(sc);
    engine.prepare();

    let output = engine.render().unwrap();
    let samples = output.channel(0).samples();
    for &s in samples {
        assert!(s >= -1.0 && s <= 1.0, "softclip output should be in [-1,1], got {s}");
    }
    // tanh(10) should be very close to 1
    assert!(samples[0] > 0.99, "tanh(10) should be near 1.0, got {}", samples[0]);
}

#[test]
fn test_softclip_passes_small_signals() {
    // tanh(0.1) ≈ 0.0997 — nearly linear for small signals
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(0.1)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let sc = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    engine.graph_mut().connect(input, sc, 0);
    engine.graph_mut().connect(drive, sc, 1);
    engine.graph_mut().set_sink(sc);
    engine.prepare();

    let output = engine.render().unwrap();
    let s = output.channel(0).samples()[0];
    assert!((s - 0.1).abs() < 0.01, "small signal should pass nearly unchanged, got {s}");
}

#[test]
fn test_softclip_drive_increases_saturation() {
    // Low drive
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let sc = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    engine.graph_mut().connect(input, sc, 0);
    engine.graph_mut().connect(drive, sc, 1);
    engine.graph_mut().set_sink(sc);
    engine.prepare();
    let out_low = engine.render().unwrap().channel(0).samples()[0];

    // High drive
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(10.0)));
    let sc = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    engine.graph_mut().connect(input, sc, 0);
    engine.graph_mut().connect(drive, sc, 1);
    engine.graph_mut().set_sink(sc);
    engine.prepare();
    let out_high = engine.render().unwrap().channel(0).samples()[0];

    assert!(
        out_high > out_low,
        "higher drive should produce more saturation: low={out_low}, high={out_high}"
    );
    assert!(out_high > 0.99, "drive=10 on 0.5 should be near 1.0, got {out_high}");
}

#[test]
fn test_softclip_symmetry() {
    // Positive input
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(0.7)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(3.0)));
    let sc = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    engine.graph_mut().connect(input, sc, 0);
    engine.graph_mut().connect(drive, sc, 1);
    engine.graph_mut().set_sink(sc);
    engine.prepare();
    let pos = engine.render().unwrap().channel(0).samples()[0];

    // Negative input
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(-0.7)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(3.0)));
    let sc = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    engine.graph_mut().connect(input, sc, 0);
    engine.graph_mut().connect(drive, sc, 1);
    engine.graph_mut().set_sink(sc);
    engine.prepare();
    let neg = engine.render().unwrap().channel(0).samples()[0];

    assert!(
        (pos + neg).abs() < 1e-6,
        "softclip should be symmetric: pos={pos}, neg={neg}"
    );
}

#[test]
fn test_overdrive_bounded_output() {
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(5.0)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(10.0)));
    let tone = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let mix = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let od = engine.graph_mut().add_node(Box::new(Overdrive::new()));
    engine.graph_mut().connect(input, od, 0);
    engine.graph_mut().connect(drive, od, 1);
    engine.graph_mut().connect(tone, od, 2);
    engine.graph_mut().connect(mix, od, 3);
    engine.graph_mut().set_sink(od);
    engine.prepare();

    let output = engine.render_offline(4);
    for &s in &output[0] {
        assert!(
            s.abs() <= 1.5,
            "overdrive output should be bounded, got {s}"
        );
    }
}

#[test]
fn test_overdrive_dry_wet_mix() {
    let config = EngineConfig::default();

    // mix = 0.0 should pass input through unchanged
    let mut engine = Engine::new(config);
    let input = engine.graph_mut().add_node(Box::new(Const::new(0.3)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(5.0)));
    let tone = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let mix = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let od = engine.graph_mut().add_node(Box::new(Overdrive::new()));
    engine.graph_mut().connect(input, od, 0);
    engine.graph_mut().connect(drive, od, 1);
    engine.graph_mut().connect(tone, od, 2);
    engine.graph_mut().connect(mix, od, 3);
    engine.graph_mut().set_sink(od);
    engine.prepare();

    let output = engine.render().unwrap();
    let s = output.channel(0).samples()[0];
    assert!(
        (s - 0.3).abs() < 1e-6,
        "mix=0 should pass dry signal, got {s}"
    );
}

#[test]
fn test_overdrive_asymmetry() {
    // Positive input
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(3.0)));
    let tone = engine.graph_mut().add_node(Box::new(Const::new(1.0))); // bright to minimize filter effect
    let mix = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let od = engine.graph_mut().add_node(Box::new(Overdrive::new()));
    engine.graph_mut().connect(input, od, 0);
    engine.graph_mut().connect(drive, od, 1);
    engine.graph_mut().connect(tone, od, 2);
    engine.graph_mut().connect(mix, od, 3);
    engine.graph_mut().set_sink(od);
    engine.prepare();
    // Render several blocks to let the tone filter settle
    let out_pos = engine.render_offline(10);
    let pos = *out_pos[0].last().unwrap();

    // Negative input
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(-0.5)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(3.0)));
    let tone = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let mix = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let od = engine.graph_mut().add_node(Box::new(Overdrive::new()));
    engine.graph_mut().connect(input, od, 0);
    engine.graph_mut().connect(drive, od, 1);
    engine.graph_mut().connect(tone, od, 2);
    engine.graph_mut().connect(mix, od, 3);
    engine.graph_mut().set_sink(od);
    engine.prepare();
    let out_neg = engine.render_offline(10);
    let neg = *out_neg[0].last().unwrap();

    // Asymmetric clipping: |pos| != |neg|
    assert!(
        (pos.abs() - neg.abs()).abs() > 0.01,
        "overdrive should be asymmetric: pos={pos}, neg={neg}"
    );
}

#[test]
fn test_softclip_zero_input() {
    // tanh(0) = 0 regardless of drive
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(100.0)));
    let sc = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    engine.graph_mut().connect(input, sc, 0);
    engine.graph_mut().connect(drive, sc, 1);
    engine.graph_mut().set_sink(sc);
    engine.prepare();

    let output = engine.render().unwrap();
    let s = output.channel(0).samples()[0];
    assert!(s.abs() < 1e-6, "tanh(0) should be 0, got {s}");
}

#[test]
fn test_softclip_zero_drive() {
    // drive=0 means tanh(0 * x) = tanh(0) = 0 for any input
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(0.8)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let sc = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    engine.graph_mut().connect(input, sc, 0);
    engine.graph_mut().connect(drive, sc, 1);
    engine.graph_mut().set_sink(sc);
    engine.prepare();

    let output = engine.render().unwrap();
    let s = output.channel(0).samples()[0];
    assert!(s.abs() < 1e-6, "drive=0 should produce silence, got {s}");
}

#[test]
fn test_softclip_negative_input_bounds() {
    // Large negative input should be bounded to [-1, 0)
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(-10.0)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let sc = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    engine.graph_mut().connect(input, sc, 0);
    engine.graph_mut().connect(drive, sc, 1);
    engine.graph_mut().set_sink(sc);
    engine.prepare();

    let output = engine.render().unwrap();
    let s = output.channel(0).samples()[0];
    assert!(s >= -1.0 && s <= 0.0, "tanh(-10) should be in [-1, 0], got {s}");
    assert!(s < -0.99, "tanh(-10) should be near -1.0, got {s}");
}

#[test]
fn test_softclip_with_oscillator() {
    // SoftClip on a sine oscillator should produce bounded output
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(phase, osc, 1);

    let drive = engine.graph_mut().add_node(Box::new(Const::new(5.0)));
    let sc = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    engine.graph_mut().connect(osc, sc, 0);
    engine.graph_mut().connect(drive, sc, 1);
    engine.graph_mut().set_sink(sc);
    engine.prepare();

    let output = engine.render_offline(10);
    for &s in &output[0] {
        assert!(s >= -1.0 && s <= 1.0, "softclip on oscillator should be bounded, got {s}");
    }
    // Should have non-trivial output
    let max = output[0].iter().copied().fold(0.0f32, f32::max);
    assert!(max > 0.3, "softclip on sine should produce output, max={max}");
}

#[test]
fn test_overdrive_zero_drive() {
    // drive=0 means gained=0 for all input, so clipped=0, tone filter → 0, output=mix*0=0
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(0.8)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let tone = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let mix = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let od = engine.graph_mut().add_node(Box::new(Overdrive::new()));
    engine.graph_mut().connect(input, od, 0);
    engine.graph_mut().connect(drive, od, 1);
    engine.graph_mut().connect(tone, od, 2);
    engine.graph_mut().connect(mix, od, 3);
    engine.graph_mut().set_sink(od);
    engine.prepare();

    let output = engine.render_offline(5);
    let last = *output[0].last().unwrap();
    assert!(last.abs() < 0.01, "drive=0, mix=1 should produce near silence, got {last}");
}

#[test]
fn test_overdrive_tone_dark_vs_bright() {
    // Run a saw through overdrive with high drive to generate harmonics,
    // then compare dark (tone=0) vs bright (tone=1). Dark should have lower RMS.
    let render_with_tone = |tone_val: f32| -> f32 {
        let mut engine = Engine::new(EngineConfig::default());
        let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
        let osc = engine.graph_mut().add_node(Box::new(Saw::new()));
        engine.graph_mut().connect(freq, osc, 0);

        let drive = engine.graph_mut().add_node(Box::new(Const::new(8.0)));
        let tone = engine.graph_mut().add_node(Box::new(Const::new(tone_val)));
        let mix = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
        let od = engine.graph_mut().add_node(Box::new(Overdrive::new()));
        engine.graph_mut().connect(osc, od, 0);
        engine.graph_mut().connect(drive, od, 1);
        engine.graph_mut().connect(tone, od, 2);
        engine.graph_mut().connect(mix, od, 3);
        engine.graph_mut().set_sink(od);
        engine.prepare();

        // Render enough blocks for tone filter to settle
        let output = engine.render_offline(20);
        // Compute RMS of last block
        let samples = &output[0][output[0].len() - 128..];
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    };

    let rms_dark = render_with_tone(0.0);
    let rms_bright = render_with_tone(1.0);

    assert!(
        rms_bright > rms_dark,
        "bright tone should have higher RMS than dark: bright={rms_bright}, dark={rms_dark}"
    );
}

#[test]
fn test_overdrive_half_mix() {
    // mix=0.5 should blend dry and wet equally
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(0.4)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(5.0)));
    let tone = engine.graph_mut().add_node(Box::new(Const::new(1.0))); // bright to minimize filter lag
    let mix = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let od = engine.graph_mut().add_node(Box::new(Overdrive::new()));
    engine.graph_mut().connect(input, od, 0);
    engine.graph_mut().connect(drive, od, 1);
    engine.graph_mut().connect(tone, od, 2);
    engine.graph_mut().connect(mix, od, 3);
    engine.graph_mut().set_sink(od);
    engine.prepare();

    let output = engine.render_offline(10);
    let s = *output[0].last().unwrap();
    // Output should be between dry (0.4) and fully wet value
    // At mix=0.5: out = 0.5 * 0.4 + 0.5 * wet
    // wet is tanh(5*0.4) = tanh(2.0) ≈ 0.964 (positive side)
    // So output ≈ 0.2 + 0.5 * ~0.964 ≈ 0.682
    assert!(s > 0.2 && s < 0.95, "half mix should blend dry and wet, got {s}");
    // Should differ from both pure dry (0.4) and pure wet
    assert!((s - 0.4).abs() > 0.05, "half mix should differ from dry, got {s}");
}

#[test]
fn test_overdrive_negative_clipping_bounded() {
    // Very large negative input: cubic clip on negative side clamps at -1.5,
    // so output of clipping stage is bounded. Verify with settled tone filter.
    let mut engine = Engine::new(EngineConfig::default());
    let input = engine.graph_mut().add_node(Box::new(Const::new(-10.0)));
    let drive = engine.graph_mut().add_node(Box::new(Const::new(10.0)));
    let tone = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let mix = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let od = engine.graph_mut().add_node(Box::new(Overdrive::new()));
    engine.graph_mut().connect(input, od, 0);
    engine.graph_mut().connect(drive, od, 1);
    engine.graph_mut().connect(tone, od, 2);
    engine.graph_mut().connect(mix, od, 3);
    engine.graph_mut().set_sink(od);
    engine.prepare();

    let output = engine.render_offline(20);
    let last = *output[0].last().unwrap();
    // Cubic soft clip of -1.5: -1.5 - (-1.5)^3/3.375 = -1.5 - (-3.375/3.375) = -1.5 + 1.0 = -0.5
    // After tone filter settles to this constant, output should be near -1.0
    assert!(last >= -1.5 && last <= 0.0, "negative clipping should be bounded, got {last}");
}

#[test]
fn test_overdrive_with_oscillator() {
    // Overdrive on a sine should produce non-silent bounded output
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(phase, osc, 1);

    let drive = engine.graph_mut().add_node(Box::new(Const::new(5.0)));
    let tone = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let mix = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let od = engine.graph_mut().add_node(Box::new(Overdrive::new()));
    engine.graph_mut().connect(osc, od, 0);
    engine.graph_mut().connect(drive, od, 1);
    engine.graph_mut().connect(tone, od, 2);
    engine.graph_mut().connect(mix, od, 3);
    engine.graph_mut().set_sink(od);
    engine.prepare();

    let output = engine.render_offline(10);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.1, "overdrive on sine should produce output, max={max}");
    assert!(max <= 1.5, "overdrive output should be bounded, max={max}");
}

#[test]
fn test_dsl_softclip_compiles() {
    use microsynth::dsl::{self, UGenRegistry};

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let source = r#"
        synthdef distorted freq=440.0 =
            let sig = sinOsc freq 0.0
            softClip sig 3.0
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs[0].name(), "distorted");

    // Render to verify no crash
    let mut engine = Engine::new(EngineConfig::default());
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let output = engine.render_offline(10);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.0, "softClip DSL synth should produce output");
}

#[test]
fn test_dsl_overdrive_compiles() {
    use microsynth::dsl::{self, UGenRegistry};

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let source = r#"
        synthdef overdriven freq=440.0 =
            let sig = saw freq
            overdrive sig 5.0 0.5 1.0
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs[0].name(), "overdriven");

    // Render to verify no crash
    let mut engine = Engine::new(EngineConfig::default());
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let output = engine.render_offline(10);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.0, "overdrive DSL synth should produce output");
}

// ============================================================================
// Band-limited oscillator tests
// ============================================================================

#[test]
fn test_blsaw_range() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let osc = engine.graph_mut().add_node(Box::new(BlSaw::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render_offline(10);
    for &s in &output[0] {
        assert!(s >= -1.01 && s <= 1.01, "BlSaw sample {s} out of range");
    }
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.5, "BlSaw should produce significant output, max was {max}");
}

#[test]
fn test_blpulse_square_symmetry() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let width = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let osc = engine.graph_mut().add_node(Box::new(BlPulse::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(width, osc, 1);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render_offline(10);
    let positive = output[0].iter().filter(|&&s| s > 0.0).count();
    let negative = output[0].iter().filter(|&&s| s < 0.0).count();
    let total = positive + negative;
    // With width=0.5, expect roughly equal positive and negative samples
    let ratio = positive as f32 / total as f32;
    assert!(
        ratio > 0.4 && ratio < 0.6,
        "BlPulse with width=0.5 should be ~symmetric, got ratio {ratio}"
    );
}

#[test]
fn test_bltri_range() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let osc = engine.graph_mut().add_node(Box::new(BlTri::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().set_sink(osc);
    engine.prepare();

    let output = engine.render_offline(20);
    for &s in &output[0] {
        assert!(s >= -1.5 && s <= 1.5, "BlTri sample {s} out of range");
    }
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.3, "BlTri should produce significant output, max was {max}");
}

// ============================================================================
// Physical model tests
// ============================================================================

#[test]
fn test_pluck_trigger_produces_output() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let decay = engine.graph_mut().add_node(Box::new(Const::new(0.99)));
    let trig = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let pluck = engine.graph_mut().add_node(Box::new(Pluck::new()));
    engine.graph_mut().connect(freq, pluck, 0);
    engine.graph_mut().connect(decay, pluck, 1);
    engine.graph_mut().connect(trig, pluck, 2);
    engine.graph_mut().set_sink(pluck);
    engine.prepare();

    let output = engine.render_offline(4);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.01, "Pluck should produce output after trigger, max was {max}");
}

#[test]
fn test_pluck_decays_to_silence() {
    let config = EngineConfig { sample_rate: 44100.0, block_size: 64 };
    let mut engine = Engine::new(config);
    // Use Impulse at very low freq to trigger once at the start
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let decay = engine.graph_mut().add_node(Box::new(Const::new(0.95)));
    let trig = engine.graph_mut().add_node(Box::new(Impulse::new()));
    let trig_freq = engine.graph_mut().add_node(Box::new(Const::new(0.1)));
    engine.graph_mut().connect(trig_freq, trig, 0);
    let pluck = engine.graph_mut().add_node(Box::new(Pluck::new()));
    engine.graph_mut().connect(freq, pluck, 0);
    engine.graph_mut().connect(decay, pluck, 1);
    engine.graph_mut().connect(trig, pluck, 2);
    engine.graph_mut().set_sink(pluck);
    engine.prepare();

    // Render many blocks to let it decay
    let output = engine.render_offline(500);
    // Check the last 64 samples are near-silent
    let tail = &output[0][output[0].len() - 64..];
    let tail_max = tail.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        tail_max < 0.01,
        "Pluck should decay to near-silence, tail max was {tail_max}"
    );
}

#[test]
fn test_bowed_produces_output() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(220.0)));
    let pressure = engine.graph_mut().add_node(Box::new(Const::new(0.5)));
    let position = engine.graph_mut().add_node(Box::new(Const::new(0.13)));
    let bowed = engine.graph_mut().add_node(Box::new(Bowed::new()));
    engine.graph_mut().connect(freq, bowed, 0);
    engine.graph_mut().connect(pressure, bowed, 1);
    engine.graph_mut().connect(position, bowed, 2);
    engine.graph_mut().set_sink(bowed);
    engine.prepare();

    // Bowed model needs some blocks to start oscillating
    let output = engine.render_offline(50);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.001, "Bowed should produce output with pressure=0.5, max was {max}");
}

#[test]
fn test_bowed_silence_at_zero_pressure() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(220.0)));
    let pressure = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let position = engine.graph_mut().add_node(Box::new(Const::new(0.13)));
    let bowed = engine.graph_mut().add_node(Box::new(Bowed::new()));
    engine.graph_mut().connect(freq, bowed, 0);
    engine.graph_mut().connect(pressure, bowed, 1);
    engine.graph_mut().connect(position, bowed, 2);
    engine.graph_mut().set_sink(bowed);
    engine.prepare();

    let output = engine.render_offline(10);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max < 0.01, "Bowed should be near-silent with pressure=0, max was {max}");
}

// ============================================================================
// FmOsc feedback tests
// ============================================================================

#[test]
fn test_fmosc_produces_output() {
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let ratio = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let index = engine.graph_mut().add_node(Box::new(Const::new(2.0)));
    let feedback = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let fm = engine.graph_mut().add_node(Box::new(FmOsc::new()));
    engine.graph_mut().connect(freq, fm, 0);
    engine.graph_mut().connect(ratio, fm, 1);
    engine.graph_mut().connect(index, fm, 2);
    engine.graph_mut().connect(feedback, fm, 3);
    engine.graph_mut().set_sink(fm);
    engine.prepare();

    let output = engine.render_offline(10);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.1, "FmOsc should produce audible output, max was {max}");
    assert!(max <= 1.0, "FmOsc output should be in [-1, 1], max was {max}");
}

#[test]
fn test_fmosc_feedback_changes_timbre() {
    // Render with feedback=0 and feedback=0.7, compare RMS — they should differ
    fn render_fm_rms(fb_val: f32) -> f32 {
        let mut engine = Engine::new(EngineConfig::default());
        let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
        let ratio = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
        let index = engine.graph_mut().add_node(Box::new(Const::new(2.0)));
        let feedback = engine.graph_mut().add_node(Box::new(Const::new(fb_val)));
        let fm = engine.graph_mut().add_node(Box::new(FmOsc::new()));
        engine.graph_mut().connect(freq, fm, 0);
        engine.graph_mut().connect(ratio, fm, 1);
        engine.graph_mut().connect(index, fm, 2);
        engine.graph_mut().connect(feedback, fm, 3);
        engine.graph_mut().set_sink(fm);
        engine.prepare();

        let output = engine.render_offline(50);
        let sum_sq: f32 = output[0].iter().map(|s| s * s).sum();
        (sum_sq / output[0].len() as f32).sqrt()
    }

    let rms_no_fb = render_fm_rms(0.0);
    let rms_with_fb = render_fm_rms(0.7);
    let diff = (rms_no_fb - rms_with_fb).abs();
    assert!(
        diff > 0.01,
        "Feedback should change timbre: rms_no_fb={rms_no_fb}, rms_with_fb={rms_with_fb}"
    );
}

// ============================================================================
// BiquadNotch tests
// ============================================================================

#[test]
fn test_notch_passes_dc() {
    // A notch filter should pass DC (it only rejects a narrow band).
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let freq = engine.graph_mut().add_node(Box::new(Const::new(1000.0)));
    let q = engine.graph_mut().add_node(Box::new(Const::new(1.0)));
    let filt = engine.graph_mut().add_node(Box::new(BiquadNotch::new()));
    engine.graph_mut().connect(src, filt, 0);
    engine.graph_mut().connect(freq, filt, 1);
    engine.graph_mut().connect(q, filt, 2);
    engine.graph_mut().set_sink(filt);
    engine.prepare();

    let output = engine.render_offline(100);
    let last = *output[0].last().unwrap();
    assert!(
        (last - 1.0).abs() < 0.01,
        "Notch should pass DC, got {last}"
    );
}

#[test]
fn test_notch_attenuates_at_center() {
    // Feed a sine at the notch frequency — it should be heavily attenuated.
    let notch_freq = 1000.0;
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(notch_freq)));
    let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(phase, osc, 1);

    let filt_freq = engine.graph_mut().add_node(Box::new(Const::new(notch_freq)));
    let q = engine.graph_mut().add_node(Box::new(Const::new(10.0))); // narrow notch
    let filt = engine.graph_mut().add_node(Box::new(BiquadNotch::new()));
    engine.graph_mut().connect(osc, filt, 0);
    engine.graph_mut().connect(filt_freq, filt, 1);
    engine.graph_mut().connect(q, filt, 2);
    engine.graph_mut().set_sink(filt);
    engine.prepare();

    // Let the filter settle then check output level
    let output = engine.render_offline(200);
    let tail: Vec<f32> = output[0].iter().rev().take(64).copied().collect();
    let rms: f32 = (tail.iter().map(|s| s * s).sum::<f32>() / tail.len() as f32).sqrt();
    assert!(
        rms < 0.1,
        "Notch should heavily attenuate sine at center freq, rms was {rms}"
    );
}

// ============================================================================
// AllpassFilter tests
// ============================================================================

#[test]
fn test_allpass_unity_gain() {
    // An allpass filter should pass all frequencies at unity gain.
    // Feed a DC signal; after settling, output should equal input.
    let mut engine = Engine::new(EngineConfig::default());
    let src = engine.graph_mut().add_node(Box::new(Const::new(0.8)));
    let freq = engine.graph_mut().add_node(Box::new(Const::new(1000.0)));
    let q = engine.graph_mut().add_node(Box::new(Const::new(0.707)));
    let filt = engine.graph_mut().add_node(Box::new(AllpassFilter::new()));
    engine.graph_mut().connect(src, filt, 0);
    engine.graph_mut().connect(freq, filt, 1);
    engine.graph_mut().connect(q, filt, 2);
    engine.graph_mut().set_sink(filt);
    engine.prepare();

    let output = engine.render_offline(100);
    let last = *output[0].last().unwrap();
    assert!(
        (last - 0.8).abs() < 0.01,
        "Allpass should pass DC at unity gain, got {last}"
    );
}

#[test]
fn test_allpass_preserves_rms() {
    // Feed a sine through allpass — output RMS should be close to input RMS.
    let mut engine_dry = Engine::new(EngineConfig::default());
    let freq = engine_dry.graph_mut().add_node(Box::new(Const::new(440.0)));
    let phase = engine_dry.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc = engine_dry.graph_mut().add_node(Box::new(SinOsc::new()));
    engine_dry.graph_mut().connect(freq, osc, 0);
    engine_dry.graph_mut().connect(phase, osc, 1);
    engine_dry.graph_mut().set_sink(osc);
    engine_dry.prepare();
    let dry_output = engine_dry.render_offline(100);
    let dry_rms: f32 = (dry_output[0].iter().map(|s| s * s).sum::<f32>() / dry_output[0].len() as f32).sqrt();

    let mut engine_wet = Engine::new(EngineConfig::default());
    let freq2 = engine_wet.graph_mut().add_node(Box::new(Const::new(440.0)));
    let phase2 = engine_wet.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc2 = engine_wet.graph_mut().add_node(Box::new(SinOsc::new()));
    engine_wet.graph_mut().connect(freq2, osc2, 0);
    engine_wet.graph_mut().connect(phase2, osc2, 1);

    let filt_freq = engine_wet.graph_mut().add_node(Box::new(Const::new(1000.0)));
    let q = engine_wet.graph_mut().add_node(Box::new(Const::new(0.707)));
    let filt = engine_wet.graph_mut().add_node(Box::new(AllpassFilter::new()));
    engine_wet.graph_mut().connect(osc2, filt, 0);
    engine_wet.graph_mut().connect(filt_freq, filt, 1);
    engine_wet.graph_mut().connect(q, filt, 2);
    engine_wet.graph_mut().set_sink(filt);
    engine_wet.prepare();
    let wet_output = engine_wet.render_offline(100);
    let wet_rms: f32 = (wet_output[0].iter().map(|s| s * s).sum::<f32>() / wet_output[0].len() as f32).sqrt();

    let ratio = wet_rms / dry_rms;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "Allpass should preserve RMS: dry={dry_rms}, wet={wet_rms}, ratio={ratio}"
    );
}

// ============================================================================
// FreqShift tests
// ============================================================================

#[test]
fn test_freqshift_zero_shift_passthrough() {
    // With shift=0, FreqShift should pass the signal through (approximately).
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(phase, osc, 1);

    let shift = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let fs = engine.graph_mut().add_node(Box::new(FreqShift::new()));
    engine.graph_mut().connect(osc, fs, 0);
    engine.graph_mut().connect(shift, fs, 1);
    engine.graph_mut().set_sink(fs);
    engine.prepare();

    let output = engine.render_offline(50);
    let rms: f32 = (output[0].iter().map(|s| s * s).sum::<f32>() / output[0].len() as f32).sqrt();
    assert!(
        rms > 0.3,
        "FreqShift with shift=0 should pass signal through, rms was {rms}"
    );
}

#[test]
fn test_freqshift_produces_output() {
    // With a non-zero shift, FreqShift should still produce output.
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(phase, osc, 1);

    let shift = engine.graph_mut().add_node(Box::new(Const::new(100.0)));
    let fs = engine.graph_mut().add_node(Box::new(FreqShift::new()));
    engine.graph_mut().connect(osc, fs, 0);
    engine.graph_mut().connect(shift, fs, 1);
    engine.graph_mut().set_sink(fs);
    engine.prepare();

    let output = engine.render_offline(50);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        max > 0.1,
        "FreqShift should produce audible output with shift=100, max was {max}"
    );
}

// ============================================================================
// WaveFolder tests
// ============================================================================

#[test]
fn test_wavefolder_bounded_output() {
    // Even with high drive, wavefolder output should stay in [-1, 1]
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(440.0)));
    let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
    engine.graph_mut().connect(freq, osc, 0);
    engine.graph_mut().connect(phase, osc, 1);

    let drive = engine.graph_mut().add_node(Box::new(Const::new(5.0)));
    let sym = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let fold = engine.graph_mut().add_node(Box::new(WaveFolder::new()));
    engine.graph_mut().connect(osc, fold, 0);
    engine.graph_mut().connect(drive, fold, 1);
    engine.graph_mut().connect(sym, fold, 2);
    engine.graph_mut().set_sink(fold);
    engine.prepare();

    let output = engine.render_offline(10);
    for s in &output[0] {
        assert!(
            s.abs() <= 1.0001,
            "WaveFolder output should be bounded to [-1,1], got {s}"
        );
    }
}

#[test]
fn test_wavefolder_drive_increases_harmonics() {
    // Higher drive should produce more zero crossings (more harmonics)
    let count_crossings = |drive_val: f32| -> usize {
        let mut engine = Engine::new(EngineConfig::default());
        let freq = engine.graph_mut().add_node(Box::new(Const::new(100.0)));
        let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
        let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
        engine.graph_mut().connect(freq, osc, 0);
        engine.graph_mut().connect(phase, osc, 1);

        let drive = engine.graph_mut().add_node(Box::new(Const::new(drive_val)));
        let sym = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
        let fold = engine.graph_mut().add_node(Box::new(WaveFolder::new()));
        engine.graph_mut().connect(osc, fold, 0);
        engine.graph_mut().connect(drive, fold, 1);
        engine.graph_mut().connect(sym, fold, 2);
        engine.graph_mut().set_sink(fold);
        engine.prepare();

        let output = engine.render_offline(10);
        let samples = &output[0];
        let mut crossings = 0;
        for w in samples.windows(2) {
            if (w[0] >= 0.0) != (w[1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings
    };

    let low_drive = count_crossings(1.0);
    let high_drive = count_crossings(4.0);
    assert!(
        high_drive > low_drive,
        "Higher drive should produce more zero crossings: low={low_drive}, high={high_drive}"
    );
}

#[test]
fn test_wavefolder_symmetry_changes_output() {
    // With symmetry != 0, output should differ from symmetry == 0
    let render_with_sym = |sym_val: f32| -> Vec<f32> {
        let mut engine = Engine::new(EngineConfig::default());
        let freq = engine.graph_mut().add_node(Box::new(Const::new(100.0)));
        let phase = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
        let osc = engine.graph_mut().add_node(Box::new(SinOsc::new()));
        engine.graph_mut().connect(freq, osc, 0);
        engine.graph_mut().connect(phase, osc, 1);

        let drive = engine.graph_mut().add_node(Box::new(Const::new(3.0)));
        let sym = engine.graph_mut().add_node(Box::new(Const::new(sym_val)));
        let fold = engine.graph_mut().add_node(Box::new(WaveFolder::new()));
        engine.graph_mut().connect(osc, fold, 0);
        engine.graph_mut().connect(drive, fold, 1);
        engine.graph_mut().connect(sym, fold, 2);
        engine.graph_mut().set_sink(fold);
        engine.prepare();

        engine.render_offline(5)
            .into_iter()
            .flatten()
            .collect()
    };

    let sym_0 = render_with_sym(0.0);
    let sym_05 = render_with_sym(0.5);

    // Outputs should differ
    let diff: f32 = sym_0.iter().zip(sym_05.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>() / sym_0.len() as f32;
    assert!(
        diff > 0.01,
        "Different symmetry values should produce different output, avg diff was {diff}"
    );
}

// ============================================================================
// LFO tests
// ============================================================================

#[test]
fn test_lfo_unipolar_sine() {
    // LFO with shape=0 (sine) should output in [0, 1]
    // 5 Hz LFO needs ~140 blocks (64 samples each) for one full cycle at 44100 Hz
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(5.0)));
    let shape = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let lfo_node = engine.graph_mut().add_node(Box::new(Lfo::new()));
    engine.graph_mut().connect(freq, lfo_node, 0);
    engine.graph_mut().connect(shape, lfo_node, 1);
    engine.graph_mut().set_sink(lfo_node);
    engine.prepare();

    let output = engine.render_offline(200);
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for s in &output[0] {
        min = min.min(*s);
        max = max.max(*s);
    }
    assert!(min >= -0.001, "LFO sine min should be >= 0, got {min}");
    assert!(max <= 1.001, "LFO sine max should be <= 1, got {max}");
    assert!(max > 0.9, "LFO sine should reach near 1.0, got max={max}");
    assert!(min < 0.1, "LFO sine should reach near 0.0, got min={min}");
}

#[test]
fn test_lfo_unipolar_all_shapes() {
    // All 4 shapes should produce output in [0, 1]
    for shape_val in [0.0f32, 1.0, 2.0, 3.0] {
        let mut engine = Engine::new(EngineConfig::default());
        let freq = engine.graph_mut().add_node(Box::new(Const::new(10.0)));
        let shape = engine.graph_mut().add_node(Box::new(Const::new(shape_val)));
        let lfo_node = engine.graph_mut().add_node(Box::new(Lfo::new()));
        engine.graph_mut().connect(freq, lfo_node, 0);
        engine.graph_mut().connect(shape, lfo_node, 1);
        engine.graph_mut().set_sink(lfo_node);
        engine.prepare();

        let output = engine.render_offline(200);
        for s in &output[0] {
            assert!(
                *s >= -0.001 && *s <= 1.001,
                "LFO shape={shape_val} output should be in [0,1], got {s}"
            );
        }
    }
}

#[test]
fn test_lfo_square_gating() {
    // Shape=3 (square) should produce values near 0 or 1 only
    let mut engine = Engine::new(EngineConfig::default());
    let freq = engine.graph_mut().add_node(Box::new(Const::new(5.0)));
    let shape = engine.graph_mut().add_node(Box::new(Const::new(3.0)));
    let lfo_node = engine.graph_mut().add_node(Box::new(Lfo::new()));
    engine.graph_mut().connect(freq, lfo_node, 0);
    engine.graph_mut().connect(shape, lfo_node, 1);
    engine.graph_mut().set_sink(lfo_node);
    engine.prepare();

    let output = engine.render_offline(200);
    for s in &output[0] {
        assert!(
            (*s - 0.0).abs() < 0.01 || (*s - 1.0).abs() < 0.01,
            "LFO square should produce 0 or 1, got {s}"
        );
    }
}

#[test]
fn test_lfo_shapes_differ() {
    // Different shapes should produce different waveforms
    let render_shape = |shape_val: f32| -> Vec<f32> {
        let mut engine = Engine::new(EngineConfig::default());
        let freq = engine.graph_mut().add_node(Box::new(Const::new(5.0)));
        let shape = engine.graph_mut().add_node(Box::new(Const::new(shape_val)));
        let lfo_node = engine.graph_mut().add_node(Box::new(Lfo::new()));
        engine.graph_mut().connect(freq, lfo_node, 0);
        engine.graph_mut().connect(shape, lfo_node, 1);
        engine.graph_mut().set_sink(lfo_node);
        engine.prepare();

        engine.render_offline(200)
            .into_iter()
            .flatten()
            .collect()
    };

    let sine = render_shape(0.0);
    let tri = render_shape(1.0);
    let saw = render_shape(2.0);
    let square = render_shape(3.0);

    // Each pair should differ
    let diff_sine_tri: f32 = sine.iter().zip(tri.iter())
        .map(|(a, b)| (a - b).abs()).sum::<f32>() / sine.len() as f32;
    let diff_tri_saw: f32 = tri.iter().zip(saw.iter())
        .map(|(a, b)| (a - b).abs()).sum::<f32>() / tri.len() as f32;
    let diff_saw_square: f32 = saw.iter().zip(square.iter())
        .map(|(a, b)| (a - b).abs()).sum::<f32>() / saw.len() as f32;

    assert!(diff_sine_tri > 0.01, "Sine and triangle should differ");
    assert!(diff_tri_saw > 0.01, "Triangle and saw should differ");
    assert!(diff_saw_square > 0.01, "Saw and square should differ");
}

#[test]
fn test_dsl_wavefolder_compiles() {
    use microsynth::dsl::{self, UGenRegistry};

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let source = r#"
        synthdef test freq=440.0 =
            let osc = sinOsc freq 0.0
            waveFolder osc 3.0 0.0
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs[0].name(), "test");

    let mut engine = Engine::new(EngineConfig::default());
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let output = engine.render_offline(10);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.0, "waveFolder DSL synth should produce output");
}

#[test]
fn test_dsl_lfo_compiles() {
    use microsynth::dsl::{self, UGenRegistry};

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let source = r#"
        synthdef wobble freq=55.0 =
            let osc = blSaw freq
            let m = lfo 5.0 0.0
            let cutoff = m * 4000.0 + 100.0
            lpf osc cutoff 8.0
    "#;
    let defs = dsl::compile(source, &reg).unwrap();
    assert_eq!(defs[0].name(), "wobble");

    let mut engine = Engine::new(EngineConfig::default());
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let output = engine.render_offline(10);
    let max = output[0].iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max > 0.0, "lfo DSL synth should produce output");
}
