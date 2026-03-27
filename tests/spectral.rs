//! Integration tests for spectral processing UGens.

use microsynth::dsl::compiler::UGenRegistry;
use microsynth::ugens::register_builtins;
use microsynth::{Engine, EngineConfig};

/// Helper: render a DSL source string and return all output samples.
fn render_dsl(source: &str, duration_secs: f32) -> Vec<f32> {
    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);
    let defs = microsynth::dsl::compile(source, &reg).expect("DSL compile failed");
    let def = &defs[0];

    let sample_rate = 44100.0;
    let block_size = 64;
    let config = EngineConfig {
        sample_rate,
        block_size,
    };
    let mut engine = Engine::new(config);
    let synth = engine.instantiate_synthdef(def);

    // Auto-set gate=1 if present.
    for param in synth.params() {
        if param.name == "gate" {
            engine.set_param(&synth, "gate", 1.0);
        }
    }

    let sink = synth.output_node();
    engine.graph_mut().set_sink(sink);
    engine.prepare();

    let num_blocks = (duration_secs * sample_rate / block_size as f32) as usize;
    let mut samples = Vec::new();
    for _ in 0..num_blocks {
        if let Some(buf) = engine.render() {
            let ch = buf.channel(0).samples();
            samples.extend_from_slice(ch);
        }
    }
    samples
}

/// Helper: compute RMS of a slice.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

#[test]
fn spectral_freeze_dsl_compiles() {
    let source = "synthdef test freq=440.0 trig=0.0 = spectralFreeze (sinOsc freq 0.0) trig";
    let samples = render_dsl(source, 0.5);
    let tail = &samples[samples.len() / 2..];
    assert!(rms(tail) > 0.001, "SpectralFreeze should produce output");
}

#[test]
fn pitch_shift_dsl_compiles() {
    let source = "synthdef test freq=440.0 = pitchShift (sinOsc freq 0.0) 1.0";
    let samples = render_dsl(source, 0.5);
    let tail = &samples[samples.len() / 2..];
    assert!(rms(tail) > 0.001, "PitchShift should produce output");
}

#[test]
fn spectral_gate_passthrough() {
    let source = "synthdef test freq=440.0 = spectralGate (sinOsc freq 0.0) 0.0";
    let samples = render_dsl(source, 0.5);
    let tail = &samples[samples.len() / 2..];
    assert!(
        rms(tail) > 0.1,
        "SpectralGate with threshold=0 should pass through, RMS={}",
        rms(tail)
    );
}

#[test]
fn spectral_gate_high_threshold() {
    let source = "synthdef test freq=440.0 = spectralGate (sinOsc freq 0.0) 0.99";
    let samples = render_dsl(source, 0.5);
    // Should compile and run without crashing.
    assert!(!samples.is_empty());
}

#[test]
fn spectral_blur_passthrough() {
    let source = "synthdef test freq=440.0 = spectralBlur (sinOsc freq 0.0) 0.0";
    let samples = render_dsl(source, 0.5);
    let tail = &samples[samples.len() / 2..];
    assert!(
        rms(tail) > 0.1,
        "SpectralBlur with blur=0 should pass through, RMS={}",
        rms(tail)
    );
}

#[test]
fn spectral_blur_high_value() {
    let source = "synthdef test freq=440.0 = spectralBlur (sinOsc freq 0.0) 0.99";
    let samples = render_dsl(source, 0.5);
    let tail = &samples[samples.len() / 2..];
    assert!(
        rms(tail) > 0.01,
        "SpectralBlur with high blur should still produce output"
    );
}

#[test]
fn spectral_filter_dsl_compiles() {
    let source = "synthdef test freq=440.0 = spectralFilter (sinOsc freq 0.0) 1000.0 500.0 2.0";
    let samples = render_dsl(source, 0.5);
    let tail = &samples[samples.len() / 2..];
    assert!(rms(tail) > 0.001, "SpectralFilter should produce output");
}

#[test]
fn convolution_dry_passthrough() {
    // Convolution with no IR loaded passes through dry signal.
    let source = "synthdef test freq=440.0 = convolution (sinOsc freq 0.0) 1.0";
    let samples = render_dsl(source, 0.5);
    let tail = &samples[samples.len() / 2..];
    assert!(
        rms(tail) > 0.1,
        "Convolution without IR should pass through dry"
    );
}

#[test]
fn all_spectral_ugens_compile_in_dsl() {
    // Verify all spectral UGens are accessible from DSL.
    let sources = [
        "synthdef t freq=440.0 = spectralFreeze (sinOsc freq 0.0) 0.0",
        "synthdef t freq=440.0 = pitchShift (sinOsc freq 0.0) 1.0",
        "synthdef t freq=440.0 = spectralFilter (sinOsc freq 0.0) 1000.0 500.0 1.0",
        "synthdef t freq=440.0 = spectralGate (sinOsc freq 0.0) 0.1",
        "synthdef t freq=440.0 = spectralBlur (sinOsc freq 0.0) 0.5",
        "synthdef t freq=440.0 = convolution (sinOsc freq 0.0) 0.5",
    ];

    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    for source in &sources {
        let result = microsynth::dsl::compile(source, &reg);
        assert!(
            result.is_ok(),
            "Failed to compile: {source}\nError: {:?}",
            result.err()
        );
    }
}
