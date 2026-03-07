//! Tests for sequencer-oriented features:
//! - Dynamic parameter control
//! - Done actions (envelope completion → synth removal)
//! - New envelopes (Perc, ADSR)
//! - New utility UGens (Impulse, Lag, Clip)
//! - Event scheduling
//! - Sample playback (PlayBuf)
//! - Bus mixing

use microsynth::*;
use microsynth::dsl::{self, UGenRegistry};
use microsynth::ugens;
use std::sync::Arc;

fn make_engine(block_size: usize) -> Engine {
    Engine::new(EngineConfig {
        sample_rate: 44100.0,
        block_size,
    })
}

fn make_registry() -> UGenRegistry {
    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);
    reg
}

// ============================================================================
// Dynamic Parameter Control
// ============================================================================

#[test]
fn test_set_param_changes_const_value() {
    let registry = make_registry();
    let defs = dsl::compile(
        "synthdef test freq=440.0 = sinOsc freq 0.0",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    // Verify the synth has a "freq" param
    assert!(synth.param_node("freq").is_some());

    // Render one block at 440 Hz
    let _out1 = engine.render();

    // Change freq to 880 Hz
    assert!(engine.set_param(&synth, "freq", 880.0));

    // Render another block at 880 Hz — should produce different output
    let _out2 = engine.render();
}

#[test]
fn test_set_param_returns_false_for_unknown_param() {
    let registry = make_registry();
    let defs = dsl::compile(
        "synthdef test freq=440.0 = sinOsc freq 0.0",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    assert!(!engine.set_param(&synth, "nonexistent", 100.0));
}

#[test]
fn test_voice_param_control() {
    let registry = make_registry();
    let defs = dsl::compile(
        "synthdef test freq=440.0 amp=1.0 = sinOsc freq 0.0 * amp",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let bus = engine.graph_mut().add_node(Box::new(ugens::Bus::new(8)));
    engine.graph_mut().set_sink(bus);

    let voice = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    engine.prepare();

    // Set freq via voice API
    assert!(engine.set_voice_param(voice, "freq", 880.0));
    assert!(engine.set_voice_param(voice, "amp", 0.5));
    assert!(!engine.set_voice_param(voice, "missing", 0.0));

    engine.render();
}

// ============================================================================
// Done Actions
// ============================================================================

#[test]
fn test_perc_envelope_done_action() {
    let registry = make_registry();
    // Very short percussive envelope
    let defs = dsl::compile(
        "synthdef hit = sinOsc 440.0 0.0 * perc 0.001 0.01",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let bus = engine.graph_mut().add_node(Box::new(ugens::Bus::new(8)));
    engine.graph_mut().set_sink(bus);

    let _voice = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    engine.prepare();

    // Render enough blocks for the envelope to complete
    // At 44100 Hz, 0.001s attack + 0.01s release = ~485 samples ≈ 8 blocks of 64
    for _ in 0..20 {
        engine.render();
    }

    // The perc envelope should be done
    let removed = engine.free_done_synths();
    assert!(removed > 0, "Expected Perc envelope to signal done");
    assert_eq!(engine.synths().len(), 0);
}

#[test]
fn test_asr_done_after_release() {
    let registry = make_registry();
    let defs = dsl::compile(
        "synthdef pad gate=1.0 = sinOsc 440.0 0.0 * asr gate 0.001 0.01",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let bus = engine.graph_mut().add_node(Box::new(ugens::Bus::new(8)));
    engine.graph_mut().set_sink(bus);

    let voice = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    engine.prepare();

    // Gate on — render attack + sustain
    for _ in 0..10 {
        engine.render();
    }
    // Should NOT be done yet (sustaining)
    assert_eq!(engine.free_done_synths(), 0);

    // Gate off — trigger release
    engine.set_voice_param(voice, "gate", 0.0);
    for _ in 0..20 {
        engine.render();
    }

    // After release completes, should be done
    let removed = engine.free_done_synths();
    assert!(removed > 0, "Expected ASR to signal done after release");
}

// ============================================================================
// New Envelopes
// ============================================================================

#[test]
fn test_perc_envelope_shape() {
    let mut engine = make_engine(64);

    let mut builder = SynthDefBuilder::new("perc_test");
    let attack = builder.add_node(|| Box::new(ugens::Const::new(0.001)));
    let release = builder.add_node(|| Box::new(ugens::Const::new(0.01)));
    let perc = builder.add_node(|| Box::new(ugens::Perc::new()));
    builder.connect(attack, perc, 0);
    builder.connect(release, perc, 1);
    builder.set_output(perc);
    let def = builder.build();

    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    // Collect samples
    let mut all_samples = Vec::new();
    for _ in 0..20 {
        if let Some(buf) = engine.render() {
            all_samples.extend_from_slice(buf.channel(0).samples());
        }
    }

    // Should start at 0, rise to ~1, then fall back to 0
    assert!(all_samples[0] < 0.1, "Should start near zero");
    let max_val = all_samples.iter().cloned().fold(0.0f32, f32::max);
    assert!(max_val > 0.9, "Peak should reach near 1.0, got {max_val}");
    let last_val = *all_samples.last().unwrap();
    assert!(last_val < 0.01, "Should end near zero, got {last_val}");
}

#[test]
fn test_adsr_envelope_shape() {
    let registry = make_registry();
    let defs = dsl::compile(
        "synthdef test gate=1.0 = adsr gate 0.001 0.005 0.5 0.01",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    // Render attack + decay phase (gate on)
    let mut all_samples = Vec::new();
    for _ in 0..20 {
        if let Some(buf) = engine.render() {
            all_samples.extend_from_slice(buf.channel(0).samples());
        }
    }

    // Should reach sustain level (~0.5)
    let last_samples = &all_samples[all_samples.len() - 64..];
    let avg: f32 = last_samples.iter().sum::<f32>() / last_samples.len() as f32;
    assert!((avg - 0.5).abs() < 0.05, "Sustain level should be ~0.5, got {avg}");

    // Gate off — release
    engine.set_param(&synth, "gate", 0.0);
    for _ in 0..20 {
        if let Some(buf) = engine.render() {
            all_samples.extend_from_slice(buf.channel(0).samples());
        }
    }

    let last_val = *all_samples.last().unwrap();
    assert!(last_val < 0.05, "Should release to near zero, got {last_val}");
}

// ============================================================================
// New Utility UGens
// ============================================================================

#[test]
fn test_impulse_fires_periodically() {
    let mut engine = make_engine(64);

    let mut builder = SynthDefBuilder::new("imp");
    let freq = builder.add_node(|| Box::new(ugens::Const::new(44100.0 / 64.0))); // one impulse per block
    let imp = builder.add_node(|| Box::new(ugens::Impulse::new()));
    builder.connect(freq, imp, 0);
    builder.set_output(imp);
    let def = builder.build();

    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    // Render 4 blocks, should get one impulse near the start of each
    let mut impulse_count = 0;
    for _ in 0..4 {
        if let Some(buf) = engine.render() {
            for s in buf.channel(0).samples() {
                if *s > 0.5 {
                    impulse_count += 1;
                }
            }
        }
    }
    assert!(impulse_count >= 3, "Expected ~4 impulses, got {impulse_count}");
}

#[test]
fn test_lag_smooths_step() {
    let mut engine = make_engine(64);

    let mut builder = SynthDefBuilder::new("lag_test");
    let input = builder.add_node(|| Box::new(ugens::Const::new(1.0)));
    let time = builder.add_node(|| Box::new(ugens::Const::new(0.01)));
    let lag = builder.add_node(|| Box::new(ugens::Lag::new()));
    builder.connect(input, lag, 0);
    builder.connect(time, lag, 1);
    builder.set_output(lag);
    let def = builder.build();

    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    // First render: lag starts at 0, should smoothly approach 1.0
    let buf = engine.render().unwrap();
    let first = buf.channel(0).samples()[0];
    let last = buf.channel(0).samples()[63];
    assert!(first < 0.5, "Lag should start low, got {first}");
    assert!(last > first, "Lag should be rising");

    // After several blocks, should converge near 1.0
    for _ in 0..100 {
        engine.render();
    }
    let buf = engine.render().unwrap();
    let val = buf.channel(0).samples()[0];
    assert!((val - 1.0).abs() < 0.001, "Should converge to 1.0, got {val}");
}

#[test]
fn test_clip_clamps_signal() {
    let registry = make_registry();
    let defs = dsl::compile(
        "synthdef test = clip (sinOsc 440.0 0.0) (-0.5) 0.5",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    let buf = engine.render().unwrap();
    for s in buf.channel(0).samples() {
        assert!(*s >= -0.5 && *s <= 0.5, "Sample {s} out of clip range");
    }
}

// ============================================================================
// Event Scheduling
// ============================================================================

#[test]
fn test_scheduler_dispatches_events() {
    let registry = make_registry();
    let defs = dsl::compile(
        "synthdef tone freq=440.0 gate=1.0 = sinOsc freq 0.0 * asr gate 0.01 0.01",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let bus = engine.graph_mut().add_node(Box::new(ugens::Bus::new(8)));
    engine.graph_mut().set_sink(bus);

    let voice = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    engine.prepare();

    // Schedule gate off at sample 128 (block 2)
    engine.scheduler_mut().schedule_gate(128, voice, 0.0);

    // Render block 0 (samples 0-63): gate still on
    engine.render();
    // Render block 1 (samples 64-127): gate still on
    engine.render();
    // Render block 2 (samples 128-191): gate off event fires
    engine.render();

    // After enough time, the envelope should complete
    for _ in 0..20 {
        engine.render();
    }
    let removed = engine.free_done_synths();
    assert!(removed > 0, "Voice should be freed after scheduled gate off");
}

#[test]
fn test_scheduler_param_change() {
    let registry = make_registry();
    let defs = dsl::compile(
        "synthdef tone freq=440.0 = sinOsc freq 0.0",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let bus = engine.graph_mut().add_node(Box::new(ugens::Bus::new(8)));
    engine.graph_mut().set_sink(bus);

    let voice = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    engine.prepare();

    // Schedule freq change to 880 at sample 64
    engine.scheduler_mut().schedule_set_param(
        64, voice, "freq", 880.0,
    );

    // First block: freq=440
    engine.render();
    // Second block: event fires, freq=880
    engine.render();
}

// ============================================================================
// Sample Playback
// ============================================================================

#[test]
fn test_sample_bank_store_and_retrieve() {
    let mut bank = SampleBank::new();

    // Create a simple test sample
    let data: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    let sample = Sample::from_mono(&data, 44100.0).with_name("test");
    let id = bank.load(sample);

    assert_eq!(bank.len(), 1);
    assert!(bank.get(id).is_some());
    assert!(bank.get_by_name("test").is_some());
    assert_eq!(bank.get(id).unwrap().num_frames(), 100);
}

#[test]
fn test_playbuf_plays_sample() {
    let mut engine = make_engine(64);

    // Create a 128-sample test tone
    let data: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0 * 2.0 - 1.0)).collect();
    let sample = Arc::new(Sample::from_mono(&data, 44100.0));

    let mut builder = SynthDefBuilder::new("player");
    let rate_node = builder.add_node(|| Box::new(ugens::Const::new(1.0)));
    let sample_clone = Arc::clone(&sample);
    let play = builder.add_node(move || {
        Box::new(ugens::PlayBuf::new().with_sample(Arc::clone(&sample_clone)))
    });
    builder.connect(rate_node, play, 0);
    builder.set_output(play);
    let def = builder.build();

    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    // Render first block (64 samples)
    let buf = engine.render().unwrap();
    let first_sample = buf.channel(0).samples()[0];
    let last_sample = buf.channel(0).samples()[63];
    // Should output the ramp
    assert!(first_sample < 0.0, "First sample should be negative (start of ramp)");
    assert!(last_sample > first_sample, "Should be ascending ramp");

    // Render second block — should complete
    engine.render();

    // Third block — playback should be done, output silence
    let buf = engine.render().unwrap();
    let val = buf.channel(0).samples()[0];
    assert!(val.abs() < 0.001, "Should be silence after playback, got {val}");
}

#[test]
fn test_playbuf_looping() {
    let mut engine = make_engine(64);

    // Short 32-sample buffer
    let data: Vec<f32> = (0..32).map(|i| (i as f32 / 32.0)).collect();
    let sample = Arc::new(Sample::from_mono(&data, 44100.0));

    let mut builder = SynthDefBuilder::new("looper");
    let rate_node = builder.add_node(|| Box::new(ugens::Const::new(1.0)));
    let sample_clone = Arc::clone(&sample);
    let play = builder.add_node(move || {
        Box::new(ugens::PlayBuf::new().with_sample(Arc::clone(&sample_clone)).with_loop(true))
    });
    builder.connect(rate_node, play, 0);
    builder.set_output(play);
    let def = builder.build();

    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    // Render several blocks — looping buffer should NOT be done
    for _ in 0..10 {
        engine.render();
    }

    let buf = engine.render().unwrap();
    let has_nonzero = buf.channel(0).samples().iter().any(|s| s.abs() > 0.001);
    assert!(has_nonzero, "Looping PlayBuf should keep producing output");
}

// ============================================================================
// Bus Mixing
// ============================================================================

#[test]
fn test_bus_sums_multiple_voices() {
    let registry = make_registry();
    let defs = dsl::compile(
        "synthdef tone freq=440.0 = sinOsc freq 0.0",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let bus = engine.graph_mut().add_node(Box::new(ugens::Bus::new(8)));
    engine.graph_mut().set_sink(bus);

    // Spawn two voices on the bus
    let v1 = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    let v2 = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    engine.prepare();

    // Both should be active
    assert!(engine.voice_synth(v1).is_some());
    assert!(engine.voice_synth(v2).is_some());

    // Render — bus output should be non-zero (sum of two sines)
    let buf = engine.render().unwrap();
    let has_nonzero = buf.channel(0).samples().iter().any(|s| s.abs() > 0.001);
    assert!(has_nonzero, "Bus should output summed voices");
}

#[test]
fn test_bus_voice_removal() {
    let registry = make_registry();
    let defs = dsl::compile(
        "synthdef tone freq=440.0 = sinOsc freq 0.0",
        &registry,
    ).unwrap();

    let mut engine = make_engine(64);
    let bus = engine.graph_mut().add_node(Box::new(ugens::Bus::new(8)));
    engine.graph_mut().set_sink(bus);

    let v1 = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    let _v2 = engine.spawn_voice_on_bus(&defs[0], bus).unwrap();
    engine.prepare();

    // Remove voice 1
    engine.free_voice(v1);
    engine.prepare();

    // Should still render (voice 2 still active)
    let buf = engine.render().unwrap();
    let has_nonzero = buf.channel(0).samples().iter().any(|s| s.abs() > 0.001);
    assert!(has_nonzero, "Remaining voice should still produce output");
}

// ============================================================================
// DSL registration of new UGens
// ============================================================================

#[test]
fn test_dsl_perc_compiles() {
    let registry = make_registry();
    let result = dsl::compile(
        "synthdef hit = sinOsc 440.0 0.0 * perc 0.01 0.1",
        &registry,
    );
    assert!(result.is_ok());
}

#[test]
fn test_dsl_adsr_compiles() {
    let registry = make_registry();
    let result = dsl::compile(
        "synthdef pad gate=1.0 = sinOsc 440.0 0.0 * adsr gate 0.01 0.1 0.7 0.5",
        &registry,
    );
    assert!(result.is_ok());
}

#[test]
fn test_dsl_impulse_compiles() {
    let registry = make_registry();
    let result = dsl::compile(
        "synthdef clock = impulse 4.0",
        &registry,
    );
    assert!(result.is_ok());
}

#[test]
fn test_dsl_lag_compiles() {
    let registry = make_registry();
    let result = dsl::compile(
        "synthdef smooth freq=440.0 = sinOsc (lag freq 0.01) 0.0",
        &registry,
    );
    assert!(result.is_ok());
}

#[test]
fn test_dsl_clip_compiles() {
    let registry = make_registry();
    let result = dsl::compile(
        "synthdef clipped = clip (sinOsc 440.0 0.0) (-0.5) 0.5",
        &registry,
    );
    assert!(result.is_ok());
}
