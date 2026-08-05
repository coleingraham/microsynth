//! Scratch measurement: does Compressor-preset + SoftClip hold a -1 dBTP
//! true-peak ceiling on a hot test signal? Not part of the test suite --
//! run manually with `cargo run --example measure_limiter`.

use microsynth::ugens::*;
use microsynth::*;

/// Plays back a fixed sample buffer, one sample per tick, looping if the
/// render runs longer than the buffer.
struct PlaybackSource {
    data: Vec<f32>,
    pos: usize,
}

impl UGen for PlaybackSource {
    fn spec(&self) -> node::UGenSpec {
        static OUTPUTS: &[node::OutputSpec] = &[node::OutputSpec {
            name: "out",
            rate: context::Rate::Audio,
        }];
        node::UGenSpec {
            name: "PlaybackSource",
            category: node::UGenCategory::Utility,
            inputs: &[],
            outputs: OUTPUTS,
        }
    }
    fn init(&mut self, _context: &context::ProcessContext) {}
    fn reset(&mut self) {
        self.pos = 0;
    }
    fn process(
        &mut self,
        _context: &context::ProcessContext,
        _inputs: &[&buffer::AudioBuffer],
        output: &mut buffer::AudioBuffer,
    ) {
        let out = output.channel_mut(0).samples_mut();
        for s in out.iter_mut() {
            *s = self.data[self.pos % self.data.len()];
            self.pos += 1;
        }
    }
}

/// Standard cubic (Catmull-Rom) oversampling estimate of true peak. Not a
/// full ITU-R BS.1770 Annex-2 FIR, but reveals the same phenomenon: inter-
/// sample peaks a naive sample-peak read misses.
fn true_peak_linear(samples: &[f32], oversample: usize) -> f32 {
    let n = samples.len() as isize;
    let get = |i: isize| -> f32 {
        let idx = i.clamp(0, n - 1) as usize;
        samples[idx]
    };
    let mut peak = 0.0f32;
    for i in 0..samples.len() as isize {
        let p0 = get(i - 1);
        let p1 = get(i);
        let p2 = get(i + 1);
        let p3 = get(i + 2);
        for k in 0..oversample {
            let t = k as f32 / oversample as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            let v = 0.5
                * ((2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
            peak = peak.max(v.abs());
        }
    }
    peak
}

fn lin_to_db(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

fn build_hot_signal(sr: f32, len: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f32 / sr;
        // Overdriven 3 kHz tone (+2.9 dBFS) -- close enough to Nyquist/4 that
        // clipping harmonics land near Nyquist, which is where inter-sample
        // ringing shows up hardest. Plus an abrupt transient burst partway
        // through to stress the compressor's attack time.
        let mut s = 1.4 * (core::f32::consts::TAU * 3000.0 * t).sin();
        if (400..416).contains(&i) {
            s = if i % 2 == 0 { 1.8 } else { -1.8 };
        }
        v.push(s);
    }
    v
}

fn measure(label: &str, threshold: f32, ratio: f32, attack: f32, release: f32, drive: f32) {
    let sr = 44100.0;
    let signal = build_hot_signal(sr, 2048);
    let input_true_peak = true_peak_linear(&signal, 8);

    let mut engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: 64,
    });
    let src = engine.graph_mut().add_node(Box::new(PlaybackSource {
        data: signal,
        pos: 0,
    }));
    let thresh = engine
        .graph_mut()
        .add_node(Box::new(Const::new(threshold)));
    let ratio_n = engine.graph_mut().add_node(Box::new(Const::new(ratio)));
    let attack_n = engine.graph_mut().add_node(Box::new(Const::new(attack)));
    let release_n = engine.graph_mut().add_node(Box::new(Const::new(release)));
    let makeup_n = engine.graph_mut().add_node(Box::new(Const::new(0.0)));
    let comp = engine.graph_mut().add_node(Box::new(Compressor::new()));
    let clip = engine.graph_mut().add_node(Box::new(SoftClip::new()));
    let drive_n = engine.graph_mut().add_node(Box::new(Const::new(drive)));

    engine.graph_mut().connect(src, comp, 0); // in
    engine.graph_mut().connect(src, comp, 1); // sidechain = self
    engine.graph_mut().connect(thresh, comp, 2);
    engine.graph_mut().connect(ratio_n, comp, 3);
    engine.graph_mut().connect(attack_n, comp, 4);
    engine.graph_mut().connect(release_n, comp, 5);
    engine.graph_mut().connect(makeup_n, comp, 6);

    engine.graph_mut().connect(comp, clip, 0);
    engine.graph_mut().connect(drive_n, clip, 1);

    engine.graph_mut().set_sink(clip);
    engine.prepare();

    let output = engine.render_offline(2048 / 64);
    let ch0 = &output[0];
    let sample_peak = ch0.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    let true_peak = true_peak_linear(ch0, 8);

    println!(
        "{label}: input true-peak {:.3} dBTP | output sample-peak {:.3} dBFS, true-peak {:.3} dBTP {}",
        lin_to_db(input_true_peak),
        lin_to_db(sample_peak),
        lin_to_db(true_peak),
        if lin_to_db(true_peak) <= -1.0 { "HOLDS -1 dBTP" } else { "EXCEEDS -1 dBTP" }
    );
}

fn measure_limiter(label: &str, ceiling: f32, release: f32) {
    let sr = 44100.0;
    let signal = build_hot_signal(sr, 2048);

    let mut engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: 64,
    });
    let src = engine.graph_mut().add_node(Box::new(PlaybackSource {
        data: signal,
        pos: 0,
    }));
    let ceiling_n = engine.graph_mut().add_node(Box::new(Const::new(ceiling)));
    let release_n = engine.graph_mut().add_node(Box::new(Const::new(release)));
    let lim = engine.graph_mut().add_node(Box::new(Limiter::new()));

    engine.graph_mut().connect(src, lim, 0);
    engine.graph_mut().connect(ceiling_n, lim, 1);
    engine.graph_mut().connect(release_n, lim, 2);
    engine.graph_mut().set_sink(lim);
    engine.prepare();

    let output = engine.render_offline(2048 / 64);
    let ch0 = &output[0];
    let sample_peak = ch0.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    let true_peak = true_peak_linear(ch0, 8);

    println!(
        "{label}: output sample-peak {:.3} dBFS, true-peak {:.3} dBTP {}",
        lin_to_db(sample_peak),
        lin_to_db(true_peak),
        if lin_to_db(true_peak) <= ceiling { "HOLDS ceiling" } else { "EXCEEDS ceiling" }
    );
}

fn main() {
    println!("Hot signal: overdriven 3kHz sine (+2.9 dBFS) + transient burst\n");

    println!("-- Compressor preset + SoftClip --");
    measure("compressor(-1dB,20:1,1ms/50ms)+softclip(drive=1.0)", -1.0, 20.0, 0.001, 0.05, 1.0);
    measure("compressor(-3dB,20:1,1ms/50ms)+softclip(drive=1.0)", -3.0, 20.0, 0.001, 0.05, 1.0);
    measure("compressor(-1dB,20:1,0.1ms/50ms)+softclip(drive=2.0)", -1.0, 20.0, 0.0001, 0.05, 2.0);
    measure("compressor(-6dB,20:1,0.1ms/50ms)+softclip(drive=3.0)", -6.0, 20.0, 0.0001, 0.05, 3.0);

    println!("\n-- Dedicated Limiter (look-ahead, true-peak-aware) --");
    measure_limiter("limiter(ceiling=-1dBTP,release=50ms)", -1.0, 0.05);
    measure_limiter("limiter(ceiling=-1dBTP,release=200ms)", -1.0, 0.2);

    // Even hotter / more pathological signal: full-scale near-Nyquist-ish
    // content plus a sharp step, to stress the look-ahead harder.
    println!("\n-- Limiter on an even hotter signal --");
    let sr = 44100.0;
    let mut hot2 = Vec::with_capacity(2048);
    for i in 0..2048usize {
        let t = i as f32 / sr;
        let mut s = 2.0 * (core::f32::consts::TAU * 8000.0 * t).sin();
        if i % 200 < 4 {
            s = if i % 2 == 0 { 3.0 } else { -3.0 };
        }
        hot2.push(s);
    }
    let input_tp = true_peak_linear(&hot2, 8);
    println!("input true-peak: {:.3} dBTP", lin_to_db(input_tp));

    let mut engine = Engine::new(EngineConfig { sample_rate: sr, block_size: 64 });
    let src = engine.graph_mut().add_node(Box::new(PlaybackSource { data: hot2, pos: 0 }));
    let ceiling_n = engine.graph_mut().add_node(Box::new(Const::new(-1.0)));
    let release_n = engine.graph_mut().add_node(Box::new(Const::new(0.05)));
    let lim = engine.graph_mut().add_node(Box::new(Limiter::new()));
    engine.graph_mut().connect(src, lim, 0);
    engine.graph_mut().connect(ceiling_n, lim, 1);
    engine.graph_mut().connect(release_n, lim, 2);
    engine.graph_mut().set_sink(lim);
    engine.prepare();
    let output = engine.render_offline(2048 / 64);
    let ch0 = &output[0];
    let sample_peak = ch0.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    let true_peak = true_peak_linear(ch0, 8);
    println!(
        "output sample-peak {:.3} dBFS, true-peak {:.3} dBTP {}",
        lin_to_db(sample_peak),
        lin_to_db(true_peak),
        if lin_to_db(true_peak) <= -1.0 { "HOLDS -1 dBTP" } else { "EXCEEDS -1 dBTP" }
    );

    println!("\n-- Limiter edge cases --");
    measure_edge_cases();
}

fn measure_edge_cases() {
    let sr = 44100.0;

    // Silence: should stay silent, no NaN.
    let silence = vec![0.0f32; 512];
    let mut engine = Engine::new(EngineConfig { sample_rate: sr, block_size: 64 });
    let src = engine.graph_mut().add_node(Box::new(PlaybackSource { data: silence, pos: 0 }));
    let ceiling_n = engine.graph_mut().add_node(Box::new(Const::new(-1.0)));
    let release_n = engine.graph_mut().add_node(Box::new(Const::new(0.05)));
    let lim = engine.graph_mut().add_node(Box::new(Limiter::new()));
    engine.graph_mut().connect(src, lim, 0);
    engine.graph_mut().connect(ceiling_n, lim, 1);
    engine.graph_mut().connect(release_n, lim, 2);
    engine.graph_mut().set_sink(lim);
    engine.prepare();
    let output = engine.render_offline(8);
    let max = output[0].iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    let has_nan = output[0].iter().any(|x| x.is_nan());
    println!("silence: max |out| = {max}, has_nan = {has_nan}");

    // Full-scale DC.
    let dc = vec![1.0f32; 512];
    let mut engine = Engine::new(EngineConfig { sample_rate: sr, block_size: 64 });
    let src = engine.graph_mut().add_node(Box::new(PlaybackSource { data: dc, pos: 0 }));
    let ceiling_n = engine.graph_mut().add_node(Box::new(Const::new(-1.0)));
    let release_n = engine.graph_mut().add_node(Box::new(Const::new(0.05)));
    let lim = engine.graph_mut().add_node(Box::new(Limiter::new()));
    engine.graph_mut().connect(src, lim, 0);
    engine.graph_mut().connect(ceiling_n, lim, 1);
    engine.graph_mut().connect(release_n, lim, 2);
    engine.graph_mut().set_sink(lim);
    engine.prepare();
    let output = engine.render_offline(8);
    let last = *output[0].last().unwrap();
    println!("full-scale DC: settled output = {last:.4} ({:.3} dBFS)", lin_to_db(last.abs()));

    // Alternating near-Nyquist burst only (the most adversarial local case).
    let mut burst = vec![0.0f32; 512];
    for i in 200..260 {
        burst[i] = if i % 2 == 0 { 1.8 } else { -1.8 };
    }
    let input_tp = true_peak_linear(&burst, 8);
    let mut engine = Engine::new(EngineConfig { sample_rate: sr, block_size: 64 });
    let src = engine.graph_mut().add_node(Box::new(PlaybackSource { data: burst, pos: 0 }));
    let ceiling_n = engine.graph_mut().add_node(Box::new(Const::new(-1.0)));
    let release_n = engine.graph_mut().add_node(Box::new(Const::new(0.05)));
    let lim = engine.graph_mut().add_node(Box::new(Limiter::new()));
    engine.graph_mut().connect(src, lim, 0);
    engine.graph_mut().connect(ceiling_n, lim, 1);
    engine.graph_mut().connect(release_n, lim, 2);
    engine.graph_mut().set_sink(lim);
    engine.prepare();
    let output = engine.render_offline(8);
    let true_peak = true_peak_linear(&output[0], 8);
    println!(
        "Nyquist-alternating burst: input true-peak {:.3} dBTP -> output true-peak {:.3} dBTP {}",
        lin_to_db(input_tp),
        lin_to_db(true_peak),
        if lin_to_db(true_peak) <= -1.0 { "HOLDS -1 dBTP" } else { "EXCEEDS -1 dBTP" }
    );
}
