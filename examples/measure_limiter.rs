//! Scratch measurement: does Compressor-preset + SoftClip hold a -1 dBTP
//! true-peak ceiling on a hot test signal? Not part of the test suite --
//! run manually with `cargo run --example measure_limiter`.
//!
//! Also covers the stereo cross-channel defect found on real pilot-genre
//! program material (rendered through the actual `composer forward`
//! pipeline, independently confirmed with `ffmpeg -af ebur128`): the
//! limiter's original look-ahead buffer was a single `DelayLine` shared
//! across channels, so channel 1's look-ahead reads mostly landed on
//! channel 0's samples instead of its own. No mono synthetic signal (all of
//! this file's other cases) could have exposed that -- it only showed up on
//! genuine two-different-channels stereo content. See `measure_stereo_case`
//! below for the always-runnable synthetic reproduction (derived from that
//! real finding, not chosen blind), and `measure_real_material` for an
//! *optional* path that verifies against an actual rendered mix if you
//! provide one -- deliberately NOT bundled with this repo (microsynth is
//! public; pilot/genre program material doesn't belong in it), pass a path
//! to a raw interleaved-stereo f32le PCM file as the first CLI argument to
//! use it: `cargo run --example measure_limiter -- /path/to/mix.f32`.

use microsynth::ugens::*;
use microsynth::*;
use std::env;
use std::fs;

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

/// Two-channel source with genuinely different content per channel -- see
/// the module doc: this is what exposed the shared-look-ahead-buffer defect
/// that every mono case in this file could not.
struct StereoPlaybackSource {
    left: Vec<f32>,
    right: Vec<f32>,
    pos: usize,
}

impl UGen for StereoPlaybackSource {
    fn spec(&self) -> node::UGenSpec {
        static OUTPUTS: &[node::OutputSpec] = &[node::OutputSpec {
            name: "out",
            rate: context::Rate::Audio,
        }];
        node::UGenSpec {
            name: "StereoPlaybackSource",
            category: node::UGenCategory::Utility,
            inputs: &[],
            outputs: OUTPUTS,
        }
    }
    fn init(&mut self, _context: &context::ProcessContext) {}
    fn reset(&mut self) {
        self.pos = 0;
    }
    fn output_channels(&self, _input_channels: &[usize]) -> usize {
        2
    }
    fn process(
        &mut self,
        _context: &context::ProcessContext,
        _inputs: &[&buffer::AudioBuffer],
        output: &mut buffer::AudioBuffer,
    ) {
        let n = output.channel(0).samples().len();
        for i in 0..n {
            let l = self.left.get(self.pos).copied().unwrap_or(0.0);
            let r = self.right.get(self.pos).copied().unwrap_or(0.0);
            output.channel_mut(0).samples_mut()[i] = l;
            output.channel_mut(1).samples_mut()[i] = r;
            self.pos += 1;
        }
    }
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
    let thresh = engine.graph_mut().add_node(Box::new(Const::new(threshold)));
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
        if lin_to_db(true_peak) <= -1.0 {
            "HOLDS -1 dBTP"
        } else {
            "EXCEEDS -1 dBTP"
        }
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
        if lin_to_db(true_peak) <= ceiling {
            "HOLDS ceiling"
        } else {
            "EXCEEDS ceiling"
        }
    );
}

fn main() {
    println!("Hot signal: overdriven 3kHz sine (+2.9 dBFS) + transient burst\n");

    println!("-- Compressor preset + SoftClip --");
    measure(
        "compressor(-1dB,20:1,1ms/50ms)+softclip(drive=1.0)",
        -1.0,
        20.0,
        0.001,
        0.05,
        1.0,
    );
    measure(
        "compressor(-3dB,20:1,1ms/50ms)+softclip(drive=1.0)",
        -3.0,
        20.0,
        0.001,
        0.05,
        1.0,
    );
    measure(
        "compressor(-1dB,20:1,0.1ms/50ms)+softclip(drive=2.0)",
        -1.0,
        20.0,
        0.0001,
        0.05,
        2.0,
    );
    measure(
        "compressor(-6dB,20:1,0.1ms/50ms)+softclip(drive=3.0)",
        -6.0,
        20.0,
        0.0001,
        0.05,
        3.0,
    );

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

    let mut engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: 64,
    });
    let src = engine
        .graph_mut()
        .add_node(Box::new(PlaybackSource { data: hot2, pos: 0 }));
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
        if lin_to_db(true_peak) <= -1.0 {
            "HOLDS -1 dBTP"
        } else {
            "EXCEEDS -1 dBTP"
        }
    );

    println!("\n-- Limiter edge cases --");
    measure_edge_cases();

    println!("\n-- Limiter stereo (independent per-channel content) --");
    measure_stereo_case();

    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "--characterize") {
        characterize_against_ffmpeg();
    } else if let Some(path) = args.first() {
        measure_real_material(path);
    } else {
        println!(
            "\n(pass a raw interleaved-stereo f32le PCM file path as the first argument to \
             also verify against real material, e.g. a WAV converted with \
             `ffmpeg -i mix.wav -f f32le -acodec pcm_f32le mix.f32`, or pass \
             `--characterize` to sweep gain/density against an independent ffmpeg \
             true-peak measurement -- requires ffmpeg on PATH)"
        );
    }
}

/// Characterize the internal (self) true-peak estimator's error against an
/// INDEPENDENT measurement, across gain and signal density, so the finding
/// is a scaling relationship rather than a single number. `ffmpeg`'s
/// `loudnorm` filter analysis pass emits `input_tp`, a higher-precision
/// true-peak reading than the `ebur128` filter's 1-decimal summary --
/// deliberately NOT the same technique as `true_peak_linear` above, which
/// is what's being checked.
///
/// Requires `ffmpeg` on PATH; run manually with
/// `cargo run --example measure_limiter -- --characterize`.
fn characterize_against_ffmpeg() {
    use std::path::PathBuf;
    use std::process::Command;

    // DEBUG cross-check: the same independently-parameterized (48-tap Hann)
    // reference used by tests/ugens.rs's regression test, run here so its
    // agreement with ffmpeg on the SAME rendered audio can be inspected
    // directly instead of trusted blindly.
    fn hann_reference_dbtp(samples: &[f32], oversample: usize) -> f32 {
        const REF_HALF_TAPS: isize = 24;
        let n = samples.len() as isize;
        let get = |i: isize| -> f32 {
            let idx = i.clamp(0, n - 1) as usize;
            samples[idx]
        };
        let sinc = |x: f32| -> f32 {
            if x.abs() < 1e-7 {
                1.0
            } else {
                let px = core::f32::consts::PI * x;
                px.sin() / px
            }
        };
        let bh = |n: f32, taps: f32| -> f32 {
            const A0: f32 = 0.358_75;
            const A1: f32 = 0.488_29;
            const A2: f32 = 0.141_28;
            const A3: f32 = 0.011_68;
            let x = core::f32::consts::TAU * n / taps;
            A0 - A1 * x.cos() + A2 * (2.0 * x).cos() - A3 * (3.0 * x).cos()
        };
        let taps = (2 * REF_HALF_TAPS) as f32;
        let mut peak = 0.0f32;
        for i in 0..samples.len() as isize {
            peak = peak.max(get(i).abs());
            for k in 1..oversample {
                let t = k as f32 / oversample as f32;
                let mut acc = 0.0f32;
                for tap in -REF_HALF_TAPS..REF_HALF_TAPS {
                    let sample = get(i + tap);
                    let dist = tap as f32 - t + 1.0;
                    let window = bh(tap as f32 + REF_HALF_TAPS as f32, taps);
                    acc += sample * sinc(dist) * window;
                }
                peak = peak.max(acc.abs());
            }
        }
        20.0 * peak.max(1e-9).log10()
    }

    fn write_wav_f32(channels: &[Vec<f32>], sample_rate: f32, path: &PathBuf) {
        let num_channels = channels.len() as u16;
        let num_samples = channels.first().map_or(0, |c| c.len());
        let bits_per_sample: u16 = 32;
        let byte_rate = sample_rate as u32 * num_channels as u32 * (bits_per_sample as u32 / 8);
        let block_align = num_channels * (bits_per_sample / 8);
        let data_size = num_samples as u32 * num_channels as u32 * (bits_per_sample as u32 / 8);
        let file_size = 36 + data_size;

        let mut buf: Vec<u8> = Vec::with_capacity(44 + data_size as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        buf.extend_from_slice(&num_channels.to_le_bytes());
        buf.extend_from_slice(&(sample_rate as u32).to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_size.to_le_bytes());
        for i in 0..num_samples {
            for ch in channels {
                let sample = ch.get(i).copied().unwrap_or(0.0);
                buf.extend_from_slice(&sample.to_le_bytes());
            }
        }
        fs::write(path, &buf).expect("failed to write WAV file");
    }

    /// `loudnorm`'s analysis-pass `input_tp`, independent of the TP target
    /// parameter -- more precision than the `ebur128` filter's 1-decimal
    /// `Peak:` summary line.
    fn ffmpeg_true_peak_dbtp(path: &PathBuf) -> f32 {
        let out = Command::new("ffmpeg")
            .args([
                "-nostats",
                "-i",
                path.to_str().unwrap(),
                "-af",
                "loudnorm=I=-14:TP=-1:print_format=json",
                "-f",
                "null",
                "-",
            ])
            .output()
            .expect("failed to run ffmpeg -- is it on PATH?");
        let stderr = String::from_utf8_lossy(&out.stderr);
        let needle = "\"input_tp\"";
        let pos = stderr
            .find(needle)
            .unwrap_or_else(|| panic!("no \"input_tp\" in ffmpeg output:\n{stderr}"));
        let rest = &stderr[pos + needle.len()..];
        let colon = rest.find(':').unwrap();
        let after_colon = &rest[colon + 1..];
        let quote1 = after_colon.find('"').unwrap();
        let after_q1 = &after_colon[quote1 + 1..];
        let quote2 = after_q1.find('"').unwrap();
        after_q1[..quote2]
            .parse()
            .unwrap_or_else(|e| panic!("could not parse input_tp {:?}: {e}", &after_q1[..quote2]))
    }

    /// "sparse": a single overdriven tone plus one short hard transient
    /// burst per channel -- the same shape `measure_stereo_case` above and
    /// `tests/ugens.rs`'s `hot_test_signal_variant` use.
    fn sparse_signal(sr: f32, len: usize, tone_freq: f32, burst_start: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(len);
        for i in 0..len {
            let t = i as f32 / sr;
            let mut s = 1.4 * (core::f32::consts::TAU * tone_freq * t).sin();
            if (burst_start..burst_start + 16).contains(&i) {
                s = if i % 2 == 0 { 1.8 } else { -1.8 };
            }
            v.push(s);
        }
        v
    }

    /// "dense": several simultaneous overdriven tones near/above the
    /// original -1.4..1.8 hot-signal amplitude range, spaced across the
    /// spectrum including close to Nyquist, plus MULTIPLE hard transient
    /// bursts -- closer in harmonic density to real limited/compressed
    /// program material (many simultaneous near-Nyquist components) than
    /// the sparse single-tone case, which is what a purely local 4-point
    /// spline is least equipped to reconstruct accurately.
    fn dense_signal(sr: f32, len: usize, burst_stride: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(len);
        let tones = [1200.0, 3000.0, 6000.0, 9000.0, 12000.0, 16000.0];
        for i in 0..len {
            let t = i as f32 / sr;
            let mut s = 0.0f32;
            for (k, f) in tones.iter().enumerate() {
                let amp = 0.55 - 0.05 * k as f32;
                s += amp * (core::f32::consts::TAU * f * t).sin();
            }
            if i % burst_stride < 12 {
                s = if i % 2 == 0 { 1.9 } else { -1.9 };
            }
            v.push(s);
        }
        v
    }

    let sr = 44100.0;
    let len = 8192usize;
    let dir = std::env::temp_dir().join(format!("microsynth-characterize-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("failed to create scratch dir");

    println!(
        "\n-- estimator error vs. independent ffmpeg measurement --\n\
         density   gain    ceiling   self-estimate(dBTP)   ffmpeg(dBTP)   error(ffmpeg-self)   verdict(ffmpeg)"
    );

    for (density_name, left, right) in [
        (
            "sparse",
            sparse_signal(sr, len, 3000.0, 400),
            sparse_signal(sr, len, 4500.0, 1900),
        ),
        (
            "dense",
            dense_signal(sr, len, 700),
            dense_signal(sr, len, 900),
        ),
    ] {
        for gain_db in [6.0f32, 12.0, 18.0] {
            for ceiling_db in [-1.0f32, -3.0, -6.0] {
                let g = 10f32.powf(gain_db / 20.0);
                let gl: Vec<f32> = left.iter().map(|s| s * g).collect();
                let gr: Vec<f32> = right.iter().map(|s| s * g).collect();
                let block_size = 64;

                let mut engine = Engine::new(EngineConfig {
                    sample_rate: sr,
                    block_size,
                });
                let src = engine.graph_mut().add_node(Box::new(StereoPlaybackSource {
                    left: gl,
                    right: gr,
                    pos: 0,
                }));
                let ceiling_n = engine.graph_mut().add_node(Box::new(Const::new(ceiling_db)));
                let release_n = engine.graph_mut().add_node(Box::new(Const::new(0.05)));
                let lim = engine.graph_mut().add_node(Box::new(Limiter::new()));
                engine.graph_mut().connect(src, lim, 0);
                engine.graph_mut().connect(ceiling_n, lim, 1);
                engine.graph_mut().connect(release_n, lim, 2);
                engine.graph_mut().set_sink(lim);
                engine.prepare();

                let num_blocks = len / block_size + 1;
                let output = engine.render_offline(num_blocks);

                let self_tp = lin_to_db(true_peak_linear(&output[0], 8)).max(lin_to_db(
                    true_peak_linear(&output[1], 8),
                ));
                let hann_tp = hann_reference_dbtp(&output[0], 8).max(hann_reference_dbtp(&output[1], 8));

                let wav_path = dir.join(format!("{density_name}-{gain_db}dB-{ceiling_db}dBTP.wav"));
                write_wav_f32(&output, sr, &wav_path);
                let ffmpeg_tp = ffmpeg_true_peak_dbtp(&wav_path);

                let error = ffmpeg_tp - self_tp;
                let verdict = if ffmpeg_tp <= ceiling_db { "HOLDS" } else { "BREACH" };
                println!(
                    "{density_name:<9} {gain_db:>+5.1}dB  {ceiling_db:>+5.1}dBTP  {self_tp:>10.3}  hann48={hann_tp:>8.3}  {ffmpeg_tp:>10.3}     {error:>+10.3}          {verdict}"
                );
            }
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

/// Two independently-hot channels with different frequency/burst timing.
/// Always runnable, no external material needed -- the permanent regression
/// test (`test_limiter_holds_ceiling_independently_per_stereo_channel` in
/// `tests/ugens.rs`) is this same shape.
fn measure_stereo_case() {
    let sr = 44100.0;
    let mut left = Vec::with_capacity(4096);
    let mut right = Vec::with_capacity(4096);
    for i in 0..4096usize {
        let t = i as f32 / sr;
        let mut l = 1.4 * (core::f32::consts::TAU * 3000.0 * t).sin();
        if (400..416).contains(&i) {
            l = if i % 2 == 0 { 1.8 } else { -1.8 };
        }
        let mut r = 1.4 * (core::f32::consts::TAU * 4500.0 * t).sin();
        if (1900..1916).contains(&i) {
            r = if i % 2 == 0 { 1.8 } else { -1.8 };
        }
        left.push(l);
        right.push(r);
    }

    let mut engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: 64,
    });
    let src = engine.graph_mut().add_node(Box::new(StereoPlaybackSource {
        left,
        right,
        pos: 0,
    }));
    let ceiling_n = engine.graph_mut().add_node(Box::new(Const::new(-1.0)));
    let release_n = engine.graph_mut().add_node(Box::new(Const::new(0.05)));
    let lim = engine.graph_mut().add_node(Box::new(Limiter::new()));
    engine.graph_mut().connect(src, lim, 0);
    engine.graph_mut().connect(ceiling_n, lim, 1);
    engine.graph_mut().connect(release_n, lim, 2);
    engine.graph_mut().set_sink(lim);
    engine.prepare();

    let output = engine.render_offline(4096 / 64);
    let left_tp = lin_to_db(true_peak_linear(&output[0], 8));
    let right_tp = lin_to_db(true_peak_linear(&output[1], 8));
    println!(
        "stereo (different content per channel): left true-peak {left_tp:.3} dBTP {}, right true-peak {right_tp:.3} dBTP {}",
        if left_tp <= -1.0 { "HOLDS" } else { "EXCEEDS" },
        if right_tp <= -1.0 { "HOLDS" } else { "EXCEEDS" },
    );
}

/// Loads a raw interleaved-stereo f32le PCM file (e.g. via
/// `ffmpeg -i mix.wav -f f32le -acodec pcm_f32le mix.f32`) and runs it
/// through the limiter at a range of gains, comparing against an
/// independent ffmpeg true-peak measurement is left to the caller (this
/// only reports this crate's own oversampled estimate). Deliberately reads
/// from an external path rather than a bundled fixture -- see the module
/// doc for why.
fn measure_real_material(path: &str) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let mut left = Vec::with_capacity(bytes.len() / 8);
    let mut right = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        left.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        right.push(f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]));
    }
    println!(
        "\n-- Real material: {path} ({:.1}s) --",
        left.len() as f32 / 44100.0
    );

    for gain_db in [0.0, 6.0, 12.0, 18.0, 24.0] {
        let sr = 44100.0;
        let g = 10f32.powf(gain_db / 20.0);
        let gl: Vec<f32> = left.iter().map(|s| s * g).collect();
        let gr: Vec<f32> = right.iter().map(|s| s * g).collect();
        let block_size = 64;

        let mut engine = Engine::new(EngineConfig {
            sample_rate: sr,
            block_size,
        });
        let src = engine.graph_mut().add_node(Box::new(StereoPlaybackSource {
            left: gl,
            right: gr,
            pos: 0,
        }));
        let ceiling_n = engine.graph_mut().add_node(Box::new(Const::new(-1.0)));
        let release_n = engine.graph_mut().add_node(Box::new(Const::new(0.05)));
        let lim = engine.graph_mut().add_node(Box::new(Limiter::new()));
        engine.graph_mut().connect(src, lim, 0);
        engine.graph_mut().connect(ceiling_n, lim, 1);
        engine.graph_mut().connect(release_n, lim, 2);
        engine.graph_mut().set_sink(lim);
        engine.prepare();

        let num_blocks = left.len() / block_size + 1;
        let output = engine.render_offline(num_blocks);
        let left_tp = lin_to_db(true_peak_linear(&output[0], 8));
        let right_tp = lin_to_db(true_peak_linear(&output[1], 8));
        println!(
            "  gain={gain_db:+.1}dB: left true-peak {left_tp:.3} dBTP {}, right true-peak {right_tp:.3} dBTP {}",
            if left_tp <= -0.95 { "HOLDS" } else { "EXCEEDS" },
            if right_tp <= -0.95 {
                "HOLDS"
            } else {
                "EXCEEDS"
            },
        );
    }
}

fn measure_edge_cases() {
    let sr = 44100.0;

    // Silence: should stay silent, no NaN.
    let silence = vec![0.0f32; 512];
    let mut engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: 64,
    });
    let src = engine.graph_mut().add_node(Box::new(PlaybackSource {
        data: silence,
        pos: 0,
    }));
    let ceiling_n = engine.graph_mut().add_node(Box::new(Const::new(-1.0)));
    let release_n = engine.graph_mut().add_node(Box::new(Const::new(0.05)));
    let lim = engine.graph_mut().add_node(Box::new(Limiter::new()));
    engine.graph_mut().connect(src, lim, 0);
    engine.graph_mut().connect(ceiling_n, lim, 1);
    engine.graph_mut().connect(release_n, lim, 2);
    engine.graph_mut().set_sink(lim);
    engine.prepare();
    let output = engine.render_offline(8);
    let max = output[0]
        .iter()
        .cloned()
        .fold(0.0f32, |a, b| a.max(b.abs()));
    let has_nan = output[0].iter().any(|x| x.is_nan());
    println!("silence: max |out| = {max}, has_nan = {has_nan}");

    // Full-scale DC.
    let dc = vec![1.0f32; 512];
    let mut engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: 64,
    });
    let src = engine
        .graph_mut()
        .add_node(Box::new(PlaybackSource { data: dc, pos: 0 }));
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
    println!(
        "full-scale DC: settled output = {last:.4} ({:.3} dBFS)",
        lin_to_db(last.abs())
    );

    // Alternating near-Nyquist burst only (the most adversarial local case).
    let mut burst = vec![0.0f32; 512];
    for (i, v) in burst.iter_mut().enumerate().take(260).skip(200) {
        *v = if i % 2 == 0 { 1.8 } else { -1.8 };
    }
    let input_tp = true_peak_linear(&burst, 8);
    let mut engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: 64,
    });
    let src = engine.graph_mut().add_node(Box::new(PlaybackSource {
        data: burst,
        pos: 0,
    }));
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
        if lin_to_db(true_peak) <= -1.0 {
            "HOLDS -1 dBTP"
        } else {
            "EXCEEDS -1 dBTP"
        }
    );
}
