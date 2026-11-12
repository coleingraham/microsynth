//! The resampler measurement behind `sampleVoice`'s default kernel.
//!
//! Each kernel (0 linear, 1 cubic Hermite, 2 step-tracking windowed sinc) plays
//! a harmonic test tone transposed -12, -7, 0, +7 and +12 semitones through the
//! voice itself (keycenter 60, keytrack 100). From a Blackman-Harris-windowed
//! 32768-point spectrum of the steady state:
//!
//! - **aliasing**: the strongest spectral peak more than 8 bins away from every
//!   expected harmonic (`k · f_out` below Nyquist), in dB below the
//!   fundamental. Interpolation images and harmonics folded back from above
//!   Nyquist both land there.
//! - **pitch error**: the fundamental's peak, refined by a parabola on the log
//!   magnitude, against `f0 · 2^(s/12)`, in cents.
//!
//! The synthetic tone is a band-limited sawtooth on C4 (harmonics 1/k up to
//! 20 kHz) at 44.1 kHz, plus the same tone recorded at 48 kHz played at the
//! keycenter (a plain rate conversion). Run with `--nocapture` to print the
//! table. A real catalog sample is measured the same way when
//! `SV_REAL_SAMPLE` names a raw little-endian f32 mono file
//! (`SV_REAL_RATE` its rate, `SV_REAL_F0` its fundamental in Hz); the
//! report's second table comes from that run.

#![cfg(feature = "ir")]

use microsynth::ir::{IrEdge, IrNode, IrParam, IrSynthDef, IrTableBinding, SynthDefClass};
use microsynth::sampler::program::{Envelope, Region, SampleProgram};
use microsynth::sampler::{SamplePcm, SamplePcmBank, SampleProgramBank};
use microsynth::*;

const SR: f32 = 44_100.0;
const N: usize = 32_768;
const SEMITONES: [i32; 5] = [-12, -7, 0, 7, 12];
const KERNELS: [(&str, f32); 3] = [("linear", 0.0), ("hermite", 1.0), ("sinc", 2.0)];

fn saw(f0: f64, rate: f64, frames: usize) -> Vec<f32> {
    let kmax = (20_000.0 / f0) as usize;
    (0..frames)
        .map(|i| {
            let t = i as f64 / rate;
            let mut v = 0.0;
            for k in 1..=kmax {
                v += (core::f64::consts::TAU * k as f64 * f0 * t).sin() / k as f64;
            }
            (v * 0.3) as f32
        })
        .collect()
}

/// Render `secs` of the voice playing `pcm` (keycenter 60) at `key` with kernel `q`.
fn play(pcm: &SamplePcm, key: f32, q: f32, secs: f32) -> Vec<f32> {
    let mut sb = SamplePcmBank::new();
    sb.register(pcm.clone());
    let mut pb = SampleProgramBank::new();
    let prog = SampleProgram {
        name: "probe".into(),
        spec_sha256: [0; 32],
        samples: vec![*pcm.hash()],
        regions: vec![Region {
            amp_veltrack: 0.0,
            ampeg: Envelope::default(),
            ..Region::default()
        }],
    };
    let pid = pb.register(prog, &sb).unwrap();
    let mut reg = dsl::UGenRegistry::new();
    ugens::register_builtins(&mut reg);
    ugens::register_table_bound_builtins(&mut reg);
    let p = |name: &str, default: f32, port: usize| {
        (
            IrNode::Param {
                name: name.into(),
                default,
            },
            port,
        )
    };
    let params = [
        p("freq", 440.0 * ((key - 69.0) / 12.0).exp2(), 0),
        p("quality", q, 6),
    ];
    let mut nodes: Vec<IrNode> = params.iter().map(|x| x.0.clone()).collect();
    nodes.push(IrNode::UGen {
        kind: "sampleVoice".into(),
        consts: vec![],
    });
    let ir = IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "probe".into(),
        class: SynthDefClass::Source,
        output_channels: 2,
        nodes,
        edges: params
            .iter()
            .enumerate()
            .map(|(i, x)| IrEdge {
                from: i,
                to: 2,
                to_input: x.1,
            })
            .collect(),
        params: vec![IrParam {
            name: "freq".into(),
            node: 0,
            input: 0,
            default: 0.0,
        }],
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding {
            node: 2,
            table_id: pid.0,
        }],
        output_node: 2,
    };
    let def = ir
        .compile_with_banks(&reg, &CoeffTableBank::new(), Some(&pb))
        .unwrap();
    let mut engine = Engine::new(EngineConfig {
        sample_rate: SR,
        block_size: 128,
    });
    let s = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(s.output_node());
    engine.prepare();
    let mut out = Vec::new();
    while out.len() < (secs * SR) as usize {
        out.extend_from_slice(engine.render().unwrap().channel(0).samples());
    }
    out
}

/// In-place radix-2 FFT of (re, im).
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -core::f64::consts::TAU / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (ang * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * c - im[b] * s;
                let ti = re[b] * s + im[b] * c;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// (aliasing dB below the fundamental, pitch error in cents, level of the
/// harmonic nearest 16 kHz against its 1/k share, dB).
fn measure(x: &[f32], f_expected: f64) -> (f64, f64, f64) {
    let mut re: Vec<f64> = (0..N)
        .map(|i| {
            let t = i as f64 / (N - 1) as f64 * core::f64::consts::TAU;
            let w =
                0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos() - 0.01168 * (3.0 * t).cos();
            x[i] as f64 * w
        })
        .collect();
    let mut im = vec![0.0; N];
    fft(&mut re, &mut im);
    let mag: Vec<f64> = (0..N / 2)
        .map(|i| (re[i] * re[i] + im[i] * im[i]).sqrt().max(1e-30))
        .collect();
    let df = SR as f64 / N as f64;
    // a harmonic's peak near its expected bin, parabolic refinement
    let peak = |f: f64| -> (f64, f64) {
        let c = (f / df).round() as usize;
        let k0 = (c - 6..=c + 6)
            .max_by(|&a, &b| mag[a].total_cmp(&mag[b]))
            .unwrap();
        let (a, b, g) = (mag[k0 - 1].ln(), mag[k0].ln(), mag[k0 + 1].ln());
        let delta = 0.5 * (a - g) / (a - 2.0 * b + g);
        ((k0 as f64 + delta) * df, mag[k0])
    };
    let (f_meas, fund) = peak(f_expected);
    // pitch from the highest harmonic below 5 kHz (the estimator's bias
    // shrinks by its number)
    let kh = (5_000.0 / f_expected).floor().max(1.0);
    let (fh, _) = peak(kh * f_expected);
    let cents = 1200.0 * (fh / (kh * f_expected)).log2();
    let k16 = (16_000.0 / f_expected).round();
    let (_, a16) = peak(k16 * f_expected);
    let droop = 20.0 * (a16 / fund * k16).log10();
    // strongest non-harmonic peak
    let mut harmonic = vec![false; N / 2];
    let mut k = 1.0;
    while k * f_meas < SR as f64 / 2.0 {
        let c = (k * f_meas / df).round() as isize;
        for i in c - 8..=c + 8 {
            if i >= 0 && (i as usize) < N / 2 {
                harmonic[i as usize] = true;
            }
        }
        k += 1.0;
    }
    let mut worst: f64 = 1e-30;
    for i in 16..N / 2 - 2 {
        if !harmonic[i] {
            worst = worst.max(mag[i]);
        }
    }
    (20.0 * (worst / fund).log10(), cents, droop)
}

fn table(
    name: &str,
    pcm: &SamplePcm,
    f0: f64,
    rows: &[(String, f32, f64)],
) -> Vec<(String, String, f64, f64)> {
    let mut out = Vec::new();
    println!(
        "\n{name}\n\n| transposition | kernel | aliasing (dB re fundamental) | pitch error (cents) | 16 kHz harmonic vs 1/k (dB) | render time (x real time) |"
    );
    println!("|---|---|---:|---:|---:|---:|");
    for (label, key, f_out) in rows {
        let dur = (N as f32 / SR) + 0.3;
        for (kname, q) in KERNELS {
            let t = std::time::Instant::now();
            let y = play(pcm, *key, q, dur);
            let speed = dur as f64 / t.elapsed().as_secs_f64();
            let (alias, cents, droop) = measure(&y[(0.2 * SR) as usize..], *f_out);
            println!("| {label} | {kname} | {alias:.1} | {cents:+.3} | {droop:+.2} | {speed:.0} |");
            out.push((label.clone(), kname.to_string(), alias, cents));
        }
    }
    let _ = f0;
    out
}

#[test]
fn resampler_table_on_a_synthetic_sawtooth() {
    let f0 = 261.625_565_300_6;
    let pcm = SamplePcm::new("saw", [1; 32], SR, &[&saw(f0, SR as f64, 4 * 44_100)[..]]).unwrap();
    let rows: Vec<(String, f32, f64)> = SEMITONES
        .iter()
        .map(|&s| {
            (
                format!("{s:+} st"),
                60.0 + s as f32,
                f0 * (s as f64 / 12.0).exp2(),
            )
        })
        .collect();
    let res = table("synthetic sawtooth, 44.1 kHz file", &pcm, f0, &rows);
    // the same tone recorded at 48 kHz, at the keycenter: a rate conversion
    let pcm48 = SamplePcm::new(
        "saw48",
        [2; 32],
        48_000.0,
        &[&saw(f0, 48_000.0, 4 * 48_000)[..]],
    )
    .unwrap();
    let res48 = table(
        "synthetic sawtooth, 48 kHz file",
        &pcm48,
        f0,
        &[("0 st".into(), 60.0, f0)],
    );

    let get = |rs: &[(String, String, f64, f64)], l: &str, k: &str| {
        rs.iter()
            .find(|r| r.0 == l && r.1 == k)
            .map(|r| (r.2, r.3))
            .unwrap()
    };
    for s in SEMITONES {
        let l = format!("{s:+} st");
        let (sinc_alias, sinc_cents) = get(&res, &l, "sinc");
        let (lin_alias, _) = get(&res, &l, "linear");
        assert!(sinc_cents.abs() < 0.01, "{l}: sinc pitch {sinc_cents}");
        if s != 0 {
            assert!(sinc_alias < -60.0, "{l}: sinc aliasing {sinc_alias}");
            assert!(
                sinc_alias < lin_alias - 20.0,
                "{l}: sinc not better than linear"
            );
        }
    }
    let (a48, c48) = get(&res48, "0 st", "sinc");
    assert!(a48 < -60.0 && c48.abs() < 0.01, "48 kHz: {a48} {c48}");
}

#[test]
fn resampler_table_on_a_real_sample() {
    let Ok(path) = std::env::var("SV_REAL_SAMPLE") else {
        println!("SV_REAL_SAMPLE not set: skipped");
        return;
    };
    let rate: f32 = std::env::var("SV_REAL_RATE").unwrap().parse().unwrap();
    let f0: f64 = std::env::var("SV_REAL_F0").unwrap().parse().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let x: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let pcm = SamplePcm::new("real", [3; 32], rate, &[&x[..]]).unwrap();
    // reference pitch: the sample played at its keycenter
    let rows: Vec<(String, f32, f64)> = SEMITONES
        .iter()
        .map(|&s| {
            (
                format!("{s:+} st"),
                60.0 + s as f32,
                f0 * (s as f64 / 12.0).exp2(),
            )
        })
        .collect();
    table(&format!("real sample {path}"), &pcm, f0, &rows);
}
