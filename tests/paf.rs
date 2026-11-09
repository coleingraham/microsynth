//! `paf`: the phase-aligned formant generator.
//!
//! The spectral peak must sit at `center`, the −6 dB width must be `bw`,
//! every line must be a harmonic of `freq` wherever `center` lies, a sweep of
//! `center` across harmonics must be continuous, and the node must stay
//! bounded and finite for any input.

use microsynth::dsl;
use microsynth::node::UGen;
use microsynth::ugens::Paf;
use microsynth::*;

mod common;
use common::builtin_registry;

const SR: f32 = 44100.0;
const BLOCK: usize = 64;

/// Render `secs` of a single-def source as mono samples at `sr`.
fn render_at(src: &str, secs: f32, sr: f32) -> Vec<f32> {
    let registry = builtin_registry();
    let def = dsl::compile(src, &registry)
        .unwrap_or_else(|e| panic!("compile failed: {e:?}\n{src}"))
        .remove(0);
    let mut engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: BLOCK,
    });
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let blocks = (secs * sr / BLOCK as f32).ceil() as usize;
    let mut out = Vec::with_capacity(blocks * BLOCK);
    for _ in 0..blocks {
        let buf = engine.render().expect("a sink is set");
        out.extend_from_slice(buf.channel(0).samples());
    }
    out.truncate((secs * sr) as usize);
    out
}

fn render(src: &str, secs: f32) -> Vec<f32> {
    render_at(src, secs, SR)
}

fn paf(f0: f32, center: f32, bw: f32) -> String {
    format!("synthdef p = paf {f0:.6} {center:.6} {bw:.6} 1.0 0.0")
}

/// Amplitude at `freq` of a Hann-windowed DFT, normalised so a unit sine
/// reads ~1.
fn bin_at(x: &[f32], freq: f64, sr: f64) -> f64 {
    let len = x.len() as f64;
    let (mut re, mut im, mut wsum) = (0.0f64, 0.0f64, 0.0f64);
    for (n, &s) in x.iter().enumerate() {
        let win = 0.5 - 0.5 * (core::f64::consts::TAU * n as f64 / len).cos();
        let w = core::f64::consts::TAU * freq * n as f64 / sr;
        re += win * s as f64 * w.cos();
        im -= win * s as f64 * w.sin();
        wsum += win;
    }
    2.0 * (re * re + im * im).sqrt() / wsum
}

fn bin(x: &[f32], freq: f64) -> f64 {
    bin_at(x, freq, SR as f64)
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}

/// An excerpt of whole periods of `f0` after a short settle: `periods`
/// periods long, so harmonics land on bins and leakage is the window's.
fn whole_periods(x: &[f32], f0: f64, periods: usize) -> &[f32] {
    let start = 2048;
    let n = (periods as f64 * SR as f64 / f0).round() as usize;
    &x[start..start + n]
}

#[test]
fn the_strongest_line_is_the_centre_harmonic() {
    for &(f0, center, bw) in &[
        (110.0f32, 880.0f32, 150.0f32),
        (220.0, 660.0, 300.0),
        (98.0, 2450.0, 400.0),
        (440.0, 3520.0, 600.0),
        (130.81, 392.43, 120.0),
    ] {
        let x = render(&paf(f0, center, bw), 1.5);
        let seg = whole_periods(&x, f0 as f64, 100);
        let k = (center / f0).round() as usize;
        let lines: Vec<f64> = (1..=(8000.0 / f0) as usize)
            .map(|m| bin(seg, m as f64 * f0 as f64))
            .collect();
        let (best, peak) =
            lines.iter().enumerate().fold(
                (0, 0.0),
                |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
            );
        assert_eq!(
            best + 1,
            k,
            "f0 {f0} center {center}: peak at harmonic {}",
            best + 1
        );
        // amp is the spectral peak: the centre line reads 1
        assert!(
            (peak - 1.0).abs() < 0.01,
            "f0 {f0} center {center}: peak {peak}"
        );
    }
}

#[test]
fn a_centre_between_harmonics_splits_the_peak_as_the_crossfade_says() {
    // c = k + alpha: lines k and k+1 are (1 − α) + α r and (1 − α) r + α
    let (f0, bw) = (200.0f64, 500.0f64);
    let r = (-2.0 * f0 / bw).exp2();
    for &alpha in &[0.25f64, 0.5, 0.8] {
        let center = (5.0 + alpha) * f0;
        let x = render(&paf(f0 as f32, center as f32, bw as f32), 1.5);
        let seg = whole_periods(&x, f0, 100);
        let lk = bin(seg, 5.0 * f0);
        let lk1 = bin(seg, 6.0 * f0);
        let want_k = (1.0 - alpha) + alpha * r;
        let want_k1 = (1.0 - alpha) * r + alpha;
        assert!(
            (db(lk) - db(want_k)).abs() < 0.1,
            "α {alpha}: {lk} vs {want_k}"
        );
        assert!(
            (db(lk1) - db(want_k1)).abs() < 0.1,
            "α {alpha}: {lk1} vs {want_k1}"
        );
        // and the two are the strongest lines
        for m in [3usize, 4, 7, 8] {
            assert!(bin(seg, m as f64 * f0) < lk.min(lk1), "α {alpha} m {m}");
        }
    }
}

/// The −6 dB full width of the line envelope around `center`, interpolating
/// the line levels (in dB) linearly in frequency.
fn minus6_width(seg: &[f32], f0: f64, center: f64) -> f64 {
    let k = (center / f0).round() as i64;
    let peak = db(bin(seg, k as f64 * f0));
    let edge = |dir: i64| -> f64 {
        let mut prev = (k as f64 * f0, peak);
        let mut m = k + dir;
        while m >= 1 {
            let f = m as f64 * f0;
            let l = db(bin(seg, f));
            if l <= peak - 6.0206 {
                let t = (prev.1 - (peak - 6.0206)) / (prev.1 - l);
                return prev.0 + t * (f - prev.0);
            }
            prev = (f, l);
            m += dir;
        }
        0.0
    };
    edge(1) - edge(-1)
}

#[test]
fn the_minus_6_db_bandwidth_is_bw() {
    for &(f0, center, bw) in &[
        (100.0f64, 1000.0f64, 400.0f64),
        (110.0, 2420.0, 900.0),
        (130.81, 784.86, 300.0),
        (65.41, 523.28, 160.0),
        (196.0, 3136.0, 1500.0),
        (55.0, 4400.0, 250.0),
    ] {
        let x = render(&paf(f0 as f32, center as f32, bw as f32), 2.0);
        let seg = whole_periods(&x, f0, 80);
        let w = minus6_width(seg, f0, center);
        eprintln!(
            "f0 {f0} center {center} bw {bw}: -6 dB width {w:.1} ({:+.1} %)",
            100.0 * (w / bw - 1.0)
        );
        assert!(
            (w / bw - 1.0).abs() < 0.02,
            "f0 {f0} center {center} bw {bw}: −6 dB width {w:.1}"
        );
    }
}

#[test]
fn every_line_is_a_harmonic_of_freq() {
    // centre off any harmonic, narrow and wide: energy between the harmonics
    // stays at the window's leakage floor (no component at `center`, no beating)
    for &(f0, center, bw) in &[
        (110.0f64, 913.0f64, 120.0f64),
        (220.0, 1371.0, 700.0),
        (82.41, 2777.0, 300.0),
    ] {
        let x = render(&paf(f0 as f32, center as f32, bw as f32), 2.5);
        let seg = whole_periods(&x, f0, 160);
        let peak = (1..=60)
            .map(|m| bin(seg, m as f64 * f0))
            .fold(0.0f64, f64::max);
        let mut worst = f64::MIN;
        for m in 1..60 {
            for frac in [0.25f64, 0.5, 0.75] {
                worst = worst.max(db(bin(seg, (m as f64 + frac) * f0)) - db(peak));
            }
        }
        // and nothing at the centre frequency itself beyond the harmonics' skirts
        let at_center = db(bin(seg, center)) - db(peak);
        eprintln!(
            "f0 {f0} center {center} bw {bw}: between-harmonic {worst:.1} dB, at center {at_center:.1} dB"
        );
        assert!(
            worst < -80.0,
            "f0 {f0} center {center}: between-harmonic {worst:.1} dB"
        );
        assert!(
            at_center < -80.0,
            "f0 {f0} center {center}: at center {at_center:.1} dB"
        );
    }
}

#[test]
fn the_output_has_no_dc_even_when_the_formant_reaches_0_hz() {
    for &(f0, center, bw) in &[
        (200.0f32, 300.0f32, 600.0f32),
        (100.0, 50.0, 400.0),
        (150.0, 0.0, 300.0),
        (220.0, 880.0, 5000.0),
    ] {
        let x = render(&paf(f0, center, bw), 1.0);
        let seg = whole_periods(&x, f0 as f64, 40);
        let mean = seg.iter().map(|&v| v as f64).sum::<f64>() / seg.len() as f64;
        let rms = (seg.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / seg.len() as f64).sqrt();
        assert!(
            mean.abs() < 1e-4 * rms.max(1.0),
            "f0 {f0} center {center} bw {bw}: mean {mean} rms {rms}"
        );
    }
}

/// RMS over windows of `win` samples.
fn window_rms(x: &[f32], win: usize) -> Vec<f64> {
    x.chunks_exact(win)
        .map(|c| (c.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / win as f64).sqrt())
        .collect()
}

#[test]
fn a_centre_sweep_across_harmonics_is_continuous() {
    // f0 200 Hz, centre 600 → 1400 Hz over 2 s (four harmonics crossed)
    let f0 = 200.0f64;
    for &bw in &[400.0f64, 800.0] {
        let src = format!("synthdef s = paf {f0:.1} (line 600.0 1400.0 2.0) {bw:.1} 1.0 0.0");
        let x = render(&src, 2.0);
        let steady: Vec<Vec<f32>> = [600.0f32, 800.0, 1000.0, 1200.0, 1400.0]
            .iter()
            .map(|&c| render(&paf(f0 as f32, c, bw as f32), 0.5))
            .collect();
        // no clicks: the largest sample step stays within the steady tone's
        let step = |v: &[f32]| {
            v.windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0f32, f32::max)
        };
        let s_sweep = step(&x[1000..]);
        let s_steady = steady
            .iter()
            .map(|v| step(&v[1000..]))
            .fold(0.0f32, f32::max);
        assert!(
            s_sweep < 1.15 * s_steady,
            "bw {bw}: step {s_sweep} vs steady {s_steady}"
        );
        // level ripple: RMS over whole periods (5 ms = 220.5 samples, so 4 periods)
        let win = (4.0 * SR as f64 / f0).round() as usize;
        let rms = window_rms(&x[2048..x.len() - 2048], win);
        let (lo, hi) = rms
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let ripple = db(hi) - db(lo);
        eprintln!(
            "sweep bw {bw}: ripple {ripple:.2} dB, max step {s_sweep:.3} (steady {s_steady:.3})"
        );
        assert!(ripple < 1.0, "bw {bw}: level ripple {ripple:.2} dB");
    }
}

#[test]
fn a_narrow_formant_between_harmonics_dips_by_the_documented_amount() {
    // power at α = 0.5 relative to α = 0: (1 + r)² / (2(1 + r²))
    let f0 = 200.0f64;
    for &bw in &[100.0f64, 200.0, 250.0] {
        let r = (-2.0 * f0 / bw).exp2();
        let want = 10.0 * ((1.0 + r).powi(2) / (2.0 * (1.0 + r * r))).log10();
        let rms_of = |center: f64| {
            let x = render(&paf(f0 as f32, center as f32, bw as f32), 1.0);
            let seg = whole_periods(&x, f0, 100);
            (seg.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / seg.len() as f64).sqrt()
        };
        let got = db(rms_of(1100.0)) - db(rms_of(1000.0));
        assert!(
            (got - want).abs() < 0.15,
            "bw {bw}: dip {got:.2} dB, formula {want:.2} dB"
        );
    }
}

#[test]
fn aliasing_follows_the_documented_bound() {
    // a wide formant on a high note: the first line past Nyquist is
    // −12 dB · (fs/2 − center) / bw re the peak, and folded lines stay under it
    let (f0, center, bw) = (1800.0f64, 9000.0f64, 6000.0f64);
    let x = render(&paf(f0 as f32, center as f32, bw as f32), 1.0);
    let seg = &x[2048..2048 + 32768];
    let peak = bin(seg, 5.0 * f0);
    let bound = -12.0412 * (SR as f64 / 2.0 - center) / bw;
    // the strongest line not at a harmonic below Nyquist
    let mut worst = f64::MIN;
    let mut f = 50.0;
    while f < SR as f64 / 2.0 - 100.0 {
        let near_harmonic = ((f / f0).round() * f0 - f).abs() < 60.0;
        if !near_harmonic {
            worst = worst.max(db(bin(seg, f)) - db(peak));
        }
        f += 10.0;
    }
    eprintln!("alias: worst non-harmonic {worst:.1} dB, bound {bound:.1} dB");
    assert!(
        worst < bound + 3.0,
        "worst alias {worst:.1} dB, bound {bound:.1} dB"
    );
    // and a speech-like formant has nothing measurable folded back
    let x = render(&paf(220.0, 3000.0, 250.0), 1.0);
    let seg = whole_periods(&x, 220.0, 120);
    let peak = bin(seg, 3080.0);
    let mid = (1..60)
        .map(|m| db(bin(seg, (m as f64 + 0.5) * 220.0)) - db(peak))
        .fold(f64::MIN, f64::max);
    assert!(mid < -90.0, "speech formant: {mid:.1} dB between harmonics");
}

#[test]
fn bounded_and_finite_for_extreme_inputs() {
    let bound = (1.0 + 0.99) / (1.0 - 0.99) + 1.0 + 1e-3;
    let vals_f0 = [
        "0.0",
        "0.000001",
        "20.0",
        "2000.0",
        "20000.0",
        "1000000000.0",
        "(0.0 - 5.0)",
    ];
    let vals_c = ["0.0", "(0.0 - 100.0)", "8000.0", "21000.0", "1000000000.0"];
    let vals_bw = ["0.0", "(0.0 - 1.0)", "0.000000001", "50.0", "1000000000.0"];
    for f0 in vals_f0 {
        for c in vals_c {
            for bw in vals_bw {
                let x = render(&format!("synthdef b = paf {f0} {c} {bw} 1.0 0.0"), 0.05);
                for &v in &x {
                    assert!(v.is_finite(), "f0 {f0} c {c} bw {bw}: non-finite");
                    assert!((v as f64).abs() <= bound, "f0 {f0} c {c} bw {bw}: {v}");
                }
            }
        }
    }
    // non-finite inputs read as their defaults (220 Hz, 800 Hz, 100 Hz, 1)
    let mut node = Paf::new();
    let ctx = ProcessContext::new(SR, BLOCK);
    node.init(&ctx);
    let mut reference = Paf::new();
    reference.init(&ctx);
    let mut bad = AudioBuffer::new(1, BLOCK);
    let mut out = AudioBuffer::new(1, BLOCK);
    let mut want = AudioBuffer::new(1, BLOCK);
    for (n, v) in bad.channel_mut(0).samples_mut().iter_mut().enumerate() {
        *v = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][n % 3];
    }
    for _ in 0..50 {
        node.process(
            &ctx,
            &[Some(&bad), Some(&bad), Some(&bad), Some(&bad), None],
            &mut out,
        );
        reference.process(&ctx, &[None, None, None, None, None], &mut want);
        assert_eq!(out.channel(0).samples(), want.channel(0).samples());
    }
    assert!(want.channel(0).samples().iter().any(|v| v.abs() > 0.01));
    // f0 across 20 Hz .. 2 kHz, centres to 8 kHz, bandwidths 30 Hz .. 3 kHz
    for f0 in [20.0f32, 55.0, 220.0, 880.0, 2000.0] {
        for c in [100.0f32, 1000.0, 4000.0, 8000.0] {
            for bw in [30.0f32, 300.0, 3000.0] {
                let r = (-2.0 * f0 as f64 / bw as f64).exp2().min(0.99);
                let pk = (1.0 + r) / (1.0 - r) + 1.0;
                let x = render(&paf(f0, c, bw), 0.1);
                let m = x.iter().fold(0.0f32, |a, v| a.max(v.abs())) as f64;
                assert!(m <= pk * 1.0001, "f0 {f0} c {c} bw {bw}: peak {m} > {pk}");
            }
        }
    }
}

#[test]
fn zero_bandwidth_is_the_bare_carriers() {
    // bw ≤ 0: a sine at a harmonic centre, two weighted harmonics between
    let x = render(&paf(200.0, 1000.0, 0.0), 1.0);
    let seg = whole_periods(&x, 200.0, 100);
    assert!((bin(seg, 1000.0) - 1.0).abs() < 0.01);
    assert!(db(bin(seg, 800.0)) < -90.0 && db(bin(seg, 1200.0)) < -90.0);
    let x = render(&paf(200.0, 1050.0, 0.0), 1.0);
    let seg = whole_periods(&x, 200.0, 100);
    assert!((bin(seg, 1000.0) - 0.75).abs() < 0.01);
    assert!((bin(seg, 1200.0) - 0.25).abs() < 0.01);
}

#[test]
fn phase_reset_restarts_the_period() {
    // two renders that differ only in when the reset arrives line up after it
    let a = render("synthdef a = paf 137.0 900.0 300.0 1.0 (impulse 2.0)", 1.0);
    let b = render("synthdef b = paf 137.0 900.0 300.0 1.0 0.0", 1.0);
    // after the impulse's second click `a` is `b` restarted, so it matches `b`
    // from its own start
    let clicks: Vec<usize> = render("synthdef i = impulse 2.0", 1.0)
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > 0.0)
        .map(|(i, _)| i)
        .collect();
    assert!(clicks.len() >= 2, "{clicks:?}");
    let start = clicks[1];
    let n = 4000;
    let d = a[start..start + n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(d < 1e-3, "after reset: max diff {d}");
}

#[test]
fn audio_rate_freq_keeps_the_formant_in_place() {
    // a vibrato on f0 moves the harmonics under a fixed formant: the line
    // nearest the centre stays the strongest and the output stays bounded
    let x = render(
        "synthdef v = paf (130.81 * (1.0 + sinOsc 5.5 0.0 * 0.03)) 730.0 200.0 1.0 0.0",
        2.0,
    );
    assert!(x.iter().all(|v| v.is_finite() && v.abs() < 20.0));
    let frames: Vec<f64> = x[4096..]
        .chunks(4410)
        .map(|c| (c.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / c.len() as f64).sqrt())
        .collect();
    let (lo, hi) = frames
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    // FM through a fixed formant is AM of the partials, but the total level moves little
    assert!(
        db(hi) - db(lo) < 3.0,
        "level swing {:.2} dB",
        db(hi) - db(lo)
    );
}

#[test]
fn a_vowel_is_a_sum_of_pafs_and_the_doc_example_compiles() {
    let src = "synthdef ah freq=220.0 gate=1.0 =\n  \
               let v = paf freq 730.0 160.0 1.0 0.0 + paf freq 1090.0 170.0 0.5 0.0 + paf freq 2440.0 200.0 0.25 0.0\n  \
               v * asr gate 0.05 0.2 * 0.2";
    let x = render(src, 1.0);
    let seg = whole_periods(&x, 220.0, 100);
    // three peaks near the formants: harmonics 3 (660), 5 (1100), 11 (2420)
    let l = |m: usize| db(bin(seg, m as f64 * 220.0));
    assert!(l(3) > l(1) && l(3) > l(2) + 3.0, "F1 peak");
    assert!(l(5) > l(7) + 3.0, "F2 peak");
    assert!(l(11) > l(9) + 3.0 && l(11) > l(14) + 3.0, "F3 peak");
}

#[test]
fn renders_the_same_at_48k_with_the_peak_on_the_harmonic() {
    let x = render_at(&paf(110.0, 880.0, 150.0), 1.5, 48_000.0);
    let n = (100.0 * 48_000.0f64 / 110.0).round() as usize;
    let seg = &x[2048..2048 + n];
    let k8 = bin_at(seg, 880.0, 48_000.0);
    assert!((k8 - 1.0).abs() < 0.01, "{k8}");
    assert!(bin_at(seg, 770.0, 48_000.0) < k8 && bin_at(seg, 990.0, 48_000.0) < k8);
}

#[test]
fn ir_form_with_inline_consts_matches_the_dsl() {
    let registry = builtin_registry();
    let ir_json = r#"{
        "format_version": 3, "name": "pi", "class": "Source", "output_channels": 1,
        "nodes": [
            {"Param": {"name": "freq", "default": 196.0}},
            {"UGen": {"kind": "paf", "consts": [[1, 1500.0], [2, 240.0], [3, 0.7]]}}
        ],
        "edges": [{"from": 0, "to": 1, "to_input": 0}],
        "params": [{"name": "freq", "node": 0, "input": 0, "default": 196.0}],
        "audio_inputs": [],
        "output_node": 1
    }"#;
    let ir = microsynth::ir::IrSynthDef::from_json(ir_json).expect("parse");
    ir.validate(&registry).expect("validate");
    let def = ir.compile(&registry).expect("compile");
    let mut engine = Engine::new(EngineConfig {
        sample_rate: SR,
        block_size: BLOCK,
    });
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let mut y = Vec::new();
    for _ in 0..200 {
        y.extend_from_slice(engine.render().unwrap().channel(0).samples());
    }
    let x = render(
        "synthdef pd = paf 196.0 1500.0 240.0 0.7 0.0",
        200.0 * BLOCK as f32 / SR,
    );
    let d = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(d < 1e-6, "IR vs DSL: {d}");
}
