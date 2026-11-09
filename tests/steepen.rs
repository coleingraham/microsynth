//! `steepen`: waveform steepening with exact de-emphasis.
//!
//! At `drive` 0 the node must be a pure delay; a steepened sine must follow
//! Fubini's harmonic series (smooth, no holes, sawtooth-like at the shock);
//! the de-emphasis must be undone exactly; the steepening must keytrack;
//! and the node must reproduce, sample for sample, the graph of `onePole`
//! and `delay` nodes it replaces.

use microsynth::dsl;
use microsynth::node::UGen;
use microsynth::ugens::{STEEPEN_REF_HZ, Steepen, steepen_pool_len};
use microsynth::*;

mod common;
use common::builtin_registry;

const SR: f32 = 44100.0;
const BLOCK: usize = 64;
const C4: f64 = STEEPEN_REF_HZ;
const TAU: f64 = core::f64::consts::TAU;

// ------------------------------------------------------------------ harness

/// Render `secs` of a single-def source as mono samples at `sr`.
fn render_at(src: &str, secs: f64, sr: f32) -> Vec<f32> {
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
    let n = (secs * sr as f64) as usize;
    let mut out = Vec::with_capacity(n + BLOCK);
    while out.len() < n {
        out.extend_from_slice(engine.render().expect("a sink is set").channel(0).samples());
    }
    out.truncate(n);
    out
}

fn render(src: &str, secs: f64) -> Vec<f32> {
    render_at(src, secs, SR)
}

/// Feed whole signals straight into one `Steepen` (port order: in, freq,
/// drive, k1, k2, stages, ref_hz, ref_amp, tau; `None` = unconnected).
fn run_node(sr: f32, inputs: &[Option<Vec<f32>>]) -> Vec<f32> {
    let mut node = Steepen::new();
    let ctx = ProcessContext::new(sr, BLOCK);
    node.init(&ctx);
    let n = inputs[0].as_ref().unwrap().len();
    let mut out = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let len = BLOCK.min(n - start);
        let bufs: Vec<Option<AudioBuffer>> = inputs
            .iter()
            .map(|s| {
                s.as_ref().map(|v| {
                    let mut b = AudioBuffer::new(1, len);
                    b.channel_mut(0)
                        .samples_mut()
                        .copy_from_slice(&v[start..start + len]);
                    b
                })
            })
            .collect();
        let refs: Vec<Option<&AudioBuffer>> = bufs.iter().map(|b| b.as_ref()).collect();
        let mut o = AudioBuffer::new(1, len);
        node.process(&ctx, &refs, &mut o);
        out.extend_from_slice(o.channel(0).samples());
        start += len;
    }
    out
}

fn konst(v: f32, n: usize) -> Option<Vec<f32>> {
    Some(vec![v; n])
}

fn sine(f: f64, amp: f64, n: usize, sr: f64) -> Vec<f32> {
    (0..n)
        .map(|i| (amp * (TAU * f * i as f64 / sr).sin()) as f32)
        .collect()
}

/// Amplitude at `freq` of a Hann-windowed DFT, normalised so a unit sine
/// reads ~1.
fn bin_at(x: &[f32], freq: f64, sr: f64) -> f64 {
    let len = x.len() as f64;
    let (mut re, mut im, mut wsum) = (0.0f64, 0.0f64, 0.0f64);
    for (n, &s) in x.iter().enumerate() {
        let win = 0.5 - 0.5 * (TAU * n as f64 / len).cos();
        let w = TAU * freq * n as f64 / sr;
        re += win * s as f64 * w.cos();
        im -= win * s as f64 * w.sin();
        wsum += win;
    }
    2.0 * (re * re + im * im).sqrt() / wsum
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}

/// Whole periods of `f0` after a 0.25 s settle (as many as fit, up to
/// 120), so harmonics land on bins.
fn whole_periods(x: &[f32], f0: f64, sr: f64) -> &[f32] {
    let start = (0.25 * sr) as usize;
    let periods = (((x.len() - start) as f64 * f0 / sr).floor()).min(120.0);
    let n = (periods * sr / f0).round() as usize;
    &x[start..(start + n).min(x.len())]
}

/// Harmonics 1..=n in dB re harmonic 1.
fn envelope(x: &[f32], f0: f64, n: usize, sr: f64) -> Vec<f64> {
    let seg = whole_periods(x, f0, sr);
    let h: Vec<f64> = (1..=n).map(|k| bin_at(seg, k as f64 * f0, sr)).collect();
    h.iter().map(|&v| db(v / h[0])).collect()
}

/// Largest dip of h2..h(n-1) below the line through its two neighbours (dB).
fn worst_hole(env: &[f64]) -> f64 {
    (1..env.len() - 1)
        .map(|k| (env[k - 1] + env[k + 1]) / 2.0 - env[k])
        .fold(f64::MIN, f64::max)
}

/// Least-squares slope (dB per harmonic) of h3..=h10.
fn slope_h3_h10(env: &[f64]) -> f64 {
    let xs: Vec<f64> = (3..=10).map(|k| k as f64).collect();
    let ys: Vec<f64> = (3..=10).map(|k| env[k - 1]).collect();
    let (mx, my) = (xs.iter().sum::<f64>() / 8.0, ys.iter().sum::<f64>() / 8.0);
    let num: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    num / den
}

/// Bessel J_n(x) by the trapezoid rule on Bessel's integral (periodic
/// integrand, so it converges geometrically).
fn bessel_j(n: usize, x: f64) -> f64 {
    let m = 4096;
    let mut s = 0.0;
    for i in 0..m {
        let t = core::f64::consts::PI * (i as f64 + 0.5) / m as f64;
        s += (n as f64 * t - x * t.sin()).cos();
    }
    s / m as f64
}

/// Fubini: the harmonics of a sine steepened to index `sigma`, dB re h1.
fn fubini(sigma: f64, n: usize) -> Vec<f64> {
    let b = |k: usize| 2.0 * bessel_j(k, k as f64 * sigma) / (k as f64 * sigma);
    (1..=n).map(|k| db(b(k) / b(1))).collect()
}

/// The digital one-pole `(1 − a)/(1 − a z⁻¹)` magnitude at `f`.
fn pole_gain(k: f64, f0: f64, f: f64, sr: f64) -> f64 {
    let a = (1.0 - TAU * k * f0 / sr).clamp(0.0, 0.9999);
    let w = TAU * f / sr;
    (1.0 - a) / (1.0 - 2.0 * a * w.cos() + a * a).sqrt()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The automatic base delay in samples per stage.
fn auto_t0(f0: f64, stages: usize, sr: f64) -> usize {
    (2.0 * sr / (TAU * f0 * stages as f64)).ceil() as usize
}

// ------------------------------------------------------------------ pass-through

#[test]
fn drive_zero_is_a_pure_delay() {
    // A rich input (harmonics, an inharmonic partial, noise, a DC step):
    // at drive 0 the output is the input delayed by K · t0 samples.
    let n = 20_000;
    let x: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / SR as f64;
            let noise =
                (((i as u64).wrapping_mul(2_654_435_761) % 1000) as f64 / 1000.0 - 0.5) * 0.2;
            let dc = if i > 9000 { 0.25 } else { 0.0 };
            (0.6 * (TAU * 261.6 * t).sin()
                + 0.3 * (TAU * 1046.0 * t + 0.3).sin()
                + 0.2 * (TAU * 3371.0 * t).sin()
                + noise
                + dc) as f32
        })
        .collect();
    for &(f0, stages, k1, k2) in &[
        (C4, 16.0f32, 0.573f32, 1.5f32),
        (130.81, 16.0, 0.573, 1.5),
        (392.0, 8.0, 0.573, 0.0),
        (98.0, 32.0, 0.0, 0.0),
        (1000.0, 1.0, 2.0, 3.0),
    ] {
        let y = run_node(
            SR,
            &[
                Some(x.clone()),
                konst(f0 as f32, n),
                konst(0.0, n),
                konst(k1, n),
                konst(k2, n),
                konst(stages, n),
                None,
                None,
                None,
            ],
        );
        let lat = stages as usize * auto_t0(f0, stages as usize, SR as f64);
        let err = (lat..n)
            .map(|i| (y[i] - x[i - lat]).abs())
            .fold(0.0f32, f32::max);
        let head = y[..lat].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            err <= 1e-6 && head == 0.0,
            "f0 {f0} K {stages} k {k1}/{k2}: latency {lat}, max error {err:e}, head {head}"
        );
    }
}

// ------------------------------------------------------------------ Fubini

/// The stage scheme in f64, as documented: K explicit steps, each sample
/// written then read `t0 − Δ·z` samples back with linear interpolation.
fn scheme(x: &[f64], sigma: f64, f0: f64, stages: usize, sr: f64) -> Vec<f64> {
    let t0 = auto_t0(f0, stages, sr) as f64;
    let gain = sigma * sr / (TAU * f0 * stages as f64);
    let mut z = x.to_vec();
    for _ in 0..stages {
        let src = z.clone();
        for (n, zn) in z.iter_mut().enumerate() {
            let d = (t0 - gain * src[n]).clamp(0.0, 2.0 * t0);
            let (di, fr) = (d as usize, d - (d as usize) as f64);
            let at = |k: usize| if k <= n { src[n - k] } else { 0.0 };
            *zn = at(di) + fr * (at(di + 1) - at(di));
        }
    }
    z
}

#[test]
fn the_stages_are_the_documented_scheme() {
    // no de-emphasis, a sine of peak 1 at drive sigma: the node is the scheme
    let n = 22_050;
    for &(f0, sigma, k) in &[
        (C4, 0.5f64, 16usize),
        (C4, 1.0, 16),
        (110.0, 0.9, 32),
        (523.25, 0.7, 4),
    ] {
        let x = sine(f0, 1.0, n, SR as f64);
        let want = scheme(
            &x.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            sigma,
            f0,
            k,
            SR as f64,
        );
        let y = run_node(
            SR,
            &[
                Some(x),
                konst(f0 as f32, n),
                konst(sigma as f32, n),
                konst(0.0, n),
                konst(0.0, n),
                konst(k as f32, n),
                None,
                None,
                None,
            ],
        );
        let err = y
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-6, "f0 {f0} sigma {sigma} K {k}: {err:e}");
    }
}

/// A sine of peak `amp` at `f0`, steepened at `drive` (ref_amp = `amp`).
fn steepened(f0: f64, amp: f64, drive: f32, k1: f32, k2: f32, stages: f32, sr: f32) -> Vec<f32> {
    let n = (1.0 * sr) as usize;
    run_node(
        sr,
        &[
            Some(sine(f0, amp, n, sr as f64)),
            konst(f0 as f32, n),
            konst(drive, n),
            konst(k1, n),
            konst(k2, n),
            konst(stages, n),
            None,
            konst(amp as f32, n),
            None,
        ],
    )
}

fn steepened_sine(f0: f64, drive: f32, k1: f32, k2: f32, sr: f32) -> Vec<f32> {
    steepened(f0, 1.0, drive, k1, k2, 16.0, sr)
}

/// A steepened sine against Fubini: h2 and h3 close, the rest below it
/// (the explicit stages fall faster than the implicit solution), and the
/// envelope smooth: falling, no hole.
fn check_against_fubini(env: &[f64], sigma: f64, what: &str) {
    let want = fubini(sigma, env.len());
    for k in 1..3 {
        assert!(
            (env[k] - want[k]).abs() <= 0.6,
            "{what}: h{} {:.2} vs Fubini {:.2}",
            k + 1,
            env[k],
            want[k]
        );
    }
    // h4..h8 within 6 dB under it (at K 16 the gap grows by about 0.7 dB
    // per harmonic at the shock), all of them under it
    for k in 3..env.len() {
        let floor = if k < 8 { want[k] - 6.0 } else { f64::MIN };
        assert!(
            env[k] <= want[k] + 0.1 && env[k] >= floor,
            "{what}: h{} {:.2} vs Fubini {:.2}",
            k + 1,
            env[k],
            want[k]
        );
    }
    assert!(
        env.windows(2).all(|w| w[1] < w[0]),
        "{what}: not falling {env:.2?}"
    );
    // the brass rows' rule was 6 dB; Fubini's own series has up to 1.7 dB
    // by this measure (its fall decelerates)
    assert!(
        worst_hole(env) <= 2.0,
        "{what}: hole {:.2} dB in {env:.2?}",
        worst_hole(env)
    );
}

#[test]
fn a_steepened_sine_is_smooth_and_approaches_fubini() {
    // No de-emphasis: the node steepens the sine itself, sigma = drive.
    for &(drive, n_h) in &[(0.5f32, 8usize), (1.0, 12)] {
        let env = envelope(&steepened_sine(C4, drive, 0.0, 0.0, SR), C4, n_h, SR as f64);
        check_against_fubini(&env, drive as f64, &format!("drive {drive}"));
        // more stages, closer to the implicit solution
        let env32 = envelope(
            &steepened(C4, 1.0, drive, 0.0, 0.0, 32.0, SR),
            C4,
            n_h,
            SR as f64,
        );
        let want = fubini(drive as f64, n_h);
        assert!(
            (env32[7] - want[7]).abs() < (env[7] - want[7]).abs(),
            "drive {drive}: h8 K32 {:.2} K16 {:.2} Fubini {:.2}",
            env32[7],
            env[7],
            want[7]
        );
    }
    // at the shock the envelope is sawtooth-like: inside the
    // [-3.5, -1.5] dB/harmonic band the brass rows were held to
    let env = envelope(&steepened_sine(C4, 1.0, 0.0, 0.0, SR), C4, 12, SR as f64);
    let s = slope_h3_h10(&env);
    assert!(
        (-3.5..=-1.5).contains(&s),
        "slope at the shock {s:.2} dB/harmonic {env:.2?}"
    );
}

#[test]
fn the_re_emphasis_undoes_the_de_emphasis_exactly() {
    // Default corners: the de-emphasised sine (peak |H(f0)|) is steepened to
    // the same sigma as a bare sine of that peak, then H^-1 restores the
    // tilt, so harmonic n reads the bare steepening's times |H(f0)|/|H(n f0)|.
    let (k1, k2) = (0.573, 1.5);
    let sr = SR as f64;
    let h = |f: f64| pole_gain(k1, C4, f, sr) * pole_gain(k2, C4, f, sr);
    for &drive in &[0.5f32, 1.0] {
        let env = envelope(
            &steepened_sine(C4, drive, k1 as f32, k2 as f32, SR),
            C4,
            12,
            sr,
        );
        let bare = envelope(&steepened(C4, h(C4), drive, 0.0, 0.0, 16.0, SR), C4, 12, sr);
        let want: Vec<f64> = bare
            .iter()
            .enumerate()
            .map(|(i, b)| b + db(h(C4) / h((i + 1) as f64 * C4)))
            .collect();
        let err = max_abs_diff(&env, &want);
        assert!(
            err <= 0.05,
            "drive {drive}: {env:.2?} vs {want:.2?} ({err:.3} dB)"
        );
    }
}

#[test]
fn the_steepening_keytracks() {
    // The same sine level steepens alike at C3, C4 and G4 (corners and time
    // scale follow freq): the envelopes match C4's.
    for &(k1, k2) in &[(0.573f32, 1.5f32), (0.0, 0.0)] {
        let c4 = envelope(&steepened_sine(C4, 0.8, k1, k2, SR), C4, 10, SR as f64);
        for &f0 in &[130.8128f64, 391.9954] {
            let e = envelope(&steepened_sine(f0, 0.8, k1, k2, SR), f0, 10, SR as f64);
            let err = max_abs_diff(&e, &c4);
            assert!(
                err <= 1.2,
                "k {k1}/{k2} f0 {f0}: {e:.2?} vs C4 {c4:.2?} ({err:.2} dB)"
            );
        }
    }
}

#[test]
fn renders_the_same_steepening_at_48k() {
    let sr = 48_000.0;
    let env = envelope(&steepened_sine(C4, 0.5, 0.0, 0.0, sr), C4, 8, sr as f64);
    check_against_fubini(&env, 0.5, "48 kHz");
    let env44 = envelope(&steepened_sine(C4, 0.5, 0.0, 0.0, SR), C4, 8, SR as f64);
    assert!(
        max_abs_diff(&env, &env44) <= 0.5,
        "{env:.2?} vs 44.1 kHz {env44:.2?}"
    );
}

// ------------------------------------------------------------------ the graph it replaces

/// A brass-like source: h2 strongest, then a falling series.
const PARTIALS: &[(f64, f64)] = &[
    (0.35, 0.0),
    (1.0, 0.4),
    (0.55, 1.1),
    (0.5, 2.0),
    (0.3, 0.7),
    (0.12, 1.9),
    (0.1, 0.2),
    (0.06, 2.5),
    (0.03, 1.3),
    (0.03, 0.6),
    (0.015, 2.2),
    (0.01, 0.9),
];

fn source_expr(f0: f64) -> String {
    PARTIALS
        .iter()
        .enumerate()
        .map(|(i, (a, ph))| format!("sinOsc {:.6} {ph:.4} * {a:.4}", f0 * (i + 1) as f64))
        .collect::<Vec<_>>()
        .join(" + ")
}

/// The graph: two keytracked `onePole` de-emphasis stages, K `delay` stages
/// with time t0 − Δ·z, and the exact inverses built from `delay` at 1/fs.
#[allow(clippy::too_many_arguments)]
fn graph_src(f0: f64, d: f64, k1: f64, k2: f64, k: usize, smax: f64, tconst: f64) -> String {
    let sr = SR as f64;
    let mut s = format!(
        "synthdef g =\n  let x = {}\n  let f = {f0:.6}\n",
        source_expr(f0)
    );
    for (i, kk) in [k1, k2].iter().enumerate() {
        s += &format!(
            "  let c{i} = f * {:.9}\n  let a{i} = 1.0 + f * {:.9}\n",
            TAU * kk / sr,
            -TAU * kk / sr
        );
    }
    s += "  let p0 = onePole x a0\n  let p1 = onePole p0 a1\n";
    s += &format!("  let kt = {:.9} / f\n", C4 / (smax * k as f64));
    s += &format!("  let cstep = {d:.6} * kt\n  let t0 = {tconst:.9} * kt\n");
    s += "  let z0 = p1\n";
    for j in 0..k {
        s += &format!("  let z{} = delay z{j} (t0 - cstep * z{j})\n", j + 1);
    }
    s += &format!("  let w0 = z{k}\n");
    s += &format!(
        "  let w1 = (w0 - delay w0 {:.12} * a0) / c0\n  let w2 = (w1 - delay w1 {:.12} * a1) / c1\n  w2\n",
        1.0 / sr,
        1.0 / sr
    );
    s
}

#[test]
fn matches_the_delay_graph_it_replaces() {
    // smax is the source's own de-emphasised peak slope at C4 (any value
    // works, both sides use it); tconst = 1.875 · pk · D as the graph had.
    let (k1, k2, k) = (0.573f64, 1.5f64, 16usize);
    let (smax, dmax) = (300.0f64, 0.7f64);
    let tconst = 1.875 * 0.3 * dmax;
    let sr = SR as f64;
    let h = |f: f64| pole_gain(k1, C4, f, sr) * pole_gain(k2, C4, f, sr);
    let ref_amp = smax / (TAU * C4 * h(C4));
    let tau = tconst / smax;
    for &f0 in &[130.8128f64, C4, 391.9954] {
        for &d in &[0.05f64, 0.35, 0.7] {
            let g = render(&graph_src(f0, d, k1, k2, k, smax, tconst), 0.8);
            let node = render(
                &format!(
                    "synthdef n =\n  let x = {}\n  steepen x {f0:.6} {d:.6} {k1} {k2} {k}.0 {C4:.6} {ref_amp:.9} {tau:.12}",
                    source_expr(f0)
                ),
                0.8,
            );
            let (eg, en) = (envelope(&g, f0, 16, sr), envelope(&node, f0, 16, sr));
            let err = max_abs_diff(&eg, &en);
            assert!(
                err <= 0.01,
                "f0 {f0} d {d}: graph {eg:.2?}\nnode {en:.2?} ({err:.3} dB)"
            );
            // and sample for sample (the graph runs in f32)
            let seg = (0.25 * sr) as usize..g.len();
            let peak = g[seg.clone()].iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let diff = seg.map(|i| (g[i] - node[i]).abs()).fold(0.0f32, f32::max);
            assert!(
                diff <= 1e-3 * peak,
                "f0 {f0} d {d}: sample diff {diff} (peak {peak})"
            );
        }
    }
}

// ------------------------------------------------------------------ memory, bounds, DC

#[test]
fn buffers_stay_under_64_kb_per_voice() {
    for &(sr, want) in &[
        (44100.0f32, 25_536usize),
        (48_000.0, 27_520),
        (96_000.0, 51_968),
    ] {
        // two channels' rings, each steepen_pool_len values
        let mut node = Steepen::new();
        node.init(&ProcessContext::new(sr, BLOCK));
        assert_eq!(node.buffer_bytes(), want, "{sr} Hz");
        assert_eq!(node.buffer_bytes(), 2 * steepen_pool_len(sr as f64) * 8);
        assert!(node.buffer_bytes() <= 64 * 1024);
    }
}

#[test]
fn dc_passes_with_unit_gain_and_nothing_accumulates() {
    let n = 44_100;
    // a DC level under full drive comes out as the same level
    let y = run_node(
        SR,
        &[
            konst(0.4, n),
            konst(C4 as f32, n),
            konst(1.0, n),
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    );
    let tail = &y[n - 2000..];
    assert!(
        tail.iter().all(|v| (v - 0.4).abs() < 1e-5),
        "{:?}",
        &tail[..4]
    );
    // a zero-mean sine steepened to the shock keeps a small mean
    let y = steepened_sine(C4, 1.0, 0.573, 1.5, SR);
    let seg = whole_periods(&y, C4, SR as f64);
    let mean = seg.iter().map(|&v| v as f64).sum::<f64>() / seg.len() as f64;
    let rms = (seg.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / seg.len() as f64).sqrt();
    // the explicit stages do not conserve the mean exactly (second order
    // in the step): a little DC at the shock, as any asymmetric shaper
    assert!(mean.abs() < 0.02 * rms, "mean {mean:e} rms {rms}");
}

#[test]
fn bounded_and_finite_for_extreme_inputs() {
    let n = 8192;
    let wild: Vec<f32> = (0..n)
        .map(|i| match i % 7 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => -1e30,
            3 => 1e-30,
            _ => (i as f32 * 0.37).sin() * 1e6,
        })
        .collect();
    for freq in [-5.0f32, 0.0, 1e-9, 20.0, 22_049.0, 1e12, f32::NAN] {
        for drive in [-1.0f32, 0.0, 0.5, 2.0, 1e9, f32::NAN] {
            for (stages, tau) in [
                (16.0f32, 0.0f32),
                (1.0, 1e3),
                (32.0, 1e-4),
                (0.0, -1.0),
                (1e9, f32::NAN),
            ] {
                let y = run_node(
                    SR,
                    &[
                        Some(wild.clone()),
                        konst(freq, n),
                        konst(drive, n),
                        konst(0.573, n),
                        konst(1.5, n),
                        konst(stages, n),
                        konst(f32::NAN, n),
                        konst(0.0, n),
                        konst(tau, n),
                    ],
                );
                assert!(
                    y.iter().all(|v| v.is_finite()),
                    "freq {freq} drive {drive} K {stages}"
                );
            }
        }
    }
    // a normal input with a huge drive stays bounded: the time shift clips
    // at ±t0, so the output is a re-ordering of nearby input samples, tilted
    let x = sine(C4, 1.0, n, SR as f64);
    let y = run_node(
        SR,
        &[
            Some(x),
            konst(C4 as f32, n),
            konst(1e9, n),
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    );
    let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak.is_finite() && peak < 1e4, "peak {peak}");
}

#[test]
fn non_finite_controls_read_as_their_defaults() {
    let n = 6000;
    let x = sine(C4, 0.8, n, SR as f64);
    let nan = konst(f32::NAN, n);
    let a = run_node(
        SR,
        &[
            Some(x.clone()),
            nan.clone(),
            konst(0.6, n),
            nan.clone(),
            nan.clone(),
            nan.clone(),
            nan.clone(),
            nan.clone(),
            nan,
        ],
    );
    let b = run_node(
        SR,
        &[
            Some(x),
            None,
            konst(0.6, n),
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    );
    assert_eq!(a, b);
}

#[test]
fn a_change_of_stages_mid_note_is_safe() {
    let n = 20_000;
    let stages: Vec<f32> = (0..n)
        .map(|i| [16.0, 4.0, 32.0, 1.0][(i / 5000) % 4])
        .collect();
    let y = run_node(
        SR,
        &[
            Some(sine(C4, 1.0, n, SR as f64)),
            konst(C4 as f32, n),
            konst(0.9, n),
            None,
            None,
            Some(stages),
            None,
            None,
            None,
        ],
    );
    // finite and bounded; the change itself clicks (a step through the
    // inverse filters), which is why `stages` is a per-instrument constant
    let peak = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak.is_finite() && peak < 100.0, "peak {peak}");
}

#[test]
fn audio_rate_drive_is_continuous() {
    // a drive envelope sweeping 0 -> 1 -> 0 over a held note: no sample
    // step larger than the steady shock tone's
    let n = 44_100;
    let x = sine(C4, 1.0, n, SR as f64);
    let ramp: Vec<f32> = (0..n)
        .map(|i| (core::f64::consts::PI * i as f64 / n as f64).sin() as f32)
        .collect();
    let swept = run_node(
        SR,
        &[
            Some(x.clone()),
            konst(C4 as f32, n),
            Some(ramp),
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    );
    let steady = run_node(
        SR,
        &[
            Some(x),
            konst(C4 as f32, n),
            konst(1.0, n),
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    );
    let step = |y: &[f32]| {
        y[2000..]
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(
        step(&swept) <= step(&steady) * 1.05,
        "{} vs {}",
        step(&swept),
        step(&steady)
    );
}

#[test]
fn ir_form_with_inline_consts_matches_the_dsl() {
    let registry = builtin_registry();
    let ir_json = r#"{
        "format_version": 3, "name": "si", "class": "Source", "output_channels": 1,
        "nodes": [
            {"Param": {"name": "freq", "default": 196.0}},
            {"UGen": {"kind": "sinOsc", "consts": [[1, 0.0]]}},
            {"UGen": {"kind": "steepen", "consts": [[2, 0.8], [3, 0.573], [4, 1.5], [5, 16.0]]}}
        ],
        "edges": [
            {"from": 0, "to": 1, "to_input": 0},
            {"from": 1, "to": 2, "to_input": 0},
            {"from": 0, "to": 2, "to_input": 1}
        ],
        "params": [{"name": "freq", "node": 0, "input": 0, "default": 196.0}],
        "audio_inputs": [],
        "output_node": 2
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
        "synthdef sd = steepen (sinOsc 196.0 0.0) 196.0 0.8 0.573 1.5 16.0 261.625565 1.0 0.0",
        200.0 * BLOCK as f64 / SR as f64,
    );
    let d = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(d < 1e-6, "IR vs DSL: {d}");
}

#[test]
fn the_doc_example_compiles_and_brightens_with_velocity() {
    let src = |vel: f32| {
        format!(
            "synthdef horn freq=261.63 vel={vel} gate=1.0 =\n  \
             let env = asr gate 0.03 0.06\n  \
             let src = blSaw freq * 0.3 + sinOsc freq 0.0 * 0.7\n  \
             steepen src freq (vel * vel * env) 0.573 1.5 16.0 261.63 1.0 0.0 * env * 0.3"
        )
    };
    let centroid = |y: &[f32]| {
        let seg = whole_periods(y, 261.63, SR as f64);
        let (mut num, mut den) = (0.0, 0.0);
        for k in 1..=30 {
            let a = bin_at(seg, 261.63 * k as f64, SR as f64);
            num += k as f64 * a * a;
            den += a * a;
        }
        num / den
    };
    let (soft, loud) = (render(&src(0.2), 1.0), render(&src(1.0), 1.0));
    assert!(
        centroid(&loud) > 1.2 * centroid(&soft),
        "{} vs {}",
        centroid(&loud),
        centroid(&soft)
    );
}
