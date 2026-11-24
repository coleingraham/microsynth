//! `guitarString`: a plucked steel string (two-polarisation waveguide).
//!
//! The tests drive the node directly (and, where the engine's done rule or
//! the DSL matters, through a compiled def) and measure what it plays:
//! tuning (both polarisations, held and moving), stability under vibrato,
//! partial frequencies and decays, inharmonicity, beating, the pluck comb,
//! velocity and brightness, determinism, block-size independence, hygiene,
//! release and done, the `bridge` tap, the string set and residual read from
//! a bound table, and agreement with an independent modal oracle.

// The tests edit a default string set field by field, which reads best
// next to the measurement that needs it.
#![allow(clippy::field_reassign_with_default)]

use microsynth::context::ProcessContext;
use microsynth::node::UGen;
use microsynth::ugens::{GS_N_PORTS, GsStringSet, GuitarString, gs_note_design};
use microsynth::*;
use std::sync::Arc;

mod common;

// Port indices (as documented in the module).
const FREQ: usize = 0;
const GATE: usize = 1;
const VEL: usize = 2;
const STRING: usize = 3;
const POS: usize = 4;
const ANGLE: usize = 5;
const HARD: usize = 6;
const WIDTH: usize = 7;
const DAMP: usize = 8;
const EXC: usize = 9;
const SEED: usize = 10;
const REL: usize = 11;
const VARY: usize = 12;
const TAP: usize = 13;

const LN_1000: f64 = 6.907_755_278_982_137;
const PI: f64 = std::f64::consts::PI;

/// A port's drive: a constant or a function of the sample index.
#[derive(Clone)]
enum Sig {
    C(f32),
    F(std::rc::Rc<dyn Fn(usize) -> f32>),
}

fn f<F: Fn(usize) -> f32 + 'static>(g: F) -> Sig {
    Sig::F(std::rc::Rc::new(g))
}

/// Render `n` samples of one node at `sr` with block size `block`.
fn render_with(
    node: &mut GuitarString,
    sr: f32,
    block: usize,
    n: usize,
    ports: &[(usize, Sig)],
    seed: u32,
) -> Vec<f32> {
    let ctx = ProcessContext::new(sr, block);
    node.init(&ctx);
    node.reseed_noise(seed);
    let mut out = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let len = block.min(n - start);
        let bufs: Vec<Option<AudioBuffer>> = (0..GS_N_PORTS)
            .map(|k| {
                ports.iter().rev().find(|(p, _)| *p == k).map(|(_, s)| {
                    let mut b = AudioBuffer::new(1, len);
                    for i in 0..len {
                        b.channel_mut(0).samples_mut()[i] = match s {
                            Sig::C(v) => *v,
                            Sig::F(g) => g(start + i),
                        };
                    }
                    b
                })
            })
            .collect();
        let inputs: Vec<Option<&AudioBuffer>> = bufs.iter().map(|b| b.as_ref()).collect();
        let mut o = AudioBuffer::new(1, len);
        node.process(&ctx, &inputs, &mut o);
        out.extend_from_slice(&o.channel(0).samples()[..len]);
        start += len;
    }
    out
}

fn render_set(set: GsStringSet, sr: f32, secs: f64, ports: &[(usize, Sig)]) -> Vec<f32> {
    let mut node = GuitarString::with_set(set);
    render_with(&mut node, sr, 64, (secs * sr as f64) as usize, ports, 1)
}

fn render(sr: f32, secs: f64, ports: &[(usize, Sig)]) -> Vec<f32> {
    render_set(GsStringSet::default(), sr, secs, ports)
}

/// Ports for a plain test note: no per-note variation, V only.
fn plain(freq: f32) -> Vec<(usize, Sig)> {
    vec![
        (FREQ, Sig::C(freq)),
        (VARY, Sig::C(0.0)),
        (ANGLE, Sig::C(0.0)),
    ]
}

/// A set whose loss is the same long T60 everywhere (for stability tests).
fn long_set(t60: f64) -> GsStringSet {
    let mut s = GsStringSet::default();
    for st in &mut s.strings {
        st.t60 = [t60; 4];
    }
    s
}

// --- Analysis --------------------------------------------------------------

/// Complex amplitude of `sig` at `f` in Hann windows of `win` seconds every
/// `hop` seconds from `t0` to `t1`: `(centre time, re, im)`.
fn demod(
    sig: &[f32],
    sr: f64,
    f: f64,
    t0: f64,
    t1: f64,
    win: f64,
    hop: f64,
) -> Vec<(f64, f64, f64)> {
    let n = (win * sr) as usize;
    let mut frames = Vec::new();
    let mut t = t0;
    while t + win <= t1 && ((t + win) * sr) as usize <= sig.len() {
        let s0 = (t * sr) as usize;
        let (mut re, mut im, mut wsum) = (0.0, 0.0, 0.0);
        for k in 0..n {
            let w = 0.5 - 0.5 * (2.0 * PI * k as f64 / n as f64).cos();
            let idx = s0 + k;
            let ph = 2.0 * PI * f * idx as f64 / sr;
            let x = sig[idx] as f64 * w;
            re += x * ph.cos();
            im -= x * ph.sin();
            wsum += w;
        }
        frames.push((s0 as f64 / sr + win / 2.0, 2.0 * re / wsum, 2.0 * im / wsum));
        t += hop;
    }
    frames
}

/// Least-squares slope and intercept of `y` on `x`.
fn line_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}

/// A partial's frequency (Hz) and decay rate (1/s, amplitude) measured near
/// `f_guess` between `t0` and `t1`, from the phase slope and the log
/// magnitude slope of windowed demodulation (refined three times).
fn partial(sig: &[f32], sr: f64, f_guess: f64, t0: f64, t1: f64, win: f64) -> (f64, f64) {
    let mut fq = f_guess;
    let mut sigma = 0.0;
    for _ in 0..3 {
        let fr = demod(sig, sr, fq, t0, t1, win, win / 4.0);
        let ts: Vec<f64> = fr.iter().map(|x| x.0).collect();
        let mut ph: Vec<f64> = fr.iter().map(|x| x.2.atan2(x.1)).collect();
        for i in 1..ph.len() {
            while ph[i] - ph[i - 1] > PI {
                ph[i] -= 2.0 * PI;
            }
            while ph[i] - ph[i - 1] < -PI {
                ph[i] += 2.0 * PI;
            }
        }
        let (slope, _) = line_fit(&ts, &ph);
        fq += slope / (2.0 * PI);
        let lm: Vec<f64> = fr
            .iter()
            .map(|x| (x.1 * x.1 + x.2 * x.2).sqrt().max(1e-30).ln())
            .collect();
        sigma = -line_fit(&ts, &lm).0;
    }
    (fq, sigma)
}

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2()
}

fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

/// Fit `B` from partials `(n, f_n)`: `(f_n/n)^2` is linear in `n^2` with
/// slope `f0^2 B`.
fn fit_b(parts: &[(u32, f64)]) -> f64 {
    let x: Vec<f64> = parts.iter().map(|(n, _)| (*n as f64).powi(2)).collect();
    let y: Vec<f64> = parts.iter().map(|(n, f)| (f / *n as f64).powi(2)).collect();
    let (slope, icpt) = line_fit(&x, &y);
    slope / icpt
}

fn centroid(x: &[f32], sr: f64) -> f64 {
    let n = x.len();
    let bins = 600usize;
    let df = (0.45 * sr) / bins as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for b in 1..bins {
        let f = b as f64 * df;
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &v) in x.iter().enumerate() {
            let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos();
            let ph = 2.0 * PI * f * i as f64 / sr;
            re += v as f64 * w * ph.cos();
            im -= v as f64 * w * ph.sin();
        }
        let m = (re * re + im * im).sqrt();
        num += f * m;
        den += m;
    }
    num / den
}

// --- The documented law, re-implemented ------------------------------------

/// The decay-rate law of the module doc, written from the documentation:
/// `σ(f)` per second on string `s` (0-based) for a note at fret ratio `k`,
/// polarisation weight `hv` (1 for V).
fn law_sigma(set: &GsStringSet, s: usize, k: f64, hv: f64, f: f64) -> f64 {
    let st = &set.strings[s];
    let sig0 = LN_1000 / st.t60[0];
    let br = st.bridge_share * sig0;
    let mut knots = vec![(st.open_hz.ln(), ((1.0 - st.bridge_share) * sig0).ln())];
    for (j, fa) in set.anchor_hz.iter().enumerate() {
        if *fa > st.open_hz * 1.01 {
            let sa = LN_1000 / st.t60[j + 1];
            knots.push((fa.ln(), (sa - br).max(0.1 * sa).ln()));
        }
    }
    let lf = f.ln();
    let li = if lf <= knots[0].0 {
        knots[0].1
    } else if lf >= knots.last().unwrap().0 {
        knots.last().unwrap().1
    } else {
        let i = knots.iter().position(|k| k.0 >= lf).unwrap();
        let t = (lf - knots[i - 1].0) / (knots[i].0 - knots[i - 1].0);
        knots[i - 1].1 + t * (knots[i].1 - knots[i - 1].1)
    };
    hv * br * k + li.exp()
}

/// Partial number of a stiff string at `f`, by bisection.
fn n_of(f: f64, f0: f64, b: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, f / f0 + 1.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid * f0 * (1.0 + b * mid * mid).sqrt() < f {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn tau(f: f64, f0: f64, b: f64) -> f64 {
    let e = f * 1e-6;
    (n_of(f + e, f0, b) - n_of(f - e, f0, b)) / (2.0 * e)
}

/// The string the documented rule picks.
fn rule_string(set: &GsStringSet, f: f64) -> usize {
    let tol = 2f64.powf(0.25 / 12.0);
    let mut best = 5;
    let mut best_o = 0.0;
    for (i, s) in set.strings.iter().enumerate() {
        if s.open_hz <= f * tol && s.open_hz > best_o {
            best = i;
            best_o = s.open_hz;
        }
    }
    best
}

// --- Tuning ----------------------------------------------------------------

/// The first partial sits at `freq` within a cent from E2 to E7, at 22.05,
/// 44.1, 48 and 96 kHz, with every loop filter's phase included.
#[test]
fn tuning_within_a_cent_from_e2_to_e7_at_four_rates() {
    let mut worst = 0.0f64;
    let mut report = String::new();
    for sr in [22_050.0f32, 44_100.0, 48_000.0, 96_000.0] {
        for midi in (40..=100).step_by(4) {
            let fr = 440.0 * 2f64.powf((midi as f64 - 69.0) / 12.0);
            if fr > sr as f64 / 8.0 {
                continue;
            }
            let sig = render(sr, 0.9, &plain(fr as f32));
            let (f1, _) = partial(&sig, sr as f64, fr, 0.1, 0.9, 0.2);
            let c = cents(f1, fr);
            worst = worst.max(c.abs());
            report += &format!(" {sr}/{midi}:{c:+.3}");
        }
    }
    println!("tuning error (cents):{report}\nworst {worst:.4}");
    assert!(worst < 1.0, "worst tuning error {worst:.3} cents:{report}");
}

/// The H polarisation sounds its string's `split` flat of `freq`.
#[test]
fn h_polarisation_is_tuned_split_flat() {
    let sr = 44_100.0;
    let set = GsStringSet::default();
    let mut report = String::new();
    for fr in [82.406_89, 196.0, 329.627_56, 660.0] {
        let s = rule_string(&set, fr);
        let want = fr * 2f64.powf(-set.strings[s].split_cents / 1200.0);
        let mut p = plain(fr as f32);
        p.push((ANGLE, Sig::C(90.0)));
        p.push((TAP, Sig::C(0.0)));
        let sig = render(sr, 1.5, &p);
        let (f1, _) = partial(&sig, sr as f64, want, 0.1, 1.5, 0.5);
        let c = cents(f1, want);
        report += &format!(" {fr:.1}: {c:+.3} c;");
        assert!(c.abs() < 1.0, "H off by {c:.3} cents:{report}");
    }
    println!("H tuning:{report}");
}

/// A bend lands in tune.
#[test]
fn a_bend_lands_in_tune() {
    let sr = 44100.0;
    let fr = 196.0f32;
    let up = fr * 2f32.powf(2.0 / 12.0);
    let mut ports = plain(fr);
    ports.push((
        FREQ,
        f(move |i| {
            let t = i as f32 / sr;
            let x = ((t - 0.1) / 0.1).clamp(0.0, 1.0);
            fr * (up / fr).powf(x)
        }),
    ));
    let sig = render(sr, 1.2, &ports);
    let (f1, _) = partial(&sig, sr as f64, up as f64, 0.4, 1.2, 0.2);
    let c = cents(f1, up as f64);
    println!("bend lands at {c:+.3} cents");
    assert!(c.abs() < 1.0, "bend lands at {c:.3} cents");
}

/// Pitch modulation adds no energy. ±50 cents at 6 Hz and ±2 semitones at
/// 12 Hz for 4 s on strings with a 60 s T60 everywhere: after the attack,
/// no 0.5 s RMS window exceeds the previous one, the decay rate of the RMS
/// stays within 10 % of the same note held still, and the largest
/// second-difference (a click measure) stays within 3x the held note's.
#[test]
fn vibrato_adds_no_energy() {
    let sr = 44_100.0f32;
    let set = long_set(60.0);
    let mut report = String::new();
    for (fr, depth_c, rate) in [
        (110.0f32, 50.0f32, 6.0f32),
        (196.0, 50.0, 6.0),
        (329.6, 50.0, 6.0),
        (440.0, 50.0, 6.0),
        (110.0, 200.0, 12.0),
        (329.6, 200.0, 12.0),
        (1318.5, 200.0, 12.0),
    ] {
        let still = render_set(set, sr, 4.0, &plain(fr));
        let mut p = plain(fr);
        p.push((
            FREQ,
            f(move |i| {
                let t = i as f32 / sr;
                fr * 2f32.powf(depth_c / 1200.0 * (2.0 * std::f32::consts::PI * rate * t).sin())
            }),
        ));
        let moving = render_set(set, sr, 4.0, &p);
        let w = (0.5 * sr) as usize;
        let wins =
            |x: &[f32]| -> Vec<f64> { (1..8).map(|k| rms(&x[k * w..(k + 1) * w])).collect() };
        let (rs, rm) = (wins(&still), wins(&moving));
        for k in 1..rm.len() {
            assert!(
                rm[k] <= rm[k - 1] * 1.0001,
                "{fr} Hz ±{depth_c} c at {rate} Hz: window {k} RMS {} > {}",
                rm[k],
                rm[k - 1]
            );
        }
        let ts: Vec<f64> = (0..rs.len()).map(|k| k as f64 * 0.5).collect();
        let ls: Vec<f64> = rs.iter().map(|x| x.ln()).collect();
        let lm: Vec<f64> = rm.iter().map(|x| x.ln()).collect();
        let (ds, dm) = (-line_fit(&ts, &ls).0, -line_fit(&ts, &lm).0);
        let click = |x: &[f32]| {
            x[(0.3 * sr) as usize..]
                .windows(3)
                .fold(0.0f32, |m, v| m.max((v[2] - 2.0 * v[1] + v[0]).abs()))
        };
        let (cs, cm) = (click(&still), click(&moving));
        report += &format!(
            "\n  {fr} Hz ±{depth_c} c at {rate} Hz: RMS decay {dm:.4}/s vs still {ds:.4}/s, \
             click {:.2}x, RMS 0.5-1 s {:.3} → 3.5-4 s {:.3}",
            cm / cs,
            rm[0],
            rm[rm.len() - 1]
        );
        assert!(
            (dm / ds - 1.0).abs() < 0.10 || (dm - ds).abs() < 0.02,
            "decay changed:{report}"
        );
        assert!(cm < 3.0 * cs, "clicks:{report}");
    }
    println!("vibrato:{report}");
}

// --- Partials: decay, inharmonicity, oracle --------------------------------

/// A set shaped like an archtop's measured octave-band T60s (register
/// C3–A3: 3.6 s at 1 kHz, 1.5 at 2 kHz, 2.0 at 4 kHz, 1.35 at 8 kHz), with
/// the anchors at 1, 3 and 8 kHz.
fn archtop_like() -> GsStringSet {
    let mut s = GsStringSet::default();
    for st in &mut s.strings {
        st.t60 = [6.0, 3.6, 1.7, 1.35];
    }
    s
}

/// Partial decays follow the documented law within 20 % on every partial
/// below 8 kHz, and within 10 % at the anchors, for the default set and an
/// archtop-shaped set, on wound and plain strings, open and fretted. The
/// anchors are not met exactly because the loss design keeps every
/// frequency from outlasting the note (above the fundamental no frequency
/// loses less per pass than the fundamental; above the last anchor at least
/// 90 % of the last anchor's; DC at least half the fundamental's), which
/// costs up to about 8 % at an anchor on the archtop-shaped low E.
#[test]
fn partial_decays_follow_the_loss_law() {
    let sr = 44100.0;
    let (mut worst, mut worst_anchor) = (0.0f64, 0.0f64);
    let mut report = String::new();
    for (name, set) in [
        ("default", GsStringSet::default()),
        ("archtop-like", archtop_like()),
    ] {
        for fr in [82.406_89, 123.47, 246.941_65, 392.0] {
            let s = rule_string(&set, fr);
            let k = fr / set.strings[s].open_hz;
            let des = gs_note_design(sr as f64, &set, fr, -1.0);
            let sig = render_set(set, sr, 3.0, &plain(fr as f32));
            for (n, fp) in des.partials.iter().enumerate() {
                let n = n as u32 + 1;
                if *fp > 8000.0 {
                    break;
                }
                let want = law_sigma(&set, s, k, 1.0, *fp);
                let t1 = (0.2 + 2.0 * LN_1000 / want / 3.0).clamp(0.6, 2.9);
                let (_, got) = partial(&sig, sr as f64, *fp, 0.2, t1, 0.15);
                let err = got / want - 1.0;
                let at_anchor = n == 1
                    || set
                        .anchor_hz
                        .iter()
                        .any(|a| (fp / a).ln().abs() < (1.0 + 0.5 * des.f0 / a).ln());
                worst = worst.max(err.abs());
                if at_anchor {
                    worst_anchor = worst_anchor.max(err.abs());
                }
                if n <= 3 || at_anchor || n.is_multiple_of(5) {
                    report += &format!(
                        "\n  {name} f {fr:.1} n {n} ({fp:.0} Hz{}): T60 law {:.2} got {:.2} ({:+.1} %)",
                        if at_anchor { ", anchor" } else { "" },
                        LN_1000 / want,
                        LN_1000 / got,
                        100.0 * err
                    );
                }
            }
        }
    }
    println!(
        "decays:{report}\nworst {:.1} %, at anchors {:.1} %",
        100.0 * worst,
        100.0 * worst_anchor
    );
    assert!(
        worst_anchor < 0.10,
        "anchor decay error {:.1} %:{report}",
        100.0 * worst_anchor
    );
    assert!(worst < 0.20, "decay error {:.1} %:{report}", 100.0 * worst);
}

/// Inharmonicity round-trips. On all six strings at frets 0, 5 and 12, at
/// 44.1 and 22.05 kHz: the loop plays its design within 0.5 cents (a miss is
/// a loop fault), and `B` fitted from the measured partials below 5 kHz is
/// within 5 % of the requested `B` (the design is pre-warped for this). The
/// worst partial of the design against `n f0 sqrt(1 + B n^2)` is reported:
/// the cascade's phase ripples between sections, most at the top of the
/// designed band.
#[test]
fn inharmonicity_round_trips() {
    let set = GsStringSet::default();
    let mut report = String::new();
    let (mut worst_b, mut worst_play) = (0.0f64, 0.0f64);
    let mut at_play = String::new();
    let (mut worst_design, mut at_design) = (0.0f64, String::new());
    for sr in [44_100.0f32, 22_050.0] {
        for s in 1..=6usize {
            for fret in [0, 5, 12] {
                let fr = set.strings[s - 1].open_hz * 2f64.powf(fret as f64 / 12.0);
                let des = gs_note_design(sr as f64, &set, fr, s as f32);
                let mut p = plain(fr as f32);
                p.push((STRING, Sig::C(s as f32)));
                let sig = render(sr, 1.6, &p);
                let mut parts = Vec::new();
                for (n, fd) in des.partials.iter().enumerate() {
                    let n = n as u32 + 1;
                    if *fd > 5000.0 || n > 30 {
                        break;
                    }
                    let ideal = n as f64 * des.f0 * (1.0 + des.b * (n * n) as f64).sqrt();
                    let want_sigma =
                        law_sigma(&set, s - 1, fr / set.strings[s - 1].open_hz, 1.0, *fd);
                    let t1 = (0.1 + 50.0 / 60.0 * LN_1000 / want_sigma).clamp(0.5, 0.9);
                    let (fm, _) = partial(&sig, sr as f64, *fd, 0.1, t1, 0.2);
                    if cents(fm, *fd).abs() > worst_play {
                        worst_play = cents(fm, *fd).abs();
                        at_play = format!(
                            "{sr} s{s} fret {fret} n {n} ({fd:.0} Hz, window to {t1:.2} s)"
                        );
                    }
                    let cd = cents(*fd, ideal);
                    if cd.abs() > worst_design {
                        worst_design = cd.abs();
                        at_design = format!(
                            "{sr} s{s} fret {fret} n {n} ({ideal:.0} Hz): {cd:+.2} c of a {:.0} c stretch",
                            cents(ideal, n as f64 * des.f0)
                        );
                    }
                    parts.push((n, fm));
                }
                let b = fit_b(&parts);
                let err = b / des.b - 1.0;
                worst_b = worst_b.max(err.abs());
                report += &format!(
                    "\n  {sr} s{s} fret {fret}: B {:.2e} fitted {:.2e} ({:+.1} %), designed for {:.2e}, {} partials",
                    des.b,
                    b,
                    100.0 * err,
                    des.b_design,
                    parts.len()
                );
            }
        }
    }
    println!(
        "inharmonicity:{report}\nworst B {:.1} %; played vs design {worst_play:.3} c at {at_play}; worst partial vs ideal {at_design}",
        100.0 * worst_b
    );
    assert!(
        worst_play < 0.5,
        "the loop misses its design by {worst_play:.3} c"
    );
    assert!(worst_b < 0.05, "B error {:.1} %:{report}", 100.0 * worst_b);
}

/// What the pluck should put on partial `n`: twice the pluck waveform's
/// spectrum there over the period, times the output gain 0.5 (the module
/// doc's waveform, written from the documentation).
fn pluck_level(sr: f64, freq: f64, fp: f64, set: &GsStringSet, s: usize, vel: f64) -> f64 {
    let st = set.strings[s];
    let k = freq / st.open_hz;
    let pp = sr / freq;
    let beta0 = 140.0 / (set.scale_mm / k);
    let m = (beta0 * pp).round();
    let beta = m / pp;
    let hard = 0.5;
    let w_eff = 10.0 * (1.25 - 0.5 * vel) * (1.0 - 0.5 * hard);
    let ramp = (w_eff * 1e-3 / (2.0 * set.scale_mm * 1e-3 * st.open_hz) * sr).clamp(1.0, 64.0);
    let corner =
        (set.vel_corner_hz * (vel * (0.5 + 0.5 * hard)).powf(set.vel_corner_exp)).min(0.45 * sr);
    let r = (-2.0 * PI * corner / sr).exp();
    let step = |t: f64| {
        if t <= 0.0 {
            0.0
        } else if t >= ramp {
            1.0
        } else {
            0.5 * (1.0 - (PI * t / ramp).cos())
        }
    };
    let q = |t: f64| {
        if t <= -1.0 {
            return 0.0;
        }
        if t < 0.0 {
            return (t + 1.0) * step(1.0);
        }
        let i = t.floor();
        let fr = t - i;
        step(i + 1.0) * (1.0 - fr) + step(i + 2.0) * fr
    };
    let w = 2.0 * PI * fp / sr;
    let (mut re, mut im, mut y) = (0.0, 0.0, 0.0);
    for t in 0..((pp + ramp) as usize + 4000) {
        let tf = t as f64;
        let u = vel * ((1.0 - beta) * q(tf) - q(tf - m) + beta * q(tf - pp));
        y = (1.0 - r) * u + r * y;
        re += y * (w * tf).cos();
        im -= y * (w * tf).sin();
    }
    0.5 * 2.0 * (re * re + im * im).sqrt() / pp
}

/// The waveguide against an independent modal oracle: a sum of damped
/// partials at `n f0 sqrt(1 + B n^2)` (the requested stiffness), with the
/// decays of the note's loss filter (from `gs_note_design`) and the
/// documented pluck's levels. Compared on every partial below 5 kHz (up to
/// 30): frequency within 4 cents or 10 % of the partial's stretch, decay
/// within 5 %, level within 1.5 dB rms over the first 12; and `B` fitted
/// from the waveguide within 5 % of `B` fitted from the oracle.
#[test]
fn waveguide_matches_the_modal_oracle() {
    let sr = 44100.0;
    let set = GsStringSet::default();
    let mut report = String::new();
    let (mut wf, mut wd, mut wl, mut wb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for fr in [82.406_89, 110.0, 246.941_65, 392.0] {
        let s = rule_string(&set, fr);
        let des = gs_note_design(sr as f64, &set, fr, -1.0);
        let sig = render(sr, 2.5, &plain(fr as f32));
        // The oracle.
        let nmax = 40u32;
        let n_s = (2.5 * sr as f64) as usize;
        let mut orc = vec![0.0f64; n_s];
        let mut params = Vec::new();
        for n in 1..=nmax {
            let fp = n as f64 * des.f0 * (1.0 + des.b * (n * n) as f64).sqrt();
            if fp > 0.45 * sr as f64 {
                break;
            }
            let w = 2.0 * PI * fp / sr as f64;
            let sigma = -des.loss[0].mag(w).ln() / tau(fp, des.f0, des.b);
            let a = pluck_level(sr as f64, fr, fp, &set, s, 0.8);
            params.push((fp, sigma));
            let dec = (-sigma / sr as f64).exp();
            let (mut amp, mut ph) = (a, 0.0f64);
            for o in orc.iter_mut() {
                *o += amp * ph.sin();
                amp *= dec;
                ph += w;
            }
        }
        let orc: Vec<f32> = orc.into_iter().map(|x| x as f32).collect();
        let (mut pw, mut po, mut lvl) = (Vec::new(), Vec::new(), Vec::new());
        for (n, (fp, sigma)) in params.iter().enumerate() {
            let n = n as u32 + 1;
            if *fp > 5000.0 || n > 30 {
                break;
            }
            let guess = des.partials.get(n as usize - 1).copied().unwrap_or(*fp);
            let t1 = (0.2 + 0.8 * LN_1000 / sigma).clamp(0.6, 1.4);
            let (fa, sa) = partial(&sig, sr as f64, guess, 0.2, t1, 0.15);
            let (fo, so) = partial(&orc, sr as f64, *fp, 0.2, t1, 0.15);
            let dc = cents(fa, fo);
            let stretch = cents(*fp, n as f64 * des.f0);
            let allowed = (0.1 * stretch).max(4.0);
            wf = wf.max(dc.abs() / allowed);
            let dd = sa / so - 1.0;
            wd = wd.max(dd.abs());
            pw.push((n, fa));
            po.push((n, fo));
            if n <= 12 {
                let lev = |x: &[f32], f: f64, s: f64| {
                    let fr_ = demod(x, sr as f64, f, 0.1, 0.4, 0.1, 0.025);
                    let v: Vec<f64> = fr_
                        .iter()
                        .map(|(t, re, im)| ((re * re + im * im).sqrt() * (s * t).exp()).ln())
                        .collect();
                    20.0 / std::f64::consts::LN_10 * v.iter().sum::<f64>() / v.len() as f64
                };
                lvl.push(lev(&sig, fa, sa) - lev(&orc, fo, so));
            }
            if n <= 3 || n.is_multiple_of(5) {
                report += &format!(
                    "\n  f {fr:.1} n {n}: {dc:+.2} c (stretch {stretch:.1} c), decay {:+.1} %",
                    100.0 * dd
                );
            }
        }
        let rmsl = (lvl.iter().map(|x| x * x).sum::<f64>() / lvl.len() as f64).sqrt();
        wl = wl.max(rmsl);
        let (bw, bo) = (fit_b(&pw), fit_b(&po));
        wb = wb.max((bw / bo - 1.0).abs());
        report += &format!(
            "\n  f {fr:.1}: B waveguide {bw:.3e} oracle {bo:.3e}, level rms {rmsl:.2} dB, {} partials",
            pw.len()
        );
    }
    println!(
        "oracle:{report}\nworst freq {wf:.2} of its bound, decay {:.1} %, level rms {wl:.2} dB, B {:.1} %",
        100.0 * wd,
        100.0 * wb
    );
    assert!(wf <= 1.0, "frequency off the oracle:{report}");
    assert!(
        wd < 0.05,
        "decay off the oracle by {:.1} %:{report}",
        100.0 * wd
    );
    assert!(wl < 1.5, "levels off the oracle by {wl:.2} dB rms:{report}");
    assert!(
        wb < 0.05,
        "B off the oracle by {:.1} %:{report}",
        100.0 * wb
    );
}

// --- Polarisations ------------------------------------------------------

/// With both polarisations sounding, a partial's envelope beats at the
/// difference of the two loops' frequencies: `f_n (1 - 2^(-split/1200))`.
#[test]
fn polarisation_pair_beats_at_its_detune() {
    let sr = 44100.0;
    let mut report = String::new();
    for (fr, n, split) in [
        (246.941_65, 10u32, 2.0),
        (246.941_65, 6, 3.0),
        (110.0, 12, 2.0),
    ] {
        let mut set = GsStringSet::default();
        set.hmix = 1.0;
        let s = rule_string(&set, fr);
        set.strings[s].split_cents = split;
        set.strings[s].hv = 1.0;
        let des = gs_note_design(sr as f64, &set, fr, -1.0);
        let mut p = plain(fr as f32);
        p.push((ANGLE, Sig::C(45.0)));
        let sig = render_set(set, sr, 4.0, &p);
        let fv = des.partials[n as usize - 1];
        let want = fv * (1.0 - 2f64.powf(-split / 1200.0));
        let fr_ = demod(&sig, sr as f64, fv, 0.05, 4.0, 0.04, 0.005);
        let ts: Vec<f64> = fr_.iter().map(|x| x.0).collect();
        let lm: Vec<f64> = fr_
            .iter()
            .map(|x| (x.1 * x.1 + x.2 * x.2).sqrt().max(1e-30).ln())
            .collect();
        let (sl, ic) = line_fit(&ts, &lm);
        let res: Vec<f64> = ts.iter().zip(&lm).map(|(t, y)| y - (sl * t + ic)).collect();
        let mut best = (0.0, 0.0);
        let mut fb = 0.2;
        while fb < 12.0 {
            let (mut re, mut im) = (0.0, 0.0);
            for (t, r) in ts.iter().zip(&res) {
                re += r * (2.0 * PI * fb * t).cos();
                im += r * (2.0 * PI * fb * t).sin();
            }
            let m = re * re + im * im;
            if m > best.1 {
                best = (fb, m);
            }
            fb += 0.005;
        }
        let err = best.0 / want - 1.0;
        report += &format!(
            "\n  f {fr:.1} n {n} split {split}: want {want:.3} Hz got {:.3} ({:+.1} %)",
            best.0,
            100.0 * err
        );
        assert!(err.abs() < 0.08, "beat off:{report}");
    }
    println!("beats:{report}");
}

/// H loses `hv` times V's bridge loss: with `hv` < 1 the H loop's
/// fundamental outlasts V's.
#[test]
fn h_polarisation_loses_less_to_the_bridge() {
    let sr = 44100.0;
    let fr = 82.406_89;
    let mut set = GsStringSet::default();
    set.hmix = 1.0;
    let decay = |angle: f32| {
        let mut p = plain(fr as f32);
        p.push((ANGLE, Sig::C(angle)));
        let sig = render_set(set, sr, 2.5, &p);
        let want = if angle == 0.0 {
            fr
        } else {
            fr * 2f64.powf(-set.strings[5].split_cents / 1200.0)
        };
        partial(&sig, sr as f64, want, 0.2, 2.4, 0.3).1
    };
    let (v, h) = (decay(0.0), decay(90.0));
    println!(
        "fundamental T60: V {:.2} s, H {:.2} s",
        LN_1000 / v,
        LN_1000 / h
    );
    assert!(h < 0.8 * v, "H should decay slower: V {v:.3}/s, H {h:.3}/s");
}

// --- Pluck --------------------------------------------------------------

/// Plucking at 1/5 of the speaking length notches partials 5 and 10.
#[test]
fn pluck_position_notches_the_predicted_partials() {
    let sr = 44100.0;
    let fr = 110.0;
    let mut p = plain(fr as f32);
    p.push((POS, Sig::C(645.0 / 5.0)));
    let des = gs_note_design(sr as f64, &GsStringSet::default(), fr, -1.0);
    let sig = render(sr, 0.4, &p);
    let lvl = |n: u32| {
        let fr_ = demod(
            &sig,
            sr as f64,
            des.partials[n as usize - 1],
            0.02,
            0.32,
            0.3,
            0.3,
        );
        20.0 * ((fr_[0].1.powi(2) + fr_[0].2.powi(2)).sqrt()).log10()
    };
    let mut report = String::new();
    for notch in [5u32, 10] {
        let around = 0.5 * (lvl(notch - 1) + lvl(notch + 1));
        let depth = around - lvl(notch);
        report += &format!(" n{notch}: {depth:.1} dB below its neighbours;");
        assert!(depth > 20.0, "notch at {notch} only {depth:.1} dB deep");
    }
    println!("comb:{report}");
}

/// Velocity changes the timbre: the attack's spectral centroid rises
/// monotonically with velocity, by at least 500 Hz from 30 to 120 (MIDI)
/// on the D string, and the level by about 12 dB; and the string set's
/// velocity corner and exponent set the range (a lower corner, a duller
/// top).
#[test]
fn velocity_brightens_the_attack() {
    let sr = 44100.0;
    let run = |set: GsStringSet| {
        let mut out = Vec::new();
        for v in [30.0f64, 50.0, 70.0, 90.0, 110.0, 120.0] {
            let mut p = vec![(FREQ, Sig::C(146.832_38)), (VARY, Sig::C(0.0))];
            p.push((VEL, Sig::C((v / 127.0) as f32)));
            let sig = render_set(set, sr, 0.1, &p);
            out.push((
                v,
                centroid(&sig[..4096], sr as f64),
                20.0 * (peak(&sig) as f64).log10(),
            ));
        }
        out
    };
    let d = run(GsStringSet::default());
    let mut report = String::new();
    for w in d.windows(2) {
        assert!(w[1].1 > w[0].1, "centroid not monotonic: {d:?}");
    }
    for (v, c, p) in &d {
        report += &format!(" v{v}: {c:.0} Hz {p:.1} dB;");
    }
    let (c30, c120) = (d[0].1, d[5].1);
    assert!(
        c120 - c30 >= 500.0,
        "centroid rose only {:.0} Hz:{report}",
        c120 - c30
    );
    assert!(
        d[5].2 - d[0].2 >= 10.0,
        "level rose only {:.1} dB:{report}",
        d[5].2 - d[0].2
    );
    let mut dull = GsStringSet::default();
    dull.vel_corner_hz = 4000.0;
    let dd = run(dull);
    assert!(
        dd[5].1 < c120 - 200.0,
        "corner did not darken the top: {dd:?}"
    );
    println!("velocity:{report}; corner 4 kHz top {:.0} Hz", dd[5].1);
}

// --- Determinism, variation, hygiene --------------------------------------

/// Same voice seed: identical; another voice seed or `seed`: a different
/// pluck; with `vary` 0 the voice seed makes no difference.
#[test]
fn deterministic_with_seeded_per_note_variation() {
    let sr = 44100.0;
    let n = 20_000;
    let run = |seed: u32, ports: &[(usize, Sig)]| {
        let mut node = GuitarString::new();
        render_with(&mut node, sr, 64, n, ports, seed)
    };
    let base = vec![(FREQ, Sig::C(196.0))];
    assert_eq!(run(7, &base), run(7, &base), "same seed must repeat");
    assert_ne!(
        run(7, &base),
        run(8, &base),
        "another voice seed varies the pluck"
    );
    let mut seeded = base.clone();
    seeded.push((SEED, Sig::C(3.0)));
    assert_ne!(
        run(7, &base),
        run(7, &seeded),
        "the seed port varies the pluck"
    );
    let mut fixed = base.clone();
    fixed.push((VARY, Sig::C(0.0)));
    assert_eq!(
        run(7, &fixed),
        run(8, &fixed),
        "vary 0 removes the variation"
    );
    let mut re = base.clone();
    re.push((GATE, f(|i| if (i / 5000) % 2 == 0 { 1.0 } else { 0.0 })));
    let out = run(7, &re);
    let a = &out[..2000];
    let b = &out[10_000..12_000];
    assert!(
        a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-4),
        "re-plucks repeat exactly"
    );
}

/// Bit-exact for every block size from 1 to 128, through a gate fall, a
/// vibrato, a palm mute, a `rel` change and `exc` input.
#[test]
fn block_size_independent() {
    let sr = 44100.0;
    let n = 30_000;
    let ports = vec![
        (
            FREQ,
            f(|i| 220.0 * (1.0 + 0.01 * (i as f32 * 0.0007).sin())),
        ),
        (GATE, f(|i| if i < 20_000 { 1.0 } else { 0.0 })),
        (
            DAMP,
            f(|i| {
                if (8000..12_000).contains(&i) {
                    0.5
                } else {
                    0.0
                }
            }),
        ),
        (REL, f(|i| if i < 21_000 { 0.15 } else { 0.4 })),
        (EXC, f(|i| if i % 9000 == 4000 { 0.3 } else { 0.0 })),
    ];
    let mut reference: Option<Vec<f32>> = None;
    for block in [1usize, 2, 3, 7, 16, 31, 64, 100, 128] {
        let mut node = GuitarString::new();
        let out = render_with(&mut node, sr, block, n, &ports, 5);
        match &reference {
            None => reference = Some(out),
            Some(r) => assert!(
                r.iter().zip(&out).all(|(a, b)| a.to_bits() == b.to_bits()),
                "block {block} differs"
            ),
        }
    }
}

/// Non-finite inputs on any port leave the output finite and bounded; a
/// non-finite, zero or negative `freq` is silence and the node reports done.
#[test]
fn non_finite_and_zero_inputs() {
    let sr = 44100.0;
    for k in 0..GS_N_PORTS {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -1e30] {
            let ports = vec![(FREQ, Sig::C(220.0)), (k, Sig::C(bad))];
            let out = render(sr, 0.3, &ports);
            let bound = if k == EXC { 40.0 } else { 4.0 };
            assert!(
                out.iter().all(|x| x.is_finite() && x.abs() < bound),
                "port {k} = {bad}"
            );
        }
    }
    for bad in [0.0f32, -100.0, f32::NAN] {
        let mut node = GuitarString::new();
        let out = render_with(&mut node, sr, 64, 8820, &[(FREQ, Sig::C(bad))], 1);
        assert!(out.iter().all(|&x| x == 0.0), "freq {bad} must be silent");
        assert!(node.is_done(), "freq {bad} must report done");
    }
    let mut node = GuitarString::new();
    let out = render_with(
        &mut node,
        sr,
        64,
        8820,
        &[(FREQ, f(|i| if i < 4410 { 220.0 } else { 0.0 }))],
        1,
    );
    assert!(peak(&out[..4410]) > 0.05);
    assert!(out[4410..].iter().all(|&x| x == 0.0));
    assert!(node.is_done());
}

/// A gate that rises while `freq` is 0 leaves the pluck pending: a `freq`
/// arriving a block later plays it; one arriving after 0.1 s does not, and
/// the node then reports done; while pending it does not.
#[test]
fn a_pluck_waits_for_freq() {
    let sr = 44100.0;
    let mut node = GuitarString::new();
    let out = render_with(
        &mut node,
        sr,
        64,
        8820,
        &[(FREQ, f(|i| if i < 64 { 0.0 } else { 220.0 }))],
        1,
    );
    assert!(out[..64].iter().all(|&x| x == 0.0));
    assert!(
        peak(&out[64..2000]) > 0.05,
        "the pending pluck did not play"
    );

    let mut node = GuitarString::new();
    let ctx = ProcessContext::new(sr, 64);
    node.init(&ctx);
    let zero = {
        let mut b = AudioBuffer::new(1, 64);
        b.channel_mut(0).samples_mut().fill(0.0);
        b
    };
    let mut inputs: Vec<Option<&AudioBuffer>> = vec![None; GS_N_PORTS];
    inputs[FREQ] = Some(&zero);
    let mut o = AudioBuffer::new(1, 64);
    node.process(&ctx, &inputs, &mut o);
    assert!(!node.is_done(), "done while a pluck is pending");
    for _ in 0..80 {
        node.process(&ctx, &inputs, &mut o);
    }
    assert!(node.is_done(), "done once the pending pluck has expired");
    let late = render_with(
        &mut GuitarString::new(),
        sr,
        64,
        13_230,
        &[(FREQ, f(|i| if i < 8820 { 0.0 } else { 220.0 }))],
        1,
    );
    assert!(late.iter().all(|&x| x == 0.0), "an expired pluck played");
}

/// Every port and every string-set value swept over its range and beyond,
/// at 22.05, 44.1, 48 and 96 kHz: nothing panics, and the output stays
/// finite and bounded (the extreme corners included: a 200 mm scale, a
/// 20 Hz open string, a note at the top of the range).
#[test]
fn stable_and_panic_free_for_every_value() {
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let corners: [(f32, f32, f32, f32); 3] = [
        // (sr, scale, open of string 6, freq)
        (44_100.0, 200.0, 20.0, 3000.0),
        (96_000.0, 200.0, 82.4, 12_000.0),
        (22_050.0, 200.0, 20.0, 20_000.0),
    ];
    for (sr, scale, o6, fr) in corners {
        let mut set = GsStringSet::default();
        set.scale_mm = scale as f64;
        set.strings[5].open_hz = o6 as f64;
        let out = render_set(set, sr, 0.3, &[(FREQ, Sig::C(fr)), (STRING, Sig::C(6.0))]);
        assert!(
            out.iter().all(|x| x.is_finite() && x.abs() < 4.0),
            "corner {sr} {scale} {o6} {fr}"
        );
    }
    let rates = [22_050.0f32, 44_100.0, 48_000.0, 96_000.0];
    for trial in 0..80 {
        let sr = rates[trial % 4];
        let mut set = GsStringSet::default();
        set.scale_mm = 200.0 + 1000.0 * rnd();
        set.hmix = rnd();
        set.vel_corner_hz = 100.0 + 30_000.0 * rnd();
        set.vel_corner_exp = 6.0 * rnd();
        for st in &mut set.strings {
            st.open_hz = 20.0 * (2f64).powf(10.0 * rnd());
            st.b = 0.01 * rnd().powi(3);
            for t in &mut st.t60 {
                *t = 0.01 * (12_000f64).powf(rnd());
            }
            st.bridge_share = rnd();
            st.split_cents = 100.0 * rnd() - 50.0;
            st.hv = 4.0 * rnd();
        }
        let ranges: [(usize, f64, f64); 11] = [
            (FREQ, 10.0, 30_000.0),
            (VEL, -1.0, 2.0),
            (STRING, -2.0, 8.0),
            (POS, -50.0, 2000.0),
            (ANGLE, -200.0, 200.0),
            (HARD, -1.0, 2.0),
            (WIDTH, -5.0, 100.0),
            (DAMP, -1.0, 2.0),
            (REL, -1.0, 10.0),
            (VARY, -1.0, 2.0),
            (SEED, -1e6, 1e6),
        ];
        let mut ports: Vec<(usize, Sig)> = Vec::new();
        for &(k, lo, hi) in &ranges {
            ports.push((k, Sig::C((lo + (hi - lo) * rnd()) as f32)));
        }
        let out = render_set(set, sr, 0.25, &ports);
        let pk = peak(&out);
        assert!(
            out.iter().all(|x| x.is_finite()),
            "trial {trial} not finite"
        );
        assert!(pk < 4.0, "trial {trial} peak {pk}");
    }
}

// --- Release, damping, done -------------------------------------------------

/// A gate fall damps the string (a ramp, not a cut) and the node reports
/// done; a held note frees itself once rung out; `damp` shortens the decay
/// while held.
#[test]
fn release_damps_and_reports_done() {
    let sr = 44100.0;
    let mut node = GuitarString::new();
    let n = (2.0 * sr) as usize;
    let off = (0.5 * sr) as usize;
    let ports = vec![
        (FREQ, Sig::C(110.0)),
        (GATE, f(move |i| if i < off { 1.0 } else { 0.0 })),
        (REL, Sig::C(0.15)),
    ];
    let out = render_with(&mut node, sr, 64, n, &ports, 1);
    let held = peak(&out[off - 2000..off]);
    let after = peak(&out[off..off + 220]);
    assert!(
        after > 0.5 * held,
        "release cut the note: {after} vs {held}"
    );
    let tail = peak(&out[off + (0.3 * sr) as usize..off + (0.35 * sr) as usize]);
    assert!(tail < held * 1e-3, "not damped: {tail} vs {held}");
    assert!(node.is_done(), "a released string must report done");
    assert!(out[(1.0 * sr) as usize..].iter().all(|&x| x == 0.0));

    let mut node = GuitarString::new();
    let out = render_with(
        &mut node,
        sr,
        64,
        (8.0 * sr) as usize,
        &[(FREQ, Sig::C(880.0))],
        1,
    );
    assert!(
        node.is_done(),
        "a held string that has rung out must report done"
    );
    assert_eq!(*out.last().unwrap(), 0.0);

    let t60 = |damp: f32| {
        let mut p = plain(110.0);
        p.push((DAMP, Sig::C(damp)));
        let sig = render(sr, 1.0, &p);
        LN_1000 / partial(&sig, sr as f64, 110.0, 0.05, 0.25, 0.08).1
    };
    let (open_t, muted) = (t60(0.0), t60(1.0));
    println!("fundamental T60 open {open_t:.2} s, muted {muted:.3} s");
    assert!(
        muted < 0.3 && muted < open_t / 10.0,
        "palm mute too weak: {muted}"
    );
}

/// The engine frees a voice whose output is the string times a gain, and a
/// string plus a `modalBank` body only once both have rung out; an `exc`
/// source that reports done does not free a sounding string.
#[test]
fn engine_frees_the_voice_when_the_string_is_done() {
    use microsynth::dsl;
    let reg = common::builtin_registry();
    let run = |src: &str, secs: f32| {
        let def = dsl::compile(src, &reg).unwrap().remove(0);
        let mut e = Engine::new(EngineConfig {
            sample_rate: 44100.0,
            block_size: 64,
        });
        let s = e.instantiate_synthdef(&def);
        e.graph_mut().set_sink(s.output_node());
        e.prepare();
        let blocks = (secs * 44100.0 / 64.0) as usize;
        for b in 0..blocks {
            e.render();
            if e.free_done_synths() > 0 {
                return Some(b as f32 * 64.0 / 44100.0);
            }
        }
        None
    };
    let gate = "(1.0 - (line 0.0 1.0 0.3))";
    let at = run(
        &format!("synthdef g = guitarString 220.0 (clip ({gate} * 1000.0) 0.0 1.0) * 0.5"),
        3.0,
    )
    .expect("the voice is freed");
    assert!((0.3..1.5).contains(&at), "freed at {at}");
    let body = "modalBank s 100.0 1.0 0.0 1.0 0.0 1.03 0.5 2.0";
    let at_body = run(
        &format!(
            "synthdef g =\n  let s = guitarString 220.0 (clip ({gate} * 1000.0) 0.0 1.0)\n  s * 0.5 + {body}"
        ),
        12.0,
    )
    .expect("the voice with a body is freed");
    assert!(at_body > at + 0.5, "body cut short: {at_body} vs {at}");
    let at_exc = run(
        "synthdef g = guitarString 220.0 0.0 0.8 (-1.0) 140.0 20.0 0.5 10.0 0.0 (whiteNoise * perc 0.001 0.01) * 0.5",
        3.0,
    );
    assert!(
        at_exc.is_none_or(|t| t > 0.5),
        "freed with the burst at {at_exc:?}"
    );
}

// --- Outputs ------------------------------------------------------------

/// `bridge` (tap 1) is the vertical bridge force: identical to `out` when
/// the set's `hmix` is 0, different otherwise; and a `modalBank` body fed by
/// the string renders exactly the resonator bank applied to the string's
/// own output.
#[test]
fn bridge_tap_and_the_body_stage() {
    let sr = 44100.0;
    let with = |tap: f32, hmix: f64| {
        let mut set = GsStringSet::default();
        set.hmix = hmix;
        render_set(set, sr, 0.5, &[(FREQ, Sig::C(146.8)), (TAP, Sig::C(tap))])
    };
    assert_eq!(with(0.0, 0.0), with(1.0, 0.0));
    assert_ne!(with(0.0, 0.3), with(1.0, 0.3));

    use microsynth::dsl;
    let reg = common::builtin_registry();
    let render_def = |src: &str| {
        let def = dsl::compile(src, &reg).unwrap().remove(0);
        let mut e = Engine::new(EngineConfig {
            sample_rate: sr,
            block_size: 64,
        });
        let s = e.instantiate_synthdef(&def);
        e.graph_mut().set_sink(s.output_node());
        e.prepare();
        let mut out = Vec::new();
        for _ in 0..200 {
            out.extend_from_slice(e.render().unwrap().channel(0).samples());
        }
        out
    };
    let string_src =
        "guitarString 146.8 1.0 0.8 (-1.0) 140.0 20.0 0.5 10.0 0.0 0.0 0.0 0.15 0.0 1.0";
    let s = render_def(&format!("synthdef a = {string_src}"));
    let body = render_def(&format!(
        "synthdef b = modalBank ({string_src}) 100.0 1.0 0.0 1.0 0.0 1.88 0.7 0.44"
    ));
    let fk = 188.0f64;
    let rho = (-LN_1000 / (0.44 * sr as f64)).exp();
    let om = 2.0 * PI * fk / sr as f64;
    let (c, sn) = (rho * om.cos(), rho * om.sin());
    let tilt = 1.0 / (1.0 + (fk / 25_600.0).powi(4)).sqrt();
    let (mut re, mut im) = (0.0f64, 0.0f64);
    let mut worst = 0.0f64;
    for (x, y) in s.iter().zip(&body) {
        let (a, b) = (re, im);
        re = c * a - sn * b + 0.7 * tilt * *x as f64;
        im = sn * a + c * b;
        worst = worst.max((im - *y as f64).abs());
    }
    assert!(worst < 1e-5, "modalBank of the string differs by {worst}");
    assert!(peak(&body) > 0.01);
}

/// Output channels are copies of one mono string driven from input channel
/// 0: a stereo `exc` with different channels gives identical channels, and
/// changing input channel 1 changes nothing.
#[test]
fn reads_channel_0_and_copies_to_every_channel() {
    let sr = 44100.0;
    let render2 = |right: f32| {
        let mut node = GuitarString::new();
        let ctx = ProcessContext::new(sr, 64);
        node.init(&ctx);
        let mut out = [Vec::new(), Vec::new()];
        for blk in 0..40 {
            let mut exc = AudioBuffer::new(2, 64);
            for i in 0..64 {
                let g = blk * 64 + i;
                exc.channel_mut(0).samples_mut()[i] = if g % 700 == 3 { 0.2 } else { 0.0 };
                exc.channel_mut(1).samples_mut()[i] = if g % 300 == 1 { right } else { 0.0 };
            }
            let mut inputs: Vec<Option<&AudioBuffer>> = vec![None; GS_N_PORTS];
            inputs[EXC] = Some(&exc);
            let mut o = AudioBuffer::new(2, 64);
            node.process(&ctx, &inputs, &mut o);
            out[0].extend_from_slice(o.channel(0).samples());
            out[1].extend_from_slice(o.channel(1).samples());
        }
        out
    };
    let a = render2(0.5);
    let b = render2(-0.9);
    assert_eq!(a[0], a[1]);
    assert_eq!(a[0], b[0]);
}

// --- Table binding: the string set and the residual -------------------------

fn residual_entry(ir: &[f32], rate: f32) -> PitchEntry {
    PitchEntry {
        f0_hz: 0.0,
        inharmonicity_stretch: 0.0,
        partial_freqs: vec![],
        k_channels: 1,
        j_noise: ir.len() as u32,
        coefficients: ir.to_vec(),
        metadata: vec![("residual_rate_hz".into(), rate)],
    }
}

fn set_entry(set: &GsStringSet) -> PitchEntry {
    PitchEntry {
        f0_hz: 0.0,
        inharmonicity_stretch: 0.0,
        partial_freqs: vec![],
        k_channels: 0,
        j_noise: 0,
        coefficients: vec![],
        metadata: set.to_metadata(),
    }
}

fn table(entries: Vec<PitchEntry>) -> Arc<CoeffTable> {
    Arc::new(CoeffTable {
        name: "gs".into(),
        entries,
    })
}

/// A bound residual is convolved into the pluck: a unit impulse plays
/// exactly as the unbound node, and `δ + 0.5 δ(t - k)` plays the unbound
/// note plus half of itself `k` samples later (the system is linear).
#[test]
fn bound_residual_is_commuted_into_the_pluck() {
    let sr = 44100.0;
    let n = 20_000;
    let ports = vec![(FREQ, Sig::C(196.0)), (VARY, Sig::C(0.0))];
    let plain_out = render_with(&mut GuitarString::new(), sr, 64, n, &ports, 1);
    let mut unit = GuitarString::with_table(table(vec![residual_entry(&[1.0], sr)]));
    let unit_out = render_with(&mut unit, sr, 64, n, &ports, 1);
    let worst = plain_out
        .iter()
        .zip(&unit_out)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-6, "unit residual differs by {worst}");

    let k = 37;
    let mut ir = vec![0.0f32; k + 1];
    ir[0] = 1.0;
    ir[k] = 0.5;
    let mut two = GuitarString::with_table(table(vec![residual_entry(&ir, sr)]));
    let two_out = render_with(&mut two, sr, 64, n, &ports, 1);
    let mut worst = 0.0f32;
    for i in 0..n {
        let want = plain_out[i] + if i >= k { 0.5 * plain_out[i - k] } else { 0.0 };
        worst = worst.max((two_out[i] - want).abs());
    }
    assert!(worst < 1e-4, "two-tap residual differs by {worst}");
}

/// A residual at another rate is resampled linearly: `[1, 0, 0.5]` at
/// 22.05 kHz is `[1, 0.5, 0, 0.25, 0.5, 0.25]` at 44.1 kHz, and the note
/// plays exactly the unbound note convolved with that.
#[test]
fn residual_at_another_rate_is_resampled() {
    let sr = 44100.0;
    let n = 15_000;
    let ports = vec![(FREQ, Sig::C(220.0)), (VARY, Sig::C(0.0))];
    let plain_out = render_with(&mut GuitarString::new(), sr, 64, n, &ports, 1);
    let mut bound =
        GuitarString::with_table(table(vec![residual_entry(&[1.0, 0.0, 0.5], 22_050.0)]));
    let out = render_with(&mut bound, sr, 64, n, &ports, 1);
    let r = [1.0f32, 0.5, 0.0, 0.25, 0.5, 0.25];
    let mut worst = 0.0f32;
    for i in 0..n {
        let want: f32 = r
            .iter()
            .enumerate()
            .filter(|(k, _)| i >= *k)
            .map(|(k, c)| c * plain_out[i - k])
            .sum();
        worst = worst.max((out[i] - want).abs());
    }
    assert!(worst < 1e-4, "resampled residual differs by {worst}");
}

/// The string set travels in the table: a node bound to a table carrying a
/// set plays exactly as a bare node built on that set (and with no residual
/// entry its residual is a unit impulse); a table with neither entry plays
/// the defaults; the set's metadata round-trips.
#[test]
fn string_set_is_read_from_the_table() {
    let sr = 44100.0;
    let n = 12_000;
    let mut set = GsStringSet::default();
    set.strings[1].open_hz = 250.0;
    set.strings[1].b = 1.0e-4;
    set.strings[1].t60 = [3.0, 2.0, 1.0, 0.5];
    set.hmix = 0.5;
    set.vel_corner_exp = 0.7;
    let ports = vec![(FREQ, Sig::C(250.0)), (STRING, Sig::C(2.0))];
    let from_set = render_with(&mut GuitarString::with_set(set), sr, 64, n, &ports, 3);
    let mut bound = GuitarString::with_table(table(vec![set_entry(&set)]));
    assert_eq!(
        bound.string_set(),
        &GsStringSet::from_table(&table(vec![set_entry(&set)])).unwrap()
    );
    let from_table = render_with(&mut bound, sr, 64, n, &ports, 3);
    // f32 metadata: compare within float rounding.
    let worst = from_set
        .iter()
        .zip(&from_table)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-4, "table set differs by {worst}");
    assert_ne!(
        from_set,
        render_with(&mut GuitarString::new(), sr, 64, n, &ports, 3)
    );
    let mut empty = GuitarString::with_table(table(vec![]));
    assert_eq!(
        render_with(&mut empty, sr, 64, n, &ports, 3),
        render_with(&mut GuitarString::new(), sr, 64, n, &ports, 3)
    );
}

/// The table-bound form resolves through an IR `table_bindings` entry.
#[test]
fn table_bound_form_resolves_through_the_ir() {
    use microsynth::ir::{IrNode, IrSynthDef, IrTableBinding, SynthDefClass};
    let mut reg = microsynth::dsl::compiler::UGenRegistry::new();
    microsynth::register_builtins(&mut reg);
    microsynth::register_table_bound_builtins(&mut reg);
    let mut bank = CoeffTableBank::new();
    let ir_resp: Vec<f32> = (0..400)
        .map(|i| (-(i as f32) / 60.0).exp() * if i % 2 == 0 { 1.0 } else { -0.6 })
        .collect();
    let id = bank.register(CoeffTable {
        name: "gs".into(),
        entries: vec![
            set_entry(&GsStringSet::default()),
            residual_entry(&ir_resp, 48_000.0),
        ],
    });
    let ir = IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "gs_probe".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![IrNode::UGen {
            kind: "guitarString".into(),
            consts: vec![(0, 196.0)],
        }],
        edges: vec![],
        params: vec![],
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding {
            node: 0,
            table_id: id.0,
        }],
        output_node: 0,
    };
    ir.validate(&reg).expect("valid IR");
    let def = ir.compile_with_tables(&reg, &bank).expect("resolves");
    let mut engine = Engine::new(EngineConfig {
        sample_rate: 44100.0,
        block_size: 128,
    });
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let out = engine.render_offline(100).remove(0);
    assert!(peak(&out) > 0.05 && out.iter().all(|x| x.is_finite()));
    let entry = reg.table_bound_entry("guitarString").expect("registered");
    assert_eq!(entry.category, UGenCategory::Physical);
}

// --- String assignment -------------------------------------------------------

/// The built-in rule picks the highest string at or below the note; an
/// explicit `string` overrides it.
#[test]
fn string_rule_and_override() {
    let sr = 44100.0;
    let note = |freq: f32, string: f32| {
        let mut p = plain(freq);
        p.push((STRING, Sig::C(string)));
        render(sr, 0.5, &p)
    };
    assert_eq!(note(220.0, -1.0), note(220.0, 3.0));
    assert_ne!(note(220.0, -1.0), note(220.0, 4.0));
    assert_eq!(note(246.941_65, -1.0), note(246.941_65, 2.0));
    assert_eq!(note(73.42, -1.0), note(73.42, 6.0));
}

// --- Second review: hard loss constraint, B region, residual rate ---------

/// A random but table-reachable string set (every value within
/// `GsStringSet::from_table`'s ranges, anchors in any order).
fn random_set(rnd: &mut impl FnMut() -> f64, t60_max: f64) -> GsStringSet {
    let mut set = GsStringSet::default();
    set.scale_mm = 200.0 + 1000.0 * rnd();
    for a in &mut set.anchor_hz {
        *a = 50.0 * (800f64).powf(rnd());
    }
    for st in &mut set.strings {
        st.open_hz = 40.0 * (2f64).powf(5.0 * rnd());
        st.b = 2e-3 * rnd().powi(3);
        for t in &mut st.t60 {
            *t = 0.05 * (t60_max / 0.05).powf(rnd());
        }
        st.bridge_share = rnd();
        st.hv = 4.0 * rnd();
    }
    set
}

fn xorshift(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The loss constraint is hard: over random table-reachable string sets,
/// notes and rates, no frequency (DC to Nyquist) loses less per pass than
/// half the fundamental's loss, so nothing outlasts the note.
#[test]
fn no_frequency_outlasts_the_note_for_random_sets() {
    use microsynth::ugens::GsLossForm;
    let mut rnd = xorshift(0x9e37_79b9_7f4a_7c15);
    let rates = [22_050.0f64, 44_100.0, 48_000.0, 96_000.0];
    let (mut n, mut worst) = (0usize, f64::INFINITY);
    let mut forms = [0usize; 3];
    for trial in 0..600 {
        let sr = rates[trial % 4];
        let set = random_set(&mut rnd, 120.0);
        let s = (rnd() * 6.0) as usize % 6;
        let f = (set.strings[s].open_hz * 2f64.powf(24.0 * rnd() / 12.0)).min(sr / 8.0);
        let d = gs_note_design(sr, &set, f, (s + 1) as f32);
        for (p, loss) in d.loss.iter().enumerate() {
            n += 1;
            forms[match d.loss_form[p] {
                GsLossForm::Shelf => 0,
                GsLossForm::OnePole => 1,
                GsLossForm::Constant => 2,
            }] += 1;
            let fp = if p == 0 { f } else { d.h_freq };
            let l0 = -loss.mag(2.0 * PI * fp / sr).ln();
            let mut weakest = f64::INFINITY;
            for k in 0..=512 {
                weakest = weakest.min(-loss.mag(PI * k as f64 / 512.0).ln());
            }
            let ratio = weakest / l0;
            worst = worst.min(ratio);
            assert!(
                ratio >= 0.5,
                "trial {trial} pol {p}: weakest loss {weakest:.3e} vs fundamental {l0:.3e} ({:?})",
                d.loss_form[p]
            );
        }
    }
    println!(
        "random sets: {n} loss filters: shelf {}, one-pole {}, constant {}; worst weakest/fundamental {worst:.3}",
        forms[0], forms[1], forms[2]
    );
}

/// The reviewer's failing case (a 24 kHz component that held forever at
/// 48 kHz) now dies with the note: the node reports done; and random sets
/// with short decays report done within a few times their longest T60.
#[test]
fn odd_sets_still_report_done() {
    let mut set = GsStringSet::default();
    set.strings[2].t60 = [0.127, 0.711, 84.7, 0.788];
    set.anchor_hz = [2998.0, 4607.0, 1176.0];
    let mut node = GuitarString::with_set(set);
    let out = render_with(
        &mut node,
        48_000.0,
        64,
        3 * 48_000,
        &[(FREQ, Sig::C(237.8)), (STRING, Sig::C(3.0))],
        1,
    );
    assert!(node.is_done(), "the reviewer's case still rings");
    assert!(out[2 * 48_000..].iter().all(|&x| x == 0.0));

    let mut rnd = xorshift(42);
    for trial in 0..12 {
        let set = random_set(&mut rnd, 1.0);
        let s = trial % 6;
        let f = set.strings[s].open_hz * 2f64.powf(12.0 * rnd() / 12.0);
        let t60_max = set.strings[s].t60.iter().fold(0.0f64, |m, t| m.max(*t));
        let secs = 6.0 * t60_max + 1.0;
        let mut node = GuitarString::with_set(set);
        render_with(
            &mut node,
            44_100.0,
            128,
            (secs * 44_100.0) as usize,
            &[(FREQ, Sig::C(f as f32)), (STRING, Sig::C((s + 1) as f32))],
            1,
        );
        assert!(node.is_done(), "trial {trial}: not done after {secs:.1} s");
    }
}

/// The reachable region of the B pre-warp, as documented: at 44.1 kHz (the
/// render rate) the realised `B` (`b_fit`) is within 4 % of the request for
/// open-string `b` from 1e-5 to 2e-3 on every string at frets 0, 7 and 12;
/// at 22.05 kHz the same from 3e-5. A table's `b` is clamped to 2e-3.
#[test]
fn b_round_trips_over_the_documented_region() {
    for (sr, lowest) in [(44_100.0f64, 1e-5), (22_050.0, 3e-5)] {
        let mut worst = (0.0f64, String::new());
        for bo in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3] {
            if bo < lowest {
                continue;
            }
            for s in 1..=6usize {
                for fret in [0, 7, 12] {
                    let mut set = GsStringSet::default();
                    set.strings[s - 1].b = bo;
                    let f = set.strings[s - 1].open_hz * 2f64.powf(fret as f64 / 12.0);
                    let d = gs_note_design(sr, &set, f, s as f32);
                    let r = d.b_fit.expect("enough partials") / d.b - 1.0;
                    if r.abs() > worst.0 {
                        worst = (
                            r.abs(),
                            format!("b {bo:.0e} s{s} fret {fret}: {:+.2} %", 100.0 * r),
                        );
                    }
                }
            }
        }
        println!("{sr}: worst B round trip {}", worst.1);
        assert!(worst.0 < 0.04, "{sr}: {}", worst.1);
    }
    let mut big = GsStringSet::default();
    big.strings[0].b = 1e-2;
    let read = GsStringSet::from_table(&table(vec![set_entry(&big)])).unwrap();
    assert_eq!(read.strings[0].b, microsynth::ugens::GS_MAX_B);
}

/// A residual entry declaring a rate outside 4–384 kHz is ignored (a
/// 3-sample residual declared at 1 Hz used to play at a peak of 15.85).
#[test]
fn residual_with_an_implausible_rate_is_ignored() {
    let sr = 44100.0;
    let n = 8000;
    let ports = vec![(FREQ, Sig::C(220.0)), (VARY, Sig::C(0.0))];
    let plain_out = render_with(&mut GuitarString::new(), sr, 64, n, &ports, 1);
    for rate in [1.0f32, 3_000.0, 500_000.0] {
        let mut node =
            GuitarString::with_table(table(vec![residual_entry(&[1.0, 0.8, 0.6], rate)]));
        let out = render_with(&mut node, sr, 64, n, &ports, 1);
        assert_eq!(out, plain_out, "rate {rate} was not ignored");
    }
    // At the bottom of the range the residual is used (resampled, it is about
    // 33 samples near 1: a loud but legitimate body response).
    let mut node = GuitarString::with_table(table(vec![residual_entry(&[1.0, 0.8, 0.6], 4_000.0)]));
    let out = render_with(&mut node, sr, 64, n, &ports, 1);
    assert_ne!(out, plain_out, "a 4 kHz residual must be used");
    assert!(out.iter().all(|x| x.is_finite()));
}
