//! `guitarString`: a plucked steel string (two-polarisation waveguide).
//!
//! The tests drive the node directly (and, where the engine's done rule or
//! the DSL matters, through a compiled def) and measure what it plays:
//! tuning, partial frequencies and decays, inharmonicity, beating, the pluck
//! comb, velocity and brightness, determinism, block-size independence,
//! hygiene, release and done, the `bridge` tap, and agreement with an
//! independent modal oracle built from the same physical parameters.

use microsynth::context::ProcessContext;
use microsynth::node::UGen;
use microsynth::ugens::{
    GS_DEFAULT_STRINGS, GS_F_HI, GS_FIXED_PORTS, GuitarString, gs_string_port,
};
use microsynth::*;

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
const SPLIT: usize = 11;
const HMIX: usize = 12;
const HV: usize = 13;
const REL: usize = 14;
const VARY: usize = 16;
const TAP: usize = 17;
const N_PORTS: usize = GS_FIXED_PORTS + 30;

const LN_1000: f64 = 6.907_755_278_982_137;

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
        let bufs: Vec<Option<AudioBuffer>> = (0..N_PORTS)
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

fn render(sr: f32, secs: f64, ports: &[(usize, Sig)]) -> Vec<f32> {
    let mut node = GuitarString::new();
    render_with(&mut node, sr, 64, (secs * sr as f64) as usize, ports, 1)
}

/// Ports for a plain test note: no per-note variation.
fn plain(freq: f32) -> Vec<(usize, Sig)> {
    vec![(FREQ, Sig::C(freq)), (VARY, Sig::C(0.0))]
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
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / n as f64).cos();
            let idx = s0 + k;
            let ph = 2.0 * std::f64::consts::PI * f * idx as f64 / sr;
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
/// magnitude slope of windowed demodulation (refined twice).
fn partial(sig: &[f32], sr: f64, f_guess: f64, t0: f64, t1: f64, win: f64) -> (f64, f64) {
    let mut fq = f_guess;
    let mut sigma = 0.0;
    for _ in 0..3 {
        let fr = demod(sig, sr, fq, t0, t1, win, win / 4.0);
        let ts: Vec<f64> = fr.iter().map(|x| x.0).collect();
        let mut ph: Vec<f64> = fr.iter().map(|x| x.2.atan2(x.1)).collect();
        for i in 1..ph.len() {
            while ph[i] - ph[i - 1] > std::f64::consts::PI {
                ph[i] -= 2.0 * std::f64::consts::PI;
            }
            while ph[i] - ph[i - 1] < -std::f64::consts::PI {
                ph[i] += 2.0 * std::f64::consts::PI;
            }
        }
        let (slope, _) = line_fit(&ts, &ph);
        fq += slope / (2.0 * std::f64::consts::PI);
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

/// The default open pitches.
fn open(s: usize) -> f64 {
    GS_DEFAULT_STRINGS[s - 1][0] as f64
}

// --- The documented mapping, re-implemented independently -----------------

/// What the module doc says a note at `freq` should be: the ideal
/// fundamental, `B`, the loss filters `(g, a)` of V and H, and the pluck's
/// geometry. Written from the documentation, not from the node's code.
struct Spec {
    sr: f64,
    f0: f64,
    b: f64,
    /// (g, a) per polarisation.
    loss: [(f64, f64); 2],
    /// Pitch of H relative to V.
    h_ratio: f64,
    /// Pluck: amplitude per polarisation, comb lag m, period P, ramp, dyn-lp R.
    amp: [f64; 2],
    m: [usize; 2],
    period: [f64; 2],
    ramp: f64,
    dyn_r: f64,
}

#[derive(Clone, Copy)]
struct NoteArgs {
    freq: f64,
    string: Option<usize>,
    vel: f64,
    pos: f64,
    angle_deg: f64,
    hard: f64,
    width: f64,
    split: f64,
    hv: f64,
}

impl NoteArgs {
    fn new(freq: f64) -> Self {
        NoteArgs {
            freq,
            string: None,
            vel: 0.8,
            pos: 140.0,
            angle_deg: 20.0,
            hard: 0.5,
            width: 10.0,
            split: 2.0,
            hv: 0.4,
        }
    }

    fn ports(&self) -> Vec<(usize, Sig)> {
        let mut p = plain(self.freq as f32);
        p.push((VEL, Sig::C(self.vel as f32)));
        p.push((POS, Sig::C(self.pos as f32)));
        p.push((ANGLE, Sig::C(self.angle_deg as f32)));
        p.push((HARD, Sig::C(self.hard as f32)));
        p.push((WIDTH, Sig::C(self.width as f32)));
        p.push((SPLIT, Sig::C(self.split as f32)));
        p.push((HV, Sig::C(self.hv as f32)));
        if let Some(s) = self.string {
            p.push((STRING, Sig::C(s as f32)));
        }
        p
    }
}

/// Partial number of a stiff string at `f`, by bisection on
/// `f = n f0 sqrt(1 + B n^2)`.
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

/// |g (1 + a) / (1 + a e^{-jw})|.
fn onepole_mag(g: f64, a: f64, w: f64) -> f64 {
    g * (1.0 + a) / (1.0 + 2.0 * a * w.cos() + a * a).sqrt()
}

fn spec(sr: f64, na: NoteArgs) -> Spec {
    let opens: Vec<f64> = (1..=6).map(open).collect();
    let s = na.string.unwrap_or_else(|| {
        let tol = 2f64.powf(0.25 / 12.0);
        let mut best = 6;
        let mut best_o = 0.0;
        for (i, &o) in opens.iter().enumerate() {
            if o <= na.freq * tol && o > best_o {
                best = i + 1;
                best_o = o;
            }
        }
        best
    });
    let row = GS_DEFAULT_STRINGS[s - 1];
    let (o, b_open, t, h, r) = (
        row[0] as f64,
        row[1] as f64,
        row[2] as f64,
        row[3] as f64,
        row[4] as f64,
    );
    let k = na.freq / o;
    let b = b_open * k * k;
    let f0 = na.freq / (1.0 + b).sqrt();
    let sig_t = LN_1000 / t;
    let sig_h = LN_1000 / h;
    let intrinsic = |f: f64| {
        let ia = (1.0 - r) * sig_t;
        let ib = (sig_h - r * sig_t).max(0.0);
        (ia + (ib - ia) * (f - o) / (GS_F_HI - o)).max(0.0)
    };
    let fa = na.freq;
    let fb = GS_F_HI.max(2.5 * fa).min(0.4 * sr);
    let tau = |f: f64| {
        let e = f * 1e-6;
        (n_of(f + e, f0, b) - n_of(f - e, f0, b)) / (2.0 * e)
    };
    let mut loss = [(0.0, 0.0); 2];
    for (p, l) in loss.iter_mut().enumerate() {
        let br = r * sig_t * k * if p == 0 { 1.0 } else { na.hv };
        let ma = (-(br + intrinsic(fa)) * tau(fa)).exp();
        let mb = (-(br + intrinsic(fb)) * tau(fb)).exp();
        let (wa, wb) = (
            2.0 * std::f64::consts::PI * fa / sr,
            2.0 * std::f64::consts::PI * fb / sr,
        );
        // Bisect the pole for the magnitude ratio.
        let ratio = |a: f64| onepole_mag(1.0, a, wb) / onepole_mag(1.0, a, wa);
        let target = mb / ma;
        let (mut lo, mut hi) = (-0.95f64, 0.0f64);
        let a = if target >= 1.0 {
            0.0
        } else if ratio(lo) > target {
            lo
        } else {
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if ratio(mid) > target {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            0.5 * (lo + hi)
        };
        let g = ma / onepole_mag(1.0, a, wa);
        *l = (g, a);
    }
    let scale = 645.0;
    let beta = na.pos / (scale / k);
    let ang = na.angle_deg.to_radians();
    let h_ratio = 2f64.powf(-na.split / 1200.0);
    let period = [sr / na.freq, sr / (na.freq * h_ratio)];
    let m = [
        (beta * period[0]).round() as usize,
        (beta * period[1]).round() as usize,
    ];
    let w_eff = na.width * (1.25 - 0.5 * na.vel) * (1.0 - 0.5 * na.hard);
    let c = 2.0 * scale * 1e-3 * o;
    let ramp = (w_eff * 1e-3 / c * sr).clamp(1.0, 64.0);
    let bright = na.vel * (0.5 + 0.5 * na.hard);
    let f_dyn = (500.0 * 2f64.powf(6.0 * bright)).min(0.45 * sr);
    Spec {
        sr,
        f0,
        b,
        loss,
        h_ratio,
        amp: [na.vel * ang.cos(), na.vel * ang.sin()],
        m,
        period,
        ramp,
        dyn_r: (-2.0 * std::f64::consts::PI * f_dyn / sr).exp(),
    }
}

impl Spec {
    /// Partial `n`'s frequency (polarisation `p`).
    fn freq(&self, n: u32, p: usize) -> f64 {
        let nf = n as f64;
        let f = nf * self.f0 * (1.0 + self.b * nf * nf).sqrt();
        if p == 0 { f } else { f * self.h_ratio }
    }

    /// Partial `n`'s amplitude decay rate (1/s), polarisation `p`.
    fn sigma(&self, n: u32, p: usize) -> f64 {
        let f = self.freq(n, p);
        let f0 = if p == 0 {
            self.f0
        } else {
            self.f0 * self.h_ratio
        };
        let e = f * 1e-6;
        let tau = (n_of(f + e, f0, self.b) - n_of(f - e, f0, self.b)) / (2.0 * e);
        let (g, a) = self.loss[p];
        -onepole_mag(g, a, 2.0 * std::f64::consts::PI * f / self.sr).ln() / tau
    }

    /// Partial `n`'s initial amplitude at the output (polarisation `p`, before
    /// `hmix`): twice the pluck waveform's spectrum at the partial over the
    /// period, times the output gain 0.5.
    fn level(&self, n: u32, p: usize) -> f64 {
        let w = 2.0 * std::f64::consts::PI * self.freq(n, p) / self.sr;
        let pp = self.period[p];
        let beta = self.m[p] as f64 / pp;
        let ramp = self.ramp;
        let step = |t: f64| {
            if t <= 0.0 {
                0.0
            } else if t >= ramp {
                1.0
            } else {
                0.5 * (1.0 - (std::f64::consts::PI * t / ramp).cos())
            }
        };
        // Sampled step response q[t] = S(t + 1), as the doc's residual-free case.
        // Linear between samples, from 0 at t = -1 (the sample before it).
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
        let n_len = (pp + ramp + 8.0) as usize + 4000;
        let (mut re, mut im, mut y) = (0.0, 0.0, 0.0);
        for t in 0..n_len {
            let tf = t as f64;
            let u =
                self.amp[p] * ((1.0 - beta) * q(tf) - q(tf - self.m[p] as f64) + beta * q(tf - pp));
            y = (1.0 - self.dyn_r) * u + self.dyn_r * y;
            re += y * (w * tf).cos();
            im -= y * (w * tf).sin();
        }
        0.5 * 2.0 * (re * re + im * im).sqrt() / pp
    }
}

/// An independent oracle: the sum of damped inharmonic partials with the
/// spec's frequencies, decays and levels (V and H, H scaled by `hmix`).
fn oracle(sp: &Spec, secs: f64, nmax: u32, hmix: f64) -> Vec<f32> {
    let n = (secs * sp.sr) as usize;
    let mut out = vec![0.0f64; n];
    for p in 0..2 {
        let mix = if p == 0 { 1.0 } else { hmix };
        if sp.amp[p].abs() < 1e-9 || mix == 0.0 {
            continue;
        }
        for k in 1..=nmax {
            let f = sp.freq(k, p);
            if f > 0.45 * sp.sr {
                break;
            }
            let (a, s) = (sp.level(k, p) * mix, sp.sigma(k, p));
            let w = 2.0 * std::f64::consts::PI * f / sp.sr;
            let dec = (-s / sp.sr).exp();
            let (mut amp, mut ph) = (a, 0.0f64);
            for o in out.iter_mut() {
                *o += amp * ph.sin();
                amp *= dec;
                ph += w;
            }
        }
    }
    out.into_iter().map(|x| x as f32).collect()
}

/// Fit `B` from measured partial frequencies `(n, f_n)`: `(f_n/n)^2` is
/// linear in `n^2` with slope `f0^2 B`.
fn fit_b(parts: &[(u32, f64)]) -> f64 {
    let x: Vec<f64> = parts.iter().map(|(n, _)| (*n as f64).powi(2)).collect();
    let y: Vec<f64> = parts.iter().map(|(n, f)| (f / *n as f64).powi(2)).collect();
    let (slope, icpt) = line_fit(&x, &y);
    slope / icpt
}

fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

fn centroid(x: &[f32], sr: f64) -> f64 {
    // Magnitude-weighted mean frequency of a Hann-windowed DFT (direct, coarse).
    let n = x.len();
    let bins = 600usize;
    let df = (0.45 * sr) / bins as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for b in 1..bins {
        let f = b as f64 * df;
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &v) in x.iter().enumerate() {
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
            let ph = 2.0 * std::f64::consts::PI * f * i as f64 / sr;
            re += v as f64 * w * ph.cos();
            im -= v as f64 * w * ph.sin();
        }
        let m = (re * re + im * im).sqrt();
        num += f * m;
        den += m;
    }
    num / den
}

// --- Tuning ----------------------------------------------------------------

/// The first partial sits at `freq` within a cent from E2 to E6, at 44.1 and
/// 48 kHz, with the loss, dispersion and Thiran filters' phase included.
#[test]
fn tuning_within_a_cent_from_e2_to_e6() {
    let mut worst = 0.0f64;
    let mut report = String::new();
    for sr in [44100.0f32, 48000.0] {
        for midi in (40..=88).step_by(3) {
            let fr = 440.0 * 2f64.powf((midi as f64 - 69.0) / 12.0);
            let mut ports = plain(fr as f32);
            ports.push((ANGLE, Sig::C(0.0)));
            let sig = render(sr, 0.9, &ports);
            let (f1, _) = partial(&sig, sr as f64, fr, 0.1, 0.9, 0.2);
            let c = cents(f1, fr);
            worst = worst.max(c.abs());
            report += &format!(" {sr}/{midi}:{c:+.3}");
        }
    }
    println!("tuning error (cents):{report}\nworst {worst:.4}");
    assert!(worst < 1.0, "worst tuning error {worst:.3} cents:{report}");
}

/// A vibrato keeps the pitch on target: the mean pitch over whole vibrato
/// cycles stays within a cent, and a held bend lands within a cent.
#[test]
fn a_bend_lands_in_tune() {
    let sr = 44100.0;
    let fr = 196.0f32;
    let up = fr * 2f32.powf(2.0 / 12.0);
    let mut ports = plain(fr);
    ports.push((ANGLE, Sig::C(0.0)));
    // Bend up a whole tone over 0.1..0.2 s, then hold.
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

// --- Partials: decay, inharmonicity, oracle --------------------------------

/// Partial decays follow the documented loss law (two anchors, the one-pole
/// in between, the loop's round-trip time) within 10 %, on a wound and a
/// plain string, open and fretted.
#[test]
fn partial_decays_follow_the_loss_model() {
    let sr = 44100.0;
    let mut worst = 0.0f64;
    let mut report = String::new();
    for (fr, nmax) in [
        (82.406_89, 40u32),
        (123.47, 30),
        (246.941_65, 20),
        (392.0, 12),
    ] {
        let mut na = NoteArgs::new(fr);
        na.angle_deg = 0.0;
        let sp = spec(sr as f64, na);
        let sig = render(sr, 3.0, &na.ports());
        for n in [1u32, 2, 3, 5, 8, 12, 16, 20, 30, 40] {
            if n > nmax || sp.freq(n, 0) > 6000.0 {
                continue;
            }
            let want = sp.sigma(n, 0);
            let t1 = (0.2 + 2.0 * LN_1000 / want / 3.0).min(2.9);
            let (_, got) = partial(&sig, sr as f64, sp.freq(n, 0), 0.2, t1, 0.15);
            let err = got / want - 1.0;
            worst = worst.max(err.abs());
            report += &format!(
                "\n  f {fr:.1} n {n}: T60 want {:.2} got {:.2} ({:+.1} %)",
                LN_1000 / want,
                LN_1000 / got,
                100.0 * err
            );
        }
    }
    println!("decays:{report}\nworst {:.1} %", 100.0 * worst);
    assert!(
        worst < 0.10,
        "worst decay error {:.1} %:{report}",
        100.0 * worst
    );
}

/// Inharmonicity, in two parts that fail for different reasons.
///
/// 1. **The waveguide plays its design.** Every partial below 5 kHz sits
///    within 0.5 cents of the frequency the note's dispersion design predicts
///    from its own phase (`gs_designed_partials`). A miss here is a fault in
///    the loop (filter order, tuning solve, state handling).
/// 2. **The design approximates the stiff string.** Each predicted partial
///    below 5 kHz is within 4 cents of `n f0 sqrt(1 + B n^2)`, or 10 % of
///    its own stretch (against `n f0`) where that is larger, and `B` fitted
///    from the measured partials is within 25 % of the requested `B`. This
///    bound records the design's accuracy as built, the documented figure
///    in the module doc; it is not derived from perception and not
///    measurement slack. Its cause: at most eight second-order all-pass
///    sections are placed on bands of 2π of the target phase, so the
///    cascade's phase ripples between section centres, most at the top of
///    the designed band where the bands are widest and the stretch is
///    largest (16.5 cents of a 225-cent stretch at 4.9 kHz on the low E at
///    fret 12). On the plain strings one to three sections carry a small
///    stretch, so the absolute floor binds (3.9 cents of 21 at 5 kHz on the
///    open B, which is also why its fitted `B` is 22 % high). More sections
///    or a better-weighted design would tighten it; tightening the bound
///    means changing the design, not this number.
///
/// Partials are measured from the design's predicted frequency: seeding the
/// analysis with the ideal frequency instead (up to 45 Hz away near 5 kHz)
/// biases the windowed estimate towards the seed.
#[test]
fn inharmonicity_matches_the_requested_b() {
    use microsynth::ugens::gs_designed_partials;
    let sr = 44100.0;
    let mut report = String::new();
    let mut worst_b = 0.0f64;
    let (mut worst_play, mut worst_design) = (0.0f64, 0.0f64);
    let (mut at_play, mut at_design) = (String::new(), String::new());
    // Low E open and at frets 5 and 12, A open, D open, B open, high E fret 7.
    for fr in [82.406_89, 110.0, 164.81, 146.832_38, 246.941_65, 493.88] {
        let mut na = NoteArgs::new(fr);
        na.angle_deg = 0.0;
        if (fr - 110.0).abs() < 1.0 || (fr - 164.81).abs() < 1.0 {
            na.string = Some(6);
        }
        let sp = spec(sr as f64, na);
        let designed = gs_designed_partials(sr as f64, fr, sp.b, sp.loss[0].1, 30);
        let sig = render(sr, 1.6, &na.ports());
        let mut parts = Vec::new();
        for n in 1..=30u32 {
            let ideal = sp.freq(n, 0);
            if ideal > 5000.0 {
                break;
            }
            let des = designed[n as usize - 1];
            // Measure over the partial's first 50 dB (at most 0.1..0.9 s).
            let t1 = (0.1 + 50.0 / 60.0 * LN_1000 / sp.sigma(n, 0)).clamp(0.5, 0.9);
            let (fm, _) = partial(&sig, sr as f64, des, 0.1, t1, 0.2);
            let (cp, cd) = (cents(fm, des), cents(des, ideal));
            let stretch = cents(ideal, n as f64 * sp.f0);
            let allowed = (0.1 * stretch).max(4.0);
            if cd.abs() / allowed > worst_design {
                worst_design = cd.abs() / allowed;
                at_design = format!(
                    "f {fr:.1} n {n} ({ideal:.0} Hz): {cd:+.2} c of a {stretch:.0} c stretch"
                );
            }
            if cp.abs() > worst_play {
                worst_play = cp.abs();
                at_play = format!("f {fr:.1} n {n}: {cp:+.3} c");
            }
            parts.push((n, fm));
        }
        let b = fit_b(&parts);
        let err = b / sp.b - 1.0;
        worst_b = worst_b.max(err.abs());
        report += &format!(
            "\n  f {fr:.1} (B {:.2e}): fitted {:.2e} ({:+.1} %), {} partials",
            sp.b,
            b,
            100.0 * err,
            parts.len()
        );
    }
    println!(
        "inharmonicity:{report}\nworst B {:.1} %; played vs design {at_play}; design vs ideal {at_design}",
        100.0 * worst_b
    );
    assert!(worst_play < 0.5, "the loop misses its design: {at_play}");
    assert!(
        worst_design <= 1.0,
        "the design misses the stiff string: {at_design}"
    );
    assert!(worst_b < 0.25, "B error {:.1} %:{report}", 100.0 * worst_b);
}

/// The waveguide matches an independent modal oracle (damped inharmonic
/// partials with the spec's frequencies, decays and levels) on partial
/// frequency (within 3 cents), decay (within 10 %) and initial level
/// (within 1.5 dB rms over the first 12 partials).
#[test]
fn waveguide_matches_the_modal_oracle() {
    let sr = 44100.0;
    let mut report = String::new();
    let (mut wf, mut wd, mut wl) = (0.0f64, 0.0f64, 0.0f64);
    for fr in [110.0, 246.941_65] {
        let mut na = NoteArgs::new(fr);
        na.angle_deg = 0.0;
        let sp = spec(sr as f64, na);
        let sig = render(sr, 2.5, &na.ports());
        let orc = oracle(&sp, 2.5, 40, 0.3);
        let mut lvl_err = Vec::new();
        for n in 1..=12u32 {
            let fw = sp.freq(n, 0);
            let (fa, sa) = partial(&sig, sr as f64, fw, 0.2, 1.4, 0.15);
            let (fo, so) = partial(&orc, sr as f64, fw, 0.2, 1.4, 0.15);
            let dc = cents(fa, fo);
            let dd = sa / so - 1.0;
            // Initial level: extrapolate each to t = 0 from the early frames.
            let lev = |x: &[f32], s: f64| {
                let fr = demod(x, sr as f64, fw, 0.1, 0.4, 0.1, 0.025);
                let v: Vec<f64> = fr
                    .iter()
                    .map(|(t, re, im)| ((re * re + im * im).sqrt() * (s * t).exp()).ln())
                    .collect();
                20.0 / std::f64::consts::LN_10 * v.iter().sum::<f64>() / v.len() as f64
            };
            let dl = lev(&sig, sa) - lev(&orc, so);
            lvl_err.push(dl);
            wf = wf.max(dc.abs());
            wd = wd.max(dd.abs());
            report += &format!(
                "\n  f {fr:.1} n {n}: {dc:+.2} c, decay {:+.1} %, level {dl:+.2} dB",
                100.0 * dd
            );
        }
        let rms = (lvl_err.iter().map(|x| x * x).sum::<f64>() / lvl_err.len() as f64).sqrt();
        wl = wl.max(rms);
    }
    println!(
        "oracle:{report}\nworst {wf:.2} c, {:.1} %, level rms {wl:.2} dB",
        100.0 * wd
    );
    assert!(
        wf < 3.0,
        "frequency off the oracle by {wf:.2} cents:{report}"
    );
    assert!(
        wd < 0.10,
        "decay off the oracle by {:.1} %:{report}",
        100.0 * wd
    );
    assert!(wl < 1.5, "levels off the oracle by {wl:.2} dB rms:{report}");
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
        let mut na = NoteArgs::new(fr);
        na.angle_deg = 45.0;
        na.split = split;
        na.hv = 1.0; // equal decays, so the beat is deep and steady
        let sp = spec(sr as f64, na);
        let mut ports = na.ports();
        ports.push((HMIX, Sig::C(1.0)));
        let sig = render(sr, 4.0, &ports);
        let want = sp.freq(n, 0) - sp.freq(n, 1);
        // Envelope of the partial, then its dominant modulation frequency.
        let fr_ = demod(&sig, sr as f64, sp.freq(n, 0), 0.05, 4.0, 0.04, 0.005);
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
                re += r * (2.0 * std::f64::consts::PI * fb * t).cos();
                im += r * (2.0 * std::f64::consts::PI * fb * t).sin();
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
    let decay = |angle: f64| {
        let mut na = NoteArgs::new(fr);
        na.angle_deg = angle;
        let mut ports = na.ports();
        ports.push((HMIX, Sig::C(1.0)));
        let sig = render(sr, 2.5, &ports);
        partial(&sig, sr as f64, fr, 0.2, 2.4, 0.3).1
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
    let mut na = NoteArgs::new(fr);
    na.angle_deg = 0.0;
    na.pos = 645.0 / 5.0;
    let sp = spec(sr as f64, na);
    let sig = render(sr, 0.4, &na.ports());
    let lvl = |n: u32| {
        let fr_ = demod(&sig, sr as f64, sp.freq(n, 0), 0.02, 0.32, 0.3, 0.3);
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
/// on the D string, and the level rises by about 12 dB.
#[test]
fn velocity_brightens_the_attack() {
    let sr = 44100.0;
    let mut last = 0.0;
    let mut report = String::new();
    let mut c30 = 0.0;
    let mut p30 = 0.0;
    for v in [30.0f64, 50.0, 70.0, 90.0, 110.0, 120.0] {
        let mut na = NoteArgs::new(146.832_38);
        na.vel = v / 127.0;
        let sig = render(sr, 0.1, &na.ports());
        let c = centroid(&sig[..4096], sr as f64);
        let p = 20.0 * (peak(&sig) as f64).log10();
        report += &format!(" v{v}: {c:.0} Hz {p:.1} dB;");
        assert!(c > last, "centroid not monotonic:{report}");
        last = c;
        if v == 30.0 {
            c30 = c;
            p30 = p;
        }
        if v == 120.0 {
            assert!(
                c - c30 >= 500.0,
                "centroid rose only {:.0} Hz:{report}",
                c - c30
            );
            assert!(
                p - p30 >= 10.0,
                "level rose only {:.1} dB:{report}",
                p - p30
            );
        }
    }
    println!("velocity:{report}");
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
    // Two plucks in one voice differ.
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
/// vibrato, a palm mute and `exc` input.
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
    for k in 0..N_PORTS {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e30, -1e30] {
            let ports = vec![(FREQ, Sig::C(220.0)), (k, Sig::C(bad))];
            let out = render(sr, 0.3, &ports);
            // `exc` is an audio input clamped to ±10, so its output may be large.
            let bound = if k == EXC { 40.0 } else { 4.0 };
            assert!(
                out.iter().all(|x| x.is_finite() && x.abs() < bound),
                "port {k} = {bad}"
            );
        }
    }
    for bad in [0.0f32, -100.0, f32::NAN] {
        let mut node = GuitarString::new();
        let out = render_with(&mut node, sr, 64, 4410, &[(FREQ, Sig::C(bad))], 1);
        assert!(out.iter().all(|&x| x == 0.0), "freq {bad} must be silent");
        assert!(node.is_done(), "freq {bad} must report done");
    }
    // freq falling to 0 mid-note stops the string.
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

/// Every port swept over its range and beyond, in random combinations: the
/// output stays finite and bounded and the string still rings out.
#[test]
fn stable_for_every_port_value() {
    let sr = 44100.0;
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let ranges: [(usize, f64, f64); 15] = [
        (FREQ, 10.0, 9000.0),
        (VEL, -1.0, 2.0),
        (STRING, -2.0, 8.0),
        (POS, -50.0, 2000.0),
        (ANGLE, -200.0, 200.0),
        (HARD, -1.0, 2.0),
        (WIDTH, -5.0, 100.0),
        (DAMP, -1.0, 2.0),
        (SPLIT, -100.0, 100.0),
        (HMIX, -1.0, 3.0),
        (HV, -1.0, 6.0),
        (REL, -1.0, 10.0),
        (gs_string_port(6, 1), -0.01, 0.05),
        (gs_string_port(6, 2), -1.0, 500.0),
        (gs_string_port(6, 3), -1.0, 500.0),
    ];
    for trial in 0..60 {
        let mut ports: Vec<(usize, Sig)> = Vec::new();
        for &(k, lo, hi) in &ranges {
            ports.push((k, Sig::C((lo + (hi - lo) * rnd()) as f32)));
        }
        ports.push((STRING, Sig::C(6.0)));
        let out = render(sr, 0.5, &ports);
        let pk = peak(&out);
        assert!(
            out.iter().all(|x| x.is_finite()),
            "trial {trial} not finite"
        );
        assert!(pk < 4.0, "trial {trial} peak {pk}");
    }
}

// --- Release, damping, done -------------------------------------------------

/// A gate fall damps the string (the release ramp, not a cut): it reaches
/// -60 dB well within `rel`'s T60 plus the ramp, and the node reports done
/// soon after; a held note frees itself once it has rung out; `damp`
/// shortens the decay while held.
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
    // No cut: the first 5 ms after the fall keep most of the level.
    let after = peak(&out[off..off + 220]);
    assert!(
        after > 0.5 * held,
        "release cut the note: {after} vs {held}"
    );
    // -60 dB re the held level within 0.3 s.
    let tail = peak(&out[off + (0.3 * sr) as usize..off + (0.35 * sr) as usize]);
    assert!(tail < held * 1e-3, "not damped: {tail} vs {held}");
    assert!(node.is_done(), "a released string must report done");
    assert!(out[(1.0 * sr) as usize..].iter().all(|&x| x == 0.0));

    // Held: a high note rings out and frees itself.
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

    // Palm mute: damp 1 makes the fundamental decay near `rel`.
    let t60 = |damp: f32| {
        let mut p = plain(110.0);
        p.push((DAMP, Sig::C(damp)));
        p.push((ANGLE, Sig::C(0.0)));
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
    // gate falls at 0.3 s.
    let gate = "(1.0 - (line 0.0 1.0 0.3))";
    let at = run(
        &format!("synthdef g = guitarString 220.0 (clip ({gate} * 1000.0) 0.0 1.0) * 0.5"),
        3.0,
    )
    .expect("the voice is freed");
    assert!((0.3..1.5).contains(&at), "freed at {at}");
    // With a body: freed later than the string alone (the modes ring on).
    let body = "modalBank s 100.0 1.0 0.0 1.0 0.0 1.03 0.5 2.0";
    let at_body = run(
        &format!(
            "synthdef g =\n  let s = guitarString 220.0 (clip ({gate} * 1000.0) 0.0 1.0)\n  s * 0.5 + {body}"
        ),
        12.0,
    )
    .expect("the voice with a body is freed");
    assert!(at_body > at + 0.5, "body cut short: {at_body} vs {at}");
    // An excitation burst that reports done does not free the string.
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
/// `hmix` is 0, different otherwise; and a `modalBank` body fed by the
/// string renders exactly the resonator bank applied to the string's own
/// output.
#[test]
fn bridge_tap_and_the_body_stage() {
    let sr = 44100.0;
    let with = |tap: f32, hmix: f32| {
        let mut p = vec![(FREQ, Sig::C(146.8))];
        p.push((TAP, Sig::C(tap)));
        p.push((HMIX, Sig::C(hmix)));
        render(sr, 0.5, &p)
    };
    assert_eq!(with(0.0, 0.0), with(1.0, 0.0));
    assert_ne!(with(0.0, 0.3), with(1.0, 0.3));

    // modalBank after the string, through the DSL, against the same
    // resonator computed here from the string's output.
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
    let string_src = "guitarString 146.8 1.0 0.8 (-1.0) 140.0 20.0 0.5 10.0 0.0 0.0 0.0 2.0 0.3 0.4 0.15 645.0 0.0 1.0";
    let s = render_def(&format!("synthdef a = {string_src}"));
    let body = render_def(&format!(
        "synthdef b = modalBank ({string_src}) 100.0 1.0 0.0 1.0 0.0 1.88 0.7 0.44"
    ));
    // One resonator: f = 188 Hz, gain 0.7, T60 0.44 s at C4 with alpha 0 (so
    // no keytracking), flat input weight.
    let fk = 188.0f64;
    let rho = (-LN_1000 / (0.44 * sr as f64)).exp();
    let om = 2.0 * std::f64::consts::PI * fk / sr as f64;
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
            let mut inputs: Vec<Option<&AudioBuffer>> = vec![None; N_PORTS];
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

// --- The commuted residual (table-bound form) ------------------------------

fn residual_table(ir: &[f32], rate: f32) -> CoeffTable {
    CoeffTable {
        name: "residual".into(),
        entries: vec![PitchEntry {
            f0_hz: 0.0,
            inharmonicity_stretch: 0.0,
            partial_freqs: vec![],
            k_channels: 1,
            j_noise: ir.len() as u32,
            coefficients: ir.to_vec(),
            metadata: vec![("residual_rate_hz".into(), rate)],
        }],
    }
}

/// A bound residual is convolved into the pluck: a unit impulse plays
/// exactly as the unbound node, and `δ + 0.5 δ(t - k)` plays the unbound
/// note plus half of itself `k` samples later (the system is linear).
#[test]
fn bound_residual_is_commuted_into_the_pluck() {
    use std::sync::Arc;
    let sr = 44100.0;
    let n = 20_000;
    let ports = vec![(FREQ, Sig::C(196.0)), (VARY, Sig::C(0.0))];
    let mut unbound = GuitarString::new();
    let plain_out = render_with(&mut unbound, sr, 64, n, &ports, 1);
    let mut unit = GuitarString::with_table(Arc::new(residual_table(&[1.0], sr)));
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
    let mut two = GuitarString::with_table(Arc::new(residual_table(&ir, sr)));
    let two_out = render_with(&mut two, sr, 64, n, &ports, 1);
    let mut worst = 0.0f32;
    for i in 0..n {
        let want = plain_out[i] + if i >= k { 0.5 * plain_out[i - k] } else { 0.0 };
        worst = worst.max((two_out[i] - want).abs());
    }
    assert!(worst < 1e-4, "two-tap residual differs by {worst}");
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
    let id = bank.register(residual_table(&ir_resp, 48_000.0));
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

/// The built-in rule picks the highest string at or below the note: G3 on
/// the G string (open), A3 on the G string (fret 2), B3 on the B string; an
/// explicit `string` overrides it, which changes the timbre (another `B` and
/// loss).
#[test]
fn string_rule_and_override() {
    let sr = 44100.0;
    let note = |freq: f32, string: f32| {
        let mut p = plain(freq);
        p.push((STRING, Sig::C(string)));
        render(sr, 0.5, &p)
    };
    // A3 by rule = A3 on string 3 explicitly; and differs from string 4.
    assert_eq!(note(220.0, -1.0), note(220.0, 3.0));
    assert_ne!(note(220.0, -1.0), note(220.0, 4.0));
    // B3 by rule = string 2.
    assert_eq!(note(246.941_65, -1.0), note(246.941_65, 2.0));
    // A note below E2 plays on the lowest string.
    assert_eq!(note(73.42, -1.0), note(73.42, 6.0));
}
