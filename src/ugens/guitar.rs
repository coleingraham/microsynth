//! `guitarString`: one plucked steel string of a six-string guitar, as a
//! digital waveguide with two polarisations.
//!
//! # Model
//!
//! Each polarisation is a single delay loop (one round trip of the string):
//!
//! ```text
//! x[n] = y[n] + u[n]                     (u: the pluck, plus `exc`)
//! y    = Disp · Damp · Loss · Thiran · z^(-N) x
//! ```
//!
//! `x` is the wave arriving at and leaving the bridge, read as the force the
//! string puts on the bridge. `N` is a whole number of samples, `Thiran` a
//! third-order Thiran all-pass for the fractional part of the period, `Loss`
//! a one-pole loss filter, `Damp` the release/palm-mute filter and `Disp` a
//! cascade of all-pass sections for the string's stiffness. The vertical
//! polarisation (`V`, normal to the top) and the horizontal one (`H`) are two
//! independent loops: `H` is tuned `split` cents flat of `V` and loses less to
//! the bridge (`hv` times `V`'s bridge loss). There is no cross-coupling.
//!
//! # Ports
//!
//! All optional, read from input channel 0; the output is mono, copied to
//! every output channel.
//!
//! | port | meaning | default |
//! |---|---|---|
//! | `freq` | sounding pitch in Hz (the first partial), per sample | 440 |
//! | `gate` | a rise plucks, a fall damps (release) | 1 |
//! | `vel` | velocity 0..1, at the pluck | 0.8 |
//! | `string` | 1..6 (1 = high E), or -1: the built-in rule (below); at the pluck | -1 |
//! | `pos` | pluck point, mm from the saddle; at the pluck | 140 |
//! | `angle` | pluck angle from the normal, degrees; at the pluck | 20 |
//! | `hard` | 0 flesh, 0.5 nail, 1 plectrum; at the pluck | 0.5 |
//! | `width` | contact width, mm; at the pluck | 10 |
//! | `damp` | extra damping 0..1 (palm mute, a hand on the string), per sample | 0 |
//! | `exc` | external excitation, injected at the pluck point, per sample | 0 |
//! | `seed` | per-note variation seed, at the pluck | 0 |
//! | `split` | V–H detune, cents; at the pluck | 2 |
//! | `hmix` | share of `H` in `out` | 0.3 |
//! | `hv` | `H`'s bridge loss as a fraction of `V`'s; at the pluck | 0.4 |
//! | `rel` | release decay `T_rel`, seconds (T60 of the fundamental once damped) | 0.15 |
//! | `scale` | scale length, mm; at the pluck | 645 |
//! | `vary` | depth of the seeded per-note variation, 0..1; at the pluck | 1 |
//! | `tap` | which signal is output: 0 `out`, 1 `bridge` | 0 |
//! | `oS bS tS hS rS` | per string `S` = 1..6 (below); at the pluck | standard tuning, light steel set |
//!
//! # Per-string constants (fittable)
//!
//! Every string is described by measurable quantities, not filter
//! coefficients:
//!
//! - `oS`: open pitch, Hz (default E4 B3 G3 D3 A2 E2).
//! - `bS`: inharmonicity coefficient `B` of the open string
//!   (`f_n = n f0 sqrt(1 + B n^2)`).
//! - `tS`: T60 of the open string's fundamental, seconds.
//! - `hS`: T60 of a partial at 2 kHz on the open string, seconds.
//! - `rS`: the bridge's share (0..1) of the open fundamental's decay rate.
//!
//! # From the constants to the loop (the documented mapping)
//!
//! At the pluck, with `freq = f` on string `S` (open pitch `o`):
//!
//! - **Fret.** `k = f / o` (the length ratio, `k < 1` plays the string tuned
//!   down). Speaking length `L_f = scale / k`; pluck point `β = pos / L_f`.
//! - **Stiffness.** `B = bS · k²` (`B ∝ 1/T L²` at fixed tension). The ideal
//!   string's fundamental is `f0 = f / sqrt(1 + B)`, so the first partial
//!   sounds at `freq` exactly, and the loop is tuned so partial `n` lies at
//!   `n f0 sqrt(1 + B n²)`.
//! - **Decay rates** (`σ = ln 1000 / T60`, per second) are the sum of a
//!   bridge term and an intrinsic term. On the open string `σ(o) = σ_t` and
//!   `σ(2 kHz) = σ_h` (from `tS`, `hS`). The bridge term is frequency
//!   independent and proportional to the round-trip rate:
//!   `σ_br = rS · σ_t · k` (`V`), `hv · σ_br` (`H`). The intrinsic term is
//!   linear in frequency through the two open-string anchors:
//!   `σ_int(o) = (1 − rS) σ_t`, `σ_int(2 kHz) = σ_h − rS σ_t` (floored at 0).
//!   So a fretted note loses more to the bridge, and its intrinsic loss is
//!   the open string's at the same absolute frequency.
//! - **Loss filter.** `Loss(z) = g (1 + a) / (1 + a z^-1)` is solved in
//!   closed form so that its per-round-trip attenuation matches `σ` exactly at
//!   two anchors: the fundamental `f` and `f_B = max(2 kHz, 2.5 f)` (below
//!   0.4 fs). A partial's decay rate is `σ_n = −ln|Loss(ω_n)| / τ_n`, where
//!   `τ_n = dn/df` at `f_n` is the loop's round-trip time there (the stiff
//!   string's group delay); between and beyond the anchors the decay follows
//!   the one-pole's shape. `|Loss| ≤ g < 1` at every frequency, so the loop is
//!   stable for any value of any port (all other loop filters are all-passes).
//! - **Dispersion.** The all-pass cascade is designed at the pluck from
//!   `B`, `f0` and the loss filter's own phase: either up to eight
//!   second-order sections placed on bands of 2π of the target phase (the
//!   band-splitting design of Abel, Välimäki and Smith; pole radius
//!   `exp(-0.9 · band width)`), or one to four equal
//!   first-order sections (coefficient by golden-section search), whichever
//!   puts partials 2..60 below 10 kHz closer to `n f0 sqrt(1 + B n²)`; none
//!   when `B` is 0 or neither helps. Accuracy as built (partials below
//!   5 kHz): within 4 cents, or 10 % of the partial's own stretch where that
//!   is larger (16.5 cents of a 225-cent stretch at 4.9 kHz on the low E at
//!   fret 12; 3.9 cents of a 21-cent stretch at 5 kHz on the open B).
//!   [`gs_designed_partials`] returns the design's own prediction.
//! - **Tuning.** The integer delay and the Thiran fraction are solved so the
//!   loop's total phase at `freq` is exactly 2π, including the loss, damping
//!   and dispersion filters' phase there (one Newton step corrects the Thiran
//!   filter's own phase error). This is re-solved whenever `freq` or the
//!   damping changes; the integer part changes only when the Thiran delay
//!   leaves 2.2..3.8 samples.
//!
//! # The pluck
//!
//! The pluck injects one period of the bridge-force waveform of a string
//! released from a triangular shape at `β`: `1 − β` for `β P` samples, then
//! `−β` for the rest of the period `P` (zero mean, so no DC; the position
//! comb `sin(n π β)` and the `1/n` slope of a pluck are both in it). Its two
//! edges are raised-cosine ramps lasting the contact time `w_eff / c`
//! (`c = 2 · scale · o`, the wave speed; at least one sample), with
//! `w_eff = width · (1.25 − 0.5 vel) · (1 − 0.5 hard)`. The waveform is
//! built from the sampled step response `Q[t]` (the ramp, convolved with the
//! residual when one is bound), read at fractional times by linear
//! interpolation from 0 at `t = -1`. The waveform then
//! passes a one-pole "dynamic level" lowpass (Jaffe and Smith) at
//! `500 Hz · 2^(6 b)`, `b = vel · (0.5 + 0.5 hard)`. So velocity changes the
//! timbre (a narrower contact and an opening lowpass) as well as the level
//! (`vel`, linear). `V` takes `cos(angle)` of it, `H` `sin(angle)`.
//!
//! **Per-note variation.** Each pluck draws, from the voice's seed (the
//! engine reseeds every spawned voice), the `seed` port and a pluck counter:
//! pluck point ±10 mm, angle ±10°, level ±1 dB and split ±0.5 cents, all
//! scaled by `vary`. Repeated notes are not identical; a render is
//! reproducible.
//!
//! **`exc`** is injected into both loops (by `cos`/`sin` of the angle)
//! through the same position comb, `exc[n] − exc[n − β P]`, and is not
//! shaped by the contact or the dynamic lowpass.
//!
//! # The commuted body residual (table-bound form)
//!
//! Bound to a coefficient table (the IR's `table_bindings`, as
//! `partialsNoise` is), the node reads an impulse response from the first
//! entry whose metadata carries `residual_rate_hz`: that entry's channel-0
//! coefficient row (`k_channels` 1, no partials, `j_noise` = the length) is
//! the response, sampled at `residual_rate_hz` (resampled linearly to the
//! engine rate when the node is built; at most [`GS_MAX_RESIDUAL_S`]
//! seconds). The pluck waveform is convolved with it (commuted synthesis:
//! body ∗ string ∗ pluck = string ∗ (body ∗ pluck)), so the residual colours
//! both `out` and `bridge`. Unbound, the response is a unit impulse.
//!
//! # Outputs
//!
//! The engine gives a node one output, so `tap` selects it: 0 = `out`
//! (`0.5 · (x_V + hmix · x_H)`, the bridge-force mix that drives the top),
//! 1 = `bridge` (`0.5 · x_V`, the vertical bridge force alone, the coupling
//! signal for a body mode or a sympathetic string). With `hmix` 0 the two are
//! identical. The low body modes go after the string, e.g. a `modalBank`
//! with `freq` held at 100 and each mode's ratio `f_k / 100`.
//!
//! # String assignment
//!
//! `string` -1: the highest string whose open pitch is at or below the note
//! (within a quarter tone); a note below every open string plays on the
//! lowest string tuned down. The chosen string is latched per pluck.
//!
//! # Release and damping
//!
//! A gate fall (or `damp` > 0) ramps an extra loss filter in with a 10 ms
//! time constant: per round trip `g_d = exp(−d · ln 1000 / (f · rel))`, pole
//! `−0.5 d` (the highs die first), `d = max(damp, released)`. A rise during
//! the note re-plucks: the new pluck is added to the ringing string.
//!
//! # Pitch changes
//!
//! `freq` is read every sample: a bend or vibrato re-solves the delay (the
//! loss and dispersion filters stay as designed at the pluck, which ignores
//! the small change of `B` and of the per-second loss with tension). The
//! Thiran filter is recursive, so a fast change of its coefficient leaves a
//! small transient; slides and fast glides are not modelled.
//!
//! # Done
//!
//! Once plucked (or excited through `exc`), when no excitation is still
//! being injected and a follower of `|x_V| + |x_H|` (20 ms) falls below
//! 1e-5 (about −100 dBFS), the loops are cleared and the node reports done;
//! its output is silent from then on. A held note frees itself once it has
//! rung out, like a real pluck. `freq` ≤ 0 (or not finite) silences the
//! string at once and reports done.
//!
//! # Hygiene and cost
//!
//! Non-finite inputs read as their defaults (`freq` as 0, `exc` as 0); every
//! port is clamped to a physical range (`exc` to ±10). State is f64 except the delay lines
//! (f32). Per voice: two delay lines and an `exc` history of
//! `fs / 20 + 8` samples each (f32, about 26.6 KB at 44.1 kHz), the
//! residual's step response (f32), and under 1 KB of filter state. Nothing
//! allocates while rendering. Per sample: about 120 multiplies with eight
//! dispersion sections on both loops, plus the delay re-solve (a few
//! dozen trigonometric calls) on samples where `freq` or the damping moves.

use crate::buffer::{AudioBuffer, MAX_BLOCK_SIZE, read_input};
use crate::coeff_table::CoeffTable;
use crate::context::{ProcessContext, Rate};
use crate::dsl::compiler::{TableUGenFactory, UGenRegistry};
use crate::node::{InputSpec, OutputSpec, UGen, UGenCategory};
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;
/// `ln 1000`: a decay of rate `σ` falls 60 dB in `ln 1000 / σ` seconds.
const LN_1000: f64 = 6.907_755_278_982_137;

/// Lowest pitch (sizes the delay lines).
pub const GS_MIN_FREQ: f32 = 20.0;
/// Highest pitch, as a fraction of the sample rate.
const MAX_FREQ_FRAC: f64 = 1.0 / 8.0;
/// Most second-order dispersion sections.
pub const GS_MAX_SECTIONS: usize = 8;
/// Most first-order dispersion sections.
const MAX_FIRST: usize = 4;
/// Band-width factor of the dispersion sections' pole radius. Searching it
/// per note on the partial cost was tried and rejected: it trades the low
/// partials' accuracy for the high ones' (5.5 cents at partial 2 on the low
/// E at fret 12).
const DISP_ETA: f64 = 0.9;
/// The upper loss anchor on the open string, Hz.
pub const GS_F_HI: f64 = 2000.0;
/// Longest commuted residual, seconds.
pub const GS_MAX_RESIDUAL_S: f64 = 0.2;
/// Longest contact ramp, samples.
const MAX_RAMP: usize = 64;
/// Thiran delay window before the integer delay is moved.
const THIRAN_LO: f64 = 2.2;
const THIRAN_HI: f64 = 3.8;
/// Done threshold on the follower.
const DONE_LEVEL: f64 = 1e-5;
/// Follower time constant, seconds.
const FOLLOW_S: f64 = 0.02;
/// Damping ramp time constant, seconds.
const DAMP_S: f64 = 0.01;
/// `exc` is clamped to ±this.
const EXC_MAX: f32 = 10.0;
/// Output gain.
const OUT_GAIN: f64 = 0.5;

/// Number of ports before the per-string ports.
pub const GS_FIXED_PORTS: usize = 18;
/// Fields per string: `o b t h r`.
pub const GS_STRING_FIELDS: usize = 5;

/// Port index of string `s` (1..=6)'s field `field` (0 `o`, 1 `b`, 2 `t`,
/// 3 `h`, 4 `r`).
pub const fn gs_string_port(s: usize, field: usize) -> usize {
    GS_FIXED_PORTS + GS_STRING_FIELDS * (s - 1) + field
}

/// Default per-string constants, string 1 (high E) to 6 (low E): open pitch
/// (Hz), `B`, T60 at the fundamental (s), T60 at 2 kHz (s), bridge share.
/// Open pitches are standard tuning; `B` is computed for a light 12–53 steel
/// set on a 645 mm scale (wound strings from estimated core diameters); the
/// decays are provisional estimates, to be replaced by fitted values.
pub const GS_DEFAULT_STRINGS: [[f32; GS_STRING_FIELDS]; 6] = [
    [329.627_56, 1.9e-5, 4.0, 2.2, 0.6],
    [246.941_65, 6.1e-5, 4.5, 2.0, 0.6],
    [195.997_72, 1.1e-5, 5.0, 1.6, 0.65],
    [146.832_38, 2.1e-5, 5.5, 1.4, 0.7],
    [110.0, 4.9e-5, 6.0, 1.2, 0.75],
    [82.406_89, 1.1e-4, 6.5, 1.0, 0.8],
];

// Port indices.
const P_FREQ: usize = 0;
const P_GATE: usize = 1;
const P_VEL: usize = 2;
const P_STRING: usize = 3;
const P_POS: usize = 4;
const P_ANGLE: usize = 5;
const P_HARD: usize = 6;
const P_WIDTH: usize = 7;
const P_DAMP: usize = 8;
const P_EXC: usize = 9;
const P_SEED: usize = 10;
const P_SPLIT: usize = 11;
const P_HMIX: usize = 12;
const P_HV: usize = 13;
const P_REL: usize = 14;
const P_SCALE: usize = 15;
const P_VARY: usize = 16;
const P_TAP: usize = 17;

/// Every port name, in order (the fixed ports, then `o1 b1 t1 h1 r1` …).
const PORT_NAMES: [&str; GS_FIXED_PORTS + 6 * GS_STRING_FIELDS] = [
    "freq", "gate", "vel", "string", "pos", "angle", "hard", "width", "damp", "exc", "seed",
    "split", "hmix", "hv", "rel", "scale", "vary", "tap", "o1", "b1", "t1", "h1", "r1", "o2", "b2",
    "t2", "h2", "r2", "o3", "b3", "t3", "h3", "r3", "o4", "b4", "t4", "h4", "r4", "o5", "b5", "t5",
    "h5", "r5", "o6", "b6", "t6", "h6", "r6",
];

/// The port specs (all optional, audio rate).
static GS_INPUTS: [InputSpec; GS_FIXED_PORTS + 6 * GS_STRING_FIELDS] = {
    const BLANK: InputSpec = InputSpec {
        name: "",
        rate: Rate::Audio,
        required: false,
    };
    let mut a = [BLANK; GS_FIXED_PORTS + 6 * GS_STRING_FIELDS];
    let mut k = 0;
    while k < a.len() {
        a[k].name = PORT_NAMES[k];
        k += 1;
    }
    a
};

/// Default value of port `k`.
fn port_default(k: usize) -> f32 {
    match k {
        P_FREQ => 440.0,
        P_GATE => 1.0,
        P_VEL => 0.8,
        P_STRING => -1.0,
        P_POS => 140.0,
        P_ANGLE => 20.0,
        P_HARD => 0.5,
        P_WIDTH => 10.0,
        P_DAMP | P_EXC | P_SEED | P_TAP => 0.0,
        P_SPLIT => 2.0,
        P_HMIX => 0.3,
        P_HV => 0.4,
        P_REL => 0.15,
        P_SCALE => 645.0,
        P_VARY => 1.0,
        _ => {
            let i = k - GS_FIXED_PORTS;
            GS_DEFAULT_STRINGS[i / GS_STRING_FIELDS][i % GS_STRING_FIELDS]
        }
    }
}

#[inline]
fn finite_or(v: f32, d: f32) -> f32 {
    if v.is_finite() { v } else { d }
}

// ---------------------------------------------------------------------------
// Design helpers (f64, closed form; no allocation).
// ---------------------------------------------------------------------------

/// Partial number of a stiff string at frequency `f` (continuous):
/// `f = n f0 sqrt(1 + B n²)` solved for `n`.
fn stiff_n(f: f64, f0: f64, b: f64) -> f64 {
    let x = f / f0;
    if b <= 0.0 {
        return x;
    }
    let s = (1.0 + 4.0 * b * x * x).sqrt();
    (2.0 * x * x / (s + 1.0)).sqrt()
}

/// `dn/df` of [`stiff_n`]: the round-trip time (s) of the loop at `f`.
fn stiff_dn_df(f: f64, f0: f64, b: f64) -> f64 {
    if b <= 0.0 || f <= 0.0 {
        return 1.0 / f0;
    }
    let x = f / f0;
    let s = (1.0 + 4.0 * b * x * x).sqrt();
    let n = stiff_n(f, f0, b);
    x / (n * s * f0)
}

/// Phase lag of `1 / (1 + a z^-1)` at `w`.
fn lp_lag(a: f64, w: f64) -> f64 {
    (-a * w.sin()).atan2(1.0 + a * w.cos())
}

/// Group delay (samples) of `1 / (1 + a z^-1)` at `w`.
fn lp_gd(a: f64, w: f64) -> f64 {
    let c = w.cos();
    (-a * c - a * a) / (1.0 + 2.0 * a * c + a * a)
}

/// Phase lag of the first-order all-pass `(a + z^-1) / (1 + a z^-1)`.
fn ap1_lag(a: f64, w: f64) -> f64 {
    w - 2.0 * (a * w.sin()).atan2(1.0 + a * w.cos())
}

/// Phase lag of the second-order all-pass with poles `ρ e^{±jθ}`.
fn ap2_lag(rho: f64, th: f64, w: f64) -> f64 {
    let mut lag = 0.0;
    for s in [1.0, -1.0] {
        let d = w - s * th;
        lag += w + 2.0 * (rho * d.sin()).atan2(1.0 - rho * d.cos());
    }
    lag
}

/// Bisect `f(w) = target` on `[lo, hi]` (`f(lo) < target ≤ f(hi)` assumed;
/// returns the bracket's end nearer the crossing otherwise).
fn bisect(mut lo: f64, mut hi: f64, target: f64, f: impl Fn(f64) -> f64) -> f64 {
    for _ in 0..56 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// A dispersion all-pass cascade.
#[derive(Clone, Copy, Debug, PartialEq)]
enum DispKind {
    None,
    /// `n` second-order sections, poles `rho[k] e^{±j th[k]}`.
    Second,
    /// `n` equal first-order sections with coefficient `a1`.
    First,
}

#[derive(Clone, Copy, Debug)]
struct DispDesign {
    kind: DispKind,
    n: usize,
    rho: [f64; GS_MAX_SECTIONS],
    th: [f64; GS_MAX_SECTIONS],
    a1: f64,
}

impl DispDesign {
    const NONE: DispDesign = DispDesign {
        kind: DispKind::None,
        n: 0,
        rho: [0.0; GS_MAX_SECTIONS],
        th: [0.0; GS_MAX_SECTIONS],
        a1: 0.0,
    };

    fn lag(&self, w: f64) -> f64 {
        match self.kind {
            DispKind::None => 0.0,
            DispKind::Second => (0..self.n)
                .map(|k| ap2_lag(self.rho[k], self.th[k], w))
                .sum(),
            DispKind::First => self.n as f64 * ap1_lag(self.a1, w),
        }
    }
}

/// Weighted mean squared relative phase error (in cents²) of the loop's
/// partials 2..=60 below `min(10 kHz, 0.45 fs)` against the stiff string,
/// with the fundamental tuned exactly; `lag` is the loop filters' phase lag.
fn disp_cost(f0: f64, b: f64, sr: f64, lag: &impl Fn(f64) -> f64) -> f64 {
    let w1 = TWO_PI * f0 * (1.0 + b).sqrt() / sr;
    let d = (TWO_PI - lag(w1)) / w1;
    let f_top = (10_000.0f64).min(0.45 * sr);
    let (mut j, mut wsum) = (0.0, 0.0);
    for n in 2..=60u32 {
        let nf = n as f64;
        let fnn = nf * f0 * (1.0 + b * nf * nf).sqrt();
        if fnn > f_top {
            break;
        }
        let wn = TWO_PI * fnn / sr;
        let r =
            (d * wn + lag(wn) - TWO_PI * nf) / (TWO_PI * nf) * (1200.0 / core::f64::consts::LN_2);
        j += r * r / nf;
        wsum += 1.0 / nf;
    }
    if wsum > 0.0 { j / wsum } else { 0.0 }
}

/// Second-order band-splitting design (see the module doc).
fn design_second(f0: f64, b: f64, alp: f64, sr: f64, eta: f64) -> Option<DispDesign> {
    let g = |w: f64| TWO_PI * stiff_n(w * sr / TWO_PI, f0, b) - lp_lag(alp, w);
    let gd = |w: f64| sr * stiff_dn_df(w * sr / TWO_PI, f0, b) - lp_gd(alp, w);
    let excess = |w: f64| g(w) - gd(w) * w;
    let w_lim = TWO_PI * (10_000.0f64).min(0.45 * sr) / sr;
    let w_top = 0.96 * PI;
    let k_lim = (excess(w_lim) / TWO_PI - 1e-9).ceil();
    let k_top = (excess(w_top) / TWO_PI).floor();
    let k = k_lim.min(k_top).min(GS_MAX_SECTIONS as f64);
    if k.is_nan() || k < 1.0 {
        return None;
    }
    let target = TWO_PI * k;
    let w_max = bisect(1e-6, w_top, target, excess);
    let dd = gd(w_max);
    let psi = |w: f64| g(w) - dd * w;
    let mut out = DispDesign::NONE;
    out.kind = DispKind::Second;
    out.n = k as usize;
    let mut prev_edge = 0.0;
    for i in 0..out.n {
        let edge = bisect(0.0, w_max, TWO_PI * (i as f64 + 1.0), psi);
        let mid = bisect(0.0, w_max, TWO_PI * (i as f64 + 0.5), psi);
        out.rho[i] = (-eta * (edge - prev_edge).max(1e-6)).exp().min(0.999_999);
        out.th[i] = mid;
        prev_edge = edge;
    }
    Some(out)
}

/// Equal first-order sections, `m` of them, coefficient by golden section.
fn design_first(f0: f64, b: f64, alp: f64, sr: f64, m: usize) -> (DispDesign, f64) {
    let cost = |a: f64| {
        disp_cost(f0, b, sr, &|w: f64| {
            m as f64 * ap1_lag(a, w) + lp_lag(alp, w)
        })
    };
    let gr = 0.618_033_988_749_894_9;
    let (mut lo, mut hi) = (-0.95f64, 0.0f64);
    let mut c = hi - gr * (hi - lo);
    let mut d = lo + gr * (hi - lo);
    let (mut fc, mut fd) = (cost(c), cost(d));
    for _ in 0..32 {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - gr * (hi - lo);
            fc = cost(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + gr * (hi - lo);
            fd = cost(d);
        }
    }
    let a = 0.5 * (lo + hi);
    let mut out = DispDesign::NONE;
    out.kind = DispKind::First;
    out.n = m;
    out.a1 = a;
    (out, cost(a))
}

/// Choose the dispersion design for a note (see the module doc). Candidates
/// whose delay at the fundamental would leave less than `max_lag_samples`
/// for the delay line are skipped.
fn design_dispersion(f0: f64, b: f64, alp: f64, sr: f64, max_delay: f64) -> DispDesign {
    if b.is_nan() || b <= 1e-9 {
        return DispDesign::NONE;
    }
    let w1 = TWO_PI * f0 * (1.0 + b).sqrt() / sr;
    let fits = |d: &DispDesign| d.lag(w1) / w1 <= max_delay;
    let mut best = DispDesign::NONE;
    let mut best_j = disp_cost(f0, b, sr, &|w: f64| lp_lag(alp, w));
    if let Some(d) = design_second(f0, b, alp, sr, DISP_ETA)
        && fits(&d)
    {
        let j = disp_cost(f0, b, sr, &|w: f64| d.lag(w) + lp_lag(alp, w));
        if j < best_j {
            best = d;
            best_j = j;
        }
    }
    for m in 1..=MAX_FIRST {
        let (d, j) = design_first(f0, b, alp, sr, m);
        if j < best_j && fits(&d) {
            best = d;
            best_j = j;
        }
    }
    best
}

/// The partial frequencies (Hz) a note's loop is designed to produce: the
/// dispersion cascade [`GuitarString`] designs for a note at `freq` with
/// inharmonicity `b` and loss-filter pole `loss_a`, and partials `1..=nmax`
/// solved from the loop's phase (the fundamental tuned exactly; the Thiran
/// filter taken as an exact fractional delay, which it is within 0.001
/// sample below 5 kHz). This is the design's own prediction, for tests and
/// for a fitting tool that wants the realised stretch rather than the ideal
/// `n f0 sqrt(1 + B n²)`.
pub fn gs_designed_partials(sr: f64, freq: f64, b: f64, loss_a: f64, nmax: usize) -> Vec<f64> {
    let f0 = freq / (1.0 + b).sqrt();
    let disp = note_dispersion(sr, freq, b, loss_a);
    let lag = |w: f64| disp.lag(w) + lp_lag(loss_a, w);
    let w1 = TWO_PI * freq / sr;
    let d = (TWO_PI - lag(w1)) / w1;
    let mut out = Vec::with_capacity(nmax);
    for n in 1..=nmax {
        let nf = n as f64;
        let target = nf * f0 * (1.0 + b * nf * nf).sqrt();
        let wt = TWO_PI * target / sr;
        let hi = (1.5 * wt).min(PI);
        let w = bisect(0.5 * wt, hi, TWO_PI * nf, |w| d * w + lag(w));
        out.push(w * sr / TWO_PI);
    }
    out
}

/// The dispersion design for a note at `freq` (the first partial) with
/// inharmonicity `b` and loss pole `loss_a`, leaving the delay line at least
/// one sample and the Thiran filter its minimum delay.
fn note_dispersion(sr: f64, freq: f64, b: f64, loss_a: f64) -> DispDesign {
    let f0 = freq / (1.0 + b).sqrt();
    let w1 = TWO_PI * freq / sr;
    let room = sr / freq - 1.0 - THIRAN_LO - 1.0 - lp_lag(loss_a, w1) / w1;
    design_dispersion(f0, b, loss_a, sr, room.max(0.0))
}

/// One-pole loss filter `g (1 + a) / (1 + a z^-1)` matching per-round-trip
/// attenuations `ma` at `wa` and `mb` at `wb` (`0 < m < 1`). Returns `(g, a)`.
fn design_loss(ma: f64, wa: f64, mb: f64, wb: f64) -> (f64, f64) {
    let ma = ma.clamp(1e-6, 0.999_999_9);
    let mut a = 0.0;
    if wb > wa * 1.05 && mb < ma {
        let r2 = (mb / ma) * (mb / ma);
        let (ca, cb) = (wa.cos(), wb.cos());
        let p = (ca - r2 * cb) / (1.0 - r2);
        a = if p > 1.0 {
            -p + (p * p - 1.0).sqrt()
        } else {
            -0.95
        };
        a = a.clamp(-0.95, 0.0);
    }
    let g = ma * (1.0 + 2.0 * a * wa.cos() + a * a).sqrt() / (1.0 + a);
    (g.clamp(0.0, 0.999_999_9), a)
}

/// Thiran third-order coefficients `(a1, a2, a3)` for delay `d`.
fn thiran3(d: f64) -> [f64; 3] {
    let n = 3.0;
    let binom = [3.0, 3.0, 1.0];
    let mut a = [0.0; 3];
    for k in 1..=3 {
        let kf = k as f64;
        let mut prod = 1.0;
        for i in 0..=3 {
            let i = i as f64;
            prod *= (d - n + i) / (d - n + kf + i);
        }
        let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
        a[k - 1] = sign * binom[k - 1] * prod;
    }
    a
}

/// Phase lag of the Thiran all-pass with coefficients `a` at `w`.
fn thiran_lag(a: &[f64; 3], w: f64) -> f64 {
    // D(w) = 1 + a1 e^{-jw} + a2 e^{-2jw} + a3 e^{-3jw}; lag = 3w + 2 arg D.
    let re = 1.0 + a[0] * w.cos() + a[1] * (2.0 * w).cos() + a[2] * (3.0 * w).cos();
    let im = -(a[0] * w.sin() + a[1] * (2.0 * w).sin() + a[2] * (3.0 * w).sin());
    3.0 * w + 2.0 * im.atan2(re)
}

/// A small deterministic hash stream (splitmix64).
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    fn bipolar(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

// ---------------------------------------------------------------------------
// One polarisation.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Biquad {
    a1: f64,
    a2: f64,
    s1: f64,
    s2: f64,
}

/// One polarisation's loop. The delay line lives in [`GuitarString::lines`].
#[derive(Clone, Copy)]
struct Pol {
    /// Pitch of this polarisation (its first partial), Hz.
    freq: f64,
    n_int: usize,
    thiran: [f64; 3],
    th_s: [f64; 3],
    /// Loss filter `g (1 + a) / (1 + a z^-1)`.
    loss_b: f64,
    loss_a: f64,
    loss_s: f64,
    /// Damping filter, same form.
    damp_b: f64,
    damp_a: f64,
    damp_s: f64,
    disp: [Biquad; GS_MAX_SECTIONS],
    first_s: [f64; MAX_FIRST],
    /// Excitation: amplitude, period (samples), comb lag `m`, pulse-high
    /// share `1 − β'`, dynamic-lowpass coefficient and state.
    amp: f64,
    period: f64,
    m: usize,
    beta: f64,
    dyn_r: f64,
    dyn_s: f64,
}

impl Default for Pol {
    fn default() -> Self {
        Pol {
            freq: 0.0,
            n_int: 1,
            thiran: thiran3(3.0),
            th_s: [0.0; 3],
            loss_b: 0.0,
            loss_a: 0.0,
            loss_s: 0.0,
            damp_b: 1.0,
            damp_a: 0.0,
            damp_s: 0.0,
            disp: [Biquad::default(); GS_MAX_SECTIONS],
            first_s: [0.0; MAX_FIRST],
            amp: 0.0,
            period: 1.0,
            m: 1,
            beta: 0.5,
            dyn_r: 0.0,
            dyn_s: 0.0,
        }
    }
}

impl Pol {
    fn clear_state(&mut self) {
        self.th_s = [0.0; 3];
        self.loss_s = 0.0;
        self.damp_s = 0.0;
        for q in &mut self.disp {
            q.s1 = 0.0;
            q.s2 = 0.0;
        }
        self.first_s = [0.0; MAX_FIRST];
        self.dyn_s = 0.0;
    }

    /// Run the loop filters on the delay line's output.
    #[inline]
    fn filters(&mut self, x: f64, design: &DispDesign) -> f64 {
        // Thiran (transposed direct form II).
        let [a1, a2, a3] = self.thiran;
        let y = a3 * x + self.th_s[0];
        self.th_s[0] = a2 * x - a1 * y + self.th_s[1];
        self.th_s[1] = a1 * x - a2 * y + self.th_s[2];
        self.th_s[2] = x - a3 * y;
        // Loss and damping one-poles.
        let y = self.loss_b * y - self.loss_a * self.loss_s;
        self.loss_s = y;
        let y = self.damp_b * y - self.damp_a * self.damp_s;
        self.damp_s = y;
        // Dispersion.
        let mut y = y;
        match design.kind {
            DispKind::None => {}
            DispKind::Second => {
                for q in &mut self.disp[..design.n] {
                    // (a2 + a1 z^-1 + z^-2) / (1 + a1 z^-1 + a2 z^-2)
                    let out = q.a2 * y + q.s1;
                    q.s1 = q.a1 * y - q.a1 * out + q.s2;
                    q.s2 = y - q.a2 * out;
                    y = out;
                }
            }
            DispKind::First => {
                let a = design.a1;
                for s in &mut self.first_s[..design.n] {
                    // (a + z^-1) / (1 + a z^-1)
                    let out = a * y + *s;
                    *s = y - a * out;
                    y = out;
                }
            }
        }
        y
    }

    /// The excitation's step response at (possibly fractional) time `t`.
    #[inline]
    fn q_at(q: &[f32], q_end: f64, t: f64) -> f64 {
        if t <= -1.0 {
            return 0.0;
        }
        if t < 0.0 {
            // Between the zero before the response and its first sample.
            return (t + 1.0) * q.first().copied().unwrap_or(0.0) as f64;
        }
        let i = t as usize;
        if i + 1 >= q.len() {
            return if i < q.len() {
                let fr = t - i as f64;
                q[i] as f64 * (1.0 - fr) + q_end * fr
            } else {
                q_end
            };
        }
        let fr = t - i as f64;
        q[i] as f64 * (1.0 - fr) + q[i + 1] as f64 * fr
    }

    /// The pluck waveform at sample `t` after the pluck (before the dynamic
    /// lowpass).
    #[inline]
    fn pluck_sample(&self, q: &[f32], q_end: f64, t: f64) -> f64 {
        self.amp
            * ((1.0 - self.beta) * Self::q_at(q, q_end, t)
                - Self::q_at(q, q_end, t - self.m as f64)
                + self.beta * Self::q_at(q, q_end, t - self.period))
    }
}

// ---------------------------------------------------------------------------
// The node.
// ---------------------------------------------------------------------------

/// The guitar string node. See the module doc.
pub struct GuitarString {
    sample_rate: f64,
    /// Delay lines (V, H) and the `exc` history, `len` samples each.
    lines: [Vec<f32>; 2],
    exc_hist: Vec<f32>,
    len: usize,
    write: usize,
    pols: [Pol; 2],
    disp: DispDesign,
    /// The residual (engine rate), unit impulse when unbound.
    residual: Vec<f32>,
    /// The bound residual as read from the table: (rate, samples).
    residual_source: Option<(f64, Vec<f32>)>,
    /// `exc` weights (V, H): cosine and sine of the angle.
    exc_w: [f64; 2],
    /// Step response of contact ∗ residual, and its final value.
    q: Vec<f32>,
    q_end: f64,
    /// Samples since the pluck; the excitation runs while `t < exc_len`.
    t: f64,
    exc_len: f64,
    exciting: bool,
    /// Latched per pluck.
    string: usize,
    rel: f64,
    hmix: f64,
    /// Damping: current, target, ramp coefficient; release flag.
    d: f64,
    d_coef: f64,
    released: bool,
    follow: f64,
    follow_k: f64,
    configured: bool,
    plucked: bool,
    done: bool,
    prev_gate: f32,
    last_freq: f64,
    voice_seed: u32,
    plucks: u64,
}

impl Default for GuitarString {
    fn default() -> Self {
        Self::new()
    }
}

impl GuitarString {
    /// An unbound node: the residual is a unit impulse.
    pub fn new() -> Self {
        GuitarString {
            sample_rate: 44100.0,
            lines: [Vec::new(), Vec::new()],
            exc_hist: Vec::new(),
            len: 0,
            write: 0,
            pols: [Pol::default(); 2],
            disp: DispDesign::NONE,
            residual: alloc::vec![1.0],
            residual_source: None,
            exc_w: [1.0, 0.0],
            q: Vec::new(),
            q_end: 1.0,
            t: 0.0,
            exc_len: 0.0,
            exciting: false,
            string: 0,
            rel: 0.15,
            hmix: 0.3,
            d: 0.0,
            d_coef: 0.0,
            released: false,
            follow: 0.0,
            follow_k: 0.0,
            configured: false,
            plucked: false,
            done: false,
            prev_gate: 0.0,
            last_freq: f64::NAN,
            voice_seed: 0,
            plucks: 0,
        }
    }

    /// A node bound to `table`: the residual is read from the first entry
    /// carrying `residual_rate_hz` (see the module doc); without one it is a
    /// unit impulse. Resampling to the engine rate happens in `init`.
    pub fn with_table(table: Arc<CoeffTable>) -> Self {
        let mut node = Self::new();
        node.residual_source = None;
        for e in &table.entries {
            let rate = e
                .metadata
                .iter()
                .find(|(k, _)| k == RESIDUAL_RATE_KEY)
                .map(|(_, v)| *v);
            if let (Some(rate), Some(row)) = (rate, e.channel(0))
                && rate.is_finite()
                && rate > 0.0
                && !row.is_empty()
            {
                node.residual_source = Some((rate as f64, row.to_vec()));
                break;
            }
        }
        node
    }
}

/// Metadata key naming a residual entry's sample rate.
pub const RESIDUAL_RATE_KEY: &str = "residual_rate_hz";

impl GuitarString {
    fn read(inputs: &[Option<&AudioBuffer>], k: usize, i: usize) -> f32 {
        let d = port_default(k);
        finite_or(read_input(inputs.get(k).copied().flatten(), 0, i, d), d)
    }

    /// The pitch at sample `i`: non-finite reads as 0 (silence).
    fn read_freq(inputs: &[Option<&AudioBuffer>], i: usize) -> f64 {
        let v = read_input(inputs.get(P_FREQ).copied().flatten(), 0, i, 440.0);
        if v.is_finite() { v as f64 } else { 0.0 }
    }

    fn max_freq(&self) -> f64 {
        self.sample_rate * MAX_FREQ_FRAC
    }

    /// The string for a note: the `string` port, or the built-in rule.
    fn choose_string(f: f64, port: f32, opens: &[f64; 6]) -> usize {
        let p = port.round();
        if (1.0..=6.0).contains(&p) {
            return p as usize - 1;
        }
        let tol = (0.25f64 / 12.0).exp2();
        let mut best: Option<usize> = None;
        for (s, &o) in opens.iter().enumerate() {
            if o <= f * tol && best.is_none_or(|b| o > opens[b]) {
                best = Some(s);
            }
        }
        best.unwrap_or_else(|| {
            // Below every open string: the lowest string.
            (0..6)
                .min_by(|&a, &b| opens[a].total_cmp(&opens[b]))
                .unwrap_or(5)
        })
    }

    /// Design both loops for a note at `freq` and (when `pluck`) start the
    /// excitation. Inputs at sample `i`.
    fn configure(&mut self, inputs: &[Option<&AudioBuffer>], i: usize, freq: f64, pluck: bool) {
        let sr = self.sample_rate;
        let rd = |k: usize| Self::read(inputs, k, i) as f64;
        let mut opens = [0.0f64; 6];
        for (s, o) in opens.iter_mut().enumerate() {
            *o = rd(gs_string_port(s + 1, 0)).clamp(GS_MIN_FREQ as f64, self.max_freq());
        }
        let s = Self::choose_string(freq, Self::read(inputs, P_STRING, i), &opens);
        self.string = s;
        let o = opens[s];
        let b_open = rd(gs_string_port(s + 1, 1)).clamp(0.0, 0.01);
        let t_open = rd(gs_string_port(s + 1, 2)).clamp(0.01, 120.0);
        let h_open = rd(gs_string_port(s + 1, 3)).clamp(0.01, 120.0);
        let share = rd(gs_string_port(s + 1, 4)).clamp(0.0, 1.0);

        // Per-note variation.
        self.plucks = self.plucks.wrapping_add(1);
        let seed_bits = (Self::read(inputs, P_SEED, i)).to_bits() as u64;
        let mut mix = Mix(((self.voice_seed as u64) << 32)
            ^ seed_bits.wrapping_mul(0x2545_F491_4F6C_DD1D)
            ^ self.plucks.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let vary = rd(P_VARY).clamp(0.0, 1.0);
        let (j_pos, j_angle, j_level, j_split) = (
            mix.bipolar() * 10.0 * vary,
            mix.bipolar() * 10.0 * vary,
            mix.bipolar() * vary,
            mix.bipolar() * 0.5 * vary,
        );

        let scale = rd(P_SCALE).clamp(200.0, 1200.0);
        let k = freq / o;
        let b = b_open * k * k;
        let len_f = scale / k;
        let pos = (rd(P_POS) + j_pos).clamp(1.0, len_f - 1.0);
        let beta = (pos / len_f).clamp(0.005, 0.95);
        let angle = ((rd(P_ANGLE) + j_angle).clamp(-90.0, 90.0)).to_radians();
        let split = (rd(P_SPLIT) + j_split).clamp(-50.0, 50.0);
        let hv = rd(P_HV).clamp(0.0, 4.0);
        self.hmix = rd(P_HMIX).clamp(0.0, 1.0);
        self.exc_w = [angle.cos(), angle.sin()];
        self.rel = rd(P_REL).clamp(0.005, 3.0);

        // Decay law (module doc).
        let sig_t = LN_1000 / t_open;
        let sig_h = LN_1000 / h_open;
        let sig_br_open = share * sig_t;
        let (ia, ib) = ((1.0 - share) * sig_t, (sig_h - sig_br_open).max(0.0));
        let intrinsic = |f: f64| -> f64 {
            if GS_F_HI > o * 1.01 {
                (ia + (ib - ia) * (f - o) / (GS_F_HI - o)).max(0.0)
            } else {
                ia
            }
        };
        let f0i = freq / (1.0 + b).sqrt();
        let fa = freq;
        let fb = (GS_F_HI.max(2.5 * fa)).min(0.4 * sr);
        let (wa, wb) = (TWO_PI * fa / sr, TWO_PI * fb / sr);
        let (ta, tb) = (stiff_dn_df(fa, f0i, b), stiff_dn_df(fb, f0i, b));
        let sig_br = sig_br_open * k;

        let vel = rd(P_VEL).clamp(0.0, 1.0);
        let hard = rd(P_HARD).clamp(0.0, 1.0);
        let width = rd(P_WIDTH).clamp(0.1, 40.0);
        let level = vel * (j_level / 20.0 * core::f64::consts::LN_10).exp();
        let w_eff = width * (1.25 - 0.5 * vel) * (1.0 - 0.5 * hard);
        let c = 2.0 * scale * 1e-3 * o;
        let ramp = (w_eff * 1e-3 / c * sr).clamp(1.0, MAX_RAMP as f64);
        let bright = vel * (0.5 + 0.5 * hard);
        let f_dyn = (500.0 * (6.0 * bright).exp2()).min(0.45 * sr);
        let dyn_r = (-TWO_PI * f_dyn / sr).exp();

        for (p, pol) in self.pols.iter_mut().enumerate() {
            let pf = if p == 0 {
                freq
            } else {
                freq * (-split / 1200.0).exp2()
            };
            pol.freq = pf;
            let br = if p == 0 { sig_br } else { hv * sig_br };
            let ma = (-(br + intrinsic(fa)) * ta).exp();
            let mb = (-(br + intrinsic(fb)) * tb).exp();
            let (g, a) = design_loss(ma, wa, mb, wb);
            pol.loss_b = g * (1.0 + a);
            pol.loss_a = a;
            let period = sr / pf;
            let m = ((beta * period).round() as usize).clamp(1, (period as usize).max(2) - 1);
            pol.period = period;
            pol.m = m;
            pol.beta = m as f64 / period;
            if pluck {
                pol.amp = level * if p == 0 { angle.cos() } else { angle.sin() };
                pol.dyn_r = dyn_r;
            }
        }

        // Dispersion, designed on V with V's loss phase; shared by H.
        self.disp = note_dispersion(sr, freq, b, self.pols[0].loss_a);
        for pol in &mut self.pols {
            if self.disp.kind == DispKind::Second {
                for (k, q) in pol.disp.iter_mut().enumerate().take(self.disp.n) {
                    q.a1 = -2.0 * self.disp.rho[k] * self.disp.th[k].cos();
                    q.a2 = self.disp.rho[k] * self.disp.rho[k];
                }
            }
        }

        if pluck {
            self.build_step_response(ramp);
            let longest = self.pols[0].period.max(self.pols[1].period);
            self.exc_len = longest + self.q.len() as f64 + 2.0;
            self.t = 0.0;
            self.exciting = true;
            self.plucked = true;
            self.released = false;
        }
        self.configured = true;
        self.last_freq = f64::NAN;
    }

    /// Step response of the contact ramp convolved with the residual.
    fn build_step_response(&mut self, ramp: f64) {
        let kh = ramp.ceil() as usize;
        let s_hat = |t: f64| -> f64 {
            if t <= 0.0 {
                0.0
            } else if t >= ramp {
                1.0
            } else {
                0.5 * (1.0 - (PI * t / ramp).cos())
            }
        };
        let lr = self.residual.len();
        let n = (lr + kh + 1).min(self.q.capacity());
        self.q.clear();
        let mut acc = 0.0f64;
        for t in 0..n {
            // q[t] = Σ_k h[k] r[t - k], h[k] = S(k+1) - S(k).
            let mut v = 0.0;
            for kk in 0..=kh.min(t) {
                let ri = t - kk;
                if ri < lr {
                    v += (s_hat(kk as f64 + 1.0) - s_hat(kk as f64)) * self.residual[ri] as f64;
                }
            }
            acc += v;
            self.q.push(acc as f32);
        }
        self.q_end = acc;
    }

    /// Re-solve both loops' delays for the current pitch and damping.
    fn retune(&mut self, freq: f64) {
        let sr = self.sample_rate;
        let split_ratio = if self.pols[0].freq > 0.0 {
            self.pols[1].freq / self.pols[0].freq
        } else {
            1.0
        };
        for (p, pol) in self.pols.iter_mut().enumerate() {
            let pf = if p == 0 { freq } else { freq * split_ratio };
            let w1 = TWO_PI * pf / sr;
            let lag = lp_lag(pol.loss_a, w1) + lp_lag(pol.damp_a, w1) + self.disp.lag(w1);
            let total = ((TWO_PI - lag) / w1).max(1.0 + THIRAN_LO);
            let mut d = total - pol.n_int as f64;
            if !(THIRAN_LO..=THIRAN_HI).contains(&d) || pol.n_int < 1 {
                pol.n_int = ((total - 3.0).round().max(1.0) as usize).min(self.len - 1);
                d = total - pol.n_int as f64;
            }
            let d0 = d.clamp(2.01, 64.0);
            let a = thiran3(d0);
            let err = thiran_lag(&a, w1) / w1 - d0;
            let d1 = (d0 - err).clamp(2.01, 64.0);
            pol.thiran = thiran3(d1);
        }
    }

    /// Update the damping filters for the current `d`.
    fn set_damping(&mut self, freq: f64) {
        for pol in &mut self.pols {
            let f = if pol.freq > 0.0 {
                pol.freq
            } else {
                freq.max(1.0)
            };
            let g = (-self.d * LN_1000 / (f * self.rel)).exp();
            let a = -0.5 * self.d;
            pol.damp_b = g * (1.0 + a);
            pol.damp_a = a;
        }
    }

    fn silence(&mut self) {
        for l in &mut self.lines {
            l.fill(0.0);
        }
        self.exc_hist.fill(0.0);
        for p in &mut self.pols {
            p.clear_state();
        }
        self.exciting = false;
        self.follow = 0.0;
        self.done = true;
    }
}

impl UGen for GuitarString {
    fn spec(&self) -> crate::node::UGenSpec {
        static OUTPUTS: [OutputSpec; 1] = [OutputSpec {
            name: "out",
            rate: Rate::Audio,
        }];
        crate::node::UGenSpec {
            name: "GuitarString",
            category: UGenCategory::Physical,
            inputs: &GS_INPUTS,
            outputs: &OUTPUTS,
        }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate as f64;
        let sr = self.sample_rate;
        self.len = (sr / GS_MIN_FREQ as f64) as usize + 8;
        for l in &mut self.lines {
            l.clear();
            l.resize(self.len, 0.0);
        }
        self.exc_hist.clear();
        self.exc_hist.resize(self.len, 0.0);
        if let Some((rate, row)) = self.residual_source.take() {
            // Resample linearly to the engine rate, at most GS_MAX_RESIDUAL_S.
            let n = ((row.len() as f64) * sr / rate).ceil() as usize;
            let n = n.clamp(1, (GS_MAX_RESIDUAL_S * sr) as usize);
            let mut r = Vec::with_capacity(n);
            for j in 0..n {
                let x = j as f64 * rate / sr;
                let i0 = x as usize;
                let fr = x - i0 as f64;
                let a = finite_or(row.get(i0).copied().unwrap_or(0.0), 0.0) as f64;
                let b = finite_or(row.get(i0 + 1).copied().unwrap_or(0.0), 0.0) as f64;
                r.push((a + (b - a) * fr) as f32);
            }
            self.residual = r;
            self.residual_source = Some((rate, row));
        }
        self.q = Vec::with_capacity(self.residual.len() + MAX_RAMP + 2);
        self.follow_k = (-1.0 / (FOLLOW_S * sr)).exp();
        self.d_coef = 1.0 - (-1.0 / (DAMP_S * sr)).exp();
        self.reset();
    }

    fn reset(&mut self) {
        for l in &mut self.lines {
            l.fill(0.0);
        }
        self.exc_hist.fill(0.0);
        self.write = 0;
        self.pols = [Pol::default(); 2];
        self.disp = DispDesign::NONE;
        self.t = 0.0;
        self.exc_len = 0.0;
        self.exciting = false;
        self.d = 0.0;
        self.released = false;
        self.follow = 0.0;
        self.configured = false;
        self.plucked = false;
        self.done = false;
        self.prev_gate = 0.0;
        self.last_freq = f64::NAN;
        self.plucks = 0;
    }

    fn reseed_noise(&mut self, seed: u32) {
        self.voice_seed = seed;
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let block = output.block_size();
        let mut mono = [0.0f32; MAX_BLOCK_SIZE];
        if self.len == 0 {
            for ch in 0..output.num_channels() {
                output.channel_mut(ch).samples_mut()[..block].fill(0.0);
            }
            return;
        }
        let exc_buf = inputs.get(P_EXC).copied().flatten();
        let damp_buf = inputs.get(P_DAMP).copied().flatten();
        let tap_buf = inputs.get(P_TAP).copied().flatten();

        for (i, o) in mono[..block].iter_mut().enumerate() {
            let raw_freq = Self::read_freq(inputs, i);
            let gate = Self::read(inputs, P_GATE, i);
            let rising = gate > 0.0 && self.prev_gate <= 0.0;
            let falling = gate <= 0.0 && self.prev_gate > 0.0;
            self.prev_gate = gate;

            if raw_freq <= 0.0 {
                // Silence: no pluck, and a sounding string stops at once.
                if !self.done && (self.plucked || rising) {
                    self.plucked = true;
                    self.silence();
                }
                *o = 0.0;
                continue;
            }
            let freq = raw_freq.clamp(GS_MIN_FREQ as f64, self.max_freq());

            if rising {
                self.done = false;
                self.configure(inputs, i, freq, true);
                self.set_damping(freq);
            } else if !self.configured {
                self.configure(inputs, i, freq, false);
                self.set_damping(freq);
            }
            if falling {
                self.released = true;
            }

            // Damping ramp.
            let damp = finite_or(read_input(damp_buf, 0, i, 0.0), 0.0).clamp(0.0, 1.0) as f64;
            let target = if self.released { 1.0 } else { damp };
            let mut moved = false;
            if self.d != target {
                let nd = self.d + (target - self.d) * self.d_coef;
                self.d = if (nd - target).abs() < 1e-6 {
                    target
                } else {
                    nd
                };
                self.set_damping(freq);
                moved = true;
            }
            if moved || freq.to_bits() != self.last_freq.to_bits() {
                self.retune(freq);
                self.last_freq = freq;
            }

            let exc =
                finite_or(read_input(exc_buf, 0, i, 0.0), 0.0).clamp(-EXC_MAX, EXC_MAX) as f64;
            if exc != 0.0 {
                self.plucked = true;
                self.done = false;
            }
            if self.done {
                *o = 0.0;
                continue;
            }

            let len = self.len;
            let w = self.write;
            self.exc_hist[w] = exc as f32;
            let mut x = [0.0f64; 2];
            for (p, xo) in x.iter_mut().enumerate() {
                let pol = &mut self.pols[p];
                let read = self.lines[p][(w + len - pol.n_int) % len] as f64;
                let y = pol.filters(read, &self.disp);
                // Excitation.
                let mut u = 0.0;
                if self.exciting {
                    let raw = pol.pluck_sample(&self.q, self.q_end, self.t);
                    pol.dyn_s = (1.0 - pol.dyn_r) * raw + pol.dyn_r * pol.dyn_s;
                    u = pol.dyn_s;
                }
                let past = self.exc_hist[(w + len - pol.m.min(len - 1)) % len] as f64;
                u += self.exc_w[p] * (exc - past);
                let xv = y + u;
                let xv = if xv.is_finite() { xv } else { 0.0 };
                self.lines[p][w] = xv as f32;
                *xo = xv;
            }
            self.write = (w + 1) % len;

            if self.exciting {
                self.t += 1.0;
                // Over once the waveform has ended and the dynamic lowpass has
                // drained (a backstop four seconds later in any case).
                let drained = self.pols[0].dyn_s.abs() < 1e-12 && self.pols[1].dyn_s.abs() < 1e-12;
                if self.t >= self.exc_len
                    && (drained || self.t >= self.exc_len + 4.0 * self.sample_rate)
                {
                    self.exciting = false;
                }
            }

            self.follow = self.follow * self.follow_k
                + (1.0 - self.follow_k) * (x[0].abs() + x[1].abs() + exc.abs());
            let tap = finite_or(read_input(tap_buf, 0, i, 0.0), 0.0);
            let v = if tap >= 0.5 {
                OUT_GAIN * x[0]
            } else {
                OUT_GAIN * (x[0] + self.hmix * x[1])
            };
            *o = v as f32;

            if self.plucked && !self.exciting && self.follow < DONE_LEVEL {
                self.silence();
            }
        }

        for ch in 0..output.num_channels() {
            output.channel_mut(ch).samples_mut()[..block].copy_from_slice(&mono[..block]);
        }
    }
}

/// Register the table-bound form of `guitarString` (the commuted residual
/// read from a bound table). The bare form is registered with the other
/// built-ins; a node with a `table_bindings` entry resolves to this one.
pub fn register(reg: &mut UGenRegistry) {
    let factory: TableUGenFactory = Arc::new(|table| Box::new(GuitarString::with_table(table)));
    reg.register_table_bound(
        "guitarString",
        factory,
        UGenCategory::Physical,
        &GS_INPUTS,
        &[OutputSpec {
            name: "out",
            rate: Rate::Audio,
        }],
    );
}
