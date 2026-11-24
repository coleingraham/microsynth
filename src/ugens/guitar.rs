//! `guitarString`: one plucked steel string of a six-string guitar, as a
//! digital waveguide with two polarisations.
//!
//! # Model
//!
//! Each polarisation is a single delay loop (one round trip of the string):
//!
//! ```text
//! x[n] = y[n] + u[n]                     (u: the pluck, plus `exc`)
//! y    = Disp · Damp · Loss · Thiran · Lagrange(z^(-R)) x
//! ```
//!
//! `x` is the wave arriving at and leaving the bridge, read as the force the
//! string puts on the bridge. The delay line is read `R` samples back; `R` is
//! a whole number for a held pitch and is read through a fifth-order
//! Lagrange interpolator only while the pitch or the damping moves (below).
//! `Thiran` is a third-order Thiran all-pass for the fractional part of the
//! period at the pluck, `Loss` the loss filter, `Damp` the release/palm-mute
//! filter and `Disp` an all-pass dispersion cascade for the string's
//! stiffness. The vertical polarisation (`V`, normal to the top) and the
//! horizontal one (`H`) are two independent loops: `H` is tuned the string's
//! `split` cents flat of `V` and loses its `hv` times `V`'s bridge loss.
//! There is no cross-coupling.
//!
//! # Ports (per-note controls)
//!
//! All optional, read from input channel 0; the output is mono, copied to
//! every output channel.
//!
//! | port | meaning | read | default |
//! |---|---|---|---|
//! | `freq` | sounding pitch in Hz (the first partial) | every sample | 440 |
//! | `gate` | a rise plucks, a fall damps (release) | every sample | 1 |
//! | `vel` | velocity 0..1 | at the pluck | 0.8 |
//! | `string` | 1..6 (1 = high E), or -1: the built-in rule (below) | at the pluck | -1 |
//! | `pos` | pluck point, mm from the saddle | at the pluck | 140 |
//! | `angle` | pluck angle from the normal, degrees | at the pluck | 20 |
//! | `hard` | 0 flesh, 0.5 nail, 1 plectrum | at the pluck | 0.5 |
//! | `width` | contact width, mm | at the pluck | 10 |
//! | `damp` | extra damping 0..1 (palm mute, a hand on the string) | every sample | 0 |
//! | `exc` | external excitation, injected at the pluck point (clamped to ±10) | every sample | 0 |
//! | `seed` | per-note variation seed | at the pluck | 0 |
//! | `rel` | release decay `T_rel`: T60 (s) of the fundamental once fully damped | every sample | 0.15 |
//! | `vary` | depth of the seeded per-note variation, 0..1 | at the pluck | 1 |
//! | `tap` | which signal is output: 0 `out`, 1 `bridge` | every sample | 0 |
//!
//! `rel` is read every sample because a hand's damping speed is a playing
//! choice that can change at the release itself (a score can set it with the
//! note-off); a change re-solves the damping filter.
//!
//! # The string set (instrument constants)
//!
//! Every constant of the instrument lives in a [`GsStringSet`], not in
//! ports: a coefficient-table entry bound to the node carries it (see
//! "Table binding"), so every voice of a part reads the same strings, and a
//! bare node uses [`GsStringSet::default`] (standard tuning, a light steel
//! set, provisional decays). Per string: open pitch, inharmonicity `B` of
//! the open string, T60 of the open string's fundamental and T60 at three
//! anchor frequencies (shared by the six strings, default 1, 3 and 8 kHz),
//! the bridge's share of the open fundamental's decay, the V–H detune
//! `split` (cents) and `hv`. Per instrument: scale length, `hmix` (`H`'s
//! share of `out`, a property of where the top and the microphone see the
//! string), and the velocity→brightness mapping (corner and exponent). All
//! are measurable quantities, and are mapped to the loop at the pluck:
//!
//! - **Fret.** `k = f / o` (the length ratio, `k < 1` plays the string tuned
//!   down). Speaking length `L_f = scale / k`; pluck point `β = pos / L_f`.
//! - **Stiffness.** `B = b · k²` (`B ∝ 1/T L²` at fixed tension). The ideal
//!   fundamental is `f0 = f / sqrt(1 + B)`, so the first partial sounds at
//!   `freq` exactly, and partial `n` is meant to lie at `n f0 sqrt(1 + B n²)`.
//! - **Decay rates** (`σ = ln 1000 / T60`, per second) are a bridge term plus
//!   an intrinsic term. The bridge term is frequency independent and
//!   proportional to the round-trip rate: `σ_br = share · σ_0 · k` (`V`),
//!   `hv · σ_br` (`H`), where `σ_0` is the open fundamental's rate. The
//!   intrinsic term is interpolated linearly in log frequency and log rate
//!   through knots at the open fundamental (`(1 − share) σ_0`) and at the
//!   anchors (the anchor's rate less the open string's bridge term, floored
//!   at a tenth of the anchor's rate), and held flat beyond the end knots. So
//!   a fretted note loses more to the bridge, and its intrinsic loss is the
//!   open string's at the same absolute frequency.
//! - **Loss filter.** `Loss(z) = g (1 + a)/(1 + a z^-1) · (1 + c)/(1 + b) ·
//!   (1 + b z^-1)/(1 + c z^-1)`: a one-pole and a first-order shelf, four
//!   coefficients for four targets: the note's fundamental and the anchors
//!   above it (below 0.45 fs). They are solved at the pluck (damped
//!   Gauss–Newton from four starts) so that the per-round-trip attenuation
//!   matches `σ` there; with fewer targets the shelf is dropped (two: the
//!   closed-form one-pole; one: a constant gain). The solve is constrained so
//!   that no frequency outlasts the note: above the fundamental no frequency
//!   loses less per round trip than the fundamental, above the last anchor
//!   none less than 90 % of the last anchor's loss (the measured plateau),
//!   and DC (the only mode below the fundamental, barely excited by a
//!   zero-mean pluck) at least half the fundamental's. Without these, a shelf
//!   can leave a near-Nyquist or near-DC component ringing for tens of
//!   seconds; with them an anchor can miss by up to about 8 % where the
//!   anchors ask for something the constraints forbid. A partial's decay rate is
//!   `σ_n = −ln|Loss(ω_n)| / τ_n`, where `τ_n = dn/df` at `f_n` is the loop's
//!   round-trip time there. A one-pole alone cannot follow measured decays
//!   (on an archtop's octave bands it leaves factors of 1.6 to 5); with the
//!   shelf the anchors match exactly and the other octave bands come within
//!   about 25 %, except where the measured curve is not monotone (a body
//!   mode taking energy, which belongs to a coupled bridge). `g` is lowered
//!   if needed so that `|Loss| < 1` at every frequency, so the loop is stable
//!   for any string set. [`gs_note_design`] returns the filters a note gets.
//! - **Dispersion.** The all-pass cascade is designed at the pluck from `B`,
//!   `f0` and the loss filter's own phase: either up to eight second-order
//!   sections placed on bands of 2π of the target phase (the band-splitting
//!   design of Abel, Välimäki and Smith; pole radius `exp(−0.9 · band
//!   width)`), or one to four equal first-order sections, whichever puts
//!   partials 2..60 below 10 kHz closer to `n f0 sqrt(1 + B n²)`; none when
//!   `B` is 0 or neither helps. A cascade over- or under-states `B`
//!   (5–36 % before correction, by string, fret and rate), so the design is
//!   pre-warped against `B_fit`, the `B` fitted in closed form to the
//!   partials the loop will play (1..30 below `min(5 kHz, 0.45 fs)`, solved
//!   from the loop's phase with the delay and the Thiran filter as fixed at
//!   the pluck): for second-order sections `B'` is iterated (`B' ← B' · B /
//!   B_fit`, at most six times, the closest kept); for first-order sections
//!   the coefficient is found by bisection so that `B_fit = B` (iterating
//!   `B'` through the coefficient search jumped between its local minima).
//!   During the pre-warp the integer delay is held at the first design's,
//!   because a step of the integer delay moves the Thiran fraction by a
//!   whole sample and with it (at low rates) the high partials. Measured:
//!   `B` round-trips within 1 % on all six strings at frets 0, 5 and 12, at
//!   44.1 and 22.05 kHz. Individual partials still ripple around the
//!   stretched series (most at the top of the designed band; see the
//!   tests).
//! - **Tuning.** The integer delay `N` and the Thiran fraction are solved at
//!   the pluck so the loop's total phase at `freq` is exactly 2π, including
//!   every filter's phase there (one Newton step corrects the Thiran filter's
//!   own phase error).
//!
//! # Pitch changes and damping: the time-varying part
//!
//! The Thiran filter is recursive: re-solving it while the pitch moves (as
//! the first version did) pumps energy into the loop, and a vibrato grew
//! without bound. So the Thiran filter is fixed at the pluck, and a pitch
//! change moves the delay line's read point instead: `R` is re-solved
//! whenever `freq`, the damping or `rel` changes, so the total loop phase at
//! the current `freq` is again 2π (every filter's phase at the new pitch
//! included), and a non-integer `R` is read through a six-tap (fifth-order)
//! Lagrange interpolator, an FIR whose gain is at most 1 at every fraction
//! on its central interval, so the moving part cannot add energy. At a
//! fraction of 0 it is the plain tap, so a held note is untouched. Its
//! high-frequency loss acts only while `R` is fractional: at a fraction of
//! 0.5 and 44.1 kHz, 0.0014 dB per pass at 4 kHz and 0.26 dB at 10 kHz (a
//! third-order interpolator loses 15 times more at 4 kHz, which made an E6
//! vibrato decay 25 % faster). Measured (60 s T60 strings, 4 s): a ±50-cent
//! 6 Hz vibrato changes the RMS decay rate by at most 1.3 % (A2 to A4), a
//! ±2-semitone 12 Hz one by about 1 % up to E4 and 10 % on E6; the RMS never
//! rises; the largest second difference stays under the held note's. The
//! loss and dispersion filters keep their pluck-time design (the small
//! change of `B` and of the per-second loss with tension is ignored). `R` is
//! held at 3 samples or more, so a pitch that would need less (above about
//! `fs / 8`, after the filters' delay) plays flat.
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
//! interpolation from 0 at `t = -1`. It then passes a one-pole "dynamic
//! level" lowpass (Jaffe and Smith) with its corner at
//! `corner · (vel · (0.5 + 0.5 hard))^exponent` (the string set's
//! `vel_corner_hz` and `vel_corner_exp`, default 16 kHz and 1.5; capped at
//! 0.45 fs). So velocity changes the timbre as well as the level (`vel`,
//! linear), and a fit sets the mapping's range. `V` takes `cos(angle)` of the
//! pluck, `H` `sin(angle)`.
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
//! **Cost at the pluck.** A pluck designs two loss filters (a small
//! constrained least-squares solve from seven starts), the dispersion
//! cascade with its pre-warp (the larger part: candidate searches and up to
//! six re-designs, each solving 30 partials from the loop's phase), and the
//! excitation's step response (contact ramp ∗ residual: about
//! `ramp × residual length` multiplies, 8 800 for a 4-sample ramp and a 50 ms
//! residual at 44.1 kHz). Measured: about 3 ms of CPU per pluck (release
//! build, E2–A6 at 44.1 and 96 kHz), once per note, without allocating. In a
//! piece of a thousand notes that is about 3 s, comparable to the rendering
//! itself.
//!
//! # Table binding
//!
//! Bound to a coefficient table (the IR's `table_bindings`, as
//! `partialsNoise` is), the node reads two kinds of entry, by metadata key
//! (no new table format; `docs/coeff-table-bank-format.md`):
//!
//! - **The string set**: the first entry carrying `gs.string_set` (any
//!   value). Keys (any missing key keeps its default): `gs.scale_mm`,
//!   `gs.hmix`, `gs.vel_corner_hz`, `gs.vel_corner_exp`, `gs.anchor_hz.1` ..
//!   `gs.anchor_hz.3`, and per string `S` = 1..6: `gs.sS.open_hz`, `gs.sS.b`,
//!   `gs.sS.t60.0` (open fundamental) .. `gs.sS.t60.3` (at the anchors),
//!   `gs.sS.bridge_share`, `gs.sS.split_cents`, `gs.sS.hv`. Values are
//!   clamped to physical ranges when read ([`GsStringSet::from_table`]).
//! - **The commuted body residual**: the first entry carrying
//!   `residual_rate_hz`; its channel-0 coefficient row (`k_channels` 1, no
//!   partials, `j_noise` = the length) is an impulse response sampled at that
//!   rate (resampled linearly to the engine rate when the node is built; at
//!   most [`GS_MAX_RESIDUAL_S`] seconds). The pluck waveform is convolved with
//!   it (commuted synthesis: body ∗ string ∗ pluck = string ∗ (body ∗
//!   pluck)), so the residual colours both `out` and `bridge`.
//!
//! The binding also gives the node a stable handle shared by the voices of a
//! part, which a later shared session (same-string chokes, sympathetic
//! strings) needs.
//!
//! # Outputs
//!
//! The engine gives a node one output, so `tap` is a mode switch: 0 = `out`
//! (`0.5 · (x_V + hmix · x_H)`, the bridge-force mix that drives the top),
//! 1 = `bridge` (`0.5 · x_V`, the vertical bridge force alone). Coupling
//! between strings is meant to happen inside a shared state, not through the
//! graph, so `tap` exists for analysis and for a body stage that wants the
//! force alone; no other output is reserved. The low body modes go after the
//! string, e.g. a `modalBank` with `freq` held at 100 and each mode's ratio
//! `f_k / 100`.
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
//! the note re-plucks: the new pluck is added to the ringing string (the
//! filters are re-designed for it; dispersion state is cleared when the
//! design's kind or section count changes).
//!
//! # Done
//!
//! Once plucked (or excited through `exc`), when no excitation is still
//! being injected and a follower of `|x_V| + |x_H|` (20 ms) falls below
//! 1e-5 (about −100 dBFS), the loops are cleared and the node reports done;
//! its output is silent from then on. A held note frees itself once it has
//! rung out, like a real pluck. `freq` ≤ 0 (or not finite) is silence: a
//! sounding string stops at once, and the node reports done. A gate that
//! rises while `freq` ≤ 0 leaves a pluck pending for up to 0.1 s (or until
//! the gate falls), so a `freq` that arrives a block after the gate still
//! plays; the node does not report done while a pluck is pending.
//!
//! # Hygiene and cost
//!
//! Non-finite inputs read as their defaults (`freq` as 0, `exc` as 0); every
//! port and string-set value is clamped to a physical range, with every
//! computed bound kept ordered, so no input can panic. `freq` is clamped to
//! `20 Hz .. fs / 8` (silently). State is f64 except the delay lines (f32).
//! Per voice: two delay lines and an `exc` history of `fs / 20 + 8` samples
//! each (f32, about 26.6 KB at 44.1 kHz), the residual's step response (f32),
//! and under 1.5 KB of filter state and string set. Nothing allocates while
//! rendering. Per sample: about 130 multiplies with eight dispersion
//! sections on both loops, plus, on samples where `freq`, the damping or
//! `rel` moves, the read-point re-solve (a few dozen trigonometric calls) and
//! a six-tap interpolation.

use crate::buffer::{AudioBuffer, MAX_BLOCK_SIZE, read_input};
use crate::coeff_table::CoeffTable;
use crate::context::{ProcessContext, Rate};
use crate::dsl::compiler::{TableUGenFactory, UGenRegistry};
use crate::node::{InputSpec, OutputSpec, UGen, UGenCategory};
use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
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
/// Number of loss anchors above the open fundamental.
pub const GS_ANCHORS: usize = 3;
/// Longest commuted residual, seconds.
pub const GS_MAX_RESIDUAL_S: f64 = 0.2;
/// Longest contact ramp, samples.
const MAX_RAMP: usize = 64;
/// Thiran delay window at the pluck.
const THIRAN_LO: f64 = 2.2;
/// Shortest read point (the interpolator reads two samples nearer).
const MIN_READ: f64 = 3.0;
/// Done threshold on the follower.
const DONE_LEVEL: f64 = 1e-5;
/// Follower time constant, seconds.
const FOLLOW_S: f64 = 0.02;
/// Damping ramp time constant, seconds.
const DAMP_S: f64 = 0.01;
/// How long a pluck waits for a positive `freq`, seconds.
const PENDING_S: f64 = 0.1;
/// `exc` is clamped to ±this.
const EXC_MAX: f32 = 10.0;
/// Output gain.
const OUT_GAIN: f64 = 0.5;
/// Partials and band the B pre-warp fits over.
const PREWARP_PARTIALS: usize = 30;
const PREWARP_TOP_HZ: f64 = 5000.0;
/// Largest loss-filter gain anywhere.
const LOSS_MAX: f64 = 0.999_999_9;

/// Number of ports.
pub const GS_N_PORTS: usize = 14;

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
const P_REL: usize = 11;
const P_VARY: usize = 12;
const P_TAP: usize = 13;

const PORT_NAMES: [&str; GS_N_PORTS] = [
    "freq", "gate", "vel", "string", "pos", "angle", "hard", "width", "damp", "exc", "seed", "rel",
    "vary", "tap",
];

/// The port specs (all optional, audio rate).
static GS_INPUTS: [InputSpec; GS_N_PORTS] = {
    const BLANK: InputSpec = InputSpec {
        name: "",
        rate: Rate::Audio,
        required: false,
    };
    let mut a = [BLANK; GS_N_PORTS];
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
        P_REL => 0.15,
        P_VARY => 1.0,
        _ => 0.0,
    }
}

#[inline]
fn finite_or(v: f32, d: f32) -> f32 {
    if v.is_finite() { v } else { d }
}

/// `x` limited to `lo..=hi` without panicking: a NaN reads as `lo`, and a
/// computed `hi` below `lo` collapses to `lo`.
#[inline]
fn clampf(x: f64, lo: f64, hi: f64) -> f64 {
    let hi = if hi >= lo { hi } else { lo };
    if x.is_nan() { lo } else { x.max(lo).min(hi) }
}

// ---------------------------------------------------------------------------
// The string set.
// ---------------------------------------------------------------------------

/// One string's constants (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GsString {
    /// Open pitch, Hz.
    pub open_hz: f64,
    /// Inharmonicity coefficient of the open string.
    pub b: f64,
    /// T60 (s): `[0]` of the open fundamental, `[1..]` at the anchors.
    pub t60: [f64; GS_ANCHORS + 1],
    /// The bridge's share of the open fundamental's decay rate, 0..1.
    pub bridge_share: f64,
    /// V–H detune, cents (H flat).
    pub split_cents: f64,
    /// H's bridge loss as a fraction of V's.
    pub hv: f64,
}

/// The instrument's constants: six strings and the instrument-wide values
/// (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GsStringSet {
    /// String 1 (high E) to 6 (low E).
    pub strings: [GsString; 6],
    /// Loss anchor frequencies above the open fundamental, Hz, ascending.
    pub anchor_hz: [f64; GS_ANCHORS],
    /// Scale length, mm.
    pub scale_mm: f64,
    /// `H`'s share of `out`.
    pub hmix: f64,
    /// Dynamic-lowpass corner at full velocity and hardness, Hz.
    pub vel_corner_hz: f64,
    /// Exponent of the corner's velocity law.
    pub vel_corner_exp: f64,
}

impl Default for GsStringSet {
    /// Standard tuning; `B` computed for a light 12–53 steel set on a 645 mm
    /// scale (wound strings from estimated core diameters); decays, bridge
    /// shares and `hv` provisional estimates; `split` from an archtop's
    /// measured V–H detune by register (about 8 cents on the low strings,
    /// 1 cent on the plain ones). To be replaced by fitted values.
    fn default() -> Self {
        let s = |open_hz, b, t60, bridge_share, split_cents| GsString {
            open_hz,
            b,
            t60,
            bridge_share,
            split_cents,
            hv: 0.4,
        };
        GsStringSet {
            strings: [
                s(329.627_56, 1.9e-5, [4.0, 2.6, 1.6, 1.0], 0.6, 0.8),
                s(246.941_65, 6.1e-5, [4.5, 2.8, 1.5, 0.9], 0.6, 1.5),
                s(195.997_72, 1.1e-5, [5.0, 2.4, 1.2, 0.8], 0.65, 2.2),
                s(146.832_38, 2.1e-5, [5.5, 2.2, 1.0, 0.7], 0.7, 3.0),
                s(110.0, 4.9e-5, [6.0, 2.0, 0.9, 0.6], 0.75, 6.0),
                s(82.406_89, 1.1e-4, [6.5, 1.8, 0.8, 0.5], 0.8, 8.0),
            ],
            anchor_hz: [1000.0, 3000.0, 8000.0],
            scale_mm: 645.0,
            hmix: 0.3,
            vel_corner_hz: 16_000.0,
            vel_corner_exp: 1.5,
        }
    }
}

/// Metadata key marking the string-set entry.
pub const STRING_SET_KEY: &str = "gs.string_set";
/// Metadata key naming a residual entry's sample rate.
pub const RESIDUAL_RATE_KEY: &str = "residual_rate_hz";

impl GsStringSet {
    /// The string set of the first entry of `table` carrying
    /// [`STRING_SET_KEY`], every missing key at its default; `None` when no
    /// entry carries the key. Values are clamped to physical ranges.
    pub fn from_table(table: &CoeffTable) -> Option<Self> {
        let e = table
            .entries
            .iter()
            .find(|e| e.metadata.iter().any(|(k, _)| k == STRING_SET_KEY))?;
        let get = |k: &str| {
            e.metadata
                .iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| *v as f64)
                .filter(|v| v.is_finite())
        };
        let mut set = Self::default();
        let put = |v: &mut f64, k: &str, lo: f64, hi: f64| {
            if let Some(x) = get(k) {
                *v = clampf(x, lo, hi);
            }
        };
        put(&mut set.scale_mm, "gs.scale_mm", 200.0, 1200.0);
        put(&mut set.hmix, "gs.hmix", 0.0, 1.0);
        put(&mut set.vel_corner_hz, "gs.vel_corner_hz", 100.0, 50_000.0);
        put(&mut set.vel_corner_exp, "gs.vel_corner_exp", 0.0, 6.0);
        for (j, a) in set.anchor_hz.iter_mut().enumerate() {
            put(a, &format!("gs.anchor_hz.{}", j + 1), 50.0, 40_000.0);
        }
        for (i, s) in set.strings.iter_mut().enumerate() {
            let p = format!("gs.s{}.", i + 1);
            put(
                &mut s.open_hz,
                &(p.clone() + "open_hz"),
                GS_MIN_FREQ as f64,
                20_000.0,
            );
            put(&mut s.b, &(p.clone() + "b"), 0.0, 0.01);
            for (j, t) in s.t60.iter_mut().enumerate() {
                put(t, &format!("{p}t60.{j}"), 0.01, 120.0);
            }
            put(&mut s.bridge_share, &(p.clone() + "bridge_share"), 0.0, 1.0);
            put(
                &mut s.split_cents,
                &(p.clone() + "split_cents"),
                -50.0,
                50.0,
            );
            put(&mut s.hv, &(p + "hv"), 0.0, 4.0);
        }
        Some(set)
    }

    /// The metadata pairs that encode this set (the inverse of
    /// [`from_table`](Self::from_table)), for a producer of tables.
    pub fn to_metadata(&self) -> Vec<(String, f32)> {
        let mut m: Vec<(String, f32)> = alloc::vec![
            (STRING_SET_KEY.into(), 1.0),
            ("gs.scale_mm".into(), self.scale_mm as f32),
            ("gs.hmix".into(), self.hmix as f32),
            ("gs.vel_corner_hz".into(), self.vel_corner_hz as f32),
            ("gs.vel_corner_exp".into(), self.vel_corner_exp as f32),
        ];
        for (j, a) in self.anchor_hz.iter().enumerate() {
            m.push((format!("gs.anchor_hz.{}", j + 1), *a as f32));
        }
        for (i, s) in self.strings.iter().enumerate() {
            let p = format!("gs.s{}.", i + 1);
            m.push((format!("{p}open_hz"), s.open_hz as f32));
            m.push((format!("{p}b"), s.b as f32));
            for (j, t) in s.t60.iter().enumerate() {
                m.push((format!("{p}t60.{j}"), *t as f32));
            }
            m.push((format!("{p}bridge_share"), s.bridge_share as f32));
            m.push((format!("{p}split_cents"), s.split_cents as f32));
            m.push((format!("{p}hv"), s.hv as f32));
        }
        m
    }

    /// The string (0-based) for a note at `f`: `port` 1..6, or the built-in
    /// rule.
    fn choose(&self, f: f64, port: f32) -> usize {
        let p = port.round();
        if (1.0..=6.0).contains(&p) {
            return p as usize - 1;
        }
        let tol = (0.25f64 / 12.0).exp2();
        let mut best: Option<usize> = None;
        for (s, st) in self.strings.iter().enumerate() {
            if st.open_hz <= f * tol && best.is_none_or(|b| st.open_hz > self.strings[b].open_hz) {
                best = Some(s);
            }
        }
        best.unwrap_or_else(|| {
            (0..6)
                .min_by(|&a, &b| self.strings[a].open_hz.total_cmp(&self.strings[b].open_hz))
                .unwrap_or(5)
        })
    }

    /// The decay rate law of string `s` for a note at `freq` (fret ratio
    /// `k`): `σ(f)` per second for polarisation V (`hv` = 1) or H.
    fn sigma(&self, s: usize, k: f64, hv: f64, f: f64) -> f64 {
        let st = &self.strings[s];
        let sig0 = LN_1000 / st.t60[0];
        let br = st.bridge_share * sig0;
        // Knots (log f, log σ_int), ascending in f.
        let mut kf = [0.0f64; GS_ANCHORS + 1];
        let mut ks = [0.0f64; GS_ANCHORS + 1];
        let mut n = 0;
        kf[0] = st.open_hz.ln();
        ks[0] = ((1.0 - st.bridge_share) * sig0).max(1e-3 * sig0).ln();
        n += 1;
        for j in 0..GS_ANCHORS {
            let fa = self.anchor_hz[j];
            if fa > st.open_hz * 1.01 && fa.ln() > kf[n - 1] {
                let sa = LN_1000 / st.t60[j + 1];
                kf[n] = fa.ln();
                ks[n] = (sa - br).max(0.1 * sa).ln();
                n += 1;
            }
        }
        let lf = f.max(1e-3).ln();
        let lint = if lf <= kf[0] {
            ks[0]
        } else if lf >= kf[n - 1] {
            ks[n - 1]
        } else {
            let mut v = ks[n - 1];
            for i in 1..n {
                if lf <= kf[i] {
                    let t = (lf - kf[i - 1]) / (kf[i] - kf[i - 1]);
                    v = ks[i - 1] + t * (ks[i] - ks[i - 1]);
                    break;
                }
            }
            v
        };
        hv * br * k + lint.exp()
    }
}

// ---------------------------------------------------------------------------
// Design helpers (f64, closed form or bounded iteration; no allocation).
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

/// `|1 + a e^{-jw}|`.
fn pz_mag(a: f64, w: f64) -> f64 {
    (1.0 + 2.0 * a * w.cos() + a * a).sqrt()
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
fn bisect(lo: f64, hi: f64, target: f64, f: impl Fn(f64) -> f64) -> f64 {
    bisect_n(lo, hi, target, f, 56)
}

/// [`bisect`] with `iters` halvings.
fn bisect_n(mut lo: f64, mut hi: f64, target: f64, f: impl Fn(f64) -> f64, iters: u32) -> f64 {
    for _ in 0..iters {
        let mid = 0.5 * (lo + hi);
        if f(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The loss filter `g (1 + a)/(1 + a z^-1) · (1 + c)/(1 + b) · (1 + b
/// z^-1)/(1 + c z^-1)`. `b = c = 0` is the one-pole alone.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GsLoss {
    pub g: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl GsLoss {
    /// `|Loss(e^{jw})|`.
    pub fn mag(&self, w: f64) -> f64 {
        self.g * (1.0 + self.a) / pz_mag(self.a, w) * (1.0 + self.c) / (1.0 + self.b)
            * pz_mag(self.b, w)
            / pz_mag(self.c, w)
    }

    /// Phase lag at `w`.
    pub fn lag(&self, w: f64) -> f64 {
        lp_lag(self.a, w) + lp_lag(self.c, w) - lp_lag(self.b, w)
    }

    /// Group delay (samples) at `w`.
    fn gd(&self, w: f64) -> f64 {
        lp_gd(self.a, w) + lp_gd(self.c, w) - lp_gd(self.b, w)
    }

    /// `ln |Loss|` with `g = 1`.
    fn shape(a: f64, b: f64, c: f64, w: f64) -> f64 {
        ((1.0 + a) / pz_mag(a, w) * (1.0 + c) / (1.0 + b) * pz_mag(b, w) / pz_mag(c, w)).ln()
    }
}

/// Closed-form one-pole through two targets: per-round-trip magnitude `ma`
/// at `wa` and `mb` at `wb`. Returns `a`.
fn onepole_a(ma: f64, wa: f64, mb: f64, wb: f64) -> f64 {
    if wb > wa * 1.05 && mb < ma {
        let r2 = (mb / ma) * (mb / ma);
        let (ca, cb) = (wa.cos(), wb.cos());
        let p = (ca - r2 * cb) / (1.0 - r2);
        let a = if p > 1.0 {
            -p + (p * p - 1.0).sqrt()
        } else {
            -0.95
        };
        clampf(a, -0.95, 0.0)
    } else {
        0.0
    }
}

/// Solve a 3×3 (or smaller) linear system in place (Gaussian elimination
/// with partial pivoting). Returns false if singular.
fn solve3(m: usize, a: &mut [[f64; 3]; 3], b: &mut [f64; 3]) -> bool {
    for col in 0..m {
        let mut piv = col;
        for r in col + 1..m {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-300 {
            return false;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let prow = a[col];
        for r in col + 1..m {
            let f = a[r][col] / prow[col];
            for (x, p) in a[r][col..m].iter_mut().zip(&prow[col..m]) {
                *x -= f * p;
            }
            b[r] -= f * b[col];
        }
    }
    for col in (0..m).rev() {
        let mut s = b[col];
        for c in col + 1..m {
            s -= a[col][c] * b[c];
        }
        b[col] = s / a[col][col];
    }
    true
}

/// Design the loss filter for targets `(w_i, λ_i)` (per-round-trip loss in
/// nepers), the first being the fundamental (matched exactly through `g`).
/// See the module doc.
fn design_loss(targets: &[(f64, f64)]) -> GsLoss {
    let n = targets.len();
    if n == 0 {
        return GsLoss {
            g: 1.0,
            ..GsLoss::default()
        };
    }
    let (w0, l0) = targets[0];
    let finish = |a: f64, b: f64, c: f64| -> GsLoss {
        let g = (-l0 - GsLoss::shape(a, b, c, w0)).exp();
        let mut loss = GsLoss { g, a, b, c };
        // Keep |Loss| < 1 everywhere (checked on a grid and at the ends).
        let mut peak = 0.0f64;
        for i in 0..=128 {
            peak = peak.max(loss.mag(PI * i as f64 / 128.0));
        }
        if !(peak.is_finite() && loss.g.is_finite()) {
            return GsLoss {
                g: (-l0).exp().min(LOSS_MAX),
                ..GsLoss::default()
            };
        }
        if peak > LOSS_MAX {
            loss.g *= LOSS_MAX / peak;
        }
        loss
    };
    if n == 1 {
        return finish(0.0, 0.0, 0.0);
    }
    let a0 = onepole_a((-l0).exp(), w0, (-targets[n - 1].1).exp(), targets[n - 1].0);
    if n == 2 {
        return finish(a0, 0.0, 0.0);
    }
    // Unknowns: [a, c, b] (the first m of them; the rest 0). Residuals: the
    // targets after the first, then a penalty that keeps the filter's peak
    // below 1 (g is set by the first target, so a shape whose peak rises
    // above the fundamental's gain must be paid for here, not rescaled).
    let m = (n - 1).min(3);
    let rows = (n - 1).min(3) + 1;
    let resid = |p: &[f64; 3], r: &mut [f64; 4]| -> f64 {
        let (a, c, b) = (p[0], p[1], if m == 3 { p[2] } else { 0.0 });
        let lg = -l0 - GsLoss::shape(a, b, c, w0);
        let mut sse = 0.0;
        for i in 1..n.min(4) {
            let (w, l) = targets[i];
            r[i - 1] = -(lg + GsLoss::shape(a, b, c, w)) - l;
            sse += r[i - 1] * r[i - 1];
        }
        // No frequency may outlast the note: above the fundamental the loss
        // per pass is at least the fundamental's, and above the last target
        // at least 90 % of the last target's (the plateau); at DC (the only
        // mode below the fundamental, barely excited by a zero-mean pluck)
        // at least half the fundamental's; and the gain is below 1.
        let (w_last, l_last) = targets[n.min(4) - 1];
        let mut over = (lg + GsLoss::shape(a, b, c, 0.0) + 0.5 * l0).max(0.0);
        for k in 0..=48 {
            let w = w0 + (PI - w0) * k as f64 / 48.0;
            let lim = if w >= w_last { -0.9 * l_last } else { -l0 };
            over = over.max(lg + GsLoss::shape(a, b, c, w) - lim.min(LOSS_MAX.ln()));
        }
        r[rows - 1] = 10.0 * over;
        sse += r[rows - 1] * r[rows - 1];
        sse
    };
    let bounds = [(-0.99, 0.0), (-0.99, 0.99), (-0.99, 0.99)];
    let starts = [
        [a0, 0.0, 0.0],
        [-0.9, 0.0, -0.89],
        [-0.5, 0.5, 0.0],
        [-0.95, -0.3, -0.94],
        [-0.97, 0.1, -0.968],
        [-0.8, 0.0, -0.79],
        [a0, -0.5, 0.0],
    ];
    let mut best = ([a0, 0.0, 0.0], f64::INFINITY);
    let mut r = [0.0; 4];
    for start in starts {
        let mut p = start;
        for (k, (lo, hi)) in bounds.iter().enumerate() {
            p[k] = clampf(p[k], *lo, *hi);
        }
        let mut sse = resid(&p, &mut r);
        let mut lambda = 1e-3;
        for _ in 0..60 {
            if !sse.is_finite() || sse < 1e-24 {
                break;
            }
            // Jacobian (forward differences).
            let mut jac = [[0.0f64; 3]; 4];
            let mut r1 = [0.0; 4];
            for k in 0..m {
                let mut q = p;
                let h = 1e-7;
                q[k] += h;
                resid(&q, &mut r1);
                for i in 0..rows {
                    jac[i][k] = (r1[i] - r[i]) / h;
                }
            }
            let mut improved = false;
            for _ in 0..8 {
                let mut a = [[0.0f64; 3]; 3];
                let mut g = [0.0f64; 3];
                for i in 0..m {
                    for j in 0..m {
                        a[i][j] = (0..rows).map(|t| jac[t][i] * jac[t][j]).sum();
                    }
                    a[i][i] *= 1.0 + lambda;
                    a[i][i] += 1e-12;
                    g[i] = -(0..rows).map(|t| jac[t][i] * r[t]).sum::<f64>();
                }
                if !solve3(m, &mut a, &mut g) {
                    lambda *= 10.0;
                    continue;
                }
                let mut q = p;
                for k in 0..m {
                    q[k] = clampf(q[k] + g[k], bounds[k].0, bounds[k].1);
                }
                let mut rq = [0.0; 4];
                let s = resid(&q, &mut rq);
                if s.is_finite() && s < sse {
                    p = q;
                    r = rq;
                    sse = s;
                    lambda = (lambda * 0.3).max(1e-9);
                    improved = true;
                    break;
                }
                lambda *= 10.0;
            }
            if !improved {
                break;
            }
        }
        if sse < best.1 {
            best = (p, sse);
        }
    }
    let p = best.0;
    finish(p[0], if m == 3 { p[2] } else { 0.0 }, p[1])
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
fn design_second(f0: f64, b: f64, loss: &GsLoss, sr: f64) -> Option<DispDesign> {
    let g = |w: f64| TWO_PI * stiff_n(w * sr / TWO_PI, f0, b) - loss.lag(w);
    let gd = |w: f64| sr * stiff_dn_df(w * sr / TWO_PI, f0, b) - loss.gd(w);
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
        out.rho[i] = (-DISP_ETA * (edge - prev_edge).max(1e-6))
            .exp()
            .min(0.999_999);
        out.th[i] = mid;
        prev_edge = edge;
    }
    Some(out)
}

/// Equal first-order sections, `m` of them, coefficient by golden section.
fn design_first(f0: f64, b: f64, loss: &GsLoss, sr: f64, m: usize) -> (DispDesign, f64) {
    let cost = |a: f64| disp_cost(f0, b, sr, &|w: f64| m as f64 * ap1_lag(a, w) + loss.lag(w));
    let gr = 0.618_033_988_749_894_9;
    let (mut lo, mut hi) = (-0.95f64, 0.0f64);
    let mut c = hi - gr * (hi - lo);
    let mut d = lo + gr * (hi - lo);
    let (mut fc, mut fd) = (cost(c), cost(d));
    for _ in 0..28 {
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

/// Choose the dispersion design for an ideal fundamental `f0` and `B`
/// (see the module doc). Candidates whose delay at the fundamental would
/// exceed `max_delay` samples are skipped.
fn design_dispersion(f0: f64, b: f64, loss: &GsLoss, sr: f64, max_delay: f64) -> DispDesign {
    if b.is_nan() || b <= 1e-9 {
        return DispDesign::NONE;
    }
    let w1 = TWO_PI * f0 * (1.0 + b).sqrt() / sr;
    let fits = |d: &DispDesign| d.lag(w1) / w1 <= max_delay;
    let mut best = DispDesign::NONE;
    let mut best_j = disp_cost(f0, b, sr, &|w: f64| loss.lag(w));
    if let Some(d) = design_second(f0, b, loss, sr)
        && fits(&d)
    {
        let j = disp_cost(f0, b, sr, &|w: f64| d.lag(w) + loss.lag(w));
        if j < best_j {
            best = d;
            best_j = j;
        }
    }
    for m in 1..=MAX_FIRST {
        let (d, j) = design_first(f0, b, loss, sr, m);
        if j < best_j && fits(&d) {
            best = d;
            best_j = j;
        }
    }
    best
}

/// The partials (Hz) the loop plays with dispersion `disp` and loss `loss`
/// for a first partial at `freq`, the requested `b` setting the search
/// windows; written into `out`, returns how many (all below
/// `min(top, 0.45 fs)`). The delay and the Thiran filter are fixed as the
/// node fixes them at the pluck, and the Thiran filter's own phase is
/// included, so this is what the loop plays.
#[allow(clippy::too_many_arguments)]
fn loop_partials(
    disp: &DispDesign,
    loss: &GsLoss,
    freq: f64,
    b: f64,
    sr: f64,
    top: f64,
    pref: Option<usize>,
    out: &mut [f64],
) -> usize {
    let w1 = TWO_PI * freq / sr;
    let len = (sr / GS_MIN_FREQ as f64) as usize + 8;
    let (n_int, th) = fix_delay(disp.lag(w1) + loss.lag(w1), w1, len, pref);
    let lag = |w: f64| disp.lag(w) + loss.lag(w) + thiran_lag(&th, w);
    let d = n_int as f64;
    let f0 = freq / (1.0 + b).sqrt();
    let top = top.min(0.45 * sr);
    let mut count = 0;
    for (k, o) in out.iter_mut().enumerate() {
        let nf = (k + 1) as f64;
        let target = nf * f0 * (1.0 + b * nf * nf).sqrt();
        if target > top {
            break;
        }
        let wt = TWO_PI * target / sr;
        let w = bisect_n(
            0.6 * wt,
            (1.6 * wt).min(PI),
            TWO_PI * nf,
            |w| d * w + lag(w),
            40,
        );
        *o = w * sr / TWO_PI;
        count += 1;
    }
    count
}

/// `B` fitted to partials `(n, f_n)`, n = 1.., in closed form: `(f_n / n)²`
/// is linear in `n²` with slope `f0² B`.
fn fit_b(parts: &[f64]) -> Option<f64> {
    let n = parts.len();
    if n < 4 {
        return None;
    }
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (k, f) in parts.iter().enumerate() {
        let nf = (k + 1) as f64;
        let x = nf * nf;
        let y = (f / nf) * (f / nf);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let nn = n as f64;
    let slope = (nn * sxy - sx * sy) / (nn * sxx - sx * sx);
    let icpt = (sy - slope * sx) / nn;
    let b = slope / icpt;
    if b.is_finite() { Some(b) } else { None }
}

/// The dispersion design for a note with its first partial at `freq` and
/// inharmonicity `b`, pre-warped so the design's fitted `B` matches `b`
/// (see the module doc). Returns the design, the `B'` it was made for, and
/// the integer delay the pre-warp held (the node uses the same).
fn note_dispersion(sr: f64, freq: f64, b: f64, loss: &GsLoss) -> (DispDesign, f64, Option<usize>) {
    let w1 = TWO_PI * freq / sr;
    let room = sr / freq - 1.0 - THIRAN_LO - MIN_READ - loss.lag(w1) / w1;
    let design = |bb: f64| design_dispersion(freq / (1.0 + bb).sqrt(), bb, loss, sr, room.max(0.0));
    let first = design(b);
    if b.is_nan() || b <= 1e-9 {
        return (first, b, None);
    }
    let len = (sr / GS_MIN_FREQ as f64) as usize + 8;
    let pref = Some(fix_delay(first.lag(w1) + loss.lag(w1), w1, len, None).0);
    let mut parts = [0.0f64; PREWARP_PARTIALS];
    if first.kind == DispKind::First {
        // Equal first-order sections: solve for the coefficient whose fitted
        // B is the requested one (bisection; more negative = more stretch),
        // keeping the section count. The coefficient search of the design
        // has local minima, so iterating B' through it can jump past the
        // target.
        let fit_at = |a1: f64, parts: &mut [f64; PREWARP_PARTIALS]| -> f64 {
            let mut d = first;
            d.a1 = a1;
            let n = loop_partials(&d, loss, freq, b, sr, PREWARP_TOP_HZ, pref, parts);
            fit_b(&parts[..n]).map_or(f64::NAN, |bf| (bf / b).ln())
        };
        let (mut lo, mut hi) = (-0.95f64, 0.0f64);
        let (flo, fhi) = (fit_at(lo, &mut parts), fit_at(hi, &mut parts));
        if flo.is_finite() && fhi.is_finite() && flo > 0.0 && fhi < 0.0 {
            for _ in 0..24 {
                let mid = 0.5 * (lo + hi);
                let fm = fit_at(mid, &mut parts);
                if !fm.is_finite() {
                    break;
                }
                if fm > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let mut d = first;
            d.a1 = 0.5 * (lo + hi);
            if d.lag(w1) / w1 <= room.max(0.0) {
                return (d, b, pref);
            }
        }
        return (first, b, pref);
    }
    let mut best = (first, b, f64::INFINITY);
    let mut bp = b;
    let mut d = first;
    for _ in 0..6 {
        let n = loop_partials(&d, loss, freq, b, sr, PREWARP_TOP_HZ, None, &mut parts);
        let Some(bf) = fit_b(&parts[..n]) else {
            break;
        };
        if bf <= 0.0 {
            break;
        }
        let err = (bf / b).ln().abs();
        if err < best.2 {
            best = (d, bp, err);
        }
        if err < 0.01 {
            break;
        }
        bp = clampf(bp * b / bf, 0.2 * b, 5.0 * b);
        d = design(bp);
    }
    if best.2.is_infinite() {
        return (first, b, None);
    }
    (best.0, best.1, None)
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

/// The integer delay and Thiran coefficients that make the loop's phase at
/// `w1` exactly 2π, given the other filters' lag there (see the module doc).
fn fix_delay(lag_others: f64, w1: f64, len: usize, pref: Option<usize>) -> (usize, [f64; 3]) {
    let total = (TWO_PI - lag_others) / w1;
    let n_int = match pref {
        // Keep a given integer delay while the fraction stays in 2.05..4.5
        // (the pre-warp holds it so the design's partials move continuously).
        Some(p) if (2.05..=4.5).contains(&(total - p as f64)) && p as f64 >= MIN_READ => p,
        _ => clampf((total - 3.0).round(), MIN_READ, (len - 4) as f64) as usize,
    };
    let d0 = clampf(total - n_int as f64, 2.01, 64.0);
    let a = thiran3(d0);
    let err = thiran_lag(&a, w1) / w1 - d0;
    (n_int, thiran3(clampf(d0 - err, 2.01, 64.0)))
}

/// A note's design, as the node makes it at the pluck (without per-note
/// variation): for tests and for a fitting tool.
#[derive(Clone, Debug)]
pub struct GsNoteDesign {
    /// The string, 0-based (0 = high E).
    pub string: usize,
    /// Requested `B` at this fret.
    pub b: f64,
    /// The `B'` the dispersion cascade was designed for (the pre-warp).
    pub b_design: f64,
    /// Ideal fundamental `freq / sqrt(1 + B)`.
    pub f0: f64,
    /// Loss filters of V and H.
    pub loss: [GsLoss; 2],
    /// Pitch of H's first partial.
    pub h_freq: f64,
    /// Partial frequencies V's loop plays (up to 30, below 0.45 fs).
    pub partials: Vec<f64>,
}

/// The loss targets of a note: `(w, λ)` at the fundamental and the anchors
/// above it (see the module doc).
#[allow(clippy::too_many_arguments)]
fn loss_targets(
    set: &GsStringSet,
    s: usize,
    k: f64,
    hv: f64,
    freq: f64,
    f0i: f64,
    b: f64,
    sr: f64,
    out: &mut [(f64, f64); GS_ANCHORS + 2],
) -> usize {
    let lam = |f: f64| set.sigma(s, k, hv, f) * stiff_dn_df(f, f0i, b);
    let w = |f: f64| TWO_PI * f / sr;
    let mut n = 0;
    out[n] = (w(freq), lam(freq));
    n += 1;
    for &fa in &set.anchor_hz {
        if fa > 1.25 * freq && fa < 0.45 * sr && fa > set.strings[s].open_hz * 1.01 {
            out[n] = (w(fa), lam(fa));
            n += 1;
        }
    }
    if n == 1 {
        let fb = (2.5 * freq).min(0.4 * sr);
        if fb > 1.05 * freq {
            out[n] = (w(fb), lam(fb));
            n += 1;
        }
    }
    n
}

/// The design `guitarString` makes for a note at `freq` on `set` (`string`
/// as the port: 1..6 or -1), with `split` and the pluck at their string-set
/// values and no per-note variation.
pub fn gs_note_design(sr: f64, set: &GsStringSet, freq: f64, string: f32) -> GsNoteDesign {
    let s = set.choose(freq, string);
    let st = set.strings[s];
    let k = freq / st.open_hz;
    let b = st.b * k * k;
    let f0 = freq / (1.0 + b).sqrt();
    let mut tg = [(0.0, 0.0); GS_ANCHORS + 2];
    let mut loss = [GsLoss::default(); 2];
    for (p, l) in loss.iter_mut().enumerate() {
        let hv = if p == 0 { 1.0 } else { st.hv };
        let n = loss_targets(set, s, k, hv, freq, f0, b, sr, &mut tg);
        *l = design_loss(&tg[..n]);
    }
    let (disp, b_design, pref) = note_dispersion(sr, freq, b, &loss[0]);
    let mut parts = [0.0f64; PREWARP_PARTIALS];
    let n = loop_partials(&disp, &loss[0], freq, b, sr, 0.45 * sr, pref, &mut parts);
    GsNoteDesign {
        string: s,
        b,
        b_design,
        f0,
        loss,
        h_freq: freq * (-st.split_cents / 1200.0).exp2(),
        partials: parts[..n].to_vec(),
    }
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
    /// Pitch of this polarisation at the pluck (its first partial), Hz.
    freq: f64,
    /// Integer delay at the pluck.
    n_int: usize,
    /// Read point now (samples), and its integer part and fraction.
    read: f64,
    read_i: usize,
    read_mu: f64,
    thiran: [f64; 3],
    th_s: [f64; 3],
    loss: GsLoss,
    loss_s: f64,
    shelf_x: f64,
    shelf_s: f64,
    /// Damping filter `damp_b / (1 + damp_a z^-1)`.
    damp_b: f64,
    damp_a: f64,
    damp_s: f64,
    disp: [Biquad; GS_MAX_SECTIONS],
    first_s: [f64; MAX_FIRST],
    /// Excitation: amplitude, period (samples), comb lag `m`, pulse share
    /// `β'`, dynamic-lowpass coefficient and state.
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
            n_int: 2,
            read: 2.0,
            read_i: 2,
            read_mu: 0.0,
            thiran: thiran3(3.0),
            th_s: [0.0; 3],
            loss: GsLoss::default(),
            loss_s: 0.0,
            shelf_x: 0.0,
            shelf_s: 0.0,
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
    fn clear_disp(&mut self) {
        for q in &mut self.disp {
            q.s1 = 0.0;
            q.s2 = 0.0;
        }
        self.first_s = [0.0; MAX_FIRST];
    }

    fn clear_state(&mut self) {
        self.th_s = [0.0; 3];
        self.loss_s = 0.0;
        self.shelf_x = 0.0;
        self.shelf_s = 0.0;
        self.damp_s = 0.0;
        self.clear_disp();
        self.dyn_s = 0.0;
    }

    /// Read the delay line `line` (write index `w`) at the read point.
    #[inline]
    fn read_line(&self, line: &[f32], w: usize) -> f64 {
        let len = line.len();
        let at = |d: usize| line[(w + len - d) % len] as f64;
        let i = self.read_i;
        if self.read_mu == 0.0 {
            return at(i);
        }
        // Six-point (fifth-order) Lagrange through delays i-2 ..= i+3 at
        // i + μ: weight of offset k is Π_{j≠k} (μ − j) / (k − j).
        let mu = self.read_mu;
        let mut y = 0.0;
        for k in -2i32..=3 {
            let mut h = 1.0;
            for j in -2i32..=3 {
                if j != k {
                    h *= (mu - j as f64) / (k - j) as f64;
                }
            }
            y += h * at((i as i32 + k) as usize);
        }
        y
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
        // Loss: one-pole, then the shelf.
        let l = &self.loss;
        let y = l.g * (1.0 + l.a) * y - l.a * self.loss_s;
        self.loss_s = y;
        let k = (1.0 + l.c) / (1.0 + l.b);
        let s = k * (y + l.b * self.shelf_x) - l.c * self.shelf_s;
        self.shelf_x = y;
        self.shelf_s = s;
        // Damping one-pole.
        let y = self.damp_b * s - self.damp_a * self.damp_s;
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
    set: GsStringSet,
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
    rel: f64,
    /// Damping: current, ramp coefficient; release flag.
    d: f64,
    d_coef: f64,
    released: bool,
    follow: f64,
    follow_k: f64,
    configured: bool,
    plucked: bool,
    done: bool,
    /// Samples a pending pluck still waits for a positive `freq`.
    pending: usize,
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
    /// An unbound node: the default string set, and a unit-impulse residual.
    pub fn new() -> Self {
        Self::with_set(GsStringSet::default())
    }

    /// An unbound node on string set `set`.
    pub fn with_set(set: GsStringSet) -> Self {
        GuitarString {
            sample_rate: 44100.0,
            set,
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
            rel: 0.15,
            d: 0.0,
            d_coef: 0.0,
            released: false,
            follow: 0.0,
            follow_k: 0.0,
            configured: false,
            plucked: false,
            done: false,
            pending: 0,
            prev_gate: 0.0,
            last_freq: f64::NAN,
            voice_seed: 0,
            plucks: 0,
        }
    }

    /// A node bound to `table`: the string set and the residual come from
    /// it (see the module doc); either missing keeps its default.
    /// Resampling the residual to the engine rate happens in `init`.
    pub fn with_table(table: Arc<CoeffTable>) -> Self {
        let mut node = Self::with_set(GsStringSet::from_table(&table).unwrap_or_default());
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

    /// The node's string set.
    pub fn string_set(&self) -> &GsStringSet {
        &self.set
    }

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
        (self.sample_rate * MAX_FREQ_FRAC).max(GS_MIN_FREQ as f64)
    }

    /// Design both loops for a note at `freq` and start the excitation.
    /// Inputs at sample `i`. With `pluck` false (the first sample with no
    /// pluck yet) only the geometry is set, so `exc` alone can play.
    fn configure(&mut self, inputs: &[Option<&AudioBuffer>], i: usize, freq: f64, pluck: bool) {
        let sr = self.sample_rate;
        let rd = |k: usize| Self::read(inputs, k, i) as f64;
        let set = self.set;
        let s = set.choose(freq, Self::read(inputs, P_STRING, i));
        let st = set.strings[s];

        // Per-note variation.
        self.plucks = self.plucks.wrapping_add(1);
        let seed_bits = (Self::read(inputs, P_SEED, i)).to_bits() as u64;
        let mut mix = Mix(((self.voice_seed as u64) << 32)
            ^ seed_bits.wrapping_mul(0x2545_F491_4F6C_DD1D)
            ^ self.plucks.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let vary = clampf(rd(P_VARY), 0.0, 1.0);
        let (j_pos, j_angle, j_level, j_split) = (
            mix.bipolar() * 10.0 * vary,
            mix.bipolar() * 10.0 * vary,
            mix.bipolar() * vary,
            mix.bipolar() * 0.5 * vary,
        );

        let scale = set.scale_mm;
        let k = freq / st.open_hz;
        let b = st.b * k * k;
        let len_f = scale / k;
        let pos = clampf(rd(P_POS) + j_pos, 1.0, len_f - 1.0);
        let beta = clampf(pos / len_f, 0.005, 0.95);
        let angle = clampf(rd(P_ANGLE) + j_angle, -90.0, 90.0).to_radians();
        let split = clampf(st.split_cents + j_split, -50.0, 50.0);
        self.exc_w = [angle.cos(), angle.sin()];

        let vel = clampf(rd(P_VEL), 0.0, 1.0);
        let hard = clampf(rd(P_HARD), 0.0, 1.0);
        let width = clampf(rd(P_WIDTH), 0.1, 40.0);
        let level = vel * (j_level / 20.0 * core::f64::consts::LN_10).exp();
        let w_eff = width * (1.25 - 0.5 * vel) * (1.0 - 0.5 * hard);
        let c = 2.0 * scale * 1e-3 * st.open_hz;
        let ramp = clampf(w_eff * 1e-3 / c * sr, 1.0, MAX_RAMP as f64);
        let bright = vel * (0.5 + 0.5 * hard);
        let f_dyn = clampf(
            set.vel_corner_hz * bright.powf(set.vel_corner_exp),
            1.0,
            0.45 * sr,
        );
        let dyn_r = (-TWO_PI * f_dyn / sr).exp();

        let f0i = freq / (1.0 + b).sqrt();
        let mut tg = [(0.0, 0.0); GS_ANCHORS + 2];
        for (p, pol) in self.pols.iter_mut().enumerate() {
            let pf = if p == 0 {
                freq
            } else {
                freq * (-split / 1200.0).exp2()
            };
            pol.freq = pf;
            let hv = if p == 0 { 1.0 } else { st.hv };
            let n = loss_targets(&set, s, k, hv, freq, f0i, b, sr, &mut tg);
            pol.loss = design_loss(&tg[..n]);
            let period = sr / pf;
            let m = (beta * period).round().max(1.0) as usize;
            let m = m.min((period as usize).max(2) - 1);
            pol.period = period;
            pol.m = m;
            pol.beta = m as f64 / period;
            if pluck {
                pol.amp = level * if p == 0 { angle.cos() } else { angle.sin() };
                pol.dyn_r = dyn_r;
            }
        }

        // Dispersion, designed on V with V's loss phase; shared by H. State
        // is cleared when the cascade's form changes.
        let (disp, _, pref) = note_dispersion(sr, freq, b, &self.pols[0].loss);
        let changed = disp.kind != self.disp.kind || disp.n != self.disp.n;
        self.disp = disp;
        for pol in &mut self.pols {
            if changed {
                pol.clear_disp();
            }
            if self.disp.kind == DispKind::Second {
                for (k, q) in pol.disp.iter_mut().enumerate().take(self.disp.n) {
                    q.a1 = -2.0 * self.disp.rho[k] * self.disp.th[k].cos();
                    q.a2 = self.disp.rho[k] * self.disp.rho[k];
                }
            }
        }

        // Fix the integer delay and the Thiran fraction at this pitch.
        self.rel = clampf(rd(P_REL), 0.005, 3.0);
        self.set_damping(freq);
        for (p, pol) in self.pols.iter_mut().enumerate() {
            let w1 = TWO_PI * pol.freq / sr;
            let lag = pol.loss.lag(w1) + lp_lag(pol.damp_a, w1) + self.disp.lag(w1);
            let (n_int, th) = fix_delay(lag, w1, self.len, if p == 0 { pref } else { None });
            pol.thiran = th;
            pol.n_int = n_int;
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

    /// Re-solve both loops' read points for the current pitch and damping
    /// (the Thiran filter stays as fixed at the pluck).
    fn retune(&mut self, freq: f64) {
        let sr = self.sample_rate;
        let split_ratio = if self.pols[0].freq > 0.0 {
            self.pols[1].freq / self.pols[0].freq
        } else {
            1.0
        };
        let hi = (self.len - 4) as f64;
        for (p, pol) in self.pols.iter_mut().enumerate() {
            let pf = if p == 0 { freq } else { freq * split_ratio };
            let w1 = TWO_PI * pf / sr;
            let lag = pol.loss.lag(w1)
                + lp_lag(pol.damp_a, w1)
                + self.disp.lag(w1)
                + thiran_lag(&pol.thiran, w1);
            let r = clampf((TWO_PI - lag) / w1, MIN_READ, hi);
            // A held note plays the plain tap.
            let r = if (r - pol.n_int as f64).abs() < 1e-6 {
                pol.n_int as f64
            } else {
                r
            };
            pol.read = r;
            pol.read_i = r.floor() as usize;
            pol.read_mu = r - r.floor();
        }
    }

    /// Update the damping filters for the current `d` and `rel`.
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
        self.sample_rate = (context.sample_rate as f64).max(1000.0);
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
            let n = n.clamp(1, ((GS_MAX_RESIDUAL_S * sr) as usize).max(1));
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
        self.pending = 0;
        self.prev_gate = 0.0;
        self.last_freq = f64::NAN;
        self.plucks = 0;
    }

    fn reseed_noise(&mut self, seed: u32) {
        self.voice_seed = seed;
    }

    fn is_done(&self) -> bool {
        self.done && self.pending == 0
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
        let rel_buf = inputs.get(P_REL).copied().flatten();

        for (i, o) in mono[..block].iter_mut().enumerate() {
            let raw_freq = Self::read_freq(inputs, i);
            let gate = Self::read(inputs, P_GATE, i);
            let rising = gate > 0.0 && self.prev_gate <= 0.0;
            let falling = gate <= 0.0 && self.prev_gate > 0.0;
            self.prev_gate = gate;
            if falling {
                self.pending = 0;
            }

            if raw_freq <= 0.0 {
                // Silence: a sounding string stops at once; a rising gate
                // waits for a positive freq.
                if rising {
                    self.pending = (PENDING_S * self.sample_rate) as usize;
                } else if self.pending > 0 {
                    self.pending -= 1;
                }
                if !self.done {
                    self.plucked = true;
                    self.silence();
                }
                *o = 0.0;
                continue;
            }
            let freq = clampf(raw_freq, GS_MIN_FREQ as f64, self.max_freq());

            let pluck_now = rising || (self.pending > 0 && gate > 0.0);
            if pluck_now {
                self.pending = 0;
                self.done = false;
                self.configure(inputs, i, freq, true);
            } else if !self.configured {
                self.configure(inputs, i, freq, false);
            }
            if falling {
                self.released = true;
            }

            // Damping ramp and `rel`.
            let damp = clampf(
                finite_or(read_input(damp_buf, 0, i, 0.0), 0.0) as f64,
                0.0,
                1.0,
            );
            let rel = clampf(
                finite_or(read_input(rel_buf, 0, i, 0.15), 0.15) as f64,
                0.005,
                3.0,
            );
            let target = if self.released { 1.0 } else { damp };
            let mut moved = false;
            if rel != self.rel {
                self.rel = rel;
                moved = self.d != 0.0;
            }
            if self.d != target {
                let nd = self.d + (target - self.d) * self.d_coef;
                self.d = if (nd - target).abs() < 1e-6 {
                    target
                } else {
                    nd
                };
                moved = true;
            }
            if moved {
                self.set_damping(freq);
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
                let read = pol.read_line(&self.lines[p], w);
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
                OUT_GAIN * (x[0] + self.set.hmix * x[1])
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

/// Register the table-bound form of `guitarString` (the string set and the
/// commuted residual read from a bound table). The bare form is registered
/// with the other built-ins; a node with a `table_bindings` entry resolves
/// to this one.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ProcessContext {
        ProcessContext::new(44_100.0, 64)
    }

    /// Run `n` blocks at a constant `freq`, `string` and gate pattern.
    fn run(node: &mut GuitarString, freq: f32, string: f32, gate: f32, n: usize) {
        let c = ctx();
        let mk = |v: f32| {
            let mut b = AudioBuffer::new(1, 64);
            b.channel_mut(0).samples_mut().fill(v);
            b
        };
        let (fb, sb, gb) = (mk(freq), mk(string), mk(gate));
        let mut inputs: Vec<Option<&AudioBuffer>> = alloc::vec![None; GS_N_PORTS];
        inputs[P_FREQ] = Some(&fb);
        inputs[P_STRING] = Some(&sb);
        inputs[P_GATE] = Some(&gb);
        let mut o = AudioBuffer::new(1, 64);
        for _ in 0..n {
            node.process(&c, &inputs, &mut o);
        }
    }

    /// The dispersion cascade's state is cleared when a re-pluck changes the
    /// design's kind or section count: first-order state left from an
    /// earlier note must not come back into the loop after a note that used
    /// second-order sections.
    #[test]
    fn dispersion_state_is_cleared_when_the_design_changes() {
        let mut node = GuitarString::new();
        node.init(&ctx());
        // A first-order design (open high E), then a second-order one (low E),
        // then the first-order note again, each a re-pluck of the ringing node.
        run(&mut node, 329.627_56, 1.0, 1.0, 20);
        assert_eq!(node.disp.kind, DispKind::First);
        assert!(node.pols[0].first_s.iter().any(|&x| x != 0.0));
        run(&mut node, 82.406_89, 6.0, 0.0, 1);
        run(&mut node, 82.406_89, 6.0, 1.0, 1);
        assert_eq!(node.disp.kind, DispKind::Second);
        // Clearing happened at the change: the first-order state is zero and
        // stays zero while second-order sections run.
        assert!(node.pols[0].first_s.iter().all(|&x| x == 0.0));
        run(&mut node, 82.406_89, 6.0, 1.0, 20);
        assert!(node.pols[0].first_s.iter().all(|&x| x == 0.0));
        let sec: f64 = node.pols[0]
            .disp
            .iter()
            .map(|q| q.s1.abs() + q.s2.abs())
            .sum();
        assert!(sec > 0.0);
        // Back to first order: the second-order state is cleared.
        run(&mut node, 329.627_56, 1.0, 0.0, 1);
        let c = ctx();
        let mk = |v: f32| {
            let mut b = AudioBuffer::new(1, 1);
            b.channel_mut(0).samples_mut().fill(v);
            b
        };
        let (fb, sb, gb) = (mk(329.627_56), mk(1.0), mk(1.0));
        let mut inputs: Vec<Option<&AudioBuffer>> = alloc::vec![None; GS_N_PORTS];
        inputs[P_FREQ] = Some(&fb);
        inputs[P_STRING] = Some(&sb);
        inputs[P_GATE] = Some(&gb);
        let mut o = AudioBuffer::new(1, 1);
        node.process(&c, &inputs, &mut o);
        assert_eq!(node.disp.kind, DispKind::First);
        assert!(node.pols[0].disp.iter().all(|q| q.s1 == 0.0 && q.s2 == 0.0));
    }

    /// `clampf` never panics, whatever its bounds.
    #[test]
    fn clampf_is_total() {
        assert_eq!(clampf(5.0, 1.0, -3.0), 1.0);
        assert_eq!(clampf(f64::NAN, 1.0, 2.0), 1.0);
        assert_eq!(clampf(1.5, 1.0, 2.0), 1.5);
        assert_eq!(clampf(9.0, 1.0, 2.0), 2.0);
    }
}
