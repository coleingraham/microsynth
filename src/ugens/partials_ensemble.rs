//! `partialsEnsemble`: a `partialsBlend` bank that sounds like a section of
//! players rather than one, at about the cost of one voice, with a stereo
//! output.
//!
//! **What a section does to a partial.** `N` players on one note each sound
//! partial `m` at a slightly different frequency (their tuning differs by a
//! few cents, and each has their own vibrato) and at an unrelated phase. With
//! the section's mean frequency `f_m` taken out, the sum is
//!
//! ```text
//! s_m(t) = Re( z_m(t) · e^{i 2π f_m t} ),   z_m(t) = N^{-1/2} Σ_j e^{i ψ_jm(t)}
//! ```
//!
//! a single sinusoid at `f_m` times a slowly moving complex envelope `z_m`.
//! Its power `|z_m|²` has mean 1 (the sum is normalised to the power of one
//! player) and, for independent phases, variance `1 − 1/N`: one player does
//! not fluctuate, and from a handful of players on the level of each partial
//! moves nearly as much as it ever will (`N → ∞` gives a circular complex
//! Gaussian `z`, a Rayleigh amplitude, power fluctuating with standard
//! deviation equal to its mean). What does keep changing with `N` is the
//! *shape* of the movement: for small `N` it is a beating of a few lines, for
//! large `N` a smooth random wander. How *fast* it moves is set by the spread
//! of the players' frequencies at that partial, which is the spread in cents
//! times `f_m`: high partials wander faster than low ones, as in a real
//! section.
//!
//! This bank computes `z_m` at a control rate (every `ctrl` samples, linear
//! between) and renders each partial as ONE sinusoid modulated by it, so the
//! sine count is that of one voice whatever `N`. `z_m = α + β g_m(t)`, `g_m` a
//! unit-power circular complex Gaussian process (seeded white noise through
//! two one-pole low-passes per component), `α² + β² = 1`. The coherent part
//! `α = N^{-1/4}` is what makes "`N` players" mean something: it gives
//! `Var |z|² = 1 − 1/N`, the exact second moment of `N` equal players with
//! independent phases (a Rician envelope standing in for the sum of `N`
//! phasors). The low-pass is set so `g_m`'s correlation time equals that of
//! `N → ∞` players whose frequencies at partial `m` are Gaussian with
//! standard deviation `σ_m = f_m · σ_c · ln 2 / 1200`, where
//! `σ_c² = spread² + (vib²/2)(1 − 1/N)`: the static detune plus the part of
//! the players' vibrato that is not common to the section (`vib` is the
//! per-player vibrato depth in cents, peak; the common part, of RMS `1/√N` of
//! one player's, is the def's own vibrato's business). Gaussian frequencies
//! give `|r(τ)| = exp(−2π² σ_m² τ²)`, `1/e` at `τ_e = 1/(π √2 σ_m)`; two
//! one-poles of rate `a` give `r(τ) = (1 + aτ) e^{−aτ}`, `1/e` at
//! `aτ = 2.1462`, so `a = 2.1462 · π √2 · σ_m`.
//!
//! **Known properties.** The two-pole spectrum has heavier tails than the
//! Gaussian line a large section converges to (its RMS width is 1.5 σ_m when
//! the correlation time is matched). The players' vibrato enters only as
//! bandwidth, so the low partials, which carry most of the level, wander
//! faster than in a section whose vibratos keep a fixed relationship over a
//! second: on held four-part chords twelve "players" here move 2.8–7.0 dB
//! against 2.7–4.9 dB for twelve detuned copies of a voice. Each partial is
//! an independent process, whereas a real player's detune is shared by all
//! of their partials.
//!
//! **Stereo.** The output has two channels, stereo by design. `width` 0 gives
//! the same `z` to both (a centred section, both channels equal); the right
//! channel's random part is `cos(wπ/2) g_L + sin(wπ/2) g_R` with `g_R` an
//! independent process, so `width` 1 gives two uncorrelated desks sharing
//! the coherent part. Both channels come from one pass over one state (there
//! is no per-channel copy of the state to drift apart), and the inputs are
//! control signals read from their first channel. The noise bands are shared
//! and identical in both channels.
//!
//! **`spread` 0 with `players` > 1.** The default `spread` is 0 and the
//! default `vib` is 0: players with no detune and no vibrato of their own.
//! The model then holds each partial at one random amplitude and phase for
//! the whole note (a sum of phasors that never move relative to one another):
//! no movement within a note, but a different draw per note, so a scale's
//! note levels scatter (a 6.6 dB range over 12 notes at 12 players in the
//! tests) and two notes never match.
//! That is what `N` identical, perfectly steady players at random phases
//! would give, and it is kept as the default so `players` > 1 with no detune
//! stays honest to the model. A section that moves needs a spread: the
//! catalog's string sections use `spread` 5 (cents) with `vib` set to the
//! instrument's own vibrato depth (19–27 cents), `onset` 0.013, `width` 0.7,
//! at `players` 6 (small) and 10⁶ (large).
//!
//! **Onset.** `onset` (seconds, ≤ 50 ms) smears the attack: the gains are
//! averaged over four taps delayed `0, onset/3, 2·onset/3, onset`, as the
//! players' attacks would be (the def's own envelope, outside the bank, is
//! not smeared). The tap ring is allocated once, at `init`, for the longest
//! onset: `(⌈0.05 · sample_rate⌉ + 2) × 4` floats per voice, 35 kB at
//! 44.1 kHz and 77 kB at 96 kHz, whether or not `onset` is used (the rest of
//! the bank's state is about 64 bytes per partial, 5 kB for 80 partials).
//!
//! **Out-of-range inputs.** A non-finite gain plays as 0 for that sample, and
//! `freq` ≤ 0 or non-finite silences the partials for that sample (their
//! phases hold), as it does in `partialsBlend`; the noise bands behave as
//! `partialsBlend`'s.
//!
//! **Phases.** `phase` (default 1) scatters the partials' carrier phases at
//! note start: partial `m` starts at `phase · u_m` cycles, `u_m` uniform in
//! `[0, 1]` (at 0 every partial starts at phase 0, as `partialsBlend`'s do).
//! Every random draw here (the carrier phases and the Gaussian processes)
//! comes from streams seeded by the voice's seed (the one
//! [`UGen::reseed_noise`] hands the voice), so a render is reproducible.
//!
//! **Control-rate amplitudes.** Each partial's amplitude (the blended
//! coefficient × main-lobe gain × Nyquist fade) is computed per tick too and
//! interpolated, where `partialsBlend` computes it per sample. With `players`
//! 1, `spread` 0, `vib` 0, `onset` 0, `phase` 0 both channels equal
//! `partialsBlend`'s output to within the interpolation error (a test checks
//! this).
//!
//! **Sine.** The carriers use a short single-precision polynomial for sine
//! and cosine of a phase in turns ([`sin_cos_turns_f32`], max error about
//! 3e-7, only `+`, `*` and `floor`, so wasm and native agree).
//!
//! **Done rule.** A source: it never reports done, and like the other banks
//! it stops silence (a def frees its voice through the envelope that
//! multiplies the bank's output).
//!
//! Ports: `freq`, `gain0`..`gain3` (as `partialsBlend`), `players` (1),
//! `spread` (cents, 0), `vib` (cents peak, 0), `onset` (s, 0), `width`
//! (0..1, 0), `ctrl` (samples, 16, 1..64), `phase` (1). All but `freq` and the
//! gains are read once, at the voice's first sample.

use crate::buffer::{AudioBuffer, read_input};
use crate::coeff_table::CoeffTable;
use crate::context::ProcessContext;
use crate::context::Rate;
use crate::dsl::compiler::{TableUGenFactory, UGenRegistry};
use crate::node::{InputSpec, OutputSpec, UGen, UGenCategory, UGenSpec};
use crate::ugens::partials::{DEFAULT_FREQ_HZ, UNIT_VARIANCE_BIPOLAR_SCALE, lerp, nyquist_fade};
use crate::ugens::partials_blend::{MAX_BLEND_CHANNELS, PartialsBlend};
use crate::ugens::rng::Rng;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Longest control period, in samples.
pub const MAX_CTRL: usize = 64;
/// Longest onset smear, in seconds.
pub const MAX_ONSET_S: f32 = 0.05;
/// Taps of the onset smear.
const ONSET_TAPS: usize = 4;

const SALT_GAUSS_L: u32 = 0x6A55_1A01;
const SALT_GAUSS_R: u32 = 0x6A55_1A02;
/// Salt that separates the phase-scatter stream from the other streams drawn
/// from the same voice seed.
const PHASE_SCATTER_SALT: u32 = 0x5CA7_7E12;

/// `aτ` at which `(1 + aτ) e^{−aτ} = 1/e`.
const TWO_POLE_1E: f32 = 2.146_193;
const LN2_OVER_1200: f32 = core::f32::consts::LN_2 / 1200.0;

const INPUT_NAMES: [&str; 12] = [
    "freq", "gain0", "gain1", "gain2", "gain3", "players", "spread", "vib", "onset", "width",
    "ctrl", "phase",
];
static INPUTS: [InputSpec; 12] = {
    const fn port(name: &'static str) -> InputSpec {
        InputSpec {
            name,
            rate: Rate::Audio,
            required: false,
        }
    }
    [
        port(INPUT_NAMES[0]),
        port(INPUT_NAMES[1]),
        port(INPUT_NAMES[2]),
        port(INPUT_NAMES[3]),
        port(INPUT_NAMES[4]),
        port(INPUT_NAMES[5]),
        port(INPUT_NAMES[6]),
        port(INPUT_NAMES[7]),
        port(INPUT_NAMES[8]),
        port(INPUT_NAMES[9]),
        port(INPUT_NAMES[10]),
        port(INPUT_NAMES[11]),
    ]
};
const P_PLAYERS: usize = 5;
const P_SPREAD: usize = 6;
const P_VIB: usize = 7;
const P_ONSET: usize = 8;
const P_WIDTH: usize = 9;
const P_CTRL: usize = 10;
const P_PHASE: usize = 11;

/// `(sin 2πt, cos 2πt)` in single precision for a finite phase `t` in turns.
///
/// Quadrant from `round(4t)`, then degree-7 (sine) and degree-8 (cosine)
/// Taylor polynomials on `|x| ≤ π/4`: truncation below 3.2e-6 and 2.5e-7
/// there before rounding (the measured maximum error is in this module's
/// tests). Only `+`, `*` and `floor`, so every target gives the same bits.
#[inline(always)]
pub fn sin_cos_turns_f32(t: f32) -> (f32, f32) {
    let t = t - t.floor();
    let q = (4.0 * t + 0.5).floor();
    let x = core::f32::consts::TAU * (t - 0.25 * q);
    let z = x * x;
    let s = x * (1.0 + z * (-1.0 / 6.0 + z * (1.0 / 120.0 + z * (-1.0 / 5040.0))));
    let c = 1.0 + z * (-0.5 + z * (1.0 / 24.0 + z * (-1.0 / 720.0 + z * (1.0 / 40320.0))));
    match (q as i32) & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Set each partial's starting phase to `scatter` times a uniform draw over
/// one cycle, from a stream seeded by the voice's seed.
fn scatter_phases(phase: &mut [f32], seed: u32, scatter: f32) {
    let mut rng = Rng::new(seed ^ PHASE_SCATTER_SALT);
    for p in phase.iter_mut() {
        let u = 0.5 * (rng.next_bipolar() + 1.0); // [0, 1]
        let v = scatter * u;
        *p = v - v.floor();
    }
}

/// Unit-variance, approximately Gaussian draw (sum of four uniforms).
#[inline]
fn gauss(rng: &mut Rng) -> f32 {
    // each uniform on [-1, 1] has variance 1/3; four sum to 4/3
    let s = rng.next_bipolar() + rng.next_bipolar() + rng.next_bipolar() + rng.next_bipolar();
    s * 0.866_025_4
}

/// Note-constant settings, latched at the first sample.
#[derive(Clone, Copy)]
struct Settings {
    players: f32,
    spread: f32,
    vib: f32,
    onset: f32,
    width: f32,
    ctrl: usize,
    phase: f32,
}

/// The ensemble bank. See the module doc.
pub struct PartialsEnsemble {
    bank: PartialsBlend,
    started: bool,
    set: Settings,
    /// Coherent amplitude `α` and random amplitude `β`.
    alpha: f32,
    beta: f32,
    /// Right channel's mix of the left and its own process.
    w_cos: f32,
    w_sin: f32,
    /// Per partial, two one-pole stages × (re, im) × (L, R).
    g_state: Vec<[f32; 8]>,
    rng_l: Rng,
    rng_r: Rng,
    /// Per partial interpolators: current (re_L, im_L, re_R, im_R) and the
    /// per-sample increment toward the next tick's value.
    cur: Vec<[f32; 4]>,
    inc: Vec<[f32; 4]>,
    /// Samples left until the next control tick.
    until_tick: usize,
    stretch: f32,
    /// Onset smear: a ring of the gains, `MAX_BLEND_CHANNELS` per sample.
    ring: Vec<f32>,
    ring_len: usize,
    ring_pos: usize,
    taps: [usize; ONSET_TAPS],
    smear: bool,
    seed: u32,
}

impl PartialsEnsemble {
    /// Construct bound to `table` (as [`PartialsBlend::new`]).
    pub fn new(table: Arc<CoeffTable>) -> Self {
        let bank = PartialsBlend::new(table);
        let m = bank.m_max;
        PartialsEnsemble {
            started: false,
            set: Settings {
                players: 1.0,
                spread: 0.0,
                vib: 0.0,
                onset: 0.0,
                width: 0.0,
                ctrl: 16,
                phase: 1.0,
            },
            alpha: 1.0,
            beta: 0.0,
            w_cos: 1.0,
            w_sin: 0.0,
            g_state: alloc::vec![[0.0; 8]; m],
            rng_l: Rng::new(SALT_GAUSS_L),
            rng_r: Rng::new(SALT_GAUSS_R),
            cur: alloc::vec![[0.0; 4]; m],
            inc: alloc::vec![[0.0; 4]; m],
            until_tick: 0,
            stretch: 1.0,
            ring: Vec::new(),
            ring_len: 0,
            ring_pos: 0,
            taps: [0; ONSET_TAPS],
            smear: false,
            seed: crate::ugens::partials::PARTIALS_NOISE_DEFAULT_SEED,
            bank,
        }
    }

    /// Partial oscillators this bank runs.
    pub fn partials(&self) -> usize {
        self.bank.m_max
    }

    fn latch(&mut self, inputs: &[Option<&AudioBuffer>]) {
        let rd = |p: usize, d: f32| {
            let v = read_input(inputs.get(p).copied().flatten(), 0, 0, d);
            if v.is_finite() { v } else { d }
        };
        let s = Settings {
            players: rd(P_PLAYERS, 1.0).max(1.0),
            spread: rd(P_SPREAD, 0.0).max(0.0),
            vib: rd(P_VIB, 0.0).max(0.0),
            onset: rd(P_ONSET, 0.0).clamp(0.0, MAX_ONSET_S),
            width: rd(P_WIDTH, 0.0).clamp(0.0, 1.0),
            ctrl: (rd(P_CTRL, 16.0).round() as i64).clamp(1, MAX_CTRL as i64) as usize,
            phase: rd(P_PHASE, 1.0).clamp(0.0, 1.0),
        };
        self.set = s;
        if s.phase > 0.0 {
            scatter_phases(&mut self.bank.phase, self.seed, s.phase);
        }
        let (ws, wc) = sin_cos_turns_f32(0.25 * s.width);
        self.w_cos = wc;
        self.w_sin = ws;
        self.rng_l = Rng::new(self.seed ^ SALT_GAUSS_L);
        self.rng_r = Rng::new(self.seed ^ SALT_GAUSS_R);
        let alpha = s.players.powf(-0.25);
        self.alpha = alpha;
        self.beta = (1.0 - alpha * alpha).max(0.0).sqrt();
        // stationary start: both stages at one Gaussian draw of the
        // component's variance (1/2), as narrow-band states nearly are
        let h = core::f32::consts::FRAC_1_SQRT_2;
        for st in self.g_state.iter_mut() {
            let (lr, li) = (gauss(&mut self.rng_l) * h, gauss(&mut self.rng_l) * h);
            let (rr, ri) = (gauss(&mut self.rng_r) * h, gauss(&mut self.rng_r) * h);
            *st = [lr, lr, li, li, rr, rr, ri, ri];
        }
        // onset smear ring
        let d = (s.onset * self.bank.sample_rate).round() as usize;
        self.smear = d > 0;
        if d > 0 {
            self.ring_len = d + 1;
            if self.ring.len() < self.ring_len * MAX_BLEND_CHANNELS {
                // sized in `init` for the longest onset; only a sample rate
                // above the one `init` saw can reach here
                self.ring.resize(self.ring_len * MAX_BLEND_CHANNELS, 0.0);
            }
            for (t, tap) in self.taps.iter_mut().enumerate() {
                *tap = (t * d + (ONSET_TAPS - 1) / 2) / (ONSET_TAPS - 1);
            }
            self.ring.iter_mut().for_each(|x| *x = 0.0);
            self.ring_pos = 0;
        }
    }

    /// One control tick at pitch `f0` and gains `g`: set every partial's
    /// interpolator toward its new complex amplitude (L and R).
    fn tick(&mut self, f0: f32, g: &[f32; MAX_BLEND_CHANNELS], first: bool) {
        let bank = &self.bank;
        let (k_n, m_n) = (bank.k, bank.m_max);
        let sr = bank.sample_rate.max(1.0);
        let nyquist = sr * 0.5;
        let p = self.set.ctrl;
        let inv_p = 1.0 / p as f32;
        let dt = p as f32 / sr;
        if bank.entries.is_empty() {
            return;
        }
        let (lo, hi, t) = bank.bracket(f0);
        let e_lo = &bank.entries[lo];
        let e_hi = &bank.entries[hi];
        let stretch = lerp(e_lo.stretch, e_hi.stretch, t);
        let mainlobe_gain = lerp(e_lo.mainlobe_gain, e_hi.mainlobe_gain, t);
        self.stretch = stretch;

        // the one-pole coefficient for partial m is q^(m+1)
        let n = self.set.players;
        let vib_var = 0.5 * self.set.vib * self.set.vib * (1.0 - 1.0 / n);
        let sigma_c = (self.set.spread * self.set.spread + vib_var).sqrt();
        let a1 = TWO_POLE_1E
            * core::f32::consts::PI
            * core::f32::consts::SQRT_2
            * f0
            * stretch
            * sigma_c
            * LN2_OVER_1200;
        let q = (-a1 * dt).exp();
        let mut pm = q;
        let beta = self.beta;

        for m in 0..m_n {
            let mut coeff = 0.0f32;
            for (k, gk) in g.iter().enumerate().take(k_n) {
                let idx = k * m_n + m;
                coeff += gk * lerp(e_lo.partials[idx], e_hi.partials[idx], t);
            }
            let freq_m = (m + 1) as f32 * f0 * stretch;
            let amp = if coeff != 0.0 {
                coeff * mainlobe_gain * nyquist_fade(freq_m, nyquist)
            } else {
                0.0
            };

            let (zl_re, zl_im, zr_re, zr_im);
            {
                // two one-pole stages per component, input scaled so each
                // component's output variance is 1/2
                let pp = pm;
                pm *= q;
                let st = &mut self.g_state[m];
                if pp < 1.0 {
                    let b = 1.0 - pp;
                    let one_m_p2 = 1.0 - pp * pp;
                    let sum_h2 = b * b * b * b * (1.0 + pp * pp) / (one_m_p2 * one_m_p2 * one_m_p2);
                    let scale = UNIT_VARIANCE_BIPOLAR_SCALE * (0.5 / sum_h2.max(1e-30)).sqrt() * b;
                    let xs = [
                        self.rng_l.next_bipolar(),
                        self.rng_l.next_bipolar(),
                        self.rng_r.next_bipolar(),
                        self.rng_r.next_bipolar(),
                    ];
                    for c in 0..4 {
                        let s1 = pp * st[2 * c] + scale * xs[c];
                        let s2 = pp * st[2 * c + 1] + b * s1;
                        st[2 * c] = s1;
                        st[2 * c + 1] = s2;
                    }
                }
                let (gl_re, gl_im, gr_re, gr_im) = (st[1], st[3], st[5], st[7]);
                zl_re = self.alpha + beta * gl_re;
                zl_im = beta * gl_im;
                zr_re = self.alpha + beta * (self.w_cos * gl_re + self.w_sin * gr_re);
                zr_im = beta * (self.w_cos * gl_im + self.w_sin * gr_im);
            }
            let target = [amp * zl_re, amp * zl_im, amp * zr_re, amp * zr_im];
            let cur = &mut self.cur[m];
            let inc = &mut self.inc[m];
            if first || !cur.iter().all(|v| v.is_finite()) {
                // a fresh note, or (defensively) a state that went non-finite:
                // restart the interpolator at the target
                *cur = target;
                *inc = [0.0; 4];
            } else {
                for c in 0..4 {
                    inc[c] = (target[c] - cur[c]) * inv_p;
                }
            }
        }
    }
}

impl UGen for PartialsEnsemble {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "PartialsEnsemble",
            category: UGenCategory::Oscillator,
            inputs: &INPUTS,
            outputs: &[OutputSpec {
                name: "out",
                rate: Rate::Audio,
            }],
        }
    }

    fn output_channels(&self, _input_channels: &[usize]) -> usize {
        2
    }

    fn init(&mut self, context: &ProcessContext) {
        self.bank.init(context);
        let max = (MAX_ONSET_S * context.sample_rate).ceil() as usize + 2;
        self.ring = alloc::vec![0.0; max * MAX_BLEND_CHANNELS];
    }

    fn reset(&mut self) {
        self.bank.reset();
        self.started = false;
        self.until_tick = 0;
        self.seed = crate::ugens::partials::PARTIALS_NOISE_DEFAULT_SEED;
        self.smear = false;
        self.ring.iter_mut().for_each(|x| *x = 0.0);
        self.ring_len = 0;
        self.ring_pos = 0;
        self.taps = [0; ONSET_TAPS];
        self.cur.iter_mut().for_each(|c| *c = [0.0; 4]);
        self.inc.iter_mut().for_each(|c| *c = [0.0; 4]);
        self.g_state.iter_mut().for_each(|g| *g = [0.0; 8]);
        self.stretch = 1.0;
    }

    fn reseed_noise(&mut self, seed: u32) {
        self.bank.reseed_noise(seed);
        self.seed = seed;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let first_block = !self.started;
        if first_block {
            self.started = true;
            self.latch(inputs);
        }
        let freq_buf = inputs.first().copied().flatten();
        let gain_bufs: [Option<&AudioBuffer>; MAX_BLEND_CHANNELS] =
            core::array::from_fn(|k| inputs.get(1 + k).copied().flatten());
        let sr = self.bank.sample_rate.max(1.0);
        let inv_sr = 1.0 / sr;
        let k_n = self.bank.k;
        let j_n = self.bank.j_max;
        let ctrl = self.set.ctrl;
        let n_out = output.num_channels();
        let len = output.channel(0).samples().len();
        for ch in 0..n_out {
            output.channel_mut(ch).samples_mut().fill(0.0);
        }

        let mut f0s = [0.0f32; MAX_CTRL];
        // per sample: 1 where `freq` is positive and finite, else 0 (the
        // partials are silent there, as partialsBlend's are at a stopped phase)
        let mut on = [0.0f32; MAX_CTRL];
        let mut gs = [[0.0f32; MAX_BLEND_CHANNELS]; MAX_CTRL];
        let mut noise_out = [0.0f32; MAX_CTRL];
        let mut first_tick = first_block;
        let mut i0 = 0usize;
        while i0 < len {
            if self.until_tick == 0 {
                self.until_tick = ctrl;
            }
            let seg = self.until_tick.min(len - i0);
            // inputs for this segment (pitch and the smeared gains)
            for s in 0..seg {
                let i = i0 + s;
                let f0_raw = read_input(freq_buf, 0, i, DEFAULT_FREQ_HZ);
                f0s[s] = if f0_raw.is_finite() {
                    f0_raw.max(0.0)
                } else {
                    0.0
                };
                on[s] = if f0s[s] > 0.0 { 1.0 } else { 0.0 };
                let mut g = [0.0f32; MAX_BLEND_CHANNELS];
                for (k, gk) in g.iter_mut().enumerate().take(k_n) {
                    let v = read_input(gain_bufs[k], 0, i, if k == 0 { 1.0 } else { 0.0 });
                    // a non-finite gain plays as 0 for that sample (as a
                    // non-finite `freq` does), so it can never reach the
                    // interpolators or the smear ring
                    *gk = if v.is_finite() { v } else { 0.0 };
                }
                if self.smear {
                    let base = self.ring_pos * MAX_BLEND_CHANNELS;
                    self.ring[base..base + MAX_BLEND_CHANNELS].copy_from_slice(&g);
                    let mut acc = [0.0f32; MAX_BLEND_CHANNELS];
                    for &d in &self.taps {
                        let at = (self.ring_pos + self.ring_len - d) % self.ring_len;
                        let b = at * MAX_BLEND_CHANNELS;
                        for (a, r) in acc.iter_mut().zip(&self.ring[b..b + k_n]) {
                            *a += r;
                        }
                    }
                    for (gk, a) in g.iter_mut().zip(&acc).take(k_n) {
                        *gk = a * (1.0 / ONSET_TAPS as f32);
                    }
                    self.ring_pos = (self.ring_pos + 1) % self.ring_len;
                }
                gs[s] = g;
            }
            // a tick at the start of each control period
            if self.until_tick == ctrl {
                let g0 = gs[0];
                self.tick(f0s[0], &g0, first_tick);
                first_tick = false;
            }

            // partials: one carrier per partial, complex amplitude per channel
            let stretch = self.stretch;
            let all_on = on[..seg].iter().all(|&o| o > 0.0);
            {
                let (left, right) = output.channel_pair_mut(0, 1);
                let left = &mut left[i0..i0 + seg];
                let right = &mut right[i0..i0 + seg];
                for (m, ph) in self.bank.phase.iter_mut().enumerate() {
                    // the same expression, in the same order, as
                    // partialsBlend's phase step
                    let mf = (m + 1) as f32;
                    let cur = &mut self.cur[m];
                    let inc = self.inc[m];
                    let silent = cur.iter().all(|&v| v == 0.0) && inc.iter().all(|&v| v == 0.0);
                    if silent {
                        for &f0 in f0s.iter().take(seg) {
                            *ph += mf * f0 * stretch * inv_sr;
                            *ph -= ph.floor();
                        }
                        continue;
                    }
                    let [mut a0, mut a1, mut a2, mut a3] = *cur;
                    if all_on {
                        for s in 0..seg {
                            *ph += mf * f0s[s] * stretch * inv_sr;
                            *ph -= ph.floor();
                            // step toward the tick's target before use, so the
                            // target is reached on the period's last sample
                            // (exact at `ctrl` 1)
                            a0 += inc[0];
                            a1 += inc[1];
                            a2 += inc[2];
                            a3 += inc[3];
                            let (sn, cs) = sin_cos_turns_f32(*ph);
                            // Im(z e^{iφ}) with φ = 2π·ph (the bank's partials
                            // are sines): re·sin + im·cos
                            left[s] += a0 * sn + a1 * cs;
                            right[s] += a2 * sn + a3 * cs;
                        }
                    } else {
                        // a segment with stopped samples (freq <= 0 or not
                        // finite): no output there rather than a frozen
                        // carrier's DC
                        for s in 0..seg {
                            *ph += mf * f0s[s] * stretch * inv_sr;
                            *ph -= ph.floor();
                            a0 += inc[0];
                            a1 += inc[1];
                            a2 += inc[2];
                            a3 += inc[3];
                            if on[s] > 0.0 {
                                let (sn, cs) = sin_cos_turns_f32(*ph);
                                left[s] += a0 * sn + a1 * cs;
                                right[s] += a2 * sn + a3 * cs;
                            }
                        }
                    }
                    *cur = [a0, a1, a2, a3];
                }
            }

            // noise bands, shared by both channels (as partialsBlend's)
            if !self.bank.live_bands.is_empty() && !self.bank.entries.is_empty() {
                for s in 0..seg {
                    let (lo, hi, t) = self.bank.bracket(f0s[s]);
                    let bank = &mut self.bank;
                    let e_lo = &bank.entries[lo];
                    let e_hi = &bank.entries[hi];
                    let noise_gain = lerp(e_lo.noise_gain, e_hi.noise_gain, t);
                    let noise_in = bank.rng.next_bipolar() * UNIT_VARIANCE_BIPOLAR_SCALE;
                    let mut acc = 0.0f32;
                    for &j in &bank.live_bands {
                        let (b0, b1, b2, a1, a2) = bank.noise_biquad[j];
                        let filtered = bank.noise_state[j].tick(noise_in, b0, b1, b2, a1, a2);
                        let mut power = 0.0f32;
                        for (k, gk) in gs[s].iter().enumerate().take(k_n) {
                            let idx = k * j_n + j;
                            let a = gk * lerp(e_lo.noise[idx], e_hi.noise[idx], t);
                            power += a * a;
                        }
                        if power != 0.0 {
                            acc += power.sqrt()
                                * noise_gain
                                * bank.noise_power_compensation[j]
                                * filtered;
                        }
                    }
                    noise_out[s] = acc;
                }
                let (left, right) = output.channel_pair_mut(0, 1);
                for s in 0..seg {
                    left[i0 + s] += noise_out[s];
                    right[i0 + s] += noise_out[s];
                }
            }

            self.until_tick -= seg;
            i0 += seg;
        }
    }
}

/// Register the `partialsEnsemble` table-bound kind.
pub fn register(reg: &mut UGenRegistry) {
    let factory: TableUGenFactory = Arc::new(|table| Box::new(PartialsEnsemble::new(table)));
    reg.register_table_bound(
        "partialsEnsemble",
        factory,
        UGenCategory::Oscillator,
        &INPUTS,
        &[OutputSpec {
            name: "out",
            rate: Rate::Audio,
        }],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coeff_table::PitchEntry;

    /// One entry at 440 Hz with a single partial of coefficient 1.
    fn one_partial(m: usize) -> Arc<CoeffTable> {
        let mut coefficients = alloc::vec![0.0f32; m];
        coefficients[m - 1] = 1.0;
        Arc::new(CoeffTable {
            name: "one".into(),
            entries: alloc::vec![PitchEntry {
                f0_hz: 440.0,
                inharmonicity_stretch: 1.0,
                partial_freqs: (1..=m).map(|k| k as f32 * 440.0).collect(),
                k_channels: 1,
                j_noise: 0,
                coefficients,
                metadata: alloc::vec![("mainlobe_gain".into(), 1.0)],
            }],
        })
    }

    fn bank(m: usize, s: Settings, seed: u32) -> PartialsEnsemble {
        bank_at(44100.0, m, s, seed)
    }

    fn bank_at(sr: f32, m: usize, s: Settings, seed: u32) -> PartialsEnsemble {
        let mut b = PartialsEnsemble::new(one_partial(m));
        b.init(&ProcessContext::new(sr, 128));
        b.reseed_noise(seed);
        // latch through the port defaults, then overwrite with `s`
        b.latch(&[]);
        b.set = s;
        let n = s.players;
        let a = n.powf(-0.25);
        b.alpha = a;
        b.beta = (1.0 - a * a).sqrt();
        let (ws, wc) = sin_cos_turns_f32(0.25 * s.width);
        b.w_cos = wc;
        b.w_sin = ws;
        b
    }

    fn stat(players: f32, spread: f32, width: f32) -> Settings {
        Settings {
            players,
            spread,
            vib: 0.0,
            onset: 0.0,
            width,
            ctrl: 16,
            phase: 1.0,
        }
    }

    /// The complex amplitudes (L, R) of the bank's last partial over `n`
    /// ticks at 440 Hz.
    fn z_series(b: &mut PartialsEnsemble, n: usize) -> Vec<[f32; 4]> {
        let m = b.bank.m_max - 1;
        let g = [1.0, 0.0, 0.0, 0.0];
        (0..n)
            .map(|i| {
                b.tick(440.0, &g, i == 0);
                // the target is cur + P·inc (first tick: cur)
                let p = b.set.ctrl as f32;
                let c = b.cur[m];
                let d = b.inc[m];
                core::array::from_fn(|k| c[k] + p * d[k])
            })
            .collect()
    }

    #[test]
    fn sin_cos_turns_f32_is_accurate() {
        let mut worst = 0.0f64;
        for i in 0..200_000 {
            let t = i as f32 / 200_000.0;
            let (s, c) = sin_cos_turns_f32(t);
            let x = core::f64::consts::TAU * t as f64;
            worst = worst.max((s as f64 - x.sin()).abs());
            worst = worst.max((c as f64 - x.cos()).abs());
        }
        assert!(worst < 1e-6, "worst {worst}");
    }

    #[test]
    fn power_has_unit_mean_and_variance_one_minus_one_over_n() {
        for &n in &[3.0f32, 12.0, 1e6] {
            // 8 cents at 440 Hz: sigma 2 Hz, correlation time ~0.11 s =
            // ~300 ticks; 400k ticks hold ~1300 correlation times
            let mut b = bank(1, stat(n, 8.0, 0.0), 77);
            let z = z_series(&mut b, 400_000);
            let p: Vec<f64> = z
                .iter()
                .map(|v| (v[0] as f64).powi(2) + (v[1] as f64).powi(2))
                .collect();
            let mean = p.iter().sum::<f64>() / p.len() as f64;
            let var = p.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / p.len() as f64;
            let want = 1.0 - 1.0 / n as f64;
            assert!((mean - 1.0).abs() < 0.08, "N={n}: mean power {mean}");
            assert!(
                (var - want).abs() < 0.15 * want.max(0.3),
                "N={n}: var {var} want {want}"
            );
        }
    }

    #[test]
    fn correlation_time_follows_spread_and_partial_number() {
        // |r(tau)| of the random part falls to 1/e at tau_e = 1/(pi sqrt2 sigma),
        // at two sample rates
        for &(sr, m, spread) in &[
            (44100.0f32, 1usize, 8.0f32),
            (44100.0, 4, 8.0),
            (44100.0, 1, 2.0),
            (48000.0, 1, 8.0),
            (48000.0, 4, 8.0),
        ] {
            let mut b = bank_at(sr, m, stat(1e9, spread, 0.0), 5);
            let z = z_series(&mut b, 300_000);
            let sigma = (m as f64) * 440.0 * spread as f64 * core::f64::consts::LN_2 / 1200.0;
            let tau_e = 1.0 / (core::f64::consts::PI * core::f64::consts::SQRT_2 * sigma);
            let dt = 16.0 / sr as f64;
            let r = |lag: usize| -> f64 {
                let (mut re, mut im, mut p) = (0.0f64, 0.0f64, 0.0f64);
                for i in 0..z.len() - lag {
                    let (a, bb) = (z[i], z[i + lag]);
                    re += (a[0] * bb[0] + a[1] * bb[1]) as f64;
                    im += (a[1] * bb[0] - a[0] * bb[1]) as f64;
                    p += (a[0] * a[0] + a[1] * a[1]) as f64;
                }
                (re * re + im * im).sqrt() / p
            };
            let lag = (tau_e / dt).round() as usize;
            let got = r(lag);
            let want = (-1.0f64).exp();
            assert!(
                (got - want).abs() < 0.06,
                "m={m} spread={spread}: r(tau_e) {got}"
            );
        }
    }

    #[test]
    fn width_decorrelates_the_random_part_only() {
        for &(n, w) in &[(1e9f32, 0.0f32), (1e9, 1.0), (12.0, 1.0), (12.0, 0.5)] {
            let mut b = bank(1, stat(n, 8.0, w), 9);
            let z = z_series(&mut b, 200_000);
            let (mut re, mut pl, mut pr) = (0.0f64, 0.0f64, 0.0f64);
            for v in &z {
                re += (v[0] * v[2] + v[1] * v[3]) as f64;
                pl += (v[0] * v[0] + v[1] * v[1]) as f64;
                pr += (v[2] * v[2] + v[3] * v[3]) as f64;
            }
            let rho = re / (pl * pr).sqrt();
            let a2 = (n as f64).powf(-0.5);
            let want = a2 + (1.0 - a2) * (core::f64::consts::FRAC_PI_2 * w as f64).cos();
            assert!(
                (rho - want).abs() < 0.06,
                "N={n} w={w}: rho {rho} want {want}"
            );
        }
    }

    #[test]
    fn one_player_does_not_fluctuate() {
        let mut b = bank(1, stat(1.0, 8.0, 1.0), 3);
        for v in z_series(&mut b, 1000) {
            assert!((v[0] - 1.0).abs() < 1e-6 && v[1].abs() < 1e-6, "{v:?}");
            assert!((v[2] - 1.0).abs() < 1e-6 && v[3].abs() < 1e-6, "{v:?}");
        }
    }
}
