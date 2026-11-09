//! Waveform steepening: the nonlinear propagation of a wave down a bore.
//!
//! - [`Steepen`]: a brass-like brightening that keeps the harmonic envelope
//!   smooth. Each compression of the input travels ahead and each
//!   rarefaction falls behind, as a loud pressure wave does in a long
//!   trumpet or horn bore, so the waveform steepens towards a shock (a
//!   sawtooth-like front) as `drive` rises.

use crate::buffer::{AudioBuffer, channel_wrapped, read_input, require_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use alloc::vec::Vec;
use core::f64::consts::TAU;

/// Most delay stages a [`Steepen`] node runs (`stages` is clamped to 1 ..= this).
pub const STEEPEN_MAX_STAGES: usize = 32;

/// The automatic base delay, and the displacement clamp, hold a total time
/// shift of this many times the shock displacement of the reference sine:
/// `drive · (input amplitude / ref_amp)` up to 2 stays inside the buffers.
pub const STEEPEN_HEADROOM: f64 = 2.0;

/// The lowest `freq` (Hz) the node keytracks to; lower values are clamped.
/// Together with [`STEEPEN_HEADROOM`] it sets the buffer size.
pub const STEEPEN_F0_MIN: f64 = 20.0;

/// The pitch (Hz) at which `drive` 1 is the shock of the reference sine, by
/// default: C4.
pub const STEEPEN_REF_HZ: f64 = 261.625_565_300_598_6;

/// `freq` and `ref_hz` are clamped below this fraction of the sample rate.
const NYQUIST_GUARD: f64 = 0.49;

/// The de-emphasis pole is clamped to `0 ..= A_MAX` (OnePole's own limit).
const A_MAX: f64 = 0.9999;

/// Default corner factors, × `freq`: 0.573 f0 (150 Hz at C4) and 1.5 f0.
const K1_DEFAULT: f32 = 0.573;
const K2_DEFAULT: f32 = 1.5;
const STAGES_DEFAULT: f32 = 16.0;

/// Ring slack per stage: two guard samples for the interpolated read, and
/// one more so a full-scale displacement never touches the write cursor.
const RING_SLACK: usize = 6;

/// Samples of delay memory per channel at `sample_rate`, over all stages:
/// `2 · HEADROOM · fs / (2π · F0_MIN)` plus [`RING_SLACK`] per stage slot.
/// At 44.1 kHz that is 1596 `f64`s, 12.8 kB; a node keeps two channels.
pub fn steepen_pool_len(sample_rate: f64) -> usize {
    let span = 2.0 * STEEPEN_HEADROOM * sample_rate / (TAU * STEEPEN_F0_MIN);
    span.ceil() as usize + RING_SLACK * STEEPEN_MAX_STAGES
}

/// One de-emphasis pole for corner factor `k` at fundamental `f0` and sample
/// rate `sr`: `a = 1 − 2π·k·f0/fs` clamped to `0 ..= 0.9999` (the
/// forward-Euler pole, as `onePole` with that coefficient), or `None` for
/// `k ≤ 0` (no stage).
#[inline]
fn pole(k: f64, f0: f64, sr: f64) -> Option<f64> {
    if k > 0.0 {
        Some((1.0 - TAU * k * f0 / sr).clamp(0.0, A_MAX))
    } else {
        None
    }
}

/// `|H(e^{jω})|` of the one-pole `(1 − a) / (1 − a z⁻¹)` at frequency `f`.
#[inline]
fn pole_gain(a: f64, f: f64, sr: f64) -> f64 {
    let w = TAU * f / sr;
    (1.0 - a) / (1.0 - 2.0 * a * w.cos() + a * a).sqrt()
}

#[inline]
fn finite_or(x: f32, default: f32) -> f64 {
    if x.is_finite() {
        x as f64
    } else {
        default as f64
    }
}

/// Waveform steepening (Burgers / Fubini bore distortion) with exact
/// de-emphasis: the brightening of a loud brass tone, with a smooth harmonic
/// envelope and no holes at any drive below the shock.
///
/// The input `x` is lowpassed into an "internal pressure" `p` by one or two
/// keytracked one-pole de-emphasis filters, `p` is time-warped by `K`
/// short modulated delay stages (each sample of a stage's input `z` is read
/// back `t0 − Δ·z` seconds later: compressions advance, rarefactions lag),
/// and the result is re-emphasised by the exact inverse filters:
///
/// ```text
/// a_i = 1 − 2π·k_i·f0/fs                    p = H₂ H₁ x,   H_i = (1 − a_i) / (1 − a_i z⁻¹)
/// z₀ = p,   z_{j+1}(n) = z_j(n − t0·fs + Δ·fs·z_j(n))    (j = 0 … K−1, linear interpolation)
/// y = H₁⁻¹ H₂⁻¹ z_K,                          H_i⁻¹ :  y(n) = (w(n) − a_i·w(n−1)) / (1 − a_i)
/// Δ = drive · ref_hz / (smax · K · f0)        smax = ref_amp · 2π · ref_hz · |H₁ H₂|(ref_hz)
/// ```
///
/// Each stage is one explicit step of the method of characteristics for
/// Burgers' equation, so `K` stages approximate the implicit (Fubini)
/// solution `p(t) = p₀(t + β·p)` with `β = K·Δ`. The steepening index
/// `σ = β · max|dp/dt|` reaches 1, the shock, when a sine of peak `ref_amp`
/// at `ref_hz` meets `drive` 1: `smax` is that sine's peak slope after the
/// de-emphasis. Fubini's harmonics of a steepened sine, `2 J_n(nσ) / (nσ)`,
/// fall smoothly with no zeros below the shock (as `n^(-4/3)` at it). The
/// explicit stages are first-order: at `K` = 16 a steepened sine's h2 and
/// h3 are Fubini's within 0.6 dB and the higher harmonics fall faster (h8
/// 4.5 dB under it at `σ` 0.5, 3.7 dB at 1), still smooth and falling
/// (about −2.6 dB per harmonic over h3–h10 at the shock); more stages come
/// closer.
///
/// **Keytracking.** The corners (`k_i · freq`) and the time scale
/// (`Δ ∝ 1/freq`) both follow `freq`, so the de-emphasis gain at the
/// fundamental, and with it `σ` for a given input level, are the same at
/// every pitch: a note an octave down steepens as much as one at
/// `ref_hz`, not twice as much (C3 and G4 envelopes within about 1 dB of
/// C4's over h1–h10; the forward-Euler corners and the interpolation
/// account for the rest).
///
/// **Small-signal gain 1.** As `drive · p → 0` every stage is a pure delay
/// and the inverse filters undo the de-emphasis exactly: the node is a
/// delay of `K · t0`. DC passes with gain 1 (both filter pairs have unit
/// DC gain) and the inverse filters are FIR, so nothing accumulates. The
/// explicit stages conserve the mean only to first order: a zero-mean sine
/// steepened to the shock comes out with a DC of about 1.3 % of its RMS,
/// as an asymmetric waveshaper's would; follow the node with a DC blocker.
///
/// **Base delay.** Each stage needs a base delay `t0` at least as long as
/// the largest advance it applies.
/// - `tau ≤ 0` (default): automatic, a whole number of samples,
///   `t0 = ⌈HEADROOM · fs / (2π · f0 · K)⌉`, fixed at the first sample the
///   node processes (channel 0's; a later glide keeps it). With a whole-sample base
///   delay, `drive` 0 is an exact delay of `K · t0` samples and a small
///   `drive` adds little interpolation loss.
/// - `tau > 0`: `t0 = tau · ref_hz / (K · f0)` seconds, the **total** base
///   delay `tau` at `ref_hz`, split over the stages and keytracked
///   continuously. A fractional `t0` puts the linear interpolation's
///   lowpass on the signal even at `drive` 0 (at C4, 16 stages and a
///   1.4-sample `t0`, about −1.5 dB at h8 and −6 dB at h16). This is what a
///   graph of `delay` nodes with a keytracked base time does; it is here so
///   such a graph can be replaced by this node sample for sample.
///
/// Each stage's displacement is clamped to `±t0` (its read stays between
/// 0 and `2·t0` samples back), so an input louder than the headroom
/// clips the time shift instead of reading outside the buffer.
///
/// **Memory.** The rings hold `t0 ≤ HEADROOM · fs / (2π · F0_MIN · K)`
/// per stage, `2 · HEADROOM · fs / (2π · 20 Hz)` + 6 per stage slot `f64`s
/// in all ([`steepen_pool_len`]): 1596 values, 12.8 kB per channel at
/// 44.1 kHz. Each of two channels keeps its own rings and filter states,
/// so a voice holds **25.5 kB at 44.1 kHz** (27.5 kB at 48 kHz, 52 kB at
/// 96 kHz), whatever `stages` is, all allocated in `init`. A third or later
/// channel runs on channel 1's state, as the delay-line kinds do.
/// The rings are `f64` because each inverse filter amplifies high
/// frequencies by up to `(1 + a)/(1 − a)` (the default pair, at C4: about
/// 70 dB at Nyquist), and `f32` storage noise would come back that much
/// louder.
///
/// Inputs (all but `in` optional):
/// - `in` (required): the signal; for a brass voice, the source before
///   the bell.
/// - `freq`: the note's f0 in Hz (default 261.63), clamped to
///   20 Hz … 0.49 · fs; keytracks the corners and the time scale.
/// - `drive`: the steepening, 0 = none, 1 = the shock of a sine of peak
///   `ref_amp` at `ref_hz` (default 0), clamped to 0 … 2; audio-rate.
/// - `k1`, `k2`: de-emphasis corners as multiples of `freq` (defaults
///   0.573 and 1.5: 150 Hz and 392 Hz at C4). `k2 ≤ 0` leaves one stage,
///   `k1 ≤ 0` and `k2 ≤ 0` none (the input itself is steepened).
/// - `stages`: number of delay stages `K`, 1 … 32 (default 16).
/// - `ref_hz`: the pitch at which `drive` 1 is the shock (default C4,
///   261.6256 Hz).
/// - `ref_amp`: the peak of the reference sine (default 1). Set it to the
///   level at which the instrument's own loudest note should reach the shock
///   at `drive` 1.
/// - `tau`: total base delay in seconds at `ref_hz` (default 0 = automatic,
///   above).
///
/// `k1`, `k2`, `stages`, `ref_hz`, `ref_amp` and `tau` are read once per
/// block (their first sample). They are per-instrument constants: a change
/// of `stages` mid-note refills the stages and clicks.
/// Non-finite inputs read as their defaults (`in` as 0).
pub struct Steepen {
    sample_rate: f64,
    /// One state per channel: see filters::OnePole's process().
    chans: [ChannelState; 2],
    ring_len: usize,
    stages: usize,
    /// The automatic base delay (samples per stage), fixed at the first sample.
    auto_t0: Option<f64>,
}

/// One channel's delay rings and filter states.
struct ChannelState {
    /// `stages` rings of `ring_len` samples each, back to back.
    pool: Vec<f64>,
    write_pos: usize,
    /// De-emphasis states and the inverse filters' previous inputs.
    p1: f64,
    p2: f64,
    w1: f64,
    w2: f64,
    /// The last internal pressure, to refill the stages when `stages` changes.
    last_p: f64,
}

impl ChannelState {
    fn new() -> Self {
        ChannelState {
            pool: Vec::new(),
            write_pos: 0,
            p1: 0.0,
            p2: 0.0,
            w1: 0.0,
            w2: 0.0,
            last_p: 0.0,
        }
    }

    fn clear(&mut self) {
        self.pool.fill(0.0);
        self.write_pos = 0;
        self.p1 = 0.0;
        self.p2 = 0.0;
        self.w1 = 0.0;
        self.w2 = 0.0;
        self.last_p = 0.0;
    }

    /// Refill the rings with the last internal pressure and restart the
    /// inverse filters from it.
    fn refill(&mut self) {
        self.pool.fill(self.last_p);
        self.w1 = self.last_p;
        self.w2 = self.last_p;
        self.write_pos = 0;
    }
}

impl Default for Steepen {
    fn default() -> Self {
        Self::new()
    }
}

impl Steepen {
    pub fn new() -> Self {
        Steepen {
            sample_rate: 44100.0,
            chans: [ChannelState::new(), ChannelState::new()],
            ring_len: 0,
            stages: 0,
            auto_t0: None,
        }
    }

    /// Bytes of delay memory this node holds (after `init`), both channels.
    pub fn buffer_bytes(&self) -> usize {
        self.chans
            .iter()
            .map(|c| c.pool.len() * core::mem::size_of::<f64>())
            .sum()
    }

    /// Re-partition the rings for `k` stages. They are refilled with the
    /// last internal pressure, and the inverse filters restarted from it, so
    /// a change mid-note steps the output by at most the signal's own range
    /// instead of an inverse-filtered click.
    fn set_stages(&mut self, k: usize) {
        self.stages = k;
        self.ring_len = self.chans[0].pool.len() / k;
        for c in &mut self.chans {
            c.refill();
        }
        self.auto_t0 = None;
    }

    fn clear_state(&mut self) {
        for c in &mut self.chans {
            c.clear();
        }
        self.auto_t0 = None;
    }
}

impl UGen for Steepen {
    ugen_spec!(
        "Steepen",
        category = Effect,
        inputs = ["in"],
        optional_inputs = [
            "freq", "drive", "k1", "k2", "stages", "ref_hz", "ref_amp", "tau"
        ],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate as f64;
        let len = steepen_pool_len(self.sample_rate);
        for c in &mut self.chans {
            c.pool = alloc::vec![0.0; len];
        }
        self.stages = 0;
        self.clear_state();
    }

    fn reset(&mut self) {
        self.clear_state();
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let in_buf = require_input(inputs, 0, self.spec().name, "in");
        let port = |k: usize| inputs.get(k).copied().flatten();
        let (freq_buf, drive_buf, k1_buf, k2_buf) = (port(1), port(2), port(3), port(4));
        let (stages_buf, ref_buf, amp_buf, tau_buf) = (port(5), port(6), port(7), port(8));
        if self.chans[0].pool.is_empty() {
            return;
        }
        let sr = self.sample_rate;
        let top = NYQUIST_GUARD * sr;

        // Per-block constants (first sample of channel 0).
        let k1 = finite_or(read_input(k1_buf, 0, 0, K1_DEFAULT), K1_DEFAULT);
        let k2 = finite_or(read_input(k2_buf, 0, 0, K2_DEFAULT), K2_DEFAULT);
        let stages = finite_or(read_input(stages_buf, 0, 0, STAGES_DEFAULT), STAGES_DEFAULT)
            .round()
            .clamp(1.0, STEEPEN_MAX_STAGES as f64) as usize;
        let ref_def = STEEPEN_REF_HZ as f32;
        let ref_hz =
            finite_or(read_input(ref_buf, 0, 0, ref_def), ref_def).clamp(STEEPEN_F0_MIN, top);
        let ref_amp = finite_or(read_input(amp_buf, 0, 0, 1.0), 1.0)
            .abs()
            .max(1e-12);
        let tau = finite_or(read_input(tau_buf, 0, 0, 0.0), 0.0);
        if stages != self.stages {
            self.set_stages(stages);
        }
        let kf = stages as f64;
        let len = self.ring_len;
        // the largest base delay (samples) whose ±t0 swing fits a ring
        let t0_max = ((len - 3) as f64 / 2.0).floor();

        // drive 1 = the shock of a sine of peak ref_amp at ref_hz
        let mut href = 1.0;
        for k in [k1, k2] {
            if let Some(a) = pole(k, ref_hz, sr) {
                href *= pole_gain(a, ref_hz, sr);
            }
        }
        let smax = ref_amp * TAU * ref_hz * href;
        // samples of displacement per unit drive, per unit p, at f0 = 1 Hz
        let disp_scale = sr * ref_hz / (smax * kf);

        // One state per channel: see filters::OnePole's process(). A third
        // or later channel runs on channel 1's, as the delay-line kinds do.
        let auto_t0 = &mut self.auto_t0;
        for ch in 0..output.num_channels() {
            let st = &mut self.chans[ch.min(1)];
            let (mut p1, mut p2, mut w1, mut w2, mut pos) =
                (st.p1, st.p2, st.w1, st.w2, st.write_pos);
            let mut last_p = st.last_p;
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let x = in_ch[i];
                let x = if x.is_finite() { x as f64 } else { 0.0 };
                let def_f0 = STEEPEN_REF_HZ as f32;
                let f0 = finite_or(read_input(freq_buf, ch, i, def_f0), def_f0)
                    .clamp(STEEPEN_F0_MIN, top);
                let drive =
                    finite_or(read_input(drive_buf, ch, i, 0.0), 0.0).clamp(0.0, STEEPEN_HEADROOM);

                // de-emphasis: the internal pressure
                let a1 = pole(k1, f0, sr);
                let a2 = pole(k2, f0, sr);
                let mut p = x;
                if let Some(a) = a1 {
                    p1 = (1.0 - a) * p + a * p1;
                    p = p1;
                }
                if let Some(a) = a2 {
                    p2 = (1.0 - a) * p + a * p2;
                    p = p2;
                }

                last_p = p;

                // base delay (samples per stage)
                let t0 = if tau > 0.0 {
                    (tau * sr * ref_hz / (kf * f0)).min(t0_max)
                } else {
                    *auto_t0.get_or_insert_with(|| {
                        (STEEPEN_HEADROOM * sr / (TAU * f0 * kf))
                            .ceil()
                            .clamp(1.0, t0_max)
                    })
                };
                let gain = drive * disp_scale / f0;

                // K stages: write, then read t0 − gain·z samples back
                let mut z = p;
                for s in 0..stages {
                    let ring = &mut st.pool[s * len..(s + 1) * len];
                    ring[pos] = z;
                    let d = (t0 - gain * z).clamp(0.0, 2.0 * t0);
                    let di = d as usize;
                    let fr = d - di as f64;
                    let a = ring[(pos + len - di) % len];
                    let b = ring[(pos + len - di - 1) % len];
                    z = a + fr * (b - a);
                }
                pos = (pos + 1) % len;

                // the exact inverse filters
                let mut y = z;
                if let Some(a) = a1 {
                    let v = (y - a * w1) / (1.0 - a);
                    w1 = y;
                    y = v;
                }
                if let Some(a) = a2 {
                    let v = (y - a * w2) / (1.0 - a);
                    w2 = y;
                    y = v;
                }
                let y = y as f32;
                *out_sample = if y.is_finite() { y } else { 0.0 };
            }

            // flush subnormals so a silent tail costs nothing
            let flush = |v: f64| if v.abs() < 1e-30 { 0.0 } else { v };
            st.p1 = flush(p1);
            st.p2 = flush(p2);
            st.w1 = flush(w1);
            st.w2 = flush(w2);
            st.last_p = flush(last_p);
            st.write_pos = pos;
        }
    }
}
