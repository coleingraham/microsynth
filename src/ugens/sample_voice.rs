//! `sampleVoice`: one note of a compiled sample program (an SFZ instrument
//! compiled to the canonical spec and uploaded as a program; see
//! `docs/sample-program-format.md`).
//!
//! The node is bound to a registered program at graph-build time (an IR
//! `table_bindings` entry whose id names a program, because the kind is
//! program-bound); it plays the regions the note selects and nothing else.
//! Behaviour follows sfizz 1.2.3 wherever the program leaves a choice, and
//! the places it differs are listed at the end.
//!
//! **Ports** (all optional, read from channel 0; the output is stereo):
//! - `freq` (Hz, default 261.6256): the note. `n = 69 + 12 log2(freq / 440)`;
//!   the region is selected by `round(n)` (clamped to 0..=127, halves away
//!   from zero) and the pitch follows `n` itself, so a fractional part is a
//!   tuning offset scaled by each region's `pitch_keytrack` (sfizz's reading
//!   of a retuned key). A tuning correction beyond ±50 cents therefore
//!   selects the neighbouring key; octave corrections belong in the note.
//!   `freq` is re-read every block, so a glide retunes sounding regions.
//! - `gate` (default 1): a rise starts a note, a fall releases it.
//! - `vel` (0..1, default 100/127): velocity at the rise,
//!   `round(vel · 127)`.
//! - `amp` (default 1): a linear gain on the output (no velocity in it: the
//!   program's own velocity tracking sets the level, as in sfizz).
//! - `seed` (default 0): seeds the `rand` draws (region selection by
//!   `lorand`/`hirand`, `amp_random`), so a render is reproducible.
//! - `bend` (-1..1, default 0): pitch bend in each region's bend range.
//! - `quality` (default [`DEFAULT_INTERP`]): the resampling kernel, 0 linear,
//!   1 cubic Hermite (sfizz's default), 2 step-tracking windowed sinc. Read
//!   at the rise.
//!
//! **Region selection** (sfizz's `registerNoteOn`): for every region whose
//! key window holds the note, in program order, the region's sequence
//! counter advances; it starts if its trigger fits (`attack`; `first` with
//! no other note of the program held; `legato` with one), its sequence
//! position comes up (`seq_position` of `seq_length`), the velocity is in its
//! window and the note's `rand` value is in `[lorand, hirand)`. The counters
//! and the held-note count live in the program's shared session, so round
//! robins advance across the voices of one program; `rand` is
//! `hash(seed, event serial)`, the serial counting every note-on and
//! note-off of the program. A note-off starts `release_key` regions, and
//! `release` regions while an attack region of a compatible note
//! still sounds anywhere in the program; their volume drops by `rt_decay`
//! dB per second the note was held, and they play at the note-on velocity.
//!
//! **Pitch**: `cents = keytrack · (n - keycenter) + tune + 100 · transpose +
//! bend`, played at `2^(cents/1200) · file_rate / engine_rate` frames per
//! sample, from `offset` to the region's last frame.
//!
//! **Level**: `volume` dB (+ a deterministic `amp_random` draw), times the
//! velocity gain (`amp_velcurve` points, else `(vel/127)²`, scaled by
//! `amp_veltrack`), times the `xf_vel` crossfade gains, times the amp
//! envelope (sfizz's: linear attack from `start`, hold, exponential decay
//! (-78 dB over the decay time) to `sustain`, exponential release to -80 dB
//! then a 50 ms linear fade; a sustain below 0.32 % releases at once).
//! Mono samples pan with sfizz's sin/cos law and +3 dB; stereo samples get
//! pan, `width` and +6 dB.
//!
//! **Loops**: `continuous` loops for the whole note, `sustain` until the
//! release, `one_shot` ignores note-off. Every loop crossfades its end into
//! the audio before its start over [`LOOP_CROSSFADE_S`] (1 ms, sfizz's
//! default, clamped to the frames before the loop start) on an S curve.
//!
//! **Chokes and polyphony**: a region starting in group `g` turns off every
//! sounding note-on region of the program with `off_by = g` (unless it is in
//! `g` itself on the same note), with its `off_mode` (`fast` 6 ms,
//! `normal` its own release, `time` its `off_time`); a choked note fires its
//! release regions. `note_polyphony` and `polyphony` steal as sfizz does
//! (lowest velocity first on one note and group; oldest of the region).
//! Every voice of the program sees the others through the shared session
//! table: an off requested by another voice takes effect at its sample if
//! the target renders later in the block, else at the start of its next
//! block (at most one engine block late).
//!
//! **Filters** (static, per region): 1-pole low/high-pass, RBJ biquads
//! (`Q = 10^(resonance/20)`) cascaded for 4 and 6 poles, peaking and
//! shelving; cutoff tracks key and velocity in cents. EQ bands are RBJ
//! peaking/shelving with `Q` from the bandwidth in octaves.
//!
//! **Done**: after the note started, when no region sounds and no release
//! region is still waiting for the gate to fall; the output is silent from
//! then on, so the done rule frees the voice.
//!
//! **Where it differs from sfizz.** Offsets from `offset_random` are never
//! honoured (a program carries none; rendering stays sample-deterministic).
//! The kernel defaults to the measured choice ([`DEFAULT_INTERP`]), not
//! sfizz's Hermite. Gate edges land on engine blocks (the engine applies
//! scheduled gates at block starts). Group-level polyphony limits other than
//! a region's own `polyphony` are not modelled. A note-off releases only
//! this voice's note, not every sounding note of the same key.
//!
//! **Resampler** (`tests/sample_voice_resampler.rs`): a band-limited
//! sawtooth on C4 played through each kernel; aliasing is the strongest
//! non-harmonic spectral peak in dB below the fundamental:
//!
//! | transposition | linear | Hermite | sinc (default) |
//! |---|---:|---:|---:|
//! | -12 st | -45.0 | -45.8 | -99.9 |
//! | -7 st | -47.2 | -46.0 | -98.0 |
//! | 0 st (exact copy) | -95.0 | -95.0 | -95.0 |
//! | +7 st | -38.5 | -36.7 | -91.9 |
//! | +12 st | -32.7 | -32.7 | -94.5 |
//! | 0 st, 48 kHz file | -48.5 | -47.7 | -95.0 |
//!
//! Pitch error is under 0.01 cents for every kernel. The sinc (16 zero
//! crossings, Kaiser beta 8.6, cutoff 0.88 of Nyquist scaled by the step)
//! renders a voice at a third to a half of the speed of Hermite; a note at
//! its keycenter in a sample at the engine rate is copied exactly by every
//! kernel.
//!
//! **Memory**: the samples belong to the program (`BoundProgram`); a voice
//! holds [`MAX_PLAYERS`] region players inline and allocates nothing while
//! rendering. See [`voice_bytes`].

use crate::buffer::{AudioBuffer, read_input};
use crate::context::{ProcessContext, Rate};
use crate::dsl::compiler::{ProgramUGenFactory, UGenRegistry};
use crate::node::{InputSpec, OutputSpec, UGen, UGenCategory};
use crate::sampler::bank::{BoundProgram, PlayerRecord};
use crate::sampler::interp::{Interp, SincSpan, read_pair};
use crate::sampler::pcm::SamplePcm;
use crate::sampler::program::{EqKind, FilterKind, LoopMode, OffMode, Region, Trigger};
use crate::ugens::filters::{
    BiquadState, biquad_bpf_coeffs, biquad_high_shelf_coeffs, biquad_hpf_coeffs,
    biquad_low_shelf_coeffs, biquad_lpf_coeffs, biquad_notch_coeffs, biquad_peaking_coeffs,
};
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// The kernel a voice uses when `quality` is unconnected: chosen by the
/// resampler measurement (see the module doc of
/// [`crate::sampler::interp`] and `tests/sample_voice_resampler.rs`).
pub const DEFAULT_INTERP: Interp = Interp::Sinc;
/// Loop crossfade length in seconds of the sample (sfizz's default).
pub const LOOP_CROSSFADE_S: f32 = 1e-3;
/// Region players one voice can hold at once; a note that selects more
/// regions plays the first ones in program order.
pub const MAX_PLAYERS: usize = 32;
/// Filter stages (filters plus EQ bands) per region.
pub const MAX_FILTER_SLOTS: usize = 6;

const OFF_FAST_S: f32 = 6e-3;
const EG_RELEASE_THRESHOLD: f32 = 1e-4;
const EG_TRANSITION_S: f32 = 50e-3;
const SUSTAIN_FREE_RUNNING: f32 = 0.0032;
const MONO_PAN_GAIN: f32 = 1.412_537_5;
const STEREO_PAN_GAIN: f32 = 1.995_262_3;

// ---------------------------------------------------------------------------
// Amp envelope (sfizz's ADSREnvelope, per sample)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EgState {
    Delay,
    Attack,
    Hold,
    Decay,
    Sustain,
    Release,
    Fadeout,
    Done,
}

#[derive(Debug, Clone, Copy)]
struct Env {
    state: EgState,
    value: f32,
    start: f32,
    delay: i64,
    attack_step: f32,
    hold: i64,
    decay_rate: f32,
    sustain: f32,
    release_rate: f32,
    transition_delta: f32,
    free_running: bool,
    sample_rate: f32,
}

fn lin_rate(t: f32, sr: f32) -> f32 {
    if t <= 0.0 { 1.0 } else { 1.0 / (sr * t) }
}

fn exp_rate(t: f32, sr: f32) -> f32 {
    if t <= 0.0 {
        0.0
    } else {
        (-9.0 / (t.max(OFF_FAST_S) * sr)).exp()
    }
}

fn secs_to_samples(t: f32, sr: f32) -> i64 {
    if t <= 0.0 { 0 } else { (t * sr) as i64 }
}

impl Env {
    fn new(region: &Region, extra_delay: i64, sr: f32, oscillator_one_shot: bool) -> Self {
        let e = &region.ampeg;
        let sustain = (e.sustain / 100.0).clamp(0.0, 1.0);
        let start = (e.start / 100.0).clamp(0.0, 1.0);
        Env {
            state: EgState::Delay,
            value: start,
            start,
            delay: extra_delay + secs_to_samples(e.delay, sr),
            attack_step: lin_rate(e.attack, sr),
            hold: secs_to_samples(e.hold, sr),
            decay_rate: exp_rate(e.decay, sr),
            sustain,
            release_rate: exp_rate(e.release, sr),
            transition_delta: 0.0,
            free_running: sustain <= SUSTAIN_FREE_RUNNING || oscillator_one_shot,
            sample_rate: sr,
        }
    }

    fn done(&self) -> bool {
        self.state == EgState::Done
    }
    fn released(&self) -> bool {
        self.state >= EgState::Release
    }
    fn remaining_delay(&self) -> i64 {
        if self.state == EgState::Delay {
            self.delay
        } else {
            0
        }
    }
    fn release(&mut self) {
        if self.state < EgState::Release {
            self.state = EgState::Release;
        }
    }
    fn set_release_time(&mut self, t: f32) {
        self.release_rate = exp_rate(t, self.sample_rate);
    }

    #[inline]
    fn next(&mut self) -> f32 {
        loop {
            match self.state {
                EgState::Delay => {
                    if self.delay > 0 {
                        self.delay -= 1;
                        self.value = self.start;
                        return self.value;
                    }
                    self.state = EgState::Attack;
                }
                EgState::Attack => {
                    self.value += self.attack_step;
                    if self.value < 1.0 {
                        return self.value;
                    }
                    self.value = 1.0;
                    self.state = EgState::Hold;
                }
                EgState::Hold => {
                    if self.hold > 0 {
                        self.hold -= 1;
                        return self.value;
                    }
                    self.state = EgState::Decay;
                }
                EgState::Decay => {
                    self.value *= self.decay_rate;
                    if self.value > self.sustain {
                        return self.value;
                    }
                    self.state = EgState::Sustain;
                    self.value = self.value.max(self.sustain);
                    self.transition_delta =
                        (self.sustain - self.value) / (self.sample_rate * EG_TRANSITION_S);
                }
                EgState::Sustain => {
                    if self.free_running {
                        self.state = EgState::Release;
                        continue;
                    }
                    if self.value > self.sustain {
                        self.value += self.transition_delta;
                    }
                    return self.value;
                }
                EgState::Release => {
                    let previous = self.value;
                    self.value *= self.release_rate;
                    if self.value > EG_RELEASE_THRESHOLD {
                        return self.value;
                    }
                    self.state = EgState::Fadeout;
                    self.value = previous;
                    self.transition_delta = -self.value.max(EG_RELEASE_THRESHOLD)
                        / (self.sample_rate * EG_TRANSITION_S);
                }
                EgState::Fadeout => {
                    self.value += self.transition_delta;
                    if self.value > 0.0 {
                        return self.value;
                    }
                    self.state = EgState::Done;
                    self.value = 0.0;
                }
                EgState::Done => return 0.0,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Static filters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum FilterShape {
    Off,
    /// y = (1 - p) x + p y1
    Lp1 {
        p: f32,
    },
    /// y = (1 + p)/2 (x - x1) + p y1
    Hp1 {
        p: f32,
    },
    /// One RBJ biquad run `stages` times.
    Biquad {
        c: [f32; 5],
        stages: u8,
    },
}

#[derive(Clone, Copy)]
struct FilterSlot {
    shape: FilterShape,
    /// Per channel: up to three biquad stages, or (x1, y1) of a one-pole.
    state: [[BiquadState; 3]; 2],
    one_pole: [(f32, f32); 2],
}

impl FilterSlot {
    const OFF: FilterSlot = FilterSlot {
        shape: FilterShape::Off,
        state: [[BiquadState::ZERO; 3]; 2],
        one_pole: [(0.0, 0.0); 2],
    };

    #[inline]
    fn tick(&mut self, ch: usize, x: f32) -> f32 {
        match self.shape {
            FilterShape::Off => x,
            FilterShape::Lp1 { p } => {
                let (_, y1) = self.one_pole[ch];
                let y = (1.0 - p) * x + p * y1;
                self.one_pole[ch] = (x, y);
                y
            }
            FilterShape::Hp1 { p } => {
                let (x1, y1) = self.one_pole[ch];
                let y = 0.5 * (1.0 + p) * (x - x1) + p * y1;
                self.one_pole[ch] = (x, y);
                y
            }
            FilterShape::Biquad { c, stages } => {
                let mut v = x;
                for s in 0..stages as usize {
                    v = self.state[ch][s].tick(v, c[0], c[1], c[2], c[3], c[4]);
                }
                v
            }
        }
    }
}

fn cents_factor(c: f32) -> f32 {
    (c / 1200.0).exp2()
}

/// Build a region's filters and EQ bands for `note` at MIDI velocity `vel`.
fn build_filters(
    region: &Region,
    note: u8,
    vel: u8,
    sr: f32,
) -> ([FilterSlot; MAX_FILTER_SLOTS], u8) {
    let mut slots = [FilterSlot::OFF; MAX_FILTER_SLOTS];
    let mut n = 0usize;
    let v = vel as f32 / 127.0;
    let nyq = 0.499 * sr;
    for f in &region.filters {
        if n == MAX_FILTER_SLOTS {
            break;
        }
        let mut cutoff = f.cutoff
            * cents_factor(f.keytrack * (note as f32 - f.keycenter as f32))
            * cents_factor(f.veltrack * v);
        cutoff = cutoff.clamp(0.0, 20_000.0).min(nyq).max(1.0);
        let q = 10f32.powf(f.resonance / 20.0).max(1e-3);
        let p = (-core::f32::consts::TAU * cutoff / sr).exp();
        let bq = |c: (f32, f32, f32, f32, f32), stages: u8| FilterShape::Biquad {
            c: [c.0, c.1, c.2, c.3, c.4],
            stages,
        };
        let shape = match f.kind {
            FilterKind::Lpf1p => FilterShape::Lp1 { p },
            FilterKind::Hpf1p => FilterShape::Hp1 { p },
            FilterKind::Lpf2p => bq(biquad_lpf_coeffs(cutoff, q, sr), 1),
            FilterKind::Hpf2p => bq(biquad_hpf_coeffs(cutoff, q, sr), 1),
            FilterKind::Lpf4p => bq(biquad_lpf_coeffs(cutoff, q, sr), 2),
            FilterKind::Hpf4p => bq(biquad_hpf_coeffs(cutoff, q, sr), 2),
            FilterKind::Lpf6p => bq(biquad_lpf_coeffs(cutoff, q, sr), 3),
            FilterKind::Hpf6p => bq(biquad_hpf_coeffs(cutoff, q, sr), 3),
            FilterKind::Bpf2p => bq(biquad_bpf_coeffs(cutoff, q, sr), 1),
            FilterKind::Brf2p => bq(biquad_notch_coeffs(cutoff, q, sr), 1),
            FilterKind::Peq => bq(biquad_peaking_coeffs(cutoff, q, f.gain, sr), 1),
            FilterKind::Lsh => bq(biquad_low_shelf_coeffs(cutoff, q, f.gain, sr), 1),
            FilterKind::Hsh => bq(biquad_high_shelf_coeffs(cutoff, q, f.gain, sr), 1),
        };
        slots[n].shape = shape;
        n += 1;
    }
    for b in &region.eq {
        if n == MAX_FILTER_SLOTS {
            break;
        }
        let freq = (b.freq + b.vel2freq * v).clamp(1.0, nyq);
        let gain = b.gain + b.vel2gain * v;
        let w0 = core::f32::consts::TAU * freq / sr;
        let q =
            1.0 / (2.0 * (0.5 * core::f32::consts::LN_2 * b.bw.max(1e-3) * w0 / w0.sin()).sinh());
        let c = match b.kind {
            EqKind::Peak => biquad_peaking_coeffs(freq, q, gain, sr),
            EqKind::LowShelf => biquad_low_shelf_coeffs(freq, q, gain, sr),
            EqKind::HighShelf => biquad_high_shelf_coeffs(freq, q, gain, sr),
        };
        slots[n].shape = FilterShape::Biquad {
            c: [c.0, c.1, c.2, c.3, c.4],
            stages: 1,
        };
        n += 1;
    }
    (slots, n as u8)
}

// ---------------------------------------------------------------------------
// Gain helpers
// ---------------------------------------------------------------------------

/// sfizz's velocity gain: the curve (or `v²`) scaled by `amp_veltrack`.
fn velocity_gain(region: &Region, curve: Option<&[f32]>, vel: u8) -> f32 {
    let v = vel as f32 / 127.0;
    let g = match curve {
        Some(c) => c[vel.min(127) as usize],
        None => v * v,
    };
    let vt = region.amp_veltrack / 100.0;
    let g = vt.abs() * (1.0 - g);
    if vt < 0.0 { g } else { 1.0 - g }
}

/// sfizz's `xfin`/`xfout` velocity gains, in MIDI velocities.
fn xf_gain(region: &Region, vel: u8) -> f32 {
    let Some(xf) = region.xf_vel else {
        return 1.0;
    };
    let v = vel as f32;
    let shape = |pos: f32| {
        if xf.gain_curve { pos } else { pos.sqrt() }
    };
    let mut g = 1.0;
    // fade in
    if v < xf.in_lo {
        g = 0.0;
    } else {
        let len = xf.in_hi - xf.in_lo - 1.0;
        if len > 0.0 && v < xf.in_hi {
            g *= shape((v - xf.in_lo) / len);
        }
    }
    // fade out
    let len = xf.out_hi - xf.out_lo - 1.0;
    if len > 0.0 && v > xf.out_lo {
        let pos = (v - xf.out_lo) / len;
        g *= if pos > 1.0 { 0.0 } else { shape(1.0 - pos) };
    }
    g
}

/// Output matrix `[ll, rl, lr, rr]`: `out_l = ll·l + rl·r`,
/// `out_r = lr·l + rr·r` (sfizz's pan, width and position stages).
fn pan_matrix(region: &Region, stereo: bool) -> [f32; 4] {
    let half_pi = core::f32::consts::FRAC_PI_2;
    let p = ((region.pan / 100.0 + 1.0) * 0.5).clamp(0.0, 1.0);
    let (pl, pr) = ((p * half_pi).cos(), (p * half_pi).sin());
    if !stereo {
        let (l, r) = (pl * MONO_PAN_GAIN, pr * MONO_PAN_GAIN);
        return [l, 0.0, r, 0.0];
    }
    let w = ((region.width / 100.0 + 1.0) * 0.5).clamp(0.0, 1.0);
    let (c1, c2) = ((w * half_pi).cos(), (w * half_pi).sin());
    let k = STEREO_PAN_GAIN * core::f32::consts::FRAC_1_SQRT_2;
    [k * c2 * pl, k * c1 * pr, k * c1 * pl, k * c2 * pr]
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform draw in `[0, 1)` from `(seed, serial, salt)`.
fn rand01(seed: u32, serial: u64, salt: u64) -> f32 {
    let h = splitmix64(splitmix64((seed as u64) << 32 ^ salt) ^ serial);
    (h >> 40) as f32 / (1u64 << 24) as f32
}

// ---------------------------------------------------------------------------
// Region player
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct LoopRt {
    start: u32,
    end: u32,
    size: f64,
    xf: u32,
    xf_out_start: u32,
}

struct Player {
    slot: u32,
    pcm: Option<Arc<SamplePcm>>,
    interp: Interp,
    pos: f64,
    step: f64,
    span: SincSpan,
    /// File rate / engine rate.
    rate_ratio: f64,
    /// Region pitch offset in cents without the note and bend.
    base_cents: f32,
    keytrack: f32,
    keycenter: f32,
    bend_up: f32,
    bend_down: f32,
    /// Index at or past which the sample has ended.
    last: u32,
    lp: Option<LoopRt>,
    loop_mode: LoopMode,
    data_delay: i64,
    data_ended: bool,
    env: Env,
    gain: f32,
    matrix: [f32; 4],
    filters: [FilterSlot; MAX_FILTER_SLOTS],
    n_filters: u8,
    on_note: bool,
    offed: bool,
    off_mode: OffMode,
    off_time: f32,
    release_time: f32,
}

impl Player {
    fn looping(&self) -> bool {
        match self.loop_mode {
            LoopMode::Continuous => self.lp.is_some(),
            LoopMode::Sustain => self.lp.is_some() && !self.env.released(),
            _ => false,
        }
    }

    fn set_pitch(&mut self, note_f: f32, bend: f32) {
        let bend_cents = if bend > 0.0 {
            bend * self.bend_up
        } else {
            -bend * self.bend_down
        };
        let cents = self.keytrack * (note_f - self.keycenter) + self.base_cents + bend_cents;
        let step = cents_factor(cents) as f64 * self.rate_ratio;
        if step != self.step {
            self.step = step;
            self.span = SincSpan::for_step(step);
        }
    }

    /// One frame of source audio (before filters), advancing the position.
    #[inline]
    fn read(&mut self, sinc: &crate::sampler::interp::SincTable) -> (f32, f32) {
        let Some(pcm) = self.pcm.as_ref() else {
            return (0.0, 0.0);
        };
        if self.data_ended {
            return (0.0, 0.0);
        }
        let looping = self.looping();
        if looping {
            let lp = self.lp.unwrap();
            while self.pos >= lp.end as f64 + 1.0 {
                self.pos -= lp.size;
            }
        } else if self.pos >= self.last as f64 {
            self.data_ended = true;
            return (0.0, 0.0);
        }
        let a = pcm.padded(0);
        let b = (pcm.channels() == 2).then(|| pcm.padded(1));
        let unity = self.step == 1.0 && self.pos.fract() == 0.0;
        let (mut l, mut r) = if unity {
            let i = self.pos as usize + crate::sampler::pcm::PCM_PAD;
            (a[i], b.map_or(0.0, |b| b[i]))
        } else {
            read_pair(self.interp, a, b, self.pos, sinc, self.span)
        };
        if looping {
            let lp = self.lp.unwrap();
            let idx = self.pos.floor();
            if lp.xf > 0 && idx >= lp.start as f64 && idx >= lp.xf_out_start as f64 {
                let t = ((self.pos - lp.xf_out_start as f64) / lp.xf as f64) as f32;
                let s_curve = |x: f32| {
                    let x = x.clamp(0.0, 1.0);
                    0.5 * (1.0 - (core::f32::consts::PI * x).cos())
                };
                let g_out = s_curve(1.0 - t);
                l *= g_out;
                r *= g_out;
                let in_pos = self.pos - lp.size;
                if in_pos.floor() >= 0.0 {
                    let (il, ir) = if unity {
                        let i = in_pos as usize + crate::sampler::pcm::PCM_PAD;
                        (a[i], b.map_or(0.0, |b| b[i]))
                    } else {
                        read_pair(self.interp, a, b, in_pos, sinc, self.span)
                    };
                    let g_in = s_curve(t);
                    l += g_in * il;
                    r += g_in * ir;
                }
            }
        }
        self.pos += self.step;
        (l, r)
    }
}

// ---------------------------------------------------------------------------
// The voice
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct NoteState {
    note: u8,
    vel: u8,
    note_f: f32,
    on_time: u64,
    seed: u32,
    interp: Interp,
    /// The note's release regions were fired by a choke.
    choke_released: bool,
}

/// One note of a bound sample program. See the module doc.
pub struct SampleVoice {
    prog: Arc<BoundProgram>,
    voice: u64,
    sample_rate: f32,
    gate_prev: bool,
    gate_held: bool,
    note: Option<NoteState>,
    started: bool,
    players: Vec<Player>,
    next_slot: u32,
    /// Pending offs this block: (slot, sample index).
    offs: Vec<(u32, usize)>,
    /// Regions a note wanted to start beyond [`MAX_PLAYERS`].
    dropped: u32,
    /// Block scratch (left, right), sized once.
    buf_l: Vec<f32>,
    buf_r: Vec<f32>,
}

/// Bytes one voice holds (its struct plus its preallocated region players
/// and off list); the samples belong to the program.
pub fn voice_bytes() -> usize {
    core::mem::size_of::<SampleVoice>()
        + MAX_PLAYERS * core::mem::size_of::<Player>()
        + MAX_PLAYERS * core::mem::size_of::<(u32, usize)>()
        + 2 * crate::buffer::MAX_BLOCK_SIZE * core::mem::size_of::<f32>()
}

impl SampleVoice {
    /// A voice of `prog` (registers itself in the program's session).
    pub fn new(prog: Arc<BoundProgram>) -> Self {
        let voice = prog.session.with(|s| s.new_voice());
        SampleVoice {
            prog,
            voice,
            sample_rate: 44_100.0,
            gate_prev: false,
            gate_held: false,
            note: None,
            started: false,
            players: Vec::with_capacity(MAX_PLAYERS),
            next_slot: 0,
            offs: Vec::with_capacity(MAX_PLAYERS),
            dropped: 0,
            buf_l: alloc::vec![0.0; crate::buffer::MAX_BLOCK_SIZE],
            buf_r: alloc::vec![0.0; crate::buffer::MAX_BLOCK_SIZE],
        }
    }

    /// Regions the last notes could not start for want of a player slot.
    pub fn dropped_regions(&self) -> u32 {
        self.dropped
    }

    /// Start region `ri` (choke and polyphony checks against the session
    /// first, as sfizz does). Runs with the session locked.
    #[allow(clippy::too_many_arguments)]
    fn start_region(
        &mut self,
        s: &mut crate::sampler::bank::Session,
        ri: u32,
        ns: &NoteState,
        serial: u64,
        on_note: bool,
        at: u64,
        held_secs: f32,
    ) {
        let prog = self.prog.clone();
        let region = prog.region(ri);
        // chokes: every sounding note-on region with off_by == this group
        for rec in s.players.iter_mut() {
            if !rec.offed
                && rec.on_note
                && rec.off_by == Some(region.group)
                && (rec.group != region.group || rec.note != ns.note)
            {
                rec.offed = true;
                rec.off_at = Some(at);
            }
        }
        // note polyphony: same group and note; lowest velocity first
        if let Some(np) = region.note_polyphony {
            let mut count = s
                .players
                .iter()
                .filter(|r| !r.offed && r.group == region.group && r.note == ns.note)
                .count() as u32;
            while count > np.saturating_sub(1) {
                let cand = s
                    .players
                    .iter()
                    .filter(|r| {
                        !r.offed && r.group == region.group && r.note == ns.note && r.vel <= ns.vel
                    })
                    .min_by_key(|r| (r.vel, r.start))
                    .copied();
                let Some(c) = cand else { break };
                s.off_with_sisters(c.voice, c.start, at);
                count -= 1;
            }
        }
        // region polyphony: oldest instance of this region
        if let Some(poly) = region.polyphony {
            let count = s
                .players
                .iter()
                .filter(|r| !r.offed && r.region == ri)
                .count() as u32;
            if count >= poly
                && let Some(c) = s
                    .players
                    .iter()
                    .filter(|r| !r.offed && r.region == ri)
                    .min_by_key(|r| r.start)
                    .copied()
            {
                s.off_with_sisters(c.voice, c.start, at);
            }
        }
        if self.players.len() >= MAX_PLAYERS {
            self.dropped += 1;
            return;
        }
        let sr = self.sample_rate;
        let pcm = prog.sample_of(ri).cloned();
        let frames = pcm.as_ref().map_or(u32::MAX, |p| p.frames() as u32);
        let last = match region.end_frames {
            Some(e) => e.min(frames.saturating_sub(1)),
            None => frames.saturating_sub(1),
        };
        let file_rate = pcm.as_ref().map_or(sr, |p| p.sample_rate());
        let lp = match (region.loop_mode, region.loop_points) {
            (LoopMode::Continuous | LoopMode::Sustain, Some((a, b))) if b < frames => {
                let end = b.min(last).max(a);
                let xf = ((LOOP_CROSSFADE_S * file_rate).round() as u32).min(a);
                Some(LoopRt {
                    start: a,
                    end,
                    size: (end + 1 - a) as f64,
                    xf,
                    xf_out_start: end + 1 - xf,
                })
            }
            _ => None,
        };
        let region_delay = secs_to_samples(region.delay_s, sr);
        let oscillator_one_shot = pcm.is_none() && region.loop_mode == LoopMode::OneShot;
        let env = Env::new(region, region_delay, sr, oscillator_one_shot);
        let mut vol_db = region.volume_db;
        if region.amp_random_db > 0.0 {
            vol_db += region.amp_random_db * rand01(ns.seed, serial, 0x1_0000 + ri as u64);
        }
        if matches!(region.trigger, Trigger::Release | Trigger::ReleaseKey) {
            vol_db -= region.rt_decay_db_per_s * held_secs;
        }
        let gain = 10f32.powf(vol_db / 20.0)
            * velocity_gain(region, prog.velcurve(ri), ns.vel)
            * xf_gain(region, ns.vel);
        let stereo = pcm.as_ref().is_some_and(|p| p.channels() == 2);
        let (filters, n_filters) = build_filters(region, ns.note, ns.vel, sr);
        let slot = self.next_slot;
        self.next_slot = self.next_slot.wrapping_add(1);
        let mut p = Player {
            slot,
            pcm,
            interp: ns.interp,
            pos: region.offset_frames as f64,
            step: 0.0,
            span: SincSpan::for_step(1.0),
            rate_ratio: file_rate as f64 / sr as f64,
            base_cents: region.tune_cents + 100.0 * region.transpose as f32,
            keytrack: region.pitch_keytrack,
            keycenter: region.pitch_keycenter as f32,
            bend_up: region.bend_up_cents,
            bend_down: region.bend_down_cents,
            last,
            lp,
            loop_mode: region.loop_mode,
            data_delay: region_delay,
            data_ended: false,
            env,
            gain,
            matrix: pan_matrix(region, stereo),
            filters,
            n_filters,
            on_note,
            offed: false,
            off_mode: region.off_mode,
            off_time: region.off_time,
            release_time: region.ampeg.release,
        };
        p.set_pitch(ns.note_f, 0.0);
        s.players.push(PlayerRecord {
            voice: self.voice,
            slot,
            region: ri,
            group: region.group,
            off_by: region.off_by,
            note: ns.note,
            vel: ns.vel,
            on_note,
            start: serial,
            offed: false,
            off_at: None,
        });
        self.players.push(p);
    }

    /// Take the offs other voices (or this one) requested for this voice's
    /// regions; `t0` is the block's first sample time.
    fn collect_offs(&mut self, s: &mut crate::sampler::bank::Session, t0: u64, n: usize) {
        for rec in s.players.iter_mut() {
            if rec.voice == self.voice
                && let Some(at) = rec.off_at.take()
            {
                let idx = at.saturating_sub(t0).min(n.saturating_sub(1) as u64) as usize;
                if self.offs.len() < self.offs.capacity() {
                    self.offs.push((rec.slot, idx));
                }
            }
        }
    }

    fn note_on(&mut self, freq: f32, vel: f32, seed: f32, quality: Option<f32>, at: u64) {
        let note_f = if freq > 0.0 && freq.is_finite() {
            69.0 + 12.0 * (freq / 440.0).log2()
        } else {
            60.0
        };
        let note = note_f.round().clamp(0.0, 127.0) as u8;
        let vel = if vel.is_finite() {
            (vel * 127.0).round().clamp(0.0, 127.0) as u8
        } else {
            100
        };
        let interp = quality.map_or(DEFAULT_INTERP, |q| Interp::from_quality(q, DEFAULT_INTERP));
        let ns = NoteState {
            note,
            vel,
            note_f,
            on_time: at,
            seed: seed.to_bits(),
            interp,
            choke_released: false,
        };
        self.note = Some(ns);
        self.started = true;
        self.gate_held = true;
        let prog = self.prog.clone();
        prog.session.with(|s| {
            s.event_serial += 1;
            let serial = s.event_serial;
            s.held += 1;
            let held = s.held;
            let rand = rand01(ns.seed, serial, 0);
            for &ri in prog.regions_for_key(note) {
                let r = prog.region(ri);
                let (len, pos) = r.seq.unwrap_or((1, 1));
                let c = s.seq_counter[ri as usize];
                s.seq_counter[ri as usize] = c.wrapping_add(1);
                let seq_on = c % len == pos - 1;
                s.seq_on[ri as usize] = seq_on;
                let trig_ok = match r.trigger {
                    Trigger::Attack => true,
                    Trigger::First => held == 1,
                    Trigger::Legato => held > 1,
                    Trigger::Release | Trigger::ReleaseKey => false,
                };
                let rand_ok = r.rand.is_none_or(|(lo, hi)| lo <= rand && rand < hi);
                if trig_ok && seq_on && (r.lovel..=r.hivel).contains(&vel) && rand_ok {
                    self.start_region(s, ri, &ns, serial, true, at, 0.0);
                }
            }
        });
    }

    /// Fire the note's release regions (sfizz's `noteOffDispatch`).
    fn dispatch_release(&mut self, at: u64) {
        let Some(ns) = self.note else { return };
        let prog = self.prog.clone();
        let held_secs = at.saturating_sub(ns.on_time) as f32 / self.sample_rate;
        prog.session.with(|s| {
            s.event_serial += 1;
            let serial = s.event_serial;
            let rand = rand01(ns.seed, serial, 0);
            for &ri in prog.regions_for_key(ns.note) {
                let r = prog.region(ri);
                if !matches!(r.trigger, Trigger::Release | Trigger::ReleaseKey) {
                    continue;
                }
                let rand_ok = r.rand.is_none_or(|(lo, hi)| lo <= rand && rand < hi);
                if !(s.seq_on[ri as usize] && (r.lovel..=r.hivel).contains(&ns.vel) && rand_ok) {
                    continue;
                }
                if r.trigger == Trigger::Release {
                    let attack_playing = s.players.iter().any(|rec| {
                        rec.on_note
                            && (r.lokey..=r.hikey).contains(&rec.note)
                            && (r.lovel..=r.hivel).contains(&rec.vel)
                    });
                    if !attack_playing {
                        continue;
                    }
                }
                self.start_region(s, ri, &ns, serial, false, at, held_secs);
            }
        });
    }

    fn note_off(&mut self, at: u64) {
        if !self.gate_held {
            return;
        }
        self.gate_held = false;
        self.prog
            .session
            .with(|s| s.held = s.held.saturating_sub(1));
        // release this note's attack regions (one-shots ignore note-off)
        let mut killed = false;
        for p in self.players.iter_mut() {
            if !p.on_note || p.loop_mode == LoopMode::OneShot {
                continue;
            }
            if p.env.remaining_delay() > 0 {
                p.env.state = EgState::Done;
                killed = true;
            } else {
                p.env.release();
            }
        }
        if killed {
            self.reap();
        }
        self.dispatch_release(at);
    }

    /// Turn off the player in `slot` at block index `i` (`at` absolute).
    fn apply_off(&mut self, slot: u32, at: u64) {
        let mut fire_release = false;
        if let Some(p) = self.players.iter_mut().find(|p| p.slot == slot) {
            if p.offed {
                return;
            }
            p.offed = true;
            match p.off_mode {
                OffMode::Fast => p.env.set_release_time(OFF_FAST_S),
                OffMode::Time => p.env.set_release_time(p.off_time),
                OffMode::Normal => p.env.set_release_time(p.release_time),
            }
            if p.env.remaining_delay() > 0 {
                p.env.state = EgState::Done;
            } else {
                p.env.release();
            }
            fire_release = p.on_note;
        }
        self.reap();
        if fire_release
            && self.gate_held
            && let Some(ns) = self.note.as_mut()
            && !ns.choke_released
        {
            ns.choke_released = true;
            self.dispatch_release(at);
        }
    }

    /// Drop finished players and their session records.
    fn reap(&mut self) {
        if !self.players.iter().any(|p| p.env.done()) {
            return;
        }
        let voice = self.voice;
        let prog = self.prog.clone();
        prog.session.with(|s| {
            for p in self.players.iter().filter(|p| p.env.done()) {
                s.players
                    .retain(|r| !(r.voice == voice && r.slot == p.slot));
            }
        });
        self.players.retain(|p| !p.env.done());
    }

    /// Render every player over `a..b` into `out_l`/`out_r`.
    fn render(&mut self, a: usize, b: usize, out_l: &mut [f32], out_r: &mut [f32]) {
        if a >= b {
            return;
        }
        let prog = self.prog.clone();
        let sinc = prog.sinc();
        let mut ended = [0u32; MAX_PLAYERS];
        let mut n_ended = 0usize;
        for p in self.players.iter_mut() {
            let stereo = p.pcm.as_ref().is_some_and(|x| x.channels() == 2);
            let [ll, rl, lr, rr] = p.matrix;
            for j in a..b {
                let e = p.env.next();
                if p.env.done() {
                    break;
                }
                let (mut l, mut r) = if p.data_delay > 0 {
                    p.data_delay -= 1;
                    (0.0, 0.0)
                } else {
                    let was_ended = p.data_ended;
                    let v = p.read(sinc);
                    if p.data_ended && !was_ended && !p.offed {
                        // sfizz's off(fast) at the end of the sample
                        p.offed = true;
                        p.env.set_release_time(OFF_FAST_S);
                        p.env.release();
                        if n_ended < MAX_PLAYERS {
                            ended[n_ended] = p.slot;
                            n_ended += 1;
                        }
                    }
                    v
                };
                for k in 0..p.n_filters as usize {
                    l = p.filters[k].tick(0, l);
                    if stereo {
                        r = p.filters[k].tick(1, r);
                    }
                }
                let g = p.gain * e;
                out_l[j] += g * (ll * l + rl * r);
                out_r[j] += g * (lr * l + rr * r);
            }
        }
        if n_ended > 0 {
            let voice = self.voice;
            let ended = &ended[..n_ended];
            self.prog.session.with(|s| {
                for rec in s.players.iter_mut() {
                    if rec.voice == voice && ended.contains(&rec.slot) {
                        rec.offed = true;
                    }
                }
            });
        }
        self.reap();
    }

    fn is_finished(&self) -> bool {
        if !self.started || !self.players.is_empty() {
            return false;
        }
        match self.note {
            Some(ns) if self.gate_held => !self.prog.has_release_for(ns.note, ns.vel),
            _ => true,
        }
    }
}

impl Drop for SampleVoice {
    fn drop(&mut self) {
        let voice = self.voice;
        let held = self.gate_held;
        self.prog.session.with(|s| {
            s.players.retain(|r| r.voice != voice);
            if held {
                s.held = s.held.saturating_sub(1);
            }
        });
    }
}

impl UGen for SampleVoice {
    ugen_spec!(
        "SampleVoice",
        category = Oscillator,
        inputs = [],
        optional_inputs = ["freq", "gate", "vel", "amp", "seed", "bend", "quality"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        let voice = self.voice;
        let held = self.gate_held;
        self.prog.session.with(|s| {
            s.players.retain(|r| r.voice != voice);
            if held {
                s.held = s.held.saturating_sub(1);
            }
        });
        self.players.clear();
        self.offs.clear();
        self.gate_prev = false;
        self.gate_held = false;
        self.note = None;
        self.started = false;
    }

    fn output_channels(&self, _input_channels: &[usize]) -> usize {
        2
    }

    fn process(
        &mut self,
        context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let n = context.block_size;
        output.channel_mut(0).samples_mut()[..n].fill(0.0);
        output.channel_mut(1).samples_mut()[..n].fill(0.0);
        let freq = inputs[0];
        let gate = inputs[1];
        let vel = inputs[2];
        let amp = inputs[3];
        let seed = inputs[4];
        let bend = inputs[5];
        let quality = inputs[6];
        let t0 = context.sample_offset;

        // Offs requested since the last block.
        let prog = self.prog.clone();
        prog.session.with(|s| self.collect_offs(s, t0, n));

        // Retune sounding regions from this block's freq and bend.
        if let Some(ns) = self.note {
            let f = read_input(freq, 0, 0, 0.0);
            let note_f = if f > 0.0 && f.is_finite() {
                69.0 + 12.0 * (f / 440.0).log2()
            } else {
                ns.note_f
            };
            let b = read_input(bend, 0, 0, 0.0);
            let b = if b.is_finite() {
                b.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            for p in self.players.iter_mut() {
                p.set_pitch(note_f, b);
            }
        }

        let mut out_l = core::mem::take(&mut self.buf_l);
        let mut out_r = core::mem::take(&mut self.buf_r);
        if out_l.len() < n {
            out_l.resize(n, 0.0);
            out_r.resize(n, 0.0);
        }
        out_l[..n].fill(0.0);
        out_r[..n].fill(0.0);
        let mut cur = 0usize;
        loop {
            // next event: a gate edge or a pending off
            let mut next_gate = n;
            for i in cur..n {
                let g = read_input(gate, 0, i, 1.0) > 0.0;
                if g != self.gate_prev {
                    next_gate = i;
                    break;
                }
            }
            let next_off = self
                .offs
                .iter()
                .map(|&(_, i)| i)
                .filter(|&i| i >= cur)
                .min()
                .unwrap_or(n);
            let ev = next_gate.min(next_off);
            self.render(cur, ev, &mut out_l, &mut out_r);
            cur = ev;
            if ev >= n {
                break;
            }
            if next_off == ev {
                while let Some(k) = self.offs.iter().position(|&(_, i)| i == ev) {
                    let (slot, _) = self.offs.swap_remove(k);
                    self.apply_off(slot, t0 + ev as u64);
                }
                continue;
            }
            // gate edge at `ev`
            let g = read_input(gate, 0, ev, 1.0) > 0.0;
            self.gate_prev = g;
            if g {
                self.note_on(
                    read_input(freq, 0, ev, 261.625_58),
                    read_input(vel, 0, ev, 100.0 / 127.0),
                    read_input(seed, 0, ev, 0.0),
                    quality.map(|q| q.channel(0).samples()[ev]),
                    t0 + ev as u64,
                );
                // offs this note-on put on this voice's own regions
                let voice = self.voice;
                while let Some((slot, at)) = prog.session.with(|s| {
                    s.players
                        .iter_mut()
                        .filter(|r| r.voice == voice)
                        .find_map(|r| r.off_at.take().map(|at| (r.slot, at)))
                }) {
                    self.apply_off(slot, at);
                }
            } else {
                self.note_off(t0 + ev as u64);
            }
        }
        self.offs.clear();

        if let Some(a) = amp {
            let a = a.channel(0).samples();
            for i in 0..n {
                out_l[i] *= a[i];
                out_r[i] *= a[i];
            }
        }
        output.channel_mut(0).samples_mut()[..n].copy_from_slice(&out_l[..n]);
        output.channel_mut(1).samples_mut()[..n].copy_from_slice(&out_r[..n]);
        self.buf_l = out_l;
        self.buf_r = out_r;
    }

    fn is_done(&self) -> bool {
        self.is_finished()
    }
}

/// Register the `sampleVoice` program-bound kind.
pub fn register(reg: &mut UGenRegistry) {
    let factory: ProgramUGenFactory = Arc::new(|prog| Box::new(SampleVoice::new(prog)));
    let port = |name| InputSpec {
        name,
        rate: Rate::Audio,
        required: false,
    };
    reg.register_program_bound(
        "sampleVoice",
        factory,
        UGenCategory::Oscillator,
        &[
            port("freq"),
            port("gate"),
            port("vel"),
            port("amp"),
            port("seed"),
            port("bend"),
            port("quality"),
        ],
        &[OutputSpec {
            name: "out",
            rate: Rate::Audio,
        }],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region() -> Region {
        Region::default()
    }

    #[test]
    fn velocity_gain_follows_sfizz() {
        let r = region();
        assert!((velocity_gain(&r, None, 127) - 1.0).abs() < 1e-6);
        let v = 64.0f32 / 127.0;
        assert!((velocity_gain(&r, None, 64) - v * v).abs() < 1e-6);
        let r0 = Region {
            amp_veltrack: 0.0,
            ..region()
        };
        assert_eq!(velocity_gain(&r0, None, 1), 1.0);
        let r50 = Region {
            amp_veltrack: 50.0,
            ..region()
        };
        assert!((velocity_gain(&r50, None, 0) - 0.5).abs() < 1e-6);
        let rn = Region {
            amp_veltrack: -100.0,
            ..region()
        };
        assert!((velocity_gain(&rn, None, 127) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn xf_gain_matches_sfizz_ranges() {
        let r = Region {
            xf_vel: Some(crate::sampler::program::VelCrossfade {
                in_lo: 20.0,
                in_hi: 61.0,
                out_lo: 80.0,
                out_hi: 121.0,
                gain_curve: true,
            }),
            ..region()
        };
        assert_eq!(xf_gain(&r, 10), 0.0);
        assert_eq!(xf_gain(&r, 20), 0.0);
        assert!((xf_gain(&r, 40) - 0.5).abs() < 1e-6);
        assert_eq!(xf_gain(&r, 70), 1.0);
        assert!((xf_gain(&r, 100) - 0.5).abs() < 1e-6);
        assert_eq!(xf_gain(&r, 127), 0.0);
    }

    #[test]
    fn centred_pan_is_unity_within_sfizz_rounding() {
        let m = pan_matrix(&region(), false);
        assert!((m[0] - 0.998_81).abs() < 1e-4 && (m[2] - 0.998_81).abs() < 1e-4);
        let m = pan_matrix(&region(), true);
        // full width, centre: L -> L, R -> R at 0.9976
        assert!((m[0] - 0.997_63).abs() < 1e-4 && m[1].abs() < 1e-6);
        assert!(m[2].abs() < 1e-6 && (m[3] - 0.997_63).abs() < 1e-4);
    }

    #[test]
    fn envelope_shapes_follow_sfizz() {
        let sr = 1000.0;
        let r = Region {
            ampeg: crate::sampler::program::Envelope {
                delay: 0.0,
                attack: 0.01,
                hold: 0.005,
                decay: 0.1,
                sustain: 50.0,
                release: 0.1,
                start: 0.0,
            },
            ..region()
        };
        let mut e = Env::new(&r, 0, sr, false);
        // linear attack in steps of 1/10, then 5 samples held at 1
        let mut v = alloc::vec::Vec::new();
        loop {
            let x = e.next();
            v.push(x);
            if x == 1.0 {
                break;
            }
        }
        assert!(
            (v[0] - 0.1).abs() < 1e-6 && (10..=11).contains(&v.len()),
            "{v:?}"
        );
        for _ in 0..4 {
            assert_eq!(e.next(), 1.0);
        }
        let d = e.next();
        assert!((d - (-9.0f32 / 100.0).exp()).abs() < 1e-6, "{d}");
        for _ in 0..400 {
            e.next();
        }
        assert!((e.next() - 0.5).abs() < 1e-6);
        e.release();
        let rv = e.next();
        assert!((rv - 0.5 * (-9.0f32 / 100.0).exp()).abs() < 1e-6);
        let mut k = 0;
        while !e.done() && k < 10_000 {
            e.next();
            k += 1;
        }
        // -80 dB at 9/100 nepers per sample, then 50 ms of fade
        assert!(k > 80 && k < 150, "{k}");
    }

    #[test]
    fn rand01_is_deterministic_and_spread() {
        assert_eq!(rand01(1, 2, 0), rand01(1, 2, 0));
        assert_ne!(rand01(1, 2, 0), rand01(2, 2, 0));
        let mean: f32 = (0..1000).map(|i| rand01(7, i, 0)).sum::<f32>() / 1000.0;
        assert!((mean - 0.5).abs() < 0.05);
        assert!((0..1000).all(|i| (0.0..1.0).contains(&rand01(3, i, 9))));
    }
}
