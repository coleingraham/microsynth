//! FM synthesis UGen.
//!
//! Frequency modulation oscillator for DX7-style timbres, metallic bells,
//! and electric piano sounds.

use crate::buffer::{AudioBuffer, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use core::f32::consts::TAU;

// --- FmOsc ---

/// Two-operator FM synthesis oscillator.
///
/// A carrier sine oscillator whose instantaneous frequency is modulated by
/// a modulator sine oscillator. The modulator's amplitude (scaled by `index`)
/// determines the amount of frequency deviation, which controls the harmonic
/// richness of the output.
///
/// `output = sin(2π * carrier_phase + index * sin(2π * mod_phase + feedback * prev_mod))`
///
/// Inputs:
/// - `freq`: carrier frequency in Hz (default 440)
/// - `ratio`: modulator-to-carrier frequency ratio (default 1.0).
///   Integer ratios (1, 2, 3...) produce harmonic spectra; non-integer
///   ratios produce inharmonic/metallic timbres.
/// - `index`: modulation index (default 1.0). Controls brightness:
///   0 = pure sine, 1–3 = mild FM, 5+ = bright/metallic.
///   The modulator's peak deviation = index * mod_freq.
/// - `feedback`: modulator self-feedback (default 0.0, range 0.0–1.0).
///   The modulator feeds its previous output back into its own phase,
///   producing increasingly harsh, noise-like timbres. Essential for
///   DX7-style brass and sync lead sounds.
///
/// Classic patches:
/// - Electric piano (DX7): ratio=1, index=1.5–3, feedback=0
/// - Bell: ratio=1.4, index=3–8, feedback=0
/// - Brass: ratio=1, index=0→5 (envelope on index), feedback=0.3–0.7
/// - Bass: ratio=0.5, index=2–4, feedback=0
pub struct FmOsc {
    carrier_phase: f32,
    mod_phase: f32,
    prev_mod_out: f32,
    sample_rate: f32,
}

impl Default for FmOsc {
    fn default() -> Self {
        Self::new()
    }
}

impl FmOsc {
    pub fn new() -> Self {
        FmOsc {
            carrier_phase: 0.0,
            mod_phase: 0.0,
            prev_mod_out: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for FmOsc {
    ugen_spec!(
        "FmOsc",
        category = Oscillator,
        inputs = [],
        optional_inputs = ["freq", "ratio", "index", "feedback"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.carrier_phase = 0.0;
        self.mod_phase = 0.0;
        self.prev_mod_out = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied().flatten();
        let ratio_buf = inputs.get(1).copied().flatten();
        let index_buf = inputs.get(2).copied().flatten();
        let feedback_buf = inputs.get(3).copied().flatten();
        let inv_sr = 1.0 / self.sample_rate;

        // Snapshot once, before the channel loop: every channel must start
        // from the same block-start state (see filters::OnePole's
        // process() comment for the read-back-inside-loop bug this avoids).
        let carrier_phase_start = self.carrier_phase;
        let mod_phase_start = self.mod_phase;
        let prev_mod_out_start = self.prev_mod_out;

        for ch in 0..output.num_channels() {
            let mut carrier_phase = carrier_phase_start;
            let mut mod_phase = mod_phase_start;
            let mut prev_mod_out = prev_mod_out_start;
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let freq = read_input(freq_buf, ch, i, 440.0);
                let ratio = read_input(ratio_buf, ch, i, 1.0).max(0.0);
                let index = read_input(index_buf, ch, i, 1.0).max(0.0);
                let feedback = read_input(feedback_buf, ch, i, 0.0).clamp(0.0, 1.0);

                let mod_freq = freq * ratio;

                // Modulator with self-feedback: feeds previous output back into phase
                let modulator = (mod_phase * TAU + feedback * prev_mod_out).sin();
                prev_mod_out = modulator;

                // Carrier with phase modulation
                *out_sample = (carrier_phase * TAU + index * modulator).sin();

                // Advance phases
                carrier_phase += freq * inv_sr;
                carrier_phase -= carrier_phase.floor();
                mod_phase += mod_freq * inv_sr;
                mod_phase -= mod_phase.floor();
            }

            if ch == 0 {
                self.carrier_phase = carrier_phase;
                self.mod_phase = mod_phase;
                self.prev_mod_out = prev_mod_out;
            }
        }
    }
}

// --- FmOp ---

/// OPL-family operator waveforms, selected by `FmOp`'s `wave` input.
///
/// Indices 0–3 are the OPL2 (YM3812) set, 4–7 the extra OPL3 (YMF262)
/// shapes. Each is a function of the operator phase `p` in `[0, 1)`.
/// Shapes 1, 2, 3 and 5 are not zero-mean: they carry DC, as on the chips,
/// so a carrier using them wants a DC-blocking high-pass downstream.
pub mod fm_wave {
    /// 0: sine.
    pub const SINE: u32 = 0;
    /// 1: half-sine (positive half, silent negative half).
    pub const HALF_SINE: u32 = 1;
    /// 2: absolute sine (both halves positive; doubles the pitch).
    pub const ABS_SINE: u32 = 2;
    /// 3: quarter-sine / "pulse-sine" (rising quarter of each half, then 0).
    pub const QUARTER_SINE: u32 = 3;
    /// 4: alternating sine (OPL3; a full sine at twice the rate in the first
    /// half, silent second half).
    pub const ALTERNATING_SINE: u32 = 4;
    /// 5: camel sine (OPL3; absolute double-rate sine in the first half).
    pub const CAMEL_SINE: u32 = 5;
    /// 6: square (OPL3).
    pub const SQUARE: u32 = 6;
    /// 7: logarithmic sawtooth / "derived square" (OPL3; an exponential
    /// decay from +1 over the first half, its mirror image toward -1 over
    /// the second).
    pub const LOG_SAW: u32 = 7;
    /// Number of shapes; `wave` values are rounded and clamped to
    /// `0..COUNT`.
    pub const COUNT: u32 = 8;
}

/// Evaluate OPL waveform `wave` (see [`fm_wave`]) at phase `p` in cycles.
/// `p` may be any real; only its fractional part matters.
#[inline]
pub fn fm_wave_shape(wave: u32, p: f32) -> f32 {
    let p = p - p.floor();
    let s = (p * TAU).sin();
    match wave {
        fm_wave::HALF_SINE => {
            if p < 0.5 {
                s
            } else {
                0.0
            }
        }
        fm_wave::ABS_SINE => s.abs(),
        fm_wave::QUARTER_SINE => {
            if (p % 0.5) < 0.25 {
                s.abs()
            } else {
                0.0
            }
        }
        fm_wave::ALTERNATING_SINE => {
            if p < 0.5 {
                (2.0 * p * TAU).sin()
            } else {
                0.0
            }
        }
        fm_wave::CAMEL_SINE => {
            if p < 0.5 {
                (2.0 * p * TAU).sin().abs()
            } else {
                0.0
            }
        }
        fm_wave::SQUARE => {
            if p < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        fm_wave::LOG_SAW => {
            // 2^(-16 p): 0 dB at the start of the half, about -48 dB at its
            // end (the OPL3 attenuation ramp), mirrored into the second half.
            if p < 0.5 {
                (-16.0 * p).exp2()
            } else {
                -(-16.0 * (1.0 - p)).exp2()
            }
        }
        _ => s,
    }
}

/// One phase-modulation operator: the building block of DX7 / YM2612 /
/// OPL-style FM patches.
///
/// ```text
/// out = level * wave(phase + (pm + fb) / 2π)
/// fb  = feedback * π * (y[n-1] + y[n-2]) / 2      (y = the unscaled wave)
/// ```
///
/// Operators are wired into algorithms by ordinary graph edges: a
/// modulator's `out` goes into a carrier's `pm` port (sum several
/// modulators with `+` first), carriers are summed into the voice output.
///
/// Inputs (all optional, all audio-rate):
/// - `freq`: operator frequency in **Hz** (default 440). A ratio operator is
///   `freq * ratio` in the graph; a fixed-frequency operator (DX7 "FIXED"
///   mode) is a constant. Taking Hz rather than a ratio keeps one idiom for
///   both and leaves keytracking to the graph.
/// - `pm`: phase modulation input in **radians** (default 0). Because
///   `level` scales the output directly, a modulator's `level` *is* the
///   peak phase deviation it imposes on the operator it feeds, i.e. the
///   classical modulation index `I` (Chowning's `d/m`). A 2-op stack
///   `fmOp f*r 0 I ⟶ fmOp f (that) 1` is exactly `fmOsc f r I 0`.
/// - `level`: output amplitude (default 1). Meant for an envelope: on a
///   carrier it is loudness, on a modulator it is the index envelope.
/// - `feedback`: self phase modulation (default 0). `feedback * π` radians
///   times the mean of the operator's last two unscaled outputs, the DX7 /
///   OPL scheme (the two-sample mean is what keeps high settings from
///   oscillating at Nyquist). `1.0` matches DX7 feedback 7 (each DX7 step
///   below halves it: `feedback = 2^(fb - 7)`). On a sine operator `0.25`
///   brightens it, `0.5` (DX7 6) is close to a sawtooth (harmonic `k` at
///   about `1/k`), `1.0` is a sawtooth smeared with noise, and above that it
///   is mostly noise; values up to `4.0` are accepted, reaching OPL's FB 7
///   (4π) for percussion. Feedback acts on the unscaled wave, so it
///   does not fall with `level`; drive this port with an envelope for
///   feedback that decays.
/// - `wave`: waveform index (default 0, sine), read per sample, rounded and
///   clamped to `0..=7`; see [`fm_wave`] for the OPL2/OPL3 set.
/// - `phase_reset`: on a rising edge (previous sample `<= 0`, this one
///   `> 0`) the phase restarts at 0, for deterministic attacks (DX7
///   "oscillator key sync").
///
/// Aliasing: like the chips, `FmOp` is not band-limited. A sine operator
/// with index `I` has a Carson bandwidth of about `2(I+1)·f_mod`; partials
/// past Nyquist fold back. High indices on high notes, high feedback, and
/// the discontinuous OPL shapes (square, half/quarter sines, log-saw) all
/// alias audibly, as they do on an OPL3 at 49.7 kHz and more so here at
/// 44.1/48 kHz. Keytrack the index downward (rate/level scaling) where that
/// matters.
pub struct FmOp {
    phase: f32,
    y1: f32,
    y2: f32,
    prev_reset: f32,
    sample_rate: f32,
}

impl Default for FmOp {
    fn default() -> Self {
        Self::new()
    }
}

impl FmOp {
    pub fn new() -> Self {
        FmOp {
            phase: 0.0,
            y1: 0.0,
            y2: 0.0,
            prev_reset: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for FmOp {
    ugen_spec!(
        "FmOp",
        category = Oscillator,
        inputs = [],
        optional_inputs = ["freq", "pm", "level", "feedback", "wave", "phase_reset"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
        self.prev_reset = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied().flatten();
        let pm_buf = inputs.get(1).copied().flatten();
        let level_buf = inputs.get(2).copied().flatten();
        let feedback_buf = inputs.get(3).copied().flatten();
        let wave_buf = inputs.get(4).copied().flatten();
        let reset_buf = inputs.get(5).copied().flatten();
        let inv_sr = 1.0 / self.sample_rate;
        let inv_tau = 1.0 / TAU;

        // Every channel starts from the same block-start state (see FmOsc).
        let (phase0, y1_0, y2_0, reset0) = (self.phase, self.y1, self.y2, self.prev_reset);

        for ch in 0..output.num_channels() {
            let (mut phase, mut y1, mut y2, mut prev_reset) = (phase0, y1_0, y2_0, reset0);
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let freq = read_input(freq_buf, ch, i, 440.0);
                let pm = read_input(pm_buf, ch, i, 0.0);
                let level = read_input(level_buf, ch, i, 1.0);
                let feedback = read_input(feedback_buf, ch, i, 0.0).clamp(0.0, 4.0);
                let wave = read_input(wave_buf, ch, i, 0.0);
                let reset = read_input(reset_buf, ch, i, 0.0);

                if reset > 0.0 && prev_reset <= 0.0 {
                    phase = 0.0;
                    y1 = 0.0;
                    y2 = 0.0;
                }
                prev_reset = reset;

                let wave_idx = if wave.is_finite() {
                    (wave + 0.5).floor().clamp(0.0, (fm_wave::COUNT - 1) as f32) as u32
                } else {
                    fm_wave::SINE
                };

                let fb = feedback * core::f32::consts::PI * 0.5 * (y1 + y2);
                let y = if wave_idx == fm_wave::SINE {
                    // Same expression as FmOsc's carrier, so a 2-op stack of
                    // FmOps reproduces it to rounding.
                    (phase * TAU + (pm + fb)).sin()
                } else {
                    fm_wave_shape(wave_idx, phase + (pm + fb) * inv_tau)
                };
                let y = if y.is_finite() { y } else { 0.0 };
                y2 = y1;
                y1 = y;
                *out_sample = level * y;

                phase += freq * inv_sr;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
                self.y1 = y1;
                self.y2 = y2;
                self.prev_reset = prev_reset;
            }
        }
    }
}
