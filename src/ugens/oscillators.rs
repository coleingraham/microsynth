//! Oscillator UGens: SinOsc, Saw, Pulse, Tri, Phasor.
//!
//! All oscillators use a phase accumulator in [0, 1) and produce output
//! in the range [-1, 1] (except Phasor which outputs [0, 1)).
//!
//! Inputs:
//! - `freq`: frequency in Hz (audio rate, per-sample modulation supported)
//! - `phase` (SinOsc only): phase offset in radians
//! - `width` (Pulse only): pulse width in [0, 1]

use crate::buffer::{AudioBuffer, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use core::f32::consts::TAU;

// --- SinOsc ---

/// Sine oscillator.
///
/// Inputs: freq (Hz), phase (radians offset).
/// Output: sin(2*pi*phase_accumulator + phase_offset) in [-1, 1].
pub struct SinOsc {
    phase: f32,
    sample_rate: f32,
}

impl Default for SinOsc {
    fn default() -> Self {
        Self::new()
    }
}

impl SinOsc {
    pub fn new() -> Self {
        SinOsc {
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for SinOsc {
    ugen_spec!(
        "SinOsc",
        category = Oscillator,
        inputs = ["freq", "phase"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied();
        let phase_offset_buf = inputs.get(1).copied();
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            // Each channel shares the same phase accumulator state for now
            // (multichannel expansion means separate SinOsc instances per channel)
            let mut phase = self.phase;
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let freq = read_input(freq_buf, ch, i, 440.0);
                let phase_offset = read_input(phase_offset_buf, ch, i, 0.0);

                *out_sample = (phase * TAU + phase_offset).sin();
                phase += freq * inv_sr;
                // Keep phase in [0, 1) to prevent precision loss over time
                phase -= phase.floor();
            }

            // Only update stored phase once (first channel drives it)
            if ch == 0 {
                self.phase = phase;
            }
        }
    }
}

// --- Phasor ---

/// Phasor: ramp from 0 to 1 at the given frequency, then wrap.
///
/// Inputs: freq (Hz).
/// Output: [0, 1) sawtooth ramp.
pub struct Phasor {
    phase: f32,
    sample_rate: f32,
}

impl Default for Phasor {
    fn default() -> Self {
        Self::new()
    }
}

impl Phasor {
    pub fn new() -> Self {
        Phasor {
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for Phasor {
    ugen_spec!(
        "Phasor",
        category = Oscillator,
        inputs = ["freq"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied();
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut phase = self.phase;
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let freq = read_input(freq_buf, ch, i, 440.0);

                *out_sample = phase;
                phase += freq * inv_sr;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
            }
        }
    }
}

// --- Saw ---

/// Naive sawtooth oscillator (non-band-limited).
///
/// Inputs: freq (Hz).
/// Output: [-1, 1] sawtooth wave (ramps up, resets down).
pub struct Saw {
    phase: f32,
    sample_rate: f32,
}

impl Default for Saw {
    fn default() -> Self {
        Self::new()
    }
}

impl Saw {
    pub fn new() -> Self {
        Saw {
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for Saw {
    ugen_spec!(
        "Saw",
        category = Oscillator,
        inputs = ["freq"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied();
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut phase = self.phase;
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let freq = read_input(freq_buf, ch, i, 440.0);

                // Map phase [0,1) to [-1,1)
                *out_sample = 2.0 * phase - 1.0;
                phase += freq * inv_sr;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
            }
        }
    }
}

// --- Pulse ---

/// Naive pulse/square oscillator (non-band-limited).
///
/// Inputs: freq (Hz), width (pulse width in [0, 1], default 0.5 = square).
/// Output: +1 or -1.
pub struct Pulse {
    phase: f32,
    sample_rate: f32,
}

impl Default for Pulse {
    fn default() -> Self {
        Self::new()
    }
}

impl Pulse {
    pub fn new() -> Self {
        Pulse {
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for Pulse {
    ugen_spec!(
        "Pulse",
        category = Oscillator,
        inputs = ["freq", "width"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied();
        let width_buf = inputs.get(1).copied();
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut phase = self.phase;
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let freq = read_input(freq_buf, ch, i, 440.0);
                let width = read_input(width_buf, ch, i, 0.5);

                *out_sample = if phase < width { 1.0 } else { -1.0 };
                phase += freq * inv_sr;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
            }
        }
    }
}

// --- Tri ---

/// Naive triangle oscillator (non-band-limited).
///
/// Inputs: freq (Hz).
/// Output: [-1, 1] triangle wave.
pub struct Tri {
    phase: f32,
    sample_rate: f32,
}

impl Default for Tri {
    fn default() -> Self {
        Self::new()
    }
}

impl Tri {
    pub fn new() -> Self {
        Tri {
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for Tri {
    ugen_spec!(
        "Tri",
        category = Oscillator,
        inputs = ["freq"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied();
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut phase = self.phase;
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let freq = read_input(freq_buf, ch, i, 440.0);

                // Triangle: rise from -1 to +1 in first half, fall +1 to -1 in second
                *out_sample = if phase < 0.5 {
                    4.0 * phase - 1.0
                } else {
                    3.0 - 4.0 * phase
                };
                phase += freq * inv_sr;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
            }
        }
    }
}
