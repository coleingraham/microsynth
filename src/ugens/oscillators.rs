//! Oscillator UGens: SinOsc, Saw, Pulse, Tri, Phasor.
//!
//! All oscillators use a phase accumulator in [0, 1) and produce output
//! in the range [-1, 1] (except Phasor which outputs [0, 1)).
//!
//! Inputs:
//! - `freq`: frequency in Hz (audio rate, per-sample modulation supported)
//! - `phase` (SinOsc only): phase offset in radians
//! - `width` (Pulse only): pulse width in [0, 1]

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
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

impl SinOsc {
    pub fn new() -> Self {
        SinOsc {
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

static SINOSC_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "freq", rate: Rate::Audio },
    InputSpec { name: "phase", rate: Rate::Audio },
];
static SINOSC_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for SinOsc {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "SinOsc", inputs: &SINOSC_INPUTS, outputs: &SINOSC_OUTPUTS }
    }

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

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(440.0);
                let phase_offset = phase_offset_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.0);

                out[i] = (phase * TAU + phase_offset).sin();
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

impl Phasor {
    pub fn new() -> Self {
        Phasor { phase: 0.0, sample_rate: 44100.0 }
    }
}

static PHASOR_INPUTS: [InputSpec; 1] = [InputSpec { name: "freq", rate: Rate::Audio }];
static PHASOR_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Phasor {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Phasor", inputs: &PHASOR_INPUTS, outputs: &PHASOR_OUTPUTS }
    }

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

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(440.0);

                out[i] = phase;
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

impl Saw {
    pub fn new() -> Self {
        Saw { phase: 0.0, sample_rate: 44100.0 }
    }
}

static SAW_INPUTS: [InputSpec; 1] = [InputSpec { name: "freq", rate: Rate::Audio }];
static SAW_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Saw {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Saw", inputs: &SAW_INPUTS, outputs: &SAW_OUTPUTS }
    }

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

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(440.0);

                // Map phase [0,1) to [-1,1)
                out[i] = 2.0 * phase - 1.0;
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

impl Pulse {
    pub fn new() -> Self {
        Pulse { phase: 0.0, sample_rate: 44100.0 }
    }
}

static PULSE_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "freq", rate: Rate::Audio },
    InputSpec { name: "width", rate: Rate::Audio },
];
static PULSE_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Pulse {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Pulse", inputs: &PULSE_INPUTS, outputs: &PULSE_OUTPUTS }
    }

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

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(440.0);
                let width = width_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.5);

                out[i] = if phase < width { 1.0 } else { -1.0 };
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

impl Tri {
    pub fn new() -> Self {
        Tri { phase: 0.0, sample_rate: 44100.0 }
    }
}

static TRI_INPUTS: [InputSpec; 1] = [InputSpec { name: "freq", rate: Rate::Audio }];
static TRI_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Tri {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Tri", inputs: &TRI_INPUTS, outputs: &TRI_OUTPUTS }
    }

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

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(440.0);

                // Triangle: rise from -1 to +1 in first half, fall +1 to -1 in second
                out[i] = if phase < 0.5 {
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
