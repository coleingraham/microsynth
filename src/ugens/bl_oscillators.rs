//! Band-limited oscillators using polyBLEP anti-aliasing.
//!
//! - [`BlSaw`]: Band-limited sawtooth
//! - [`BlPulse`]: Band-limited pulse/square with variable width
//! - [`BlTri`]: Band-limited triangle (leaky-integrated polyBLEP square)
//!
//! All use a polynomial bandlimited step (polyBLEP) correction to reduce
//! aliasing at discontinuities, providing ~30 dB of alias rejection over
//! naive waveforms with minimal CPU cost.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};

/// 2-point polynomial bandlimited step correction.
///
/// `t`: current phase in [0, 1)
/// `dt`: phase increment per sample (freq / sample_rate), must be positive
#[inline]
fn poly_blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        let t = t / dt;
        2.0 * t - t * t - 1.0
    } else if t > 1.0 - dt {
        let t = (t - 1.0) / dt;
        t * t + 2.0 * t + 1.0
    } else {
        0.0
    }
}

// --- BlSaw ---

/// Band-limited sawtooth oscillator using polyBLEP.
///
/// Inputs: freq (Hz).
/// Output: [-1, 1] sawtooth wave.
pub struct BlSaw {
    phase: f32,
    sample_rate: f32,
}

impl BlSaw {
    pub fn new() -> Self {
        BlSaw { phase: 0.0, sample_rate: 44100.0 }
    }
}

static BLSAW_INPUTS: [InputSpec; 1] = [InputSpec { name: "freq", rate: Rate::Audio }];
static BLSAW_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for BlSaw {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "BlSaw", inputs: &BLSAW_INPUTS, outputs: &BLSAW_OUTPUTS }
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
                let dt = (freq * inv_sr).abs();

                // Naive saw: phase [0,1) → [-1,1)
                let mut sample = 2.0 * phase - 1.0;
                // polyBLEP correction at the wrap discontinuity
                sample -= poly_blep(phase, dt);

                out[i] = sample;
                phase += dt;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
            }
        }
    }
}

// --- BlPulse ---

/// Band-limited pulse/square oscillator using polyBLEP.
///
/// Inputs: freq (Hz), width (pulse width [0, 1], default 0.5 = square).
/// Output: [-1, 1] pulse wave.
pub struct BlPulse {
    phase: f32,
    sample_rate: f32,
}

impl BlPulse {
    pub fn new() -> Self {
        BlPulse { phase: 0.0, sample_rate: 44100.0 }
    }
}

static BLPULSE_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "freq", rate: Rate::Audio },
    InputSpec { name: "width", rate: Rate::Audio },
];
static BLPULSE_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for BlPulse {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "BlPulse", inputs: &BLPULSE_INPUTS, outputs: &BLPULSE_OUTPUTS }
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
                let dt = (freq * inv_sr).abs();
                let width = width_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.5)
                    .clamp(dt.max(0.01), (1.0 - dt).min(0.99));

                // Naive pulse
                let mut sample = if phase < width { 1.0 } else { -1.0 };

                // polyBLEP at rising edge (phase ~ 0)
                sample += poly_blep(phase, dt);
                // polyBLEP at falling edge (phase ~ width)
                let phase_shifted = (phase - width + 1.0) % 1.0;
                sample -= poly_blep(phase_shifted, dt);

                out[i] = sample;
                phase += dt;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
            }
        }
    }
}

// --- BlTri ---

/// Band-limited triangle oscillator.
///
/// Derived by leaky-integrating a band-limited square wave (polyBLEP).
/// The leak coefficient is frequency-adaptive to maintain consistent amplitude.
///
/// Inputs: freq (Hz).
/// Output: approximately [-1, 1] triangle wave.
pub struct BlTri {
    phase: f32,
    sample_rate: f32,
    integrator: f32,
}

impl BlTri {
    pub fn new() -> Self {
        BlTri { phase: 0.0, sample_rate: 44100.0, integrator: 0.0 }
    }
}

static BLTRI_INPUTS: [InputSpec; 1] = [InputSpec { name: "freq", rate: Rate::Audio }];
static BLTRI_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for BlTri {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "BlTri", inputs: &BLTRI_INPUTS, outputs: &BLTRI_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.integrator = 0.0;
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
            let mut integrator = self.integrator;
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(440.0);
                let dt = (freq * inv_sr).abs();

                // Band-limited square wave (width = 0.5)
                let mut square = if phase < 0.5 { 1.0 } else { -1.0 };
                square += poly_blep(phase, dt);
                let phase_shifted = (phase - 0.5 + 1.0) % 1.0;
                square -= poly_blep(phase_shifted, dt);

                // Leaky integration with frequency-adaptive coefficients
                integrator += square * 4.0 * dt;
                integrator *= 1.0 - 2.0 * dt;

                out[i] = integrator;
                phase += dt;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
                self.integrator = integrator;
            }
        }
    }
}
