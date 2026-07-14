//! Band-limited oscillators using polyBLEP anti-aliasing.
//!
//! - [`BlSaw`]: Band-limited sawtooth
//! - [`BlPulse`]: Band-limited pulse/square with variable width
//! - [`BlTri`]: Band-limited triangle (leaky-integrated polyBLEP square)
//!
//! All use a polynomial bandlimited step (polyBLEP) correction to reduce
//! aliasing at discontinuities, providing ~30 dB of alias rejection over
//! naive waveforms with minimal CPU cost.

use crate::buffer::{AudioBuffer, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;

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

impl Default for BlSaw {
    fn default() -> Self {
        Self::new()
    }
}

impl BlSaw {
    pub fn new() -> Self {
        BlSaw {
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for BlSaw {
    ugen_spec!(
        "BlSaw",
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
                let dt = (freq * inv_sr).abs();

                // Naive saw: phase [0,1) → [-1,1)
                let mut sample = 2.0 * phase - 1.0;
                // polyBLEP correction at the wrap discontinuity
                sample -= poly_blep(phase, dt);

                *out_sample = sample;
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

impl Default for BlPulse {
    fn default() -> Self {
        Self::new()
    }
}

impl BlPulse {
    pub fn new() -> Self {
        BlPulse {
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for BlPulse {
    ugen_spec!(
        "BlPulse",
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
                let dt = (freq * inv_sr).abs();
                let width =
                    read_input(width_buf, ch, i, 0.5).clamp(dt.max(0.01), (1.0 - dt).min(0.99));

                // Naive pulse
                let mut sample = if phase < width { 1.0 } else { -1.0 };

                // polyBLEP at rising edge (phase ~ 0)
                sample += poly_blep(phase, dt);
                // polyBLEP at falling edge (phase ~ width)
                let phase_shifted = (phase - width + 1.0) % 1.0;
                sample -= poly_blep(phase_shifted, dt);

                *out_sample = sample;
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

impl Default for BlTri {
    fn default() -> Self {
        Self::new()
    }
}

impl BlTri {
    pub fn new() -> Self {
        BlTri {
            phase: 0.0,
            sample_rate: 44100.0,
            integrator: 0.0,
        }
    }
}

impl UGen for BlTri {
    ugen_spec!(
        "BlTri",
        category = Oscillator,
        inputs = ["freq"],
        outputs = ["out"]
    );

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

            for (i, out_sample) in out.iter_mut().enumerate() {
                let freq = read_input(freq_buf, ch, i, 440.0);
                let dt = (freq * inv_sr).abs();

                // Band-limited square wave (width = 0.5)
                let mut square = if phase < 0.5 { 1.0 } else { -1.0 };
                square += poly_blep(phase, dt);
                let phase_shifted = (phase - 0.5 + 1.0) % 1.0;
                square -= poly_blep(phase_shifted, dt);

                // Leaky integration with frequency-adaptive coefficients
                integrator += square * 4.0 * dt;
                integrator *= 1.0 - 2.0 * dt;

                *out_sample = integrator;
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
