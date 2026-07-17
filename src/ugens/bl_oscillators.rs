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

// --- Band-limited phase-accumulator oscillators ---
//
// Like the naive oscillators in `oscillators`, these share one machine: hold a
// phase in [0, 1), emit a sample, advance by dt = |freq| / sample_rate, wrap.
// They differ in the waveform function and, independently, in whether they take
// an extra shaping input (BlPulse) or carry extra per-sample state (BlTri).

/// Generate a band-limited (polyBLEP) phase-accumulator oscillator UGen.
///
/// Variants supply `sample`, which receives the current `phase` and the phase
/// increment `dt` (polyBLEP needs `dt` to size its correction window), and
/// returns the output sample. Two optional axes extend that signature:
///
/// - `extra = ("<port>", <default>)` declares a second input port; its raw
///   per-sample value is passed to `sample` as a third argument. Any clamping
///   belongs in `sample`, which is where `dt` is in scope.
/// - `state = (<field>, <init>)` adds an `f32` field carried across samples and
///   channels; `sample` receives it as a final `&mut f32` argument.
macro_rules! bl_osc {
    (
        $(#[$meta:meta])*
        $ty:ident, $name:literal,
        $(extra = ($port:literal, $default:expr),)?
        $(state = ($state:ident, $state_init:expr),)?
        sample = $sample:expr $(,)?
    ) => {
        $(#[$meta])*
        pub struct $ty {
            phase: f32,
            sample_rate: f32,
            $($state: f32,)?
        }

        impl Default for $ty {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $ty {
            pub fn new() -> Self {
                $ty {
                    phase: 0.0,
                    sample_rate: 44100.0,
                    $($state: $state_init,)?
                }
            }
        }

        impl UGen for $ty {
            ugen_spec!(
                $name,
                category = Oscillator,
                inputs = ["freq" $(, $port)?],
                outputs = ["out"]
            );

            fn init(&mut self, context: &ProcessContext) {
                self.sample_rate = context.sample_rate;
            }

            fn reset(&mut self) {
                self.phase = 0.0;
                $(self.$state = $state_init;)?
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
                    $(let mut $state = self.$state;)?
                    let out = output.channel_mut(ch).samples_mut();

                    for (i, out_sample) in out.iter_mut().enumerate() {
                        let freq = read_input(freq_buf, ch, i, 440.0);
                        let dt = (freq * inv_sr).abs();

                        *out_sample = ($sample)(
                            phase,
                            dt
                            $(, read_input(inputs.get(1).copied(), ch, i, $default))?
                            $(, &mut $state)?
                        );

                        phase += dt;
                        phase -= phase.floor();
                    }

                    if ch == 0 {
                        self.phase = phase;
                        $(self.$state = $state;)?
                    }
                }
            }
        }
    };
}

bl_osc! {
    /// Band-limited sawtooth oscillator using polyBLEP.
    ///
    /// Inputs: freq (Hz).
    /// Output: [-1, 1] sawtooth wave.
    BlSaw, "BlSaw",
    sample = |phase: f32, dt: f32| {
        // Naive saw: phase [0,1) → [-1,1)
        let sample = 2.0 * phase - 1.0;
        // polyBLEP correction at the wrap discontinuity
        sample - poly_blep(phase, dt)
    },
}

bl_osc! {
    /// Band-limited pulse/square oscillator using polyBLEP.
    ///
    /// Inputs: freq (Hz), width (pulse width [0, 1], default 0.5 = square).
    /// Output: [-1, 1] pulse wave.
    BlPulse, "BlPulse",
    extra = ("width", 0.5),
    sample = |phase: f32, dt: f32, raw_width: f32| {
        // Keep the pulse width off both edges so the polyBLEPs don't
        // overlap. When dt >= ~0.5 (freq at/above Nyquist) the band is
        // degenerate and lo would exceed hi — guard with lo.max(hi) so a
        // near-Nyquist freq pins the width instead of panicking clamp().
        let lo = dt.max(0.01);
        let hi = (1.0 - dt).min(0.99);
        let width = raw_width.clamp(lo, lo.max(hi));

        // Naive pulse
        let mut sample = if phase < width { 1.0 } else { -1.0 };

        // polyBLEP at rising edge (phase ~ 0)
        sample += poly_blep(phase, dt);
        // polyBLEP at falling edge (phase ~ width)
        let phase_shifted = (phase - width + 1.0) % 1.0;
        sample -= poly_blep(phase_shifted, dt);

        sample
    },
}

bl_osc! {
    /// Band-limited triangle oscillator.
    ///
    /// Derived by leaky-integrating a band-limited square wave (polyBLEP).
    /// The leak coefficient is frequency-adaptive to maintain consistent amplitude.
    ///
    /// Inputs: freq (Hz).
    /// Output: approximately [-1, 1] triangle wave.
    BlTri, "BlTri",
    state = (integrator, 0.0),
    sample = |phase: f32, dt: f32, integrator: &mut f32| {
        // Band-limited square wave (width = 0.5)
        let mut square = if phase < 0.5 { 1.0 } else { -1.0 };
        square += poly_blep(phase, dt);
        let phase_shifted = (phase - 0.5 + 1.0) % 1.0;
        square -= poly_blep(phase_shifted, dt);

        // Leaky integration with frequency-adaptive coefficients
        *integrator += square * 4.0 * dt;
        *integrator *= 1.0 - 2.0 * dt;
        *integrator
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A freq input at/above Nyquist drives dt >= 0.5, which inverts the pulse-
    /// width clamp bounds (lo > hi). BlPulse must pin the width, not panic.
    #[test]
    fn bl_pulse_survives_above_nyquist_freq() {
        let ctx = ProcessContext::new(44100.0, 32);
        let mut osc = BlPulse::new();
        osc.init(&ctx);

        // 40 kHz > Nyquist (22.05 kHz): dt = 40000/44100 ≈ 0.907 -> lo > hi.
        let mut freq = AudioBuffer::mono(ctx.block_size);
        freq.channel_mut(0).samples_mut().fill(40_000.0);
        let mut out = AudioBuffer::mono(ctx.block_size);

        osc.process(&ctx, &[&freq], &mut out); // must not panic
        assert!(out.channel(0).samples().iter().all(|s| s.is_finite()));
    }
}
