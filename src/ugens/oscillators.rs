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

// --- Phase-accumulator oscillators ---
//
// Every naive oscillator here is the same machine: hold a phase in [0, 1),
// emit a sample derived from it, advance by freq/sample_rate, wrap. They
// differ only in the waveform function — and, for two of them, in taking one
// extra shaping input. `phase_osc!` stamps each as a concrete named type so
// the DSL registry and `pub use oscillators::*` re-exports keep referencing
// them by name.

/// Generate a naive (non-band-limited) phase-accumulator oscillator UGen.
///
/// Variants supply `sample`, the waveform function mapping the current phase to
/// an output sample. An oscillator that takes a second shaping input declares
/// it as `extra = ("<port>", <default>)`; that port's value is then passed to
/// `sample` as a second argument, read per sample so it can be modulated at
/// audio rate.
macro_rules! phase_osc {
    (
        $(#[$meta:meta])*
        $ty:ident, $name:literal,
        $(extra = ($port:literal, $default:expr),)?
        sample = $sample:expr $(,)?
    ) => {
        $(#[$meta])*
        pub struct $ty {
            phase: f32,
            sample_rate: f32,
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
                    // Each channel shares the same phase accumulator state for now
                    // (multichannel expansion means separate instances per channel).
                    let mut phase = self.phase;
                    let out = output.channel_mut(ch).samples_mut();

                    for (i, out_sample) in out.iter_mut().enumerate() {
                        let freq = read_input(freq_buf, ch, i, 440.0);

                        *out_sample = ($sample)(
                            phase
                            $(, read_input(inputs.get(1).copied(), ch, i, $default))?
                        );

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
    };
}

phase_osc! {
    /// Sine oscillator.
    ///
    /// Inputs: freq (Hz), phase (radians offset).
    /// Output: sin(2*pi*phase_accumulator + phase_offset) in [-1, 1].
    SinOsc, "SinOsc",
    extra = ("phase", 0.0),
    sample = |phase: f32, phase_offset: f32| (phase * TAU + phase_offset).sin(),
}

phase_osc! {
    /// Phasor: ramp from 0 to 1 at the given frequency, then wrap.
    ///
    /// Inputs: freq (Hz).
    /// Output: [0, 1) sawtooth ramp.
    Phasor, "Phasor",
    sample = |phase: f32| phase,
}

phase_osc! {
    /// Naive sawtooth oscillator (non-band-limited).
    ///
    /// Inputs: freq (Hz).
    /// Output: [-1, 1] sawtooth wave (ramps up, resets down).
    Saw, "Saw",
    // Map phase [0,1) to [-1,1)
    sample = |phase: f32| 2.0 * phase - 1.0,
}

phase_osc! {
    /// Naive pulse/square oscillator (non-band-limited).
    ///
    /// Inputs: freq (Hz), width (pulse width in [0, 1], default 0.5 = square).
    /// Output: +1 or -1.
    Pulse, "Pulse",
    extra = ("width", 0.5),
    sample = |phase: f32, width: f32| if phase < width { 1.0 } else { -1.0 },
}

phase_osc! {
    /// Naive triangle oscillator (non-band-limited).
    ///
    /// Inputs: freq (Hz).
    /// Output: [-1, 1] triangle wave.
    Tri, "Tri",
    // Triangle: rise from -1 to +1 in first half, fall +1 to -1 in second
    sample = |phase: f32| if phase < 0.5 { 4.0 * phase - 1.0 } else { 3.0 - 4.0 * phase },
}
