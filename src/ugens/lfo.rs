//! LFO (Low Frequency Oscillator) UGen.
//!
//! A dedicated modulation source with multiple waveform shapes and unipolar
//! [0, 1] output, designed for filter cutoff modulation, tremolo, and other
//! synthesis modulation tasks.
//!
//! Unlike the audio oscillators (`sinOsc`, `saw`, etc.) which output bipolar
//! [-1, 1], the LFO outputs [0, 1] which maps directly to parameter ranges
//! without requiring manual offset math.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};

/// Multi-shape LFO with unipolar [0, 1] output.
///
/// Inputs:
/// - `freq`: LFO rate in Hz (default 1.0)
/// - `shape`: waveform shape (default 0.0)
///   - 0.0 = sine
///   - 1.0 = triangle
///   - 2.0 = sawtooth (ramp up)
///   - 3.0 = square
///   - Non-integer values crossfade between adjacent shapes.
///
/// Output: unipolar signal in [0, 1].
pub struct Lfo {
    phase: f32,
    sample_rate: f32,
}

impl Lfo {
    pub fn new() -> Self {
        Lfo {
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

static LFO_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "freq", rate: Rate::Audio },
    InputSpec { name: "shape", rate: Rate::Audio },
];
static LFO_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

/// Compute each waveform shape from phase [0, 1).
/// All return values in [0, 1].
#[inline]
fn lfo_sine(phase: f32) -> f32 {
    0.5 + 0.5 * (phase * core::f32::consts::TAU).sin()
}

#[inline]
fn lfo_triangle(phase: f32) -> f32 {
    // 0→1 for first half, 1→0 for second half
    let p = phase * 2.0;
    if p < 1.0 { p } else { 2.0 - p }
}

#[inline]
fn lfo_saw(phase: f32) -> f32 {
    phase
}

#[inline]
fn lfo_square(phase: f32) -> f32 {
    if phase < 0.5 { 1.0 } else { 0.0 }
}

/// Evaluate the LFO at a given phase and (possibly fractional) shape.
#[inline]
fn lfo_eval(phase: f32, shape: f32) -> f32 {
    let shape = shape.clamp(0.0, 3.0);
    let idx = shape as u32;
    let frac = shape - idx as f32;

    let a = match idx {
        0 => lfo_sine(phase),
        1 => lfo_triangle(phase),
        2 => lfo_saw(phase),
        _ => lfo_square(phase),
    };

    if frac < 1e-6 || idx >= 3 {
        return a;
    }

    let b = match idx + 1 {
        1 => lfo_triangle(phase),
        2 => lfo_saw(phase),
        _ => lfo_square(phase),
    };

    a + frac * (b - a)
}

impl UGen for Lfo {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Lfo", inputs: &LFO_INPUTS, outputs: &LFO_OUTPUTS }
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
        let shape_buf = inputs.get(1).copied();
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut phase = self.phase;
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1.0)
                    .max(0.0);
                let shape = shape_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.0);

                out[i] = lfo_eval(phase, shape);

                phase += freq * inv_sr;
                if phase >= 1.0 {
                    phase -= 1.0;
                }
            }

            if ch == 0 {
                self.phase = phase;
            }
        }
    }
}
