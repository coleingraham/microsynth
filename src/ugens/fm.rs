//! FM synthesis UGen.
//!
//! Frequency modulation oscillator for DX7-style timbres, metallic bells,
//! and electric piano sounds.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use core::f32::consts::TAU;

// --- FmOsc ---

/// Two-operator FM synthesis oscillator.
///
/// A carrier sine oscillator whose instantaneous frequency is modulated by
/// a modulator sine oscillator. The modulator's amplitude (scaled by `index`)
/// determines the amount of frequency deviation, which controls the harmonic
/// richness of the output.
///
/// `output = sin(2π * carrier_phase + index * sin(2π * mod_phase))`
///
/// Inputs:
/// - `freq`: carrier frequency in Hz (default 440)
/// - `ratio`: modulator-to-carrier frequency ratio (default 1.0).
///   Integer ratios (1, 2, 3...) produce harmonic spectra; non-integer
///   ratios produce inharmonic/metallic timbres.
/// - `index`: modulation index (default 1.0). Controls brightness:
///   0 = pure sine, 1–3 = mild FM, 5+ = bright/metallic.
///   The modulator's peak deviation = index * mod_freq.
///
/// Classic patches:
/// - Electric piano (DX7): ratio=1, index=1.5–3
/// - Bell: ratio=1.4, index=3–8
/// - Brass: ratio=1, index=0→5 (envelope on index)
/// - Bass: ratio=0.5, index=2–4
pub struct FmOsc {
    carrier_phase: f32,
    mod_phase: f32,
    sample_rate: f32,
}

impl FmOsc {
    pub fn new() -> Self {
        FmOsc {
            carrier_phase: 0.0,
            mod_phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

static FMOSC_INPUTS: [InputSpec; 3] = [
    InputSpec { name: "freq", rate: Rate::Audio },
    InputSpec { name: "ratio", rate: Rate::Audio },
    InputSpec { name: "index", rate: Rate::Audio },
];
static FMOSC_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for FmOsc {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "FmOsc", inputs: &FMOSC_INPUTS, outputs: &FMOSC_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.carrier_phase = 0.0;
        self.mod_phase = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied();
        let ratio_buf = inputs.get(1).copied();
        let index_buf = inputs.get(2).copied();
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut carrier_phase = self.carrier_phase;
            let mut mod_phase = self.mod_phase;
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(440.0);
                let ratio = ratio_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1.0)
                    .max(0.0);
                let index = index_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1.0)
                    .max(0.0);

                let mod_freq = freq * ratio;

                // Modulator output
                let modulator = (mod_phase * TAU).sin();

                // Carrier with phase modulation
                // FM via phase modulation: carrier_phase + index * modulator
                out[i] = (carrier_phase * TAU + index * modulator).sin();

                // Advance phases
                carrier_phase += freq * inv_sr;
                carrier_phase -= carrier_phase.floor();
                mod_phase += mod_freq * inv_sr;
                mod_phase -= mod_phase.floor();
            }

            if ch == 0 {
                self.carrier_phase = carrier_phase;
                self.mod_phase = mod_phase;
            }
        }
    }
}
