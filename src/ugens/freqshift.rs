//! Frequency shifter UGen.
//!
//! Single-sideband frequency shifting via Hilbert transform for Leslie speaker
//! simulation, creative detuning, and sub bass thickening.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use core::f32::consts::TAU;

// --- Hilbert Transform ---

/// Hilbert transform approximation using two parallel allpass chains.
///
/// Each chain is a cascade of 4 first-order allpass filters with coefficients
/// chosen to maintain approximately 90° phase difference between the two
/// outputs across the audio band (~20 Hz to ~20 kHz).
///
/// Coefficients from Hilbert transformer design (Anssi Klapuri / Olli Niemitalo).
const HILBERT_COEFFS_I: [f32; 4] = [
    0.6923878,
    0.9360654322959,
    0.9882295226860,
    0.9987488452737,
];
const HILBERT_COEFFS_Q: [f32; 4] = [
    0.4021921162426,
    0.8561710882420,
    0.9722909545651,
    0.9952884791278,
];

/// State for one allpass chain (4 first-order stages).
#[derive(Clone, Copy)]
struct HilbertChain {
    state: [f32; 4],
}

impl HilbertChain {
    fn new() -> Self {
        HilbertChain { state: [0.0; 4] }
    }

    fn reset(&mut self) {
        self.state = [0.0; 4];
    }

    /// Process one sample through 4 cascaded first-order allpass filters.
    #[inline]
    fn tick(&mut self, input: f32, coeffs: &[f32; 4]) -> f32 {
        let mut x = input;
        for i in 0..4 {
            let c = coeffs[i];
            // First-order allpass: y = c * x + state, new_state = x - c * y
            let y = c * x + self.state[i];
            self.state[i] = x - c * y;
            x = y;
        }
        x
    }
}

// --- FreqShift ---

/// Frequency shifter via Hilbert transform and single-sideband modulation.
///
/// Shifts all frequencies in the input signal by a fixed Hz offset.
/// Unlike pitch shifting, this does not preserve harmonic relationships —
/// a 100 Hz shift on a 440 Hz tone produces 540 Hz (not 440 * ratio).
///
/// Inputs:
/// - `in`: audio signal
/// - `shift`: frequency shift in Hz (default 0.0). Positive shifts up,
///   negative shifts down. Small values (0.5–6 Hz) create Leslie/rotary
///   speaker effects. Larger values create metallic, ring-mod-like timbres.
///
/// Uses a Hilbert transform (two parallel allpass chains producing I/Q
/// quadrature signals) followed by single-sideband modulation:
/// `output = I * cos(2π * shift * t) - Q * sin(2π * shift * t)`
pub struct FreqShift {
    chain_i: HilbertChain,
    chain_q: HilbertChain,
    osc_phase: f32,
    sample_rate: f32,
}

impl FreqShift {
    pub fn new() -> Self {
        FreqShift {
            chain_i: HilbertChain::new(),
            chain_q: HilbertChain::new(),
            osc_phase: 0.0,
            sample_rate: 44100.0,
        }
    }
}

static FREQSHIFT_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "shift", rate: Rate::Audio },
];
static FREQSHIFT_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for FreqShift {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "FreqShift", inputs: &FREQSHIFT_INPUTS, outputs: &FREQSHIFT_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.chain_i.reset();
        self.chain_q.reset();
        self.osc_phase = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let shift_buf = inputs.get(1).copied();
        let inv_sr = 1.0 / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut chain_i = self.chain_i;
            let mut chain_q = self.chain_q;
            let mut osc_phase = self.osc_phase;
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let shift = shift_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.0);

                // Hilbert transform: produce I (in-phase) and Q (quadrature) signals
                let sig_i = chain_i.tick(x, &HILBERT_COEFFS_I);
                let sig_q = chain_q.tick(x, &HILBERT_COEFFS_Q);

                // Single-sideband modulation (upper sideband)
                let angle = osc_phase * TAU;
                let cos_val = angle.cos();
                let sin_val = angle.sin();
                out[i] = sig_i * cos_val - sig_q * sin_val;

                // Advance oscillator phase
                osc_phase += shift * inv_sr;
                osc_phase -= osc_phase.floor();
            }

            if ch == 0 {
                self.chain_i = chain_i;
                self.chain_q = chain_q;
                self.osc_phase = osc_phase;
            }
        }
    }
}
