//! Distortion UGens: SoftClip, Overdrive.
//!
//! Soft clipping and overdrive effects for adding harmonic saturation.
//! These complement the hard `Clip` UGen in `utility`.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};

// --- SoftClip ---

/// Hyperbolic tangent soft clipper.
///
/// Applies `tanh(drive * in)` to produce smooth saturation that asymptotically
/// approaches ±1. Unlike hard clipping, the transition into saturation is
/// gradual, producing a warmer and more musical distortion.
///
/// Inputs:
/// - `in`: audio signal to clip
/// - `drive`: pre-gain before tanh (default 1.0). Higher values push the signal
///   further into saturation. At drive=1.0 small signals pass nearly unchanged;
///   at drive=5.0+ the output is heavily saturated.
///
/// Output is always bounded to (-1, 1).
pub struct SoftClip;

impl SoftClip {
    pub fn new() -> Self {
        SoftClip
    }
}

static SOFTCLIP_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "drive", rate: Rate::Audio },
];
static SOFTCLIP_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for SoftClip {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "SoftClip", inputs: &SOFTCLIP_INPUTS, outputs: &SOFTCLIP_OUTPUTS }
    }

    fn init(&mut self, _context: &ProcessContext) {}
    fn reset(&mut self) {}

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let drive_buf = inputs.get(1).copied();

        for ch in 0..output.num_channels() {
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let drive = drive_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1.0)
                    .max(0.0);
                out[i] = (drive * x).tanh();
            }
        }
    }
}

// --- Overdrive ---

/// Asymmetric tube-style overdrive with tone control and dry/wet mix.
///
/// Emulates the harmonic character of a tube amplifier by applying different
/// saturation curves to the positive and negative halves of the signal:
/// - Positive half: `tanh(drive * x)` (standard soft clip)
/// - Negative half: cubic soft clip `g - g³/3.375` (softer compression)
///
/// This asymmetry introduces even harmonics (2nd, 4th, ...) alongside the odd
/// harmonics from symmetric clipping, producing the "warm" character associated
/// with tube distortion.
///
/// A simple one-pole lowpass tone filter shapes the post-distortion timbre,
/// and a dry/wet mix control allows parallel blending.
///
/// Inputs:
/// - `in`: audio signal
/// - `drive`: pre-gain (default 1.0). 1.0 = clean, 5.0+ = heavy distortion.
/// - `tone`: post-distortion brightness (default 0.5). 0.0 = dark, 1.0 = bright.
/// - `mix`: dry/wet blend (default 1.0). 0.0 = fully dry, 1.0 = fully wet.
pub struct Overdrive {
    y1: f32,
}

impl Overdrive {
    pub fn new() -> Self {
        Overdrive { y1: 0.0 }
    }
}

static OVERDRIVE_INPUTS: [InputSpec; 4] = [
    InputSpec { name: "in", rate: Rate::Audio },
    InputSpec { name: "drive", rate: Rate::Audio },
    InputSpec { name: "tone", rate: Rate::Audio },
    InputSpec { name: "mix", rate: Rate::Audio },
];
static OVERDRIVE_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Overdrive {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Overdrive", inputs: &OVERDRIVE_INPUTS, outputs: &OVERDRIVE_OUTPUTS }
    }

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        self.y1 = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let in_buf = inputs[0];
        let drive_buf = inputs.get(1).copied();
        let tone_buf = inputs.get(2).copied();
        let mix_buf = inputs.get(3).copied();

        for ch in 0..output.num_channels() {
            let mut y1 = self.y1;
            let in_ch = in_buf.channel(ch % in_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let drive = drive_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1.0)
                    .max(0.0);
                let tone = tone_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                let mix = mix_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1.0)
                    .clamp(0.0, 1.0);

                // Pre-gain
                let gained = drive * x;

                // Asymmetric soft clipping
                let clipped = if gained >= 0.0 {
                    // Positive half: tanh saturation
                    gained.tanh()
                } else {
                    // Negative half: cubic soft clip for softer compression
                    // x - x^3/3 scaled to [-1.5, 0] range, normalized by 1/1.5^2
                    let g = gained.clamp(-1.5, 0.0);
                    g - (g * g * g) / 3.375
                };

                // One-pole tone filter
                // tone=0.0 -> coeff=0.95 (dark), tone=1.0 -> coeff=0.1 (bright)
                let coeff = 0.95 - tone * 0.85;
                y1 = (1.0 - coeff) * clipped + coeff * y1;

                // Dry/wet mix
                out[i] = (1.0 - mix) * x + mix * y1;
            }

            if ch == 0 {
                self.y1 = y1;
            }
        }
    }
}
