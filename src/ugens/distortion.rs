//! Distortion UGens: SoftClip, Overdrive, WaveFolder.
//!
//! Soft clipping, overdrive, and wavefolder effects for adding harmonic saturation.
//! These complement the hard `Clip` UGen in `utility`.

use crate::buffer::{AudioBuffer, channel_wrapped, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;

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

impl Default for SoftClip {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftClip {
    pub fn new() -> Self {
        SoftClip
    }
}

impl UGen for SoftClip {
    ugen_spec!(
        "SoftClip",
        category = Effect,
        inputs = ["in", "drive"],
        outputs = ["out"]
    );

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
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let drive = read_input(drive_buf, ch, i, 1.0).max(0.0);
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

impl Default for Overdrive {
    fn default() -> Self {
        Self::new()
    }
}

impl Overdrive {
    pub fn new() -> Self {
        Overdrive { y1: 0.0 }
    }
}

impl UGen for Overdrive {
    ugen_spec!(
        "Overdrive",
        category = Effect,
        inputs = ["in", "drive", "tone", "mix"],
        outputs = ["out"]
    );

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
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let drive = read_input(drive_buf, ch, i, 1.0).max(0.0);
                let tone = read_input(tone_buf, ch, i, 0.5).clamp(0.0, 1.0);
                let mix = read_input(mix_buf, ch, i, 1.0).clamp(0.0, 1.0);

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

// --- WaveFolder ---

/// Wavefolder distortion for aggressive harmonic generation.
///
/// When the driven signal exceeds ±1.0 it "folds" back instead of clipping,
/// creating dense harmonic content characteristic of neuro bass and Buchla-style
/// timbres. The fold formula uses `sin` for smooth, continuous folding:
///
///   `out = sin(π/2 * drive * in)`
///
/// At drive=1.0 the signal passes with mild shaping; at drive=4.0+ the signal
/// folds multiple times, generating rich odd and even harmonics.
///
/// Inputs:
/// - `in`: audio signal to fold
/// - `drive`: fold amount (default 1.0). Higher values = more folds = more harmonics.
/// - `symmetry`: DC offset before folding (default 0.0, range -1 to 1).
///   Non-zero values break odd-harmonic symmetry, introducing even harmonics.
pub struct WaveFolder;

impl Default for WaveFolder {
    fn default() -> Self {
        Self::new()
    }
}

impl WaveFolder {
    pub fn new() -> Self {
        WaveFolder
    }
}

impl UGen for WaveFolder {
    ugen_spec!(
        "WaveFolder",
        category = Effect,
        inputs = ["in", "drive", "symmetry"],
        outputs = ["out"]
    );

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
        let sym_buf = inputs.get(2).copied();

        let half_pi = core::f32::consts::FRAC_PI_2;

        for ch in 0..output.num_channels() {
            let in_ch = channel_wrapped(in_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let x = in_ch[i];
                let drive = read_input(drive_buf, ch, i, 1.0).max(0.0);
                let symmetry = read_input(sym_buf, ch, i, 0.0).clamp(-1.0, 1.0);

                // Apply symmetry offset, then fold via sin
                let driven = (x + symmetry) * drive;
                out[i] = (half_pi * driven).sin();
            }
        }
    }
}
