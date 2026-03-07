//! Noise UGens: WhiteNoise, PinkNoise.
//!
//! Uses a simple LCG PRNG (no_std compatible, deterministic, fast).

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{OutputSpec, UGen, UGenSpec};

/// Simple 32-bit LCG for fast, deterministic noise. Not cryptographic.
#[derive(Clone)]
struct Rng {
    state: u32,
}

impl Rng {
    fn new(seed: u32) -> Self {
        Rng { state: seed.wrapping_add(1) }
    }

    /// Returns a uniform f32 in [-1, 1].
    #[inline]
    fn next_bipolar(&mut self) -> f32 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Convert upper bits to float in [-1, 1]
        (self.state as i32 as f32) / (i32::MAX as f32)
    }
}

// --- WhiteNoise ---

/// White noise generator. Uniform random samples in [-1, 1].
///
/// No inputs. One output.
pub struct WhiteNoise {
    rng: Rng,
}

impl WhiteNoise {
    pub fn new() -> Self {
        WhiteNoise { rng: Rng::new(0xDEAD_BEEF) }
    }

    /// Create with a specific seed for deterministic output.
    pub fn with_seed(seed: u32) -> Self {
        WhiteNoise { rng: Rng::new(seed) }
    }
}

static NOISE_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for WhiteNoise {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "WhiteNoise", inputs: &[], outputs: &NOISE_OUTPUTS }
    }

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        self.rng = Rng::new(0xDEAD_BEEF);
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        _inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        for ch in 0..output.num_channels() {
            let out = output.channel_mut(ch).samples_mut();
            for sample in out.iter_mut() {
                *sample = self.rng.next_bipolar();
            }
        }
    }
}

// --- PinkNoise ---

/// Pink noise (1/f) generator using the Voss-McCartney algorithm.
///
/// Uses 16 rows of white noise octaves summed together to approximate
/// a -3 dB/octave rolloff.
///
/// No inputs. One output.
pub struct PinkNoise {
    rng: Rng,
    rows: [f32; 16],
    running_sum: f32,
    counter: u32,
}

impl PinkNoise {
    pub fn new() -> Self {
        PinkNoise {
            rng: Rng::new(0xCAFE_BABE),
            rows: [0.0; 16],
            running_sum: 0.0,
            counter: 0,
        }
    }
}

impl UGen for PinkNoise {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "PinkNoise", inputs: &[], outputs: &NOISE_OUTPUTS }
    }

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        self.rows = [0.0; 16];
        self.running_sum = 0.0;
        self.counter = 0;
        self.rng = Rng::new(0xCAFE_BABE);
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        _inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        // Normalization factor: sum of 16 rows plus one white noise sample.
        let norm = 1.0 / 17.0;

        for ch in 0..output.num_channels() {
            let out = output.channel_mut(ch).samples_mut();
            for sample in out.iter_mut() {
                self.counter = self.counter.wrapping_add(1);

                // Update one row per sample based on trailing zeros of counter.
                // Row k updates every 2^k samples.
                let tz = self.counter.trailing_zeros().min(15) as usize;
                let old = self.rows[tz];
                let new = self.rng.next_bipolar();
                self.rows[tz] = new;
                self.running_sum += new - old;

                // Add one white noise sample for the highest frequency content.
                let white = self.rng.next_bipolar();
                *sample = (self.running_sum + white) * norm;
            }
        }
    }
}
