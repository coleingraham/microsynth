//! Noise UGens: WhiteNoise, PinkNoise, VinylCrackle.
//!
//! Uses a simple LCG PRNG (no_std compatible, deterministic, fast).

use super::rng::Rng;
use crate::buffer::{AudioBuffer, MAX_BLOCK_SIZE, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;

// --- WhiteNoise ---

/// White noise generator. Uniform random samples in [-1, 1].
///
/// No inputs. One output.
pub struct WhiteNoise {
    rng: Rng,
}

impl Default for WhiteNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl WhiteNoise {
    pub fn new() -> Self {
        WhiteNoise {
            rng: Rng::new(0xDEAD_BEEF),
        }
    }

    /// Create with a specific seed for deterministic output.
    pub fn with_seed(seed: u32) -> Self {
        WhiteNoise {
            rng: Rng::new(seed),
        }
    }
}

impl UGen for WhiteNoise {
    ugen_spec!(
        "WhiteNoise",
        category = Noise,
        inputs = [],
        outputs = ["out"]
    );

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        self.rng = Rng::new(0xDEAD_BEEF);
    }

    fn reseed_noise(&mut self, seed: u32) {
        self.rng = Rng::new(seed);
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        _inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        // `self.rng` is a single shared PRNG stream, not per-channel state.
        // The old loop below mutated it directly once per output channel —
        // not the read-back-inside-loop timing bug fixed elsewhere in this
        // crate (see filters::OnePole::process's comment), but the same
        // underlying defect class: a shared mutable resource consumed once
        // per channel instead of once per sample, so channel 1 got a
        // completely different (later) slice of the noise stream than
        // channel 0. Fixed by drawing the stream once per sample into a
        // stack-allocated mono scratch buffer and copying it to every
        // output channel.
        let block_size = output.block_size();
        let mut mono = [0.0f32; MAX_BLOCK_SIZE];
        for sample in mono[..block_size].iter_mut() {
            *sample = self.rng.next_bipolar();
        }
        for ch in 0..output.num_channels() {
            output.channel_mut(ch).samples_mut()[..block_size].copy_from_slice(&mono[..block_size]);
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

impl Default for PinkNoise {
    fn default() -> Self {
        Self::new()
    }
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
    ugen_spec!(
        "PinkNoise",
        category = Noise,
        inputs = [],
        outputs = ["out"]
    );

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        self.rows = [0.0; 16];
        self.running_sum = 0.0;
        self.counter = 0;
        self.rng = Rng::new(0xCAFE_BABE);
    }

    fn reseed_noise(&mut self, seed: u32) {
        self.rows = [0.0; 16];
        self.running_sum = 0.0;
        self.counter = 0;
        self.rng = Rng::new(seed);
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        _inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        // Normalization factor: sum of 16 rows plus one white noise sample.
        let norm = 1.0 / 17.0;

        // `self.rng`/`self.rows`/`self.running_sum`/`self.counter` are a
        // single shared generator, not per-channel state — see
        // `WhiteNoise::process`'s comment for why driving it once per
        // channel (the old loop below) is a defect, and the same
        // compute-once-copy-to-all-channels fix.
        let block_size = output.block_size();
        let mut mono = [0.0f32; MAX_BLOCK_SIZE];
        for sample in mono[..block_size].iter_mut() {
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
        for ch in 0..output.num_channels() {
            output.channel_mut(ch).samples_mut()[..block_size].copy_from_slice(&mono[..block_size]);
        }
    }
}

// --- VinylCrackle ---

/// Fallback RNG seed used only if `seed` is left unconnected. A real chain
/// spec is expected to always wire `seed` from its keyed-seed derivation
/// (see the struct doc); this just keeps an unwired instance from panicking
/// or being silently mute.
const VINYL_CRACKLE_DEFAULT_SEED: u32 = 0x7A1E_C0DE;

/// Sparse impulsive click/pop noise, the vinyl-surface-noise texture.
///
/// Each sample, a coin flip (weighted by `density`) decides whether a new
/// click fires; when one does, it gets a random sign, an amplitude skewed
/// toward quiet (cubed, so most clicks are small and loud pops are rare —
/// matching how real surface noise sounds), and a short random exponential
/// decay. Between clicks the output just decays the last one to zero.
///
/// **Seeding is a cross-repo contract, not a local detail.** `seed` is the
/// literal parameter name a chain-spec author's keyed-seed derivation
/// targets (`root → (busId, stageIndex, "seed")`); this UGen only *consumes*
/// that value — it never derives its own. It is read once, from the first
/// sample of `seed` on this instance's first `process` call (or on the call
/// after a `reset`), and held for the UGen's lifetime: `seed` is meant to be
/// a chain-spec constant, not something modulated at audio rate. Equal seeds
/// produce byte-identical output (see the determinism tests in
/// `tests/ugens.rs`); different seeds must, and are checked to, produce
/// different output first — a test that would also pass with a stuck RNG is
/// not a determinism test.
///
/// Inputs:
/// - `density`: average clicks per second (default 80.0)
/// - `amount`: click loudness scale, 0.0-1.0 (default 0.3)
/// - `seed`: RNG seed (default 0x7A1E_C0DE if left unconnected — see above)
pub struct VinylCrackle {
    rng: Rng,
    seeded: bool,
    click_value: f32,
    click_decay: f32,
    sample_rate: f32,
}

impl Default for VinylCrackle {
    fn default() -> Self {
        Self::new()
    }
}

impl VinylCrackle {
    pub fn new() -> Self {
        VinylCrackle {
            rng: Rng::new(VINYL_CRACKLE_DEFAULT_SEED),
            seeded: false,
            click_value: 0.0,
            click_decay: 0.7,
            sample_rate: 44100.0,
        }
    }
}

impl UGen for VinylCrackle {
    ugen_spec!(
        "VinylCrackle",
        category = Noise,
        inputs = [],
        optional_inputs = ["density", "amount", "seed"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        // Re-latch `seed` on the next `process` call rather than reseeding
        // here directly: `reset` has no access to `inputs`, only `seed`'s
        // eventual buffer does.
        self.seeded = false;
        self.click_value = 0.0;
        self.click_decay = 0.7;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let density_buf = inputs.first().copied().flatten();
        let amount_buf = inputs.get(1).copied().flatten();
        let seed_buf = inputs.get(2).copied().flatten();

        if !self.seeded {
            let seed_val = read_input(seed_buf, 0, 0, VINYL_CRACKLE_DEFAULT_SEED as f32);
            self.rng = Rng::new(seed_val as u32);
            self.seeded = true;
        }

        let sr = self.sample_rate.max(1.0);

        // `self.rng`/`self.click_value`/`self.click_decay` are a single
        // shared generator, not per-channel state — see
        // `WhiteNoise::process`'s comment for why driving it once per
        // channel (the old loop below) is a defect, and the same
        // compute-once-copy-to-all-channels fix.
        let block_size = output.block_size();
        let mut mono = [0.0f32; MAX_BLOCK_SIZE];

        for (i, mono_sample) in mono[..block_size].iter_mut().enumerate() {
            let density = read_input(density_buf, 0, i, 80.0).max(0.0);
            let amount = read_input(amount_buf, 0, i, 0.3).clamp(0.0, 1.0);
            let p_trigger = (density / sr).min(1.0);

            let trigger_roll = self.rng.next_bipolar().abs();
            if trigger_roll < p_trigger {
                let sign = if self.rng.next_bipolar() >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let mag = self.rng.next_bipolar().abs();
                let skewed = mag * mag * mag; // mostly small clicks, rare loud pops
                self.click_value = sign * skewed * amount;
                let decay_roll = self.rng.next_bipolar().abs();
                self.click_decay = 0.5 + 0.35 * decay_roll; // ~2-6 sample tails
            } else {
                self.click_value *= self.click_decay;
            }

            *mono_sample = self.click_value;
        }

        for ch in 0..output.num_channels() {
            output.channel_mut(ch).samples_mut()[..block_size].copy_from_slice(&mono[..block_size]);
        }
    }
}
