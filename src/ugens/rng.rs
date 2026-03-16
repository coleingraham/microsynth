//! Shared PRNG for noise and physical model UGens.

/// Simple 32-bit LCG for fast, deterministic noise. Not cryptographic.
#[derive(Clone)]
pub(crate) struct Rng {
    state: u32,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng { state: seed.wrapping_add(1) }
    }

    /// Returns a uniform f32 in [-1, 1].
    #[inline]
    pub fn next_bipolar(&mut self) -> f32 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Convert upper bits to float in [-1, 1]
        (self.state as i32 as f32) / (i32::MAX as f32)
    }
}
