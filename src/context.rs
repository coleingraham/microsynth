/// Whether a signal is computed at audio rate or control rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rate {
    /// Audio rate: one sample per sample-tick (block_size samples per block).
    Audio,
    /// Control rate: one sample per block (1 sample per block).
    Control,
}

/// Global processing context passed to every node during rendering.
///
/// Contains engine-wide state that is immutable for the duration of a block.
#[derive(Debug, Clone)]
pub struct ProcessContext {
    /// Sample rate in Hz (e.g. 44100.0).
    pub sample_rate: f32,
    /// Number of samples per audio-rate block.
    pub block_size: usize,
    /// Monotonically increasing sample offset (start of current block).
    pub sample_offset: u64,
}

impl ProcessContext {
    /// Create a new context.
    pub fn new(sample_rate: f32, block_size: usize) -> Self {
        ProcessContext {
            sample_rate,
            block_size,
            sample_offset: 0,
        }
    }

    /// Current time in seconds at the start of this block.
    #[inline]
    pub fn time_secs(&self) -> f64 {
        self.sample_offset as f64 / self.sample_rate as f64
    }

    /// The number of samples in a control-rate "block" (always 1).
    #[inline]
    pub fn control_block_size(&self) -> usize {
        1
    }

    /// Get the effective block size for a given rate.
    #[inline]
    pub fn block_size_for_rate(&self, rate: Rate) -> usize {
        match rate {
            Rate::Audio => self.block_size,
            Rate::Control => 1,
        }
    }

    /// Advance the context by one block.
    pub fn advance(&mut self) {
        self.sample_offset += self.block_size as u64;
    }
}
