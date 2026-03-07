/// Maximum block size for audio processing.
/// 64 samples is standard (matches Web Audio render quantum).
pub const MAX_BLOCK_SIZE: usize = 64;

/// A single-channel buffer of samples for one block.
/// Fixed-size array on the stack — no heap allocation.
#[derive(Clone)]
pub struct Block {
    data: [f32; MAX_BLOCK_SIZE],
    len: usize,
}

impl Block {
    /// Create a zeroed block with the given length.
    pub fn new(len: usize) -> Self {
        debug_assert!(len <= MAX_BLOCK_SIZE);
        Block {
            data: [0.0; MAX_BLOCK_SIZE],
            len,
        }
    }

    /// Number of valid samples in this block.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this block has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read-only access to the sample data.
    #[inline]
    pub fn samples(&self) -> &[f32] {
        &self.data[..self.len]
    }

    /// Mutable access to the sample data.
    #[inline]
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.data[..self.len]
    }

    /// Set all samples to zero.
    #[inline]
    pub fn clear(&mut self) {
        self.data[..self.len].fill(0.0);
    }

    /// Fill all samples with a constant value.
    #[inline]
    pub fn fill(&mut self, value: f32) {
        self.data[..self.len].fill(value);
    }

    /// Set the active length of this block.
    #[inline]
    pub fn set_len(&mut self, len: usize) {
        debug_assert!(len <= MAX_BLOCK_SIZE);
        self.len = len;
    }
}

impl core::ops::Index<usize> for Block {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &f32 {
        debug_assert!(index < self.len);
        &self.data[index]
    }
}

impl core::ops::IndexMut<usize> for Block {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        debug_assert!(index < self.len);
        &mut self.data[index]
    }
}

/// A multi-channel audio buffer. Each channel is a `Block`.
///
/// The inner `Vec` is allocated once during graph preparation and never
/// resized on the render path.
pub struct AudioBuffer {
    channels: alloc::vec::Vec<Block>,
}

impl AudioBuffer {
    /// Create a new buffer with the given number of channels and block size.
    pub fn new(num_channels: usize, block_size: usize) -> Self {
        let mut channels = alloc::vec::Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            channels.push(Block::new(block_size));
        }
        AudioBuffer { channels }
    }

    /// Create a mono (single-channel) buffer.
    pub fn mono(block_size: usize) -> Self {
        Self::new(1, block_size)
    }

    /// Number of channels.
    #[inline]
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Get a reference to a single channel.
    #[inline]
    pub fn channel(&self, index: usize) -> &Block {
        &self.channels[index]
    }

    /// Get a mutable reference to a single channel.
    #[inline]
    pub fn channel_mut(&mut self, index: usize) -> &mut Block {
        &mut self.channels[index]
    }

    /// Set the number of channels, reusing existing allocations.
    /// New channels are zeroed.
    pub fn set_num_channels(&mut self, num_channels: usize, block_size: usize) {
        while self.channels.len() < num_channels {
            self.channels.push(Block::new(block_size));
        }
        self.channels.truncate(num_channels);
    }

    /// Set the block size for all channels.
    pub fn set_block_size(&mut self, block_size: usize) {
        for ch in &mut self.channels {
            ch.set_len(block_size);
        }
    }

    /// Zero out all channels.
    pub fn clear(&mut self) {
        for ch in &mut self.channels {
            ch.clear();
        }
    }

    /// Get the block size (from the first channel, or 0 if empty).
    #[inline]
    pub fn block_size(&self) -> usize {
        self.channels.first().map_or(0, |b| b.len())
    }
}
