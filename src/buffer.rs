/// Maximum block size for audio processing.
/// 128 samples matches the Web Audio render quantum.
pub const MAX_BLOCK_SIZE: usize = 128;

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

/// Read one input sample with SuperCollider-style channel wrapping.
///
/// Returns `default` when `buf` is `None` (an unconnected input port).
/// Otherwise reads sample `i` of channel `ch`, wrapping the channel index
/// modulo the buffer's channel count so that, e.g., a mono input feeds every
/// output channel. This is the canonical per-sample input read used throughout
/// the UGens; prefer it over hand-writing
/// `buf.map(|b| b.channel(ch % b.num_channels()).samples()[i]).unwrap_or(d)`.
#[inline]
pub fn read_input(buf: Option<&AudioBuffer>, ch: usize, i: usize, default: f32) -> f32 {
    match buf {
        Some(b) => b.channel(ch % b.num_channels()).samples()[i],
        None => default,
    }
}

/// Borrow a whole input channel with SuperCollider-style channel wrapping.
///
/// The block-level counterpart to [`read_input`]: where `read_input` reads one
/// sample from an optional (possibly unconnected) input port, this hoists a
/// whole channel's slice out of a *connected* input buffer once, before the
/// per-sample loop. The channel index wraps modulo the buffer's channel count,
/// so a mono input feeds every output channel.
///
/// Prefer it over hand-writing
/// `buf.channel(ch % buf.num_channels()).samples()`.
#[inline]
pub fn channel_wrapped(buf: &AudioBuffer, ch: usize) -> &[f32] {
    buf.channel(ch % buf.num_channels()).samples()
}
