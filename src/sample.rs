//! Audio sample buffer storage for sample playback.
//!
//! Provides a `SampleBank` for storing named audio buffers and a `PlayBuf`
//! UGen for playing them back with rate and loop control.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A unique identifier for a sample buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SampleId(pub u32);

/// A stored audio sample: one or more channels of f32 data at a given sample rate.
#[derive(Clone)]
pub struct Sample {
    /// Per-channel sample data.
    channels: Vec<Vec<f32>>,
    /// Sample rate this was recorded at.
    sample_rate: f32,
    /// Optional name for lookup.
    name: String,
}

impl Sample {
    /// Create a mono sample from a slice of f32 data.
    pub fn from_mono(data: &[f32], sample_rate: f32) -> Self {
        Sample {
            channels: alloc::vec![data.to_vec()],
            sample_rate,
            name: String::new(),
        }
    }

    /// Create a stereo sample from two slices.
    pub fn from_stereo(left: &[f32], right: &[f32], sample_rate: f32) -> Self {
        Sample {
            channels: alloc::vec![left.to_vec(), right.to_vec()],
            sample_rate,
            name: String::new(),
        }
    }

    /// Create a multi-channel sample.
    pub fn from_channels(channels: Vec<Vec<f32>>, sample_rate: f32) -> Self {
        Sample {
            channels,
            sample_rate,
            name: String::new(),
        }
    }

    /// Set the name of this sample.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of frames (samples per channel).
    pub fn num_frames(&self) -> usize {
        self.channels.first().map_or(0, |c| c.len())
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f32 {
        self.num_frames() as f32 / self.sample_rate
    }

    /// Get a channel's data.
    pub fn channel(&self, ch: usize) -> &[f32] {
        &self.channels[ch]
    }

    /// Sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Read a sample with linear interpolation at a fractional frame index.
    /// Returns 0.0 if out of bounds.
    pub fn read_interpolated(&self, ch: usize, index: f64) -> f32 {
        let ch_data = &self.channels[ch % self.channels.len()];
        let len = ch_data.len();
        if len == 0 {
            return 0.0;
        }

        let i = index.floor() as i64;
        let frac = (index - index.floor()) as f32;

        let s0 = if i >= 0 && (i as usize) < len {
            ch_data[i as usize]
        } else {
            0.0
        };
        let s1 = if (i + 1) >= 0 && ((i + 1) as usize) < len {
            ch_data[(i + 1) as usize]
        } else {
            0.0
        };

        s0 + frac * (s1 - s0)
    }
}

/// A bank of named and numbered audio sample buffers.
pub struct SampleBank {
    samples: BTreeMap<SampleId, Sample>,
    name_index: BTreeMap<String, SampleId>,
    next_id: u32,
}

impl SampleBank {
    /// Create an empty sample bank.
    pub fn new() -> Self {
        SampleBank {
            samples: BTreeMap::new(),
            name_index: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Load a sample into the bank. Returns its SampleId.
    pub fn load(&mut self, sample: Sample) -> SampleId {
        let id = SampleId(self.next_id);
        self.next_id += 1;
        if !sample.name.is_empty() {
            self.name_index.insert(sample.name.clone(), id);
        }
        self.samples.insert(id, sample);
        id
    }

    /// Get a sample by ID.
    pub fn get(&self, id: SampleId) -> Option<&Sample> {
        self.samples.get(&id)
    }

    /// Get a sample by name.
    pub fn get_by_name(&self, name: &str) -> Option<&Sample> {
        let id = self.name_index.get(name)?;
        self.samples.get(id)
    }

    /// Look up a SampleId by name.
    pub fn id_for_name(&self, name: &str) -> Option<SampleId> {
        self.name_index.get(name).copied()
    }

    /// Remove a sample by ID.
    pub fn remove(&mut self, id: SampleId) {
        if let Some(sample) = self.samples.remove(&id) {
            if !sample.name.is_empty() {
                self.name_index.remove(&sample.name);
            }
        }
    }

    /// Number of loaded samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl Default for SampleBank {
    fn default() -> Self {
        Self::new()
    }
}
