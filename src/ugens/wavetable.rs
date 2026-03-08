//! Wavetable oscillator UGen.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use crate::sample::Sample;
use alloc::sync::Arc;

/// Wavetable oscillator: reads through a stored waveform at a given frequency.
///
/// Unlike PlayBuf (which plays linearly at a rate), WaveTable wraps around
/// the waveform at the specified frequency, producing a pitched tone with
/// the waveform's timbre.
///
/// Inputs: freq (Hz).
///
/// The waveform is set at construction time via `WaveTable::with_waveform()`.
/// The waveform is treated as one complete cycle — the oscillator's frequency
/// determines how fast it scans through the table.
pub struct WaveTable {
    waveform: Option<Arc<Sample>>,
    phase: f64,
    sample_rate: f32,
}

impl WaveTable {
    pub fn new() -> Self {
        WaveTable {
            waveform: None,
            phase: 0.0,
            sample_rate: 44100.0,
        }
    }

    /// Set the waveform to use. The entire sample is treated as one cycle.
    pub fn with_waveform(mut self, waveform: Arc<Sample>) -> Self {
        self.waveform = Some(waveform);
        self
    }
}

static WT_INPUTS: [InputSpec; 1] = [InputSpec { name: "freq", rate: Rate::Audio }];
static WT_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for WaveTable {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "WaveTable",
            inputs: &WT_INPUTS,
            outputs: &WT_OUTPUTS,
        }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        self.phase = 0.0;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }

    fn output_channels(&self, _input_channels: &[usize]) -> usize {
        self.waveform.as_ref().map_or(1, |w| w.num_channels())
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let waveform = match &self.waveform {
            Some(w) => w,
            None => {
                output.clear();
                return;
            }
        };

        let freq_buf = inputs.first().copied();
        let table_len = waveform.num_frames() as f64;
        if table_len == 0.0 {
            output.clear();
            return;
        }
        let inv_sr = 1.0 / self.sample_rate as f64;

        for ch in 0..output.num_channels() {
            let mut phase = self.phase;
            let out = output.channel_mut(ch).samples_mut();
            let wt_ch = ch % waveform.num_channels();

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(440.0) as f64;

                // Read with wrapping interpolation
                let read_pos = phase * table_len;
                out[i] = read_interpolated_wrap(waveform.channel(wt_ch), read_pos, table_len);

                // Advance phase
                phase += freq * inv_sr;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
            }
        }
    }
}

/// Linear interpolation with wrapping for wavetable lookup.
fn read_interpolated_wrap(data: &[f32], index: f64, _len: f64) -> f32 {
    let i0 = index.floor() as usize % data.len();
    let i1 = (i0 + 1) % data.len();
    let frac = (index - index.floor()) as f32;
    data[i0] + frac * (data[i1] - data[i0])
}

// ---- Waveform generators ------------------------------------------------

const DEFAULT_TABLE_SIZE: usize = 2048;

/// Generate a single-cycle sine waveform.
pub fn generate_sine(len: usize) -> Arc<Sample> {
    use core::f32::consts::PI;
    let data: alloc::vec::Vec<f32> = (0..len)
        .map(|i| (2.0 * PI * i as f32 / len as f32).sin())
        .collect();
    Arc::new(Sample::from_mono(&data, 44100.0))
}

/// Generate a single-cycle sawtooth waveform (band-limited via additive synthesis).
pub fn generate_saw(len: usize) -> Arc<Sample> {
    let harmonics = 64;
    let data: alloc::vec::Vec<f32> = (0..len)
        .map(|i| {
            use core::f32::consts::PI;
            let phase = 2.0 * PI * i as f32 / len as f32;
            let mut val = 0.0f32;
            for h in 1..=harmonics {
                let sign = if h % 2 == 0 { -1.0 } else { 1.0 };
                val += sign * (h as f32 * phase).sin() / h as f32;
            }
            val * (2.0 / core::f32::consts::PI)
        })
        .collect();
    Arc::new(Sample::from_mono(&data, 44100.0))
}

/// Generate a single-cycle triangle waveform (band-limited via additive synthesis).
pub fn generate_tri(len: usize) -> Arc<Sample> {
    let harmonics = 64;
    let data: alloc::vec::Vec<f32> = (0..len)
        .map(|i| {
            use core::f32::consts::PI;
            let phase = 2.0 * PI * i as f32 / len as f32;
            let mut val = 0.0f32;
            for k in 0..harmonics {
                let h = 2 * k + 1; // odd harmonics only
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                val += sign * (h as f32 * phase).sin() / (h as f32 * h as f32);
            }
            val * (8.0 / (PI * PI))
        })
        .collect();
    Arc::new(Sample::from_mono(&data, 44100.0))
}

/// Generate a single-cycle square/pulse waveform (band-limited via additive synthesis).
pub fn generate_square(len: usize) -> Arc<Sample> {
    let harmonics = 64;
    let data: alloc::vec::Vec<f32> = (0..len)
        .map(|i| {
            use core::f32::consts::PI;
            let phase = 2.0 * PI * i as f32 / len as f32;
            let mut val = 0.0f32;
            for k in 0..harmonics {
                let h = 2 * k + 1; // odd harmonics only
                val += (h as f32 * phase).sin() / h as f32;
            }
            val * (4.0 / PI)
        })
        .collect();
    Arc::new(Sample::from_mono(&data, 44100.0))
}

/// Create a WaveTable pre-loaded with a sine waveform.
pub fn sine_table() -> WaveTable {
    WaveTable::new().with_waveform(generate_sine(DEFAULT_TABLE_SIZE))
}

/// Create a WaveTable pre-loaded with a sawtooth waveform.
pub fn saw_table() -> WaveTable {
    WaveTable::new().with_waveform(generate_saw(DEFAULT_TABLE_SIZE))
}

/// Create a WaveTable pre-loaded with a triangle waveform.
pub fn tri_table() -> WaveTable {
    WaveTable::new().with_waveform(generate_tri(DEFAULT_TABLE_SIZE))
}

/// Create a WaveTable pre-loaded with a square waveform.
pub fn square_table() -> WaveTable {
    WaveTable::new().with_waveform(generate_square(DEFAULT_TABLE_SIZE))
}
