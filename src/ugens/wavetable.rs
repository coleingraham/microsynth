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
