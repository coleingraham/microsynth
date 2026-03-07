//! Sample playback UGen.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use crate::sample::Sample;
use alloc::sync::Arc;

/// Plays back a sample buffer with rate and loop control.
///
/// Inputs: rate (playback rate: 1.0 = normal, 2.0 = double speed, etc.),
///         trigger (positive crossing restarts playback from the beginning).
///
/// The sample is set at construction time via `PlayBuf::with_sample()`.
/// Reports `is_done()` = true when playback reaches the end (non-looping mode).
pub struct PlayBuf {
    sample: Option<Arc<Sample>>,
    position: f64,
    playing: bool,
    looping: bool,
    done: bool,
    prev_trig: f32,
    sample_rate: f32,
}

impl PlayBuf {
    pub fn new() -> Self {
        PlayBuf {
            sample: None,
            position: 0.0,
            playing: true,
            looping: false,
            done: false,
            prev_trig: 0.0,
            sample_rate: 44100.0,
        }
    }

    /// Set the sample to play.
    pub fn with_sample(mut self, sample: Arc<Sample>) -> Self {
        self.sample = Some(sample);
        self
    }

    /// Set whether playback should loop.
    pub fn with_loop(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }
}

static PLAYBUF_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "rate", rate: Rate::Audio },
    InputSpec { name: "trig", rate: Rate::Audio },
];
static PLAYBUF_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for PlayBuf {
    fn spec(&self) -> UGenSpec {
        UGenSpec {
            name: "PlayBuf",
            inputs: &PLAYBUF_INPUTS,
            outputs: &PLAYBUF_OUTPUTS,
        }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        self.position = 0.0;
        self.playing = true;
        self.done = false;
        self.prev_trig = 0.0;
    }

    fn reset(&mut self) {
        self.position = 0.0;
        self.playing = true;
        self.done = false;
        self.prev_trig = 0.0;
    }

    fn output_channels(&self, _input_channels: &[usize]) -> usize {
        // Output channel count matches the sample's channel count
        self.sample.as_ref().map_or(1, |s| s.num_channels())
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let sample = match &self.sample {
            Some(s) => s,
            None => {
                output.clear();
                return;
            }
        };

        let rate_buf = inputs.first().copied();
        let trig_buf = inputs.get(1).copied();
        let num_frames = sample.num_frames() as f64;
        let rate_ratio = sample.sample_rate() / self.sample_rate;

        for ch in 0..output.num_channels() {
            let mut position = self.position;
            let mut playing = self.playing;
            let mut done = self.done;
            let mut prev_trig = self.prev_trig;
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                // Check trigger (restart playback)
                let trig = trig_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.0);
                if trig > 0.0 && prev_trig <= 0.0 {
                    position = 0.0;
                    playing = true;
                    done = false;
                }
                prev_trig = trig;

                let rate = rate_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(1.0);

                if playing && !done {
                    out[i] = sample.read_interpolated(ch, position);
                    position += rate as f64 * rate_ratio as f64;

                    if position >= num_frames {
                        if self.looping {
                            position %= num_frames;
                        } else {
                            position = num_frames;
                            playing = false;
                            done = true;
                        }
                    } else if position < 0.0 {
                        if self.looping {
                            position = position.rem_euclid(num_frames);
                        } else {
                            position = 0.0;
                            playing = false;
                            done = true;
                        }
                    }
                } else {
                    out[i] = 0.0;
                }
            }

            if ch == 0 {
                self.position = position;
                self.playing = playing;
                self.done = done;
                self.prev_trig = prev_trig;
            }
        }
    }

    fn is_done(&self) -> bool {
        self.done
    }
}
