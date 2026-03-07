//! Envelope UGens: Line, ASR.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};

// --- Line ---

/// Linear ramp from `start` to `end` over `dur` seconds, then holds at `end`.
///
/// Inputs: start (initial value), end (target value), dur (duration in seconds).
/// Reads input values at init time (first sample of first block).
pub struct Line {
    value: f32,
    increment: f32,
    target: f32,
    samples_remaining: u64,
    initialized: bool,
    sample_rate: f32,
}

impl Line {
    pub fn new() -> Self {
        Line {
            value: 0.0,
            increment: 0.0,
            target: 1.0,
            samples_remaining: 0,
            initialized: false,
            sample_rate: 44100.0,
        }
    }
}

static LINE_INPUTS: [InputSpec; 3] = [
    InputSpec { name: "start", rate: Rate::Audio },
    InputSpec { name: "end", rate: Rate::Audio },
    InputSpec { name: "dur", rate: Rate::Audio },
];
static LINE_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Line {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Line", inputs: &LINE_INPUTS, outputs: &LINE_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        self.initialized = false;
    }

    fn reset(&mut self) {
        self.value = 0.0;
        self.increment = 0.0;
        self.target = 1.0;
        self.samples_remaining = 0;
        self.initialized = false;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        // On first block, read start/end/dur from first sample of inputs
        if !self.initialized {
            let start = inputs.first()
                .map(|b| b.channel(0).samples()[0])
                .unwrap_or(0.0);
            let end = inputs.get(1)
                .map(|b| b.channel(0).samples()[0])
                .unwrap_or(1.0);
            let dur = inputs.get(2)
                .map(|b| b.channel(0).samples()[0])
                .unwrap_or(1.0)
                .max(0.0);

            self.value = start;
            self.target = end;
            let total_samples = (dur * self.sample_rate) as u64;
            if total_samples > 0 {
                self.increment = (end - start) / total_samples as f32;
                self.samples_remaining = total_samples;
            } else {
                self.increment = 0.0;
                self.value = end;
                self.samples_remaining = 0;
            }
            self.initialized = true;
        }

        for ch in 0..output.num_channels() {
            let mut value = self.value;
            let mut remaining = self.samples_remaining;
            let out = output.channel_mut(ch).samples_mut();

            for sample in out.iter_mut() {
                *sample = value;
                if remaining > 0 {
                    value += self.increment;
                    remaining -= 1;
                    if remaining == 0 {
                        value = self.target;
                    }
                }
            }

            if ch == 0 {
                self.value = value;
                self.samples_remaining = remaining;
            }
        }
    }
}

// --- ASR ---

/// Attack-Sustain-Release envelope triggered by a gate signal.
///
/// Inputs: gate (>0 = on, 0 = off), attack (seconds), release (seconds).
///
/// - When gate goes high: ramp from current level to 1.0 over `attack` seconds.
/// - While gate is high: hold at 1.0.
/// - When gate goes low: ramp from current level to 0.0 over `release` seconds.
pub struct ASR {
    level: f32,
    stage: AsrStage,
    sample_rate: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum AsrStage {
    Idle,
    Attack,
    Sustain,
    Release,
}

impl ASR {
    pub fn new() -> Self {
        ASR {
            level: 0.0,
            stage: AsrStage::Idle,
            sample_rate: 44100.0,
        }
    }
}

static ASR_INPUTS: [InputSpec; 3] = [
    InputSpec { name: "gate", rate: Rate::Audio },
    InputSpec { name: "attack", rate: Rate::Audio },
    InputSpec { name: "release", rate: Rate::Audio },
];
static ASR_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for ASR {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "ASR", inputs: &ASR_INPUTS, outputs: &ASR_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.level = 0.0;
        self.stage = AsrStage::Idle;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let gate_buf = inputs[0];
        let attack_buf = inputs.get(1).copied();
        let release_buf = inputs.get(2).copied();

        for ch in 0..output.num_channels() {
            let mut level = self.level;
            let mut stage = self.stage;
            let gate_ch = gate_buf.channel(ch % gate_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let gate = gate_ch[i];
                let attack_time = attack_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.01)
                    .max(0.0001);
                let release_time = release_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.1)
                    .max(0.0001);

                let gate_on = gate > 0.0;

                match stage {
                    AsrStage::Idle => {
                        if gate_on {
                            stage = AsrStage::Attack;
                        }
                    }
                    AsrStage::Attack => {
                        if !gate_on {
                            stage = AsrStage::Release;
                        } else {
                            let rate = 1.0 / (attack_time * self.sample_rate);
                            level += rate;
                            if level >= 1.0 {
                                level = 1.0;
                                stage = AsrStage::Sustain;
                            }
                        }
                    }
                    AsrStage::Sustain => {
                        if !gate_on {
                            stage = AsrStage::Release;
                        }
                        level = 1.0;
                    }
                    AsrStage::Release => {
                        if gate_on {
                            stage = AsrStage::Attack;
                        } else {
                            let rate = 1.0 / (release_time * self.sample_rate);
                            level -= rate;
                            if level <= 0.0 {
                                level = 0.0;
                                stage = AsrStage::Idle;
                            }
                        }
                    }
                }

                out[i] = level;
            }

            if ch == 0 {
                self.level = level;
                self.stage = stage;
            }
        }
    }
}
