//! Envelope UGens: Line, XLine, Perc, ExpPerc, ASR, ADSR.

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};

// --- Line ---

/// Linear ramp from `start` to `end` over `dur` seconds, then holds at `end`.
///
/// Inputs: start (initial value), end (target value), dur (duration in seconds).
/// Reads input values at init time (first sample of first block).
///
/// Reports `is_done()` = true after the ramp completes.
pub struct Line {
    value: f32,
    increment: f32,
    target: f32,
    samples_remaining: u64,
    initialized: bool,
    done: bool,
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
            done: false,
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
        self.done = false;
    }

    fn reset(&mut self) {
        self.value = 0.0;
        self.increment = 0.0;
        self.target = 1.0;
        self.samples_remaining = 0;
        self.initialized = false;
        self.done = false;
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
                if remaining == 0 && self.initialized {
                    self.done = true;
                }
            }
        }
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

// --- XLine ---

/// Exponential ramp from `start` to `end` over `dur` seconds, then holds at `end`.
///
/// Uses multiplicative interpolation: `value *= ratio` per sample.
/// If start or end is zero or they differ in sign, values are clamped to ±1e-6.
///
/// Reports `is_done()` = true after the ramp completes.
pub struct XLine {
    value: f32,
    ratio: f32,
    target: f32,
    samples_remaining: u64,
    initialized: bool,
    done: bool,
    sample_rate: f32,
}

impl XLine {
    pub fn new() -> Self {
        XLine {
            value: 0.0,
            ratio: 1.0,
            target: 1.0,
            samples_remaining: 0,
            initialized: false,
            done: false,
            sample_rate: 44100.0,
        }
    }

    /// Clamp a value away from zero, preserving sign.
    fn clamp_nonzero(v: f32) -> f32 {
        if v == 0.0 {
            1e-6
        } else if v.abs() < 1e-6 {
            v.signum() * 1e-6
        } else {
            v
        }
    }
}

static XLINE_INPUTS: [InputSpec; 3] = [
    InputSpec { name: "start", rate: Rate::Audio },
    InputSpec { name: "end", rate: Rate::Audio },
    InputSpec { name: "dur", rate: Rate::Audio },
];
static XLINE_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for XLine {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "XLine", inputs: &XLINE_INPUTS, outputs: &XLINE_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        self.initialized = false;
        self.done = false;
    }

    fn reset(&mut self) {
        self.value = 0.0;
        self.ratio = 1.0;
        self.target = 1.0;
        self.samples_remaining = 0;
        self.initialized = false;
        self.done = false;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        if !self.initialized {
            let raw_start = inputs.first()
                .map(|b| b.channel(0).samples()[0])
                .unwrap_or(1.0);
            let raw_end = inputs.get(1)
                .map(|b| b.channel(0).samples()[0])
                .unwrap_or(0.001);
            let dur = inputs.get(2)
                .map(|b| b.channel(0).samples()[0])
                .unwrap_or(1.0)
                .max(0.0);

            // Ensure same sign and non-zero for exponential interpolation
            let start = Self::clamp_nonzero(raw_start);
            let mut end = Self::clamp_nonzero(raw_end);
            if (start > 0.0) != (end > 0.0) {
                // Force same sign as start
                end = end.abs() * start.signum();
            }

            self.value = start;
            self.target = end;
            let total_samples = (dur * self.sample_rate) as u64;
            if total_samples > 0 {
                self.ratio = (end / start).powf(1.0 / total_samples as f32);
                self.samples_remaining = total_samples;
            } else {
                self.ratio = 1.0;
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
                    value *= self.ratio;
                    remaining -= 1;
                    if remaining == 0 {
                        value = self.target;
                    }
                }
            }

            if ch == 0 {
                self.value = value;
                self.samples_remaining = remaining;
                if remaining == 0 && self.initialized {
                    self.done = true;
                }
            }
        }
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

// --- Perc ---

/// Percussive envelope: attack ramp up, then release ramp down to 0.
///
/// Inputs: attack (seconds), release (seconds).
/// No gate needed — fires immediately on instantiation.
///
/// Reports `is_done()` = true after the release completes.
pub struct Perc {
    level: f32,
    stage: PercStage,
    done: bool,
    sample_rate: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum PercStage {
    Attack,
    Release,
    Done,
}

impl Perc {
    pub fn new() -> Self {
        Perc {
            level: 0.0,
            stage: PercStage::Attack,
            done: false,
            sample_rate: 44100.0,
        }
    }
}

static PERC_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "attack", rate: Rate::Audio },
    InputSpec { name: "release", rate: Rate::Audio },
];
static PERC_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Perc {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Perc", inputs: &PERC_INPUTS, outputs: &PERC_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        self.level = 0.0;
        self.stage = PercStage::Attack;
        self.done = false;
    }

    fn reset(&mut self) {
        self.level = 0.0;
        self.stage = PercStage::Attack;
        self.done = false;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let attack_buf = inputs.first().copied();
        let release_buf = inputs.get(1).copied();

        for ch in 0..output.num_channels() {
            let mut level = self.level;
            let mut stage = self.stage;
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let attack_time = attack_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.001)
                    .max(0.0001);
                let release_time = release_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.1)
                    .max(0.0001);

                match stage {
                    PercStage::Attack => {
                        let rate = 1.0 / (attack_time * self.sample_rate);
                        level += rate;
                        if level >= 1.0 {
                            level = 1.0;
                            stage = PercStage::Release;
                        }
                    }
                    PercStage::Release => {
                        let rate = 1.0 / (release_time * self.sample_rate);
                        level -= rate;
                        if level <= 0.0 {
                            level = 0.0;
                            stage = PercStage::Done;
                        }
                    }
                    PercStage::Done => {
                        level = 0.0;
                    }
                }

                out[i] = level;
            }

            if ch == 0 {
                self.level = level;
                self.stage = stage;
                if stage == PercStage::Done {
                    self.done = true;
                }
            }
        }
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

// --- ExpPerc ---

/// Exponential percussive envelope: linear attack, exponential release.
///
/// Attack stage is linear (sub-millisecond, shape inaudible).
/// Release uses `level *= coeff` where `coeff = exp(-1.0 / (release * sample_rate))`.
/// Done when level drops below 1e-6.
///
/// Reports `is_done()` = true after the release completes.
pub struct ExpPerc {
    level: f32,
    stage: ExpPercStage,
    done: bool,
    sample_rate: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum ExpPercStage {
    Attack,
    Release,
    Done,
}

impl ExpPerc {
    pub fn new() -> Self {
        ExpPerc {
            level: 0.0,
            stage: ExpPercStage::Attack,
            done: false,
            sample_rate: 44100.0,
        }
    }
}

static EXPPERC_INPUTS: [InputSpec; 2] = [
    InputSpec { name: "attack", rate: Rate::Audio },
    InputSpec { name: "release", rate: Rate::Audio },
];
static EXPPERC_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for ExpPerc {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "ExpPerc", inputs: &EXPPERC_INPUTS, outputs: &EXPPERC_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        self.level = 0.0;
        self.stage = ExpPercStage::Attack;
        self.done = false;
    }

    fn reset(&mut self) {
        self.level = 0.0;
        self.stage = ExpPercStage::Attack;
        self.done = false;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let attack_buf = inputs.first().copied();
        let release_buf = inputs.get(1).copied();

        for ch in 0..output.num_channels() {
            let mut level = self.level;
            let mut stage = self.stage;
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let attack_time = attack_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.001)
                    .max(0.0001);
                let release_time = release_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.1)
                    .max(0.0001);

                match stage {
                    ExpPercStage::Attack => {
                        let rate = 1.0 / (attack_time * self.sample_rate);
                        level += rate;
                        if level >= 1.0 {
                            level = 1.0;
                            stage = ExpPercStage::Release;
                        }
                    }
                    ExpPercStage::Release => {
                        let coeff = (-1.0 / (release_time * self.sample_rate)).exp();
                        level *= coeff;
                        if level < 1e-6 {
                            level = 0.0;
                            stage = ExpPercStage::Done;
                        }
                    }
                    ExpPercStage::Done => {
                        level = 0.0;
                    }
                }

                out[i] = level;
            }

            if ch == 0 {
                self.level = level;
                self.stage = stage;
                if stage == ExpPercStage::Done {
                    self.done = true;
                }
            }
        }
    }

    fn is_done(&self) -> bool {
        self.done
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
///
/// Reports `is_done()` = true when the envelope returns to Idle after Release.
pub struct ASR {
    level: f32,
    stage: AsrStage,
    triggered: bool,
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
            triggered: false,
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
        self.triggered = false;
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
            let mut triggered = self.triggered;
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
                            triggered = true;
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
                self.triggered = triggered;
            }
        }
    }

    fn is_done(&self) -> bool {
        // Done when we've been triggered and returned to idle
        self.triggered && self.stage == AsrStage::Idle
    }
}

// --- ADSR ---

/// Full ADSR envelope with configurable sustain level.
///
/// Inputs: gate, attack (seconds), decay (seconds), sustain (level 0-1), release (seconds).
///
/// - Gate on: Attack → peak (1.0) → Decay → Sustain level
/// - Gate off: Release → 0.0
///
/// Reports `is_done()` = true when the envelope returns to Idle after Release.
pub struct ADSR {
    level: f32,
    stage: AdsrStage,
    triggered: bool,
    sample_rate: f32,
}

#[derive(Clone, Copy, PartialEq)]
enum AdsrStage {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl ADSR {
    pub fn new() -> Self {
        ADSR {
            level: 0.0,
            stage: AdsrStage::Idle,
            triggered: false,
            sample_rate: 44100.0,
        }
    }
}

static ADSR_INPUTS: [InputSpec; 5] = [
    InputSpec { name: "gate", rate: Rate::Audio },
    InputSpec { name: "attack", rate: Rate::Audio },
    InputSpec { name: "decay", rate: Rate::Audio },
    InputSpec { name: "sustain", rate: Rate::Audio },
    InputSpec { name: "release", rate: Rate::Audio },
];
static ADSR_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for ADSR {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "ADSR", inputs: &ADSR_INPUTS, outputs: &ADSR_OUTPUTS }
    }

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.level = 0.0;
        self.stage = AdsrStage::Idle;
        self.triggered = false;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let gate_buf = inputs[0];
        let attack_buf = inputs.get(1).copied();
        let decay_buf = inputs.get(2).copied();
        let sustain_buf = inputs.get(3).copied();
        let release_buf = inputs.get(4).copied();

        for ch in 0..output.num_channels() {
            let mut level = self.level;
            let mut stage = self.stage;
            let mut triggered = self.triggered;
            let gate_ch = gate_buf.channel(ch % gate_buf.num_channels()).samples();
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let gate = gate_ch[i];
                let attack_time = attack_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.01)
                    .max(0.0001);
                let decay_time = decay_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.1)
                    .max(0.0001);
                let sustain_level = sustain_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                let release_time = release_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.1)
                    .max(0.0001);

                let gate_on = gate > 0.0;

                match stage {
                    AdsrStage::Idle => {
                        if gate_on {
                            stage = AdsrStage::Attack;
                            triggered = true;
                        }
                    }
                    AdsrStage::Attack => {
                        if !gate_on {
                            stage = AdsrStage::Release;
                        } else {
                            let rate = 1.0 / (attack_time * self.sample_rate);
                            level += rate;
                            if level >= 1.0 {
                                level = 1.0;
                                stage = AdsrStage::Decay;
                            }
                        }
                    }
                    AdsrStage::Decay => {
                        if !gate_on {
                            stage = AdsrStage::Release;
                        } else {
                            let rate = (1.0 - sustain_level) / (decay_time * self.sample_rate);
                            level -= rate;
                            if level <= sustain_level {
                                level = sustain_level;
                                stage = AdsrStage::Sustain;
                            }
                        }
                    }
                    AdsrStage::Sustain => {
                        if !gate_on {
                            stage = AdsrStage::Release;
                        }
                        level = sustain_level;
                    }
                    AdsrStage::Release => {
                        if gate_on {
                            stage = AdsrStage::Attack;
                        } else {
                            let rate = level.max(0.001) / (release_time * self.sample_rate);
                            level -= rate;
                            if level <= 0.0 {
                                level = 0.0;
                                stage = AdsrStage::Idle;
                            }
                        }
                    }
                }

                out[i] = level;
            }

            if ch == 0 {
                self.level = level;
                self.stage = stage;
                self.triggered = triggered;
            }
        }
    }

    fn is_done(&self) -> bool {
        self.triggered && self.stage == AdsrStage::Idle
    }
}
