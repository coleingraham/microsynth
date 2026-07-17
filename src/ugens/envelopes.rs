//! Envelope UGens: Line, XLine, Perc, ExpPerc, ASR, ADSR.

use crate::buffer::{AudioBuffer, channel_wrapped, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;

// --- Line / XLine ---
//
// The linear and exponential ramps share a struct, lifecycle, and per-sample
// loop; they differ only in how a step is derived from start/end/duration and
// how it is applied to the running value. `ramp_ugen!` stamps each as a
// concrete named type.

/// Clamp a value away from zero, preserving sign.
///
/// Exponential ramps are undefined through zero, so `XLine` pins its endpoints
/// just off it.
fn clamp_nonzero(v: f32) -> f32 {
    if v == 0.0 {
        1e-6
    } else if v.abs() < 1e-6 {
        v.signum() * 1e-6
    } else {
        v
    }
}

/// Generate a ramp envelope UGen that runs from `start` to `end` over `dur`
/// seconds and then holds at `end`.
///
/// Endpoints and duration are latched once, from the first sample of the first
/// block. Variants supply:
/// - `start_default` / `end_default`: values for unconnected ports.
/// - `prepare`: `fn(start, end) -> (start, end)`, conditioning the endpoints
///   before a step is derived (`XLine` uses it to keep them non-zero and
///   same-signed).
/// - `step_identity`: the step that leaves `value` unchanged — the resting
///   state, and the step used for a zero-length ramp.
/// - `step_from`: `fn(start, end, total_samples) -> step`.
/// - `apply`: `fn(value, step) -> value`, run once per sample while ramping.
macro_rules! ramp_ugen {
    (
        $(#[$meta:meta])*
        $ty:ident, $name:literal,
        start_default = $start_default:expr,
        end_default = $end_default:expr,
        prepare = $prepare:expr,
        step_identity = $step_identity:expr,
        step_from = $step_from:expr,
        apply = $apply:expr $(,)?
    ) => {
        $(#[$meta])*
        pub struct $ty {
            value: f32,
            step: f32,
            target: f32,
            samples_remaining: u64,
            initialized: bool,
            done: bool,
            sample_rate: f32,
        }

        impl Default for $ty {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $ty {
            pub fn new() -> Self {
                $ty {
                    value: 0.0,
                    step: $step_identity,
                    target: 1.0,
                    samples_remaining: 0,
                    initialized: false,
                    done: false,
                    sample_rate: 44100.0,
                }
            }
        }

        impl UGen for $ty {
            ugen_spec!(
                $name,
                category = Envelope,
                inputs = ["start", "end", "dur"],
                outputs = ["out"]
            );

            fn init(&mut self, context: &ProcessContext) {
                self.sample_rate = context.sample_rate;
                self.initialized = false;
                self.done = false;
            }

            fn reset(&mut self) {
                self.value = 0.0;
                self.step = $step_identity;
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
                // On first block, latch start/end/dur from the first sample.
                if !self.initialized {
                    let raw_start = read_input(inputs.first().copied(), 0, 0, $start_default);
                    let raw_end = read_input(inputs.get(1).copied(), 0, 0, $end_default);
                    let dur = read_input(inputs.get(2).copied(), 0, 0, 1.0).max(0.0);

                    let (start, end) = ($prepare)(raw_start, raw_end);

                    self.value = start;
                    self.target = end;
                    let total_samples = (dur * self.sample_rate) as u64;
                    if total_samples > 0 {
                        self.step = ($step_from)(start, end, total_samples);
                        self.samples_remaining = total_samples;
                    } else {
                        self.step = $step_identity;
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
                            value = ($apply)(value, self.step);
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
    };
}

ramp_ugen! {
    /// Linear ramp from `start` to `end` over `dur` seconds, then holds at `end`.
    ///
    /// Inputs: start (initial value), end (target value), dur (duration in seconds).
    /// Reads input values at init time (first sample of first block).
    ///
    /// Reports `is_done()` = true after the ramp completes.
    Line, "Line",
    start_default = 0.0,
    end_default = 1.0,
    prepare = |start: f32, end: f32| (start, end),
    step_identity = 0.0,
    step_from = |start: f32, end: f32, total: u64| (end - start) / total as f32,
    apply = |value: f32, step: f32| value + step,
}

ramp_ugen! {
    /// Exponential ramp from `start` to `end` over `dur` seconds, then holds at `end`.
    ///
    /// Uses multiplicative interpolation: `value *= ratio` per sample.
    /// If start or end is zero or they differ in sign, values are clamped to ±1e-6.
    ///
    /// Reports `is_done()` = true after the ramp completes.
    XLine, "XLine",
    start_default = 1.0,
    end_default = 0.001,
    // Ensure same sign and non-zero for exponential interpolation.
    prepare = |raw_start: f32, raw_end: f32| {
        let start = clamp_nonzero(raw_start);
        let mut end = clamp_nonzero(raw_end);
        if (start > 0.0) != (end > 0.0) {
            // Force same sign as start
            end = end.abs() * start.signum();
        }
        (start, end)
    },
    step_identity = 1.0,
    step_from = |start: f32, end: f32, total: u64| (end / start).powf(1.0 / total as f32),
    apply = |value: f32, step: f32| value * step,
}

// --- Perc / ExpPerc ---
//
// Both percussive envelopes share a stage machine, lifecycle, and per-sample
// loop; they differ only in how the release stage decays the level and when it
// declares itself finished. `perc_ugen!` stamps each as a concrete named type.

/// The stage machine shared by every percussive envelope.
#[derive(Clone, Copy, PartialEq)]
enum PercStage {
    Attack,
    Release,
    Done,
}

/// Generate a percussive (attack-then-release) envelope UGen.
///
/// The attack stage is always a linear ramp to 1.0. Variants supply:
/// - `release_step`: `fn(level, release_time, sample_rate) -> level`, the decay
///   applied once per sample during release.
/// - `release_done`: `fn(level) -> bool`, when the decayed level counts as
///   silent, ending the envelope.
macro_rules! perc_ugen {
    (
        $(#[$meta:meta])*
        $ty:ident, $name:literal, release_step = $release_step:expr, release_done = $release_done:expr $(,)?
    ) => {
        $(#[$meta])*
        pub struct $ty {
            level: f32,
            stage: PercStage,
            done: bool,
            sample_rate: f32,
        }

        impl Default for $ty {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $ty {
            pub fn new() -> Self {
                $ty {
                    level: 0.0,
                    stage: PercStage::Attack,
                    done: false,
                    sample_rate: 44100.0,
                }
            }
        }

        impl UGen for $ty {
            ugen_spec!(
                $name,
                category = Envelope,
                inputs = ["attack", "release"],
                outputs = ["out"]
            );

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

                    for (i, out_sample) in out.iter_mut().enumerate() {
                        let attack_time = read_input(attack_buf, ch, i, 0.001).max(0.0001);
                        let release_time = read_input(release_buf, ch, i, 0.1).max(0.0001);

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
                                level = ($release_step)(level, release_time, self.sample_rate);
                                if ($release_done)(level) {
                                    level = 0.0;
                                    stage = PercStage::Done;
                                }
                            }
                            PercStage::Done => {
                                level = 0.0;
                            }
                        }

                        *out_sample = level;
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
    };
}

perc_ugen! {
    /// Percussive envelope: attack ramp up, then release ramp down to 0.
    ///
    /// Inputs: attack (seconds), release (seconds).
    /// No gate needed — fires immediately on instantiation.
    ///
    /// Reports `is_done()` = true after the release completes.
    Perc, "Perc",
    release_step = |level: f32, release_time: f32, sample_rate: f32| {
        level - 1.0 / (release_time * sample_rate)
    },
    release_done = |level: f32| level <= 0.0,
}

perc_ugen! {
    /// Exponential percussive envelope: linear attack, exponential release.
    ///
    /// Attack stage is linear (sub-millisecond, shape inaudible).
    /// Release uses `level *= coeff` where `coeff = exp(-1.0 / (release * sample_rate))`.
    /// Done when level drops below 1e-6.
    ///
    /// Reports `is_done()` = true after the release completes.
    ExpPerc, "ExpPerc",
    release_step = |level: f32, release_time: f32, sample_rate: f32| {
        level * (-1.0 / (release_time * sample_rate)).exp()
    },
    release_done = |level: f32| level < 1e-6,
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

impl Default for ASR {
    fn default() -> Self {
        Self::new()
    }
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

impl UGen for ASR {
    ugen_spec!(
        "ASR",
        category = Envelope,
        inputs = ["gate", "attack", "release"],
        outputs = ["out"]
    );

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
            let gate_ch = channel_wrapped(gate_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let gate = gate_ch[i];
                let attack_time = read_input(attack_buf, ch, i, 0.01).max(0.0001);
                let release_time = read_input(release_buf, ch, i, 0.1).max(0.0001);

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

impl Default for ADSR {
    fn default() -> Self {
        Self::new()
    }
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

impl UGen for ADSR {
    ugen_spec!(
        "ADSR",
        category = Envelope,
        inputs = ["gate", "attack", "decay", "sustain", "release"],
        outputs = ["out"]
    );

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
            let gate_ch = channel_wrapped(gate_buf, ch);
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let gate = gate_ch[i];
                let attack_time = read_input(attack_buf, ch, i, 0.01).max(0.0001);
                let decay_time = read_input(decay_buf, ch, i, 0.1).max(0.0001);
                let sustain_level = read_input(sustain_buf, ch, i, 0.5).clamp(0.0, 1.0);
                let release_time = read_input(release_buf, ch, i, 0.1).max(0.0001);

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
