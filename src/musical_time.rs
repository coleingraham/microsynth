//! Musical time primitives.
//!
//! Converts between musical positions (`bar:step +tick_offset`) and absolute
//! sample offsets. Supports arbitrary time signatures, grid resolutions, and
//! PPQN-based micro-timing for swing/humanization.
//!
//! This module is pure math with no dependency on the audio graph.

/// Musical time configuration for a piece or section.
#[derive(Debug, Clone, Copy)]
pub struct TimeConfig {
    /// Beats per minute.
    pub bpm: f32,
    /// Time signature numerator (e.g. 4 for 4/4).
    pub numerator: u8,
    /// Time signature denominator (e.g. 4 for 4/4, 8 for 7/8).
    pub denominator: u8,
    /// Grid resolution: number of steps per bar (e.g. 16 for 16th-note grid in 4/4).
    pub grid_steps: u16,
    /// Pulses per quarter note for sub-grid micro-timing (e.g. 96, 480).
    /// Set to 0 to disable sub-grid resolution.
    pub ppqn: u16,
    /// Audio sample rate in Hz.
    pub sample_rate: f32,
}

/// A position in musical time.
#[derive(Debug, Clone, Copy, Default)]
pub struct MusicalPosition {
    /// Zero-indexed bar number.
    pub bar: u32,
    /// Zero-indexed step within the bar (0..grid_steps-1).
    pub step: u16,
    /// Sub-step offset in PPQN ticks. Positive = late (laid-back),
    /// negative = early (rushed/pre-trigger).
    pub tick_offset: i16,
}

impl MusicalPosition {
    /// Create a new musical position.
    pub fn new(bar: u32, step: u16, tick_offset: i16) -> Self {
        MusicalPosition {
            bar,
            step,
            tick_offset,
        }
    }
}

impl TimeConfig {
    /// Create a standard 4/4 config with 16th-note grid and 96 PPQN.
    pub fn new_4_4(bpm: f32, sample_rate: f32) -> Self {
        TimeConfig {
            bpm,
            numerator: 4,
            denominator: 4,
            grid_steps: 16,
            ppqn: 96,
            sample_rate,
        }
    }

    /// Duration of one quarter note in seconds.
    fn quarter_note_secs(&self) -> f64 {
        60.0 / self.bpm as f64
    }

    /// Duration of one bar in seconds.
    ///
    /// For time signature N/D, one bar contains N beats, each beat is
    /// (4/D) quarter notes long. So bar = N * (4/D) * quarter_note_secs.
    pub fn bar_duration_secs(&self) -> f64 {
        self.numerator as f64 * (4.0 / self.denominator as f64) * self.quarter_note_secs()
    }

    /// Duration of one grid step in seconds.
    pub fn step_duration_secs(&self) -> f64 {
        self.bar_duration_secs() / self.grid_steps as f64
    }

    /// Duration of one PPQN tick in seconds. Returns 0 if ppqn is 0.
    pub fn tick_duration_secs(&self) -> f64 {
        if self.ppqn == 0 {
            return 0.0;
        }
        self.quarter_note_secs() / self.ppqn as f64
    }

    /// Duration of one grid step in samples.
    pub fn step_duration_samples(&self) -> f64 {
        self.step_duration_secs() * self.sample_rate as f64
    }

    /// Number of PPQN ticks per grid step.
    ///
    /// For 4/4 with grid_steps=16 and ppqn=96: each step is a 16th note
    /// = 1/4 of a quarter note, so ticks_per_step = 96/4 = 24.
    pub fn ticks_per_step(&self) -> u32 {
        if self.ppqn == 0 || self.grid_steps == 0 {
            return 0;
        }
        // Quarter notes per bar = numerator * (4 / denominator)
        // Steps per quarter note = grid_steps / quarter_notes_per_bar
        // Ticks per step = ppqn / steps_per_quarter_note
        let quarter_notes_per_bar =
            self.numerator as f64 * (4.0 / self.denominator as f64);
        let steps_per_quarter = self.grid_steps as f64 / quarter_notes_per_bar;
        (self.ppqn as f64 / steps_per_quarter) as u32
    }

    /// Convert a musical position to an absolute sample offset.
    ///
    /// Negative tick offsets produce earlier positions (useful for pre-trigger
    /// and rushed micro-timing).
    pub fn position_to_samples(&self, pos: MusicalPosition) -> u64 {
        let bar_samples = self.bar_duration_secs() * self.sample_rate as f64;
        let step_samples = self.step_duration_samples();
        let tick_samples = self.tick_duration_secs() * self.sample_rate as f64;

        let total = pos.bar as f64 * bar_samples
            + pos.step as f64 * step_samples
            + pos.tick_offset as f64 * tick_samples;

        // Clamp to 0 (negative total can happen with pre-trigger offsets at bar 0)
        if total < 0.0 {
            0
        } else {
            total.round() as u64
        }
    }

    /// Convert a duration in grid steps to a duration in samples.
    pub fn steps_to_samples(&self, steps: f32) -> u64 {
        let samples = steps as f64 * self.step_duration_samples();
        samples.round() as u64
    }

    /// Convert a duration in grid steps to seconds.
    pub fn steps_to_secs(&self, steps: f32) -> f64 {
        steps as f64 * self.step_duration_secs()
    }
}
