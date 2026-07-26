//! Musical time primitives.
//!
//! Converts between musical positions (`bar:step +tick_offset`) and absolute
//! sample offsets. Supports arbitrary time signatures, grid resolutions, and
//! PPQN-based micro-timing for swing/humanization.
//!
//! This module is pure math with no dependency on the audio graph.

use crate::curve::GlideShape;
use alloc::vec::Vec;

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
#[derive(Debug, Clone, Copy, Default, PartialEq)]
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
        let quarter_notes_per_bar = self.numerator as f64 * (4.0 / self.denominator as f64);
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
        if total < 0.0 { 0 } else { total.round() as u64 }
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

    /// Convert a sequence of musical-time glide segments to absolute
    /// sample-time terms, in the order given.
    ///
    /// This is the pure half of musical-time sequencing: it has no
    /// dependency on a scheduler, voice, or the audio graph, so it can be
    /// tested (and reasoned about) on its own. Turning the result into
    /// scheduled events on a live voice is a separate step — see
    /// [`crate::musical_sequence::schedule_musical_glides`].
    ///
    /// Rounding rule: each segment's position and glide length are rounded
    /// to the nearest sample, exactly as [`TimeConfig::position_to_samples`]
    /// and [`TimeConfig::steps_to_secs`] already do — this function performs
    /// no additional rounding. Because the sample-based scheduler dispatches
    /// an event scheduled for sample N at the start of the block *containing*
    /// N (see `crate::scheduler`), a position that falls strictly between
    /// two block boundaries is not truncated or pushed to a neighboring
    /// block: it lands in whichever block's sample range contains its
    /// rounded sample time.
    ///
    /// Determinism: identical `segments` and an identical `TimeConfig`
    /// always produce an identical output sequence. Rendering the same
    /// `segments` through two `TimeConfig`s that differ only in `bpm`
    /// scales every position and every glide length proportionally, since
    /// both derive from the same tempo-relative quarter-note duration.
    pub fn sequence_to_samples(&self, segments: &[MusicalGlideSegment]) -> Vec<SampleTimeGlide> {
        segments
            .iter()
            .map(|segment| SampleTimeGlide {
                time: self.position_to_samples(segment.position),
                target: segment.target,
                glide_secs: self.steps_to_secs(segment.glide_steps) as f32,
                shape: segment.shape,
            })
            .collect()
    }
}

/// A single parameter update expressed in musical time: where it falls, what
/// value it moves to, how long the glide there takes (in grid steps,
/// fractional allowed, so the glide scales with tempo), and the
/// interpolation shape the glide follows.
///
/// Shape is composed from [`crate::curve::GlideShape`] rather than
/// reimplemented here; this type only carries it through to the eventual
/// scheduled event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MusicalGlideSegment {
    /// Where this update falls in musical time.
    pub position: MusicalPosition,
    /// The value the parameter glides to.
    pub target: f32,
    /// Glide length in grid steps (fractional allowed). Scales with tempo:
    /// converted to seconds via [`TimeConfig::steps_to_secs`].
    pub glide_steps: f32,
    /// Interpolation shape for the glide.
    pub shape: GlideShape,
}

impl MusicalGlideSegment {
    /// Create a new musical-time glide segment.
    pub fn new(
        position: MusicalPosition,
        target: f32,
        glide_steps: f32,
        shape: GlideShape,
    ) -> Self {
        MusicalGlideSegment {
            position,
            target,
            glide_steps,
            shape,
        }
    }
}

/// A [`MusicalGlideSegment`] converted to absolute sample-time terms by
/// [`TimeConfig::sequence_to_samples`].
///
/// Carries no voice or parameter name — pairing this with a target voice and
/// parameter to actually schedule it is a separate, non-pure step (see
/// [`crate::musical_sequence::schedule_musical_glides`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleTimeGlide {
    /// Absolute sample offset at which the glide should begin, rounded to
    /// the nearest sample.
    pub time: u64,
    /// The value the parameter glides to.
    pub target: f32,
    /// Glide duration in seconds (matches
    /// `Scheduler::schedule_param_glide`'s `glide_secs`).
    pub glide_secs: f32,
    /// Interpolation shape for the glide.
    pub shape: GlideShape,
}
