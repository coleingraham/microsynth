//! Pitch and tuning primitives.
//!
//! Provides MIDI↔Hz conversion, cent offset application, and tuning tables
//! for 12-TET, quarter-tone (24-TET), just intonation, and arbitrary
//! cent-per-octave arrays (e.g. Slendro, Pelog, Maqam systems).
//!
//! This module is pure math with no dependency on the audio graph.

use alloc::vec::Vec;

/// A tuning system that maps note numbers to frequencies.
///
/// Notes are integer indices (like MIDI note numbers). The table defines
/// cent offsets for each degree within one octave; octave transposition
/// is handled automatically.
#[derive(Clone, Debug)]
pub struct TuningTable {
    /// Reference note number (default: 69 = A4 in MIDI).
    pub anchor_note: u8,
    /// Frequency of the anchor note in Hz (default: 440.0).
    pub anchor_freq: f32,
    /// Cent offsets for each degree within one octave.
    /// Length determines the number of divisions per octave.
    /// For 12-TET: `[0, 100, 200, 300, ..., 1100]`.
    pub cents: Vec<f32>,
}

impl TuningTable {
    /// Standard 12-tone equal temperament with A4 = 440 Hz.
    pub fn equal_temperament_12() -> Self {
        let cents: Vec<f32> = (0..12).map(|i| i as f32 * 100.0).collect();
        TuningTable {
            anchor_note: 69,
            anchor_freq: 440.0,
            cents,
        }
    }

    /// 24-tone equal temperament (quarter-tone) with A4 = 440 Hz.
    ///
    /// Note numbers span 0..N with 24 divisions per octave, so MIDI-like
    /// note 69 becomes `69 * 2 = 138` in this space.
    pub fn equal_temperament_24() -> Self {
        let cents: Vec<f32> = (0..24).map(|i| i as f32 * 50.0).collect();
        TuningTable {
            anchor_note: 138, // A4 in 24-TET note space
            anchor_freq: 440.0,
            cents,
        }
    }

    /// 5-limit just intonation, 12 notes per octave, A4 = 440 Hz.
    ///
    /// Ratios: 1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8.
    pub fn just_intonation() -> Self {
        let ratios: [f64; 12] = [
            1.0,
            16.0 / 15.0,
            9.0 / 8.0,
            6.0 / 5.0,
            5.0 / 4.0,
            4.0 / 3.0,
            45.0 / 32.0,
            3.0 / 2.0,
            8.0 / 5.0,
            5.0 / 3.0,
            9.0 / 5.0,
            15.0 / 8.0,
        ];
        let cents: Vec<f32> = ratios
            .iter()
            .map(|r| (1200.0 * r.log2()) as f32)
            .collect();
        TuningTable {
            anchor_note: 69,
            anchor_freq: 440.0,
            cents,
        }
    }

    /// Create a tuning table from an arbitrary array of cent values (one octave).
    pub fn from_cents(cents: &[f32], anchor_note: u8, anchor_freq: f32) -> Self {
        TuningTable {
            anchor_note,
            anchor_freq,
            cents: cents.into(),
        }
    }

    /// Number of divisions per octave in this tuning.
    pub fn divisions(&self) -> usize {
        self.cents.len()
    }

    /// Convert a note number to Hz using this tuning table.
    ///
    /// Integer note numbers map directly to table degrees. Fractional values
    /// interpolate linearly between adjacent cent entries.
    pub fn note_to_hz(&self, note: f32) -> f32 {
        let divisions = self.cents.len() as f32;
        let relative = note - self.anchor_note as f32;

        // Decompose into whole octaves and fractional degree within the octave
        let mut octaves = (relative / divisions).floor();
        let mut degree_frac = relative - octaves * divisions;

        // Ensure degree_frac is in [0, divisions)
        if degree_frac < 0.0 {
            degree_frac += divisions;
            octaves -= 1.0;
        }

        // Interpolate cents between adjacent degrees
        let degree_lo = degree_frac.floor() as usize;
        let frac = degree_frac - degree_frac.floor();
        let len = self.cents.len();

        let cents_lo = self.cents[degree_lo % len];
        let cents_hi = if degree_lo + 1 < len {
            self.cents[degree_lo + 1]
        } else {
            1200.0 // wraps to next octave
        };
        let cents_interp = cents_lo + frac * (cents_hi - cents_lo);

        let total_cents = octaves as f64 * 1200.0 + cents_interp as f64;
        (self.anchor_freq as f64 * pow2_f64(total_cents / 1200.0)) as f32
    }

    /// Convert Hz to the nearest note number (approximate inverse).
    pub fn hz_to_note(&self, hz: f32) -> f32 {
        if hz <= 0.0 {
            return 0.0;
        }
        let divisions = self.cents.len() as f64;
        let total_cents = 1200.0 * (hz as f64 / self.anchor_freq as f64).log2();

        let octaves = (total_cents / 1200.0).floor();
        let remaining = total_cents - octaves * 1200.0;

        // Find the closest degree
        let mut best_degree = 0usize;
        let mut best_diff = f64::MAX;
        for (i, &c) in self.cents.iter().enumerate() {
            let diff = (remaining - c as f64).abs();
            if diff < best_diff {
                best_diff = diff;
                best_degree = i;
            }
        }

        self.anchor_note as f32 + (octaves * divisions) as f32 + best_degree as f32
    }
}

impl Default for TuningTable {
    fn default() -> Self {
        Self::equal_temperament_12()
    }
}

// ---------------------------------------------------------------------------
// Free functions (fast paths, no allocation)
// ---------------------------------------------------------------------------

/// Convert a MIDI note number to Hz using standard 12-TET.
///
/// `a4_freq` is the reference frequency for MIDI note 69 (typically 440.0).
/// Supports fractional note numbers for pitch bends.
pub fn midi_to_hz_12tet(note: f32, a4_freq: f32) -> f32 {
    a4_freq * pow2_f32((note - 69.0) / 12.0)
}

/// Convert Hz to MIDI note number using standard 12-TET.
///
/// Returns a fractional value for pitches between semitones.
pub fn hz_to_midi_12tet(hz: f32, a4_freq: f32) -> f32 {
    if hz <= 0.0 {
        return 0.0;
    }
    69.0 + 12.0 * (hz / a4_freq).log2()
}

/// Apply a cent offset to a base frequency.
///
/// `apply_cents(440.0, 100.0)` returns one semitone above A4 (~466.16 Hz).
/// `apply_cents(440.0, -50.0)` returns a quarter-tone flat.
pub fn apply_cents(base_hz: f32, cents: f32) -> f32 {
    base_hz * pow2_f32(cents / 1200.0)
}

// ---------------------------------------------------------------------------
// no_std-compatible pow2 helpers using intrinsic math
// ---------------------------------------------------------------------------

/// 2^x for f32.
fn pow2_f32(x: f32) -> f32 {
    (x * core::f32::consts::LN_2).exp()
}

/// 2^x for f64.
fn pow2_f64(x: f64) -> f64 {
    (x * core::f64::consts::LN_2).exp()
}
