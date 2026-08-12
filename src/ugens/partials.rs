//! Partials + shaped-noise ugen (MOT-636): the direct-synthesis half of
//! `motif-docs/rfcs/2026-08-08-nmf-multi-channel-timbre.md`'s "Direct
//! synthesis path" (requirements 1-8). Renders a MOT-634 coefficient-table
//! bank entry as an `M_p`-sinusoid oscillator bank at partial frequencies
//! `m * f0 * stretch` plus `J` pre-shaped filtered-noise generators mixed by
//! band gains — bypassing `W@H` -> Griffin-Lim entirely.
//!
//! ## What this ugen consumes, and what it deliberately does not
//!
//! Per-frame controls are exactly two ordinary audio-rate ports: `freq` (a
//! continuous f0 in Hz — vibrato, bends, and portamento are just an
//! audio-rate signal, no special handling) and `gain` (the RFC's `H_p(t)`,
//! the velocity proxy). Everything else the ugen needs — which partials
//! exist, their coefficient weights, the inharmonicity stretch — comes from
//! the table it is bound to (`crate::coeff_table::CoeffTable`, resolved at
//! construction; see `docs/coeff-table-bank-format.md`).
//!
//! **Channel mixing is out of scope by design**, not by omission. RFC
//! requirement 7 and this ticket's own context note both say channel mixing
//! (softmax over a pitch's K channels, and the velocity->alpha map that
//! feeds it) is upstream arithmetic, not the ugen's business. A `CoeffTable`
//! entry can carry K channels (MOT-634's format), but this ugen renders
//! exactly one of them, picked once at construction time
//! (`with_channel`, default channel 0, fallible — see its doc — when the
//! requested channel resolves for none of the table's entries) — it never
//! blends channels at render time. A caller that wants a specific pre-mixed
//! timbre uploads a table whose entries already carry that mix in the
//! selected channel; this ugen's job stops at rendering whichever fixed
//! coefficient vector each entry hands it, continuously reinterpreted as f0
//! glides across the pitch grid.
//!
//! ## The two axes of "coefficient... update" smoothing (RFC requirement 6)
//!
//! There is no separate frame-rate coefficient *stream* port in this ticket's
//! scope (no port shape in this engine can carry a variable-length per-frame
//! vector). Instead, the "coefficient vector in effect" is entirely a
//! function of the current f0: which two table entries bracket it, and the
//! crossfade weight between them (requirement 2). Because `freq` is read at
//! audio rate and the bracket/crossfade/stretch/partial-frequency math below
//! runs fresh every sample, the coefficient and gain ramps requirement 6
//! asks for fall out for free at full sample resolution — no separate
//! block-rate ramp state machine is needed on top.
//!
//! ## Birth/death and Nyquist crossing (requirement 3) without extra state
//!
//! Partial index `m` is interpolated as a convex combination of the two
//! bracketing entries' coefficient rows, zero-padded to whichever entry has
//! fewer partials (`M_p` may differ between grid points). Because the
//! crossfade weight `t` moves continuously from 0 to 1 across the *entire*
//! inter-grid span (not just at the boundary sample), a partial that only
//! exists at one endpoint fades in/out smoothly over that whole span — the
//! birth/death ramp is the interpolation itself, not a bolted-on state
//! machine, and it is continuous at grid points by construction (at `f0 ==`
//! an entry's own f0, `t` is exactly 0 or 1 in both adjacent brackets, so the
//! two segments agree there). A partial whose *frequency* (not its table
//! presence) crosses Nyquist during a glide is separately faded via
//! [`nyquist_fade`] — a continuous function of the instantaneous partial
//! frequency, so it never introduces a signal discontinuity of its own.
//!
//! ## Simplex closure (requirement 4)
//!
//! Both endpoint coefficient rows are L1-unit by the RFC's channel
//! parameterization; a convex combination of L1-unit vectors (even
//! zero-padded ones — zero-padding never changes an L1 sum) is itself
//! L1-unit. This ugen applies **no renormalization** anywhere in the render
//! path.
//!
//! For the **partial half**, that gives an exact L1 energy budget: each
//! partial's amplitude is `coefficient * mainlobe_gain * fade * gain`,
//! summed directly, and since each term is `amp_m * sin(...)` with `|sin| <=
//! 1`, the triangle inequality gives `|sum_m amp_m * sin(...)| <= gain *
//! mainlobe_gain * sum_m |coeff_m|` for every sample — total output
//! amplitude tracks `gain` exactly through an interpolated glide.
//!
//! The **noise half does not inherit that bound in the same form.** Each
//! band's contribution is `coeff_j * noise_gain * gain * filtered_j`, where
//! `filtered_j` is a resonant biquad's *output*, not its input — and a
//! stable IIR filter's worst-case gain against a bounded-magnitude input
//! (`Rng::next_bipolar()` is uniform on `[-1, 1]`) is its impulse response's
//! L1 norm `‖h_j‖₁` (`sum_n |h_j[n]|`), not 1. For [`NOISE_BAND_Q`]'s
//! constant-peak-gain resonance, `‖h_j‖₁` measures **> 1** (QA measured a
//! real rendered peak of 0.9028 at `gain = 0.9`, all L1 mass on one noise
//! band, no metadata — exceeding the naive `gain * 1.0 = 0.9` bound; see
//! `tests/partials_noise.rs`'s `noise_band_l1_bound_requires_the_filter_impulse_response_norm`,
//! which reproduces that measurement and pins the correct bound: `coeff_j *
//! noise_gain * gain * ‖h_j‖₁`, not `coeff_j * noise_gain * gain`).
//!
//! This is not a headroom risk in practice: the exporter's shipped
//! `noise_gain` values are ≈0.058 (`motif-soundmatch`'s
//! `channel_export.py`), which keeps real dictionary tables far below the
//! bound even with `‖h_j‖₁ > 1`. The point of this section is what the
//! invariant precisely *is* — exact for the partial half, filter-dependent
//! for the noise half — not a claim that shipped output risks clipping.
//! `tests/partials_noise.rs` checks the partial-only bound directly (no
//! noise mass) and separately closes the noise-band case with its own
//! filter-dependent bound.
//!
//! ## The deterministic bridge (RFC "The bridge is deterministic")
//!
//! Column mass -> sinusoid amplitude (window mainlobe gain) and bump mass ->
//! noise-generator gain (expected white-noise magnitude through the analysis
//! window) are, per the RFC, per-column scalars computed once by whichever
//! producer built the table (the analysis side knows its own window/FFT
//! config; this crate does not). This ugen reads them from each entry's
//! open metadata slot under the keys [`MAINLOBE_GAIN_KEY`] / [`NOISE_GAIN_KEY`]
//! — it applies the scalars, it does not derive them. The absent-key default
//! is `docs/coeff-table-bank-format.md`'s Metadata section's contract to
//! keep, not restated here; [`metadata_scalar`]'s call sites below are the
//! implementation of it.

use crate::buffer::{AudioBuffer, read_input};
use crate::coeff_table::CoeffTable;
use crate::context::{ProcessContext, Rate};
use crate::dsl::compiler::{TableUGenFactory, UGenRegistry};
use crate::node::{InputSpec, OutputSpec, UGen, UGenCategory};
use crate::ugens::filters::{BiquadState, biquad_bpf_coeffs};
use crate::ugens::rng::Rng;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::f32::consts::TAU;

/// Channel selected from each dictionary entry when constructed via
/// [`PartialsNoise::new`]. See the module doc's "Channel mixing is out of
/// scope by design" note.
const DEFAULT_CHANNEL: u32 = 0;

/// Metadata key (`docs/coeff-table-bank-format.md` Part 1's open metadata
/// slot) for the column-mass -> sinusoid-amplitude conversion scalar.
pub const MAINLOBE_GAIN_KEY: &str = "mainlobe_gain";
/// Metadata key for the bump-mass -> noise-generator-gain conversion scalar.
pub const NOISE_GAIN_KEY: &str = "noise_gain";

/// Default seed for the shared shaped-noise source. Mirrors `WhiteNoise`'s /
/// `PinkNoise`'s fixed-default-seed convention (`ugens/noise.rs`); a voice
/// spawn is expected to call [`UGen::reseed_noise`] for keyed, per-voice
/// determinism (see `noise.rs`'s module doc on the seeding contract).
const PARTIALS_NOISE_DEFAULT_SEED: u32 = 0x9A27_1E55;

/// Default `freq`/`gain` values for an unconnected port.
const DEFAULT_FREQ_HZ: f32 = 220.0;
const DEFAULT_GAIN: f32 = 1.0;

/// The fixed smooth-noise basis's frequency span. Table-independent by
/// design (RFC: "`N` — a **fixed** bins x J smooth noise basis"): band
/// identity (which center frequency "band index j" means) must not depend on
/// which table, or which of a table's entries, is currently active, or a
/// glide that changes `j_noise` between entries would reassign a band's
/// frequency mid-note. Mel-spaced across a fixed, generous vocal/instrument
/// range; the top end is clamped to the render sample rate's Nyquist in
/// [`PartialsNoise::init`], not here (these two constants are computed once,
/// before sample rate is known).
const NOISE_BAND_MIN_HZ: f32 = 80.0;
const NOISE_BAND_MAX_HZ: f32 = 12_000.0;

/// Resonance of each fixed noise-band bandpass. Low (wide) on purpose: the
/// RFC describes the noise basis as "smooth... bumps," not narrow spectral
/// lines, so each band should pass a broad neighborhood rather than ring at
/// a single frequency.
const NOISE_BAND_Q: f32 = 1.0;

/// Fraction of the Nyquist range, immediately below Nyquist, over which a
/// partial whose frequency is climbing toward it is faded to zero (see
/// [`nyquist_fade`]). Sized so the fade band comfortably spans multiple
/// samples' worth of frequency motion for any realistic glide rate, while
/// staying small enough that it only engages partials that are genuinely
/// about to alias, not the bulk of the spectrum.
const NYQUIST_GUARD_FRACTION: f32 = 0.05;

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Linear-interpolate coefficient `idx` between the low/high pitch-bracket
/// entries' coefficient slice, treating an out-of-range index as silent
/// (0.0) — the two bracket entries can carry different partial/noise
/// support widths.
#[inline]
fn lerp_coeff(lo: &[f32], hi: &[f32], idx: usize, t: f32) -> f32 {
    let c_lo = lo.get(idx).copied().unwrap_or(0.0);
    let c_hi = hi.get(idx).copied().unwrap_or(0.0);
    lerp(c_lo, c_hi, t)
}

/// Linear fade-to-zero for a partial whose instantaneous frequency is
/// approaching or has crossed Nyquist. `1.0` below the guard band, `0.0` at
/// or above Nyquist, linear in between — continuous in `freq`, so a glide
/// that pushes a partial through Nyquist never introduces a sample
/// discontinuity of its own (RFC requirement 3, "partials cross Nyquist
/// during glides").
#[inline]
fn nyquist_fade(freq: f32, nyquist: f32) -> f32 {
    if nyquist <= 0.0 {
        return 0.0;
    }
    let guard_start = nyquist * (1.0 - NYQUIST_GUARD_FRACTION);
    if freq <= guard_start {
        1.0
    } else if freq >= nyquist {
        0.0
    } else {
        (nyquist - freq) / (nyquist - guard_start)
    }
}

/// O'Shaughnessy mel scale, used only to space the fixed noise-band centers
/// (see [`NOISE_BAND_MIN_HZ`]/[`NOISE_BAND_MAX_HZ`]) — not a claim about
/// perceptual accuracy, just a smoothly-decreasing-resolution spacing that
/// puts more bands in the low end where "smooth noise bump" content
/// typically concentrates (breath, key/hammer noise).
#[inline]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

#[inline]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Mel-spaced, table-independent center frequencies for `j_max` fixed noise
/// bands (raw — not yet clamped to a render sample rate's Nyquist; see
/// [`PartialsNoise::init`]).
fn noise_band_centers(j_max: usize) -> Vec<f32> {
    if j_max == 0 {
        return Vec::new();
    }
    let mel_lo = hz_to_mel(NOISE_BAND_MIN_HZ);
    let mel_hi = hz_to_mel(NOISE_BAND_MAX_HZ);
    (0..j_max)
        .map(|j| {
            let frac = (j as f32 + 0.5) / j_max as f32;
            mel_to_hz(lerp(mel_lo, mel_hi, frac))
        })
        .collect()
}

fn metadata_scalar(metadata: &[(alloc::string::String, f32)], key: &str, default: f32) -> f32 {
    metadata
        .iter()
        .find(|(k, _)| k.as_str() == key)
        .map(|(_, v)| *v)
        .unwrap_or(default)
}

/// One dictionary pitch entry's precomputed, render-ready content: the
/// selected channel's coefficient row split into its partial and noise-band
/// halves, plus the entry's own deterministic-bridge scalars. Extracted once
/// at construction from the bound [`CoeffTable`] (see the module doc's
/// "Resolution is a snapshot" note carried over from MOT-634) — `process`
/// never touches the table itself.
struct EntryData {
    f0_hz: f32,
    stretch: f32,
    mainlobe_gain: f32,
    noise_gain: f32,
    /// Length `M_p` for this entry.
    partial_coeffs: Vec<f32>,
    /// Length `J` for this entry (may differ from other entries' `J`).
    noise_coeffs: Vec<f32>,
}

/// Returned by [`PartialsNoise::with_channel`] when `channel` resolves for
/// **none** of a non-empty table's entries (e.g. every entry's `k_channels
/// <= channel`) — a structurally different failure from a single bad entry
/// among otherwise-fine ones, which [`extract_entries`] still tolerates (see
/// its doc). A table with zero entries to begin with is a separate,
/// pre-existing, tolerated case (an intentionally empty/not-yet-loaded
/// dictionary) and does not produce this error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoEntriesForChannel {
    /// The channel index that was requested.
    pub channel: u32,
    /// How many entries the table had, all of which failed to resolve
    /// `channel`.
    pub table_entries: usize,
}

impl core::fmt::Display for NoEntriesForChannel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "channel {} resolved for none of this table's {} entries",
            self.channel, self.table_entries
        )
    }
}

/// Extract and sort every resolvable entry of `table`'s channel `channel`.
///
/// An entry whose `channel` is out of range for that entry (or the entry is
/// malformed — only reachable for a hand-built `CoeffTable` that bypassed
/// `CoeffTable::from_bytes`'s decode-time validation) is skipped rather than
/// panicking: a bad entry in an otherwise-fine table should not take the
/// audio thread down. Sorted by ascending `f0_hz` regardless of the table's
/// own entry order — `CoeffTable`'s doc states ascending-f0 order is "by
/// convention (not enforced)", so a consumer doing pitch-bracket search
/// must sort or verify, not assume.
///
/// This tolerance is per-entry only. A caller where **every** entry fails to
/// resolve `channel` gets a loud [`NoEntriesForChannel`] from
/// [`PartialsNoise::with_channel`], not a silently entry-less (and therefore
/// silently zero-output) ugen — see that function's doc.
fn extract_entries(table: &CoeffTable, channel: u32) -> Vec<EntryData> {
    let mut out: Vec<EntryData> = table
        .entries
        .iter()
        .filter_map(|e| {
            let row = e.channel(channel)?;
            let m = e.m_partials().min(row.len());
            let (partial_coeffs, noise_coeffs) = row.split_at(m);
            Some(EntryData {
                f0_hz: e.f0_hz,
                stretch: e.inharmonicity_stretch,
                mainlobe_gain: metadata_scalar(&e.metadata, MAINLOBE_GAIN_KEY, 1.0),
                noise_gain: metadata_scalar(&e.metadata, NOISE_GAIN_KEY, 1.0),
                partial_coeffs: partial_coeffs.to_vec(),
                noise_coeffs: noise_coeffs.to_vec(),
            })
        })
        .collect();
    out.sort_by(|a, b| a.f0_hz.total_cmp(&b.f0_hz));
    out
}

/// Partials + shaped-noise direct-synthesis ugen (MOT-636). See the module
/// doc for the full design; in short: an `M_p`-sinusoid oscillator bank at
/// `m * freq * stretch` plus `J` fixed shaped-noise bands, both continuously
/// reinterpreted from a table-bound dictionary as `freq` glides across the
/// pitch grid, scaled by `gain`.
///
/// Table-bound: constructed only via [`register`]'s
/// [`crate::dsl::compiler::UGenRegistry::register_table_bound`] registration
/// (there is no DSL text syntax for a table reference — see
/// `docs/coeff-table-bank-format.md`), never directly from DSL source.
///
/// Inputs: `freq` (continuous f0, Hz), `gain` (H(t), the velocity proxy —
/// requirement 7 keeps any velocity *semantics* upstream of this ugen).
pub struct PartialsNoise {
    entries: Vec<EntryData>,
    /// `max` noise-band count across every resolved entry — the stable size
    /// of `noise_state`/`noise_biquad`, so a given band index keeps one
    /// continuous filter identity across brackets.
    j_max: usize,
    /// One phase accumulator per partial index, sized to the `max` partial
    /// count across every resolved entry at construction — the stable
    /// identity that lets a given partial index keep one continuous phase
    /// accumulator across brackets (RFC requirement 5, "phase continuity for
    /// free").
    phase: Vec<f32>,
    noise_state: Vec<BiquadState>,
    /// `(b0, b1, b2, a1, a2)` per band — fixed once sample rate is known in
    /// [`init`](UGen::init); the bands themselves never move.
    noise_biquad: Vec<(f32, f32, f32, f32, f32)>,
    /// Raw (pre-Nyquist-clamp) band centers, computed once at construction.
    noise_center_hz: Vec<f32>,
    rng: Rng,
    sample_rate: f32,
}

impl PartialsNoise {
    /// Construct bound to `table`, rendering channel [`DEFAULT_CHANNEL`] (0)
    /// of each entry. The table-bound factory [`register`] registers
    /// constructs.
    ///
    /// # Panics
    ///
    /// Panics if `table` is non-empty but no entry resolves channel 0 (e.g.
    /// every entry's `k_channels == 0`) — see [`with_channel`](Self::with_channel)'s
    /// doc for why this is loud rather than tolerated. This is a
    /// construction-time (not audio-thread) panic: [`register`]'s factory
    /// runs during synth instantiation, before any audio renders. For a
    /// malformed table produced by [`CoeffTable::from_bytes`], this should
    /// be unreachable — decode-time validation rejects it there; it remains
    /// reachable only for a hand-built `CoeffTable` that bypassed that
    /// validation. A well-formed *empty* table (zero entries — e.g. no
    /// dictionary loaded yet) is unaffected and still constructs normally.
    pub fn new(table: Arc<CoeffTable>) -> Self {
        Self::with_channel(table, DEFAULT_CHANNEL).expect(
            "channel 0 always resolves for a well-formed, non-empty CoeffTable \
             (k_channels >= 1 on every entry); an empty table is a separate, \
             tolerated case and does not reach this expect",
        )
    }

    /// Construct bound to `table`, rendering a specific channel of each
    /// entry. See the module doc's "Channel mixing is out of scope by
    /// design" note: this is a construction-time structural choice, not a
    /// runtime input — the caller (typically the exporter that resolved
    /// which channel, or an already-mixed-to-one-channel table) picks it
    /// once, up front.
    ///
    /// Returns [`NoEntriesForChannel`] if `table` has at least one entry but
    /// `channel` resolves for none of them — this is the whole-table failure
    /// mode (every entry out of range for `channel`), which is a different
    /// and louder case than a single bad entry among otherwise-fine ones
    /// (that case is tolerated silently by [`extract_entries`], as before).
    /// A table with zero entries to begin with is unaffected: it is a valid,
    /// pre-existing "not yet loaded" case, not this error.
    pub fn with_channel(table: Arc<CoeffTable>, channel: u32) -> Result<Self, NoEntriesForChannel> {
        let table_entries = table.entries.len();
        let entries = extract_entries(&table, channel);
        if table_entries > 0 && entries.is_empty() {
            return Err(NoEntriesForChannel {
                channel,
                table_entries,
            });
        }
        let m_max = entries
            .iter()
            .map(|e| e.partial_coeffs.len())
            .max()
            .unwrap_or(0);
        let j_max = entries
            .iter()
            .map(|e| e.noise_coeffs.len())
            .max()
            .unwrap_or(0);
        let noise_center_hz = noise_band_centers(j_max);
        Ok(PartialsNoise {
            entries,
            j_max,
            phase: alloc::vec![0.0; m_max],
            noise_state: alloc::vec![BiquadState::new(); j_max],
            noise_biquad: alloc::vec![(0.0, 0.0, 0.0, 0.0, 0.0); j_max],
            noise_center_hz,
            rng: Rng::new(PARTIALS_NOISE_DEFAULT_SEED),
            sample_rate: 44100.0,
        })
    }

    /// Find the two entry indices bracketing `f0` (equal if `f0` is at or
    /// beyond either end of the grid) and the crossfade weight toward the
    /// upper one, `t` in `[0, 1]`. Binary search since `entries` is sorted
    /// ascending by `f0_hz`.
    fn bracket(&self, f0: f32) -> (usize, usize, f32) {
        let entries = &self.entries;
        let n = entries.len();
        debug_assert!(n > 0, "bracket() requires at least one entry");
        if n == 1 {
            return (0, 0, 0.0);
        }
        let last = n - 1;
        if f0 <= entries[0].f0_hz {
            return (0, 0, 0.0);
        }
        if f0 >= entries[last].f0_hz {
            return (last, last, 0.0);
        }
        let mut lo = 0usize;
        let mut hi = last;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if entries[mid].f0_hz <= f0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (f_lo, f_hi) = (entries[lo].f0_hz, entries[hi].f0_hz);
        let t = if f_hi > f_lo {
            (f0 - f_lo) / (f_hi - f_lo)
        } else {
            0.0
        };
        (lo, hi, t)
    }
}

impl UGen for PartialsNoise {
    ugen_spec!(
        "PartialsNoise",
        category = Oscillator,
        inputs = ["freq", "gain"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let nyquist = self.sample_rate * 0.5;
        for j in 0..self.j_max {
            let raw_center = self.noise_center_hz[j];
            let center = raw_center.min(nyquist * 0.9).max(20.0);
            self.noise_biquad[j] = biquad_bpf_coeffs(center, NOISE_BAND_Q, self.sample_rate);
        }
    }

    fn reset(&mut self) {
        self.phase.iter_mut().for_each(|p| *p = 0.0);
        self.noise_state
            .iter_mut()
            .for_each(|s| *s = BiquadState::new());
        self.rng = Rng::new(PARTIALS_NOISE_DEFAULT_SEED);
    }

    fn reseed_noise(&mut self, seed: u32) {
        self.rng = Rng::new(seed);
        self.noise_state
            .iter_mut()
            .for_each(|s| *s = BiquadState::new());
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied();
        let gain_buf = inputs.get(1).copied();
        let sr = self.sample_rate.max(1.0);
        let inv_sr = 1.0 / sr;
        let nyquist = sr * 0.5;

        for ch in 0..output.num_channels() {
            // Local working copies: only channel 0's final state is
            // persisted back at the end of the block, matching every other
            // stateful UGen in this crate (Param, Lag, WaveTable, ...) —
            // multichannel expansion re-runs the same evolution per channel
            // rather than tracking independent state per channel.
            let mut phase = self.phase.clone();
            let mut noise_state = self.noise_state.clone();
            let mut rng = self.rng.clone();
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let f0_raw = read_input(freq_buf, ch, i, DEFAULT_FREQ_HZ);
                let f0 = if f0_raw.is_finite() {
                    f0_raw.max(0.0)
                } else {
                    0.0
                };
                let gain = read_input(gain_buf, ch, i, DEFAULT_GAIN);

                let mut sample = 0.0f32;

                if !self.entries.is_empty() {
                    let (lo, hi, t) = self.bracket(f0);
                    let e_lo = &self.entries[lo];
                    let e_hi = &self.entries[hi];
                    let stretch = lerp(e_lo.stretch, e_hi.stretch, t);
                    let mainlobe_gain = lerp(e_lo.mainlobe_gain, e_hi.mainlobe_gain, t);
                    let noise_gain = lerp(e_lo.noise_gain, e_hi.noise_gain, t);

                    for (m, ph) in phase.iter_mut().enumerate() {
                        let coeff = lerp_coeff(&e_lo.partial_coeffs, &e_hi.partial_coeffs, m, t);

                        let partial_number = (m + 1) as f32;
                        let freq_m = partial_number * f0 * stretch;

                        // Always advance phase, even when this partial is
                        // silent (coeff == 0): "phase continuity for free"
                        // (requirement 5) means a partial that reactivates
                        // later must not jump.
                        *ph += freq_m * inv_sr;
                        *ph -= ph.floor();

                        if coeff != 0.0 {
                            let fade = nyquist_fade(freq_m, nyquist);
                            let amp = coeff * mainlobe_gain * fade * gain;
                            sample += amp * (TAU * *ph).sin();
                        }
                    }

                    if self.j_max > 0 {
                        // One shared noise source through J parallel fixed
                        // bandpass filters — spectrally equivalent to J
                        // independent generators, without J separate RNGs.
                        let noise_in = rng.next_bipolar();
                        for (j, state) in noise_state.iter_mut().enumerate() {
                            let coeff = lerp_coeff(&e_lo.noise_coeffs, &e_hi.noise_coeffs, j, t);
                            let (b0, b1, b2, a1, a2) = self.noise_biquad[j];
                            // Always tick, same reasoning as phase above:
                            // a silent band's filter memory must stay live.
                            let filtered = state.tick(noise_in, b0, b1, b2, a1, a2);
                            if coeff != 0.0 {
                                sample += coeff * noise_gain * gain * filtered;
                            }
                        }
                    }
                }

                *out_sample = sample;
            }

            if ch == 0 {
                self.phase = phase;
                self.noise_state = noise_state;
                self.rng = rng;
            }
        }
    }
}

/// Register the `partialsNoise` table-bound kind (MOT-636).
///
/// Not part of [`crate::ugens::register_builtins`]: a table-bound kind has
/// no bare `fn() -> Box<dyn UGen>` factory to register there (there is no
/// table to construct with until a graph resolves one against a bank — see
/// [`crate::dsl::compiler::UGenRegistry::register_table_bound`]'s doc).
/// Call this alongside `register_builtins` from any host that resolves
/// table-bound kinds — today that means native / `ir`-feature tooling; see
/// `docs/coeff-table-bank-format.md`'s "What this mechanism does not (yet)
/// provide" for the wasm-bundle `ms_compile`-path reachability gap this
/// registration does not change (registering the kind name is not
/// `ir`-gated, only `IrSynthDef::compile_with_tables` — the step that
/// actually resolves a binding — is).
pub fn register(reg: &mut UGenRegistry) {
    let factory: TableUGenFactory = Arc::new(|table| Box::new(PartialsNoise::new(table)));
    reg.register_table_bound(
        "partialsNoise",
        factory,
        UGenCategory::Oscillator,
        &[
            InputSpec {
                name: "freq",
                rate: Rate::Audio,
            },
            InputSpec {
                name: "gain",
                rate: Rate::Audio,
            },
        ],
        &[OutputSpec {
            name: "out",
            rate: Rate::Audio,
        }],
    );
}
