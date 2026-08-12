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
//! band's contribution is `coeff_j * noise_gain * gain * compensation_j *
//! filtered_j`, where `filtered_j` is a resonant biquad's *output*, not its
//! input, and `compensation_j` is the MOT-641 per-band power-gain scalar (see
//! [`biquad_noise_power_gain`]/`noise_power_compensation`'s doc). A stable
//! IIR filter's worst-case gain against a bounded-magnitude input (the
//! unit-variance-rescaled noise source is uniform on `[-sqrt(3), sqrt(3)]` —
//! see [`UNIT_VARIANCE_BIPOLAR_SCALE`]'s doc) is its impulse response's L1
//! norm `‖h_j‖₁` (`sum_n |h_j[n]|`), not 1. For [`NOISE_BAND_Q`]'s
//! constant-peak-gain resonance, `‖h_j‖₁` measures **> 1**, and
//! `compensation_j` (a function of the impulse response's L2 norm, not its
//! L1 norm) does not cancel that — the two norms differ for a resonant
//! filter's impulse response, so the naive `gain * 1.0` bound is still
//! violated after MOT-641's level fix, just by a different, compensation-
//! scaled amount. See `tests/partials_noise.rs`'s
//! `noise_band_l1_bound_requires_the_filter_impulse_response_norm`, which
//! measures the actual rendered peak and pins the correct bound: `coeff_j *
//! noise_gain * gain * compensation_j * sqrt(3) * ‖h_j‖₁`, not `coeff_j *
//! noise_gain * gain`.
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
//!
//! `noise_gain`'s own derivation (`motif-soundmatch`'s
//! `channel_export.py::noise_gain`) is explicit that it converts bump mass to
//! generator amplitude "GIVEN a unit-variance noise source and an ideal
//! (unity in-band gain) bandpass" — conditions this ugen did not actually
//! meet before MOT-641 (`Rng::next_bipolar()` is not unit-variance, and
//! [`NOISE_BAND_Q`]'s constant-peak-gain resonance is not unity-in-band-gain
//! against a wideband source). [`UNIT_VARIANCE_BIPOLAR_SCALE`] and
//! `noise_power_compensation` (see their docs) are what make those
//! conditions actually hold at render time, rather than leaving them as an
//! assumption the exporter's scalar alone cannot satisfy.
//!
//! The noise-band *placement* half of the bridge (MOT-641) works the same
//! way: [`NOISE_BAND_MIN_HZ_KEY`]/[`NOISE_BAND_MAX_HZ_KEY`] metadata (falling
//! back to [`NOISE_BAND_MIN_HZ`]/[`NOISE_BAND_MAX_HZ`] when absent) lets the
//! table state the exact span its analysis-side noise basis `N` was built
//! against, and [`noise_band_centers`]'s formula matches `channels.py`'s own
//! bump-center formula exactly — so a table that supplies its true span
//! renders noise bands at (up to `f32` rounding) the same center frequencies
//! the analysis side fit against, not an approximation of them.

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

/// Metadata keys (MOT-641) for the noise-band span a table's entries were fit
/// against, in Hz. Either absent: falls back to [`NOISE_BAND_MIN_HZ`] /
/// [`NOISE_BAND_MAX_HZ`]. Read as one table-wide value (not per-entry) by
/// [`noise_band_span`] — see that function's doc — and
/// `docs/coeff-table-bank-format.md`'s Metadata section for the wire
/// contract.
pub const NOISE_BAND_MIN_HZ_KEY: &str = "noise_band_min_hz";
pub const NOISE_BAND_MAX_HZ_KEY: &str = "noise_band_max_hz";

/// Default seed for the shared shaped-noise source. Mirrors `WhiteNoise`'s /
/// `PinkNoise`'s fixed-default-seed convention (`ugens/noise.rs`); a voice
/// spawn is expected to call [`UGen::reseed_noise`] for keyed, per-voice
/// determinism (see `noise.rs`'s module doc on the seeding contract).
const PARTIALS_NOISE_DEFAULT_SEED: u32 = 0x9A27_1E55;

/// Default `freq`/`gain` values for an unconnected port.
const DEFAULT_FREQ_HZ: f32 = 220.0;
const DEFAULT_GAIN: f32 = 1.0;

/// The fallback smooth-noise basis frequency span (MOT-641): used only when a
/// table's entries carry neither [`NOISE_BAND_MIN_HZ_KEY`] nor
/// [`NOISE_BAND_MAX_HZ_KEY`] metadata (`docs/coeff-table-bank-format.md`'s
/// documented fallback for a table without a band definition). A real fitted
/// table instead carries the exact span its analysis-side noise basis `N` was
/// built against (typically `[0, Nyquist]` at the analysis sample rate — see
/// `motif-soundmatch`'s `channels.py::noise_basis`). Band identity (which
/// center frequency "band index j" means) must not depend on which table, or
/// which of a table's entries, is currently active, or a glide that changes
/// `j_noise` between entries would reassign a band's frequency mid-note — so
/// [`noise_band_span`] reads one table-wide value, and a well-formed table's
/// exporter writes the same span into every entry. The top end is further
/// clamped to the render sample rate's Nyquist in [`PartialsNoise::init`],
/// not here (these two constants are computed once, before sample rate is
/// known).
const NOISE_BAND_MIN_HZ: f32 = 80.0;
const NOISE_BAND_MAX_HZ: f32 = 12_000.0;

/// `Rng::next_bipolar()` is uniform on `[-1, 1]`, variance `1/3` (std
/// `≈0.5774`) — not the unit-variance source the exporter's `noise_gain`
/// derivation (`motif-soundmatch`'s `channel_export.py::noise_gain`) assumes
/// ("for zero-mean white noise of unit **variance**..."). `sqrt(3)` rescales
/// a `[-1, 1]`-uniform sample to variance exactly 1 (`Var(k*U) = k^2 *
/// Var(U)`, so `k = sqrt(1 / Var(U)) = sqrt(3)`), without touching
/// `next_bipolar()`'s own documented `[-1, 1]` contract that `WhiteNoise` /
/// `PinkNoise` and every other consumer still rely on (MOT-641).
const UNIT_VARIANCE_BIPOLAR_SCALE: f32 = 1.732_050_8; // sqrt(3)

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

/// Mel-spaced center frequencies for `j_max` noise bands over `[min_hz,
/// max_hz]` (raw — not yet clamped to a render sample rate's Nyquist; see
/// [`PartialsNoise::init`]).
///
/// Band `j`'s center is the `(j+1)`-th interior point of `j_max + 2`
/// equally-mel-spaced edges spanning `[min_hz, max_hz]`, i.e. its mel-space
/// fraction is `frac_j = (j + 1) / (j_max + 1)`, not the naive per-band
/// midpoint `frac_j = (j + 0.5) / j_max` this function used before MOT-641.
/// This is not a stylistic choice: it is the *exact* formula
/// `motif-soundmatch`'s `channels.py::noise_basis` uses for its own bump
/// centers (edges via `linspace` over `j_max + 2` points, interior points
/// kept) — matching it here, not just the span, is what makes the analysis
/// and synthesis band centers coincide exactly (up to `f32` rounding) when a
/// table's [`NOISE_BAND_MIN_HZ_KEY`]/[`NOISE_BAND_MAX_HZ_KEY`] metadata
/// carries the same `[min_hz, max_hz]` the analysis side built `N` against.
fn noise_band_centers(j_max: usize, min_hz: f32, max_hz: f32) -> Vec<f32> {
    if j_max == 0 {
        return Vec::new();
    }
    let mel_lo = hz_to_mel(min_hz);
    let mel_hi = hz_to_mel(max_hz);
    (0..j_max)
        .map(|j| {
            let frac = (j as f32 + 1.0) / (j_max as f32 + 1.0);
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

/// The noise-band span (`[min_hz, max_hz]`) [`noise_band_centers`] should use
/// for `table`: one table-wide value, read from `table`'s first entry's
/// metadata (falling back to [`NOISE_BAND_MIN_HZ`]/[`NOISE_BAND_MAX_HZ`] when
/// either key is absent there). Deliberately reads the *raw* table entries
/// (not [`EntryData`], which is already channel-resolved) — the span is a
/// property of the table's analysis configuration, not of any one channel —
/// and deliberately reads only entry 0 rather than per-entry, per
/// [`NOISE_BAND_MIN_HZ`]'s doc: band identity must be one fixed thing across
/// a table, so a well-formed table's exporter writes the same span into every
/// entry and any one of them is representative. A table with zero entries
/// gets the hardcoded fallback (nothing to read).
fn noise_band_span(table: &CoeffTable) -> (f32, f32) {
    match table.entries.first() {
        Some(e) => (
            metadata_scalar(&e.metadata, NOISE_BAND_MIN_HZ_KEY, NOISE_BAND_MIN_HZ),
            metadata_scalar(&e.metadata, NOISE_BAND_MAX_HZ_KEY, NOISE_BAND_MAX_HZ),
        ),
        None => (NOISE_BAND_MIN_HZ, NOISE_BAND_MAX_HZ),
    }
}

/// The white-noise power gain `Σ h[n]²` of a stable biquad's impulse
/// response `h` — the output variance a unit-variance white-noise input
/// produces (Parseval: `Var(y) = Var(x) · Σ h[n]²` for i.i.d. `x`). MOT-641's
/// per-band [`PartialsNoise::noise_power_compensation`] is `1 /
/// sqrt(biquad_noise_power_gain(..))`, so that after compensation a
/// unit-variance input produces unit-variance output — the "ideal (unity
/// in-band gain) bandpass" `motif-soundmatch`'s `noise_gain` derivation
/// assumes, which [`NOISE_BAND_Q`]'s actual constant-peak-gain resonance does
/// not provide on its own (a resonant filter passes only its own noise
/// bandwidth, not the source's full power).
///
/// Computed by direct simulation (feed one unit impulse, sum squared
/// output) rather than a closed form: exact in the limit, and simple enough
/// to trust over a hand-derived Lyapunov solve. `n_samples` is sized from the
/// filter's own ring/decay time constant (`tau ≈ Q · sample_rate / (π ·
/// freq)` samples for a 2-pole resonator) so the truncated sum captures
/// `1 - exp(-25) ≈ 1 - 1.4e-11` of the true (infinite) energy — comfortably
/// below `f32` rounding noise — for any `freq`/`Q`/`sample_rate`, not just
/// the ones this ugen happens to use today.
fn biquad_noise_power_gain(
    freq: f32,
    q: f32,
    sample_rate: f32,
    coeffs: (f32, f32, f32, f32, f32),
) -> f32 {
    let (b0, b1, b2, a1, a2) = coeffs;
    let f = freq.max(1.0);
    let tau_samples = (q * sample_rate / (core::f32::consts::PI * f)).max(1.0);
    let n = ((25.0 * tau_samples) as usize).clamp(64, 2_000_000);
    let mut state = BiquadState::new();
    let mut sum_sq = 0.0f32;
    let mut x = 1.0f32;
    for _ in 0..n {
        let y = state.tick(x, b0, b1, b2, a1, a2);
        x = 0.0;
        sum_sq += y * y;
    }
    sum_sq
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
    /// Per-band `1 / sqrt(biquad_noise_power_gain(..))` (MOT-641), fixed
    /// alongside `noise_biquad` once sample rate is known in
    /// [`init`](UGen::init). Restores each band's filtered output to
    /// unit-variance-in/unit-variance-out — the "ideal (unity in-band gain)
    /// bandpass" `noise_gain`'s derivation assumes — since [`NOISE_BAND_Q`]'s
    /// actual constant-peak-gain resonance passes only its own noise
    /// bandwidth, not the source's full power. See
    /// [`biquad_noise_power_gain`]'s doc.
    noise_power_compensation: Vec<f32>,
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
        let (band_min_hz, band_max_hz) = noise_band_span(&table);
        let noise_center_hz = noise_band_centers(j_max, band_min_hz, band_max_hz);
        Ok(PartialsNoise {
            entries,
            j_max,
            phase: alloc::vec![0.0; m_max],
            noise_state: alloc::vec![BiquadState::new(); j_max],
            noise_biquad: alloc::vec![(0.0, 0.0, 0.0, 0.0, 0.0); j_max],
            noise_center_hz,
            noise_power_compensation: alloc::vec![1.0; j_max],
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
            let coeffs = biquad_bpf_coeffs(center, NOISE_BAND_Q, self.sample_rate);
            self.noise_biquad[j] = coeffs;
            // Guard against a degenerate (near-zero) power gain -- unreachable
            // for any center this method actually clamps to (`[20, nyquist *
            // 0.9]` Hz at Q=1 never approaches the all-pole singularity a
            // near-0 Hz or near-Nyquist center would produce), but a bound
            // saves this from ever dividing by ~0 rather than relying on that
            // reachability argument alone.
            let power_gain =
                biquad_noise_power_gain(center, NOISE_BAND_Q, self.sample_rate, coeffs).max(1e-12);
            self.noise_power_compensation[j] = 1.0 / power_gain.sqrt();
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
                        // Rescaled to unit variance (MOT-641): see
                        // `UNIT_VARIANCE_BIPOLAR_SCALE`'s doc.
                        let noise_in = rng.next_bipolar() * UNIT_VARIANCE_BIPOLAR_SCALE;
                        for (j, state) in noise_state.iter_mut().enumerate() {
                            let coeff = lerp_coeff(&e_lo.noise_coeffs, &e_hi.noise_coeffs, j, t);
                            let (b0, b1, b2, a1, a2) = self.noise_biquad[j];
                            // Always tick, same reasoning as phase above:
                            // a silent band's filter memory must stay live.
                            let filtered = state.tick(noise_in, b0, b1, b2, a1, a2);
                            if coeff != 0.0 {
                                // Per-band power-gain compensation (MOT-641):
                                // see `noise_power_compensation`'s doc.
                                let compensation = self.noise_power_compensation[j];
                                sample += coeff * noise_gain * gain * compensation * filtered;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A single band's mel-space fraction should be exactly `(j+1)/(j_max+1)`
    /// (MOT-641's edges-based formula, matching `channels.py::noise_basis`),
    /// not the pre-MOT-641 `(j+0.5)/j_max` midpoint formula.
    #[test]
    fn noise_band_centers_uses_the_edges_based_mel_fraction() {
        // j_max = 1, span [0, 8000] Hz: frac = (0+1)/(1+1) = 0.5 exactly the
        // mel midpoint of the span, closed-form checkable without floating
        // the whole linspace machinery.
        let centers = noise_band_centers(1, 0.0, 8000.0);
        assert_eq!(centers.len(), 1);
        let expected = mel_to_hz(0.5 * hz_to_mel(8000.0));
        assert!(
            (centers[0] - expected).abs() < 1e-3,
            "expected {expected}, got {}",
            centers[0]
        );

        // j_max = 3: fracs are 1/4, 2/4, 3/4 -- the middle band sits at
        // exactly the mel midpoint regardless of j_max, since (j_max/2 +
        // 0.5)/(j_max+1) == 0.5 only when j_max is odd and j is its middle
        // index; check that directly for j_max=3, j=1: frac = 2/4 = 0.5.
        let centers3 = noise_band_centers(3, 0.0, 8000.0);
        assert_eq!(centers3.len(), 3);
        assert!(
            (centers3[1] - expected).abs() < 1e-3,
            "middle band of an odd j_max should sit at the mel midpoint: \
             expected {expected}, got {}",
            centers3[1]
        );
    }

    #[test]
    fn noise_band_centers_empty_for_zero_bands() {
        assert!(noise_band_centers(0, 0.0, 8000.0).is_empty());
    }

    #[test]
    fn noise_band_span_falls_back_to_hardcoded_defaults_when_metadata_absent() {
        let table = CoeffTable {
            name: "no-span-metadata".into(),
            entries: alloc::vec![crate::coeff_table::PitchEntry {
                f0_hz: 220.0,
                inharmonicity_stretch: 1.0,
                partial_freqs: alloc::vec![],
                k_channels: 1,
                j_noise: 0,
                coefficients: alloc::vec![],
                metadata: alloc::vec![],
            }],
        };
        assert_eq!(
            noise_band_span(&table),
            (NOISE_BAND_MIN_HZ, NOISE_BAND_MAX_HZ)
        );
    }

    #[test]
    fn noise_band_span_falls_back_for_a_zero_entry_table() {
        let table = CoeffTable {
            name: "empty".into(),
            entries: alloc::vec![],
        };
        assert_eq!(
            noise_band_span(&table),
            (NOISE_BAND_MIN_HZ, NOISE_BAND_MAX_HZ)
        );
    }

    #[test]
    fn noise_band_span_reads_the_first_entrys_metadata() {
        let table = CoeffTable {
            name: "with-span-metadata".into(),
            entries: alloc::vec![crate::coeff_table::PitchEntry {
                f0_hz: 220.0,
                inharmonicity_stretch: 1.0,
                partial_freqs: alloc::vec![],
                k_channels: 1,
                j_noise: 0,
                coefficients: alloc::vec![],
                metadata: alloc::vec![
                    (NOISE_BAND_MIN_HZ_KEY.into(), 0.0),
                    (NOISE_BAND_MAX_HZ_KEY.into(), 8000.0),
                ],
            }],
        };
        assert_eq!(noise_band_span(&table), (0.0, 8000.0));
    }

    /// `biquad_noise_power_gain`'s simulation should converge to the closed-
    /// form Parseval identity `Σ h[n]² == Var(y) / Var(x)` for a
    /// unit-variance-normalized DC-passthrough sanity case: a trivial
    /// all-pass-at-DC-only biquad (`b0=1`, everything else 0) has `h[0]=1`,
    /// `h[n]=0` otherwise, so its power gain is exactly 1.
    #[test]
    fn biquad_noise_power_gain_identity_filter_is_unity() {
        let gain = biquad_noise_power_gain(1000.0, 1.0, 44_100.0, (1.0, 0.0, 0.0, 0.0, 0.0));
        assert!(
            (gain - 1.0).abs() < 1e-6,
            "identity filter's impulse response is a single 1.0 sample, so \
             its power gain (sum of squares) must be exactly 1.0, got {gain}"
        );
    }
}
