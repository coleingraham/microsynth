//! `partialsBlend`: one partial bank that blends up to four channels of its
//! table by audio-rate gains, in place of several `partialsNoise` banks whose
//! outputs are summed.
//!
//! A def that wants a spectrum that changes within a note (an attack table
//! fading into a sustain table, a soft table crossfaded against a loud one by
//! velocity) can run one [`PartialsNoise`](super::partials::PartialsNoise)
//! per table on the same `freq` and sum them with time-varying gains. Every
//! bank starts its partials at phase 0 and advances them from the same
//! `freq`, so partial `m` of every bank is the same sinusoid and the sum is
//! a per-partial amplitude interpolation, exactly. It costs one sine per
//! partial per bank. `partialsBlend` computes that interpolation directly:
//! the tables become channels `0..K` of one table (the format's
//! `k_channels`, row-major per entry), and
//!
//! ```text
//! amp_m(t) = mainlobe_gain · fade_m · Σ_k gain_k(t) · c_k,m(f0)
//! ```
//!
//! with one phase accumulator and one sine per partial, whatever `K`. Each
//! `c_k,m(f0)` is interpolated across the pitch grid exactly as
//! `partialsNoise` interpolates its single channel (bracketing entries,
//! linear in f0, zero-padded, Nyquist fade).
//!
//! **Noise bands.** Separate banks have independent noise sources, so their
//! band outputs add in power. Here one shared source drives one set of band
//! filters, and band `j`'s gain is the root of the summed powers,
//! `noise_gain · sqrt(Σ_k (gain_k · c_k,j)²)`, which gives the same band
//! power as the separate banks (it is the same noise in distribution, not
//! sample for sample). The band placement, unit-variance source and per-band
//! power compensation are `partialsNoise`'s.
//!
//! **Equivalence.** Against `K` `partialsNoise` banks bound to the `K`
//! single-channel tables and summed, the partial half differs only by
//! floating-point summation order (relative error of order 1e-7); see
//! `tests/partials_blend.rs`.
//!
//! **Work skipped without changing the output.** A partial index whose
//! coefficient is zero in every channel of every entry can never sound, so
//! the bank is sized to the highest index that can (no phase is kept for the
//! rest); a noise band that is zero everywhere is never filtered. Both are
//! decided once, at construction.
//!
//! Ports: `freq` (Hz, default 220), `gain0` (default 1), `gain1`, `gain2`,
//! `gain3` (default 0). A table with fewer channels than four ignores the
//! extra gains; channels beyond four are not rendered. An entry lacking a
//! channel another entry has reads that channel as zero.

use crate::buffer::{AudioBuffer, read_input};
use crate::coeff_table::CoeffTable;
use crate::context::ProcessContext;
use crate::context::Rate;
use crate::dsl::compiler::{TableUGenFactory, UGenRegistry};
use crate::node::{InputSpec, OutputSpec, UGen, UGenCategory};
use crate::ugens::filters::{BiquadState, biquad_bpf_coeffs};
use crate::ugens::partials::{
    DEFAULT_FREQ_HZ, MAINLOBE_GAIN_KEY, NOISE_BAND_Q, NOISE_GAIN_KEY, PARTIALS_NOISE_DEFAULT_SEED,
    UNIT_VARIANCE_BIPOLAR_SCALE, biquad_noise_power_gain, lerp, metadata_scalar,
    noise_band_centers, noise_band_span, nyquist_fade,
};
use crate::ugens::rng::Rng;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::f32::consts::TAU;

/// Most channels one bank blends (one `gain` port each).
pub const MAX_BLEND_CHANNELS: usize = 4;

/// One pitch entry, every channel zero-padded to the bank's sizes.
struct BlendEntry {
    f0_hz: f32,
    stretch: f32,
    mainlobe_gain: f32,
    noise_gain: f32,
    /// `k * m_max + m`: channel `k`'s coefficient for partial `m`.
    partials: Vec<f32>,
    /// `k * j_max + j`: channel `k`'s coefficient for noise band `j`.
    noise: Vec<f32>,
}

/// The table-bound blending partial bank. See the module doc.
pub struct PartialsBlend {
    entries: Vec<BlendEntry>,
    /// Channels rendered (`<= MAX_BLEND_CHANNELS`).
    k: usize,
    /// Partials kept: one past the highest index nonzero anywhere.
    m_max: usize,
    /// Noise bands as the table defines them (band identity and placement).
    j_max: usize,
    /// Bands nonzero somewhere in the table; only these are filtered.
    live_bands: Vec<usize>,
    phase: Vec<f32>,
    noise_state: Vec<BiquadState>,
    noise_biquad: Vec<(f32, f32, f32, f32, f32)>,
    noise_center_hz: Vec<f32>,
    noise_power_compensation: Vec<f32>,
    rng: Rng,
    sample_rate: f32,
}

impl PartialsBlend {
    /// Construct bound to `table`, blending its channels `0..min(K, 4)`.
    /// A malformed entry (only reachable for a hand-built table that skipped
    /// [`CoeffTable::from_bytes`]'s validation) is skipped; an empty table
    /// gives a silent bank.
    pub fn new(table: Arc<CoeffTable>) -> Self {
        let raw: Vec<_> = table
            .entries
            .iter()
            .filter(|e| e.is_well_formed())
            .collect();
        let k = raw
            .iter()
            .map(|e| e.k_channels as usize)
            .max()
            .unwrap_or(0)
            .min(MAX_BLEND_CHANNELS);
        let j_max = raw.iter().map(|e| e.j_noise as usize).max().unwrap_or(0);
        // highest partial index that is nonzero in some channel of some entry
        let mut m_max = 0usize;
        for e in &raw {
            for ch in 0..k.min(e.k_channels as usize) {
                if let Some(row) = e.channel(ch as u32) {
                    let m = e.m_partials().min(row.len());
                    if let Some(last) = row[..m].iter().rposition(|&c| c != 0.0) {
                        m_max = m_max.max(last + 1);
                    }
                }
            }
        }
        let mut entries: Vec<BlendEntry> = raw
            .iter()
            .map(|e| {
                let m_e = e.m_partials();
                let mut partials = alloc::vec![0.0f32; k * m_max];
                let mut noise = alloc::vec![0.0f32; k * j_max];
                for ch in 0..k.min(e.k_channels as usize) {
                    if let Some(row) = e.channel(ch as u32) {
                        let (p, n) = row.split_at(m_e.min(row.len()));
                        for (m, &c) in p.iter().take(m_max).enumerate() {
                            partials[ch * m_max + m] = c;
                        }
                        for (j, &c) in n.iter().take(j_max).enumerate() {
                            noise[ch * j_max + j] = c;
                        }
                    }
                }
                BlendEntry {
                    f0_hz: e.f0_hz,
                    stretch: e.inharmonicity_stretch,
                    mainlobe_gain: metadata_scalar(&e.metadata, MAINLOBE_GAIN_KEY, 1.0),
                    noise_gain: metadata_scalar(&e.metadata, NOISE_GAIN_KEY, 1.0),
                    partials,
                    noise,
                }
            })
            .collect();
        entries.sort_by(|a, b| a.f0_hz.total_cmp(&b.f0_hz));
        let live_bands = (0..j_max)
            .filter(|&j| {
                entries
                    .iter()
                    .any(|e| (0..k).any(|ch| e.noise[ch * j_max + j] != 0.0))
            })
            .collect();
        let (band_min_hz, band_max_hz) = noise_band_span(&table);
        PartialsBlend {
            entries,
            k,
            m_max,
            j_max,
            live_bands,
            phase: alloc::vec![0.0; m_max],
            noise_state: alloc::vec![BiquadState::new(); j_max],
            noise_biquad: alloc::vec![(0.0, 0.0, 0.0, 0.0, 0.0); j_max],
            noise_center_hz: noise_band_centers(j_max, band_min_hz, band_max_hz),
            noise_power_compensation: alloc::vec![1.0; j_max],
            rng: Rng::new(PARTIALS_NOISE_DEFAULT_SEED),
            sample_rate: 44100.0,
        }
    }

    /// Channels this bank blends.
    pub fn channels(&self) -> usize {
        self.k
    }

    /// Partial oscillators this bank runs.
    pub fn partials(&self) -> usize {
        self.m_max
    }

    /// Bracketing entries and the weight toward the upper one (as
    /// `PartialsNoise::bracket`).
    fn bracket(&self, f0: f32) -> (usize, usize, f32) {
        let entries = &self.entries;
        let n = entries.len();
        if n <= 1 || f0 <= entries[0].f0_hz {
            return (0, 0, 0.0);
        }
        let last = n - 1;
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

impl UGen for PartialsBlend {
    ugen_spec!(
        "PartialsBlend",
        category = Oscillator,
        inputs = [],
        optional_inputs = ["freq", "gain0", "gain1", "gain2", "gain3"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let nyquist = self.sample_rate * 0.5;
        for j in 0..self.j_max {
            let center = self.noise_center_hz[j].min(nyquist * 0.9).max(20.0);
            let coeffs = biquad_bpf_coeffs(center, NOISE_BAND_Q, self.sample_rate);
            self.noise_biquad[j] = coeffs;
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
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied().flatten();
        let gain_bufs: [Option<&AudioBuffer>; MAX_BLEND_CHANNELS] =
            core::array::from_fn(|k| inputs.get(1 + k).copied().flatten());
        let sr = self.sample_rate.max(1.0);
        let inv_sr = 1.0 / sr;
        let nyquist = sr * 0.5;
        let (k_n, m_n, j_n) = (self.k, self.m_max, self.j_max);

        // Every channel starts from the block-start state (see PartialsNoise).
        let phase_snapshot = self.phase.clone();
        let noise_state_snapshot = self.noise_state.clone();
        let rng_snapshot = self.rng.clone();

        for ch in 0..output.num_channels() {
            let mut phase = phase_snapshot.clone();
            let mut noise_state = noise_state_snapshot.clone();
            let mut rng = rng_snapshot.clone();
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let f0_raw = read_input(freq_buf, ch, i, DEFAULT_FREQ_HZ);
                let f0 = if f0_raw.is_finite() {
                    f0_raw.max(0.0)
                } else {
                    0.0
                };
                let mut g = [0.0f32; MAX_BLEND_CHANNELS];
                for (k, gk) in g.iter_mut().enumerate().take(k_n) {
                    *gk = read_input(gain_bufs[k], ch, i, if k == 0 { 1.0 } else { 0.0 });
                }

                let mut sample = 0.0f32;
                if !self.entries.is_empty() {
                    let (lo, hi, t) = self.bracket(f0);
                    let e_lo = &self.entries[lo];
                    let e_hi = &self.entries[hi];
                    let stretch = lerp(e_lo.stretch, e_hi.stretch, t);
                    let mainlobe_gain = lerp(e_lo.mainlobe_gain, e_hi.mainlobe_gain, t);
                    let noise_gain = lerp(e_lo.noise_gain, e_hi.noise_gain, t);

                    for (m, ph) in phase.iter_mut().enumerate() {
                        let freq_m = (m + 1) as f32 * f0 * stretch;
                        // advance every phase, sounding or not (phase continuity)
                        *ph += freq_m * inv_sr;
                        *ph -= ph.floor();
                        let mut coeff = 0.0f32;
                        for (k, gk) in g.iter().enumerate().take(k_n) {
                            let idx = k * m_n + m;
                            coeff += gk * lerp(e_lo.partials[idx], e_hi.partials[idx], t);
                        }
                        if coeff != 0.0 {
                            let fade = nyquist_fade(freq_m, nyquist);
                            sample += coeff * mainlobe_gain * fade * (TAU * *ph).sin();
                        }
                    }

                    if !self.live_bands.is_empty() {
                        let noise_in = rng.next_bipolar() * UNIT_VARIANCE_BIPOLAR_SCALE;
                        for &j in &self.live_bands {
                            let (b0, b1, b2, a1, a2) = self.noise_biquad[j];
                            let filtered = noise_state[j].tick(noise_in, b0, b1, b2, a1, a2);
                            let mut power = 0.0f32;
                            for (k, gk) in g.iter().enumerate().take(k_n) {
                                let idx = k * j_n + j;
                                let a = gk * lerp(e_lo.noise[idx], e_hi.noise[idx], t);
                                power += a * a;
                            }
                            if power != 0.0 {
                                sample += power.sqrt()
                                    * noise_gain
                                    * self.noise_power_compensation[j]
                                    * filtered;
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

/// Register the `partialsBlend` table-bound kind.
pub fn register(reg: &mut UGenRegistry) {
    let factory: TableUGenFactory = Arc::new(|table| Box::new(PartialsBlend::new(table)));
    let port = |name| InputSpec {
        name,
        rate: Rate::Audio,
        required: false,
    };
    reg.register_table_bound(
        "partialsBlend",
        factory,
        UGenCategory::Oscillator,
        &[
            port("freq"),
            port("gain0"),
            port("gain1"),
            port("gain2"),
            port("gain3"),
        ],
        &[OutputSpec {
            name: "out",
            rate: Rate::Audio,
        }],
    );
}
