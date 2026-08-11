//! Integration tests for the partials + shaped-noise direct-synthesis ugen
//! (MOT-636, `microsynth::ugens::partials::PartialsNoise`): an
//! `M_p`-sinusoid oscillator bank plus `J` fixed shaped-noise bands, rendered
//! from a MOT-634 coefficient table bound at construction. These drive the
//! ugen directly through the low-level `Engine`/`AudioGraph` API (no DSL, no
//! IR) so the DSP behavior can be checked against closed-form expectations;
//! `table_bound_registration_resolves_and_renders` at the bottom covers the
//! `ir`-feature registration/resolution wiring end to end.

use microsynth::coeff_table::{CoeffTable, CoeffTableBank, PitchEntry};
use microsynth::ugens::partials::{MAINLOBE_GAIN_KEY, NOISE_GAIN_KEY, PartialsNoise};
use microsynth::ugens::*;
use microsynth::*;
use std::f32::consts::TAU;
use std::sync::Arc;

const SAMPLE_RATE: f32 = 44100.0;
const BLOCK_SIZE: usize = 128;

fn config() -> EngineConfig {
    EngineConfig {
        sample_rate: SAMPLE_RATE,
        block_size: BLOCK_SIZE,
    }
}

fn blocks_for(num_samples: usize) -> usize {
    num_samples.div_ceil(BLOCK_SIZE)
}

/// Render `ugen`'s mono output for `num_blocks` blocks with `freq`/`gain`
/// (input ports 0/1) held constant via `Const` nodes.
fn render_const(ugen: Box<dyn UGen>, freq: f32, gain: f32, num_blocks: usize) -> Vec<f32> {
    let mut engine = Engine::new(config());
    let freq_node = engine.graph_mut().add_node(Box::new(Const::new(freq)));
    let gain_node = engine.graph_mut().add_node(Box::new(Const::new(gain)));
    let osc = engine.graph_mut().add_node(ugen);
    engine.graph_mut().connect(freq_node, osc, 0);
    engine.graph_mut().connect(gain_node, osc, 1);
    engine.graph_mut().set_sink(osc);
    engine.prepare();
    engine.render_offline(num_blocks).remove(0)
}

/// Render `ugen`'s mono output, driving `freq` through a `Line` ramp from
/// `freq_start` to `freq_end` over `dur_secs` (holding at `freq_end`
/// thereafter), and `gain` from a `Const`.
fn render_glide(
    ugen: Box<dyn UGen>,
    freq_start: f32,
    freq_end: f32,
    dur_secs: f32,
    gain: f32,
    num_blocks: usize,
) -> Vec<f32> {
    let mut engine = Engine::new(config());
    let start = engine
        .graph_mut()
        .add_node(Box::new(Const::new(freq_start)));
    let end = engine.graph_mut().add_node(Box::new(Const::new(freq_end)));
    let dur = engine.graph_mut().add_node(Box::new(Const::new(dur_secs)));
    let line = engine.graph_mut().add_node(Box::new(Line::new()));
    engine.graph_mut().connect(start, line, 0);
    engine.graph_mut().connect(end, line, 1);
    engine.graph_mut().connect(dur, line, 2);

    let gain_node = engine.graph_mut().add_node(Box::new(Const::new(gain)));
    let osc = engine.graph_mut().add_node(ugen);
    engine.graph_mut().connect(line, osc, 0);
    engine.graph_mut().connect(gain_node, osc, 1);
    engine.graph_mut().set_sink(osc);
    engine.prepare();
    engine.render_offline(num_blocks).remove(0)
}

fn single_entry_table(
    f0_hz: f32,
    partial_coeffs: &[f32],
    noise_coeffs: &[f32],
    metadata: Vec<(String, f32)>,
) -> CoeffTable {
    let mut coefficients = partial_coeffs.to_vec();
    coefficients.extend_from_slice(noise_coeffs);
    CoeffTable {
        name: "synthetic".into(),
        entries: vec![PitchEntry {
            f0_hz,
            inharmonicity_stretch: 1.0,
            partial_freqs: (1..=partial_coeffs.len())
                .map(|m| m as f32 * f0_hz)
                .collect(),
            k_channels: 1,
            j_noise: noise_coeffs.len() as u32,
            coefficients,
            metadata,
        }],
    }
}

// ============================================================================
// 1. Analytic render test: closed-form partial amplitudes and noise-band
//    gains from a synthetic table with known coefficients.
// ============================================================================

#[test]
fn partial_amplitudes_match_closed_form_sinusoid_sum() {
    // Single entry at f0 exactly, so bracket() resolves with t == 0 and the
    // table's coefficients are used unmodified (no interpolation to
    // confound the check). mainlobe_gain defaults to 1.0 (no metadata), so
    // amplitude_m == coefficient_m * gain exactly.
    let coeffs = [0.5f32, 0.3, 0.1];
    let f0 = 220.0;
    let gain = 0.8;
    let table = single_entry_table(f0, &coeffs, &[], vec![]);
    let ugen: Box<dyn UGen> = Box::new(PartialsNoise::new(Arc::new(table)));

    let num_samples = 512;
    let out = render_const(ugen, f0, gain, blocks_for(num_samples));

    // Independent per-partial phase-accumulator recurrence — the
    // mathematical definition of "an M_p-sinusoid oscillator bank
    // integrating instantaneous frequency" the ugen implements, computed
    // here from scratch rather than by calling into it.
    let mut phases = vec![0.0f32; coeffs.len()];
    let mut expected = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let mut s = 0.0f32;
        for (m, ph) in phases.iter_mut().enumerate() {
            let freq_m = (m as f32 + 1.0) * f0;
            *ph += freq_m / SAMPLE_RATE;
            *ph -= ph.floor();
            s += coeffs[m] * gain * (TAU * *ph).sin();
        }
        expected.push(s);
    }

    for (i, (&actual, &exp)) in out
        .iter()
        .zip(expected.iter())
        .enumerate()
        .take(num_samples)
    {
        assert!(
            (actual - exp).abs() < 1e-4,
            "sample {i}: expected {exp}, got {actual}"
        );
    }
}

#[test]
fn mainlobe_gain_metadata_scales_partial_amplitude_linearly() {
    // Two otherwise-identical tables differing only in `mainlobe_gain`
    // metadata; every partial amplitude is coefficient * mainlobe_gain *
    // fade * gain, so scaling that one metadata scalar must scale every
    // output sample by the exact same factor -- a direct check that the
    // scalar in `docs/coeff-table-bank-format.md`'s open metadata slot is
    // *applied*, not silently ignored or only partially applied.
    let f0 = 220.0;
    let coeffs = [0.5f32];
    let table_a = single_entry_table(f0, &coeffs, &[], vec![]);
    let table_b = single_entry_table(f0, &coeffs, &[], vec![(MAINLOBE_GAIN_KEY.into(), 1.7)]);

    let num_samples = 512;
    let out_a = render_const(
        Box::new(PartialsNoise::new(Arc::new(table_a))),
        f0,
        1.0,
        blocks_for(num_samples),
    );
    let out_b = render_const(
        Box::new(PartialsNoise::new(Arc::new(table_b))),
        f0,
        1.0,
        blocks_for(num_samples),
    );

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate().take(num_samples) {
        assert!(
            (b - 1.7 * a).abs() < 1e-4,
            "sample {i}: expected {} (1.7x sample {a}), got {b}",
            1.7 * a
        );
    }
}

#[test]
fn noise_gain_metadata_scales_noise_band_output_linearly() {
    // Same idea as the mainlobe_gain test, over the noise-band path: no
    // partials (M_p == 0), so output is purely `coefficient * noise_gain *
    // gain * filtered_noise`. The shared noise source is deterministic
    // (fixed default seed, see PARTIALS_NOISE_DEFAULT_SEED) and identical
    // between two freshly constructed instances, so the two renders' noise
    // streams line up sample-for-sample and the *exact* per-sample ratio
    // is checkable, not just an RMS/statistical proxy.
    let f0 = 220.0;
    let noise_coeffs = [0.6f32, 0.4];
    let table_a = single_entry_table(f0, &[], &noise_coeffs, vec![]);
    let table_b = single_entry_table(f0, &[], &noise_coeffs, vec![(NOISE_GAIN_KEY.into(), 2.5)]);

    let num_samples = 1024;
    let out_a = render_const(
        Box::new(PartialsNoise::new(Arc::new(table_a))),
        f0,
        1.0,
        blocks_for(num_samples),
    );
    let out_b = render_const(
        Box::new(PartialsNoise::new(Arc::new(table_b))),
        f0,
        1.0,
        blocks_for(num_samples),
    );

    assert!(
        out_a.iter().any(|&s| s != 0.0),
        "sanity: noise-only render should not be silent"
    );
    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate().take(num_samples) {
        assert!(
            (b - 2.5 * a).abs() < 1e-4,
            "sample {i}: expected {} (2.5x sample {a}), got {b}",
            2.5 * a
        );
    }
}

// ============================================================================
// 2. f0-glide test across a grid boundary: partial birth/death ramped, no
//    click (no discontinuity above ramp-explained magnitude).
// ============================================================================

/// A two-entry table with different M_p at each grid point (birth/death) and
/// deliberately low frequencies / concentrated coefficients: a partial
/// birthing with a *hard* switch (the bug this test exists to catch) would
/// inject a same-sample amplitude jump on the order of its own coefficient
/// (0.5), which at these frequencies is many times larger than the ambient
/// per-sample change a smoothly glided, low-frequency oscillation produces
/// -- so a hard-switch regression and the smooth (correct) interpolation are
/// clearly distinguishable by the jump-vs-ambient ratio computed below.
fn birth_death_table() -> CoeffTable {
    CoeffTable {
        name: "birth-death".into(),
        entries: vec![
            PitchEntry {
                f0_hz: 55.0,
                inharmonicity_stretch: 1.0,
                partial_freqs: vec![55.0],
                k_channels: 1,
                j_noise: 0,
                coefficients: vec![1.0],
                metadata: vec![],
            },
            PitchEntry {
                f0_hz: 110.0,
                inharmonicity_stretch: 1.0,
                partial_freqs: vec![110.0, 220.0],
                k_channels: 1,
                j_noise: 0,
                coefficients: vec![0.5, 0.5],
                metadata: vec![],
            },
        ],
    }
}

#[test]
fn f0_glide_across_grid_boundary_has_no_click() {
    let table = birth_death_table();
    let ugen: Box<dyn UGen> = Box::new(PartialsNoise::new(Arc::new(table)));

    // 40 -> 140 Hz comfortably brackets both grid points (55, 110).
    let dur_secs = 0.5;
    let num_samples = (dur_secs * SAMPLE_RATE) as usize;
    let out = render_glide(ugen, 40.0, 140.0, dur_secs, 1.0, blocks_for(num_samples));

    let diffs: Vec<f32> = out.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let max_diff = diffs.iter().cloned().fold(0.0f32, f32::max);
    let mut sorted = diffs.clone();
    sorted.sort_by(f32::total_cmp);
    let median_diff = sorted[sorted.len() / 2];

    assert!(
        median_diff > 1e-6,
        "sanity: glide should produce meaningful ambient oscillation, median diff {median_diff}"
    );
    // A correct, continuously-interpolated birth/death spreads the
    // coefficient change over the whole inter-grid span (thousands of
    // samples here), so the glide's largest single-sample step should stay
    // within a small multiple of its own ambient (steady-oscillation) step
    // size. See the table's doc comment for why a hard-switch bug would
    // clear this bar by a wide margin instead.
    assert!(
        max_diff < median_diff * 10.0,
        "no discontinuity above ramp-explained magnitude: max diff {max_diff} vs median {median_diff} (ratio {})",
        max_diff / median_diff
    );
}

// ============================================================================
// 3. Energy-tracking test: output RMS tracks H through an interpolated
//    glide (simplex closure) -- exact per-sample linearity in `gain`, which
//    implies RMS proportionality as a special case.
// ============================================================================

#[test]
fn gain_scales_output_linearly_through_a_pitch_glide() {
    let table = birth_death_table();
    let dur_secs = 0.3;
    let num_samples = (dur_secs * SAMPLE_RATE) as usize;
    let num_blocks = blocks_for(num_samples);

    let out_1x = render_glide(
        Box::new(PartialsNoise::new(Arc::new(table.clone()))),
        40.0,
        140.0,
        dur_secs,
        1.0,
        num_blocks,
    );
    let out_2x = render_glide(
        Box::new(PartialsNoise::new(Arc::new(table))),
        40.0,
        140.0,
        dur_secs,
        2.0,
        num_blocks,
    );

    for (i, (&a, &b)) in out_1x
        .iter()
        .zip(out_2x.iter())
        .enumerate()
        .take(num_samples)
    {
        assert!(
            (b - 2.0 * a).abs() < 1e-4,
            "sample {i}: gain=2.0 output should be exactly 2x gain=1.0 output; expected {}, got {b}",
            2.0 * a
        );
    }
}

// ============================================================================
// 4. Simplex-closure preservation: no renormalization anywhere in the
//    render path.
// ============================================================================

#[test]
fn same_slot_interpolation_tracks_gain_exactly_with_no_renormalization() {
    // Both grid entries put full L1 mass (coefficient 1.0) on the *same*
    // partial slot (m = 0), so the interpolated coefficient is lerp(1, 1,
    // t) == 1.0 for every t -- no zero-padding, no birth/death, isolating
    // the "no renormalization" property. If the render path divided by
    // something it should not (channel count, partial count, ...), the
    // rendered peak would come out below `gain`; this table makes that
    // divergence directly visible against a known target.
    let table = CoeffTable {
        name: "same-slot".into(),
        entries: vec![
            PitchEntry {
                f0_hz: 55.0,
                inharmonicity_stretch: 1.0,
                partial_freqs: vec![55.0],
                k_channels: 1,
                j_noise: 0,
                coefficients: vec![1.0],
                metadata: vec![],
            },
            PitchEntry {
                f0_hz: 110.0,
                inharmonicity_stretch: 1.0,
                partial_freqs: vec![110.0],
                k_channels: 1,
                j_noise: 0,
                coefficients: vec![1.0],
                metadata: vec![],
            },
        ],
    };
    let gain = 0.73;
    let dur_secs = 0.3;
    let num_samples = (dur_secs * SAMPLE_RATE) as usize;
    let out = render_glide(
        Box::new(PartialsNoise::new(Arc::new(table))),
        40.0,
        140.0,
        dur_secs,
        gain,
        blocks_for(num_samples),
    );

    let peak = out
        .iter()
        .take(num_samples)
        .fold(0.0f32, |m, &s| m.max(s.abs()));
    assert!(
        (peak - gain).abs() < gain * 0.05,
        "peak amplitude should track gain ({gain}) with no renormalization, got {peak}"
    );
}

#[test]
fn birth_death_interpolation_never_exceeds_the_l1_energy_budget() {
    // Rigorous upper bound, true regardless of phase relationships -- for
    // the PARTIAL half only. `birth_death_table()` has `j_noise: 0` on both
    // entries, so this table exercises exactly that case: since both
    // endpoint coefficient rows are L1-unit and convex interpolation (even
    // zero-padded) preserves that, the interpolated row's L1 sum is exactly
    // 1.0 at every t. By the triangle inequality, |sum_m amp_m * sin(...)|
    // <= sum_m |amp_m| <= gain * 1.0 for every sample, with equality only at
    // a vanishingly unlikely phase alignment. A renormalization bug that
    // *adds* energy (double-counting, a sign error) is exactly what would
    // push a sample over this bound.
    //
    // This bound does NOT generalize to entries carrying noise mass -- a
    // resonant noise-band biquad's impulse response can have L1 norm > 1, so
    // `coeff_j * noise_gain * gain` alone is not a valid per-sample bound for
    // the noise half. See
    // `noise_band_l1_bound_requires_the_filter_impulse_response_norm` below
    // (MOT-649 F3) for the noise case and its correct, filter-dependent
    // bound, and `partials.rs`'s "Simplex closure" module doc for the full
    // restatement.
    let table = birth_death_table();
    let gain = 0.9;
    let dur_secs = 0.5;
    let num_samples = (dur_secs * SAMPLE_RATE) as usize;
    let out = render_glide(
        Box::new(PartialsNoise::new(Arc::new(table))),
        40.0,
        140.0,
        dur_secs,
        gain,
        blocks_for(num_samples),
    );

    let bound = gain * 1.0 + 1e-4;
    for (i, &s) in out.iter().enumerate().take(num_samples) {
        assert!(
            s.abs() <= bound,
            "sample {i}: {s} exceeds the partial-only L1 energy budget {bound} (gain={gain})"
        );
    }
}

// ============================================================================
// 4b. Noise-band L1 energy bound (MOT-649 F3): the partial-only bound above
// does not hold for entries carrying noise mass. This independently
// reimplements partials.rs's private mel-spaced band-center formula and
// filters.rs's private RBJ constant-peak-gain bandpass coefficients (both
// `pub(crate)`, unreachable from an integration test) to compute the
// noise-band biquad's impulse-response L1 norm -- the correct filter-
// dependent bound -- and pins the exact QA-reported violation of the naive
// bound as the closure test's red evidence. This mirrors an existing
// cross-boundary precedent: motif-soundmatch's `channel_export.py` already
// reimplements the same band-center formula for the same reason (F4). If
// `NOISE_BAND_MIN_HZ` / `NOISE_BAND_MAX_HZ` / `NOISE_BAND_Q` ever change in
// `partials.rs`, this copy must move with them.
// ============================================================================

const TEST_NOISE_BAND_MIN_HZ: f32 = 80.0;
const TEST_NOISE_BAND_MAX_HZ: f32 = 12_000.0;
const TEST_NOISE_BAND_Q: f32 = 1.0;

fn test_hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn test_mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Mirrors `partials.rs::noise_band_centers` plus the Nyquist clamp applied
/// in `PartialsNoise::init`.
fn test_clamped_band_center_hz(j_max: usize, band: usize, sample_rate: f32) -> f32 {
    let mel_lo = test_hz_to_mel(TEST_NOISE_BAND_MIN_HZ);
    let mel_hi = test_hz_to_mel(TEST_NOISE_BAND_MAX_HZ);
    let frac = (band as f32 + 0.5) / j_max as f32;
    let raw = test_mel_to_hz(mel_lo + frac * (mel_hi - mel_lo));
    let nyquist = sample_rate * 0.5;
    raw.min(nyquist * 0.9).max(20.0)
}

/// Mirrors `filters.rs::biquad_bpf_coeffs` (constant-peak-gain RBJ
/// bandpass), independently re-derived rather than called (private).
fn test_biquad_bpf_coeffs(freq: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
    let w0 = TAU * freq / sample_rate;
    let (sin_w0, cos_w0) = (w0.sin(), w0.cos());
    let alpha = sin_w0 / (2.0 * q);
    let (b0, b1, b2) = (alpha, 0.0, -alpha);
    let (a0, a1, a2) = (1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha);
    (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
}

/// `sum_n |h[n]|` for `n` in `[0, n_samples)`, via the same direct-form-II-
/// transposed recurrence as `BiquadState::tick` (independently re-derived --
/// see the section comment above). Every band this ugen uses is stable
/// (pole radius < 1), so the impulse response decays geometrically and
/// `n_samples` far beyond that decay makes the truncated sum a tight,
/// effectively-exact estimate of the true (infinite) L1 norm.
fn impulse_response_l1_norm(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32, n_samples: usize) -> f32 {
    let (mut z1, mut z2) = (0.0f32, 0.0f32);
    let mut total = 0.0f32;
    for n in 0..n_samples {
        let x = if n == 0 { 1.0 } else { 0.0 };
        let y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        total += y.abs();
    }
    total
}

#[test]
fn noise_band_l1_bound_requires_the_filter_impulse_response_norm() {
    // Exact reproduction of the QA report's F3 measurement (MOT-649):
    // sr=16000, J=4, all L1 mass on band index 2, no metadata (both bridge
    // scalars default 1.0), gain=0.9. QA measured a real rendered peak of
    // 0.9028 -- which is the naive "L1 energy budget" bound (gain * 1.0 =
    // 0.9) being violated. Reproduced here as the closure test's red
    // evidence, then checked against the correct, filter-dependent bound.
    let sr = 16_000.0f32;
    let j_max = 4usize;
    let band = 2usize;
    let f0 = 220.0; // Irrelevant: this entry carries no partial mass.
    let gain = 0.9;
    let mut noise_coeffs = vec![0.0f32; j_max];
    noise_coeffs[band] = 1.0;
    let table = single_entry_table(f0, &[], &noise_coeffs, vec![]);

    let num_samples = 200_000usize;
    let mut engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: BLOCK_SIZE,
    });
    let freq_node = engine.graph_mut().add_node(Box::new(Const::new(f0)));
    let gain_node = engine.graph_mut().add_node(Box::new(Const::new(gain)));
    let osc = engine
        .graph_mut()
        .add_node(Box::new(PartialsNoise::new(Arc::new(table))));
    engine.graph_mut().connect(freq_node, osc, 0);
    engine.graph_mut().connect(gain_node, osc, 1);
    engine.graph_mut().set_sink(osc);
    engine.prepare();
    let out = engine.render_offline(blocks_for(num_samples)).remove(0);
    let peak = out
        .iter()
        .take(num_samples)
        .fold(0.0f32, |m, &s| m.max(s.abs()));

    // Red evidence: the naive partial-only bound does not hold for the
    // noise half.
    let naive_bound = gain * 1.0;
    assert!(
        peak > naive_bound,
        "expected the naive partial-only L1 bound ({naive_bound}) to be \
         violated by the noise half -- this is the red evidence for F3; got \
         peak {peak}, which no longer demonstrates the defect"
    );
    assert!(
        (peak - 0.9028).abs() < 5e-4,
        "expected to reproduce QA's exact measured peak (0.9028); got {peak} \
         instead -- a mismatch means the RNG/filter recurrence or this \
         table's construction has drifted from the reported repro"
    );

    // Correct bound: coeff_j * noise_gain * gain * ||h_j||_1, not
    // coeff_j * noise_gain * gain. coeff_j = noise_gain = 1.0 here (all mass
    // on one band, no metadata), so the bound reduces to gain * ||h||_1.
    let (b0, b1, b2, a1, a2) = test_biquad_bpf_coeffs(
        test_clamped_band_center_hz(j_max, band, sr),
        TEST_NOISE_BAND_Q,
        sr,
    );
    let h_l1_norm = impulse_response_l1_norm(b0, b1, b2, a1, a2, 500_000);
    let correct_bound = gain * h_l1_norm + 1e-4;
    assert!(
        peak <= correct_bound,
        "sample peak {peak} exceeds the filter-dependent bound {correct_bound} \
         (gain={gain} * ||h||_1={h_l1_norm}) -- this bound should hold even \
         though the naive one does not"
    );
}

// ============================================================================
// 5. Registration + IR resolution wiring (mirrors tests/coeff_table_bank.rs).
// ============================================================================

#[test]
#[cfg(feature = "ir")]
fn table_bound_registration_resolves_and_renders() {
    use microsynth::ir::{IrNode, IrSynthDef, IrTableBinding, SynthDefClass};

    let mut reg = microsynth::dsl::compiler::UGenRegistry::new();
    microsynth::register_builtins(&mut reg);
    microsynth::register_table_bound_builtins(&mut reg);

    let mut bank = CoeffTableBank::new();
    let table = single_entry_table(220.0, &[0.5, 0.3, 0.1], &[], vec![]);
    let id = bank.register(table);

    // freq/gain fed by inline-const Param nodes (params 0/1), output is the
    // partialsNoise node itself.
    let ir = IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "partials_probe".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![
            IrNode::Param {
                name: "freq".into(),
                default: 220.0,
            },
            IrNode::Param {
                name: "gain".into(),
                default: 1.0,
            },
            IrNode::UGen {
                kind: "partialsNoise".into(),
                consts: vec![],
            },
        ],
        edges: vec![
            microsynth::ir::IrEdge {
                from: 0,
                to: 2,
                to_input: 0,
            },
            microsynth::ir::IrEdge {
                from: 1,
                to: 2,
                to_input: 1,
            },
        ],
        params: vec![],
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding {
            node: 2,
            table_id: id.0,
        }],
        output_node: 2,
    };
    ir.validate(&reg).expect("valid IR");
    let def = ir
        .compile_with_tables(&reg, &bank)
        .expect("resolves against bank");

    let mut engine = Engine::new(config());
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let output = engine.render().expect("engine should produce output");
    let samples = output.channel(0).samples();
    assert!(
        samples.iter().any(|&s| s != 0.0),
        "resolved partialsNoise node should render non-silent output"
    );
}


