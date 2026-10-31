//! `fmOp`: one phase-modulation operator with the OPL waveform set.
//!
//! A 2-op stack of `fmOp`s must reproduce `fmOsc`, the waveform table must
//! have the OPL shapes, and feedback must stay bounded at every setting.

use microsynth::dsl;
use microsynth::*;

mod common;
use common::builtin_registry;

const SR: f32 = 44100.0;
const BLOCK: usize = 64;

/// Render `secs` of a single-def source as mono samples.
fn render(src: &str, secs: f32) -> Vec<f32> {
    let registry = builtin_registry();
    let def = dsl::compile(src, &registry)
        .unwrap_or_else(|e| panic!("compile failed: {e:?}"))
        .remove(0);
    let mut engine = Engine::new(EngineConfig {
        sample_rate: SR,
        block_size: BLOCK,
    });
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let blocks = (secs * SR / BLOCK as f32).ceil() as usize;
    let mut out = Vec::with_capacity(blocks * BLOCK);
    for _ in 0..blocks {
        let buf = engine.render().expect("a sink is set");
        out.extend_from_slice(buf.channel(0).samples());
    }
    out
}

/// Magnitude of the Hann-windowed DFT at `freq` Hz, normalised so a unit
/// sine at `freq` reads ~1.
fn bin(x: &[f32], freq: f32) -> f32 {
    let len = x.len() as f64;
    let (mut re, mut im, mut wsum) = (0.0f64, 0.0f64, 0.0f64);
    for (n, &s) in x.iter().enumerate() {
        let win = 0.5 - 0.5 * (core::f64::consts::TAU * n as f64 / len).cos();
        let w = core::f64::consts::TAU * freq as f64 * n as f64 / SR as f64;
        re += win * s as f64 * w.cos();
        im -= win * s as f64 * w.sin();
        wsum += win;
    }
    (2.0 * (re * re + im * im).sqrt() / wsum) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn two_op_stack_matches_fm_osc() {
    // (carrier Hz, ratio, index): harmonic, sub-octave and inharmonic.
    for &(f, ratio, index) in &[
        (220.0f32, 1.0f32, 2.0f32),
        (110.0, 3.0, 4.5),
        (330.0, 1.4, 7.0),
    ] {
        let reference = render(
            &format!("synthdef r = fmOsc {f:.4} {ratio:.4} {index:.4} 0.0"),
            1.0,
        );
        let stack = render(
            &format!(
                "synthdef s =\n  let m = fmOp ({f:.4} * {ratio:.4}) 0.0 {index:.4} 0.0 0.0\n  \
                 fmOp {f:.4} m 1.0 0.0 0.0"
            ),
            1.0,
        );
        assert_eq!(reference.len(), stack.len());
        let d = max_abs_diff(&reference, &stack);
        assert!(
            d < 1e-3,
            "f={f} ratio={ratio} index={index}: max |fmOsc - stack| = {d}"
        );
    }
}

#[test]
fn modulator_level_is_the_modulation_index() {
    // Chowning: with c:m = 1:1 and index I, the carrier partial has
    // amplitude |J0(I) - J2(I)|... rather than lean on that sum, check the
    // pure Bessel case c:m = 1:4 where sidebands do not collide below k=1:
    // carrier at f has amplitude |J0(I)|, first sideband at f+4f has |J1(I)|.
    let f = 200.0;
    let index = 1.5;
    let x = render(
        &format!(
            "synthdef s =\n  let m = fmOp ({f:.1} * 4.0) 0.0 {index:.2} 0.0 0.0\n  \
             fmOp {f:.1} m 1.0 0.0 0.0"
        ),
        1.0,
    );
    // J0(1.5) = 0.5118, J1(1.5) = 0.5579.
    let c = bin(&x, f);
    let s1 = bin(&x, 5.0 * f);
    assert!((c - 0.5118).abs() < 0.02, "carrier {c}");
    assert!((s1 - 0.5579).abs() < 0.02, "upper sideband {s1}");
}

#[test]
fn waveform_table_has_the_opl_shapes() {
    use microsynth::ugens::fm::{fm_wave as w, fm_wave_shape as shape};
    let close = |a: f32, b: f32| (a - b).abs() < 1e-4;
    // 0 sine.
    assert!(close(shape(w::SINE, 0.25), 1.0) && close(shape(w::SINE, 0.75), -1.0));
    // 1 half-sine: negative half silenced.
    assert!(close(shape(w::HALF_SINE, 0.25), 1.0) && close(shape(w::HALF_SINE, 0.75), 0.0));
    // 2 abs-sine: negative half rectified.
    assert!(close(shape(w::ABS_SINE, 0.75), 1.0));
    // 3 quarter-sine: rising quarter of each half, then silence.
    assert!(shape(w::QUARTER_SINE, 0.2) > 0.9 && close(shape(w::QUARTER_SINE, 0.3), 0.0));
    assert!(shape(w::QUARTER_SINE, 0.7) > 0.9 && close(shape(w::QUARTER_SINE, 0.8), 0.0));
    // 4 alternating: double-rate full sine in the first half.
    assert!(close(shape(w::ALTERNATING_SINE, 0.125), 1.0));
    assert!(close(shape(w::ALTERNATING_SINE, 0.375), -1.0));
    assert!(close(shape(w::ALTERNATING_SINE, 0.75), 0.0));
    // 5 camel: rectified double-rate sine in the first half.
    assert!(close(shape(w::CAMEL_SINE, 0.375), 1.0) && close(shape(w::CAMEL_SINE, 0.75), 0.0));
    // 6 square.
    assert!(close(shape(w::SQUARE, 0.1), 1.0) && close(shape(w::SQUARE, 0.6), -1.0));
    // 7 log-saw: decays from +1 to ~-48 dB over the first half, mirrored.
    assert!(close(shape(w::LOG_SAW, 0.0), 1.0));
    assert!(shape(w::LOG_SAW, 0.49) < 0.01 && shape(w::LOG_SAW, 0.25) > 0.0);
    assert!(shape(w::LOG_SAW, 0.999) < -0.98);

    // Every shape is bounded; the expected ones carry DC.
    let n = 4096;
    for wave in 0..w::COUNT {
        let vals: Vec<f32> = (0..n).map(|k| shape(wave, k as f32 / n as f32)).collect();
        assert!(
            vals.iter().all(|v| v.abs() <= 1.0 + 1e-6),
            "wave {wave} out of range"
        );
        let mean = vals.iter().sum::<f32>() / n as f32;
        let dc = matches!(
            wave,
            w::HALF_SINE | w::ABS_SINE | w::QUARTER_SINE | w::CAMEL_SINE
        );
        assert_eq!(mean.abs() > 0.05, dc, "wave {wave} mean {mean}");
    }
    // Phases outside [0, 1) wrap.
    assert!(close(shape(w::SINE, 1.25), 1.0) && close(shape(w::SINE, -0.75), 1.0));
}

#[test]
fn wave_input_selects_the_shape_in_a_graph() {
    // abs-sine at 100 Hz: positive mean, energy at 200 Hz, none at 100 Hz.
    let x = render("synthdef s = fmOp 100.0 0.0 1.0 0.0 2.0", 1.0);
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    assert!(
        (mean - 2.0 / core::f32::consts::PI).abs() < 0.01,
        "mean {mean}"
    );
    assert!(bin(&x, 100.0) < 0.01 && bin(&x, 200.0) > 0.3);
    // Out-of-range wave values clamp to log-saw (7), negative to sine.
    let hi = render("synthdef s = fmOp 100.0 0.0 1.0 0.0 99.0", 0.1);
    let ls = render("synthdef s = fmOp 100.0 0.0 1.0 0.0 7.0", 0.1);
    assert_eq!(hi, ls);
    let lo = render("synthdef s = fmOp 100.0 0.0 1.0 0.0 (0.0 - 3.0)", 0.1);
    let sine = render("synthdef s = fmOp 100.0 0.0 1.0 0.0 0.0", 0.1);
    assert_eq!(lo, sine);
}

#[test]
fn feedback_is_bounded_and_brightens() {
    for &f in &[55.0f32, 440.0, 4000.0] {
        for &fb in &[0.0f32, 0.25, 0.5, 1.0, 2.0, 4.0] {
            for wave in [0, 3, 6] {
                let x = render(
                    &format!("synthdef s = fmOp {f:.1} 0.0 0.7 {fb:.2} {wave}.0"),
                    1.0,
                );
                assert!(
                    x.iter().all(|v| v.is_finite() && v.abs() <= 0.7 + 1e-5),
                    "f={f} fb={fb} wave={wave} not bounded"
                );
            }
        }
    }
    // Sine with rising feedback walks the DX7 path: 0 is a sine, 0.5
    // (DX7 feedback 6) is close to a sawtooth (harmonic k at ~1/k), 1.0
    // (DX7 feedback 7) is a sawtooth smeared with noise.
    let spectrum = |fb: f32| {
        let x = render(&format!("synthdef s = fmOp 200.0 0.0 1.0 {fb:.2} 0.0"), 1.0);
        let h: Vec<f32> = (1..=8).map(|k| bin(&x, 200.0 * k as f32)).collect();
        let power = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let harmonic = h.iter().map(|a| a * a / 2.0).sum::<f32>();
        (h, 1.0 - harmonic / power)
    };
    let (h0, _) = spectrum(0.0);
    let (hq, _) = spectrum(0.25);
    let (hh, noise_h) = spectrum(0.5);
    let (_, noise_1) = spectrum(1.0);
    assert!(h0[1] < 1e-3, "no feedback, no 2nd harmonic: {}", h0[1]);
    let r = |h: &[f32], k: usize| h[k - 1] / h[0];
    assert!(
        r(&hq, 2) > 0.2 && r(&hh, 2) > r(&hq, 2),
        "2nd harmonic grows"
    );
    for k in 2..=5 {
        let saw = 1.0 / k as f32;
        assert!(
            (r(&hh, k) - saw).abs() < 0.12,
            "feedback 0.5 is saw-like: h{k}/h1 = {} vs {saw}",
            r(&hh, k)
        );
    }
    assert!(
        noise_h < 0.15,
        "feedback 0.5 is still mostly harmonic: {noise_h}"
    );
    assert!(noise_1 > 0.5, "feedback 1.0 is mostly noise: {noise_1}");
    // At feedback 1 the two-sample mean keeps the output from settling into
    // a Nyquist-rate limit cycle: little energy in the top octave.
    let x = render("synthdef s = fmOp 200.0 0.0 1.0 1.0 0.0", 1.0);
    assert!(bin(&x, 22000.0) < 0.01);
}

#[test]
fn level_scales_and_phase_reset_restarts() {
    let a = render("synthdef s = fmOp 300.0 0.0 1.0 0.3 0.0", 0.2);
    let b = render("synthdef s = fmOp 300.0 0.0 0.25 0.3 0.0", 0.2);
    assert!(a.iter().zip(&b).all(|(x, y)| (x * 0.25 - y).abs() < 1e-6));

    // Drive the UGen directly: a rising edge at sample 30 restarts the
    // phase, so samples 30.. equal samples 0.. of a fresh operator.
    let ctx = ProcessContext::new(SR, BLOCK);
    let mut freq = AudioBuffer::new(1, BLOCK);
    freq.channel_mut(0).fill(523.0);
    let mut fb = AudioBuffer::new(1, BLOCK);
    fb.channel_mut(0).fill(0.4);
    let mut reset = AudioBuffer::new(1, BLOCK);
    for (i, s) in reset.channel_mut(0).samples_mut().iter_mut().enumerate() {
        *s = if i >= 30 { 1.0 } else { 0.0 };
    }
    let mut op = microsynth::ugens::fm::FmOp::new();
    op.init(&ctx);
    let mut out = AudioBuffer::new(1, BLOCK);
    let inputs = [Some(&freq), None, None, Some(&fb), None, Some(&reset)];
    op.process(&ctx, &inputs, &mut out);
    let o = out.channel(0).samples();
    for k in 0..30 {
        assert!((o[30 + k] - o[k]).abs() < 1e-5, "sample {k}");
    }
}
