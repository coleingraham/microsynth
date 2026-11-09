//! `modalBank`: a bank of damped two-pole resonators.
//!
//! An impulse into the bank must ring each mode at `ratio · freq` (within a
//! cent) with the table's decay (within 5 %), strike position must silence
//! the modes with a node at the strike point, and the bank must stay finite
//! and bounded whatever its inputs.

use microsynth::context::ProcessContext;
use microsynth::dsl;
use microsynth::ir::{IrEdge, IrNode, IrSynthDef, RenderSpec, SynthDefClass, render_ir};
use microsynth::node::UGen;
use microsynth::ugens::{MODAL_F_REF, MODAL_MAX_MODES, ModalBank, modal_port};
use microsynth::*;

mod common;
use common::builtin_registry;

const BLOCK: usize = 64;

/// Render `secs` of a single-def DSL source at `sr`, mono.
fn render_at(src: &str, secs: f32, sr: f32) -> Vec<f32> {
    let registry = builtin_registry();
    let def = dsl::compile(src, &registry)
        .unwrap_or_else(|e| panic!("compile failed: {e:?}"))
        .remove(0);
    let mut engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: BLOCK,
    });
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let blocks = (secs * sr / BLOCK as f32).ceil() as usize;
    let mut out = Vec::with_capacity(blocks * BLOCK);
    for _ in 0..blocks {
        let buf = engine.render().expect("a sink is set");
        out.extend_from_slice(buf.channel(0).samples());
    }
    out
}

/// A `modalBank` source struck by one impulse: `modes` = (ratio, gain, T60).
fn bank_src(
    freq: f32,
    damp: f32,
    pos: f32,
    bright: f32,
    alpha: f32,
    modes: &[(f32, f32, f32)],
) -> String {
    let table: Vec<String> = modes
        .iter()
        .map(|(r, g, d)| format!("({r:.6}) ({g:.6}) ({d:.6})"))
        .collect();
    format!(
        "synthdef b = modalBank (impulse 0.0) ({freq:.6}) ({damp:.6}) ({pos:.6}) ({bright:.6}) ({alpha:.6}) {}",
        table.join(" ")
    )
}

/// Hann-windowed DFT magnitude at `freq`, normalised so a unit sine reads ~1.
fn bin(x: &[f32], freq: f64, sr: f64) -> f64 {
    let len = x.len() as f64;
    let (mut re, mut im, mut wsum) = (0.0f64, 0.0f64, 0.0f64);
    for (n, &s) in x.iter().enumerate() {
        let win = 0.5 - 0.5 * (core::f64::consts::TAU * n as f64 / len).cos();
        let w = core::f64::consts::TAU * freq * n as f64 / sr;
        re += win * s as f64 * w.cos();
        im -= win * s as f64 * w.sin();
        wsum += win;
    }
    2.0 * (re * re + im * im).sqrt() / wsum
}

/// The spectral peak within ±30 cents of `near`, to 0.01 cent.
fn peak_near(x: &[f32], near: f64, sr: f64) -> f64 {
    let (mut lo, mut hi) = (
        near * 2f64.powf(-30.0 / 1200.0),
        near * 2f64.powf(30.0 / 1200.0),
    );
    // Golden-section search on the (unimodal near a line) magnitude.
    let g = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..60 {
        let a = hi - g * (hi - lo);
        let b = lo + g * (hi - lo);
        if bin(x, a, sr) < bin(x, b, sr) {
            lo = a;
        } else {
            hi = b;
        }
    }
    0.5 * (lo + hi)
}

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2()
}

/// T60 of the component at `freq`: a line fitted to its level (dB, from
/// 20 ms Hann-windowed DFT frames) between `t0` and `t1` seconds.
fn t60_at(x: &[f32], freq: f64, sr: f64, t0: f64, t1: f64) -> f64 {
    let frame = (0.05 * sr) as usize;
    let hop = (0.01 * sr) as usize;
    let (mut ts, mut ls) = (Vec::new(), Vec::new());
    let mut start = (t0 * sr) as usize;
    while start + frame <= ((t1 * sr) as usize).min(x.len()) {
        let m = bin(&x[start..start + frame], freq, sr);
        ts.push((start + frame / 2) as f64 / sr);
        ls.push(20.0 * m.max(1e-30).log10());
        start += hop;
    }
    let n = ts.len() as f64;
    let (mt, ml) = (ts.iter().sum::<f64>() / n, ls.iter().sum::<f64>() / n);
    let num: f64 = ts.iter().zip(&ls).map(|(t, l)| (t - mt) * (l - ml)).sum();
    let den: f64 = ts.iter().map(|t| (t - mt) * (t - mt)).sum();
    -60.0 / (num / den)
}

const BAR: [(f32, f32, f32); 4] = [
    (1.0, 1.0, 3.0),
    (2.756, 0.6, 1.5),
    (5.404, 0.4, 0.8),
    (8.933, 0.3, 0.5),
];

#[test]
fn impulse_response_has_lines_at_the_mode_ratios_within_a_cent() {
    for &sr in &[44100.0f32, 48000.0] {
        for &f0 in &[110.0f32, 261.6256, 523.25] {
            let y = render_at(&bank_src(f0, 1.0, 0.0, 1.0, 0.0, &BAR), 1.0, sr);
            for &(r, _, _) in &BAR {
                let want = (r * f0) as f64;
                if want > 0.45 * sr as f64 {
                    continue;
                }
                let got = peak_near(&y, want, sr as f64);
                let c = cents(got, want);
                assert!(
                    c.abs() < 1.0,
                    "sr {sr} f0 {f0} ratio {r}: line at {got:.3} Hz, {c:.3} cents off {want:.3}"
                );
            }
        }
    }
}

#[test]
fn decay_times_match_the_table_within_five_percent() {
    let sr = 44100.0f64;
    // At C4 (keytracking neutral), damp 1.
    let y = render_at(
        &bank_src(MODAL_F_REF, 1.0, 0.0, 1.0, 0.0, &BAR),
        3.5,
        sr as f32,
    );
    for &(r, _, d) in &BAR {
        let t = t60_at(
            &y,
            (r * MODAL_F_REF) as f64,
            sr,
            0.05,
            (d as f64 * 0.8).max(0.4),
        );
        assert!(
            ((t / d as f64) - 1.0).abs() < 0.05,
            "ratio {r}: T60 {t:.3} s vs table {d} s"
        );
    }
    // damp 0.5 halves every decay; alpha 1 at an octave above C4 halves it again.
    let y = render_at(
        &bank_src(2.0 * MODAL_F_REF, 0.5, 0.0, 1.0, 1.0, &BAR[..2]),
        1.5,
        sr as f32,
    );
    for &(r, _, d) in &BAR[..2] {
        let want = d as f64 * 0.25;
        let t = t60_at(&y, (2.0 * r * MODAL_F_REF) as f64, sr, 0.03, want * 0.8);
        assert!(
            ((t / want) - 1.0).abs() < 0.05,
            "ratio {r}: T60 {t:.3} s vs {want:.3} s (damp 0.5, alpha 1, one octave up)"
        );
    }
}

#[test]
fn strike_position_silences_the_modes_with_a_node_there() {
    let sr = 44100.0f64;
    let f0 = 220.0f32;
    let modes = [
        (1.0, 1.0, 2.0),
        (2.0, 1.0, 2.0),
        (3.0, 1.0, 2.0),
        (4.0, 1.0, 2.0),
    ];
    let level = |pos: f32| -> Vec<f64> {
        let y = render_at(&bank_src(f0, 1.0, pos, 1.0, 0.0, &modes), 0.5, sr as f32);
        modes
            .iter()
            .map(|&(r, _, _)| bin(&y, (r * f0) as f64, sr))
            .collect()
    };
    let off = level(0.0);
    // The middle: slots 1 and 3 (modes 2 and 4) have a node there.
    let mid = level(0.5);
    assert!(
        mid[1] < 1e-4 * off[1] && mid[3] < 1e-4 * off[3],
        "pos 0.5: {mid:?} vs {off:?}"
    );
    assert!(
        (mid[0] / off[0] - 1.0).abs() < 0.01 && (mid[2] / off[2] - 1.0).abs() < 0.01,
        "pos 0.5 keeps modes 1, 3: {mid:?}"
    );
    // A third of the way: slot 2 (mode 3) is silenced, slot 0 is weighted sin(π/3).
    let third = level(1.0 / 3.0);
    assert!(third[2] < 1e-4 * off[2], "pos 1/3: {third:?}");
    let w0 = (core::f64::consts::PI / 3.0).sin();
    assert!(
        (third[0] / off[0] - w0).abs() < 0.01,
        "pos 1/3 weights mode 1 by sin(π/3): {}",
        third[0] / off[0]
    );
}

#[test]
fn bright_is_a_lowpass_on_how_the_modes_are_struck() {
    let sr = 44100.0f64;
    let f0 = 261.6256f32;
    let lvl = |bright: f32| {
        let y = render_at(&bank_src(f0, 1.0, 0.0, bright, 0.0, &BAR), 0.5, sr as f32);
        (bin(&y, f0 as f64, sr), bin(&y, (8.933 * f0) as f64, sr))
    };
    let (lo1, hi1) = lvl(1.0);
    let (lo0, hi0) = lvl(0.3);
    let fc = 100.0 * 2f64.powf(8.0 * 0.3);
    let want = |f: f64| 1.0 / (1.0 + (f / fc).powi(4)).sqrt();
    assert!(
        (lo0 / lo1 / (want(f0 as f64) / (1.0 / (1.0 + (f0 as f64 / 25600.0).powi(4)).sqrt()))
            - 1.0)
            .abs()
            < 0.01
    );
    let f9 = (8.933 * f0) as f64;
    let expect = want(f9) / (1.0 / (1.0 + (f9 / 25600.0).powi(4)).sqrt());
    assert!(
        (hi0 / hi1 / expect - 1.0).abs() < 0.02,
        "bright 0.3 darkens the 8.933 mode: {} vs {expect}",
        hi0 / hi1
    );
    assert!(hi0 / hi1 < 0.06);
}

#[test]
fn bounded_and_finite_for_extreme_inputs() {
    let sr = 44100.0f32;
    let gsum: f32 = BAR.iter().map(|m| m.1).sum();
    // Impulse excitation: |y| <= Σ|g| for any freq/damp/pos/bright/alpha.
    for &(f, damp, pos, bright, alpha) in &[
        (1.0e-3f32, 1.0f32, 0.0f32, 1.0f32, 0.0f32),
        (20_000.0, 1.0, 0.0, 1.0, 0.0),
        (1.0e9, 1.0, 0.0, 1.0, 0.0),
        (-440.0, 1.0, 0.0, 1.0, 0.0),
        (440.0, 1.0e9, 0.0, 1.0, 0.0),
        (440.0, -5.0, 0.0, 1.0, 0.0),
        (440.0, 1.0, 7.0, -3.0, 100.0),
        (30.0, 1.0, 0.5, 9.0, -100.0),
    ] {
        let y = render_at(&bank_src(f, damp, pos, bright, alpha, &BAR), 0.5, sr);
        let pk = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            y.iter().all(|v| v.is_finite()),
            "non-finite at {f} {damp} {pos} {bright} {alpha}"
        );
        assert!(
            pk <= gsum * 1.0001,
            "peak {pk} > Σg {gsum} at {f} {damp} {pos} {bright} {alpha}"
        );
    }
    // White noise into the longest decays the bank allows stays finite and
    // under the resonators' steady-state gain.
    let y = render_at(
        "synthdef n = modalBank whiteNoise 220.0 1000000.0 0.0 1.0 0.0 1.0 1.0 1000.0 3.1 1.0 1000.0",
        3.0,
        sr,
    );
    assert!(y.iter().all(|v| v.is_finite()));
    let pk = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(pk < 1.0e4, "noise into 120 s decays: peak {pk}");
    // NaN excitation and NaN parameters read as their defaults.
    let y = render_at(
        "synthdef q = modalBank ((impulse 0.0) + (0.0 / 0.0)) (0.0 / 0.0) (0.0 / 0.0) 0.0 1.0 0.0 1.0 1.0 1.0",
        0.2,
        sr,
    );
    assert!(y.iter().all(|v| v.is_finite()));
}

#[test]
fn modes_past_nyquist_are_dropped_and_unused_slots_are_silent() {
    let sr = 44100.0f64;
    // Mode 2 at 30 kHz is dropped; slot 1 (ratio 0) is unused.
    let y = render_at(
        &bank_src(
            1000.0,
            1.0,
            0.0,
            1.0,
            0.0,
            &[(1.0, 1.0, 1.0), (0.0, 1.0, 1.0), (30.0, 1.0, 1.0)],
        ),
        0.5,
        sr as f32,
    );
    let one = render_at(
        &bank_src(1000.0, 1.0, 0.0, 1.0, 0.0, &[(1.0, 1.0, 1.0)]),
        0.5,
        sr as f32,
    );
    assert_eq!(y, one);
}

#[test]
fn is_done_only_after_it_was_struck_and_rang_out() {
    let sr = 44100.0f32;
    let mut bank = ModalBank::new();
    let ctx = ProcessContext::new(sr, BLOCK);
    bank.init(&ctx);
    let n_ports = bank.spec().inputs.len();
    assert_eq!(n_ports, modal_port(MODAL_MAX_MODES - 1, 2) + 1);
    let konst = |v: f32| {
        let mut b = AudioBuffer::new(1, BLOCK);
        b.channel_mut(0).samples_mut().fill(v);
        b
    };
    let (freq, r0, g0, d0) = (konst(440.0), konst(1.0), konst(1.0), konst(0.05));
    let silence = konst(0.0);
    let mut strike = konst(0.0);
    strike.channel_mut(0).samples_mut()[0] = 1.0;
    let mut out = AudioBuffer::new(1, BLOCK);
    let run = |bank: &mut ModalBank, x: &AudioBuffer, out: &mut AudioBuffer| {
        let mut ins: Vec<Option<&AudioBuffer>> = vec![None; n_ports];
        ins[0] = Some(x);
        ins[1] = Some(&freq);
        ins[modal_port(0, 0)] = Some(&r0);
        ins[modal_port(0, 1)] = Some(&g0);
        ins[modal_port(0, 2)] = Some(&d0);
        bank.process(&ctx, &ins, out);
    };
    // Silent input before any strike: not done (it has not sounded yet).
    run(&mut bank, &silence, &mut out);
    assert!(!bank.is_done());
    run(&mut bank, &strike, &mut out);
    assert!(!bank.is_done(), "the strike block");
    run(&mut bank, &silence, &mut out);
    assert!(!bank.is_done(), "still ringing");
    // 0.05 s T60 = -60 dB; -120 dB after 0.1 s; give it 0.15 s.
    for _ in 0..(0.15 * sr / BLOCK as f32) as usize {
        run(&mut bank, &silence, &mut out);
    }
    assert!(bank.is_done(), "rang out");
}

#[test]
fn inline_const_mode_table_in_the_ir_renders_like_the_dsl() {
    let reg = builtin_registry();
    let f0 = 330.0f32;
    let mut consts = vec![(1u32, f0), (2, 1.0), (3, 0.0), (4, 1.0), (5, 0.0)];
    for (k, &(r, g, d)) in BAR.iter().enumerate() {
        consts.push((modal_port(k, 0) as u32, r));
        consts.push((modal_port(k, 1) as u32, g));
        consts.push((modal_port(k, 2) as u32, d));
    }
    let ir = IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "bank".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![
            IrNode::UGen {
                kind: "impulse".into(),
                consts: vec![(0, 0.0)],
            },
            IrNode::UGen {
                kind: "modalBank".into(),
                consts,
            },
        ],
        edges: vec![IrEdge {
            from: 0,
            to: 1,
            to_input: 0,
        }],
        params: Vec::new(),
        audio_inputs: Vec::new(),
        table_bindings: Vec::new(),
        output_node: 1,
    };
    ir.validate(&reg).expect("valid");
    let back = IrSynthDef::from_json(&ir.to_json()).expect("round trip");
    assert_eq!(back, ir);
    let mut spec = RenderSpec::nsynth();
    spec.sample_rate = 44100.0;
    spec.duration_secs = 0.5;
    spec.gate_on_secs = 0.5;
    spec.max_tail_secs = 0.0;
    let y = render_ir(&ir, &reg, &spec).expect("render").remove(0);
    let want = render_at(&bank_src(f0, 1.0, 0.0, 1.0, 0.0, &BAR), 0.5, 44100.0);
    let n = y.len().min(want.len());
    let d = y[..n]
        .iter()
        .zip(&want[..n])
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(d < 1e-6, "IR vs DSL: max diff {d}");
}

#[test]
fn a_pitch_glide_keeps_the_ringing_amplitude() {
    // One mode that barely decays, swept 400 -> 200 Hz over 0.5 s after the
    // strike: the coupled-form resonator keeps its amplitude through the glide
    // (a direct-form biquad would not).
    let sr = 44100.0f32;
    let y = render_at(
        "synthdef g = modalBank (impulse 0.0) (line 400.0 200.0 0.5) 1.0 0.0 1.0 0.0 1.0 1.0 120.0",
        0.6,
        sr,
    );
    let win = (0.02 * sr) as usize;
    let peaks: Vec<f32> = y
        .chunks(win)
        .map(|c| c.iter().fold(0.0f32, |m, v| m.max(v.abs())))
        .collect();
    // Expected: the 120 s T60's own decay, 0.5 dB/s.
    let first = peaks[1];
    for (i, p) in peaks.iter().enumerate().take(peaks.len() - 1).skip(1) {
        let decay = 10f32.powf(-3.0 * (i - 1) as f32 * 0.02 / 120.0);
        assert!(
            (p / (first * decay) - 1.0).abs() < 0.003,
            "window {i}: amplitude {p} vs {} during the glide",
            first * decay
        );
    }
}

#[test]
fn each_channel_rings_only_from_its_own_input() {
    // Left struck, right silent: the right channel must stay silent across
    // block boundaries (one resonator state per channel), and the left must
    // equal a mono bank struck the same way.
    let sr = 44100.0f32;
    let ctx = ProcessContext::new(sr, BLOCK);
    let n_ports = modal_port(MODAL_MAX_MODES - 1, 2) + 1;
    let konst = |chans: usize, v: f32| {
        let mut b = AudioBuffer::new(chans, BLOCK);
        for c in 0..chans {
            b.channel_mut(c).samples_mut().fill(v);
        }
        b
    };
    let (f2, r2, d2) = (konst(2, 440.0), konst(2, 1.0), konst(2, 1.0));
    let (f1, r1, d1) = (konst(1, 440.0), konst(1, 1.0), konst(1, 1.0));
    let mut stereo = ModalBank::new();
    let mut mono = ModalBank::new();
    stereo.init(&ctx);
    mono.init(&ctx);
    for blk in 0..8 {
        let mut x2 = AudioBuffer::new(2, BLOCK);
        let mut x1 = AudioBuffer::new(1, BLOCK);
        if blk == 0 {
            x2.channel_mut(0).samples_mut()[0] = 1.0;
            x1.channel_mut(0).samples_mut()[0] = 1.0;
        }
        let mut ins2: Vec<Option<&AudioBuffer>> = vec![None; n_ports];
        ins2[0] = Some(&x2);
        ins2[1] = Some(&f2);
        ins2[modal_port(0, 0)] = Some(&r2);
        ins2[modal_port(0, 2)] = Some(&d2);
        let mut ins1: Vec<Option<&AudioBuffer>> = vec![None; n_ports];
        ins1[0] = Some(&x1);
        ins1[1] = Some(&f1);
        ins1[modal_port(0, 0)] = Some(&r1);
        ins1[modal_port(0, 2)] = Some(&d1);
        let mut o2 = AudioBuffer::new(2, BLOCK);
        let mut o1 = AudioBuffer::new(1, BLOCK);
        stereo.process(&ctx, &ins2, &mut o2);
        mono.process(&ctx, &ins1, &mut o1);
        assert_eq!(
            o2.channel(0).samples(),
            o1.channel(0).samples(),
            "block {blk}: left"
        );
        assert!(
            o2.channel(1).samples().iter().all(|&v| v == 0.0),
            "block {blk}: the silent right channel rings"
        );
    }
}
