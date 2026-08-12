//! Demo render for the partials + shaped-noise ugen (MOT-636): a synthetic
//! dictionary spanning five grid pitches, driven by a continuous f0 glide
//! plus vibrato, rendered offline to a WAV file. Not part of the test suite
//! -- run manually with:
//!
//! ```text
//! cargo run --release --example render_partials_demo -- [output.wav]
//! ```
//!
//! Defaults to `/Volumes/T7/nmf-channels/mot-636/demo.wav` (Cole's standing
//! rule: render artifacts go to T7, not the repo or local disk); pass a path
//! to write elsewhere. Every dictionary entry here is synthetic (hand-built
//! harmonic-series-plus-noise coefficients) -- no learned/real instrument
//! data, consistent with "no learned instrument may be compiled into
//! microsynth source" (RFC requirement 8).

#[path = "common/wav.rs"]
mod wav;

use microsynth::coeff_table::{CoeffTable, PitchEntry};
use microsynth::ugens::partials::PartialsNoise;
use microsynth::ugens::*;
use microsynth::*;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::Arc;

const SAMPLE_RATE: f32 = 44100.0;
const BLOCK_SIZE: usize = 128;
const DEFAULT_OUTPUT: &str = "/Volumes/T7/nmf-channels/mot-636/demo.wav";

/// Build a synthetic five-pitch dictionary spanning roughly one octave.
/// Each entry: `M` = 6 harmonic partials on a 1/m^p rolloff (p grows
/// slightly with pitch, so the timbre visibly brightens/darkens across the
/// glide -- otherwise every entry would sound identical and the render
/// wouldn't demonstrate table interpolation at all), plus `J` = 2 small
/// noise bands. Every entry is separately normalized to L1 == 1 (the RFC's
/// channel-parameterization invariant this ugen relies on for simplex
/// closure).
fn synthetic_dictionary() -> CoeffTable {
    let f0s = [220.0f32, 262.0, 311.0, 370.0, 440.0];
    let m_partials = 6;
    let j_noise = 2;
    let noise_mass_total = 0.12;

    let entries = f0s
        .iter()
        .enumerate()
        .map(|(k, &f0_hz)| {
            let rolloff = 1.0 + 0.08 * k as f32;
            let raw_partials: Vec<f32> = (1..=m_partials)
                .map(|m| 1.0 / (m as f32).powf(rolloff))
                .collect();
            let partial_mass: f32 = raw_partials.iter().sum();
            let partial_scale = (1.0 - noise_mass_total) / partial_mass;
            let mut coefficients: Vec<f32> =
                raw_partials.iter().map(|&w| w * partial_scale).collect();
            // Two noise bands, splitting the fixed noise budget unevenly
            // (a little more mass on the lower band) for a touch of
            // per-pitch character rather than a flat split everywhere.
            coefficients.push(noise_mass_total * 0.65);
            coefficients.push(noise_mass_total * 0.35);

            PitchEntry {
                f0_hz,
                inharmonicity_stretch: 1.0,
                partial_freqs: (1..=m_partials).map(|m| m as f32 * f0_hz).collect(),
                k_channels: 1,
                j_noise,
                coefficients,
                metadata: vec![],
            }
        })
        .collect();

    CoeffTable {
        name: "mot-636-demo".into(),
        entries,
    }
}

fn main() {
    let output = env::args().nth(1).unwrap_or_else(|| DEFAULT_OUTPUT.into());
    let output_path = Path::new(&output);

    let table = synthetic_dictionary();
    let ugen: Box<dyn UGen> = Box::new(PartialsNoise::new(Arc::new(table)));

    let dur_secs = 4.0f32;
    let num_samples = (dur_secs * SAMPLE_RATE) as usize;
    let num_blocks = num_samples.div_ceil(BLOCK_SIZE);

    let mut engine = Engine::new(EngineConfig {
        sample_rate: SAMPLE_RATE,
        block_size: BLOCK_SIZE,
    });
    let g = engine.graph_mut();

    // f0 glide: 220 -> 440 Hz (the dictionary's full grid span) over the
    // whole render.
    let glide_start = g.add_node(Box::new(Const::new(220.0)));
    let glide_end = g.add_node(Box::new(Const::new(440.0)));
    let glide_dur = g.add_node(Box::new(Const::new(dur_secs)));
    let base_freq = g.add_node(Box::new(Line::new()));
    g.connect(glide_start, base_freq, 0);
    g.connect(glide_end, base_freq, 1);
    g.connect(glide_dur, base_freq, 2);

    // Vibrato: +-1.2% sinusoidal frequency modulation at 5.5 Hz, applied as
    // freq = base_freq * (1 + depth * sin(2*pi*rate*t)).
    let vib_rate = g.add_node(Box::new(Const::new(5.5)));
    let vib_phase0 = g.add_node(Box::new(Const::new(0.0)));
    let vib_osc = g.add_node(Box::new(SinOsc::new()));
    g.connect(vib_rate, vib_osc, 0);
    g.connect(vib_phase0, vib_osc, 1);

    let vib_depth = g.add_node(Box::new(Const::new(0.012)));
    let vib_scaled = g.add_node(Box::new(BinOpUGen::new(BinOpKind::Mul)));
    g.connect(vib_osc, vib_scaled, 0);
    g.connect(vib_depth, vib_scaled, 1);

    let one = g.add_node(Box::new(Const::new(1.0)));
    let vib_modulator = g.add_node(Box::new(BinOpUGen::new(BinOpKind::Add)));
    g.connect(vib_scaled, vib_modulator, 0);
    g.connect(one, vib_modulator, 1);

    let freq = g.add_node(Box::new(BinOpUGen::new(BinOpKind::Mul)));
    g.connect(base_freq, freq, 0);
    g.connect(vib_modulator, freq, 1);

    // Gentle attack so the render doesn't click in at full gain instantly.
    let attack_start = g.add_node(Box::new(Const::new(0.0)));
    let attack_end = g.add_node(Box::new(Const::new(0.75)));
    let attack_dur = g.add_node(Box::new(Const::new(0.08)));
    let gain = g.add_node(Box::new(Line::new()));
    g.connect(attack_start, gain, 0);
    g.connect(attack_end, gain, 1);
    g.connect(attack_dur, gain, 2);

    let osc = g.add_node(ugen);
    g.connect(freq, osc, 0);
    g.connect(gain, osc, 1);
    g.set_sink(osc);

    engine.prepare();
    let channels = engine.render_offline(num_blocks);
    let samples = &channels[0][..num_samples.min(channels[0].len())];

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("failed to create output directory");
    }
    wav::write_pcm16(&[samples], SAMPLE_RATE, output_path);
    println!(
        "Wrote {} samples ({:.2}s) to {}",
        samples.len(),
        samples.len() as f32 / SAMPLE_RATE,
        output_path.display()
    );
}
