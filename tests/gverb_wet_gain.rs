//! GVerb's wet path is level-normalized to unit pink-noise RMS gain.

use microsynth::ugens::*;
use microsynth::*;

const SAMPLE_RATE: f32 = 44100.0;
const BLOCK_SIZE: usize = 64;
const SETTLE_SECONDS: f32 = 4.0;
const MEASURE_SECONDS: f32 = 4.0;

fn rms(samples: &[f32]) -> f64 {
    (samples
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt()
}

fn blocks() -> usize {
    ((SETTLE_SECONDS + MEASURE_SECONDS) * SAMPLE_RATE) as usize / BLOCK_SIZE
}

fn settle_samples() -> usize {
    (SETTLE_SECONDS * SAMPLE_RATE) as usize / BLOCK_SIZE * BLOCK_SIZE
}

fn render(source: Box<dyn UGen>, reverb: Option<(f32, f32, f32, f32)>) -> Vec<Vec<f32>> {
    let mut engine = Engine::new(EngineConfig::default());
    let graph = engine.graph_mut();
    let src = graph.add_node(source);
    let sink = match reverb {
        None => src,
        Some((roomsize, damping, wet, dry)) => {
            let verb = graph.add_node(Box::new(GVerb::new()));
            graph.connect(src, verb, 0);
            for (port, value) in [(1, roomsize), (2, damping), (3, wet), (4, dry)] {
                let node = graph.add_node(Box::new(Const::new(value)));
                graph.connect(node, verb, port);
            }
            verb
        }
    };
    graph.set_sink(sink);
    engine.prepare();
    engine.render_offline(blocks())
}

fn stereo_rms_gain(out: &[Vec<f32>], input_rms: f64) -> f64 {
    let skip = settle_samples();
    let left = rms(&out[0][skip..]) / input_rms;
    let right = rms(&out[1][skip..]) / input_rms;
    ((left * left + right * right) / 2.0).sqrt()
}

#[test]
fn wet_path_has_unit_pink_noise_gain_across_roomsize_and_damping() {
    let input = render(Box::new(PinkNoise::new()), None);
    let input_rms = rms(&input[0][settle_samples()..]);
    let settings = [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.55, 0.42),
        (0.85, 0.3),
        (1.0, 0.0),
        (0.975, 0.525),
        (0.5, 0.9965),
        (1.0, 0.99965),
        (0.3, 1.0),
    ];
    for (roomsize, damping) in settings {
        let out = render(
            Box::new(PinkNoise::new()),
            Some((roomsize, damping, 1.0, 0.0)),
        );
        let gain_db = 20.0 * stereo_rms_gain(&out, input_rms).log10();
        assert!(
            gain_db.abs() <= 1.0,
            "roomsize {roomsize} damping {damping}: wet pink-noise gain {gain_db:+.2} dB, expected within 1 dB of unity"
        );
    }
}

#[test]
fn dry_path_is_unaffected_by_wet_normalization() {
    let input = render(Box::new(WhiteNoise::with_seed(7)), None);
    let out = render(
        Box::new(WhiteNoise::with_seed(7)),
        Some((0.85, 0.3, 0.0, 1.0)),
    );
    assert_eq!(out[0], input[0]);
    assert_eq!(out[1], input[0]);
}

/// Prints `GVERB_WET_LOG_GAIN` for the current comb and allpass network. Run with
/// `cargo test --release --test gverb_wet_gain -- --ignored --nocapture` after changing
/// the reverb's structure, and paste the output over the table in `filters.rs`.
#[test]
#[ignore]
fn print_uncompensated_wet_log_gain_table() {
    const DAMPING_NODES: [f32; 29] = [
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
        0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 1.0,
    ];
    let input = render(Box::new(PinkNoise::new()), None);
    let input_rms = rms(&input[0][settle_samples()..]);
    for step in 0..21 {
        let roomsize = step as f32 / 20.0;
        let row: Vec<String> = DAMPING_NODES
            .iter()
            .map(|&damping| {
                let out = render(
                    Box::new(PinkNoise::new()),
                    Some((roomsize, damping, 1.0, 0.0)),
                );
                let measured = stereo_rms_gain(&out, input_rms);
                let raw = measured / f64::from(GVerb::wet_gain_compensation(roomsize, damping));
                format!("{:.5}", raw.ln())
            })
            .collect();
        println!("    [{}],", row.join(", "));
    }
}
