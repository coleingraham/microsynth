//! Behavior of the `WowFlutter` UGen: depth as speed deviation in percent,
//! scrape and weave, per-channel delay memory, seeding, block-size
//! invariance, the delay range, irregularity, and reset.

use microsynth::node::UGen;
use microsynth::ugens::WowFlutter;
use microsynth::{AudioBuffer, ProcessContext};

const SR: f32 = 44100.0;
const TONE_HZ: f64 = 1000.0;
/// Centre delay in samples at `SR`, and the read margin kept at each end.
const CENTER_SAMPLES: f64 = 441.0;
const READ_MARGIN: f64 = 3.0;
/// Period, in samples, of the ramp used to read back the delay.
const RAMP_PERIOD: usize = 4096;

/// Every input, passed explicitly to each render. The baseline has scrape and
/// weave off so depth measurements see only wow and flutter.
#[derive(Clone, Copy)]
struct Settings {
    wow_rate: f32,
    wow_depth: f32,
    flutter_rate: f32,
    flutter_depth: f32,
    mix: f32,
    irregularity: f32,
    scrape: f32,
    weave: f32,
    seed: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            wow_rate: 0.6,
            wow_depth: 0.12,
            flutter_rate: 9.0,
            flutter_depth: 0.05,
            mix: 1.0,
            irregularity: 0.6,
            scrape: 0.0,
            weave: 0.0,
            seed: 1.0,
        }
    }
}

/// A 1 kHz sine at half scale, computed in f64 so its own phase is exact.
fn tone(i: usize) -> f32 {
    (0.5 * (core::f64::consts::TAU * TONE_HZ * i as f64 / SR as f64).sin()) as f32
}

/// A rising ramp from 0 to just under 1, repeating every `RAMP_PERIOD` samples.
fn ramp(i: usize) -> f32 {
    (i % RAMP_PERIOD) as f32 / RAMP_PERIOD as f32
}

fn silence(_i: usize) -> f32 {
    0.0
}

fn constant(value: f32, block_size: usize) -> AudioBuffer {
    let mut buf = AudioBuffer::new(1, block_size);
    buf.channel_mut(0).samples_mut().fill(value);
    buf
}

/// Renders at least `num_samples` samples through an initialized `ugen`,
/// rounded up to whole blocks, with `left` and `right` indexed by sample from 0.
fn render_with(
    ugen: &mut WowFlutter,
    settings: Settings,
    num_samples: usize,
    block_size: usize,
    left: fn(usize) -> f32,
    right: fn(usize) -> f32,
) -> [Vec<f32>; 2] {
    let ctx = ProcessContext::new(SR, block_size);
    let params = [
        constant(settings.wow_rate, block_size),
        constant(settings.wow_depth, block_size),
        constant(settings.flutter_rate, block_size),
        constant(settings.flutter_depth, block_size),
        constant(settings.mix, block_size),
        constant(settings.irregularity, block_size),
        constant(settings.scrape, block_size),
        constant(settings.weave, block_size),
        constant(settings.seed, block_size),
    ];
    let mut out = [Vec::new(), Vec::new()];
    let mut start = 0;
    while start < num_samples {
        let mut input = AudioBuffer::new(2, block_size);
        for i in 0..block_size {
            input.channel_mut(0).samples_mut()[i] = left(start + i);
            input.channel_mut(1).samples_mut()[i] = right(start + i);
        }
        let mut inputs: Vec<Option<&AudioBuffer>> = vec![Some(&input)];
        inputs.extend(params.iter().map(Some));
        let mut output = AudioBuffer::new(2, block_size);
        ugen.process(&ctx, &inputs, &mut output);
        out[0].extend_from_slice(output.channel(0).samples());
        out[1].extend_from_slice(output.channel(1).samples());
        start += block_size;
    }
    out
}

fn render(
    settings: Settings,
    num_samples: usize,
    block_size: usize,
    left: fn(usize) -> f32,
    right: fn(usize) -> f32,
) -> [Vec<f32>; 2] {
    let mut ugen = WowFlutter::new();
    ugen.init(&ProcessContext::new(SR, block_size));
    render_with(&mut ugen, settings, num_samples, block_size, left, right)
}

fn seconds(secs: f64) -> usize {
    (secs * SR as f64) as usize
}

/// Per-cycle frequency of a rendered tone, from rising zero crossings, as a
/// fractional deviation from `TONE_HZ`, starting `skip_secs` in.
fn deviation_track(samples: &[f32], skip_secs: f64) -> Vec<f64> {
    let sr = SR as f64;
    let crossings: Vec<f64> = samples
        .windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] < 0.0 && w[1] >= 0.0)
        .map(|(i, w)| {
            let (a, b) = (w[0] as f64, w[1] as f64);
            i as f64 + a / (a - b)
        })
        .collect();
    crossings
        .windows(2)
        .filter(|w| w[0] / sr >= skip_secs)
        .map(|w| sr / (w[1] - w[0]) / TONE_HZ - 1.0)
        .collect()
}

/// The delay, in samples, applied at each output sample of a rendered ramp,
/// starting `skip_secs` in. Hermite interpolation reproduces a straight line
/// exactly, so the delay reads straight off the ramp. A sample is `None` when
/// a read anywhere in the delay line's range could reach back across the
/// ramp's wrap.
fn delay_track(samples: &[f32], skip_secs: f64) -> Vec<Option<f64>> {
    let reach = (2.0 * CENTER_SAMPLES + READ_MARGIN) as usize + 1;
    samples
        .iter()
        .enumerate()
        .skip(seconds(skip_secs))
        .map(|(n, &y)| {
            let since_wrap = n % RAMP_PERIOD;
            (since_wrap >= reach).then_some(since_wrap as f64 - y as f64 * RAMP_PERIOD as f64)
        })
        .collect()
}

/// Playback speed deviation, as a fraction, between consecutive delay readings.
fn speed_deviations(track: &[Option<f64>]) -> Vec<f64> {
    track
        .windows(2)
        .filter_map(|w| match (w[0], w[1]) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        })
        .collect()
}

/// Half the peak-to-peak range of a deviation track, in percent.
fn peak_deviation_percent(track: &[f64]) -> f64 {
    let max = track.iter().copied().fold(f64::MIN, f64::max);
    let min = track.iter().copied().fold(f64::MAX, f64::min);
    50.0 * (max - min)
}

/// Pearson correlation between a track and itself `lag` entries later.
fn lagged_correlation(track: &[f64], lag: usize) -> f64 {
    let (a, b) = (&track[..track.len() - lag], &track[lag..]);
    let n = a.len() as f64;
    let (mean_a, mean_b) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum();
    let var_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum();
    let var_b: f64 = b.iter().map(|y| (y - mean_b).powi(2)).sum();
    cov / (var_a * var_b).sqrt()
}

fn measured_depth_percent(settings: Settings, secs: f64) -> f64 {
    let [left, _] = render(settings, seconds(secs), 128, tone, tone);
    peak_deviation_percent(&deviation_track(&left, 0.5))
}

/// Every random component active, for the determinism tests.
fn all_components() -> Settings {
    Settings {
        wow_depth: 0.5,
        irregularity: 1.0,
        scrape: 1.0,
        weave: 0.0002,
        ..Settings::default()
    }
}

#[test]
fn depth_is_peak_speed_deviation_in_percent_at_any_rate() {
    let steady = Settings {
        irregularity: 0.0,
        ..Settings::default()
    };
    for wow_rate in [0.5f32, 2.0] {
        let settings = Settings {
            wow_rate,
            wow_depth: 0.5,
            flutter_depth: 0.0,
            ..steady
        };
        let measured = measured_depth_percent(settings, 0.6 + 2.0 / wow_rate as f64);
        assert!(
            (measured - 0.5).abs() <= 0.025,
            "wow at {wow_rate} Hz, depth 0.5%: measured {measured:.4}%"
        );
    }
    for flutter_rate in [6.0f32, 12.0] {
        let settings = Settings {
            flutter_rate,
            flutter_depth: 0.2,
            wow_depth: 0.0,
            ..steady
        };
        let measured = measured_depth_percent(settings, 1.5);
        assert!(
            (measured - 0.2).abs() <= 0.01,
            "flutter at {flutter_rate} Hz, depth 0.2%: measured {measured:.4}%"
        );
    }
}

#[test]
fn scrape_depth_is_peak_equivalent_speed_deviation_in_percent() {
    let settings = Settings {
        wow_depth: 0.0,
        flutter_depth: 0.0,
        scrape: 1.0,
        ..Settings::default()
    };
    let [left, _] = render(settings, seconds(4.0), 128, ramp, ramp);
    let speeds = speed_deviations(&delay_track(&left, 0.5));
    let rms = (speeds.iter().map(|s| s * s).sum::<f64>() / speeds.len() as f64).sqrt();
    let measured = 100.0 * core::f64::consts::SQRT_2 * rms;
    assert!(
        (measured - 1.0).abs() <= 0.1,
        "scrape 1%: measured {measured:.4}%"
    );
}

#[test]
fn weave_moves_the_channels_apart_within_its_peak_and_nothing_else() {
    let weave = 0.0002;
    let settings = Settings {
        wow_depth: 0.0,
        flutter_depth: 0.0,
        weave,
        ..Settings::default()
    };
    let [left, right] = render(settings, seconds(20.0), 128, ramp, ramp);
    let peak_samples = weave as f64 * SR as f64;
    let mut largest: f64 = 0.0;
    for (l, r) in delay_track(&left, 0.5)
        .into_iter()
        .zip(delay_track(&right, 0.5))
    {
        if let (Some(l), Some(r)) = (l, r) {
            let difference = l - r;
            assert!(
                difference.abs() <= peak_samples + 1e-3,
                "channels {difference:.4} samples apart, peak is {peak_samples:.4}"
            );
            assert!(
                ((l + r) / 2.0 - CENTER_SAMPLES).abs() <= 1e-3,
                "weave must not move the channels' shared timing: {:.4}",
                (l + r) / 2.0
            );
            largest = largest.max(difference.abs());
        }
    }
    assert!(
        largest >= 0.5 * peak_samples,
        "weave should reach at least half its peak in 20 s: {largest:.4} of {peak_samples:.4} samples"
    );
}

#[test]
fn extreme_settings_stay_finite_and_inside_the_delay_range() {
    let extreme = Settings {
        wow_rate: 0.1,
        wow_depth: 10.0,
        flutter_rate: 40.0,
        flutter_depth: 10.0,
        irregularity: 1.0,
        scrape: 5.0,
        weave: 0.001,
        ..Settings::default()
    };
    let [left, right] = render(extreme, seconds(5.0), 128, tone, tone);
    assert!(
        left.iter()
            .chain(&right)
            .all(|x| x.is_finite() && x.abs() <= 0.6),
        "output must stay finite and near the input's level"
    );
    let [left, right] = render(extreme, seconds(5.0), 128, ramp, ramp);
    let (lowest, highest) = (READ_MARGIN, 2.0 * CENTER_SAMPLES - READ_MARGIN);
    for delay in delay_track(&left, 0.0)
        .into_iter()
        .chain(delay_track(&right, 0.0))
        .flatten()
    {
        assert!(
            (lowest - 1e-3..=highest + 1e-3).contains(&delay),
            "delay {delay:.3} samples left the {lowest}-{highest} range"
        );
    }
}

#[test]
fn each_channel_depends_only_on_its_own_input() {
    let n = seconds(1.0);
    let beside_silence = render(all_components(), n, 128, tone, silence);
    let beside_ramp = render(all_components(), n, 128, tone, ramp);
    assert!(
        beside_silence[0].iter().any(|x| x.abs() > 0.1),
        "left should carry the tone"
    );
    assert!(
        beside_silence[1].iter().all(|&x| x == 0.0),
        "right input is silent, so right output must be silent"
    );
    assert_eq!(
        beside_silence[0], beside_ramp[0],
        "left output must not change with the right input"
    );
    let swapped_silence = render(all_components(), n, 128, silence, tone);
    let swapped_ramp = render(all_components(), n, 128, ramp, tone);
    assert_eq!(
        swapped_silence[1], swapped_ramp[1],
        "right output must not change with the left input"
    );
}

#[test]
fn identical_channels_stay_identical_without_weave() {
    let settings = Settings {
        weave: 0.0,
        ..all_components()
    };
    let [left, right] = render(settings, seconds(1.0), 128, tone, tone);
    assert_eq!(left, right);
}

#[test]
fn equal_seeds_match_and_different_seeds_differ() {
    let n = seconds(2.0);
    let a = render(
        Settings {
            seed: 11.0,
            ..all_components()
        },
        n,
        128,
        tone,
        tone,
    );
    let b = render(
        Settings {
            seed: 11.0,
            ..all_components()
        },
        n,
        128,
        tone,
        tone,
    );
    let c = render(
        Settings {
            seed: 12.0,
            ..all_components()
        },
        n,
        128,
        tone,
        tone,
    );
    assert_eq!(a, b, "equal seeds must give identical output");
    assert_ne!(a[0], c[0], "different seeds must give different output");
    assert_ne!(a[1], c[1], "different seeds must give different output");
}

#[test]
fn output_does_not_depend_on_block_size() {
    let n = 128 * 345;
    let small = render(all_components(), n, 64, tone, tone);
    let large = render(all_components(), n, 128, tone, tone);
    assert_eq!(small, large);
}

#[test]
fn depth_beyond_the_delay_range_is_scaled_down_not_clipped() {
    let settings = Settings {
        wow_rate: 0.1,
        wow_depth: 3.0,
        flutter_depth: 0.0,
        irregularity: 0.0,
        ..Settings::default()
    };
    let [left, _] = render(settings, seconds(11.0), 128, tone, tone);
    assert!(
        left.iter().all(|x| x.is_finite() && x.abs() <= 0.51),
        "output must stay finite and at the input's level"
    );
    let ceiling = core::f64::consts::TAU * 0.1 * (CENTER_SAMPLES - READ_MARGIN) / SR as f64 * 100.0;
    let measured = peak_deviation_percent(&deviation_track(&left, 0.5));
    assert!(
        (measured - ceiling).abs() <= 0.05 * ceiling,
        "3% at 0.1 Hz exceeds the delay range: expected the {ceiling:.4}% ceiling, measured {measured:.4}%"
    );
}

#[test]
fn irregularity_breaks_up_the_periodic_motion() {
    let wow_only = Settings {
        wow_rate: 2.0,
        wow_depth: 0.5,
        flutter_depth: 0.0,
        ..Settings::default()
    };
    // Four wow periods, counted in cycles of the tone.
    let lag = (4.0 / 2.0 * TONE_HZ) as usize;
    let correlation = |irregularity: f32| {
        let [left, _] = render(
            Settings {
                irregularity,
                ..wow_only
            },
            seconds(20.5),
            128,
            tone,
            tone,
        );
        lagged_correlation(&deviation_track(&left, 0.5), lag)
    };
    let steady = correlation(0.0);
    let irregular = correlation(1.0);
    assert!(
        steady > 0.95,
        "steady wow should repeat every period: {steady:.3}"
    );
    assert!(
        irregular < 0.7,
        "irregular wow should not repeat every period: {irregular:.3} (steady {steady:.3})"
    );
}

#[test]
fn reset_reproduces_a_fresh_instance() {
    let mut ugen = WowFlutter::new();
    ugen.init(&ProcessContext::new(SR, 128));
    let first = render_with(&mut ugen, all_components(), seconds(0.5), 128, tone, tone);
    ugen.reset();
    let second = render_with(&mut ugen, all_components(), seconds(0.5), 128, tone, tone);
    assert_eq!(first, second);
}
