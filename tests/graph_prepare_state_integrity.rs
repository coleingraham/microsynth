//! Regression tests for `AudioGraph::prepare()`'s node (re-)initialization.
//!
//! `prepare()` runs after every structural change to the graph -- any voice
//! spawned or freed, not just once at startup. If it re-ran `UGen::init` on
//! nodes that are already live, any UGen whose `init` resets internal state
//! (envelope stage, oscillator/ramp phase, playback position, delay-line
//! cursor, ...) would be restarted or glitched by an unrelated voice
//! spawning or freeing elsewhere in the same graph.
//!
//! These tests drive that scenario directly with percussive ("snare")
//! voices that finish and fall silent, followed by unrelated ("lead")
//! voices spawned later, and check that the finished snare voices --
//! silent, no longer needed, and never explicitly freed -- stay silent
//! instead of being resurrected by the later, unrelated spawns.

use microsynth::dsl;
use microsynth::*;

mod common;
use common::builtin_registry;

fn make_engine(block_size: usize) -> Engine {
    Engine::new(EngineConfig {
        sample_rate: 44100.0,
        block_size,
    })
}

/// Render `num_blocks` blocks and return the bus's channel-0 samples,
/// flattened in render order.
fn render_samples(engine: &mut Engine, num_blocks: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for _ in 0..num_blocks {
        if let Some(buf) = engine.render() {
            out.extend_from_slice(buf.channel(0).samples());
        }
    }
    out
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

fn segment_rms(samples: &[f32], seg_samples: usize) -> Vec<f32> {
    samples.chunks(seg_samples).map(rms).collect()
}

/// Assert two equal-length sample buffers match exactly, with a diagnostic
/// summary (first differing index, peak absolute difference, RMS of each
/// side) instead of a full-vector diff on failure.
fn assert_tails_match(tail: &[f32], baseline: &[f32], context: &str) {
    assert_eq!(tail.len(), baseline.len(), "{context}: length mismatch");
    let first_diff = tail.iter().zip(baseline.iter()).position(|(a, b)| a != b);
    let peak_abs_diff = tail
        .iter()
        .zip(baseline.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        first_diff.is_none(),
        "{context}: first differing sample at index {}, peak abs diff {peak_abs_diff}, \
         tail rms {} vs baseline rms {}",
        first_diff.unwrap(),
        rms(tail),
        rms(baseline),
    );
}

const BLOCK: usize = 64;
/// 100 blocks (~145 ms at 44.1 kHz) per segment.
const SEG_BLOCKS: usize = 100;
const SEG_SAMPLES: usize = SEG_BLOCKS * BLOCK;

/// A short, noisy "snare" hit: white noise through a fast percussive
/// envelope. Fully decays (1 ms attack + 20 ms release) well inside one
/// segment.
fn snare_def(registry: &dsl::UGenRegistry) -> SynthDef {
    dsl::compile("synthdef snare = whiteNoise * perc 0.001 0.02", registry)
        .unwrap()
        .remove(0)
}

/// A sustained "lead" tone: a pure sine through its own percussive envelope,
/// no noise UGens involved. Because it has no noise UGen, its output does
/// not depend on spawn order (only noise UGens are reseeded per spawn), so
/// it renders bit-identically regardless of what else was spawned earlier.
fn lead_def(registry: &dsl::UGenRegistry) -> SynthDef {
    dsl::compile(
        "synthdef lead = sinOsc 220.0 0.0 * perc 0.005 0.2",
        registry,
    )
    .unwrap()
    .remove(0)
}

fn new_bus_engine() -> (Engine, NodeId) {
    let mut engine = make_engine(BLOCK);
    let bus = engine
        .graph_mut()
        .add_node(Box::new(ugens::Bus::new(ugens::ChannelCount::Stereo)));
    engine.graph_mut().set_sink(bus);
    (engine, bus)
}

/// Case A: a train of snare hits with nothing spawned after the last one.
/// The tail -- rendered with no further structural changes -- must be
/// silent. Included as the paired control for case B below: it holds both
/// before and after the `prepare()` fix, since nothing perturbs the graph
/// after the last spawn.
#[test]
fn snare_train_tail_is_silent_with_no_further_spawns() {
    let registry = builtin_registry();
    let snare = snare_def(&registry);
    let (mut engine, bus) = new_bus_engine();

    // One snare hit per segment for 8 segments, then silence.
    let mut samples = Vec::new();
    for _ in 0..8 {
        engine.spawn_voice_on_bus(&snare, bus).unwrap();
        engine.prepare();
        samples.extend(render_samples(&mut engine, SEG_BLOCKS));
    }
    samples.extend(render_samples(&mut engine, SEG_BLOCKS * 8));

    let segs = segment_rms(&samples, SEG_SAMPLES);
    assert_eq!(segs.len(), 16);
    for (i, &r) in segs[8..].iter().enumerate() {
        assert_eq!(
            r,
            0.0,
            "tail segment {} should be silent, got rms {r} (all segments: {segs:?})",
            8 + i
        );
    }
}

/// Case B, the defect itself: spawning an unrelated later voice must not
/// resurrect an earlier, already-finished percussive envelope that was
/// never explicitly freed.
///
/// A single snare finishes and goes silent; a single lead note is then
/// spawned much later. From that point on, the mixed output must be
/// bit-identical to a lead-only run with no snare ever spawned -- a
/// finished, unfreed snare voice must contribute nothing further, whether
/// or not it is still sitting in the graph.
#[test]
fn later_spawn_does_not_resurrect_finished_percussive_envelope() {
    let registry = builtin_registry();
    let snare = snare_def(&registry);
    let lead = lead_def(&registry);

    // Lead-only baseline: no snare ever spawned.
    let (mut baseline, bus) = new_bus_engine();
    baseline.spawn_voice_on_bus(&lead, bus).unwrap();
    baseline.prepare();
    let baseline_tail = render_samples(&mut baseline, SEG_BLOCKS * 4);

    // Snare, left to fully decay and never freed, then the same lead spawn.
    let (mut engine, bus) = new_bus_engine();
    engine.spawn_voice_on_bus(&snare, bus).unwrap();
    engine.prepare();
    let gap = render_samples(&mut engine, SEG_BLOCKS * 4); // snare fully decays here

    // Precondition: by the end of the gap the snare has genuinely gone
    // silent (not just "should be done" -- audibly zero).
    let gap_end_rms = rms(&gap[gap.len() - SEG_SAMPLES..]);
    assert_eq!(
        gap_end_rms, 0.0,
        "precondition: snare must have decayed to silence before the lead spawns"
    );

    engine.spawn_voice_on_bus(&lead, bus).unwrap();
    engine.prepare();
    let tail = render_samples(&mut engine, SEG_BLOCKS * 4);

    assert_tails_match(
        &tail,
        &baseline_tail,
        "spawning the lead voice after the snare finished must not change the \
         output vs. a run with no snare at all -- the finished, unfreed snare \
         voice must contribute nothing",
    );
}

/// Dose-response control: the more finished-but-unfreed snare voices are
/// sitting in the graph, the larger the resurrection when an unrelated
/// voice spawns later -- monotonic in snare count. After the fix this
/// entire curve collapses to zero.
#[test]
fn resurrection_energy_is_monotonic_in_leftover_snare_count() {
    let registry = builtin_registry();
    let snare = snare_def(&registry);
    let lead = lead_def(&registry);

    let mut prev_rms = -1.0f32;
    for &count in &[1usize, 2, 4, 8] {
        let (mut engine, bus) = new_bus_engine();
        for _ in 0..count {
            engine.spawn_voice_on_bus(&snare, bus).unwrap();
        }
        engine.prepare();
        render_samples(&mut engine, SEG_BLOCKS * 4); // all snares fully decay

        engine.spawn_voice_on_bus(&lead, bus).unwrap();
        engine.prepare();
        // Look only at the first segment right after the lead spawns, before
        // the lead itself has built up any sustained level, to isolate
        // resurrected snare energy from the lead's own onset.
        let onset = render_samples(&mut engine, SEG_BLOCKS);
        let onset_rms = rms(&onset);

        assert!(
            onset_rms >= prev_rms,
            "resurrection energy should be monotonic in leftover snare count: \
             count {count} gave rms {onset_rms}, previous count gave {prev_rms}"
        );
        prev_rms = onset_rms;
    }
}

/// The minimal case: one snare, one lead note spawned much later, must
/// produce output bit-identical to lead-only. A narrower, single-count
/// restatement of `later_spawn_does_not_resurrect_finished_percussive_envelope`
/// kept separate as the smallest possible acceptance case.
#[test]
fn one_snare_then_one_distant_lead_is_bit_identical_to_lead_only() {
    let registry = builtin_registry();
    let snare = snare_def(&registry);
    let lead = lead_def(&registry);

    let (mut baseline, bus) = new_bus_engine();
    baseline.spawn_voice_on_bus(&lead, bus).unwrap();
    baseline.prepare();
    let baseline_tail = render_samples(&mut baseline, SEG_BLOCKS * 6);

    let (mut engine, bus) = new_bus_engine();
    engine.spawn_voice_on_bus(&snare, bus).unwrap();
    engine.prepare();
    render_samples(&mut engine, SEG_BLOCKS * 6);
    engine.spawn_voice_on_bus(&lead, bus).unwrap();
    engine.prepare();
    let tail = render_samples(&mut engine, SEG_BLOCKS * 6);

    assert_tails_match(&tail, &baseline_tail, "one snare + one distant lead");
}
