//! Regression coverage for the shared-seed noise-coherence mechanism:
//! `AudioGraph::reseed_node_noise` forcing multiple noise-bearing voices to
//! the same PRNG seed so their noise streams are sample-for-sample
//! identical, rather than independent.
//!
//! This property is what a crossfaded-voice render (e.g. K partialsNoise
//! voices standing in for one source, summed with time-varying gain
//! envelopes) relies on: independent per-voice noise streams would sum as
//! the RMS of K uncorrelated sources (~sqrt(K)x a single voice's RMS,
//! perceptibly duller/quieter than intended), while a shared seed makes
//! every voice's noise identical, so the sum is exactly Kx one voice's
//! signal -- coherent, not merely correlated. Before this test, that claim
//! was verified by hand (bit-exact, per the QA report) with nothing in the
//! suite guarding it: `AudioGraph::reseed_node_noise` had no test anywhere
//! outside its own definition.
//!
//! Uses `WhiteNoise` rather than `PartialsNoise` -- same mechanism
//! (`UGen::reseed_noise`, driven through `AudioGraph::reseed_node_noise`),
//! no coefficient-table setup needed to exercise it.

use microsynth::ugens::{Bus, ChannelCount, WhiteNoise};
use microsynth::{Engine, EngineConfig, SynthDef, SynthDefBuilder};

const SAMPLE_RATE: f32 = 44100.0;
const BLOCK_SIZE: usize = 128;
/// Enough samples that the *independent*-seed case's sqrt(K)x approximation
/// (a statistical property, unlike the shared-seed case's exact Kx scaling)
/// converges tightly.
const NUM_BLOCKS: usize = 40;
const VOICE_COUNT: usize = 3;
const SHARED_SEED: u32 = 0x1234_5678;

fn white_noise_def() -> SynthDef {
    let mut b = SynthDefBuilder::new("white_noise_voice");
    let idx = b.add_node(|| Box::new(WhiteNoise::new()));
    b.set_output(idx);
    b.build()
}

fn rms(samples: &[f32]) -> f32 {
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// Render one isolated `WhiteNoise` voice reseeded to `seed`, return its RMS.
fn single_voice_rms(seed: u32) -> f32 {
    let def = white_noise_def();
    let mut engine = Engine::new(EngineConfig {
        sample_rate: SAMPLE_RATE,
        block_size: BLOCK_SIZE,
    });
    let synth = engine.instantiate_synthdef(&def);
    let reseeded = engine
        .graph_mut()
        .reseed_node_noise(synth.output_node(), seed);
    assert!(
        reseeded,
        "reseed_node_noise should apply to a freshly instantiated voice"
    );
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();
    let out = engine.render_offline(NUM_BLOCKS);
    rms(&out[0])
}

/// Render `k` `WhiteNoise` voices summed through a mono `Bus`, return the
/// bus output's RMS. `shared_seed`: `Some(seed)` forces every voice to that
/// seed via `reseed_node_noise` (the coherence fix under test); `None`
/// leaves each voice at its normal `instantiate_synthdef`-derived default
/// seed (distinct per voice, by spawn order -- the pre-fix / independent
/// baseline).
fn k_voices_bus_rms(k: usize, shared_seed: Option<u32>) -> f32 {
    let def = white_noise_def();
    let mut engine = Engine::new(EngineConfig {
        sample_rate: SAMPLE_RATE,
        block_size: BLOCK_SIZE,
    });
    let bus_id = engine
        .graph_mut()
        .add_node(Box::new(Bus::new(ChannelCount::Mono)));

    for i in 0..k {
        let synth = engine.instantiate_synthdef(&def);
        if let Some(seed) = shared_seed {
            let reseeded = engine
                .graph_mut()
                .reseed_node_noise(synth.output_node(), seed);
            assert!(reseeded, "voice {i}: reseed_node_noise should apply");
        }
        engine.graph_mut().connect(synth.output_node(), bus_id, i);
    }

    engine.graph_mut().set_sink(bus_id);
    engine.prepare();
    let out = engine.render_offline(NUM_BLOCKS);
    rms(&out[0])
}

/// The positive case: voices forced to a shared seed via
/// `reseed_node_noise` produce bit-identical noise streams, so their sum
/// through a `Bus` is exactly `VOICE_COUNT`x a single voice's signal --
/// this checks that at the RMS level, tightly (no statistical slack needed:
/// identical streams summed is a deterministic scaling, not an
/// approximation).
#[test]
fn shared_seed_voices_sum_to_exactly_k_times_one_voice_rms() {
    let one = single_voice_rms(SHARED_SEED);
    let summed = k_voices_bus_rms(VOICE_COUNT, Some(SHARED_SEED));
    let expected = one * VOICE_COUNT as f32;

    assert!(
        (summed - expected).abs() / expected < 0.01,
        "shared-seed sum RMS should be {VOICE_COUNT}x one voice's RMS ({one}) = {expected}, \
         got {summed} -- coherent (identical-stream) summation is exact scaling, not an \
         approximation, so this should hold tightly"
    );
}

/// The negative case: without forcing a shared seed, each voice keeps its
/// own `instantiate_synthdef`-derived default seed (distinct per spawn
/// order), so the streams are independent and sum as uncorrelated
/// sources -- RMS ~ sqrt(VOICE_COUNT)x one voice's, and in particular
/// nowhere near VOICE_COUNT's worth. This is the failure mode the shared
/// seed exists to avoid, and the case that would catch a future
/// `channels_ir`/export-script node-layout change that stopped calling
/// `reseed_node_noise` from silently degrading a crossfade back to duller,
/// uncorrelated noise.
#[test]
fn independent_seed_voices_sum_as_uncorrelated_not_k_times() {
    let one = single_voice_rms(SHARED_SEED); // reference amplitude only
    let summed = k_voices_bus_rms(VOICE_COUNT, None);
    let k_times = one * VOICE_COUNT as f32;
    let sqrt_k_times = one * (VOICE_COUNT as f32).sqrt();

    assert!(
        summed < 0.75 * k_times,
        "independent-seed sum RMS ({summed}) should be well below {VOICE_COUNT}x one voice's \
         ({k_times}) -- this close to it would mean the voices' noise streams weren't actually \
         independent"
    );
    // Generous tolerance: this is a statistical approximation (uncorrelated
    // sum), unlike the exact scaling in the shared-seed case above.
    assert!(
        (summed - sqrt_k_times).abs() / sqrt_k_times < 0.3,
        "independent-seed sum RMS ({summed}) should be in the ballpark of sqrt({VOICE_COUNT})x \
         one voice's ({sqrt_k_times})"
    );
}
