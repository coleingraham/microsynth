//! Done actions key on the synth's output path.
//!
//! `Engine::free_done_synths` frees a synth when a done-reporting node's
//! silence reaches its output (through a `Mul` VCA, a pass-through filter,
//! or both terms of a sum). A one-shot on a side chain, such as an
//! excitation burst into a delay-line string or a pitch envelope into a
//! `freq` port, must not free a voice that is still sounding.

use microsynth::dsl;
use microsynth::*;

mod common;
use common::builtin_registry;

const SR: f32 = 44100.0;
const BLOCK: usize = 64;

fn make_engine() -> Engine {
    Engine::new(EngineConfig {
        sample_rate: SR,
        block_size: BLOCK,
    })
}

fn compile_one(src: &str) -> SynthDef {
    let registry = builtin_registry();
    dsl::compile(src, &registry)
        .unwrap_or_else(|e| panic!("compile failed: {e:?}"))
        .remove(0)
}

/// One instantiated synth rendered as the graph's sink.
struct Rig {
    engine: Engine,
    synth: Synth,
}

impl Rig {
    fn new(src: &str) -> Self {
        let def = compile_one(src);
        let mut engine = make_engine();
        let synth = engine.instantiate_synthdef(&def);
        engine.graph_mut().set_sink(synth.output_node());
        engine.prepare();
        Rig { engine, synth }
    }

    fn blocks_for(secs: f32) -> usize {
        (secs * SR / BLOCK as f32).ceil() as usize
    }

    /// Render `secs` of audio, reaping done synths after every block as a
    /// host does. Returns the peak of the rendered audio and whether the
    /// synth was freed (rendering stops at the free).
    fn run(&mut self, secs: f32) -> (f32, bool) {
        let mut peak = 0.0f32;
        for _ in 0..Self::blocks_for(secs) {
            if let Some(buf) = self.engine.render() {
                for ch in 0..buf.num_channels() {
                    for &s in buf.channel(ch).samples() {
                        peak = peak.max(s.abs());
                    }
                }
            }
            if self.engine.free_done_synths() > 0 {
                return (peak, true);
            }
        }
        (peak, false)
    }

    /// Whether any of the synth's nodes reports done (the rule before the
    /// output-path refinement).
    fn any_node_done(&self) -> bool {
        self.synth
            .node_ids()
            .iter()
            .any(|&id| self.engine.graph().node_is_done(id))
    }

    fn release(&mut self) {
        assert!(self.engine.set_param(&self.synth, "gate", 0.0));
    }
}

#[test]
fn side_chain_excitation_into_feedback_delay_does_not_free_a_held_voice() {
    // Karplus-Strong: a 6 ms noise burst excites a feedback-delay string; the
    // sustained voice is gated by an asr VCA after the string.
    let mut rig = Rig::new(
        "synthdef ks freq=220.0 gate=1.0 = \
         let env = asr gate 0.005 0.05; \
         let burst = whiteNoise * perc 0.001 0.005; \
         feedbackDelay burst (1.0 / freq) 0.995 * env",
    );
    // Well past the burst.
    let (_, freed) = rig.run(0.1);
    assert!(!freed, "the voice was freed while its gate is held");
    assert!(
        rig.any_node_done(),
        "the excitation perc should have finished (otherwise this test proves nothing)"
    );
    // Still sounding after the burst finished.
    let (peak, freed) = rig.run(0.05);
    assert!(!freed);
    assert!(peak > 1e-3, "the string should still ring, peak {peak}");

    // Releasing the asr frees the voice.
    rig.release();
    let (_, freed) = rig.run(0.2);
    assert!(
        freed,
        "the voice should be freed when its VCA envelope finishes"
    );
    assert!(rig.engine.synths().is_empty());
}

#[test]
fn side_chain_excitation_into_comb_filter_does_not_free_a_held_voice() {
    let mut rig = Rig::new(
        "synthdef ksc freq=220.0 gate=1.0 = \
         let env = asr gate 0.005 0.05; \
         combFilter (whiteNoise * expPerc 0.001 0.004) (1.0 / freq) 0.99 * env",
    );
    let (_, freed) = rig.run(0.1);
    assert!(!freed);
    assert!(rig.any_node_done());
    rig.release();
    let (_, freed) = rig.run(0.2);
    assert!(freed);
}

#[test]
fn pitch_drop_xline_on_a_freq_port_does_not_free_the_voice() {
    let mut rig = Rig::new(
        "synthdef drop freq=440.0 gate=1.0 = \
         let env = asr gate 0.005 0.05; \
         sinOsc (freq * xLine 2.0 1.0 0.02) 0.0 * env",
    );
    let (_, freed) = rig.run(0.1);
    assert!(!freed, "the pitch drop finishing must not free the voice");
    assert!(rig.any_node_done(), "the xLine should have finished");
    let (peak, freed) = rig.run(0.05);
    assert!(!freed);
    assert!(peak > 0.5, "the oscillator should still sound, peak {peak}");
    rig.release();
    let (_, freed) = rig.run(0.2);
    assert!(freed);
}

#[test]
fn perc_vca_frees_the_voice_when_it_finishes() {
    // The classic one-shot: `Mul(expPerc, sinOsc)`.
    let mut rig = Rig::new("synthdef hit = sinOsc 440.0 0.0 * expPerc 0.001 0.01");
    let (peak, freed) = rig.run(0.5);
    assert!(peak > 0.1);
    assert!(freed, "the perc VCA finishing should free the voice");
}

#[test]
fn vca_envelope_behind_a_filter_still_frees_the_voice() {
    let mut rig = Rig::new(
        "synthdef filt freq=220.0 gate=1.0 = \
         lpf (saw freq * asr gate 0.005 0.02) 2000.0 0.7",
    );
    let (_, freed) = rig.run(0.1);
    assert!(!freed);
    rig.release();
    let (_, freed) = rig.run(0.2);
    assert!(
        freed,
        "a VCA envelope behind a filter should free the voice"
    );
}

#[test]
fn summed_voices_free_only_when_every_term_is_silent() {
    // Two independently enveloped terms: the short one finishing leaves the
    // long one sounding.
    let mut rig = Rig::new(
        "synthdef two = \
         sinOsc 440.0 0.0 * perc 0.001 0.01 + sinOsc 660.0 0.0 * perc 0.001 0.2",
    );
    let (_, freed) = rig.run(0.05);
    assert!(!freed, "one term finishing must not free the sum");
    assert!(rig.any_node_done());
    let (_, freed) = rig.run(0.5);
    assert!(freed, "both terms finished: the voice should be freed");
}

#[test]
fn pluck_summed_with_a_held_voice_does_not_free_it() {
    let mut rig = Rig::new(
        "synthdef pl freq=220.0 gate=1.0 = \
         let env = asr gate 0.005 0.05; \
         (pluck freq 0.05 1.0 + saw freq * 0.2) * env",
    );
    // Hold long enough for the pluck to fall below its energy threshold.
    let mut pluck_done = false;
    for _ in 0..40 {
        let (_, freed) = rig.run(0.05);
        assert!(!freed, "the pluck finishing must not free the held voice");
        if rig.any_node_done() {
            pluck_done = true;
            break;
        }
    }
    assert!(pluck_done, "the pluck should report done within 2 s");
    rig.release();
    let (_, freed) = rig.run(0.2);
    assert!(freed);
}

/// When every done-reporting node is on the output path, the synth is freed
/// on exactly the block where the old any-node rule freed it.
#[test]
fn defs_whose_done_nodes_are_all_on_the_output_path_free_as_before() {
    let defs = [
        "synthdef a freq=220.0 gate=1.0 amp=0.3 = let env = asr gate 0.01 0.05; saw freq * env * amp",
        "synthdef b freq=220.0 gate=1.0 amp=0.3 = \
         let env = adsr gate 0.01 0.02 0.5 0.05; lpf (pulse freq 0.5) 1600.0 0.7 * env * amp",
        "synthdef c freq=220.0 gate=1.0 = let env = asr gate 0.01 0.05; (saw freq + tri freq) * env",
        "synthdef d freq=220.0 gate=1.0 = hpf (lpf (saw freq * asr gate 0.01 0.05) 3000.0 0.7) 20.0 0.7",
        "synthdef e = let ampenv = expPerc 0.001 0.05; sinOsc 60.0 0.0 * ampenv * 0.9",
    ];
    for src in defs {
        let mut rig = Rig::new(src);
        let has_gate = rig.synth.param_node("gate").is_some();
        let release_block = Rig::blocks_for(0.05);
        let mut freed_at = None;
        for block in 0..Rig::blocks_for(1.0) {
            if has_gate && block == release_block {
                rig.release();
            }
            rig.engine.render();
            let old_rule = rig.any_node_done();
            let freed = rig.engine.free_done_synths() > 0;
            assert_eq!(freed, old_rule, "{src}: block {block}");
            if freed {
                freed_at = Some(block);
                break;
            }
        }
        assert!(freed_at.is_some(), "{src}: never freed");
    }
}
