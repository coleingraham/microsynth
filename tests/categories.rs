//! UGen category tags: every registered UGen carries a category from its
//! `spec()`, exposed on the registry entry for downstream tooling.

use microsynth::UGenCategory;
use microsynth::UGenCategory::*;
use microsynth::dsl::compiler::UGenRegistry;

mod common;
use common::builtin_registry as registry;

fn category_of(reg: &UGenRegistry, name: &str) -> UGenCategory {
    reg.entry(name)
        .unwrap_or_else(|| panic!("{name} not registered"))
        .category
}

#[test]
fn registered_ugens_have_expected_categories() {
    let reg = registry();
    // (DSL name, expected category) — a representative row per category, plus
    // the whole excitation / filter / envelope surface the Source shell keys on.
    let cases: &[(&str, UGenCategory)] = &[
        // Oscillators (incl. band-limited, FM, wavetable).
        ("sinOsc", Oscillator),
        ("saw", Oscillator),
        ("pulse", Oscillator),
        ("tri", Oscillator),
        ("phasor", Oscillator),
        ("blSaw", Oscillator),
        ("blPulse", Oscillator),
        ("blTri", Oscillator),
        ("fmOsc", Oscillator),
        ("waveTable", Oscillator),
        ("sinTable", Oscillator),
        // Physical.
        ("pluck", Physical),
        ("bowed", Physical),
        // Noise.
        ("whiteNoise", Noise),
        ("pinkNoise", Noise),
        // Filters (biquads via macro, plus onepole/comb/reverb/dynamics).
        ("onePole", Filter),
        ("lpf", Filter),
        ("hpf", Filter),
        ("bpf", Filter),
        ("notch", Filter),
        ("combFilter", Filter),
        ("gverb", Filter),
        ("compressor", Filter),
        // Envelopes.
        ("line", Envelope),
        ("xLine", Envelope),
        ("perc", Envelope),
        ("expPerc", Envelope),
        ("asr", Envelope),
        ("adsr", Envelope),
        // Effects.
        ("delay", Effect),
        ("feedbackDelay", Effect),
        ("softClip", Effect),
        ("overdrive", Effect),
        ("waveFolder", Effect),
        ("chorus", Effect),
        ("flanger", Effect),
        ("phaser", Effect),
        ("freqShift", Effect),
        ("stereoWidth", Effect),
        ("pingPongDelay", Effect),
        ("bitcrusher", Effect),
        ("spectralFreeze", Effect),
        ("convolution", Effect),
        // Utility / routing.
        ("pan2", Utility),
        ("mix", Utility),
        ("lag", Utility),
        ("clip", Utility),
        ("impulse", Utility),
        ("lfo", Utility),
        ("audioIn", Utility),
    ];
    for &(name, expected) in cases {
        assert_eq!(category_of(&reg, name), expected, "category of {name}");
    }
}

#[test]
fn every_registered_ugen_has_a_category() {
    // Smoke check: every registered entry is reachable and its category is one
    // of the known variants (the type guarantees this, but this also asserts
    // the registry is non-empty and `iter()` works).
    let reg = registry();
    let mut count = 0;
    for (name, entry) in reg.iter() {
        // Touch the category so the field is genuinely exercised.
        let _ = entry.category;
        assert!(!name.is_empty());
        count += 1;
    }
    assert!(count > 40, "expected the full builtin set, got {count}");
}
