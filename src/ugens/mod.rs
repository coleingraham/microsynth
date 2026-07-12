//! Built-in UGens for the microsynth engine.
//!
//! Organized by category:
//! - `math`: Const, BinOp, Neg (arithmetic primitives)
//! - `oscillators`: SinOsc, Saw, Pulse, Tri, Phasor
//! - `bl_oscillators`: BlSaw, BlPulse, BlTri (band-limited via polyBLEP)
//! - `noise`: WhiteNoise, PinkNoise
//! - `filters`: OnePole, BiquadLPF, BiquadHPF, BiquadBPF, BiquadNotch, AllpassFilter, CombFilter, GVerb, Compressor
//! - `envelopes`: Line, XLine, Perc, ExpPerc, ASR, ADSR
//! - `delay`: Delay, FeedbackDelay
//! - `distortion`: SoftClip, Overdrive, WaveFolder
//! - `modulation`: Chorus, Flanger, Phaser
//! - `fm`: FmOsc (two-operator FM synthesis with self-feedback)
//! - `freqshift`: FreqShift (Hilbert transform frequency shifter)
//! - `lfo`: Lfo (multi-shape unipolar LFO)
//! - `stereo`: StereoWidth, PingPongDelay
//! - `bitcrush`: Bitcrusher (sample rate / bit depth reduction)
//! - `utility`: Pan2, Mix, SampleAndHold, Impulse, Lag, Clip
//! - `playbuf`: PlayBuf (sample playback)
//! - `wavetable`: WaveTable (wavetable oscillator)
//! - `physical`: Pluck (Karplus-Strong), Bowed (waveguide bowed string)
//! - `spectral`: SpectralFreeze, PitchShift, SpectralFilter, SpectralGate, SpectralBlur, Convolution

// Declared first with `#[macro_use]` so the `ugen_spec!` macro it defines is
// in scope for every UGen submodule declared below.
#[macro_use]
mod macros;

pub mod bitcrush;
pub mod bl_oscillators;
pub mod bus;
pub mod delay;
pub mod distortion;
pub mod envelopes;
pub mod filters;
pub mod fm;
pub mod freqshift;
pub mod lfo;
pub mod math;
pub mod modulation;
pub mod noise;
pub mod oscillators;
pub mod physical;
pub mod playbuf;
pub(crate) mod rng;
pub mod spectral;
pub mod stereo;
pub mod utility;
pub mod wavetable;

// Re-export everything for convenience.
pub use bitcrush::*;
pub use bl_oscillators::*;
pub use bus::*;
pub use delay::*;
pub use distortion::*;
pub use envelopes::*;
pub use filters::*;
pub use fm::*;
pub use freqshift::*;
pub use lfo::*;
pub use math::*;
pub use modulation::*;
pub use noise::*;
pub use oscillators::*;
pub use physical::*;
pub use playbuf::*;
pub use spectral::*;
pub use stereo::*;
pub use utility::*;
pub use wavetable::*;

use crate::dsl::compiler::UGenRegistry;
use alloc::boxed::Box;

/// Register all built-in UGens with a DSL registry.
///
/// This gives the DSL access to every standard UGen via its camelCase name:
/// `sinOsc`, `saw`, `pulse`, `tri`, `whiteNoise`, `pinkNoise`,
/// `onePole`, `lpf`, `hpf`, `bpf`, `line`, `asr`,
/// `delay`, `pan2`, `mix`, `sampleAndHold`.
///
/// Port specs are derived from each UGen's own `spec()` via
/// [`UGenRegistry::register_spec`], so the port lists live in exactly one
/// place — the UGen definition — rather than being re-declared here. The DSL
/// name stays explicit because it differs from the UGen's internal
/// `spec().name`, and several DSL names can map to the same UGen type.
pub fn register_builtins(reg: &mut UGenRegistry) {
    // -- Oscillators --
    reg.register_spec("sinOsc", || Box::new(SinOsc::new()));
    reg.register_spec("saw", || Box::new(Saw::new()));
    reg.register_spec("pulse", || Box::new(Pulse::new()));
    reg.register_spec("tri", || Box::new(Tri::new()));
    reg.register_spec("phasor", || Box::new(Phasor::new()));

    // -- Band-limited Oscillators (polyBLEP) --
    reg.register_spec("blSaw", || Box::new(BlSaw::new()));
    reg.register_spec("blPulse", || Box::new(BlPulse::new()));
    reg.register_spec("blTri", || Box::new(BlTri::new()));

    // -- Physical Models --
    reg.register_spec("pluck", || Box::new(Pluck::new()));
    reg.register_spec("bowed", || Box::new(Bowed::new()));

    // -- Noise --
    reg.register_spec("whiteNoise", || Box::new(WhiteNoise::new()));
    reg.register_spec("pinkNoise", || Box::new(PinkNoise::new()));

    // -- Filters --
    reg.register_spec("onePole", || Box::new(OnePole::new()));
    reg.register_spec("lpf", || Box::new(BiquadLPF::new()));
    reg.register_spec("hpf", || Box::new(BiquadHPF::new()));
    reg.register_spec("bpf", || Box::new(BiquadBPF::new()));
    reg.register_spec("notch", || Box::new(BiquadNotch::new()));
    reg.register_spec("allpass", || Box::new(AllpassFilter::new()));
    reg.register_spec("combFilter", || Box::new(CombFilter::new()));
    reg.register_spec("gverb", || Box::new(GVerb::new()));

    // -- Envelopes --
    reg.register_spec("line", || Box::new(Line::new()));
    reg.register_spec("xLine", || Box::new(XLine::new()));
    reg.register_spec("perc", || Box::new(Perc::new()));
    reg.register_spec("expPerc", || Box::new(ExpPerc::new()));
    reg.register_spec("asr", || Box::new(ASR::new()));
    reg.register_spec("adsr", || Box::new(ADSR::new()));

    // -- Delay --
    reg.register_spec("delay", || Box::new(Delay::new()));
    reg.register_spec("feedbackDelay", || Box::new(FeedbackDelay::new()));

    // -- Utility --
    reg.register_spec("pan2", || Box::new(Pan2::new()));
    reg.register_spec("mix", || Box::new(Mix::new()));
    reg.register_spec("sampleAndHold", || Box::new(SampleAndHold::new()));
    reg.register_spec("impulse", || Box::new(Impulse::new()));
    reg.register_spec("lag", || Box::new(Lag::new()));
    reg.register_spec("clip", || Box::new(Clip::new()));

    // -- Compressor --
    reg.register_spec("compressor", || Box::new(Compressor::new()));

    // -- FM Synthesis --
    reg.register_spec("fmOsc", || Box::new(FmOsc::new()));

    // -- Frequency Shifter --
    reg.register_spec("freqShift", || Box::new(FreqShift::new()));

    // -- Modulation (Chorus, Flanger, Phaser) --
    reg.register_spec("chorus", || Box::new(Chorus::new()));
    reg.register_spec("flanger", || Box::new(Flanger::new()));
    reg.register_spec("phaser", || Box::new(Phaser::new()));

    // -- Stereo Effects --
    reg.register_spec("stereoWidth", || Box::new(StereoWidth::new()));
    reg.register_spec("pingPongDelay", || Box::new(PingPongDelay::new()));

    // -- Bitcrusher --
    reg.register_spec("bitcrusher", || Box::new(Bitcrusher::new()));

    // -- Distortion --
    reg.register_spec("softClip", || Box::new(SoftClip::new()));
    reg.register_spec("overdrive", || Box::new(Overdrive::new()));
    reg.register_spec("waveFolder", || Box::new(WaveFolder::new()));

    // -- LFO --
    reg.register_spec("lfo", || Box::new(Lfo::new()));

    // -- Bus / Routing --
    reg.register_spec("audioIn", || Box::new(AudioIn));

    // -- Wavetable --
    reg.register_spec("waveTable", || Box::new(WaveTable::new()));

    // -- Spectral Processing --
    reg.register_spec("spectralFreeze", || Box::new(SpectralFreeze::new()));
    reg.register_spec("pitchShift", || Box::new(PitchShift::new()));
    reg.register_spec("spectralFilter", || Box::new(SpectralFilter::new()));
    reg.register_spec("spectralGate", || Box::new(SpectralGate::new()));
    reg.register_spec("spectralBlur", || Box::new(SpectralBlur::new()));
    reg.register_spec("convolution", || Box::new(Convolution::new()));

    // -- Pre-built wavetable oscillators --
    reg.register_spec("sinTable", || Box::new(sine_table()));
    reg.register_spec("sawTable", || Box::new(saw_table()));
    reg.register_spec("triTable", || Box::new(tri_table()));
    reg.register_spec("squareTable", || Box::new(square_table()));
}
