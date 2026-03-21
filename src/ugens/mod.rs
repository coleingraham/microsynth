//! Built-in UGens for the microsynth engine.
//!
//! Organized by category:
//! - `math`: Const, BinOp, Neg (arithmetic primitives)
//! - `oscillators`: SinOsc, Saw, Pulse, Tri, Phasor
//! - `bl_oscillators`: BlSaw, BlPulse, BlTri (band-limited via polyBLEP)
//! - `noise`: WhiteNoise, PinkNoise
//! - `filters`: OnePole, BiquadLPF, BiquadHPF, BiquadBPF, CombFilter, GVerb, Compressor
//! - `envelopes`: Line, XLine, Perc, ExpPerc, ASR, ADSR
//! - `delay`: Delay, FeedbackDelay
//! - `distortion`: SoftClip, Overdrive
//! - `modulation`: Chorus, Flanger, Phaser
//! - `fm`: FmOsc (two-operator FM synthesis)
//! - `stereo`: StereoWidth, PingPongDelay
//! - `bitcrush`: Bitcrusher (sample rate / bit depth reduction)
//! - `utility`: Pan2, Mix, SampleAndHold, Impulse, Lag, Clip
//! - `playbuf`: PlayBuf (sample playback)
//! - `wavetable`: WaveTable (wavetable oscillator)
//! - `physical`: Pluck (Karplus-Strong), Bowed (waveguide bowed string)

pub mod bitcrush;
pub mod bl_oscillators;
pub mod bus;
pub mod delay;
pub mod distortion;
pub mod envelopes;
pub mod filters;
pub mod fm;
pub mod math;
pub mod modulation;
pub mod noise;
pub mod oscillators;
pub mod physical;
pub mod playbuf;
pub(crate) mod rng;
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
pub use math::*;
pub use modulation::*;
pub use noise::*;
pub use oscillators::*;
pub use physical::*;
pub use playbuf::*;
pub use stereo::*;
pub use utility::*;
pub use wavetable::*;

use crate::context::Rate;
use crate::dsl::compiler::{UGenRegistry};
use crate::node::{InputSpec, OutputSpec};
use alloc::boxed::Box;

/// Register all built-in UGens with a DSL registry.
///
/// This gives the DSL access to every standard UGen via its camelCase name:
/// `sinOsc`, `saw`, `pulse`, `tri`, `whiteNoise`, `pinkNoise`,
/// `onePole`, `lpf`, `hpf`, `bpf`, `line`, `asr`,
/// `delay`, `pan2`, `mix`, `sampleAndHold`.
pub fn register_builtins(reg: &mut UGenRegistry) {
    // -- Oscillators --
    reg.register(
        "sinOsc",
        || Box::new(SinOsc::new()),
        &[
            InputSpec { name: "freq", rate: Rate::Audio },
            InputSpec { name: "phase", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "saw",
        || Box::new(Saw::new()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "pulse",
        || Box::new(Pulse::new()),
        &[
            InputSpec { name: "freq", rate: Rate::Audio },
            InputSpec { name: "width", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "tri",
        || Box::new(Tri::new()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "phasor",
        || Box::new(Phasor::new()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Band-limited Oscillators (polyBLEP) --
    reg.register(
        "blSaw",
        || Box::new(BlSaw::new()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "blPulse",
        || Box::new(BlPulse::new()),
        &[
            InputSpec { name: "freq", rate: Rate::Audio },
            InputSpec { name: "width", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "blTri",
        || Box::new(BlTri::new()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Physical Models --
    reg.register(
        "pluck",
        || Box::new(Pluck::new()),
        &[
            InputSpec { name: "freq", rate: Rate::Audio },
            InputSpec { name: "decay", rate: Rate::Audio },
            InputSpec { name: "trig", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "bowed",
        || Box::new(Bowed::new()),
        &[
            InputSpec { name: "freq", rate: Rate::Audio },
            InputSpec { name: "pressure", rate: Rate::Audio },
            InputSpec { name: "position", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Noise --
    reg.register(
        "whiteNoise",
        || Box::new(WhiteNoise::new()),
        &[],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "pinkNoise",
        || Box::new(PinkNoise::new()),
        &[],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Filters --
    reg.register(
        "onePole",
        || Box::new(OnePole::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "coeff", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "lpf",
        || Box::new(BiquadLPF::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "freq", rate: Rate::Audio },
            InputSpec { name: "q", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "hpf",
        || Box::new(BiquadHPF::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "freq", rate: Rate::Audio },
            InputSpec { name: "q", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "bpf",
        || Box::new(BiquadBPF::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "freq", rate: Rate::Audio },
            InputSpec { name: "q", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Comb filter --
    reg.register(
        "combFilter",
        || Box::new(CombFilter::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "delay", rate: Rate::Audio },
            InputSpec { name: "feedback", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- GVerb (Schroeder reverb) --
    reg.register(
        "gverb",
        || Box::new(GVerb::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "roomsize", rate: Rate::Audio },
            InputSpec { name: "damping", rate: Rate::Audio },
            InputSpec { name: "wet", rate: Rate::Audio },
            InputSpec { name: "dry", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Envelopes --
    reg.register(
        "line",
        || Box::new(Line::new()),
        &[
            InputSpec { name: "start", rate: Rate::Audio },
            InputSpec { name: "end", rate: Rate::Audio },
            InputSpec { name: "dur", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "xLine",
        || Box::new(XLine::new()),
        &[
            InputSpec { name: "start", rate: Rate::Audio },
            InputSpec { name: "end", rate: Rate::Audio },
            InputSpec { name: "dur", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "perc",
        || Box::new(Perc::new()),
        &[
            InputSpec { name: "attack", rate: Rate::Audio },
            InputSpec { name: "release", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "expPerc",
        || Box::new(ExpPerc::new()),
        &[
            InputSpec { name: "attack", rate: Rate::Audio },
            InputSpec { name: "release", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "asr",
        || Box::new(ASR::new()),
        &[
            InputSpec { name: "gate", rate: Rate::Audio },
            InputSpec { name: "attack", rate: Rate::Audio },
            InputSpec { name: "release", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "adsr",
        || Box::new(ADSR::new()),
        &[
            InputSpec { name: "gate", rate: Rate::Audio },
            InputSpec { name: "attack", rate: Rate::Audio },
            InputSpec { name: "decay", rate: Rate::Audio },
            InputSpec { name: "sustain", rate: Rate::Audio },
            InputSpec { name: "release", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Delay --
    reg.register(
        "delay",
        || Box::new(Delay::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "time", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "feedbackDelay",
        || Box::new(FeedbackDelay::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "time", rate: Rate::Audio },
            InputSpec { name: "feedback", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Utility --
    reg.register(
        "pan2",
        || Box::new(Pan2::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "pos", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "left", rate: Rate::Audio }],
    );
    reg.register(
        "mix",
        || Box::new(Mix::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "sampleAndHold",
        || Box::new(SampleAndHold::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "trig", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "impulse",
        || Box::new(Impulse::new()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "lag",
        || Box::new(Lag::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "time", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "clip",
        || Box::new(Clip::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "lo", rate: Rate::Audio },
            InputSpec { name: "hi", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Compressor --
    reg.register(
        "compressor",
        || Box::new(Compressor::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "sidechain", rate: Rate::Audio },
            InputSpec { name: "threshold", rate: Rate::Audio },
            InputSpec { name: "ratio", rate: Rate::Audio },
            InputSpec { name: "attack", rate: Rate::Audio },
            InputSpec { name: "release", rate: Rate::Audio },
            InputSpec { name: "makeup", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- FM Synthesis --
    reg.register(
        "fmOsc",
        || Box::new(FmOsc::new()),
        &[
            InputSpec { name: "freq", rate: Rate::Audio },
            InputSpec { name: "ratio", rate: Rate::Audio },
            InputSpec { name: "index", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Modulation (Chorus, Flanger, Phaser) --
    reg.register(
        "chorus",
        || Box::new(Chorus::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "rate", rate: Rate::Audio },
            InputSpec { name: "depth", rate: Rate::Audio },
            InputSpec { name: "mix", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "flanger",
        || Box::new(Flanger::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "rate", rate: Rate::Audio },
            InputSpec { name: "depth", rate: Rate::Audio },
            InputSpec { name: "feedback", rate: Rate::Audio },
            InputSpec { name: "mix", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "phaser",
        || Box::new(Phaser::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "rate", rate: Rate::Audio },
            InputSpec { name: "depth", rate: Rate::Audio },
            InputSpec { name: "feedback", rate: Rate::Audio },
            InputSpec { name: "mix", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Stereo Effects --
    reg.register(
        "stereoWidth",
        || Box::new(StereoWidth::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "width", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "pingPongDelay",
        || Box::new(PingPongDelay::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "time", rate: Rate::Audio },
            InputSpec { name: "feedback", rate: Rate::Audio },
            InputSpec { name: "mix", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Bitcrusher --
    reg.register(
        "bitcrusher",
        || Box::new(Bitcrusher::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "bits", rate: Rate::Audio },
            InputSpec { name: "downsample", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Distortion --
    reg.register(
        "softClip",
        || Box::new(SoftClip::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "drive", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "overdrive",
        || Box::new(Overdrive::new()),
        &[
            InputSpec { name: "in", rate: Rate::Audio },
            InputSpec { name: "drive", rate: Rate::Audio },
            InputSpec { name: "tone", rate: Rate::Audio },
            InputSpec { name: "mix", rate: Rate::Audio },
        ],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Bus / Routing --
    reg.register(
        "audioIn",
        || Box::new(AudioIn),
        &[InputSpec { name: "in", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Wavetable --
    reg.register(
        "waveTable",
        || Box::new(WaveTable::new()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );

    // -- Pre-built wavetable oscillators --
    reg.register(
        "sinTable",
        || Box::new(sine_table()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "sawTable",
        || Box::new(saw_table()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "triTable",
        || Box::new(tri_table()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
    reg.register(
        "squareTable",
        || Box::new(square_table()),
        &[InputSpec { name: "freq", rate: Rate::Audio }],
        &[OutputSpec { name: "out", rate: Rate::Audio }],
    );
}
