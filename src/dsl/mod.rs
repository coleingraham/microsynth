//! Text-based DSL for defining synthesis graphs, effects, and routing.
//!
//! The microsynth DSL uses a Haskell-inspired syntax for declaring synthesis
//! graphs (SynthDefs), audio buses, and signal routing chains. It compiles
//! to the engine's internal audio graph representation.
//!
//! # SynthDef Declarations
//!
//! A SynthDef defines a reusable synthesis template. Syntax:
//!
//! ```text
//! synthdef NAME PARAM=DEFAULT ... = BODY
//! ```
//!
//! Parameters are named with default values. The body is an expression whose
//! result is the SynthDef's audio output. Parameters can be changed at runtime
//! with optional glide/portamento for smooth transitions.
//!
//! ```text
//! -- A simple sine oscillator
//! synthdef tone freq=440.0 amp=0.5 =
//!   sinOsc freq 0.0 * amp
//!
//! -- Subtractive synth with filter and envelope
//! synthdef pad freq=440.0 gate=1.0 amp=0.5 =
//!   let osc = saw freq
//!   let env = asr gate 0.3 1.0
//!   let filt = lpf osc (freq * 4.0) 2.0
//!   filt * env * amp
//!
//! -- Percussive hi-hat
//! synthdef hihat amp=0.3 =
//!   let env = perc 0.001 0.08
//!   let sig = whiteNoise
//!   let filt = hpf sig 8000.0 1.0
//!   filt * amp * env
//! ```
//!
//! # Expressions
//!
//! ## Numeric Literals
//!
//! Floating-point numbers: `440.0`, `0.5`, `42` (integer form is also valid).
//!
//! ## Variables
//!
//! Reference parameters or let-bound names by identifier: `freq`, `osc`, `env`.
//!
//! ## Function Application
//!
//! UGens are called by juxtaposition (whitespace-separated positional arguments):
//!
//! ```text
//! sinOsc freq 0.0        -- sinOsc with freq and phase arguments
//! lpf sig cutoff q       -- low-pass filter with three arguments
//! whiteNoise             -- zero-argument UGens need no args
//! ```
//!
//! ## Arithmetic Operators
//!
//! Standard arithmetic with conventional precedence:
//!
//! - `*`, `/` — multiplication, division (higher precedence)
//! - `+`, `-` — addition, subtraction (lower precedence)
//! - `-x` — unary negation (highest precedence)
//!
//! ```text
//! osc * env * amp            -- multiply signals together
//! freq + 100.0               -- offset a frequency
//! dry * (1.0 - wet) + sig * wet  -- wet/dry crossfade
//! ```
//!
//! ## Parentheses
//!
//! Use parentheses to override precedence or group sub-expressions:
//!
//! ```text
//! lpf osc (freq * 4.0) 2.0   -- freq * 4.0 is a single argument to lpf
//! (a + b) * c                 -- addition before multiplication
//! ```
//!
//! # Let-Bindings
//!
//! Bind intermediate results to names for readability and reuse.
//!
//! ## Statement-level (newline or `;` separated)
//!
//! ```text
//! synthdef example freq=440.0 =
//!   let osc = sinOsc freq 0.0
//!   let env = perc 0.01 0.5
//!   osc * env
//! ```
//!
//! ## Inline with `in`
//!
//! ```text
//! synthdef example = let x = 3.0; y = 4.0 in x + y
//! ```
//!
//! # Comments
//!
//! Line comments start with `--` and extend to the end of the line:
//!
//! ```text
//! -- This is a comment
//! synthdef test = 1.0  -- inline comment
//! ```
//!
//! # Built-in UGens
//!
//! ## Oscillators
//!
//! | Name | Inputs | Description |
//! |------|--------|-------------|
//! | `sinOsc` | `freq` (Hz), `phase` (radians) | Sine wave oscillator \[-1, 1\] |
//! | `saw` | `freq` (Hz) | Sawtooth wave \[-1, 1\] |
//! | `pulse` | `freq` (Hz), `width` (\[0, 1\]) | Pulse/square wave with variable duty cycle |
//! | `tri` | `freq` (Hz) | Triangle wave \[-1, 1\] |
//! | `phasor` | `freq` (Hz) | Ramp oscillator \[0, 1\] |
//!
//! ```text
//! sinOsc 440.0 0.0         -- 440 Hz sine, zero phase
//! saw 110.0                -- 110 Hz sawtooth
//! pulse 220.0 0.5          -- 220 Hz square wave (50% duty)
//! pulse 220.0 0.1          -- 220 Hz narrow pulse (10% duty)
//! tri 330.0                -- 330 Hz triangle
//! ```
//!
//! ## Wavetable Oscillators
//!
//! | Name | Inputs | Description |
//! |------|--------|-------------|
//! | `waveTable` | `freq` (Hz) | Wavetable oscillator (custom waveform) |
//! | `sinTable` | `freq` (Hz) | Pre-built sine wavetable |
//! | `sawTable` | `freq` (Hz) | Pre-built sawtooth wavetable |
//! | `triTable` | `freq` (Hz) | Pre-built triangle wavetable |
//! | `squareTable` | `freq` (Hz) | Pre-built square wavetable |
//!
//! ```text
//! sinTable 440.0           -- wavetable sine at 440 Hz
//! sawTable 110.0           -- wavetable sawtooth at 110 Hz
//! ```
//!
//! ## Noise Generators
//!
//! | Name | Inputs | Description |
//! |------|--------|-------------|
//! | `whiteNoise` | _(none)_ | Uniform white noise \[-1, 1\] |
//! | `pinkNoise` | _(none)_ | 1/f pink noise (Voss-McCartney algorithm) |
//!
//! ```text
//! whiteNoise               -- full-spectrum noise
//! pinkNoise                -- spectrally weighted noise (more bass)
//! ```
//!
//! ## Filters
//!
//! | Name | Inputs | Description |
//! |------|--------|-------------|
//! | `onePole` | `in`, `coeff` (\[-1, 1\]) | One-pole low/high-pass filter |
//! | `lpf` | `in`, `freq` (Hz), `q` | Second-order Butterworth low-pass filter |
//! | `hpf` | `in`, `freq` (Hz), `q` | Second-order high-pass filter |
//! | `bpf` | `in`, `freq` (Hz), `q` | Second-order band-pass filter |
//! | `combFilter` | `in`, `delay` (sec), `feedback` | IIR feedback comb filter |
//! | `gverb` | `in`, `roomsize`, `damping`, `wet`, `dry` | Schroeder reverb (stereo output) |
//!
//! ```text
//! lpf sig 1000.0 1.0       -- low-pass at 1 kHz, Q=1
//! hpf sig 200.0 0.7        -- high-pass at 200 Hz
//! bpf noise 800.0 10.0     -- narrow band-pass at 800 Hz
//! combFilter sig 0.01 0.7  -- comb filter, 10ms delay, 70% feedback
//! gverb sig 0.8 0.5 0.3 0.7  -- reverb: room=0.8, damp=0.5, wet=0.3, dry=0.7
//! ```
//!
//! ## Envelopes
//!
//! | Name | Inputs | Description |
//! |------|--------|-------------|
//! | `line` | `start`, `end`, `dur` (sec) | Linear ramp from start to end, holds at end |
//! | `perc` | `attack` (sec), `release` (sec) | Percussive envelope (attack then release) |
//! | `asr` | `gate`, `attack` (sec), `release` (sec) | Attack-sustain-release (gate-controlled) |
//! | `adsr` | `gate`, `attack`, `decay`, `sustain`, `release` | Full ADSR envelope |
//!
//! ```text
//! perc 0.001 0.1           -- sharp attack, short release
//! asr gate 0.01 0.5        -- gate-controlled with 10ms attack, 500ms release
//! adsr gate 0.01 0.1 0.7 0.5  -- ADSR: 10ms atk, 100ms dec, 70% sus, 500ms rel
//! line 0.0 1.0 2.0         -- ramp from 0 to 1 over 2 seconds
//! ```
//!
//! ## Delay
//!
//! | Name | Inputs | Description |
//! |------|--------|-------------|
//! | `delay` | `in`, `time` (sec, max 5s) | Simple delay line (no feedback) |
//! | `feedbackDelay` | `in`, `time` (sec), `feedback` | Delay with feedback for echo/dub effects |
//!
//! `feedbackDelay` feeds the output back into the delay line:
//! `y[n] = x[n] + feedback * y[n - time]`. Use feedback values
//! between 0.0 and 0.9 for clean repeating echoes; higher values
//! (up to 0.999) produce dub-style runaway repeats.
//!
//! ```text
//! delay sig 0.25                  -- quarter-second delay (no repeats)
//! feedbackDelay sig 0.3 0.5       -- 300ms echo, 50% feedback
//! feedbackDelay sig 0.5 0.7       -- half-second dub delay
//! ```
//!
//! ## Dynamics
//!
//! | Name | Inputs | Description |
//! |------|--------|-------------|
//! | `compressor` | `in`, `sidechain`, `threshold` (dB), `ratio`, `attack` (sec), `release` (sec), `makeup` (dB) | Feed-forward compressor with sidechain |
//!
//! The compressor reduces dynamic range by attenuating signals whose
//! sidechain input exceeds the threshold. The `sidechain` input controls
//! what signal is used for level detection — pass the same signal as `in`
//! for normal compression, or a different signal for sidechain compression
//! (e.g. ducking synths when a kick hits).
//!
//! - `threshold`: level in dB above which compression begins (e.g. -10.0)
//! - `ratio`: compression ratio (e.g. 4.0 = 4:1 compression)
//! - `attack`: how fast the compressor reacts to level increases (seconds)
//! - `release`: how fast the compressor recovers after level drops (seconds)
//! - `makeup`: gain in dB added after compression to restore volume
//!
//! ```text
//! -- Self-sidechain compression (normal compressor)
//! compressor sig sig (0.0 - 10.0) 4.0 0.01 0.1 6.0
//!
//! -- Sidechain compression: duck pads when kick hits
//! -- (in a routing context, kick feeds sidechain input)
//! compressor padSig kickSig (0.0 - 20.0) 8.0 0.001 0.1 0.0
//! ```
//!
//! ### Sidechain Compression with Routing
//!
//! To use sidechain compression in a routing graph, define the compressor
//! effect as a SynthDef with `audioIn` for the main signal, and route the
//! sidechain signal separately:
//!
//! ```text
//! -- Self-compressing effect
//! synthdef busComp threshold=-10.0 ratio=4.0 makeup=6.0 =
//!   let sig = audioIn
//!   compressor sig sig threshold ratio 0.01 0.1 makeup
//!
//! bus drums 2
//! route drums => busComp => main
//! ```
//!
//! ## Utility
//!
//! | Name | Inputs | Description |
//! |------|--------|-------------|
//! | `pan2` | `in`, `pos` (\[-1, 1\]) | Equal-power stereo panner (-1=left, 1=right) |
//! | `mix` | `in` | Sum multichannel signal to mono |
//! | `sampleAndHold` | `in`, `trig` | Sample input value on trigger rising edge |
//! | `impulse` | `freq` (Hz) | Periodic impulse train (1.0 once per period) |
//! | `lag` | `in`, `time` (sec) | Exponential smoothing (low-pass) |
//! | `clip` | `in`, `lo`, `hi` | Hard clamp signal to \[lo, hi\] range |
//!
//! ```text
//! pan2 sig 0.0             -- center pan
//! pan2 sig (0.0 - 0.5)     -- pan left
//! mix stereoSig            -- downmix to mono
//! impulse 4.0              -- trigger 4 times per second
//! lag sig 0.1              -- smooth signal with 100ms lag
//! clip sig (0.0 - 1.0) 1.0 -- clip to [-1, 1]
//! ```
//!
//! ## Bus / Routing
//!
//! | Name | Inputs | Description |
//! |------|--------|-------------|
//! | `audioIn` | _(none in DSL)_ | Receives audio from a bus (for effect SynthDefs) |
//!
//! `audioIn` is used in effect SynthDefs to receive audio from the source bus.
//! See the [Effects & Routing](#effects--routing) section below.
//!
//! # Effects & Routing
//!
//! The DSL supports defining effects as SynthDefs and connecting them via
//! named audio buses. This enables flexible signal routing: source voices
//! feed into buses, buses feed through effects, and effects output to other
//! buses or the main stereo output.
//!
//! ## Defining Effect SynthDefs
//!
//! An effect SynthDef uses `audioIn` to receive audio from its source bus:
//!
//! ```text
//! -- Low-pass filter effect
//! synthdef myFilter cutoff=2000.0 q=1.0 =
//!   let sig = audioIn
//!   lpf sig cutoff q
//!
//! -- Simple echo effect
//! synthdef echo mix=0.3 =
//!   let dry = audioIn
//!   let wet = delay dry 0.25
//!   dry * (1.0 - mix) + wet * mix
//!
//! -- Reverb effect
//! synthdef myReverb wet=0.3 roomsize=0.8 damping=0.5 =
//!   gverb audioIn roomsize damping wet (1.0 - wet)
//! ```
//!
//! ## Bus Declarations
//!
//! Declare named audio buses with a channel count:
//!
//! ```text
//! bus NAME CHANNELS
//! ```
//!
//! A `main` bus (stereo) always exists by default — you do not need to
//! declare it. All inputs to a bus are summed per channel.
//!
//! ```text
//! bus drums 2              -- stereo drum bus
//! bus synths 2             -- stereo synth bus
//! bus fx 2                 -- stereo effects send bus
//! ```
//!
//! ## Route Declarations
//!
//! Define signal routing chains with `=>`:
//!
//! ```text
//! route SOURCE_BUS => EFFECT_SYNTHDEF => TARGET_BUS
//! ```
//!
//! The first and last names are buses; middle names are effect SynthDefs.
//! Longer chains pass through multiple effects and intermediate buses:
//!
//! ```text
//! route raw => preFilter => filtered
//! route filtered => postReverb => main
//! ```
//!
//! ### Fan-out
//!
//! A bus can appear as the source in multiple route declarations, sending
//! its audio to multiple destinations in parallel. This is useful for
//! send effects, parallel processing, and sidechain compression:
//!
//! ```text
//! bus synths 2
//!
//! -- Same source feeds two different effect chains
//! route synths => myFilter => main
//! route synths => myReverb => main
//! ```
//!
//! ## Complete Routing Example
//!
//! ```text
//! -- Define effect SynthDefs
//! synthdef drumFilter cutoff=4000.0 q=0.7 =
//!   let sig = audioIn
//!   lpf sig cutoff q
//!
//! synthdef padReverb wet=0.3 roomsize=0.8 damping=0.5 =
//!   gverb audioIn roomsize damping wet (1.0 - wet)
//!
//! -- Declare buses
//! bus drums 2
//! bus pads 2
//!
//! -- Route: drums get filtered, pads get reverbed, both go to main
//! route drums => drumFilter => main
//! route pads => padReverb => main
//! ```
//!
//! ## Multi-hop Routing Example
//!
//! ```text
//! synthdef preFilter cutoff=3000.0 =
//!   lpf audioIn cutoff 1.0
//!
//! synthdef postReverb wet=0.4 roomsize=0.7 damping=0.5 =
//!   gverb audioIn roomsize damping wet (1.0 - wet)
//!
//! bus raw 2
//! bus filtered 2
//!
//! -- Signal flows: raw => preFilter => filtered => postReverb => main
//! route raw => preFilter => filtered
//! route filtered => postReverb => main
//! ```
//!
//! # Rust API
//!
//! ## Compiling SynthDefs
//!
//! ```rust,ignore
//! use microsynth::dsl::{compile, UGenRegistry};
//! use microsynth::ugens::register_builtins;
//!
//! let mut registry = UGenRegistry::new();
//! register_builtins(&mut registry);
//!
//! let source = "
//!   synthdef pad freq=440.0 amp=0.5 =
//!     let osc = sinOsc freq 0.0
//!     let env = perc 0.01 0.5
//!     osc * env * amp
//! ";
//!
//! let defs = compile(source, &registry).unwrap();
//! ```
//!
//! ## Compiling with Routing
//!
//! ```rust,ignore
//! use microsynth::dsl::{compile_with_routing, UGenRegistry};
//! use microsynth::ugens::register_builtins;
//!
//! let mut registry = UGenRegistry::new();
//! register_builtins(&mut registry);
//!
//! let source = "
//!   synthdef myFx cutoff=1000.0 q=1.0 =
//!     let sig = audioIn
//!     lpf sig cutoff q
//!
//!   bus synths 2
//!   route synths => myFx => main
//! ";
//!
//! let (defs, routing) = compile_with_routing(source, &registry).unwrap();
//! ```
//!
//! ## Error Handling
//!
//! All compilation functions return [`DslError`], which wraps:
//! - [`LexError`] — invalid tokens or characters
//! - [`ParseError`] — syntax errors (with line/column position)
//! - [`CompileError`] — semantic errors (unknown UGen, wrong argument count, etc.)

pub mod ast;
pub mod compiler;
pub mod lexer;
pub mod parser;

pub use compiler::{compile_program, compile_routing, compile_synthdef, CompileError, UGenEntry, UGenRegistry};
pub use lexer::LexError;
pub use parser::ParseError;

use crate::routing::RoutingGraph;
use crate::synthdef::SynthDef;
use alloc::vec::Vec;
use core::fmt;

/// Unified error type for the DSL pipeline.
#[derive(Debug, Clone)]
pub enum DslError {
    Lex(LexError),
    Parse(ParseError),
    Compile(CompileError),
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslError::Lex(e) => write!(f, "{e}"),
            DslError::Parse(e) => write!(f, "{e}"),
            DslError::Compile(e) => write!(f, "{e}"),
        }
    }
}

impl From<LexError> for DslError {
    fn from(e: LexError) -> Self {
        DslError::Lex(e)
    }
}

impl From<ParseError> for DslError {
    fn from(e: ParseError) -> Self {
        DslError::Parse(e)
    }
}

impl From<CompileError> for DslError {
    fn from(e: CompileError) -> Self {
        DslError::Compile(e)
    }
}

/// Parse and compile DSL source into SynthDefs.
///
/// This is the main entry point: tokenize → parse → compile.
/// Bus and route declarations are ignored; use `compile_with_routing`
/// to also produce a routing graph.
pub fn compile(source: &str, registry: &UGenRegistry) -> Result<Vec<SynthDef>, DslError> {
    let tokens = lexer::tokenize(source)?;
    let mut parser = parser::Parser::new(tokens);
    let program = parser.parse_program()?;
    let defs = compile_program(&program, registry)?;
    Ok(defs)
}

/// Parse and compile DSL source into SynthDefs and a RoutingGraph.
///
/// Returns `(synthdefs, routing_graph)`. The routing graph contains
/// bus and route declarations from the source, with effect references
/// resolved against the compiled SynthDefs.
pub fn compile_with_routing(
    source: &str,
    registry: &UGenRegistry,
) -> Result<(Vec<SynthDef>, RoutingGraph), DslError> {
    let tokens = lexer::tokenize(source)?;
    let mut parser = parser::Parser::new(tokens);
    let program = parser.parse_program()?;
    let defs = compile_program(&program, registry)?;
    let routing = compile_routing(&program, &defs)?;
    Ok((defs, routing))
}
