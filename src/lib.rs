#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod buffer;
pub mod coeff_table;
pub mod context;
pub mod curve;
pub mod dsl;
pub mod engine;
pub mod graph;
#[cfg(feature = "ir")]
pub mod ir;
pub mod musical_sequence;
pub mod musical_time;
pub mod node;
pub mod routing;
pub mod sample;
pub mod scheduler;
pub mod spectral;
pub mod synthdef;
pub mod tuning;
pub mod ugens;
pub mod voice;
// Not gated on `target_arch = "wasm32"`: the raw `#[no_mangle] extern "C"`
// exports in this module don't depend on wasm-bindgen (only the `WebSynth`
// class and its neighbors do, gated separately below by `feature = "web"`),
// and compiling them on every target lets the native test suite drive the
// raw C-ABI directly instead of only exercising it by proxy.
pub mod web;

// Re-export core types for convenience.
pub use buffer::{AudioBuffer, Block, MAX_BLOCK_SIZE};
pub use coeff_table::{CoeffTable, CoeffTableBank, CoeffTableBankError, PitchEntry, TableId};
pub use context::{ProcessContext, Rate};
pub use curve::{GlideShape, GlideSpace, glide_fraction};
pub use engine::{Engine, EngineConfig};
pub use graph::AudioGraph;
pub use musical_sequence::schedule_musical_glides;
pub use musical_time::{MusicalGlideSegment, MusicalPosition, SampleTimeGlide, TimeConfig};
pub use node::{InputSpec, NodeId, OutputSpec, UGen, UGenCategory, UGenSpec};
pub use routing::{BusId, EffectId, RoutingGraph};
pub use sample::{Sample, SampleBank, SampleId};
pub use scheduler::{EventAction, Scheduler, VoiceId};
pub use spectral::{Complex, StftProcessor, WindowType};
pub use synthdef::{Synth, SynthDef, SynthDefBuilder, SynthParam};
pub use tuning::{TuningTable, apply_cents, hz_to_midi_12tet, midi_to_hz_12tet};
pub use ugens::{register_builtins, register_table_bound_builtins};
pub use voice::{
    Admission, LegatoVoice, StealPolicy, Transition, VoiceAllocator, classify_transition,
};
