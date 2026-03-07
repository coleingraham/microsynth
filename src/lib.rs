#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod buffer;
pub mod context;
pub mod dsl;
pub mod engine;
pub mod graph;
pub mod musical_time;
pub mod node;
pub mod sample;
pub mod scheduler;
pub mod synthdef;
pub mod tuning;
pub mod ugens;
#[cfg(target_arch = "wasm32")]
pub mod web;

// Re-export core types for convenience.
pub use buffer::{AudioBuffer, Block, MAX_BLOCK_SIZE};
pub use context::{ProcessContext, Rate};
pub use engine::{Engine, EngineConfig};
pub use graph::AudioGraph;
pub use musical_time::{MusicalPosition, TimeConfig};
pub use node::{InputSpec, NodeId, OutputSpec, UGen, UGenSpec};
pub use sample::{Sample, SampleBank, SampleId};
pub use scheduler::{EventAction, Scheduler, VoiceId};
pub use synthdef::{Synth, SynthDef, SynthDefBuilder, SynthParam};
pub use tuning::{TuningTable, apply_cents, hz_to_midi_12tet, midi_to_hz_12tet};
pub use ugens::register_builtins;
