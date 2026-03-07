#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod buffer;
pub mod context;
pub mod dsl;
pub mod engine;
pub mod graph;
pub mod node;
pub mod sample;
pub mod scheduler;
pub mod synthdef;
pub mod ugens;
#[cfg(target_arch = "wasm32")]
pub mod web;

// Re-export core types for convenience.
pub use buffer::{AudioBuffer, Block, MAX_BLOCK_SIZE};
pub use context::{ProcessContext, Rate};
pub use engine::{Engine, EngineConfig};
pub use graph::AudioGraph;
pub use node::{InputSpec, NodeId, OutputSpec, UGen, UGenSpec};
pub use sample::{Sample, SampleBank, SampleId};
pub use scheduler::{EventAction, Scheduler, VoiceId};
pub use synthdef::{Synth, SynthDef, SynthDefBuilder, SynthParam};
pub use ugens::register_builtins;
