#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod buffer;
pub mod context;
pub mod engine;
pub mod graph;
pub mod node;
pub mod synthdef;

// Re-export core types for convenience.
pub use buffer::{AudioBuffer, Block, MAX_BLOCK_SIZE};
pub use context::{ProcessContext, Rate};
pub use engine::{Engine, EngineConfig};
pub use graph::AudioGraph;
pub use node::{InputSpec, NodeId, OutputSpec, UGen, UGenSpec};
pub use synthdef::{Synth, SynthDef, SynthDefBuilder};
