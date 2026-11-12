//! Sample playback data: the uploaded sample audio, the compiled instrument
//! programs that play it, and the resampling kernels.
//!
//! The engine never decodes an audio file and never reads an instrument
//! description. A host decodes each sample to f32 PCM and uploads it
//! ([`pcm`], `docs/sample-bank-format.md`), compiles the instrument spec to
//! a program ([`program`], `docs/sample-program-format.md`) and uploads that;
//! a `sampleVoice` node ([`crate::ugens::sample_voice`]) is bound to a
//! registered program by id at graph-build time, the way a coefficient-table
//! node is bound to a table.

pub mod bank;
pub mod interp;
pub mod pcm;
pub mod program;

pub use bank::{BoundProgram, ProgramBindError, ProgramId, SampleProgramBank};
pub use interp::Interp;
pub use pcm::{ContentHash, PcmCodecError, PcmId, SamplePcm, SamplePcmBank};
pub use program::{ProgramCodecError, Region, SampleProgram};
