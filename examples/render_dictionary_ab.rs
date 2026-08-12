//! MOT-637/MOT-642 native render: decode a real exported MOT-634 wire-format
//! dictionary (produced by `motif-soundmatch/python/scripts/export_mot637_ab.py`),
//! register it into a [`CoeffTableBank`], resolve it against a table-bound
//! `partialsNoise` node through the real `IrSynthDef::compile_with_tables` path
//! (the production bank/id-reference mechanism -- not a hand-built `PartialsNoise`
//! the way `render_partials_demo.rs` does), and render it driven by the fit's own
//! per-frame-dominant-pitch performance: both `gain` (H(t)) and `freq` (f0(t))
//! glide per audio block (per-block linear/pitch ramps, matching RFC requirement
//! 6's "linear per-block ramps on ... gain updates" -- extended here to `freq`
//! too, RFC requirement 1's continuous-f0 glide).
//!
//! ## MOT-642: driving the whole fitted performance, not one fixed pitch
//!
//! MOT-637's original version of this example held `freq` fixed at the
//! dictionary's own highest-total-H-mass pitch for the whole render -- the
//! source arpeggio (4 notes) was inaudible even though the dictionary carried
//! every fitted pitch. This version drives `freq` from its own per-block target
//! array exactly the way `gain` already was: **sequential single-voice glide**,
//! chosen over one-voice-per-active-pitch polyphony because the source (a solo
//! vocal line) is monophonic -- one pitch sounds at a time, so a single glided
//! voice is a faithful and much simpler match for the fitted performance than
//! allocating simultaneous voices. The Python driver
//! (`export_mot637_ab.py`) computes, per STFT frame, `argmax` over pitches of
//! `fit.H[:, t]` and emits that frame's winning pitch's f0 (Hz) and its own `H`
//! value as two audio-rate envelopes; this file just glides `freq`/`gain` toward
//! each block's target from those envelopes, reusing exactly the mechanism
//! MOT-637 already validated for `gain`.
//!
//! Not part of the test suite -- run manually, from this crate's root:
//!
//! ```text
//! cargo run --release --example render_dictionary_ab -- \
//!     <dictionary.msct> <gain_env_f32_SRhz.raw> <freq_env_f32_SRhz.raw> \
//!     <sample_rate> <out.wav>
//! ```
//!
//! `<gain_env_f32_SRhz.raw>`/`<freq_env_f32_SRhz.raw>` are raw little-endian f32
//! files, one sample per audio sample at `<sample_rate>` (produced by
//! `channel_export.upsample_gain` on the Python side -- a generic frame-rate ->
//! audio-rate linear-interp helper, reused for both gain and freq). Both must
//! have the same length (the Python driver writes them from the same
//! `num_samples`); this example asserts that rather than silently truncating to
//! the shorter one.

#[path = "common/wav.rs"]
mod wav;

use microsynth::coeff_table::{CoeffTable, CoeffTableBank};
use microsynth::curve::{GlideShape, GlideSpace};
use microsynth::dsl::compiler::UGenRegistry;
use microsynth::ir::{IrNode, IrSynthDef, IrTableBinding, SynthDefClass};
use microsynth::{Engine, EngineConfig};
use std::env;
use std::fs;
use std::path::Path;

fn read_f32_raw(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    assert!(
        bytes.len().is_multiple_of(4),
        "{path:?} length {} is not a multiple of 4",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A single-node IR document: `Param("freq")` and `Param("gain")` feeding a
/// table-bound `partialsNoise` node, bound to `table_id`, as its own output.
/// `default_freq_hz` seeds the `freq` param at the melody's own first target
/// (the caller's `freq_env[0]`) rather than an arbitrary constant, so there is
/// no artificial glide-from-default at the start of block 0.
fn dictionary_ir(table_id: u32, default_freq_hz: f32) -> IrSynthDef {
    IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "mot637_dictionary_ab".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: vec![
            IrNode::Param {
                name: "freq".into(),
                default: default_freq_hz,
            },
            IrNode::Param {
                name: "gain".into(),
                default: 0.0,
            },
            IrNode::UGen {
                kind: "partialsNoise".into(),
                consts: vec![],
            },
        ],
        edges: vec![
            microsynth::ir::IrEdge {
                from: 0,
                to: 2,
                to_input: 0,
            },
            microsynth::ir::IrEdge {
                from: 1,
                to: 2,
                to_input: 1,
            },
        ],
        params: vec![
            microsynth::ir::IrParam {
                name: "freq".into(),
                node: 0,
                input: 0,
                default: default_freq_hz,
            },
            microsynth::ir::IrParam {
                name: "gain".into(),
                node: 1,
                input: 0,
                default: 0.0,
            },
        ],
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding { node: 2, table_id }],
        output_node: 2,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!(
            "usage: render_dictionary_ab <dictionary.msct> <gain_env_f32.raw> \
             <freq_env_f32.raw> <sample_rate> <out.wav>"
        );
        std::process::exit(1);
    }
    let dict_path = Path::new(&args[1]);
    let gain_env_path = Path::new(&args[2]);
    let freq_env_path = Path::new(&args[3]);
    let sample_rate: f32 = args[4].parse().expect("sample_rate must be a float");
    let out_path = Path::new(&args[5]);

    let table_bytes =
        fs::read(dict_path).unwrap_or_else(|e| panic!("failed to read {dict_path:?}: {e}"));
    let table = CoeffTable::from_bytes(&table_bytes)
        .unwrap_or_else(|e| panic!("failed to decode {dict_path:?}: {e:?}"));
    println!("Decoded '{}': {} entries", table.name, table.entries.len());
    for e in &table.entries {
        let mg = e
            .metadata
            .iter()
            .find(|(k, _)| k == "mainlobe_gain")
            .map(|(_, v)| *v);
        let ng = e
            .metadata
            .iter()
            .find(|(k, _)| k == "noise_gain")
            .map(|(_, v)| *v);
        println!(
            "  f0={:.2} Hz  M={}  J={}  K={}  mainlobe_gain={:?}  noise_gain={:?}",
            e.f0_hz,
            e.partial_freqs.len(),
            e.j_noise,
            e.k_channels,
            mg,
            ng
        );
    }

    let gain_env = read_f32_raw(gain_env_path);
    let freq_env = read_f32_raw(freq_env_path);
    assert_eq!(
        gain_env.len(),
        freq_env.len(),
        "gain envelope ({} samples, {:?}) and freq envelope ({} samples, {:?}) \
         must have the same length -- the Python driver writes both from the \
         same num_samples",
        gain_env.len(),
        gain_env_path,
        freq_env.len(),
        freq_env_path
    );
    println!(
        "Envelopes: {} samples ({:.2}s @ {} Hz)",
        gain_env.len(),
        gain_env.len() as f32 / sample_rate,
        sample_rate
    );

    // Register: the real runtime upload path's Rust-side half (Part 2/3 of
    // docs/coeff-table-bank-format.md -- this is the register() a host's
    // ms_coeff_table_register ABI export would call after the same decode step).
    let mut bank = CoeffTableBank::new();
    let table_id = bank.register(table).0;

    let mut reg = UGenRegistry::new();
    microsynth::register_builtins(&mut reg);
    microsynth::register_table_bound_builtins(&mut reg);

    let default_freq_hz = freq_env.first().copied().unwrap_or(220.0);
    let ir = dictionary_ir(table_id, default_freq_hz);
    ir.validate(&reg).expect("IR should validate");
    let def = ir
        .compile_with_tables(&reg, &bank)
        .expect("compile_with_tables should resolve the registered table");

    let block_size = 128usize;
    let mut engine = Engine::new(EngineConfig {
        sample_rate,
        block_size,
    });
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    let num_samples = gain_env.len();
    let num_blocks = num_samples.div_ceil(block_size);
    let block_dur = block_size as f32 / sample_rate;
    let mut out: Vec<f32> = Vec::with_capacity(num_blocks * block_size);

    for b in 0..num_blocks {
        // Target this block's gain/freq at its END sample (or the envelope's
        // last value once exhausted), glided from wherever the previous block
        // left off -- the per-block ramp RFC requirement 6 asks for, at native
        // block resolution. Gain glides linearly in its own (already
        // perceptually-linear) units; freq glides in pitch space
        // (`GlideSpace::Pitch`, an equal-ratio sweep) since it is a genuinely
        // a pitch parameter and MOT-642's melody data is a sequence of
        // discrete MIDI-grid targets, not an already-linear-in-Hz curve.
        let end_idx = ((b + 1) * block_size).min(num_samples) - 1;
        let gain_target = gain_env[end_idx];
        let freq_target = freq_env[end_idx];
        engine.set_param_glide(
            &synth,
            "gain",
            gain_target,
            block_dur,
            GlideShape::Linear,
            GlideSpace::Raw,
        );
        engine.set_param_glide(
            &synth,
            "freq",
            freq_target,
            block_dur,
            GlideShape::Linear,
            GlideSpace::Pitch,
        );
        if let Some(buf) = engine.render() {
            out.extend_from_slice(buf.channel(0).samples());
        }
    }
    out.truncate(num_samples);

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).expect("failed to create output directory");
    }
    wav::write_pcm16(&[out.as_slice()], sample_rate, out_path);
    println!(
        "Wrote {} samples ({:.2}s) to {}",
        out.len(),
        out.len() as f32 / sample_rate,
        out_path.display()
    );
}
