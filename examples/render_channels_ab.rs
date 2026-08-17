//! Multi-channel timbre crossfade render: decode K single-channel coefficient
//! tables produced from one fitted dictionary (one table per channel, each
//! `k_channels=1` -- the existing wire format, unchanged), build K
//! `partialsNoise` voices bound one-per-table, and sum them with each voice's
//! own time-varying gain envelope (its channel's fitted mix weight times the
//! note's overall gain) while all K voices share one `freq` envelope (they
//! are the same note, just different timbre stages). This is a driver-level
//! realization of a temporally-ordered multi-channel fit: no engine or
//! wire-format change, only K single-channel voices crossfaded by the
//! envelopes the driver supplies -- generalizes `render_dictionary_ab`'s
//! existing single-voice gain/freq-glide driver to K summed voices.
//!
//! Not part of the test suite -- run manually, from this crate's root:
//!
//! ```text
//! cargo run --release --example render_channels_ab -- \
//!     <sample_rate> <freq_env_f32_SRhz.raw> <out.wav> \
//!     <dict1.msct> <gain1_env_f32_SRhz.raw> \
//!     [<dict2.msct> <gain2_env_f32_SRhz.raw> ...]
//! ```
//!
//! One `(dict, gain_env)` pair per channel; K is however many pairs are
//! given (K=1 works too -- the plain single-voice case, useful as this
//! ticket's own K=1 comparison arm without a second binary). Every gain
//! envelope and the freq envelope must have the same length (raw
//! little-endian f32, one sample per audio sample, the same
//! frame-rate -> audio-rate upsampling convention `render_dictionary_ab`
//! already uses); this example asserts that rather than silently truncating.

#[path = "common/raw_env.rs"]
mod raw_env;
#[path = "common/wav.rs"]
mod wav;

use microsynth::coeff_table::{CoeffTable, CoeffTableBank};
use microsynth::curve::{GlideShape, GlideSpace};
use microsynth::dsl::compiler::UGenRegistry;
use microsynth::ir::{IrEdge, IrNode, IrParam, IrSynthDef, IrTableBinding, SynthDefClass};
use microsynth::{Engine, EngineConfig};
use raw_env::read_f32_raw;
use std::env;
use std::fs;
use std::path::Path;

/// K voices, each a table-bound `partialsNoise` fed the shared `freq` param
/// and its own `gain_k` param, summed via a chain of `Add` nodes (`K-1` of
/// them; K=1 has none -- the lone voice is the output directly). Node layout:
/// `0` = freq param, `1..=K` = gain params, `K+1..=2K` = voice nodes,
/// `2K+1..2K+K` = the `Add` chain (empty for K=1).
fn channels_ir(table_ids: &[u32], default_freq_hz: f32) -> IrSynthDef {
    let k = table_ids.len();
    assert!(k >= 1, "need at least one (dict, gain_env) pair");

    let mut nodes = vec![IrNode::Param {
        name: "freq".into(),
        default: default_freq_hz,
    }];
    for i in 0..k {
        nodes.push(IrNode::Param {
            name: format!("gain{i}"),
            default: 0.0,
        });
    }
    let voice_base = 1 + k;
    for _ in 0..k {
        nodes.push(IrNode::UGen {
            kind: "partialsNoise".into(),
            consts: vec![],
        });
    }

    let mut edges = Vec::new();
    let mut params = vec![IrParam {
        name: "freq".into(),
        node: 0,
        input: 0,
        default: default_freq_hz,
    }];
    for i in 0..k {
        let voice = voice_base + i;
        edges.push(IrEdge {
            from: 0,
            to: voice,
            to_input: 0,
        }); // freq -> voice i's freq input
        edges.push(IrEdge {
            from: 1 + i,
            to: voice,
            to_input: 1,
        }); // gain_i -> voice i's gain input
        params.push(IrParam {
            name: format!("gain{i}"),
            node: 1 + i,
            input: 0,
            default: 0.0,
        });
    }

    let add_base = voice_base + k;
    let output_node = if k == 1 {
        voice_base
    } else {
        // Chain: sum0 = Add(voice0, voice1); sum1 = Add(sum0, voice2); ...
        for i in 0..(k - 1) {
            let add_node = add_base + i;
            let left = if i == 0 { voice_base } else { add_node - 1 };
            let right = voice_base + i + 1;
            nodes.push(IrNode::UGen {
                kind: "Add".into(),
                consts: vec![],
            });
            edges.push(IrEdge {
                from: left,
                to: add_node,
                to_input: 0,
            });
            edges.push(IrEdge {
                from: right,
                to: add_node,
                to_input: 1,
            });
        }
        add_base + (k - 2)
    };

    let table_bindings = table_ids
        .iter()
        .enumerate()
        .map(|(i, &table_id)| IrTableBinding {
            node: voice_base + i,
            table_id,
        })
        .collect();

    IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "mot668_channels_ab".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes,
        edges,
        params,
        audio_inputs: vec![],
        table_bindings,
        output_node,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 || !(args.len() - 4).is_multiple_of(2) {
        eprintln!(
            "usage: render_channels_ab <sample_rate> <freq_env.raw> <out.wav> \
             <dict1.msct> <gain1_env.raw> [<dict2.msct> <gain2_env.raw> ...]"
        );
        std::process::exit(1);
    }
    let sample_rate: f32 = args[1].parse().expect("sample_rate must be a float");
    let freq_env_path = Path::new(&args[2]);
    let out_path = Path::new(&args[3]);
    let pair_args = &args[4..];
    let k = pair_args.len() / 2;

    let freq_env = read_f32_raw(freq_env_path);

    let mut table_ids = Vec::with_capacity(k);
    let mut gain_envs = Vec::with_capacity(k);
    let mut bank = CoeffTableBank::new();
    for i in 0..k {
        let dict_path = Path::new(&pair_args[2 * i]);
        let gain_path = Path::new(&pair_args[2 * i + 1]);
        let table_bytes =
            fs::read(dict_path).unwrap_or_else(|e| panic!("failed to read {dict_path:?}: {e}"));
        let table = CoeffTable::from_bytes(&table_bytes)
            .unwrap_or_else(|e| panic!("failed to decode {dict_path:?}: {e:?}"));
        assert!(
            table.entries.iter().all(|e| e.k_channels == 1),
            "{dict_path:?}: every entry must be k_channels=1 (one channel per \
             dictionary, per this example's own crossfade contract)"
        );
        println!(
            "Channel {i}: decoded '{}' ({} entries) from {dict_path:?}",
            table.name,
            table.entries.len()
        );
        let table_id = bank.register(table).0;
        table_ids.push(table_id);

        let gain_env = read_f32_raw(gain_path);
        assert_eq!(
            gain_env.len(),
            freq_env.len(),
            "channel {i} gain envelope ({} samples, {gain_path:?}) and freq \
             envelope ({} samples, {freq_env_path:?}) must have the same length",
            gain_env.len(),
            freq_env.len()
        );
        gain_envs.push(gain_env);
    }

    let mut reg = UGenRegistry::new();
    microsynth::register_builtins(&mut reg);
    microsynth::register_table_bound_builtins(&mut reg);

    let default_freq_hz = freq_env.first().copied().unwrap_or(220.0);
    let ir = channels_ir(&table_ids, default_freq_hz);
    ir.validate(&reg).expect("IR should validate");
    let def = ir
        .compile_with_tables(&reg, &bank)
        .expect("compile_with_tables should resolve every registered table");

    let block_size = 128usize;
    let mut engine = Engine::new(EngineConfig {
        sample_rate,
        block_size,
    });
    let synth = engine.instantiate_synthdef(&def);

    // Coherent-noise fix: the K voices are meant to be heard as ONE crossfaded
    // sound source, not K independent ones, but `instantiate_synthdef` gives
    // every node a distinct noise seed by default (`derive_noise_seed`'s own
    // node-index term -- correct for independent sources, wrong here). Force
    // every voice's noise generator to the SAME seed so their noise streams
    // are sample-for-sample identical: the crossfade gain envelopes (which sum
    // to 1 by construction, see mot668_channels_ab_export.py's own docstring)
    // then sum the shared noise coherently too, not as the RMS sum of
    // uncorrelated sources. Node layout is `channels_ir`'s own: voices occupy
    // IR indices `1+k .. 1+2k`.
    let voice_base = 1 + k;
    let shared_noise_seed = 0x5EED_0000u32;
    for i in 0..k {
        let voice_node_id = synth.node_ids()[voice_base + i];
        let reseeded = engine
            .graph_mut()
            .reseed_node_noise(voice_node_id, shared_noise_seed);
        assert!(
            reseeded,
            "voice {i} (node index {}) did not accept a noise reseed -- \
             `reseed_node_noise` returns false only when the node index \
             doesn't resolve to a live slot, which means channels_ir's node \
             layout (voices at 1+k..1+2k) has drifted from this offset. \
             Silently continuing here would degrade the coherent-noise fix \
             above to a no-op without any signal that it happened.",
            voice_node_id.index()
        );
    }

    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    let num_samples = freq_env.len();
    let num_blocks = num_samples.div_ceil(block_size);
    let block_dur = block_size as f32 / sample_rate;
    let mut out: Vec<f32> = Vec::with_capacity(num_blocks * block_size);

    for b in 0..num_blocks {
        let end_idx = ((b + 1) * block_size).min(num_samples) - 1;
        let freq_target = freq_env[end_idx];
        engine.set_param_glide(
            &synth,
            "freq",
            freq_target,
            block_dur,
            GlideShape::Linear,
            GlideSpace::Pitch,
        );
        for (i, gain_env) in gain_envs.iter().enumerate() {
            let gain_target = gain_env[end_idx];
            engine.set_param_glide(
                &synth,
                &format!("gain{i}"),
                gain_target,
                block_dur,
                GlideShape::Linear,
                GlideSpace::Raw,
            );
        }
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
        "Wrote {} samples ({:.2}s), {} channel(s) summed, to {}",
        out.len(),
        out.len() as f32 / sample_rate,
        k,
        out_path.display()
    );
}
