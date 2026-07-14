//! Offline-render convenience for a compiled IR: instantiate, drive a gate
//! schedule, render the sustain plus the natural tail, and trim/pad to a fixed
//! length. Wraps the same `Engine` pattern the CLI uses (`microsynth-cli.rs`),
//! with envelope-tail detection so one-shots and sustaining voices both render
//! cleanly.

use super::IrSynthDef;
use crate::dsl::compiler::UGenRegistry;
use crate::engine::{Engine, EngineConfig};
use crate::ir::IrError;
use alloc::string::String;
use alloc::vec::Vec;

/// How to render an IR to samples.
pub struct RenderSpec {
    pub sample_rate: f32,
    pub block_size: usize,
    /// Parameter overrides applied before rendering (e.g. `freq`, `amp`).
    pub params: Vec<(String, f32)>,
    /// When to release the `gate` (set it to 0). Ignored if the def has no
    /// `gate` param. The gate is held at 1 until this time.
    pub gate_on_secs: f32,
    /// Safety cap on how long to keep rendering after gate-off waiting for the
    /// graph to report done.
    pub max_tail_secs: f32,
    /// Exact output length. The concatenated render is trimmed or zero-padded
    /// to this many seconds.
    pub duration_secs: f32,
}

impl RenderSpec {
    /// The NSynth render convention: 16 kHz mono, gate held 3 s then released,
    /// 4 s total (see plan §5.2). `params` starts empty — set `freq`/`amp`.
    pub fn nsynth() -> Self {
        RenderSpec {
            sample_rate: 16_000.0,
            block_size: 64,
            params: Vec::new(),
            gate_on_secs: 3.0,
            max_tail_secs: 1.0,
            duration_secs: 4.0,
        }
    }
}

/// Compile `ir` and render it to `Vec<Vec<f32>>` (one buffer per channel) per
/// `spec`. The IR is compiled via [`IrSynthDef::compile`]; validate first if it
/// is untrusted.
pub fn render_ir(
    ir: &IrSynthDef,
    reg: &UGenRegistry,
    spec: &RenderSpec,
) -> Result<Vec<Vec<f32>>, IrError> {
    let def = ir.compile(reg)?;

    let mut engine = Engine::new(EngineConfig {
        sample_rate: spec.sample_rate,
        block_size: spec.block_size,
    });
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());

    let has_gate = def.param_names().iter().any(|(n, _, _)| n == "gate");
    if has_gate {
        // Hold the gate open for the sustain phase.
        engine.set_param(&synth, "gate", 1.0);
    }
    for (name, value) in &spec.params {
        engine.set_param(&synth, name, *value);
    }
    engine.prepare();

    let block = spec.block_size.max(1);
    let target_samples = (spec.duration_secs * spec.sample_rate).round() as usize;
    let gate_off_block = (spec.gate_on_secs * spec.sample_rate / block as f32).ceil() as usize;
    let max_blocks = ((spec.gate_on_secs + spec.max_tail_secs) * spec.sample_rate / block as f32)
        .ceil() as usize;

    let mut channels: Vec<Vec<f32>> = Vec::new();
    let mut rendered_samples = 0usize;
    let mut block_idx = 0usize;
    let mut gate_released = !has_gate;

    // Render until we have at least the target length, the graph reports done
    // after gate-off, or we hit the safety cap.
    loop {
        if has_gate && !gate_released && block_idx >= gate_off_block {
            engine.set_param(&synth, "gate", 0.0);
            gate_released = true;
        }

        let rendered = engine.render_offline(1);
        if channels.is_empty() && !rendered.is_empty() {
            channels.resize(rendered.len(), Vec::with_capacity(target_samples));
        }
        for (ch, buf) in rendered.iter().enumerate() {
            channels[ch].extend_from_slice(buf);
        }
        rendered_samples += rendered.first().map_or(0, |c| c.len());
        block_idx += 1;

        let have_target = rendered_samples >= target_samples;
        let done_after_release = gate_released
            && synth
                .node_ids()
                .iter()
                .any(|&id| engine.graph().node_is_done(id));

        if (have_target && (done_after_release || !has_gate)) || block_idx >= max_blocks {
            break;
        }
        // Never spin past the target if there is no gate to release.
        if have_target && !has_gate {
            break;
        }
    }

    // Trim or zero-pad each channel to exactly the target length.
    if channels.is_empty() {
        channels.push(Vec::new());
    }
    for ch in &mut channels {
        ch.resize(target_samples, 0.0);
    }
    Ok(channels)
}
