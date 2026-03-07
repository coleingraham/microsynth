//! WebAudio backend — WASM bindings for running microsynth in the browser.
//!
//! Exposes a `WebSynth` handle to JavaScript that:
//! 1. Accepts DSL source code to compile and load a SynthDef
//! 2. Renders audio blocks on demand (called from a ScriptProcessorNode or AudioWorklet)
//!
//! # Usage from JS
//!
//! ```js
//! import init, { WebSynth } from './microsynth.js';
//! await init();
//!
//! const synth = new WebSynth(44100.0, 256);
//! synth.compile_and_load(`
//!     synthdef pad freq=220.0 amp=0.3 =
//!         let sig = sinOsc freq 0.0
//!         sig * amp
//! `);
//!
//! // In your ScriptProcessorNode.onaudioprocess:
//! synth.render(outputLeftChannel, outputRightChannel);
//! ```

use alloc::string::ToString;
use alloc::vec::Vec;
use wasm_bindgen::prelude::*;

use crate::dsl::{self, UGenRegistry};
use crate::engine::{Engine, EngineConfig};
use crate::ugens::register_builtins;

/// A synthesis engine handle exposed to JavaScript.
///
/// Each `WebSynth` owns a full `Engine` with a compiled DSL graph.
/// Call `render()` from your audio callback to fill output buffers.
#[wasm_bindgen]
pub struct WebSynth {
    engine: Engine,
    registry: UGenRegistry,
    /// Number of output channels from the current graph.
    num_channels: usize,
}

#[wasm_bindgen]
impl WebSynth {
    /// Create a new synthesizer.
    ///
    /// `sample_rate`: the AudioContext sample rate (e.g. 44100 or 48000).
    /// `block_size`: samples per render call (typically 128, 256, or 512).
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32, block_size: usize) -> WebSynth {
        let mut registry = UGenRegistry::new();
        register_builtins(&mut registry);

        let config = EngineConfig { sample_rate, block_size };
        let engine = Engine::new(config);

        WebSynth {
            engine,
            registry,
            num_channels: 0,
        }
    }

    /// Compile DSL source and load the first SynthDef into the engine.
    ///
    /// Replaces whatever was previously loaded.
    /// Returns an error string on failure, or null on success.
    #[wasm_bindgen(js_name = "compileAndLoad")]
    pub fn compile_and_load(&mut self, source: &str) -> Result<(), JsError> {
        let defs = dsl::compile(source, &self.registry)
            .map_err(|e| JsError::new(&e.to_string()))?;

        if defs.is_empty() {
            return Err(JsError::new("no synthdef found in source"));
        }

        // Reset engine with fresh graph
        let config = EngineConfig {
            sample_rate: self.engine.context().sample_rate,
            block_size: self.engine.context().block_size,
        };
        self.engine = Engine::new(config);

        let synth = self.engine.instantiate_synthdef(&defs[0]);
        self.engine.graph_mut().set_sink(synth.output_node());
        self.engine.prepare();

        // Probe output channel count
        if let Some(output) = self.engine.render() {
            self.num_channels = output.num_channels();
        } else {
            self.num_channels = 1;
        }

        // Reset time after the probe render
        let config = EngineConfig {
            sample_rate: self.engine.context().sample_rate,
            block_size: self.engine.context().block_size,
        };
        self.engine = Engine::new(config);
        let synth = self.engine.instantiate_synthdef(&defs[0]);
        self.engine.graph_mut().set_sink(synth.output_node());
        self.engine.prepare();

        Ok(())
    }

    /// Render one block of audio into the provided JS Float32Arrays.
    ///
    /// `left` and `right` are the output channel buffers from the
    /// ScriptProcessorNode's `onaudioprocess` event. If the synth is mono,
    /// both channels get the same data. If stereo, channel 0 → left,
    /// channel 1 → right.
    #[wasm_bindgen]
    pub fn render(&mut self, left: &mut [f32], right: &mut [f32]) {
        // The engine block size might differ from the JS buffer size.
        // We render in engine-block-sized chunks to fill the JS buffer.
        let js_len = left.len();
        let block_size = self.engine.context().block_size;
        let mut offset = 0;

        while offset < js_len {
            let chunk = (js_len - offset).min(block_size);

            if let Some(output) = self.engine.render() {
                let nc = output.num_channels();
                // Left channel
                let src_l = output.channel(0).samples();
                let copy_len = chunk.min(src_l.len());
                left[offset..offset + copy_len].copy_from_slice(&src_l[..copy_len]);

                // Right channel
                if nc >= 2 {
                    let src_r = output.channel(1).samples();
                    let copy_len_r = chunk.min(src_r.len());
                    right[offset..offset + copy_len_r].copy_from_slice(&src_r[..copy_len_r]);
                } else {
                    // Mono: duplicate to right
                    right[offset..offset + copy_len].copy_from_slice(&src_l[..copy_len]);
                }
            } else {
                // No output — fill silence
                left[offset..offset + chunk].fill(0.0);
                right[offset..offset + chunk].fill(0.0);
            }

            offset += chunk;
        }
    }

    /// Get the number of output channels the current graph produces.
    #[wasm_bindgen(getter, js_name = "numChannels")]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Get current playback time in seconds.
    #[wasm_bindgen(getter, js_name = "currentTime")]
    pub fn current_time(&self) -> f64 {
        self.engine.time_secs()
    }

    /// Get the sample rate.
    #[wasm_bindgen(getter, js_name = "sampleRate")]
    pub fn sample_rate(&self) -> f32 {
        self.engine.context().sample_rate
    }
}

/// List all available built-in UGen names (for the web UI).
#[wasm_bindgen(js_name = "availableUGens")]
pub fn available_ugens() -> Vec<JsValue> {
    let names = [
        "sinOsc", "saw", "pulse", "tri", "phasor",
        "whiteNoise", "pinkNoise",
        "onePole", "lpf", "hpf", "bpf",
        "line", "asr",
        "delay",
        "pan2", "mix", "sampleAndHold",
    ];
    names.iter().map(|&n| JsValue::from_str(n)).collect()
}
