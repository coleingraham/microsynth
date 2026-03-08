//! WebAudio backend — WASM bindings for running microsynth in the browser.
//!
//! Provides two APIs:
//!
//! ## 1. Raw C exports (for AudioWorklet — runs in worklet thread)
//!
//! These `#[no_mangle] extern "C"` functions work without wasm-bindgen,
//! avoiding the TextEncoder/TextDecoder limitation in AudioWorkletGlobalScope.
//! The worklet processor loads the raw WASM module and calls these directly.
//!
//! ## 2. wasm-bindgen exports (for main thread — setup and fallback)
//!
//! The `WebSynth` class is used from the main thread for:
//! - A ScriptProcessorNode fallback (if AudioWorklet is unavailable)
//! - DSL compilation feedback (error messages)
//!
//! # Architecture
//!
//! ```text
//! Main Thread                          AudioWorklet Thread
//! ┌──────────────────┐                ┌─────────────────────┐
//! │  index.html       │   postMessage │  processor.js       │
//! │  - editor UI      │──────────────>│  - loads WASM raw   │
//! │  - compile button │  (DSL source) │  - calls ms_compile │
//! │  - scope display  │               │  - calls ms_render  │
//! │                   │               │  - fills outputs    │
//! └──────────────────┘                └─────────────────────┘
//! ```

use alloc::boxed::Box;
#[cfg(feature = "web")]
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::dsl::{self, UGenRegistry};
use crate::engine::{Engine, EngineConfig};
use crate::ugens::register_builtins;

#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

// ============================================================================
// Raw C exports for AudioWorklet (no wasm-bindgen needed in worklet scope)
// ============================================================================

/// Single-threaded global cell for WASM. WASM has no threads by default,
/// so this is safe in the AudioWorklet context.
struct WasmCell<T>(core::cell::UnsafeCell<T>);
unsafe impl<T> Sync for WasmCell<T> {}

impl<T> WasmCell<T> {
    const fn new(val: T) -> Self {
        WasmCell(core::cell::UnsafeCell::new(val))
    }
    /// SAFETY: Caller must ensure no concurrent access (guaranteed in single-threaded WASM).
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }
}

/// Global engine state for the worklet.
static ENGINE: WasmCell<Option<Engine>> = WasmCell::new(None);
static REGISTRY: WasmCell<Option<UGenRegistry>> = WasmCell::new(None);
/// Compiled SynthDefs available for spawning voices.
static DEFS: WasmCell<Option<Vec<crate::synthdef::SynthDef>>> = WasmCell::new(None);
/// Bus node for multi-voice mixing.
static BUS_NODE: WasmCell<Option<crate::node::NodeId>> = WasmCell::new(None);

/// Allocate `size` bytes in WASM linear memory. Returns a pointer.
/// Used by JS to write string data (DSL source) into WASM memory.
#[unsafe(no_mangle)]
pub extern "C" fn ms_alloc(size: usize) -> *mut u8 {
    let mut buf = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// Free a previously allocated buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_free(ptr: *mut u8, capacity: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, capacity);
    }
}

/// Initialize the engine with the given sample rate.
/// Block size is fixed at 128 (WebAudio render quantum).
#[unsafe(no_mangle)]
pub extern "C" fn ms_init(sample_rate: f32) {
    let mut registry = UGenRegistry::new();
    register_builtins(&mut registry);

    let config = EngineConfig {
        sample_rate,
        block_size: 128, // WebAudio render quantum
    };

    unsafe {
        *ENGINE.get_mut() = Some(Engine::new(config));
        *REGISTRY.get_mut() = Some(registry);
    }
}

/// Compile DSL source and load it into the engine.
///
/// `source_ptr` and `source_len` point to a UTF-8 string in WASM memory
/// (previously written via `ms_alloc`).
///
/// Returns 0 on success, 1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_compile(source_ptr: *const u8, source_len: usize) -> u32 {
    let source_bytes = unsafe { core::slice::from_raw_parts(source_ptr, source_len) };
    let source = match core::str::from_utf8(source_bytes) {
        Ok(s) => s,
        Err(_) => return 1,
    };

    let registry = match unsafe { REGISTRY.get_mut() }.as_ref() {
        Some(r) => r,
        None => return 1,
    };

    let defs = match dsl::compile(source, registry) {
        Ok(d) => d,
        Err(_) => return 1,
    };

    if defs.is_empty() {
        return 1;
    }

    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return 1,
    };

    // Reset engine
    let sr = engine.context().sample_rate;
    *engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: 128,
    });

    let synth = engine.instantiate_synthdef(&defs[0]);
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    0
}

/// Render 128 samples of stereo audio.
///
/// `out_left` and `out_right` must each point to 128 f32s of writable memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_render(out_left: *mut f32, out_right: *mut f32) {
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return,
    };

    let left = unsafe { core::slice::from_raw_parts_mut(out_left, 128) };
    let right = unsafe { core::slice::from_raw_parts_mut(out_right, 128) };

    if let Some(output) = engine.render() {
        let nc = output.num_channels();
        let src_l = output.channel(0).samples();
        let copy_len = 128.min(src_l.len());
        left[..copy_len].copy_from_slice(&src_l[..copy_len]);

        if nc >= 2 {
            let src_r = output.channel(1).samples();
            let copy_len_r = 128.min(src_r.len());
            right[..copy_len_r].copy_from_slice(&src_r[..copy_len_r]);
        } else {
            right[..copy_len].copy_from_slice(&src_l[..copy_len]);
        }
    } else {
        left.fill(0.0);
        right.fill(0.0);
    }
}

/// Compile DSL source and store the SynthDef(s) for voice spawning.
/// Also sets up a Bus as the graph sink for multi-voice mixing.
/// Returns 0 on success, 1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_compile_def(source_ptr: *const u8, source_len: usize) -> u32 {
    let source_bytes = unsafe { core::slice::from_raw_parts(source_ptr, source_len) };
    let source = match core::str::from_utf8(source_bytes) {
        Ok(s) => s,
        Err(_) => return 1,
    };

    let registry = match unsafe { REGISTRY.get_mut() }.as_ref() {
        Some(r) => r,
        None => return 1,
    };

    let defs = match dsl::compile(source, registry) {
        Ok(d) => d,
        Err(_) => return 1,
    };

    if defs.is_empty() {
        return 1;
    }

    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return 1,
    };

    // Reset engine for fresh voice management
    let sr = engine.context().sample_rate;
    *engine = Engine::new(EngineConfig {
        sample_rate: sr,
        block_size: 128,
    });

    // Create a bus node as the graph sink
    let bus = crate::ugens::Bus::new(32);
    let bus_id = engine.graph_mut().add_node(Box::new(bus));
    engine.graph_mut().set_sink(bus_id);
    engine.prepare();

    unsafe {
        *BUS_NODE.get_mut() = Some(bus_id);
        *DEFS.get_mut() = Some(defs);
    }

    0
}

/// Spawn a voice from the first compiled SynthDef, connected to the bus.
/// Returns the voice ID (> 0), or 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn ms_spawn_voice() -> u64 {
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return 0,
    };
    let defs = match unsafe { DEFS.get_mut() }.as_ref() {
        Some(d) => d,
        None => return 0,
    };
    let bus_id = match unsafe { BUS_NODE.get_mut() } {
        Some(id) => *id,
        None => return 0,
    };

    if defs.is_empty() {
        return 0;
    }

    match engine.spawn_voice_on_bus(&defs[0], bus_id) {
        Some(voice_id) => {
            engine.prepare();
            voice_id.0
        }
        None => 0,
    }
}

/// Set the gate parameter on a voice. gate > 0 = note on, gate = 0 = note off.
#[unsafe(no_mangle)]
pub extern "C" fn ms_voice_gate(voice_id: u64, value: f32) {
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return,
    };
    engine.set_voice_param(crate::scheduler::VoiceId(voice_id), "gate", value);
}

/// Set a named parameter on a voice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_voice_param(
    voice_id: u64,
    param_ptr: *const u8,
    param_len: usize,
    value: f32,
) {
    let param_bytes = unsafe { core::slice::from_raw_parts(param_ptr, param_len) };
    let param = match core::str::from_utf8(param_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };

    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return,
    };
    engine.set_voice_param(crate::scheduler::VoiceId(voice_id), param, value);
}

/// Free a voice by ID.
#[unsafe(no_mangle)]
pub extern "C" fn ms_free_voice(voice_id: u64) {
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return,
    };
    engine.free_voice(crate::scheduler::VoiceId(voice_id));
    engine.prepare();
}

/// Free all voices that have finished (e.g. envelope completed).
/// Returns the number of voices freed.
#[unsafe(no_mangle)]
pub extern "C" fn ms_free_done() -> u32 {
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return 0,
    };
    let count = engine.free_done_synths();
    if count > 0 {
        engine.prepare();
    }
    count as u32
}

// ============================================================================
// Musical time & tuning utility exports (stateless, no engine state needed)
// ============================================================================

/// Convert a MIDI note number to Hz using standard 12-TET.
#[unsafe(no_mangle)]
pub extern "C" fn ms_midi_to_hz(midi_note: f32, a4_freq: f32) -> f32 {
    crate::tuning::midi_to_hz_12tet(midi_note, a4_freq)
}

/// Convert Hz to MIDI note number using standard 12-TET.
#[unsafe(no_mangle)]
pub extern "C" fn ms_hz_to_midi(hz: f32, a4_freq: f32) -> f32 {
    crate::tuning::hz_to_midi_12tet(hz, a4_freq)
}

/// Apply a cent offset to a base frequency.
/// Returns `base_hz * 2^(cents / 1200)`.
#[unsafe(no_mangle)]
pub extern "C" fn ms_apply_cents(base_hz: f32, cents: f32) -> f32 {
    crate::tuning::apply_cents(base_hz, cents)
}

/// Convert a musical position (bar:step +tick_offset) to an absolute sample offset.
///
/// All time config parameters are passed as scalars (no structs over FFI).
#[unsafe(no_mangle)]
pub extern "C" fn ms_position_to_samples(
    bpm: f32,
    numerator: u8,
    denominator: u8,
    grid_steps: u16,
    ppqn: u16,
    sample_rate: f32,
    bar: u32,
    step: u16,
    tick_offset: i16,
) -> u64 {
    let config = crate::musical_time::TimeConfig {
        bpm,
        numerator,
        denominator,
        grid_steps,
        ppqn,
        sample_rate,
    };
    let pos = crate::musical_time::MusicalPosition::new(bar, step, tick_offset);
    config.position_to_samples(pos)
}

/// Convert a duration in grid steps to a duration in samples.
#[unsafe(no_mangle)]
pub extern "C" fn ms_steps_to_samples(
    bpm: f32,
    numerator: u8,
    denominator: u8,
    grid_steps: u16,
    sample_rate: f32,
    steps: f32,
) -> u64 {
    let config = crate::musical_time::TimeConfig {
        bpm,
        numerator,
        denominator,
        grid_steps,
        ppqn: 0,
        sample_rate,
    };
    config.steps_to_samples(steps)
}

/// Schedule a gate-on and auto gate-off for a voice.
/// `on_time` and `off_time` are absolute sample offsets.
#[unsafe(no_mangle)]
pub extern "C" fn ms_schedule_note(voice_id: u64, on_time: u64, off_time: u64) {
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return,
    };
    engine
        .scheduler_mut()
        .schedule_note(crate::scheduler::VoiceId(voice_id), on_time, off_time);
}

/// Schedule a note with attack-aligned pre-trigger.
///
/// `grid_time`: where the attack peak should align (absolute sample offset).
/// `attack_secs`: envelope attack time in seconds.
/// `duration_samples`: how long after `grid_time` before gate-off.
#[unsafe(no_mangle)]
pub extern "C" fn ms_schedule_note_aligned(
    voice_id: u64,
    grid_time: u64,
    attack_secs: f32,
    duration_samples: u64,
) {
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return,
    };
    engine.schedule_note_aligned(
        crate::scheduler::VoiceId(voice_id),
        grid_time,
        attack_secs,
        duration_samples,
    );
}

// ============================================================================
// wasm-bindgen exports for main thread (ScriptProcessorNode fallback + UI)
// ============================================================================

/// A synthesis engine handle exposed to JavaScript (main thread).
///
/// Used for ScriptProcessorNode fallback and for compile-time error reporting.
#[cfg(feature = "web")]
#[wasm_bindgen]
pub struct WebSynth {
    engine: Engine,
    registry: UGenRegistry,
    num_channels: usize,
}

#[cfg(feature = "web")]
#[wasm_bindgen]
impl WebSynth {
    /// Create a new synthesizer.
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

    /// Compile DSL source and load the first SynthDef.
    /// Returns an error string on failure.
    #[wasm_bindgen(js_name = "compileAndLoad")]
    pub fn compile_and_load(&mut self, source: &str) -> Result<(), JsError> {
        let defs = dsl::compile(source, &self.registry)
            .map_err(|e| JsError::new(&e.to_string()))?;

        if defs.is_empty() {
            return Err(JsError::new("no synthdef found in source"));
        }

        let sr = self.engine.context().sample_rate;
        let bs = self.engine.context().block_size;

        self.engine = Engine::new(EngineConfig {
            sample_rate: sr,
            block_size: bs,
        });

        let synth = self.engine.instantiate_synthdef(&defs[0]);
        self.engine.graph_mut().set_sink(synth.output_node());
        self.engine.prepare();

        // Probe channel count
        if let Some(output) = self.engine.render() {
            self.num_channels = output.num_channels();
        } else {
            self.num_channels = 1;
        }

        // Reset after probe
        self.engine = Engine::new(EngineConfig {
            sample_rate: sr,
            block_size: bs,
        });
        let synth = self.engine.instantiate_synthdef(&defs[0]);
        self.engine.graph_mut().set_sink(synth.output_node());
        self.engine.prepare();

        Ok(())
    }

    /// Render audio into stereo Float32Arrays (ScriptProcessorNode fallback).
    #[wasm_bindgen]
    pub fn render(&mut self, left: &mut [f32], right: &mut [f32]) {
        let js_len = left.len();
        let block_size = self.engine.context().block_size;
        let mut offset = 0;

        while offset < js_len {
            let chunk = (js_len - offset).min(block_size);

            if let Some(output) = self.engine.render() {
                let nc = output.num_channels();
                let src_l = output.channel(0).samples();
                let copy_len = chunk.min(src_l.len());
                left[offset..offset + copy_len].copy_from_slice(&src_l[..copy_len]);

                if nc >= 2 {
                    let src_r = output.channel(1).samples();
                    let copy_len_r = chunk.min(src_r.len());
                    right[offset..offset + copy_len_r].copy_from_slice(&src_r[..copy_len_r]);
                } else {
                    right[offset..offset + copy_len].copy_from_slice(&src_l[..copy_len]);
                }
            } else {
                left[offset..offset + chunk].fill(0.0);
                right[offset..offset + chunk].fill(0.0);
            }

            offset += chunk;
        }
    }

    #[wasm_bindgen(getter, js_name = "numChannels")]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    #[wasm_bindgen(getter, js_name = "currentTime")]
    pub fn current_time(&self) -> f64 {
        self.engine.time_secs()
    }

    #[wasm_bindgen(getter, js_name = "sampleRate")]
    pub fn sample_rate(&self) -> f32 {
        self.engine.context().sample_rate
    }
}

/// Validate DSL source and return error message (or empty string on success).
/// Used by the main thread UI for immediate feedback.
#[cfg(feature = "web")]
#[wasm_bindgen(js_name = "validateDSL")]
pub fn validate_dsl(source: &str) -> String {
    let mut registry = UGenRegistry::new();
    register_builtins(&mut registry);
    match dsl::compile(source, &registry) {
        Ok(defs) if defs.is_empty() => String::from("no synthdef found"),
        Ok(_) => String::new(),
        Err(e) => e.to_string(),
    }
}

/// List all available built-in UGen names.
#[cfg(feature = "web")]
#[wasm_bindgen(js_name = "availableUGens")]
pub fn available_ugens() -> Vec<JsValue> {
    let names = [
        "sinOsc", "saw", "pulse", "tri", "phasor",
        "whiteNoise", "pinkNoise",
        "onePole", "lpf", "hpf", "bpf",
        "line", "perc", "asr", "adsr",
        "delay",
        "pan2", "mix", "sampleAndHold",
        "impulse", "lag", "clip",
        "waveTable",
        "sinTable", "sawTable", "triTable", "squareTable",
    ];
    names.iter().map(|&n| JsValue::from_str(n)).collect()
}
