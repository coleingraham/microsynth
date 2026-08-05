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
//! # Glide shape/space ABI encoding
//!
//! A handful of raw exports (`ms_voice_param_glide`, `ms_schedule_musical_glides`)
//! carry a [`crate::curve::GlideShape`] and [`crate::curve::GlideSpace`] across
//! the C boundary. Both cross as plain integers plus one float, decoded by
//! [`decode_glide_shape`] / [`decode_glide_space`]:
//!
//! - `shape_kind: u32` — `0` = `Hold`, `1` = `Linear`, `2` = `Sine`,
//!   `3` = `Exponential` (any other value falls back to `Linear`).
//! - `tension: f32` — only meaningful when `shape_kind == 3`; ignored
//!   otherwise.
//! - `space_kind: u32` — `0` = `Raw`, `1` = `Pitch` (any other value falls
//!   back to `Raw`).
//!
//! This encoding is the one place these enums are represented as integers;
//! callers on the JS side should treat it as a stable contract, not
//! reimplement the shape/space vocabulary itself (see [`crate::curve`]).
//!
//! # Contract vs. diagnostic exports
//!
//! `ms_voice_param_glide`, `ms_legato_note_on`, `ms_legato_note_off`, and
//! `ms_schedule_musical_glides` are the stable contract: downstream
//! consumers build against these signatures. `ms_legato_slot_for` is not —
//! it's a read-only diagnostic accessor over internal bus-slot bookkeeping,
//! added only to make that bookkeeping testable, and may change or be
//! removed independently of the contract above.
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
use alloc::collections::BTreeMap;
use alloc::string::String as AllocString;
#[cfg(feature = "web")]
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::curve::{GlideShape, GlideSpace};
use crate::dsl::{self, UGenRegistry};
use crate::engine::{Engine, EngineConfig};
use crate::musical_sequence::schedule_musical_glides;
use crate::musical_time::{MusicalGlideSegment, MusicalPosition, TimeConfig};
use crate::ugens::register_builtins;
use crate::voice::LegatoVoice;

#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

/// Decode a [`GlideShape`] crossing the C-ABI — see the module docs for the
/// full encoding table. Unrecognized `kind` values fall back to `Linear`
/// rather than failing, since a raw export has no error channel for this
/// argument (it always returns void, following the shape of `ms_voice_param`).
fn decode_glide_shape(kind: u32, tension: f32) -> GlideShape {
    match kind {
        0 => GlideShape::Hold,
        2 => GlideShape::Sine,
        3 => GlideShape::Exponential(tension),
        _ => GlideShape::Linear,
    }
}

/// Decode a [`GlideSpace`] crossing the C-ABI — see the module docs for the
/// full encoding table. Unrecognized `kind` values fall back to `Raw`.
fn decode_glide_space(kind: u32) -> GlideSpace {
    match kind {
        1 => GlideSpace::Pitch,
        _ => GlideSpace::Raw,
    }
}

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

/// Named SynthDef registry for multi-timbral playback.
static DEF_REGISTRY: WasmCell<Option<BTreeMap<AllocString, crate::synthdef::SynthDef>>> =
    WasmCell::new(None);

/// Master effect synth (inserted between bus and graph sink).
static MASTER_SYNTH: WasmCell<Option<crate::synthdef::Synth>> = WasmCell::new(None);

/// A registered SynthDef's mono/legato voice mode: (pitch parameter name,
/// portamento seconds, tie-portamento shape, tie-portamento space — the
/// last two default to `Linear`/`Pitch` when the declaration omits them,
/// see `VoiceModeDecl`'s doc comment).
type LegatoModeEntry = (AllocString, f32, GlideShape, GlideSpace);

/// Mono/legato voice-mode metadata parsed from each registered SynthDef's
/// `voice` declaration (see `src/dsl`), keyed by SynthDef name. Populated by
/// `ms_register_def`; absent for any name whose DSL source had no `voice`
/// declaration.
static LEGATO_MODES: WasmCell<Option<BTreeMap<AllocString, LegatoModeEntry>>> = WasmCell::new(None);

/// One [`LegatoVoice`] track per SynthDef name that has been played legato at
/// least once via `ms_legato_note_on`, keyed by that name.
static LEGATO_VOICES: WasmCell<Option<BTreeMap<AllocString, LegatoVoice>>> = WasmCell::new(None);

/// The bus input slot reserved for each legato track's output, assigned once
/// on first use. Slots are handed out counting down from the bus's last
/// input index so they never collide with the low-to-high slots
/// `ms_spawn_voice_named` (via `Engine::spawn_voice_on_bus`) hands out for
/// ordinary polyphonic voices sharing the same bus.
static LEGATO_SLOTS: WasmCell<Option<BTreeMap<AllocString, usize>>> = WasmCell::new(None);

/// Initialize the engine with a Bus node as the graph sink.
/// Call once before `ms_register_def` / `ms_spawn_voice_named`.
#[unsafe(no_mangle)]
pub extern "C" fn ms_init_with_bus(sample_rate: f32) {
    let mut registry = UGenRegistry::new();
    register_builtins(&mut registry);

    let config = EngineConfig {
        sample_rate,
        block_size: 128,
    };
    let mut engine = Engine::new(config);

    // Create a stereo bus node as the graph sink. The input-slot count (voice
    // capacity) is fixed at MAX_BUS_INPUTS regardless of this channel count —
    // see `ugens::bus::ChannelCount`'s doc.
    let bus = crate::ugens::Bus::new(crate::ugens::ChannelCount::Stereo);
    let bus_id = engine.graph_mut().add_node(Box::new(bus));
    engine.graph_mut().set_sink(bus_id);
    engine.prepare();

    unsafe {
        *ENGINE.get_mut() = Some(engine);
        *REGISTRY.get_mut() = Some(registry);
        *BUS_NODE.get_mut() = Some(bus_id);
        *DEF_REGISTRY.get_mut() = Some(BTreeMap::new());
        *DEFS.get_mut() = None;
        *LEGATO_MODES.get_mut() = Some(BTreeMap::new());
        *LEGATO_VOICES.get_mut() = Some(BTreeMap::new());
        *LEGATO_SLOTS.get_mut() = Some(BTreeMap::new());
    }
}

/// Register a named SynthDef. Compiles the DSL source and stores the first
/// resulting SynthDef under the given name.
/// Returns 0 on success, 1 on error.
///
/// # Safety
/// `name_ptr`/`source_ptr` must each point to an initialized buffer of at
/// least `name_len`/`source_len` bytes that stays valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_register_def(
    name_ptr: *const u8,
    name_len: usize,
    source_ptr: *const u8,
    source_len: usize,
) -> u32 {
    let name_bytes = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
    let name = match core::str::from_utf8(name_bytes) {
        Ok(s) => s,
        Err(_) => return 1,
    };
    let source_bytes = unsafe { core::slice::from_raw_parts(source_ptr, source_len) };
    let source = match core::str::from_utf8(source_bytes) {
        Ok(s) => s,
        Err(_) => return 1,
    };

    let ugen_registry = match unsafe { REGISTRY.get_mut() }.as_ref() {
        Some(r) => r,
        None => return 1,
    };

    // Also parses this source's `voice` declarations (if any); this is
    // additive over the plain `dsl::compile` used elsewhere in this file —
    // it doesn't change what gets registered as the def, only what else gets
    // recorded alongside it.
    let (defs, voice_modes) = match dsl::compile_with_voice_modes(source, ugen_registry) {
        Ok(result) => result,
        Err(_) => return 1,
    };

    if defs.is_empty() {
        return 1;
    }

    let def_registry = match unsafe { DEF_REGISTRY.get_mut() }.as_mut() {
        Some(r) => r,
        None => return 1,
    };

    def_registry.insert(AllocString::from(name), defs.into_iter().next().unwrap());

    // Record this name's mono/legato voice mode, if its source declared one.
    if let Some(mode) = voice_modes.iter().find(|m| m.synth_name == name)
        && let Some(modes) = unsafe { LEGATO_MODES.get_mut() }.as_mut()
    {
        modes.insert(
            AllocString::from(name),
            (
                mode.pitch_param.clone(),
                mode.portamento_secs,
                mode.portamento_shape,
                mode.portamento_space,
            ),
        );
    }

    0
}

/// Set a named SynthDef as the master effect, wired between the bus and graph output.
/// Returns 0 on success, 1 on error.
///
/// # Safety
/// `name_ptr` must point to an initialized buffer of at least `name_len`
/// bytes that stays valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_set_bus_master(name_ptr: *const u8, name_len: usize) -> u32 {
    let name_bytes = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
    let name = match core::str::from_utf8(name_bytes) {
        Ok(s) => s,
        Err(_) => return 1,
    };

    let def_registry = match unsafe { DEF_REGISTRY.get_mut() }.as_ref() {
        Some(r) => r,
        None => return 1,
    };
    let def = match def_registry.get(name) {
        Some(d) => d,
        None => return 1,
    };
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return 1,
    };
    let bus_id = match unsafe { BUS_NODE.get_mut() } {
        Some(id) => *id,
        None => return 1,
    };

    let synth = engine.instantiate_synthdef(def);

    // Wire bus output → synth's audioIn node
    if let Some(audio_in_node) = synth.audio_input_node("in") {
        engine.graph_mut().connect(bus_id, audio_in_node, 0);
    } else {
        return 1;
    }

    // Set the synth's output as the new graph sink
    engine.graph_mut().set_sink(synth.output_node());
    engine.prepare();

    unsafe {
        *MASTER_SYNTH.get_mut() = Some(synth);
    }

    0
}

/// Set a parameter on the master effect synth.
///
/// # Safety
/// `param_ptr` must point to an initialized buffer of at least `param_len`
/// bytes that stays valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_master_param(param_ptr: *const u8, param_len: usize, value: f32) {
    let param_bytes = unsafe { core::slice::from_raw_parts(param_ptr, param_len) };
    let param = match core::str::from_utf8(param_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };

    let synth = match unsafe { MASTER_SYNTH.get_mut() }.as_ref() {
        Some(s) => s,
        None => return,
    };
    let node_id = match synth.param_node(param) {
        Some(id) => id,
        None => return,
    };
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return,
    };
    engine.graph_mut().set_node_value(node_id, value);
}

/// Spawn a voice from a named SynthDef onto the bus.
/// Returns voice_id > 0, or 0 on failure.
///
/// # Safety
/// `name_ptr` must point to an initialized buffer of at least `name_len`
/// bytes that stays valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_spawn_voice_named(name_ptr: *const u8, name_len: usize) -> u64 {
    let name_bytes = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
    let name = match core::str::from_utf8(name_bytes) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let def_registry = match unsafe { DEF_REGISTRY.get_mut() }.as_ref() {
        Some(r) => r,
        None => return 0,
    };
    let def = match def_registry.get(name) {
        Some(d) => d,
        None => return 0,
    };
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return 0,
    };
    let bus_id = match unsafe { BUS_NODE.get_mut() } {
        Some(id) => *id,
        None => return 0,
    };

    match engine.spawn_voice_on_bus(def, bus_id) {
        Some(voice_id) => {
            engine.prepare();
            voice_id.0
        }
        None => 0,
    }
}

/// Spawn a voice from a named SynthDef onto the bus at the given stereo pan
/// position. Same lookup/bus-slot behavior as [`ms_spawn_voice_named`]; the
/// only difference is the pan placement (see [`ms_spawn_voice_panned`]'s doc
/// for the pan value's range and the center-pan byte-identity guarantee).
///
/// This is the entry point a per-role/per-instrument render configuration
/// calls: same named-SynthDef lookup as `ms_spawn_voice_named`, plus a pan
/// value read from that configuration.
///
/// Returns voice_id > 0, or 0 on failure.
///
/// # Safety
/// `name_ptr` must point to an initialized buffer of at least `name_len`
/// bytes that stays valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_spawn_voice_named_panned(
    name_ptr: *const u8,
    name_len: usize,
    pan: f32,
) -> u64 {
    let name_bytes = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
    let name = match core::str::from_utf8(name_bytes) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let def_registry = match unsafe { DEF_REGISTRY.get_mut() }.as_ref() {
        Some(r) => r,
        None => return 0,
    };
    let def = match def_registry.get(name) {
        Some(d) => d,
        None => return 0,
    };
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return 0,
    };
    let bus_id = match unsafe { BUS_NODE.get_mut() } {
        Some(id) => *id,
        None => return 0,
    };

    match engine.spawn_voice_on_bus_panned(def, bus_id, pan) {
        Some(voice_id) => {
            engine.prepare();
            voice_id.0
        }
        None => 0,
    }
}

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
///
/// # Safety
/// `ptr`/`capacity` must come from a prior microsynth allocation that has not
/// already been freed; calling with any other value is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_free(ptr: *mut u8, capacity: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, capacity);
    }
}

/// Initialize the engine with the given sample rate.
/// Block size is fixed at 128 (WebAudio render quantum).
///
/// Resets every piece of graph-dependent session state this file tracks —
/// the mono/legato bookkeeping (`LEGATO_MODES`/`LEGATO_VOICES`/`LEGATO_SLOTS`)
/// and `BUS_NODE` — the same way `ms_init_with_bus` does, so a re-init here
/// can't leave either a stale legato track whose held `VoiceId` happens to
/// alias one of the new engine's freshly-issued sequential ids, or a stale
/// `BUS_NODE` pointing at a `NodeId` in the just-destroyed engine's graph
/// (which `ms_spawn_voice_named` would otherwise feed to
/// `Engine::spawn_voice_on_bus` — safe today only because the new, empty
/// graph doesn't have that index yet; it becomes valid-and-wrong the moment
/// a later call repopulates the graph past it).
///
/// `DEF_REGISTRY`/`DEFS` are deliberately left alone: they're graph-
/// independent templates (compiled `SynthDef`s, not live node references),
/// so a name registered before this call is still validly registered after.
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
        *BUS_NODE.get_mut() = None;
        *LEGATO_MODES.get_mut() = Some(BTreeMap::new());
        *LEGATO_VOICES.get_mut() = Some(BTreeMap::new());
        *LEGATO_SLOTS.get_mut() = Some(BTreeMap::new());
    }
}

/// Compile DSL source and load it into the engine.
///
/// `source_ptr` and `source_len` point to a UTF-8 string in WASM memory
/// (previously written via `ms_alloc`).
///
/// Returns 0 on success, 1 on error.
///
/// # Safety
/// `source_ptr` must point to an initialized buffer of at least `source_len`
/// bytes that stays valid for the call.
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
///
/// # Safety
/// `out_left` and `out_right` must each point to a writable buffer of at least
/// 128 `f32`s that stays valid for the call.
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
///
/// # Safety
/// `source_ptr` must point to an initialized buffer of at least `source_len`
/// bytes that stays valid for the call.
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

    // Create a stereo bus node as the graph sink. The input-slot count (voice
    // capacity) is fixed at MAX_BUS_INPUTS regardless of this channel count —
    // see `ugens::bus::ChannelCount`'s doc.
    let bus = crate::ugens::Bus::new(crate::ugens::ChannelCount::Stereo);
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

/// Spawn a voice from the first compiled SynthDef, connected to the bus at
/// the given stereo pan position.
///
/// `pan` ranges -1.0 (left) to +1.0 (right); 0.0 is center. Out-of-range
/// values are clamped by the underlying `Pan2` UGen. Passing `0.0` is
/// equivalent to [`ms_spawn_voice`] — both take the same direct
/// voice-to-bus connection, with no `Pan2` node inserted, so existing
/// center-pan callers see byte-identical output whether they call this or
/// `ms_spawn_voice`.
///
/// This is the config-driven entry point for per-role/per-instrument pan:
/// callers select the pan value from their own render configuration (e.g.
/// a per-role pan table) rather than this crate hardcoding any position.
///
/// Returns the voice ID (> 0), or 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn ms_spawn_voice_panned(pan: f32) -> u64 {
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

    match engine.spawn_voice_on_bus_panned(&defs[0], bus_id, pan) {
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
///
/// # Safety
/// `param_ptr` must point to an initialized buffer of at least `param_len`
/// bytes that stays valid for the call.
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

/// Set a named parameter on a voice with a shaped glide to `target` over
/// `glide_secs` seconds, instead of jumping instantly like `ms_voice_param`.
/// `shape_kind`/`tension`/`space_kind` encode the glide's interpolation
/// shape and space — see the module docs for the exact encoding.
///
/// # Safety
/// `param_ptr` must point to an initialized buffer of at least `param_len`
/// bytes that stays valid for the call.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)] // Voice, param, target, glide, and shape/space are each meaningful and independent.
pub unsafe extern "C" fn ms_voice_param_glide(
    voice_id: u64,
    param_ptr: *const u8,
    param_len: usize,
    target: f32,
    glide_secs: f32,
    shape_kind: u32,
    tension: f32,
    space_kind: u32,
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
    let shape = decode_glide_shape(shape_kind, tension);
    let space = decode_glide_space(space_kind);
    engine.set_voice_param_glide(
        crate::scheduler::VoiceId(voice_id),
        param,
        target,
        glide_secs,
        shape,
        space,
    );
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
// Mono/legato voice-mode exports
// ============================================================================
//
// A SynthDef played legato needs its own single-voice track (see
// `crate::voice::LegatoVoice`) rather than the usual independent-voice-per-
// spawn model `ms_spawn_voice_named` uses. `ms_register_def` records which
// registered names asked for this (via a `voice` declaration in their DSL
// source); these exports drive the resulting track by that same name.

/// Start or continue a legato note at `pitch` on the named SynthDef's
/// mono/legato track, creating the track (and wiring its output to the bus)
/// on first use.
///
/// `name` must have been registered via `ms_register_def` with a `voice`
/// declaration in its DSL source (see `src/dsl` module docs). Returns the
/// voice ID (> 0), or 0 if `name` is unregistered or declared no voice mode.
///
/// # Safety
/// `name_ptr` must point to an initialized buffer of at least `name_len`
/// bytes that stays valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_legato_note_on(
    name_ptr: *const u8,
    name_len: usize,
    pitch: f32,
) -> u64 {
    let name_bytes = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
    let name = match core::str::from_utf8(name_bytes) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let def = match unsafe { DEF_REGISTRY.get_mut() }
        .as_ref()
        .and_then(|r| r.get(name))
    {
        Some(d) => d,
        None => return 0,
    };
    let (pitch_param, portamento_secs, portamento_shape, portamento_space) =
        match unsafe { LEGATO_MODES.get_mut() }
            .as_ref()
            .and_then(|m| m.get(name))
        {
            Some(mode) => mode.clone(),
            None => return 0,
        };
    let tracks = match unsafe { LEGATO_VOICES.get_mut() }.as_mut() {
        Some(t) => t,
        None => return 0,
    };
    let track = tracks.entry(AllocString::from(name)).or_insert_with(|| {
        LegatoVoice::new(pitch_param, portamento_secs)
            .with_glide(portamento_shape, portamento_space)
    });
    // Snapshot the voice held *before* the call, not just whether one was
    // held at all: `LegatoVoice::note_on` can silently spawn a replacement
    // voice for a track that still looks "held" from here, if the
    // previously-held voice was reaped out from under it (e.g. by
    // `Engine::free_done_synths` after a full release) — it clears its own
    // `held` state and spawns fresh internally, after this function has
    // already read it. A `track.voice().is_none()` check taken before the
    // call cannot see that case, so it must be a value comparison taken
    // after the call instead (see the comment below).
    let prev_voice = track.voice();

    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return 0,
    };
    let voice_id = track.note_on(engine, def, pitch);

    // A voice id changing from what was held before (including from "none
    // held yet") means a brand-new synth was just spawned into the graph
    // and needs wiring to the bus — this covers both the true first note
    // and a reap-then-respawn, since `Engine::spawn_voice` always hands out
    // a fresh, never-before-used id *within one engine's lifetime*
    // (`Scheduler::alloc_voice_id` is monotonic, never reused). That
    // monotonicity is what makes this comparison sound — but the counter
    // resets to 1 whenever `ms_init`/`ms_init_with_bus` replaces the engine,
    // so this is only correct because both of them also clear
    // `LEGATO_VOICES`. If a future edit trims that reset list, a track
    // surviving a re-init could hold a `prev_voice` that collides with a
    // fresh id in the *new* engine and this comparison would wrongly see
    // "unchanged," silently skipping wiring again.
    if prev_voice != Some(voice_id)
        && let Some(bus_id) = unsafe { BUS_NODE.get_mut() }
        && let Some(slots) = unsafe { LEGATO_SLOTS.get_mut() }.as_mut()
    {
        // Reuse this track's already-reserved slot if it has one (a
        // reap-then-respawn) rather than recomputing the counting-down
        // formula, which depends on how many *other* tracks have reserved a
        // slot since — recomputing now could collide with a slot a
        // different track already claimed.
        let slot = match slots.get(name).copied() {
            Some(slot) => Some(slot),
            None => engine
                .graph()
                .node_spec(*bus_id)
                .map(|spec| spec.inputs.len())
                .map(|bus_max| {
                    let slot = bus_max.saturating_sub(1).saturating_sub(slots.len());
                    slots.insert(AllocString::from(name), slot);
                    slot
                }),
        };
        if let Some(slot) = slot {
            if let Some(synth) = engine.voice_synth(voice_id) {
                let output_node = synth.output_node();
                engine.graph_mut().connect(output_node, *bus_id, slot);
            }
            engine.prepare();
        }
    }

    voice_id.0
}

/// Release the currently-held note on a legato track, if any (begins the
/// held voice's release stage; the voice stays held for a following
/// `ms_legato_note_on` call — see `LegatoVoice::note_off`).
///
/// A no-op if `name` has no legato track yet.
///
/// # Safety
/// `name_ptr` must point to an initialized buffer of at least `name_len`
/// bytes that stays valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_legato_note_off(name_ptr: *const u8, name_len: usize) {
    let name_bytes = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
    let name = match core::str::from_utf8(name_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };

    let track = match unsafe { LEGATO_VOICES.get_mut() }
        .as_mut()
        .and_then(|t| t.get_mut(name))
    {
        Some(t) => t,
        None => return,
    };
    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return,
    };
    track.note_off(engine);
}

/// The bus input slot reserved for the named legato track, or -1 if `name`
/// has none yet (never played legato, or no engine initialized).
///
/// Purely a diagnostic accessor over `LEGATO_SLOTS` — normal playback never
/// needs it. It exists so a caller (or a test) can confirm that a track's
/// slot stays stable across a reap-then-respawn instead of drifting to a
/// value that could collide with a different track's slot (see
/// `ms_legato_note_on`'s doc comment).
///
/// # Safety
/// `name_ptr` must point to an initialized buffer of at least `name_len`
/// bytes that stays valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ms_legato_slot_for(name_ptr: *const u8, name_len: usize) -> i64 {
    let name_bytes = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
    let name = match core::str::from_utf8(name_bytes) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match unsafe { LEGATO_SLOTS.get_mut() }
        .as_ref()
        .and_then(|s| s.get(name))
    {
        Some(&slot) => slot as i64,
        None => -1,
    }
}

// ============================================================================
// Musical-time sequenced glide export
// ============================================================================

/// Schedule a sequence of musical-time, shaped parameter glides on a voice in
/// one call — the raw-C-ABI counterpart of
/// [`crate::musical_sequence::schedule_musical_glides`]. Each segment `i` is
/// read from the parallel arrays at index `i`; all arrays must have at least
/// `count` elements.
///
/// `sample_rate` is **not** a parameter here, unlike the stateless
/// `ms_position_to_samples`/`ms_steps_to_samples` (which have no engine to
/// consult and so must take it): this export is engine-coupled — it
/// dispatches through the live engine's own scheduler — so it reads the
/// initialized engine's actual sample rate instead of taking a second,
/// independently-supplied one that could silently drift from it and
/// schedule everything at the wrong time with no error signal.
///
/// - `bars_ptr[i]`, `steps_ptr[i]`, `tick_offsets_ptr[i]`: the segment's
///   musical position (see `MusicalPosition`).
/// - `targets_ptr[i]`: the value `param` glides to.
/// - `glide_steps_ptr[i]`: glide length in grid steps (fractional allowed).
/// - `shape_kinds_ptr[i]`, `tensions_ptr[i]`, `space_kinds_ptr[i]`: the
///   glide's shape and space — see the module docs for the encoding.
///
/// Returns 0 on success, 1 if no engine is initialized. `count == 0` is a
/// valid no-op and returns 0 without touching any of the `*_ptr` arguments
/// (so they may be null in that case). `voice_id` is not validated against
/// currently-live voices either: if it names a voice that is gone by the
/// time a segment's event fires (freed directly, or reaped after a legato
/// track respawned under a new id), that segment is a silent no-op rather
/// than an error, matching every other scheduled or targeted export in this
/// file (`ms_voice_param`, `ms_voice_param_glide`, `ms_schedule_note`, …).
///
/// # Safety
/// `param_ptr` must point to an initialized buffer of at least `param_len`
/// bytes. If `count > 0`, each `*_ptr` array must point to at least `count`
/// initialized elements of its element type; all must stay valid for the
/// call.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)] // Voice/param plus most of a musical TimeConfig (sample_rate comes from the engine) plus one array per segment field, all independently meaningful.
pub unsafe extern "C" fn ms_schedule_musical_glides(
    voice_id: u64,
    param_ptr: *const u8,
    param_len: usize,
    bpm: f32,
    numerator: u8,
    denominator: u8,
    grid_steps: u16,
    ppqn: u16,
    bars_ptr: *const u32,
    steps_ptr: *const u16,
    tick_offsets_ptr: *const i16,
    targets_ptr: *const f32,
    glide_steps_ptr: *const f32,
    shape_kinds_ptr: *const u32,
    tensions_ptr: *const f32,
    space_kinds_ptr: *const u32,
    count: usize,
) -> u32 {
    let param_bytes = unsafe { core::slice::from_raw_parts(param_ptr, param_len) };
    let param = match core::str::from_utf8(param_bytes) {
        Ok(s) => s,
        Err(_) => return 1,
    };

    let engine = match unsafe { ENGINE.get_mut() }.as_mut() {
        Some(e) => e,
        None => return 1,
    };

    // An empty segment list is a valid no-op — return before touching any of
    // the array pointers so a caller with nothing to schedule doesn't have
    // to pass non-null (if still aligned) pointers just to satisfy
    // `slice::from_raw_parts`'s never-null-even-when-empty requirement.
    if count == 0 {
        return 0;
    }

    let config = TimeConfig {
        bpm,
        numerator,
        denominator,
        grid_steps,
        ppqn,
        sample_rate: engine.context().sample_rate,
    };

    let bars = unsafe { core::slice::from_raw_parts(bars_ptr, count) };
    let steps = unsafe { core::slice::from_raw_parts(steps_ptr, count) };
    let tick_offsets = unsafe { core::slice::from_raw_parts(tick_offsets_ptr, count) };
    let targets = unsafe { core::slice::from_raw_parts(targets_ptr, count) };
    let glide_steps = unsafe { core::slice::from_raw_parts(glide_steps_ptr, count) };
    let shape_kinds = unsafe { core::slice::from_raw_parts(shape_kinds_ptr, count) };
    let tensions = unsafe { core::slice::from_raw_parts(tensions_ptr, count) };
    let space_kinds = unsafe { core::slice::from_raw_parts(space_kinds_ptr, count) };

    let segments: Vec<MusicalGlideSegment> = (0..count)
        .map(|i| {
            let shape = decode_glide_shape(shape_kinds[i], tensions[i]);
            let space = decode_glide_space(space_kinds[i]);
            let position = MusicalPosition::new(bars[i], steps[i], tick_offsets[i]);
            MusicalGlideSegment::new(position, targets[i], glide_steps[i], shape).with_space(space)
        })
        .collect();

    schedule_musical_glides(
        engine.scheduler_mut(),
        &config,
        crate::scheduler::VoiceId(voice_id),
        param,
        &segments,
    );
    0
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
    /// Handle to the currently loaded synth, so parameters can be addressed
    /// by name after `compileAndLoad` (see `setParamGlide`). `None` before
    /// the first successful load.
    synth: Option<crate::synthdef::Synth>,
}

#[cfg(feature = "web")]
#[wasm_bindgen]
impl WebSynth {
    /// Create a new synthesizer.
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32, block_size: usize) -> WebSynth {
        let mut registry = UGenRegistry::new();
        register_builtins(&mut registry);

        let config = EngineConfig {
            sample_rate,
            block_size,
        };
        let engine = Engine::new(config);

        WebSynth {
            engine,
            registry,
            num_channels: 0,
            synth: None,
        }
    }

    /// Compile DSL source and load the first SynthDef.
    /// Returns an error string on failure.
    #[wasm_bindgen(js_name = "compileAndLoad")]
    pub fn compile_and_load(&mut self, source: &str) -> Result<(), JsError> {
        let defs =
            dsl::compile(source, &self.registry).map_err(|e| JsError::new(&e.to_string()))?;

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
        self.synth = Some(synth);

        Ok(())
    }

    /// Set a named parameter on the loaded synth with a shaped glide to
    /// `target` over `glide_secs` seconds (see `ms_voice_param_glide`'s doc
    /// comment and the module docs for the `shape_kind`/`space_kind`
    /// encoding). Returns `false` if no synth is loaded or the name is
    /// unknown.
    #[wasm_bindgen(js_name = "setParamGlide")]
    #[allow(clippy::too_many_arguments)] // Param, target, glide, and shape/space are each meaningful and independent.
    pub fn set_param_glide(
        &mut self,
        name: &str,
        target: f32,
        glide_secs: f32,
        shape_kind: u32,
        tension: f32,
        space_kind: u32,
    ) -> bool {
        let Some(synth) = &self.synth else {
            return false;
        };
        let shape = decode_glide_shape(shape_kind, tension);
        let space = decode_glide_space(space_kind);
        self.engine
            .set_param_glide(synth, name, target, glide_secs, shape, space)
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
        "sinOsc",
        "saw",
        "pulse",
        "tri",
        "phasor",
        "whiteNoise",
        "pinkNoise",
        "onePole",
        "lpf",
        "hpf",
        "bpf",
        "line",
        "perc",
        "asr",
        "adsr",
        "delay",
        "pan2",
        "mix",
        "sampleAndHold",
        "impulse",
        "lag",
        "clip",
        "waveTable",
        "sinTable",
        "sawTable",
        "triTable",
        "squareTable",
    ];
    names.iter().map(|&n| JsValue::from_str(n)).collect()
}
