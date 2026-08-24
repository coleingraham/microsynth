//! Regression gate: the raw C-ABI export list that existed before shaped
//! parameter glides, mono/legato notes, and musical-time sequencing were
//! added to `src/web.rs` must keep its exact signatures. New exports may be
//! added, but none of these may change — the consuming render path calls
//! them with a fixed argument layout and must keep working untouched.
//!
//! This is a textual check rather than a type-level one on purpose: it
//! catches the case a type-level check can't, an incidental reformatting or
//! reordering of parameters that would still typecheck (e.g. swapping two
//! same-typed arguments) but would silently break every existing caller.
//!
//! DELIBERATE RETIREMENT: `ms_set_bus_master` is removed from
//! `PRE_EXISTING_SIGNATURES` below. It was exported dead code: zero call
//! sites anywhere reachable at runtime (the one JS wrapper that called it,
//! `processor.js`'s `setMasterEffect`, was itself only reachable from a
//! `postMessage` case nothing in the repo ever sends). This is the ONE
//! sanctioned removal from this list; any other signature disappearing is
//! still a real regression this test must catch.

const WEB_RS: &str = include_str!("../src/web.rs");

/// Every `ms_*` raw export signature that existed before this change,
/// copied verbatim from `src/web.rs`.
const PRE_EXISTING_SIGNATURES: &[&str] = &[
    "pub extern \"C\" fn ms_init_with_bus(sample_rate: f32) {",
    "pub unsafe extern \"C\" fn ms_register_def(\n    name_ptr: *const u8,\n    name_len: usize,\n    source_ptr: *const u8,\n    source_len: usize,\n) -> u32 {",
    "pub unsafe extern \"C\" fn ms_master_param(param_ptr: *const u8, param_len: usize, value: f32) {",
    "pub unsafe extern \"C\" fn ms_spawn_voice_named(name_ptr: *const u8, name_len: usize) -> u64 {",
    "pub extern \"C\" fn ms_alloc(size: usize) -> *mut u8 {",
    "pub unsafe extern \"C\" fn ms_free(ptr: *mut u8, capacity: usize) {",
    "pub extern \"C\" fn ms_init(sample_rate: f32) {",
    "pub unsafe extern \"C\" fn ms_compile(source_ptr: *const u8, source_len: usize) -> u32 {",
    "pub unsafe extern \"C\" fn ms_render(out_left: *mut f32, out_right: *mut f32) {",
    "pub unsafe extern \"C\" fn ms_compile_def(source_ptr: *const u8, source_len: usize) -> u32 {",
    "pub extern \"C\" fn ms_spawn_voice() -> u64 {",
    "pub extern \"C\" fn ms_voice_gate(voice_id: u64, value: f32) {",
    "pub unsafe extern \"C\" fn ms_voice_param(\n    voice_id: u64,\n    param_ptr: *const u8,\n    param_len: usize,\n    value: f32,\n) {",
    "pub extern \"C\" fn ms_free_voice(voice_id: u64) {",
    "pub extern \"C\" fn ms_free_done() -> u32 {",
    "pub extern \"C\" fn ms_midi_to_hz(midi_note: f32, a4_freq: f32) -> f32 {",
    "pub extern \"C\" fn ms_hz_to_midi(hz: f32, a4_freq: f32) -> f32 {",
    "pub extern \"C\" fn ms_apply_cents(base_hz: f32, cents: f32) -> f32 {",
    "pub extern \"C\" fn ms_position_to_samples(\n    bpm: f32,\n    numerator: u8,\n    denominator: u8,\n    grid_steps: u16,\n    ppqn: u16,\n    sample_rate: f32,\n    bar: u32,\n    step: u16,\n    tick_offset: i16,\n) -> u64 {",
    "pub extern \"C\" fn ms_steps_to_samples(\n    bpm: f32,\n    numerator: u8,\n    denominator: u8,\n    grid_steps: u16,\n    sample_rate: f32,\n    steps: f32,\n) -> u64 {",
    "pub extern \"C\" fn ms_schedule_note(voice_id: u64, on_time: u64, off_time: u64) {",
    "pub extern \"C\" fn ms_schedule_note_aligned(\n    voice_id: u64,\n    grid_time: u64,\n    attack_secs: f32,\n    duration_samples: u64,\n) {",
];

#[test]
fn test_all_pre_existing_raw_exports_unchanged() {
    for sig in PRE_EXISTING_SIGNATURES {
        assert!(
            WEB_RS.contains(sig),
            "pre-existing raw C-ABI signature changed or went missing:\n{sig}"
        );
    }
}

/// The new exports this change adds are additions, not replacements: none of
/// their names collide with a pre-existing one, and the pre-existing count
/// of `#[unsafe(no_mangle)]` raw exports plus these new ones matches the
/// total currently in the file (i.e. nothing was deleted).
#[test]
fn test_new_exports_are_additive_not_replacements() {
    let new_names = [
        "ms_voice_param_glide",
        "ms_legato_note_on",
        "ms_legato_note_off",
        "ms_legato_slot_for",
        "ms_schedule_musical_glides",
        "ms_routing_init",
        "ms_routing_add_bus",
        "ms_register_effect_def",
        "ms_routing_add_effect",
        "ms_routing_build",
        "ms_spawn_voice_on_routing_bus_named",
        "ms_spawn_voice_panned",
        "ms_spawn_voice_named_panned",
        // Bus/effect output taps -- two separate exports, not one, because a
        // bus's raw summed input and an effect's own processed/wet output are different
        // signals (see their doc comments in src/web.rs).
        "ms_routing_bus_output",
        "ms_routing_effect_output",
        // MOT-634: runtime coefficient-table bank + upload ABI. Register/replace/free
        // by id, plus name->id resolution -- see src/web.rs's "Coefficient-table bank
        // exports" section doc for the wire format these carry.
        "ms_coeff_table_register",
        "ms_coeff_table_replace",
        "ms_coeff_table_free",
        "ms_coeff_table_id_for_name",
        // MOT-640: wasm-ABI reachability for table-bound nodes. Takes a
        // serialized `ir::IrSynthDef` (not DSL text) and resolves its
        // `table_bindings` via `IrSynthDef::compile_with_tables` -- see
        // src/web.rs's doc comment on this export.
        "ms_compile_ir_with_tables",
        // Named-IR registration: like `ms_register_def`, but the source is
        // an `ir::IrSynthDef`'s JSON text form (`IrSynthDef::from_json`)
        // rather than DSL source -- see src/web.rs's doc comment on this
        // export.
        "ms_register_def_ir",
    ];
    for name in new_names {
        assert!(
            WEB_RS.contains(&format!("fn {name}(")),
            "expected new export {name} to be present"
        );
    }

    let total_no_mangle = WEB_RS.matches("#[unsafe(no_mangle)]").count();
    // One per pre-existing signature, plus one per new export.
    assert_eq!(
        total_no_mangle,
        PRE_EXISTING_SIGNATURES.len() + new_names.len(),
        "expected exactly the pre-existing exports plus the new ones, no fewer and no more"
    );
}
