//! Shared helpers for the integration tests.
//!
//! Each file under `tests/` is compiled as its own crate, so this module is
//! included separately into each one and any given helper will be unused in
//! most of them — hence the blanket `dead_code` allow.
#![allow(dead_code)]

use microsynth::dsl::UGenRegistry;
use microsynth::ugens::register_builtins;

/// A registry with every built-in UGen registered.
///
/// This is the registry almost every test wants. Tests that need their own
/// stub UGens build a registry directly instead (see `tests/dsl.rs`).
pub fn builtin_registry() -> UGenRegistry {
    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);
    reg
}
