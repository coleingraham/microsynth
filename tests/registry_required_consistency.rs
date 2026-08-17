//! `UGenRegistry`'s `required` flags are recorded once, at registration
//! time, by copying each port's `InputSpec::required` off a probe instance
//! (`register`/`register_spec`/`register_table_bound` in
//! `src/dsl/compiler.rs`). That copy is exactly what `IrSynthDef::validate`
//! (`src/ir/mod.rs`) and the DSL compiler's `Expr::App` (both consulted
//! *instead of* re-probing a fresh instance, for speed) trust to describe
//! every currently-constructible UGen. A hand-written registration whose
//! `InputSpec` literals drift from the UGen's own `spec()` -- most
//! plausible for `register`/`register_table_bound` call sites that spell
//! `InputSpec`s out by hand rather than deriving them via `register_spec`
//! -- would silently desync the registry from the ugens it describes,
//! with no compile-time signal. This test is that signal: it probes every
//! registered kind fresh and checks the registry's recorded `required`
//! against the probe's own `spec()`.

use microsynth::coeff_table::CoeffTable;
use microsynth::dsl::UGenRegistry;
use microsynth::ugens::{register_builtins, register_table_bound_builtins};
use std::sync::Arc;

#[test]
fn every_bare_registry_entry_matches_its_own_probe_spec() {
    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let mut checked = 0;
    for (name, entry) in reg.iter() {
        let probe = (entry.factory)();
        let spec = probe.spec();
        assert_eq!(
            entry.required.len(),
            spec.inputs.len(),
            "{name}: registry has {} required flags but spec() has {} ports",
            entry.required.len(),
            spec.inputs.len()
        );
        for (i, input) in spec.inputs.iter().enumerate() {
            assert_eq!(
                entry.required[i], input.required,
                "{name}: port {i} ({}) — registry says required={}, spec() says required={}",
                input.name, entry.required[i], input.required
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "registry should have registered builtins");
}

#[test]
fn every_table_bound_registry_entry_matches_its_own_probe_spec() {
    let mut reg = UGenRegistry::new();
    register_table_bound_builtins(&mut reg);

    // A well-formed, empty table (zero entries) constructs every table-bound
    // kind's factory without panicking — see `PartialsNoise::new`'s doc.
    let probe_table = Arc::new(CoeffTable::default());

    let mut checked = 0;
    for (name, entry) in reg.table_bound_iter() {
        let probe = (entry.factory)(probe_table.clone());
        let spec = probe.spec();
        assert_eq!(
            entry.required.len(),
            spec.inputs.len(),
            "{name}: registry has {} required flags but spec() has {} ports",
            entry.required.len(),
            spec.inputs.len()
        );
        for (i, input) in spec.inputs.iter().enumerate() {
            assert_eq!(
                entry.required[i], input.required,
                "{name}: port {i} ({}) — registry says required={}, spec() says required={}",
                input.name, entry.required[i], input.required
            );
        }
        checked += 1;
    }
    assert!(
        checked > 0,
        "registry should have registered at least one table-bound kind"
    );
}
