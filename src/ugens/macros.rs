//! Internal declarative macros shared across the UGen implementations.
//!
//! These are crate-private helpers brought into scope for every `ugens`
//! submodule via `#[macro_use] mod macros;` in `ugens/mod.rs`. They are not
//! part of the public API.

/// Generate the [`UGen::spec`](crate::node::UGen::spec) method for a UGen.
///
/// Every UGen must return a [`UGenSpec`](crate::node::UGenSpec) describing its
/// name and its input/output ports. Hand-written, that means declaring two
/// `static` arrays of [`InputSpec`](crate::node::InputSpec) /
/// [`OutputSpec`](crate::node::OutputSpec) and a `spec()` method that points at
/// them — ~15 lines of near-identical boilerplate per UGen, repeated ~60 times.
/// Since **every** built-in port runs at [`Rate::Audio`](crate::context::Rate),
/// the only real information is the UGen's name and its ordered port names.
///
/// This macro captures exactly that. It expands to a single `spec()` method, so
/// it is invoked **inside** an `impl UGen for T { ... }` block (trait method
/// bodies cannot be split across multiple `impl` blocks, so the port arrays are
/// declared as function-local `static`s inside the generated method):
///
/// ```ignore
/// impl UGen for Saw {
///     ugen_spec!("Saw", inputs = ["freq"], outputs = ["out"]);
///
///     fn init(&mut self, ctx: &ProcessContext) { self.sample_rate = ctx.sample_rate; }
///     fn reset(&mut self) { self.phase = 0.0; }
///     fn process(&mut self, /* ... */) { /* ... */ }
/// }
/// ```
///
/// The invocation above is equivalent to hand-writing a `SAW_INPUTS`/
/// `SAW_OUTPUTS` static pair plus a `spec()` that references them.
///
/// Notes:
/// - Every port is created at `Rate::Audio`; this matches all current built-in
///   UGens. A UGen that genuinely needs `Rate::Control` ports, or that computes
///   its ports at runtime (see `bus::Bus`), should hand-write `spec()` instead.
/// - Port lists may be empty, e.g. `ugen_spec!("WhiteNoise", inputs = [],
///   outputs = ["out"]);`.
/// - Fully-qualified `$crate::...` paths are used throughout, so a UGen module
///   does not need `InputSpec`/`OutputSpec`/`UGenSpec`/`Rate` in scope to invoke
///   the macro.
/// - The generated function-local `static`s (`INPUTS`/`OUTPUTS`) live in each
///   `spec()` method's own scope, so there is no name collision between UGens,
///   and — being `'static` — they are returned by reference at zero cost, just
///   like the hand-written module-level `static`s they replace.
macro_rules! ugen_spec {
    (
        $name:literal,
        inputs = [$($input:literal),* $(,)?],
        outputs = [$($output:literal),* $(,)?] $(,)?
    ) => {
        fn spec(&self) -> $crate::node::UGenSpec {
            static INPUTS: &[$crate::node::InputSpec] = &[
                $(
                    $crate::node::InputSpec {
                        name: $input,
                        rate: $crate::context::Rate::Audio,
                    },
                )*
            ];
            static OUTPUTS: &[$crate::node::OutputSpec] = &[
                $(
                    $crate::node::OutputSpec {
                        name: $output,
                        rate: $crate::context::Rate::Audio,
                    },
                )*
            ];
            $crate::node::UGenSpec {
                name: $name,
                inputs: INPUTS,
                outputs: OUTPUTS,
            }
        }
    };
}
