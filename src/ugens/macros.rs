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
    // With an explicit category, every listed port required (the common
    // case). Delegates to the `optional_inputs`-aware arm with an empty list.
    (
        $name:literal,
        category = $category:ident,
        inputs = [$($input:literal),* $(,)?],
        outputs = [$($output:literal),* $(,)?] $(,)?
    ) => {
        ugen_spec!(
            $name,
            category = $category,
            inputs = [$($input),*],
            optional_inputs = [],
            outputs = [$($output),*]
        );
    };
    // With an explicit category and a trailing list of optional ports —
    // ports the UGen's own `process` defaults when unconnected (see
    // `InputSpec::required`), rather than requiring a connection. Optional
    // ports are appended after the required ones, in the order given here:
    // port index in the resulting spec is
    // `required.len() + position in optional_inputs`, and every `process`
    // impl using this arm must read ports in that same order.
    (
        $name:literal,
        category = $category:ident,
        inputs = [$($input:literal),* $(,)?],
        optional_inputs = [$($opt:literal),* $(,)?],
        outputs = [$($output:literal),* $(,)?] $(,)?
    ) => {
        fn spec(&self) -> $crate::node::UGenSpec {
            static INPUTS: &[$crate::node::InputSpec] = &[
                $(
                    $crate::node::InputSpec {
                        name: $input,
                        rate: $crate::context::Rate::Audio,
                        required: true,
                    },
                )*
                $(
                    $crate::node::InputSpec {
                        name: $opt,
                        rate: $crate::context::Rate::Audio,
                        required: false,
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
                category: $crate::node::UGenCategory::$category,
                inputs: INPUTS,
                outputs: OUTPUTS,
            }
        }
    };
    // Without a category — defaults to `Utility`, every listed port required.
    (
        $name:literal,
        inputs = [$($input:literal),* $(,)?],
        outputs = [$($output:literal),* $(,)?] $(,)?
    ) => {
        ugen_spec!(
            $name,
            category = Utility,
            inputs = [$($input),*],
            optional_inputs = [],
            outputs = [$($output),*]
        );
    };
    // Without a category, with a trailing list of optional ports.
    (
        $name:literal,
        inputs = [$($input:literal),* $(,)?],
        optional_inputs = [$($opt:literal),* $(,)?],
        outputs = [$($output:literal),* $(,)?] $(,)?
    ) => {
        ugen_spec!(
            $name,
            category = Utility,
            inputs = [$($input),*],
            optional_inputs = [$($opt),*],
            outputs = [$($output),*]
        );
    };
}
