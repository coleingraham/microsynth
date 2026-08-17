//! Regression coverage for a DSL source that under-supplies a UGen's
//! required input ports: `Expr::App`'s compiler connects positional
//! arguments to ports as a prefix (0..args.len()), so a call with fewer
//! arguments than the UGen declares leaves the trailing ports unconnected.
//! When one of those unconnected trailing ports is required, that used to
//! reach `AudioGraph::prepare`'s fail-loud panic (or, before required/
//! optional ports were tracked at all, an out-of-bounds index inside the
//! UGen's own `process`) — a panic with no `validate` in front of it on
//! this path, unlike the IR/JSON compile path. The compiler now catches
//! the gap itself and returns a `CompileError` naming the ugen and port.

use microsynth::dsl::{UGenRegistry, compile, compile_synthdef};
use microsynth::ugens::register_builtins;
use microsynth::web;
use std::sync::Mutex;

fn registry() -> UGenRegistry {
    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);
    reg
}

/// `sampleAndHold` declares two required ports (`in`, `trig`, neither
/// optional — see `src/ugens/utility.rs`'s `SampleAndHold` spec). Supplying
/// only one positional argument leaves `trig` unconnected.
const UNDER_SUPPLIED_SOURCE: &str = "synthdef test x=1.0 = sampleAndHold x";

/// Compiling a synthdef through `compile_synthdef` (the direct, non-wasm
/// entry point every other compile path funnels through) must reject an
/// under-supplied required port as a `CompileError`, not build a graph that
/// would later panic in `AudioGraph::prepare`.
#[test]
fn compile_synthdef_rejects_an_under_supplied_required_port() {
    let reg = registry();
    let tokens = microsynth::dsl::lexer::tokenize(UNDER_SUPPLIED_SOURCE).unwrap();
    let mut parser = microsynth::dsl::parser::Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let decl = &program.defs[0];

    match compile_synthdef(decl, &reg) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("sampleAndHold"),
                "error should name the ugen: {msg}"
            );
            assert!(
                msg.contains("trig"),
                "error should name the missing port: {msg}"
            );
        }
        Ok(_) => panic!(
            "expected a CompileError for an under-supplied required port, \
             compilation succeeded instead"
        ),
    }
}

/// Same source through the top-level `dsl::compile` (tokenize + parse +
/// compile) pipeline, which is what `ms_compile` calls underneath.
#[test]
fn dsl_compile_rejects_an_under_supplied_required_port() {
    let reg = registry();
    assert!(
        compile(UNDER_SUPPLIED_SOURCE, &reg).is_err(),
        "dsl::compile should reject an under-supplied required port"
    );
}

// `ms_compile` is backed by process-global statics shared across every test
// in a binary that touches them; serialize access the same way
// tests/wasm_raw_abi.rs does.
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn alloc_str(s: &str) -> (*mut u8, usize) {
    let len = s.len();
    let ptr = web::ms_alloc(len);
    unsafe {
        core::ptr::copy_nonoverlapping(s.as_ptr(), ptr, len);
    }
    (ptr, len)
}

fn free_str(ptr: *mut u8, len: usize) {
    unsafe { web::ms_free(ptr, len) };
}

/// The C-ABI path a wasm host actually calls: `ms_compile` must return the
/// documented error code (1) for an under-supplied required port, not
/// panic. A panic here would unwind across an `extern "C"` boundary a real
/// wasm host cannot recover from -- exactly the failure mode this test
/// exists to rule out.
#[test]
fn ms_compile_returns_an_error_code_for_an_under_supplied_required_port() {
    let _guard = lock();
    web::ms_init(44100.0);
    let (ptr, len) = alloc_str(UNDER_SUPPLIED_SOURCE);
    let result = std::panic::catch_unwind(|| unsafe { web::ms_compile(ptr, len) });
    free_str(ptr, len);

    match result {
        Ok(code) => assert_eq!(
            code, 1,
            "ms_compile should return the documented error code (1), not compile a graph \
             prepare() would refuse to render"
        ),
        Err(_) => panic!(
            "ms_compile panicked instead of returning an error code -- this is the C-ABI \
             half of the under-supplied-required-port gap"
        ),
    }
}

/// `ms_compile_def` (the "compile and register, don't spawn yet" sibling of
/// `ms_compile`, used by callers that want to spawn multiple voices from one
/// compiled def) shares `dsl::compile` underneath, so it is fixed by the same
/// `Expr::App` change -- but it is worth its own test: before that fix, this
/// was the worst-behaved of the three entry points. It returned success (0),
/// registered the malformed def into `DEFS`, and only panicked later, inside
/// `ms_spawn_voice`'s `engine.prepare()` call -- at spawn time, arbitrarily
/// far from the actual compile call and with no error ever surfaced from
/// compilation itself. This asserts both halves stay fixed: the compile call
/// returns nonzero, and nothing gets registered for a later spawn to detonate
/// on.
#[test]
fn ms_compile_def_rejects_an_under_supplied_required_port_and_registers_nothing() {
    let _guard = lock();
    web::ms_init(44100.0);
    let (ptr, len) = alloc_str(UNDER_SUPPLIED_SOURCE);
    let compile_result = unsafe { web::ms_compile_def(ptr, len) };
    free_str(ptr, len);

    assert_ne!(
        compile_result, 0,
        "ms_compile_def should return a nonzero error code for an under-supplied \
         required port, not report success"
    );

    let spawn_result = web::ms_spawn_voice();
    assert_eq!(
        spawn_result, 0,
        "ms_compile_def must not have registered a def for a failed compile -- \
         ms_spawn_voice returning nonzero here would mean it spawned (and \
         engine.prepare()'d) a graph with an unconnected required port"
    );
}
