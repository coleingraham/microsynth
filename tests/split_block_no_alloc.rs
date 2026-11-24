//! A block rendered in pieces allocates no more than the same block rendered
//! whole: the per-node whole-block buffers are sized when the graph is
//! prepared, so neither the first split block nor a block after a later
//! voice spawn touches the allocator for them.

use microsynth::dsl;
use microsynth::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

mod common;
use common::builtin_registry;

struct Counting;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

fn bump() {
    // `try_with`: the allocator also runs while thread-locals are torn down.
    let _ = ON.try_with(|on| {
        if on.get() {
            let _ = COUNT.try_with(|c| c.set(c.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc_zeroed(l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Allocations made by one `render` call.
fn allocs_in_render(engine: &mut Engine) -> usize {
    COUNT.with(|c| c.set(0));
    ON.with(|o| o.set(true));
    let ok = engine.render().is_some();
    ON.with(|o| o.set(false));
    assert!(ok, "the bus is the sink");
    COUNT.with(|c| c.get())
}

/// Schedule a gate on and off at `a` and `b` samples into the block that
/// starts at `block_start`, on `voice`.
fn gate_pair(engine: &mut Engine, voice: VoiceId, block_start: u64, a: u64, b: u64) {
    engine
        .scheduler_mut()
        .schedule_gate(block_start + a, voice, 1.0);
    engine
        .scheduler_mut()
        .schedule_gate(block_start + b, voice, 0.0);
}

#[test]
fn a_split_block_allocates_no_more_than_a_whole_one() {
    const B: u64 = 128;
    let def = dsl::compile("synthdef g gate=0.0 = gate * 0.5", &builtin_registry())
        .unwrap()
        .remove(0);
    let mut engine = Engine::new(EngineConfig {
        sample_rate: 44_100.0,
        block_size: B as usize,
    });
    let bus = engine
        .graph_mut()
        .add_node(Box::new(ugens::Bus::new(ugens::ChannelCount::Stereo)));
    engine.graph_mut().set_sink(bus);
    let v1 = engine.spawn_voice_on_bus(&def, bus).unwrap();
    engine.prepare();

    allocs_in_render(&mut engine); // warm: the first block of any engine

    // The same two events, on the block start (renders whole) and inside
    // the block (renders in three pieces).
    gate_pair(&mut engine, v1, B, 0, 0);
    let whole = allocs_in_render(&mut engine);
    gate_pair(&mut engine, v1, 2 * B, 37, 50);
    let split = allocs_in_render(&mut engine);
    assert_eq!(split, whole, "first split block");

    // A voice spawned later has new nodes: its first split block is clean too.
    let v2 = engine.spawn_voice_on_bus(&def, bus).unwrap();
    engine.prepare();
    gate_pair(&mut engine, v2, 3 * B, 0, 0);
    let whole = allocs_in_render(&mut engine);
    gate_pair(&mut engine, v2, 4 * B, 41, 90);
    let split = allocs_in_render(&mut engine);
    assert_eq!(split, whole, "split block after a spawn");
}

/// A `guitarString` voice plucks, bends and releases without allocating:
/// the pluck (filter design, step response) on a split block, a pitch moved
/// every few samples inside blocks, a palm mute and a release, each block
/// allocating no more than an idle whole block.
#[test]
fn a_guitar_string_plucks_bends_and_releases_without_allocating() {
    const B: u64 = 128;
    let def = dsl::compile(
        "synthdef gs freq=220.0 gate=0.0 damp=0.0 = guitarString freq gate 0.8 (-1.0) 140.0 20.0 0.5 10.0 damp",
        &builtin_registry(),
    )
    .unwrap()
    .remove(0);
    let mut engine = Engine::new(EngineConfig {
        sample_rate: 44_100.0,
        block_size: B as usize,
    });
    let bus = engine
        .graph_mut()
        .add_node(Box::new(ugens::Bus::new(ugens::ChannelCount::Stereo)));
    engine.graph_mut().set_sink(bus);
    let v = engine.spawn_voice_on_bus(&def, bus).unwrap();
    engine.prepare();
    allocs_in_render(&mut engine); // warm
    let idle = allocs_in_render(&mut engine);

    // Pluck inside a block.
    engine.scheduler_mut().schedule_gate(2 * B + 37, v, 1.0);
    assert!(allocs_in_render(&mut engine) <= idle, "pluck block");
    // A vibrato: the pitch moves at several points inside each block.
    for blk in 3..40u64 {
        for k in 0..4u64 {
            let t = blk * B + k * 32 + 5;
            let f = 220.0 * (1.0 + 0.02 * ((t as f32) * 0.002).sin());
            engine.scheduler_mut().schedule_set_param(t, v, "freq", f);
        }
        if blk == 20 {
            engine
                .scheduler_mut()
                .schedule_set_param(blk * B + 64, v, "damp", 0.6);
        }
        assert!(allocs_in_render(&mut engine) <= idle, "vibrato block {blk}");
    }
    // Release inside a block, then let it ring out.
    engine.scheduler_mut().schedule_gate(40 * B + 90, v, 0.0);
    for blk in 40..80 {
        assert!(allocs_in_render(&mut engine) <= idle, "release block {blk}");
    }
}
