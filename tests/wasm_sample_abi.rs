//! The sample upload ABI end to end, driven through the raw `ms_*` exports
//! the way a wasm host drives them: upload a sample and a program, register
//! an IR def whose `sampleVoice` node is bound to the program, spawn a voice
//! and render it. The exports are backed by process-global statics, so the
//! tests in this file share one lock.

#![cfg(feature = "ir")]

use microsynth::ir::{IrEdge, IrNode, IrParam, IrSynthDef, IrTableBinding, SynthDefClass};
use microsynth::sampler::SamplePcm;
use microsynth::sampler::program::{Region, SampleProgram};
use microsynth::web;
use std::sync::Mutex;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Call `f` with `bytes` copied into an `ms_alloc`'d buffer.
fn with_bytes<R>(bytes: &[u8], f: impl FnOnce(*const u8, usize) -> R) -> R {
    let len = bytes.len();
    let ptr = web::ms_alloc(len);
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len) };
    let r = f(ptr, len);
    unsafe { web::ms_free(ptr, len) };
    r
}

fn sample(hash: u8) -> SamplePcm {
    SamplePcm::new("abi", [hash; 32], 44_100.0, &[&vec![0.5f32; 8_820][..]]).unwrap()
}

fn program(hash: u8) -> SampleProgram {
    SampleProgram {
        name: "abi__program".into(),
        spec_sha256: [0x5A; 32],
        samples: vec![[hash; 32]],
        regions: vec![Region {
            amp_veltrack: 0.0,
            ..Region::default()
        }],
    }
}

fn def_json(pid: u32) -> String {
    IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "abi".into(),
        class: SynthDefClass::Source,
        output_channels: 2,
        nodes: vec![
            IrNode::Param {
                name: "freq".into(),
                default: 261.625_58,
            },
            IrNode::Param {
                name: "gate".into(),
                default: 0.0,
            },
            IrNode::UGen {
                kind: "sampleVoice".into(),
                consts: vec![],
            },
        ],
        edges: vec![
            IrEdge {
                from: 0,
                to: 2,
                to_input: 0,
            },
            IrEdge {
                from: 1,
                to: 2,
                to_input: 1,
            },
        ],
        params: vec![
            IrParam {
                name: "freq".into(),
                node: 0,
                input: 0,
                default: 261.625_58,
            },
            IrParam {
                name: "gate".into(),
                node: 1,
                input: 0,
                default: 0.0,
            },
        ],
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding {
            node: 2,
            table_id: pid,
        }],
        output_node: 2,
    }
    .to_json()
}

#[test]
fn upload_bind_spawn_and_render_through_the_raw_exports() {
    let _g = lock();
    web::ms_init_with_bus(44_100.0);
    assert_eq!(web::ms_sample_bytes_total(), 0.0);
    let s = sample(1);
    let sid = with_bytes(&s.to_bytes(), |p, n| unsafe {
        web::ms_sample_register(p, n)
    });
    assert!(sid > 0);
    assert_eq!(
        with_bytes(&s.to_bytes(), |p, n| unsafe {
            web::ms_sample_register(p, n)
        }),
        sid,
        "same content hash, same id"
    );
    assert_eq!(web::ms_sample_bytes_total(), s.bytes() as f64);
    assert_eq!(
        with_bytes(&[1u8; 32], |p, n| unsafe {
            web::ms_sample_id_for_hash(p, n)
        }),
        sid
    );
    let hex = "01".repeat(32);
    assert_eq!(
        with_bytes(hex.as_bytes(), |p, n| unsafe {
            web::ms_sample_id_for_hash(p, n)
        }),
        sid
    );

    let pid = with_bytes(&program(1).to_bytes(), |p, n| unsafe {
        web::ms_sample_program_register(p, n)
    });
    assert!(pid > 0);
    assert_eq!(web::ms_sample_program_bytes(pid), s.bytes() as f64);
    assert_eq!(
        with_bytes(b"abi__program", |p, n| unsafe {
            web::ms_sample_program_id_for_name(p, n)
        }),
        pid
    );
    assert!(web::ms_sample_voice_bytes() > 0);

    let json = def_json(pid);
    let r = with_bytes(b"abi", |np, nn| {
        with_bytes(json.as_bytes(), |jp, jn| unsafe {
            web::ms_register_def_ir(np, nn, jp, jn)
        })
    });
    assert_eq!(r, 0, "the IR def binds to the program");
    let voice = with_bytes(b"abi", |p, n| unsafe { web::ms_spawn_voice_named(p, n) });
    assert!(voice > 0);
    web::ms_voice_gate(voice, 1.0);
    let mut l = vec![0.0f32; 128];
    let mut r = vec![0.0f32; 128];
    let mut peak = 0.0f32;
    for _ in 0..20 {
        unsafe { web::ms_render(l.as_mut_ptr(), r.as_mut_ptr()) };
        peak = l.iter().chain(&r).fold(peak, |m, x| m.max(x.abs()));
    }
    assert!((peak - 0.5 * 0.998_814_9).abs() < 1e-4, "{peak}");
}

#[test]
fn a_program_whose_sample_is_missing_or_a_bad_upload_is_refused() {
    let _g = lock();
    web::ms_init_with_bus(44_100.0);
    assert_eq!(
        with_bytes(&program(9).to_bytes(), |p, n| unsafe {
            web::ms_sample_program_register(p, n)
        }),
        0,
        "no sample with that hash"
    );
    assert_eq!(
        with_bytes(b"MSSPgarbage", |p, n| unsafe {
            web::ms_sample_register(p, n)
        }),
        0
    );
    // an unregistered program id fails the IR registration
    let json = def_json(77);
    let r = with_bytes(b"x", |np, nn| {
        with_bytes(json.as_bytes(), |jp, jn| unsafe {
            web::ms_register_def_ir(np, nn, jp, jn)
        })
    });
    assert_eq!(r, 1);
    // a new session starts with empty banks
    let sid = with_bytes(&sample(2).to_bytes(), |p, n| unsafe {
        web::ms_sample_register(p, n)
    });
    assert!(sid > 0);
    web::ms_init_with_bus(44_100.0);
    assert_eq!(web::ms_sample_bytes_total(), 0.0);
    web::ms_sample_free(sid);
    web::ms_sample_program_free(1);
}
