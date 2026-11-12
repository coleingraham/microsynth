//! `sampleVoice`: every program feature the voice honours, each with a test
//! that fails without it, on tiny synthetic samples.
//!
//! Most tests render whole notes through the engine: one voice per note,
//! spawned on a stereo bus, gates scheduled, finished voices freed by the
//! done rule after every block (as a host does). Sample values are chosen so
//! the output identifies what played: a constant ("DC") sample of level `x`
//! plays at `x · g`, where `g` is the centred mono pan gain (sfizz's
//! `cos(π/4) · 10^(3/20)`), the velocity gain and the envelope.

#![cfg(feature = "ir")]

use microsynth::ir::{IrEdge, IrError, IrNode, IrParam, IrSynthDef, IrTableBinding, SynthDefClass};
use microsynth::sampler::program::{
    Envelope, EqBand, EqKind, Filter, FilterKind, LoopMode, LoopSource, OffMode, Region, SILENCE,
    SampleProgram, Trigger, VelCrossfade,
};
use microsynth::sampler::{ProgramId, SamplePcm, SamplePcmBank, SampleProgramBank};
use microsynth::*;

const SR: f32 = 44_100.0;
const BLOCK: usize = 128;
/// Centred mono pan gain, `cos(π/4) · 1.4125375`.
const MONO: f32 = 0.998_814_9;

fn hash(n: u8) -> [u8; 32] {
    [n; 32]
}

fn dc(n: u8, level: f32, frames: usize) -> SamplePcm {
    SamplePcm::new(format!("dc{n}"), hash(n), SR, &[&vec![level; frames][..]]).unwrap()
}

fn sine(n: u8, freq: f32, rate: f32, frames: usize) -> SamplePcm {
    let x: Vec<f32> = (0..frames)
        .map(|i| (core::f32::consts::TAU * freq * i as f32 / rate).sin() * 0.5)
        .collect();
    SamplePcm::new(format!("sine{n}"), hash(n), rate, &[&x[..]]).unwrap()
}

fn midi_hz(k: f32) -> f32 {
    440.0 * ((k - 69.0) / 12.0).exp2()
}

/// A region on sample `s` covering every key and velocity, keycenter 60,
/// no velocity tracking (so the level is the sample's), instant attack.
fn reg(s: u32) -> Region {
    Region {
        sample: s,
        amp_veltrack: 0.0,
        ampeg: Envelope {
            release: 0.001,
            ..Envelope::default()
        },
        ..Region::default()
    }
}

fn program(samples: &[&SamplePcm], regions: Vec<Region>) -> SampleProgram {
    SampleProgram {
        name: "test".into(),
        spec_sha256: [0xAB; 32],
        samples: samples.iter().map(|s| *s.hash()).collect(),
        regions,
    }
}

struct Banks {
    _samples: SamplePcmBank,
    programs: SampleProgramBank,
    pid: ProgramId,
}

fn banks(samples: Vec<SamplePcm>, prog: SampleProgram) -> Banks {
    let mut sb = SamplePcmBank::new();
    for s in samples {
        sb.register(s);
    }
    let mut pb = SampleProgramBank::new();
    let pid = pb.register(prog, &sb).expect("program binds");
    Banks {
        _samples: sb,
        programs: pb,
        pid,
    }
}

const PORTS: [&str; 7] = ["freq", "gate", "vel", "amp", "seed", "bend", "quality"];

/// An IR def: one Param per named port (gate defaults to 0, so a spawned
/// voice waits for its scheduled note), the voice as the output.
fn voice_ir(pid: ProgramId, ports: &[&str]) -> IrSynthDef {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut params = Vec::new();
    let voice = ports.len();
    for (i, &name) in ports.iter().enumerate() {
        let default = match name {
            "freq" => midi_hz(60.0),
            "gate" => 0.0,
            "vel" => 1.0,
            "amp" => 1.0,
            "quality" => 2.0,
            _ => 0.0,
        };
        nodes.push(IrNode::Param {
            name: name.into(),
            default,
        });
        params.push(IrParam {
            name: name.into(),
            node: i,
            input: 0,
            default,
        });
        let port = PORTS.iter().position(|p| *p == name).unwrap();
        edges.push(IrEdge {
            from: i,
            to: voice,
            to_input: port,
        });
    }
    nodes.push(IrNode::UGen {
        kind: "sampleVoice".into(),
        consts: vec![],
    });
    IrSynthDef {
        format_version: microsynth::ir::FORMAT_VERSION,
        name: "sv".into(),
        class: SynthDefClass::Source,
        output_channels: 2,
        nodes,
        edges,
        params,
        audio_inputs: vec![],
        table_bindings: vec![IrTableBinding {
            node: voice,
            table_id: pid.0,
        }],
        output_node: voice,
    }
}

fn registry() -> dsl::UGenRegistry {
    let mut reg = dsl::UGenRegistry::new();
    ugens::register_builtins(&mut reg);
    ugens::register_table_bound_builtins(&mut reg);
    reg
}

fn compile(b: &Banks, ports: &[&str]) -> SynthDef {
    let reg = registry();
    let ir = voice_ir(b.pid, ports);
    ir.validate(&reg).expect("valid IR");
    ir.compile_with_banks(&reg, &CoeffTableBank::new(), Some(&b.programs))
        .expect("compiles")
}

#[derive(Clone, Copy)]
struct Note {
    at: f32,
    dur: f32,
    key: f32,
    vel: u8,
}

fn note(at: f32, dur: f32, key: f32, vel: u8) -> Note {
    Note { at, dur, key, vel }
}

struct Out {
    l: Vec<f32>,
    r: Vec<f32>,
    /// Voices alive after each block.
    alive: Vec<usize>,
}

impl Out {
    fn at(&self, t: f32) -> f32 {
        self.l[(t * SR) as usize]
    }
    fn mean(&self, a: f32, b: f32) -> f32 {
        let (i, j) = ((a * SR) as usize, (b * SR) as usize);
        (self.l[i..j].iter().map(|&x| x as f64).sum::<f64>() / (j - i) as f64) as f32
    }
    fn rms(&self, a: f32, b: f32) -> f32 {
        let (i, j) = ((a * SR) as usize, (b * SR) as usize);
        (self.l[i..j]
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            / (j - i) as f64)
            .sqrt() as f32
    }
    fn peak_after(&self, t: f32) -> f32 {
        let i = (t * SR) as usize;
        self.l[i..]
            .iter()
            .chain(&self.r[i..])
            .fold(0.0, |m, x| m.max(x.abs()))
    }
    fn alive_at(&self, t: f32) -> usize {
        self.alive[(t * SR) as usize / BLOCK]
    }
}

/// Render `notes` (one voice each) with extra per-voice params.
fn render_with(def: &SynthDef, notes: &[Note], extra: &[(&str, f32)], secs: f32) -> Out {
    let mut engine = Engine::new(EngineConfig {
        sample_rate: SR,
        block_size: BLOCK,
    });
    let bus = engine
        .graph_mut()
        .add_node(Box::new(ugens::Bus::new(ugens::ChannelCount::Stereo)));
    engine.graph_mut().set_sink(bus);
    for n in notes {
        let v = engine.spawn_voice_on_bus(def, bus).unwrap();
        engine.set_voice_param(v, "freq", midi_hz(n.key));
        engine.set_voice_param(v, "vel", n.vel as f32 / 127.0);
        for &(k, x) in extra {
            engine.set_voice_param(v, k, x);
        }
        let on = (n.at * SR) as u64;
        engine.schedule_note(v, on, ((n.dur * SR) as u64).max(1));
    }
    engine.prepare();
    let blocks = (secs * SR) as usize / BLOCK + 1;
    let mut out = Out {
        l: Vec::new(),
        r: Vec::new(),
        alive: Vec::new(),
    };
    for _ in 0..blocks {
        let buf = engine.render().expect("bus sink");
        out.l.extend_from_slice(buf.channel(0).samples());
        out.r.extend_from_slice(buf.channel(1).samples());
        if engine.free_done_synths() > 0 {
            engine.prepare();
        }
        out.alive.push(engine.synths().len());
    }
    out
}

fn render(b: &Banks, notes: &[Note], secs: f32) -> Out {
    render_with(&compile(b, &["freq", "gate", "vel"]), notes, &[], secs)
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

// ---------------------------------------------------------------------------

#[test]
fn region_is_picked_by_key() {
    let (a, b) = (dc(1, 0.25, 44_100), dc(2, 0.5, 44_100));
    let prog = program(
        &[&a, &b],
        vec![
            Region {
                hikey: 59,
                ..reg(0)
            },
            Region {
                lokey: 60,
                ..reg(1)
            },
        ],
    );
    let bk = banks(vec![a, b], prog);
    let lo = render(&bk, &[note(0.0, 0.3, 59.0, 100)], 0.2);
    let hi = render(&bk, &[note(0.0, 0.3, 60.0, 100)], 0.2);
    assert!(
        close(lo.mean(0.05, 0.1), 0.25 * MONO, 1e-5),
        "{}",
        lo.mean(0.05, 0.1)
    );
    assert!(
        close(hi.mean(0.05, 0.1), 0.5 * MONO, 1e-5),
        "{}",
        hi.mean(0.05, 0.1)
    );
}

#[test]
fn velocity_layers_and_velocity_tracking() {
    let (a, b) = (dc(1, 0.25, 44_100), dc(2, 0.5, 44_100));
    let prog = program(
        &[&a, &b],
        vec![
            Region {
                hivel: 63,
                amp_veltrack: 100.0,
                ..reg(0)
            },
            Region {
                lovel: 64,
                amp_veltrack: 100.0,
                ..reg(1)
            },
        ],
    );
    let bk = banks(vec![a, b], prog);
    let soft = render(&bk, &[note(0.0, 0.3, 60.0, 40)], 0.2);
    let loud = render(&bk, &[note(0.0, 0.3, 60.0, 127)], 0.2);
    let v = 40.0f32 / 127.0;
    assert!(close(soft.mean(0.05, 0.1), 0.25 * MONO * v * v, 1e-5));
    assert!(close(loud.mean(0.05, 0.1), 0.5 * MONO, 1e-5));
}

#[test]
fn amp_velcurve_points_set_the_gain() {
    let a = dc(1, 0.5, 44_100);
    let prog = program(
        &[&a],
        vec![Region {
            amp_veltrack: 100.0,
            amp_velcurve: vec![(64, 0.8)],
            ..reg(0)
        }],
    );
    let bk = banks(vec![a], prog);
    let o = render(&bk, &[note(0.0, 0.3, 60.0, 64)], 0.2);
    assert!(close(o.mean(0.05, 0.1), 0.5 * MONO * 0.8, 1e-5));
    let o = render(&bk, &[note(0.0, 0.3, 60.0, 32)], 0.2);
    assert!(close(o.mean(0.05, 0.1), 0.5 * MONO * 0.4, 1e-5));
}

#[test]
fn round_robin_advances_across_voices_and_is_deterministic() {
    let s: Vec<SamplePcm> = (1..=3).map(|n| dc(n, 0.1 * n as f32, 8_000)).collect();
    let refs: Vec<&SamplePcm> = s.iter().collect();
    let regions = (0..3)
        .map(|i| Region {
            seq: Some((3, i + 1)),
            ..reg(i)
        })
        .collect();
    let prog = program(&refs, regions);
    let bk = banks(s.clone(), prog);
    let notes: Vec<Note> = (0..5)
        .map(|i| note(0.2 * i as f32, 0.1, 60.0, 100))
        .collect();
    let o = render(&bk, &notes, 1.0);
    let got: Vec<f32> = (0..5)
        .map(|i| o.mean(0.2 * i as f32 + 0.02, 0.2 * i as f32 + 0.08))
        .collect();
    let want = [0.1, 0.2, 0.3, 0.1, 0.2];
    for (g, w) in got.iter().zip(want) {
        assert!(close(*g, w * MONO, 1e-5), "{got:?}");
    }
    // A second render from a fresh registration repeats the sequence exactly.
    let regions2 = (0..3)
        .map(|i| Region {
            seq: Some((3, i + 1)),
            ..reg(i)
        })
        .collect();
    let bk2 = banks(s.clone(), program(&refs, regions2));
    let o2 = render(&bk2, &notes, 1.0);
    assert_eq!(o.l, o2.l);
}

#[test]
fn rand_selection_follows_the_seed() {
    let s = vec![dc(1, 0.1, 8_000), dc(2, 0.2, 8_000)];
    let refs: Vec<&SamplePcm> = s.iter().collect();
    let prog = || {
        program(
            &refs,
            vec![
                Region {
                    rand: Some((0.0, 0.5)),
                    ..reg(0)
                },
                Region {
                    rand: Some((0.5, 1.0)),
                    ..reg(1)
                },
            ],
        )
    };
    let notes: Vec<Note> = (0..16)
        .map(|i| note(0.1 * i as f32, 0.05, 60.0, 100))
        .collect();
    let picks = |seed: f32| -> Vec<u8> {
        let bk = banks(s.clone(), prog());
        let def = compile(&bk, &["freq", "gate", "vel", "seed"]);
        let o = render_with(&def, &notes, &[("seed", seed)], 1.7);
        (0..16)
            .map(|i| {
                let m = o.mean(0.1 * i as f32 + 0.01, 0.1 * i as f32 + 0.04) / MONO;
                if close(m, 0.1, 1e-4) {
                    1
                } else if close(m, 0.2, 1e-4) {
                    2
                } else {
                    panic!("note {i} played {m}")
                }
            })
            .collect()
    };
    let a = picks(0.0);
    assert_eq!(a, picks(0.0), "same seed, same picks");
    assert!(a.contains(&1) && a.contains(&2), "{a:?}");
    assert_ne!(a, picks(1.0), "another seed draws differently");
}

#[test]
fn first_and_legato_triggers_follow_held_notes() {
    let s = vec![dc(1, 0.1, 44_100), dc(2, 0.2, 44_100)];
    let refs: Vec<&SamplePcm> = s.iter().collect();
    let prog = program(
        &refs,
        vec![
            Region {
                trigger: Trigger::First,
                ..reg(0)
            },
            Region {
                trigger: Trigger::Legato,
                ..reg(1)
            },
        ],
    );
    let bk = banks(s, prog);
    // note 1 alone; note 2 while note 1 is held; note 3 after both ended
    let notes = [
        note(0.0, 0.4, 60.0, 100),
        note(0.2, 0.4, 62.0, 100),
        note(0.8, 0.1, 64.0, 100),
    ];
    let o = render(&bk, &notes, 1.0);
    assert!(close(o.mean(0.05, 0.15), 0.1 * MONO, 1e-5));
    assert!(close(o.mean(0.25, 0.35), (0.1 + 0.2) * MONO, 1e-5));
    assert!(close(o.mean(0.82, 0.88), 0.1 * MONO, 1e-5));
}

#[test]
fn offset_and_end_bound_the_played_frames() {
    let ramp: Vec<f32> = (0..20_000).map(|i| i as f32 / 20_000.0).collect();
    let a = SamplePcm::new("ramp", hash(1), SR, &[&ramp[..]]).unwrap();
    let prog = program(
        &[&a],
        vec![Region {
            offset_frames: 10_000,
            end_frames: Some(14_000),
            ..reg(0)
        }],
    );
    let bk = banks(vec![a], prog);
    let o = render(&bk, &[note(0.0, 1.0, 60.0, 100)], 0.6);
    assert!(
        close(o.l[0], 0.5 * MONO, 1e-6),
        "starts at the offset: {}",
        o.l[0]
    );
    assert!(close(o.l[3_000], 0.65 * MONO, 1e-6));
    // ends after 4000 frames (with sfizz's fast off) and the voice is freed
    assert!(o.peak_after(4_000.0 / SR + 0.01) < 1e-4);
    assert_eq!(o.alive_at(0.5), 0);
}

#[test]
fn pitch_follows_keycenter_keytrack_tune_and_the_file_rate() {
    // a 441 Hz sine recorded at 48 kHz, keycenter 60
    let a = sine(1, 441.0, 48_000.0, 96_000);
    let measure = |region: Region, key: f32| -> f32 {
        let bk = banks(vec![a.clone()], program(&[&a], vec![region]));
        let o = render(&bk, &[note(0.0, 1.5, key, 100)], 1.2);
        zero_cross_freq(&o.l[(0.1 * SR) as usize..(1.1 * SR) as usize])
    };
    let base = measure(reg(0), 60.0);
    assert!((base / 441.0 - 1.0).abs() < 2e-4, "rate conversion: {base}");
    let up = measure(reg(0), 72.0);
    assert!((up / 882.0 - 1.0).abs() < 2e-4, "+12: {up}");
    let down = measure(reg(0), 53.0);
    assert!(
        (down / (441.0 * (-7.0f32 / 12.0).exp2()) - 1.0).abs() < 2e-4,
        "-7: {down}"
    );
    let tuned = measure(
        Region {
            tune_cents: 50.0,
            transpose: -1,
            ..reg(0)
        },
        60.0,
    );
    assert!(
        (tuned / (441.0 * (-0.5f32 / 12.0).exp2()) - 1.0).abs() < 2e-4,
        "{tuned}"
    );
    let fixed = measure(
        Region {
            pitch_keytrack: 0.0,
            ..reg(0)
        },
        72.0,
    );
    assert!((fixed / 441.0 - 1.0).abs() < 2e-4, "keytrack 0: {fixed}");
    // a fractional note is a tuning offset (+30 cents)
    let frac = measure(reg(0), 60.3);
    assert!(
        (frac / (441.0 * (0.3f32 / 12.0).exp2()) - 1.0).abs() < 2e-4,
        "{frac}"
    );
}

fn zero_cross_freq(x: &[f32]) -> f32 {
    // positive-going crossings, interpolated
    let mut first = None;
    let mut last = 0.0;
    let mut count = 0;
    for i in 1..x.len() {
        if x[i - 1] < 0.0 && x[i] >= 0.0 {
            let t = (i - 1) as f32 + x[i - 1] / (x[i - 1] - x[i]);
            if first.is_none() {
                first = Some(t);
            } else {
                count += 1;
            }
            last = t;
        }
    }
    count as f32 / ((last - first.unwrap()) / SR)
}

#[test]
fn bend_moves_the_pitch_within_the_region_range() {
    let a = sine(1, 441.0, SR, 88_200);
    let bk = banks(
        vec![a.clone()],
        program(
            &[&a],
            vec![Region {
                bend_up_cents: 1200.0,
                ..reg(0)
            }],
        ),
    );
    let def = compile(&bk, &["freq", "gate", "vel", "bend"]);
    let o = render_with(&def, &[note(0.0, 1.0, 60.0, 100)], &[("bend", 0.5)], 0.8);
    let f = zero_cross_freq(&o.l[(0.1 * SR) as usize..(0.7 * SR) as usize]);
    assert!((f / (441.0 * 0.5f32.exp2()) - 1.0).abs() < 2e-4, "{f}");
}

#[test]
fn loop_crossfade_keeps_the_wrap_continuous() {
    // 441 Hz sine: a loop of 9.25 periods jumps from the crest to zero
    let a = sine(1, 441.0, SR, 20_000);
    let max_jump = |lp: (u32, u32)| -> (f32, f32) {
        let bk = banks(
            vec![a.clone()],
            program(
                &[&a],
                vec![Region {
                    loop_mode: LoopMode::Continuous,
                    loop_source: LoopSource::File,
                    loop_points: Some(lp),
                    ..reg(0)
                }],
            ),
        );
        let o = render(&bk, &[note(0.0, 2.0, 60.0, 100)], 1.5);
        let s = &o.l[(0.5 * SR) as usize..(1.4 * SR) as usize];
        let jump = s.windows(2).fold(0.0f32, |m, w| m.max((w[1] - w[0]).abs()));
        (jump, o.rms(1.0, 1.4))
    };
    // sine slope bound: 0.5 · 2π · 441 / 44100 = 0.031 per sample
    let (jump, rms) = max_jump((1_000, 1_924));
    assert!(rms > 0.25, "still looping long past the sample end: {rms}");
    assert!(jump < 0.05, "crossfaded wrap: {jump}");
    // with no audio before the loop start the crossfade has no length
    let (jump0, _) = max_jump((0, 924));
    assert!(jump0 > 0.4, "control: an uncrossfaded wrap jumps: {jump0}");
}

#[test]
fn sustain_loops_stop_at_release_and_continuous_loops_do_not() {
    let a = sine(1, 441.0, SR, 20_000);
    let run = |mode: LoopMode| -> Out {
        let bk = banks(
            vec![a.clone()],
            program(
                &[&a],
                vec![Region {
                    loop_mode: mode,
                    loop_points: Some((1_000, 1_999)),
                    ampeg: Envelope {
                        release: 3.0,
                        ..Envelope::default()
                    },
                    ..reg(0)
                }],
            ),
        );
        render(&bk, &[note(0.0, 1.0, 60.0, 100)], 2.0)
    };
    let sus = run(LoopMode::Sustain);
    let cont = run(LoopMode::Continuous);
    // held: both loop well past the 0.45 s sample
    assert!(sus.rms(0.8, 0.95) > 0.2 && cont.rms(0.8, 0.95) > 0.2);
    // released at 1.0 s: the sustain loop plays out to the end (18000
    // frames, < 0.45 s), the continuous one keeps looping in the release
    assert!(sus.rms(1.5, 1.9) < 1e-4, "{}", sus.rms(1.5, 1.9));
    assert!(cont.rms(1.5, 1.9) > 0.02, "{}", cont.rms(1.5, 1.9));
}

#[test]
fn one_shot_ignores_note_off() {
    let a = dc(1, 0.5, 44_100);
    let run = |mode: LoopMode| {
        let bk = banks(
            vec![a.clone()],
            program(
                &[&a],
                vec![Region {
                    loop_mode: mode,
                    ..reg(0)
                }],
            ),
        );
        render(&bk, &[note(0.0, 0.1, 60.0, 100)], 0.5)
    };
    assert!(close(
        run(LoopMode::OneShot).mean(0.3, 0.4),
        0.5 * MONO,
        1e-5
    ));
    // 6 ms release floor, then sfizz's 50 ms fade
    assert!(run(LoopMode::NoLoop).peak_after(0.17) < 1e-6);
}

#[test]
fn amp_envelope_follows_sfizz_shapes() {
    let a = dc(1, 1.0, 3 * 44_100);
    let prog = program(
        &[&a],
        vec![Region {
            ampeg: Envelope {
                attack: 0.1,
                decay: 0.2,
                sustain: 50.0,
                release: 0.3,
                ..Envelope::default()
            },
            ..reg(0)
        }],
    );
    let bk = banks(vec![a], prog);
    // note-off on a block boundary
    let off = (1.0 * SR / BLOCK as f32).round() * BLOCK as f32 / SR;
    let o = render(&bk, &[note(0.0, off, 60.0, 100)], 1.6);
    // linear attack: a quarter at 25 ms, three quarters at 75 ms
    assert!(close(o.at(0.025), 0.25 * MONO, 2e-3), "{}", o.at(0.025));
    assert!(close(o.at(0.075), 0.75 * MONO, 2e-3), "{}", o.at(0.075));
    // exponential decay (-78 dB over the decay time) from 1 towards 50 %
    let d = o.at(0.1 + 0.005);
    assert!(close(d, (-9.0f32 * 0.005 / 0.2).exp() * MONO, 2e-3), "{d}");
    assert!(close(o.at(0.9), 0.5 * MONO, 1e-4));
    // exponential release: e^-4.5 of the sustain level half way through
    let r = o.at(off + 0.15);
    assert!(close(r, 0.5 * (-4.5f32).exp() * MONO, 2e-4), "{r}");
    assert!(o.peak_after(off + 0.36) < 1e-6);
}

#[test]
fn region_delay_holds_the_start() {
    let a = dc(1, 0.5, 44_100);
    let bk = banks(
        vec![a.clone()],
        program(
            &[&a],
            vec![Region {
                delay_s: 0.1,
                ..reg(0)
            }],
        ),
    );
    let o = render(&bk, &[note(0.0, 0.5, 60.0, 100)], 0.3);
    assert_eq!(o.mean(0.0, 0.099), 0.0);
    assert!(close(o.mean(0.11, 0.2), 0.5 * MONO, 1e-5));
}

#[test]
fn release_trigger_fires_at_note_off_with_rt_decay() {
    let (body, tail) = (dc(1, 0.5, 4 * 44_100), dc(2, 0.25, 22_050));
    let prog = || {
        program(
            &[&body, &tail],
            vec![
                reg(0),
                Region {
                    trigger: Trigger::Release,
                    rt_decay_db_per_s: 6.0,
                    ..reg(1)
                },
            ],
        )
    };
    let level_after = |held: f32| -> f32 {
        let bk = banks(vec![body.clone(), tail.clone()], prog());
        let held = (held * SR / BLOCK as f32).round() * BLOCK as f32 / SR;
        let o = render(&bk, &[note(0.0, held, 60.0, 100)], held + 0.6);
        assert!(
            close(o.mean(0.1, 0.2), 0.5 * MONO, 1e-5),
            "only the body while held"
        );
        o.mean(held + 0.05, held + 0.4)
    };
    let short = level_after(0.5);
    let long = level_after(1.5);
    assert!(
        close(short, 0.25 * MONO * 10f32.powf(-3.0 / 20.0), 2e-4),
        "{short}"
    );
    let ratio_db = 20.0 * (short / long).log10();
    assert!(
        (ratio_db - 6.0).abs() < 0.05,
        "6 dB per second held: {ratio_db}"
    );
}

#[test]
fn release_regions_need_a_sounding_attack_region_but_release_key_does_not() {
    // a 10 ms attack sample ends long before the note-off
    let (body, tail) = (dc(1, 0.5, 441), dc(2, 0.25, 22_050));
    let run = |trigger: Trigger| {
        let bk = banks(
            vec![body.clone(), tail.clone()],
            program(&[&body, &tail], vec![reg(0), Region { trigger, ..reg(1) }]),
        );
        render(&bk, &[note(0.0, 0.3, 60.0, 100)], 0.6).mean(0.35, 0.45)
    };
    assert_eq!(run(Trigger::Release), 0.0);
    assert!(close(run(Trigger::ReleaseKey), 0.25 * MONO, 1e-5));
}

#[test]
fn a_region_in_a_group_chokes_another_voices_off_by_region() {
    // open hat (group 1, choked by group 2) and a short closed hat (group 2)
    let (open, closed) = (dc(1, 0.5, 4 * 44_100), dc(2, 0.25, 441));
    let prog = |off_by: Option<i32>| {
        program(
            &[&open, &closed],
            vec![
                Region {
                    lokey: 46,
                    hikey: 46,
                    group: 1,
                    off_by,
                    off_mode: OffMode::Fast,
                    loop_mode: LoopMode::OneShot,
                    ..reg(0)
                },
                Region {
                    lokey: 42,
                    hikey: 42,
                    group: 2,
                    loop_mode: LoopMode::OneShot,
                    ..reg(1)
                },
            ],
        )
    };
    let notes = [note(0.0, 0.05, 46.0, 100), note(0.2, 0.05, 42.0, 100)];
    let bk = banks(vec![open.clone(), closed.clone()], prog(Some(2)));
    let o = render(&bk, &notes, 0.5);
    assert!(close(o.mean(0.1, 0.19), 0.5 * MONO, 1e-5));
    assert!(o.peak_after(0.25) < 1e-4, "choked: {}", o.peak_after(0.25));
    assert_eq!(o.alive_at(0.4), 0, "the choked voice is freed");
    // control: without off_by the open hat rings on
    let bk = banks(vec![open.clone(), closed.clone()], prog(None));
    let o = render(&bk, &notes, 0.5);
    assert!(close(o.mean(0.3, 0.4), 0.5 * MONO, 1e-5));
}

#[test]
fn a_self_choking_group_spares_a_repeat_of_the_same_note() {
    let a = dc(1, 0.25, 4 * 44_100);
    let bk = || {
        banks(
            vec![a.clone()],
            program(
                &[&a],
                vec![Region {
                    group: 3,
                    off_by: Some(3),
                    loop_mode: LoopMode::OneShot,
                    ..reg(0)
                }],
            ),
        )
    };
    let same = render(
        &bk(),
        &[note(0.0, 0.05, 60.0, 100), note(0.2, 0.05, 60.0, 100)],
        0.5,
    );
    assert!(close(same.mean(0.3, 0.4), 0.5 * MONO, 1e-5), "both sound");
    let other = render(
        &bk(),
        &[note(0.0, 0.05, 60.0, 100), note(0.2, 0.05, 62.0, 100)],
        0.5,
    );
    assert!(
        close(other.mean(0.3, 0.4), 0.25 * MONO, 1e-4),
        "the first is choked"
    );
}

#[test]
fn note_polyphony_and_region_polyphony_steal() {
    let a = dc(1, 0.25, 4 * 44_100);
    let run = |r: Region, keys: [f32; 2]| {
        let bk = banks(vec![a.clone()], program(&[&a], vec![r]));
        render(
            &bk,
            &[note(0.0, 1.0, keys[0], 100), note(0.2, 1.0, keys[1], 100)],
            0.5,
        )
        .mean(0.3, 0.4)
    };
    let fast = |r: Region| Region {
        off_mode: OffMode::Fast,
        ..r
    };
    assert!(
        close(run(reg(0), [60.0, 60.0]), 0.5 * MONO, 1e-5),
        "control"
    );
    let np = fast(Region {
        note_polyphony: Some(1),
        ..reg(0)
    });
    assert!(close(run(np.clone(), [60.0, 60.0]), 0.25 * MONO, 1e-4));
    assert!(
        close(run(np, [60.0, 62.0]), 0.5 * MONO, 1e-5),
        "other notes untouched"
    );
    let poly = fast(Region {
        polyphony: Some(1),
        ..reg(0)
    });
    assert!(close(run(poly, [60.0, 62.0]), 0.25 * MONO, 1e-4));
}

#[test]
fn velocity_crossfades_scale_the_layers() {
    let a = dc(1, 0.5, 44_100);
    let bk = banks(
        vec![a.clone()],
        program(
            &[&a],
            vec![Region {
                xf_vel: Some(VelCrossfade {
                    in_lo: 0.0,
                    in_hi: 65.0,
                    out_lo: 127.0,
                    out_hi: 127.0,
                    gain_curve: false,
                }),
                ..reg(0)
            }],
        ),
    );
    let lvl = |v: u8| render(&bk, &[note(0.0, 0.3, 60.0, v)], 0.2).mean(0.05, 0.1);
    assert!(
        close(lvl(32), 0.5 * MONO * 0.5f32.sqrt(), 1e-5),
        "power curve half way"
    );
    assert!(close(lvl(100), 0.5 * MONO, 1e-5));
}

#[test]
fn filters_and_eq_shape_the_sample() {
    let a = sine(1, 5_000.0, SR, 44_100);
    let level = |filters: Vec<Filter>, eq: Vec<EqBand>, key: f32| {
        let bk = banks(
            vec![a.clone()],
            program(
                &[&a],
                vec![Region {
                    pitch_keytrack: 0.0,
                    filters,
                    eq,
                    ..reg(0)
                }],
            ),
        );
        render(&bk, &[note(0.0, 0.6, key, 100)], 0.5).rms(0.1, 0.4)
    };
    let dry = level(vec![], vec![], 60.0);
    let lp = |keytrack: f32| Filter {
        kind: FilterKind::Lpf2p,
        cutoff: 1_000.0,
        resonance: 0.0,
        keytrack,
        keycenter: 60,
        veltrack: 0.0,
        gain: 0.0,
    };
    let db = |x: f32| 20.0 * (x / dry).log10();
    // RBJ 2-pole, Q 1, five times the cutoff: about -27.6 dB
    let att = db(level(vec![lp(0.0)], vec![], 60.0));
    assert!((-29.0..-26.0).contains(&att), "{att}");
    // keytracked three octaves up (1200 cents/key x 3 keys): 8 kHz, passes
    let tracked = db(level(vec![lp(1200.0)], vec![], 63.0));
    assert!(tracked > -1.5, "{tracked}");
    let eq = db(level(
        vec![],
        vec![EqBand {
            kind: EqKind::Peak,
            freq: 5_000.0,
            bw: 1.0,
            gain: -12.0,
            vel2freq: 0.0,
            vel2gain: 0.0,
        }],
        60.0,
    ));
    assert!((eq + 12.0).abs() < 0.2, "{eq}");
}

#[test]
fn pan_and_width_place_the_sample() {
    let a = dc(1, 0.5, 44_100);
    let bk = banks(
        vec![a.clone()],
        program(
            &[&a],
            vec![Region {
                pan: -100.0,
                ..reg(0)
            }],
        ),
    );
    let o = render(&bk, &[note(0.0, 0.3, 60.0, 100)], 0.2);
    assert!(close(o.mean(0.05, 0.1), 0.5 * 1.412_537_5, 1e-5));
    assert!(o.r[(0.05 * SR) as usize].abs() < 1e-6);
    // a stereo sample at width 0 sums to the middle
    let l = vec![0.5f32; 44_100];
    let r = vec![0.1f32; 44_100];
    let st = SamplePcm::new("st", hash(9), SR, &[&l[..], &r[..]]).unwrap();
    let bk = banks(
        vec![st.clone()],
        program(
            &[&st],
            vec![Region {
                width: 0.0,
                ..reg(0)
            }],
        ),
    );
    let o = render(&bk, &[note(0.0, 0.3, 60.0, 100)], 0.2);
    let i = (0.05 * SR) as usize;
    assert!(close(o.l[i], o.r[i], 1e-6));
    assert!(
        close(o.l[i], 0.6 * 0.997_63 * (0.5f32).sqrt(), 1e-4),
        "{}",
        o.l[i]
    );
}

#[test]
fn stereo_samples_keep_their_channels_apart() {
    let n = 44_100;
    let l: Vec<f32> = (0..n)
        .map(|i| (core::f32::consts::TAU * 441.0 * i as f32 / SR).sin())
        .collect();
    let r: Vec<f32> = (0..n)
        .map(|i| (core::f32::consts::TAU * 661.5 * i as f32 / SR).sin())
        .collect();
    let st = SamplePcm::new("st", hash(9), SR, &[&l[..], &r[..]]).unwrap();
    let bk = banks(vec![st.clone()], program(&[&st], vec![reg(0)]));
    let o = render(&bk, &[note(0.0, 0.6, 60.0, 100)], 0.5);
    let i0 = (0.1 * SR) as usize;
    for i in i0..i0 + 1000 {
        assert!(close(o.l[i], 0.997_63 * l[i], 1e-5));
        assert!(close(o.r[i], 0.997_63 * r[i], 1e-5));
    }
}

#[test]
fn amp_random_is_drawn_from_the_seed() {
    let a = dc(1, 0.5, 44_100);
    let levels = |seed: f32| -> Vec<f32> {
        let bk = banks(
            vec![a.clone()],
            program(
                &[&a],
                vec![Region {
                    amp_random_db: 6.0,
                    ..reg(0)
                }],
            ),
        );
        let def = compile(&bk, &["freq", "gate", "vel", "seed"]);
        let notes: Vec<Note> = (0..6)
            .map(|i| note(0.1 * i as f32, 0.05, 60.0, 100))
            .collect();
        let o = render_with(&def, &notes, &[("seed", seed)], 0.7);
        (0..6)
            .map(|i| o.mean(0.1 * i as f32 + 0.01, 0.1 * i as f32 + 0.04))
            .collect()
    };
    let a0 = levels(0.0);
    assert_eq!(a0, levels(0.0));
    for &x in &a0 {
        let db = 20.0 * (x / (0.5 * MONO)).log10();
        assert!((-1e-3..6.001).contains(&db), "{db}");
    }
    assert!(a0.windows(2).any(|w| w[0] != w[1]));
}

#[test]
fn silence_regions_play_nothing_but_still_choke() {
    let a = dc(1, 0.5, 4 * 44_100);
    let prog = program(
        &[&a],
        vec![
            Region {
                lokey: 60,
                hikey: 60,
                off_by: Some(5),
                loop_mode: LoopMode::OneShot,
                ..reg(0)
            },
            Region {
                sample: SILENCE,
                lokey: 61,
                hikey: 61,
                group: 5,
                ..reg(0)
            },
        ],
    );
    let bk = banks(vec![a], prog);
    let o = render(
        &bk,
        &[note(0.0, 0.05, 60.0, 100), note(0.2, 0.1, 61.0, 100)],
        0.5,
    );
    assert!(close(o.mean(0.1, 0.19), 0.5 * MONO, 1e-5));
    assert!(o.peak_after(0.25) < 1e-4);
}

#[test]
fn the_voice_is_done_only_when_every_region_has_ended() {
    let a = dc(1, 0.5, 4 * 44_100);
    let tail = dc(2, 0.25, 4_410);
    // a held looping note stays; after release + its 0.2 s release it goes
    let looped = program(
        &[&a],
        vec![Region {
            loop_mode: LoopMode::Continuous,
            loop_points: Some((1_000, 10_000)),
            ampeg: Envelope {
                release: 0.2,
                ..Envelope::default()
            },
            ..reg(0)
        }],
    );
    let o = render(
        &banks(vec![a.clone()], looped),
        &[note(0.0, 5.0, 60.0, 100)],
        5.6,
    );
    assert_eq!(o.alive_at(4.9), 1, "held");
    assert_eq!(o.alive_at(5.5), 0, "freed after the release");
    // a voice with no attack region but a release-key region waits for its gate
    let only_release = program(
        &[&tail],
        vec![Region {
            trigger: Trigger::ReleaseKey,
            ..reg(0)
        }],
    );
    let o = render(
        &banks(vec![tail], only_release),
        &[note(0.0, 0.5, 60.0, 100)],
        0.8,
    );
    assert_eq!(o.alive_at(0.4), 1, "waits for the gate");
    assert!(o.mean(0.52, 0.58) > 0.2, "plays the release region");
    assert_eq!(o.alive_at(0.75), 0);
}

#[test]
fn quality_selects_the_kernel() {
    let a = sine(1, 2_000.0, SR, 44_100);
    let bk = banks(vec![a.clone()], program(&[&a], vec![reg(0)]));
    let def = compile(&bk, &["freq", "gate", "vel", "quality"]);
    let run = |q: f32| render_with(&def, &[note(0.0, 0.5, 60.3, 100)], &[("quality", q)], 0.3).l;
    let (lin, her, sinc) = (run(0.0), run(1.0), run(2.0));
    assert_ne!(lin, her);
    assert_ne!(her, sinc);
    // unconnected = the default kernel (sinc)
    let def0 = compile(&bk, &["freq", "gate", "vel"]);
    assert_eq!(
        render_with(&def0, &[note(0.0, 0.5, 60.3, 100)], &[], 0.3).l,
        sinc
    );
}

#[test]
fn ir_binding_errors_are_named() {
    let a = dc(1, 0.5, 100);
    let bk = banks(vec![a.clone()], program(&[&a], vec![reg(0)]));
    let reg = registry();
    let ir = voice_ir(ProgramId(99), &["freq"]);
    assert_eq!(
        ir.compile_with_banks(&reg, &CoeffTableBank::new(), Some(&bk.programs))
            .err(),
        Some(IrError::ProgramNotFound(99))
    );
    let ir = voice_ir(bk.pid, &["freq"]);
    assert_eq!(
        ir.compile_with_tables(&reg, &CoeffTableBank::new()).err(),
        Some(IrError::ProgramNotFound(bk.pid.0)),
        "the table-only compile has no programs to bind"
    );
    let mut unbound = voice_ir(bk.pid, &["freq"]);
    unbound.table_bindings.clear();
    assert!(matches!(
        unbound.validate(&reg),
        Err(IrError::MissingTableBinding { .. })
    ));
}
