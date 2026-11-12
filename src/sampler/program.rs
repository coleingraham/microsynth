//! A compiled instrument program: the regions of one articulation of one
//! instrument at one preset, as a host compiles them from the canonical
//! instrument spec, in the byte format of `docs/sample-program-format.md`.
//!
//! Every value is a constant (modulation was evaluated when the spec was
//! compiled). Units follow the spec: times in seconds, tuning in cents,
//! volume in dB, pan/width/veltrack/sustain in percent, frames at the
//! sample's own rate, key and velocity bounds inclusive MIDI numbers.

use super::pcm::ContentHash;
use crate::wire::{Reader, WireError, put_f32, put_str, put_u16, put_u32};
use alloc::string::String;
use alloc::vec::Vec;

/// Wire-format magic prefix of a program.
pub const PROGRAM_MAGIC: &[u8] = b"MSPG";
/// Wire-format version of a program. Bump on any layout change.
pub const PROGRAM_FORMAT_VERSION: u16 = 1;
/// `sample` index of a region whose source is the silence generator (it
/// plays nothing but can still choke other regions).
pub const SILENCE: u32 = u32::MAX;
/// Encoded "absent" for optional frame indices.
const NONE_U32: u32 = u32::MAX;

/// How a region's sample loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Plays to the end (or until released); no loop.
    NoLoop,
    /// Plays to the end; note-off does not release it.
    OneShot,
    /// Loops for the whole note, release included.
    Continuous,
    /// Loops until released, then plays on to the end.
    Sustain,
}

/// Where a region's loop points came from (recorded, not acted on).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopSource {
    None,
    Sfz,
    File,
    Whole,
}

/// What starts a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// Note-on.
    Attack,
    /// Note-off, while an attack region of a compatible note still sounds.
    Release,
    /// Note-off, always.
    ReleaseKey,
    /// Note-on with no other note held.
    First,
    /// Note-on while another note is held.
    Legato,
}

/// How a choked (or stolen) region stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffMode {
    /// A 6 ms release.
    Fast,
    /// The region's own amp-envelope release.
    Normal,
    /// The region's `off_time` as the release.
    Time,
}

/// A static filter type the voice runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterKind {
    Lpf1p,
    Hpf1p,
    Lpf2p,
    Hpf2p,
    Bpf2p,
    Brf2p,
    Lpf4p,
    Hpf4p,
    Lpf6p,
    Hpf6p,
    Peq,
    Lsh,
    Hsh,
}

/// A static EQ band type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EqKind {
    Peak,
    LowShelf,
    HighShelf,
}

/// The amp envelope (sfizz's ADSR with delay, hold and start level).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Envelope {
    pub delay: f32,
    pub attack: f32,
    pub hold: f32,
    pub decay: f32,
    /// Percent.
    pub sustain: f32,
    pub release: f32,
    /// Percent.
    pub start: f32,
}

impl Default for Envelope {
    fn default() -> Self {
        Envelope {
            delay: 0.0,
            attack: 0.0,
            hold: 0.0,
            decay: 0.0,
            sustain: 100.0,
            release: 0.001,
            start: 0.0,
        }
    }
}

/// A velocity crossfade (`xfin_*`/`xfout_*` velocities as written).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelCrossfade {
    pub in_lo: f32,
    pub in_hi: f32,
    pub out_lo: f32,
    pub out_hi: f32,
    /// `true` for the `gain` curve, `false` for `power`.
    pub gain_curve: bool,
}

/// One static filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Filter {
    pub kind: FilterKind,
    pub cutoff: f32,
    /// dB.
    pub resonance: f32,
    /// Cents per key.
    pub keytrack: f32,
    pub keycenter: u8,
    /// Cents at full velocity.
    pub veltrack: f32,
    /// dB (peaking and shelving types).
    pub gain: f32,
}

/// One static EQ band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqBand {
    pub kind: EqKind,
    pub freq: f32,
    /// Octaves.
    pub bw: f32,
    /// dB.
    pub gain: f32,
    /// Hz at full velocity.
    pub vel2freq: f32,
    /// dB at full velocity.
    pub vel2gain: f32,
}

/// One region.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// Index into [`SampleProgram::samples`], or [`SILENCE`].
    pub sample: u32,
    pub lokey: u8,
    pub hikey: u8,
    pub lovel: u8,
    pub hivel: u8,
    pub pitch_keycenter: u8,
    /// Cents per key.
    pub pitch_keytrack: f32,
    pub tune_cents: f32,
    pub transpose: i32,
    pub offset_frames: u32,
    /// Last frame played (inclusive); `None` = the sample's last frame.
    pub end_frames: Option<u32>,
    pub loop_mode: LoopMode,
    pub loop_source: LoopSource,
    /// Inclusive loop frames; `None` when there is no loop.
    pub loop_points: Option<(u32, u32)>,
    pub volume_db: f32,
    pub amp_random_db: f32,
    pub pan: f32,
    pub width: f32,
    pub amp_veltrack: f32,
    /// `amp_velcurve_N` points (velocity, gain), in the order written.
    pub amp_velcurve: Vec<(u8, f32)>,
    pub ampeg: Envelope,
    /// (length, position), position 1-based.
    pub seq: Option<(u32, u32)>,
    /// `lorand`/`hirand`.
    pub rand: Option<(f32, f32)>,
    pub group: i32,
    pub off_by: Option<i32>,
    pub off_mode: OffMode,
    pub off_time: f32,
    pub trigger: Trigger,
    pub rt_decay_db_per_s: f32,
    pub delay_s: f32,
    pub note_polyphony: Option<u32>,
    pub polyphony: Option<u32>,
    pub bend_up_cents: f32,
    pub bend_down_cents: f32,
    pub xf_vel: Option<VelCrossfade>,
    pub filters: Vec<Filter>,
    pub eq: Vec<EqBand>,
}

impl Default for Region {
    fn default() -> Self {
        Region {
            sample: 0,
            lokey: 0,
            hikey: 127,
            lovel: 0,
            hivel: 127,
            pitch_keycenter: 60,
            pitch_keytrack: 100.0,
            tune_cents: 0.0,
            transpose: 0,
            offset_frames: 0,
            end_frames: None,
            loop_mode: LoopMode::NoLoop,
            loop_source: LoopSource::None,
            loop_points: None,
            volume_db: 0.0,
            amp_random_db: 0.0,
            pan: 0.0,
            width: 100.0,
            amp_veltrack: 100.0,
            amp_velcurve: Vec::new(),
            ampeg: Envelope::default(),
            seq: None,
            rand: None,
            group: 0,
            off_by: None,
            off_mode: OffMode::Fast,
            off_time: 0.006,
            trigger: Trigger::Attack,
            rt_decay_db_per_s: 0.0,
            delay_s: 0.0,
            note_polyphony: None,
            polyphony: None,
            bend_up_cents: 200.0,
            bend_down_cents: -200.0,
            xf_vel: None,
            filters: Vec::new(),
            eq: Vec::new(),
        }
    }
}

/// A compiled program.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SampleProgram {
    /// The spec's program id.
    pub name: String,
    /// sha256 of the canonical spec the program was compiled from, carried
    /// so a render can name exactly what it played.
    pub spec_sha256: ContentHash,
    /// The samples the regions read, by content hash.
    pub samples: Vec<ContentHash>,
    pub regions: Vec<Region>,
}

/// Errors from decoding a program.
#[derive(Debug, Clone, PartialEq)]
pub enum ProgramCodecError {
    BadMagic,
    UnsupportedVersion(u16),
    UnexpectedEof,
    BadUtf8,
    TrailingBytes,
    /// An enum tag outside its range, at region `region`.
    BadTag {
        region: usize,
        field: &'static str,
        tag: u8,
    },
    /// A region that cannot play as written (empty key/velocity window,
    /// sample index outside the table, inverted loop, non-finite number).
    BadRegion {
        region: usize,
        why: &'static str,
    },
}

impl From<WireError> for ProgramCodecError {
    fn from(e: WireError) -> Self {
        match e {
            WireError::UnexpectedEof => ProgramCodecError::UnexpectedEof,
            WireError::BadUtf8 => ProgramCodecError::BadUtf8,
        }
    }
}

fn put_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}
fn put_i32(out: &mut Vec<u8>, v: i32) {
    put_u32(out, v as u32);
}
fn put_opt_u32(out: &mut Vec<u8>, v: Option<u32>) {
    put_u32(out, v.unwrap_or(NONE_U32));
}

fn loop_mode_tag(m: LoopMode) -> u8 {
    match m {
        LoopMode::NoLoop => 0,
        LoopMode::OneShot => 1,
        LoopMode::Continuous => 2,
        LoopMode::Sustain => 3,
    }
}
fn loop_source_tag(s: LoopSource) -> u8 {
    match s {
        LoopSource::None => 0,
        LoopSource::Sfz => 1,
        LoopSource::File => 2,
        LoopSource::Whole => 3,
    }
}
fn trigger_tag(t: Trigger) -> u8 {
    match t {
        Trigger::Attack => 0,
        Trigger::Release => 1,
        Trigger::ReleaseKey => 2,
        Trigger::First => 3,
        Trigger::Legato => 4,
    }
}
fn off_mode_tag(m: OffMode) -> u8 {
    match m {
        OffMode::Fast => 0,
        OffMode::Normal => 1,
        OffMode::Time => 2,
    }
}
/// Filter type tags, in wire order.
pub const FILTER_KINDS: [FilterKind; 13] = [
    FilterKind::Lpf1p,
    FilterKind::Hpf1p,
    FilterKind::Lpf2p,
    FilterKind::Hpf2p,
    FilterKind::Bpf2p,
    FilterKind::Brf2p,
    FilterKind::Lpf4p,
    FilterKind::Hpf4p,
    FilterKind::Lpf6p,
    FilterKind::Hpf6p,
    FilterKind::Peq,
    FilterKind::Lsh,
    FilterKind::Hsh,
];
/// EQ type tags, in wire order.
pub const EQ_KINDS: [EqKind; 3] = [EqKind::Peak, EqKind::LowShelf, EqKind::HighShelf];

impl SampleProgram {
    /// Encode to the upload byte format (`docs/sample-program-format.md`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(PROGRAM_MAGIC);
        put_u16(&mut out, PROGRAM_FORMAT_VERSION);
        put_str(&mut out, &self.name);
        out.extend_from_slice(&self.spec_sha256);
        put_u32(&mut out, self.samples.len() as u32);
        for h in &self.samples {
            out.extend_from_slice(h);
        }
        put_u32(&mut out, self.regions.len() as u32);
        for r in &self.regions {
            put_u32(&mut out, r.sample);
            for b in [r.lokey, r.hikey, r.lovel, r.hivel, r.pitch_keycenter] {
                put_u8(&mut out, b);
            }
            put_f32(&mut out, r.pitch_keytrack);
            put_f32(&mut out, r.tune_cents);
            put_i32(&mut out, r.transpose);
            put_u32(&mut out, r.offset_frames);
            put_opt_u32(&mut out, r.end_frames);
            put_u8(&mut out, loop_mode_tag(r.loop_mode));
            put_u8(&mut out, loop_source_tag(r.loop_source));
            put_opt_u32(&mut out, r.loop_points.map(|p| p.0));
            put_opt_u32(&mut out, r.loop_points.map(|p| p.1));
            for v in [r.volume_db, r.amp_random_db, r.pan, r.width, r.amp_veltrack] {
                put_f32(&mut out, v);
            }
            put_u8(&mut out, r.amp_velcurve.len() as u8);
            for &(v, g) in &r.amp_velcurve {
                put_u8(&mut out, v);
                put_f32(&mut out, g);
            }
            let e = &r.ampeg;
            for v in [
                e.delay, e.attack, e.hold, e.decay, e.sustain, e.release, e.start,
            ] {
                put_f32(&mut out, v);
            }
            let (sl, sp) = r.seq.unwrap_or((0, 0));
            put_u32(&mut out, sl);
            put_u32(&mut out, sp);
            put_u8(&mut out, r.rand.is_some() as u8);
            let (rl, rh) = r.rand.unwrap_or((0.0, 1.0));
            put_f32(&mut out, rl);
            put_f32(&mut out, rh);
            put_i32(&mut out, r.group);
            put_u8(&mut out, r.off_by.is_some() as u8);
            put_i32(&mut out, r.off_by.unwrap_or(0));
            put_u8(&mut out, off_mode_tag(r.off_mode));
            put_f32(&mut out, r.off_time);
            put_u8(&mut out, trigger_tag(r.trigger));
            put_f32(&mut out, r.rt_decay_db_per_s);
            put_f32(&mut out, r.delay_s);
            put_u32(&mut out, r.note_polyphony.unwrap_or(0));
            put_u32(&mut out, r.polyphony.unwrap_or(0));
            put_f32(&mut out, r.bend_up_cents);
            put_f32(&mut out, r.bend_down_cents);
            put_u8(&mut out, r.xf_vel.is_some() as u8);
            let xf = r.xf_vel.unwrap_or(VelCrossfade {
                in_lo: 0.0,
                in_hi: 0.0,
                out_lo: 127.0,
                out_hi: 127.0,
                gain_curve: false,
            });
            for v in [xf.in_lo, xf.in_hi, xf.out_lo, xf.out_hi] {
                put_f32(&mut out, v);
            }
            put_u8(&mut out, xf.gain_curve as u8);
            put_u8(&mut out, r.filters.len() as u8);
            for f in &r.filters {
                let tag = FILTER_KINDS.iter().position(|k| *k == f.kind).unwrap_or(0) as u8;
                put_u8(&mut out, tag);
                put_f32(&mut out, f.cutoff);
                put_f32(&mut out, f.resonance);
                put_f32(&mut out, f.keytrack);
                put_u8(&mut out, f.keycenter);
                put_f32(&mut out, f.veltrack);
                put_f32(&mut out, f.gain);
            }
            put_u8(&mut out, r.eq.len() as u8);
            for b in &r.eq {
                let tag = EQ_KINDS.iter().position(|k| *k == b.kind).unwrap_or(0) as u8;
                put_u8(&mut out, tag);
                for v in [b.freq, b.bw, b.gain, b.vel2freq, b.vel2gain] {
                    put_f32(&mut out, v);
                }
            }
        }
        out
    }

    /// Decode the upload byte format, validating every region.
    pub fn from_bytes(bytes: &[u8]) -> Result<SampleProgram, ProgramCodecError> {
        let mut r = Reader::new(bytes);
        if r.take(PROGRAM_MAGIC.len())? != PROGRAM_MAGIC {
            return Err(ProgramCodecError::BadMagic);
        }
        let version = r.u16()?;
        if version > PROGRAM_FORMAT_VERSION {
            return Err(ProgramCodecError::UnsupportedVersion(version));
        }
        let name = r.string()?;
        let mut spec_sha256 = [0u8; 32];
        spec_sha256.copy_from_slice(r.take(32)?);
        let n_samples = r.usize32()?;
        let mut samples = Vec::with_capacity(n_samples.min(1 << 16));
        for _ in 0..n_samples {
            let mut h = [0u8; 32];
            h.copy_from_slice(r.take(32)?);
            samples.push(h);
        }
        let n_regions = r.usize32()?;
        let mut regions = Vec::with_capacity(n_regions.min(1 << 16));
        for i in 0..n_regions {
            let region = read_region(&mut r, i)?;
            validate_region(&region, i, samples.len())?;
            regions.push(region);
        }
        if r.pos != bytes.len() {
            return Err(ProgramCodecError::TrailingBytes);
        }
        Ok(SampleProgram {
            name,
            spec_sha256,
            samples,
            regions,
        })
    }
}

fn tag<T: Copy>(
    table: &[T],
    t: u8,
    region: usize,
    field: &'static str,
) -> Result<T, ProgramCodecError> {
    table
        .get(t as usize)
        .copied()
        .ok_or(ProgramCodecError::BadTag {
            region,
            field,
            tag: t,
        })
}

fn opt_u32(v: u32) -> Option<u32> {
    if v == NONE_U32 { None } else { Some(v) }
}

fn read_region(r: &mut Reader, i: usize) -> Result<Region, ProgramCodecError> {
    let sample = r.u32()?;
    let lokey = r.u8()?;
    let hikey = r.u8()?;
    let lovel = r.u8()?;
    let hivel = r.u8()?;
    let pitch_keycenter = r.u8()?;
    let pitch_keytrack = r.f32()?;
    let tune_cents = r.f32()?;
    let transpose = r.u32()? as i32;
    let offset_frames = r.u32()?;
    let end_frames = opt_u32(r.u32()?);
    let loop_mode = tag(
        &[
            LoopMode::NoLoop,
            LoopMode::OneShot,
            LoopMode::Continuous,
            LoopMode::Sustain,
        ],
        r.u8()?,
        i,
        "loop_mode",
    )?;
    let loop_source = tag(
        &[
            LoopSource::None,
            LoopSource::Sfz,
            LoopSource::File,
            LoopSource::Whole,
        ],
        r.u8()?,
        i,
        "loop_source",
    )?;
    let ls = opt_u32(r.u32()?);
    let le = opt_u32(r.u32()?);
    let loop_points = match (ls, le) {
        (Some(a), Some(b)) => Some((a, b)),
        (None, None) => None,
        _ => {
            return Err(ProgramCodecError::BadRegion {
                region: i,
                why: "loop start and end must both be set or both absent",
            });
        }
    };
    let volume_db = r.f32()?;
    let amp_random_db = r.f32()?;
    let pan = r.f32()?;
    let width = r.f32()?;
    let amp_veltrack = r.f32()?;
    let n_curve = r.u8()? as usize;
    let mut amp_velcurve = Vec::with_capacity(n_curve);
    for _ in 0..n_curve {
        let v = r.u8()?;
        let g = r.f32()?;
        amp_velcurve.push((v, g));
    }
    let ampeg = Envelope {
        delay: r.f32()?,
        attack: r.f32()?,
        hold: r.f32()?,
        decay: r.f32()?,
        sustain: r.f32()?,
        release: r.f32()?,
        start: r.f32()?,
    };
    let seq_len = r.u32()?;
    let seq_pos = r.u32()?;
    let seq = if seq_len == 0 {
        None
    } else {
        Some((seq_len, seq_pos))
    };
    let has_rand = r.u8()? != 0;
    let rand_lo = r.f32()?;
    let rand_hi = r.f32()?;
    let rand = has_rand.then_some((rand_lo, rand_hi));
    let group = r.u32()? as i32;
    let has_off_by = r.u8()? != 0;
    let off_by_v = r.u32()? as i32;
    let off_by = has_off_by.then_some(off_by_v);
    let off_mode = tag(
        &[OffMode::Fast, OffMode::Normal, OffMode::Time],
        r.u8()?,
        i,
        "off_mode",
    )?;
    let off_time = r.f32()?;
    let trigger = tag(
        &[
            Trigger::Attack,
            Trigger::Release,
            Trigger::ReleaseKey,
            Trigger::First,
            Trigger::Legato,
        ],
        r.u8()?,
        i,
        "trigger",
    )?;
    let rt_decay_db_per_s = r.f32()?;
    let delay_s = r.f32()?;
    let np = r.u32()?;
    let poly = r.u32()?;
    let bend_up_cents = r.f32()?;
    let bend_down_cents = r.f32()?;
    let has_xf = r.u8()? != 0;
    let xf = VelCrossfade {
        in_lo: r.f32()?,
        in_hi: r.f32()?,
        out_lo: r.f32()?,
        out_hi: r.f32()?,
        gain_curve: r.u8()? != 0,
    };
    let n_filters = r.u8()? as usize;
    let mut filters = Vec::with_capacity(n_filters);
    for _ in 0..n_filters {
        let kind = tag(&FILTER_KINDS, r.u8()?, i, "filter type")?;
        filters.push(Filter {
            kind,
            cutoff: r.f32()?,
            resonance: r.f32()?,
            keytrack: r.f32()?,
            keycenter: r.u8()?,
            veltrack: r.f32()?,
            gain: r.f32()?,
        });
    }
    let n_eq = r.u8()? as usize;
    let mut eq = Vec::with_capacity(n_eq);
    for _ in 0..n_eq {
        let kind = tag(&EQ_KINDS, r.u8()?, i, "eq type")?;
        eq.push(EqBand {
            kind,
            freq: r.f32()?,
            bw: r.f32()?,
            gain: r.f32()?,
            vel2freq: r.f32()?,
            vel2gain: r.f32()?,
        });
    }
    Ok(Region {
        sample,
        lokey,
        hikey,
        lovel,
        hivel,
        pitch_keycenter,
        pitch_keytrack,
        tune_cents,
        transpose,
        offset_frames,
        end_frames,
        loop_mode,
        loop_source,
        loop_points,
        volume_db,
        amp_random_db,
        pan,
        width,
        amp_veltrack,
        amp_velcurve,
        ampeg,
        seq,
        rand,
        group,
        off_by,
        off_mode,
        off_time,
        trigger,
        rt_decay_db_per_s,
        delay_s,
        note_polyphony: (np > 0).then_some(np),
        polyphony: (poly > 0).then_some(poly),
        bend_up_cents,
        bend_down_cents,
        xf_vel: has_xf.then_some(xf),
        filters,
        eq,
    })
}

fn validate_region(r: &Region, i: usize, n_samples: usize) -> Result<(), ProgramCodecError> {
    let bad = |why| Err(ProgramCodecError::BadRegion { region: i, why });
    if r.lokey > 127 || r.hikey > 127 || r.lovel > 127 || r.hivel > 127 || r.pitch_keycenter > 127 {
        return bad("key, velocity or keycenter outside 0..127");
    }
    if r.lokey > r.hikey || r.lovel > r.hivel {
        return bad("empty key or velocity window");
    }
    if r.sample != SILENCE && r.sample as usize >= n_samples {
        return bad("sample index outside the sample table");
    }
    if let Some((a, b)) = r.loop_points
        && a > b
    {
        return bad("loop start after loop end");
    }
    if let Some((len, pos)) = r.seq
        && (pos == 0 || pos > len)
    {
        return bad("seq_position outside 1..seq_length");
    }
    if r.amp_velcurve.iter().any(|&(v, _)| v > 127) {
        return bad("amp_velcurve velocity outside 0..127");
    }
    let e = &r.ampeg;
    let mut finite = [
        r.pitch_keytrack,
        r.tune_cents,
        r.volume_db,
        r.amp_random_db,
        r.pan,
        r.width,
        r.amp_veltrack,
        e.delay,
        e.attack,
        e.hold,
        e.decay,
        e.sustain,
        e.release,
        e.start,
        r.off_time,
        r.rt_decay_db_per_s,
        r.delay_s,
        r.bend_up_cents,
        r.bend_down_cents,
    ]
    .into_iter()
    .chain(r.amp_velcurve.iter().map(|p| p.1))
    .chain(r.rand.iter().flat_map(|p| [p.0, p.1]))
    .chain(
        r.xf_vel
            .iter()
            .flat_map(|x| [x.in_lo, x.in_hi, x.out_lo, x.out_hi]),
    )
    .chain(
        r.filters
            .iter()
            .flat_map(|f| [f.cutoff, f.resonance, f.keytrack, f.veltrack, f.gain]),
    )
    .chain(
        r.eq.iter()
            .flat_map(|b| [b.freq, b.bw, b.gain, b.vel2freq, b.vel2gain]),
    );
    if !finite.all(|v| v.is_finite()) {
        return bad("non-finite number");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn rich_region() -> Region {
        Region {
            sample: 1,
            lokey: 40,
            hikey: 52,
            lovel: 10,
            hivel: 90,
            pitch_keycenter: 45,
            pitch_keytrack: 50.0,
            tune_cents: -12.5,
            transpose: -1,
            offset_frames: 100,
            end_frames: Some(9000),
            loop_mode: LoopMode::Sustain,
            loop_source: LoopSource::File,
            loop_points: Some((2000, 8000)),
            volume_db: -3.0,
            amp_random_db: 1.5,
            pan: -20.0,
            width: 50.0,
            amp_veltrack: 73.0,
            amp_velcurve: alloc::vec![(64, 0.3), (100, 0.9)],
            ampeg: Envelope {
                delay: 0.01,
                attack: 0.02,
                hold: 0.03,
                decay: 0.4,
                sustain: 60.0,
                release: 0.7,
                start: 10.0,
            },
            seq: Some((4, 3)),
            rand: Some((0.25, 0.5)),
            group: 7,
            off_by: Some(-3),
            off_mode: OffMode::Time,
            off_time: 0.2,
            trigger: Trigger::Release,
            rt_decay_db_per_s: 6.0,
            delay_s: 0.05,
            note_polyphony: Some(2),
            polyphony: Some(5),
            bend_up_cents: 1200.0,
            bend_down_cents: -1200.0,
            xf_vel: Some(VelCrossfade {
                in_lo: 20.0,
                in_hi: 60.0,
                out_lo: 70.0,
                out_hi: 100.0,
                gain_curve: true,
            }),
            filters: alloc::vec![Filter {
                kind: FilterKind::Hpf2p,
                cutoff: 80.0,
                resonance: 3.0,
                keytrack: 100.0,
                keycenter: 60,
                veltrack: 1200.0,
                gain: 0.0,
            }],
            eq: alloc::vec![EqBand {
                kind: EqKind::Peak,
                freq: 1000.0,
                bw: 1.5,
                gain: -6.0,
                vel2freq: 200.0,
                vel2gain: 3.0,
            }],
        }
    }

    fn program() -> SampleProgram {
        SampleProgram {
            name: "lib__prog".into(),
            spec_sha256: [9; 32],
            samples: alloc::vec![[1; 32], [2; 32]],
            regions: alloc::vec![Region::default(), rich_region()],
        }
    }

    #[test]
    fn round_trips_every_field() {
        let p = program();
        assert_eq!(SampleProgram::from_bytes(&p.to_bytes()), Ok(p));
    }

    #[test]
    fn rejects_regions_that_cannot_play() {
        let mut p = program();
        p.regions[1].sample = 2;
        assert!(matches!(
            SampleProgram::from_bytes(&p.to_bytes()),
            Err(ProgramCodecError::BadRegion { region: 1, .. })
        ));
        let mut p = program();
        p.regions[0].lokey = 70;
        p.regions[0].hikey = 60;
        assert!(matches!(
            SampleProgram::from_bytes(&p.to_bytes()),
            Err(ProgramCodecError::BadRegion { region: 0, .. })
        ));
        let mut p = program();
        p.regions[1].tune_cents = f32::NAN;
        assert!(matches!(
            SampleProgram::from_bytes(&p.to_bytes()),
            Err(ProgramCodecError::BadRegion { region: 1, .. })
        ));
        let mut p = program();
        p.regions[1].seq = Some((2, 3));
        assert!(SampleProgram::from_bytes(&p.to_bytes()).is_err());
        let p = program();
        let mut b = p.to_bytes();
        b.push(1);
        assert_eq!(
            SampleProgram::from_bytes(&b),
            Err(ProgramCodecError::TrailingBytes)
        );
        let mut b = p.to_bytes();
        b[4] = 0xff;
        b[5] = 0xff;
        assert_eq!(
            SampleProgram::from_bytes(&b),
            Err(ProgramCodecError::UnsupportedVersion(0xffff))
        );
    }

    #[test]
    fn silence_regions_need_no_sample() {
        let mut p = program();
        p.samples.clear();
        p.regions = alloc::vec![Region {
            sample: SILENCE,
            ..Region::default()
        }];
        assert!(SampleProgram::from_bytes(&p.to_bytes()).is_ok());
    }
}
