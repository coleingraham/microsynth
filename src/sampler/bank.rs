//! The session's registered programs, each bound to its samples, and the
//! per-program state its voices share.
//!
//! A program is registered after its samples: every content hash in its
//! sample table must already be in the [`SamplePcmBank`], and the bound
//! program keeps its own reference to each. Voices of one program share one
//! [`Session`] (behind a [`SpinCell`]): the round-robin counters, the count
//! of held notes (for `first`/`legato` regions), the event counter that
//! seeds `rand`, and the table of sounding regions that chokes
//! (`group`/`off_by`) and polyphony limits act on. See
//! `docs/sample-program-format.md`, "Shared state".

use super::interp::SincTable;
use super::pcm::{SamplePcm, SamplePcmBank};
use super::program::{Region, SILENCE, SampleProgram, Trigger};
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};

/// A mutex for state shared by voices that the engine renders from one
/// thread: uncontended in practice, correct (it spins) if two threads ever
/// render voices of one program at once. `core` has no mutex, and the
/// shared state must be `Sync` to travel in the `Arc` a graph factory holds.
pub struct SpinCell<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

// SAFETY: access to `value` is serialised by `locked` (see `with`).
unsafe impl<T: Send> Sync for SpinCell<T> {}
unsafe impl<T: Send> Send for SpinCell<T> {}

struct Unlock<'a>(&'a AtomicBool);
impl Drop for Unlock<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl<T> SpinCell<T> {
    /// Wrap a value.
    pub fn new(value: T) -> Self {
        SpinCell {
            locked: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// Run `f` with exclusive access to the value.
    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        let _unlock = Unlock(&self.locked);
        // SAFETY: the flag is held until `_unlock` drops, after `f` returns
        // (or unwinds), so no other `with` can alias the value.
        f(unsafe { &mut *self.value.get() })
    }
}

/// One sounding region of some voice, as the program's other voices see it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PlayerRecord {
    /// The voice (unique per program session).
    pub voice: u64,
    /// The player within the voice.
    pub slot: u32,
    /// Region index.
    pub region: u32,
    pub group: i32,
    pub off_by: Option<i32>,
    /// The note that started it (for a release region, the released note).
    pub note: u8,
    /// MIDI velocity of that note.
    pub vel: u8,
    /// Started by a note-on (`attack`/`first`/`legato`), not a note-off.
    pub on_note: bool,
    /// Event serial of the start: the age order, and the sister set (every
    /// region one event started shares it).
    pub start: u64,
    /// Turned off (choked or stolen); it no longer counts and cannot be
    /// turned off again.
    pub offed: bool,
    /// A pending off: the absolute sample time it takes effect.
    pub off_at: Option<u64>,
}

/// The state a program's voices share.
pub(crate) struct Session {
    next_voice: u64,
    /// Counts note-ons and note-offs; the `rand` draws key on it.
    pub event_serial: u64,
    /// Notes currently held (gate high), for `first`/`legato`.
    pub held: u32,
    /// sfizz's per-region sequence counter.
    pub seq_counter: Vec<u32>,
    /// Whether the region's sequence position was hit at the last note-on
    /// of a key in its range (what a release region consults at note-off).
    pub seq_on: Vec<bool>,
    pub players: Vec<PlayerRecord>,
}

impl Session {
    fn new(regions: usize) -> Self {
        Session {
            next_voice: 1,
            event_serial: 0,
            held: 0,
            seq_counter: alloc::vec![0; regions],
            seq_on: alloc::vec![false; regions],
            players: Vec::with_capacity(256),
        }
    }

    /// A fresh voice id.
    pub fn new_voice(&mut self) -> u64 {
        let v = self.next_voice;
        self.next_voice += 1;
        v
    }

    /// Mark every sister of `rec` (same voice, same start event) offed at `at`.
    pub fn off_with_sisters(&mut self, voice: u64, start: u64, at: u64) {
        for r in self.players.iter_mut() {
            if r.voice == voice && r.start == start && !r.offed {
                r.offed = true;
                r.off_at = Some(at);
            }
        }
    }
}

/// Identifier of a registered program. `0` is never issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramId(pub u32);

/// Errors from registering a program.
#[derive(Debug, Clone, PartialEq)]
pub enum ProgramBindError {
    /// Sample-table entry `index` is not registered in the sample bank.
    MissingSample { index: usize },
}

/// A program bound to its samples, with the state its voices share.
pub struct BoundProgram {
    program: SampleProgram,
    /// Per sample-table entry.
    pcm: Vec<Arc<SamplePcm>>,
    /// Per key 0..=127: the regions whose key window holds it, in program order.
    by_key: Vec<Vec<u32>>,
    /// Per region: its velocity curve as 128 gains, when it has one.
    velcurve: Vec<Option<Vec<f32>>>,
    sinc: SincTable,
    sample_bytes: usize,
    pub(crate) session: SpinCell<Session>,
}

impl BoundProgram {
    /// Bind `program` to the samples in `bank`.
    pub fn bind(program: SampleProgram, bank: &SamplePcmBank) -> Result<Self, ProgramBindError> {
        let mut pcm = Vec::with_capacity(program.samples.len());
        for (index, h) in program.samples.iter().enumerate() {
            match bank.get_by_hash(h) {
                Some(p) => pcm.push(p.clone()),
                None => return Err(ProgramBindError::MissingSample { index }),
            }
        }
        let mut by_key: Vec<Vec<u32>> = (0..128).map(|_| Vec::new()).collect();
        for (i, r) in program.regions.iter().enumerate() {
            for k in r.lokey..=r.hikey {
                by_key[k as usize].push(i as u32);
            }
        }
        let velcurve = program
            .regions
            .iter()
            .map(|r| (!r.amp_velcurve.is_empty()).then(|| velcurve_table(&r.amp_velcurve)))
            .collect();
        // Count each distinct sample once.
        let mut seen: BTreeMap<*const SamplePcm, ()> = BTreeMap::new();
        let mut sample_bytes = 0;
        for p in &pcm {
            if seen.insert(Arc::as_ptr(p), ()).is_none() {
                sample_bytes += p.bytes();
            }
        }
        let n = program.regions.len();
        Ok(BoundProgram {
            program,
            pcm,
            by_key,
            velcurve,
            sinc: SincTable::new(),
            sample_bytes,
            session: SpinCell::new(Session::new(n)),
        })
    }

    /// The program as registered.
    pub fn program(&self) -> &SampleProgram {
        &self.program
    }
    /// Region `i`.
    pub fn region(&self, i: u32) -> &Region {
        &self.program.regions[i as usize]
    }
    /// The regions whose key window holds `key`, in program order.
    pub fn regions_for_key(&self, key: u8) -> &[u32] {
        &self.by_key[key.min(127) as usize]
    }
    /// The sample region `i` reads (`None` for the silence generator).
    pub fn sample_of(&self, i: u32) -> Option<&Arc<SamplePcm>> {
        let s = self.region(i).sample;
        if s == SILENCE {
            None
        } else {
            self.pcm.get(s as usize)
        }
    }
    /// Region `i`'s velocity curve (128 gains) when it has one.
    pub fn velcurve(&self, i: u32) -> Option<&[f32]> {
        self.velcurve[i as usize].as_deref()
    }
    pub(crate) fn sinc(&self) -> &SincTable {
        &self.sinc
    }
    /// Bytes of sample audio the program holds (each sample once).
    pub fn sample_bytes(&self) -> usize {
        self.sample_bytes
    }
    /// Whether some release or release-key region could fire for `key` at
    /// velocity `vel` (keeps a voice alive while its gate is held).
    pub fn has_release_for(&self, key: u8, vel: u8) -> bool {
        self.regions_for_key(key).iter().any(|&i| {
            let r = self.region(i);
            matches!(r.trigger, Trigger::Release | Trigger::ReleaseKey)
                && (r.lovel..=r.hivel).contains(&vel)
        })
    }
}

/// sfizz's velocity curve: 128 points, (0, 0) and (127, 1) unless given,
/// linear between the given points.
pub fn velcurve_table(points: &[(u8, f32)]) -> Vec<f32> {
    let mut v = alloc::vec![0.0f32; 128];
    let mut set = [false; 128];
    v[0] = 0.0;
    set[0] = true;
    v[127] = 1.0;
    set[127] = true;
    for &(k, g) in points {
        let k = k.min(127) as usize;
        v[k] = g;
        set[k] = true;
    }
    let mut last = 0usize;
    for i in 1..128 {
        if set[i] {
            for j in last + 1..i {
                let t = (j - last) as f32 / (i - last) as f32;
                v[j] = v[last] + t * (v[i] - v[last]);
            }
            last = i;
        }
    }
    v
}

/// The session's registered programs.
pub struct SampleProgramBank {
    programs: BTreeMap<ProgramId, Arc<BoundProgram>>,
    by_name: BTreeMap<alloc::string::String, ProgramId>,
    next_id: u32,
}

impl SampleProgramBank {
    /// An empty bank; the first id issued is 1.
    pub fn new() -> Self {
        SampleProgramBank {
            programs: BTreeMap::new(),
            by_name: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Bind and store a program.
    pub fn register(
        &mut self,
        program: SampleProgram,
        samples: &SamplePcmBank,
    ) -> Result<ProgramId, ProgramBindError> {
        let name = program.name.clone();
        let bound = BoundProgram::bind(program, samples)?;
        let id = ProgramId(self.next_id);
        self.next_id += 1;
        if !name.is_empty() {
            self.by_name.insert(name, id);
        }
        self.programs.insert(id, Arc::new(bound));
        Ok(id)
    }

    /// Drop the bank's reference (voices already built keep theirs).
    pub fn free(&mut self, id: ProgramId) {
        if let Some(p) = self.programs.remove(&id)
            && self.by_name.get(&p.program.name) == Some(&id)
        {
            self.by_name.remove(&p.program.name);
        }
    }

    /// Look a program up by id.
    pub fn get(&self, id: ProgramId) -> Option<&Arc<BoundProgram>> {
        self.programs.get(&id)
    }
    /// The id registered under `name` (the spec's program id).
    pub fn id_for_name(&self, name: &str) -> Option<ProgramId> {
        self.by_name.get(name).copied()
    }
    /// Number of registered programs.
    pub fn len(&self) -> usize {
        self.programs.len()
    }
    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.programs.is_empty()
    }
}

impl Default for SampleProgramBank {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::pcm::SamplePcm;

    #[test]
    fn velcurve_fills_linearly_between_points() {
        let t = velcurve_table(&[(64, 0.5)]);
        assert_eq!(t[0], 0.0);
        assert_eq!(t[64], 0.5);
        assert_eq!(t[127], 1.0);
        assert!((t[32] - 0.25).abs() < 1e-6);
        assert!((t[96] - (0.5 + 0.5 * 32.0 / 63.0)).abs() < 1e-6);
        let t = velcurve_table(&[(0, 1.0), (127, 1.0)]);
        assert!(t.iter().all(|&g| g == 1.0));
    }

    #[test]
    fn binding_needs_every_sample_and_counts_bytes_once() {
        let mut samples = SamplePcmBank::new();
        let prog = SampleProgram {
            name: "p".into(),
            spec_sha256: [0; 32],
            samples: alloc::vec![[1; 32], [1; 32]],
            regions: alloc::vec![Region::default()],
        };
        let mut bank = SampleProgramBank::new();
        assert_eq!(
            bank.register(prog.clone(), &samples),
            Err(ProgramBindError::MissingSample { index: 0 })
        );
        let pcm = SamplePcm::new("s", [1; 32], 44_100.0, &[&[0.0; 10][..]]).unwrap();
        let b = pcm.bytes();
        samples.register(pcm);
        let id = bank.register(prog, &samples).unwrap();
        assert_eq!(bank.get(id).unwrap().sample_bytes(), b);
        assert_eq!(bank.id_for_name("p"), Some(id));
        assert_eq!(bank.get(id).unwrap().regions_for_key(60), &[0]);
    }

    #[test]
    fn spin_cell_serialises_access() {
        let c = SpinCell::new(0u32);
        c.with(|v| *v += 1);
        assert_eq!(c.with(|v| *v), 1);
    }
}
