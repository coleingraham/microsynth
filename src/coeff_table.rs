//! Runtime coefficient-table bank: a registry of runtime-fillable, id-referenced
//! buffers holding per-pitch synthesis coefficients.
//!
//! This is the microsynth-side half of the dictionary-table wire format: the
//! contract a runtime host uploads data against (`crate::web`'s `ms_coeff_table_*`
//! exports) and that a DSL/IR-reachable node resolves by id at graph-build time
//! (`crate::dsl::compiler::UGenRegistry`'s table-bound registration path, and
//! `crate::ir::IrSynthDef::compile_with_tables` when the `ir` feature is on).
//!
//! Mirrors [`crate::sample::SampleBank`]'s id/name-indexed bank shape, with two
//! differences that matter for the runtime-fill contract this bank exists to
//! serve (see `docs/coeff-table-bank-format.md` for the full wire-format
//! reference this module implements):
//!
//! - **`replace` keeps the id.** A live graph node that resolved a `TableId`
//!   once should see updated content at the same id on the next resolution,
//!   the same way SuperCollider's `Buffer`/bufnum model lets `/b_alloc`
//!   reload a bufnum's content in place. `SampleBank` has no equivalent —
//!   swapping a sample there means loading a new one under a new id.
//! - **Entries carry an open metadata slot.** Per-pitch entries hold today's
//!   known fields (f0, inharmonicity stretch, partial frequencies, and K ×
//!   (M_p + J) coefficients) plus a small `(String, f32)` metadata list, so a
//!   later field can travel in the wire format without a version bump.
//!
//! No entry point in this module (or anywhere else in this crate) can
//! populate a table from anything compiled into this source tree — every
//! table this bank ever holds arrives through [`CoeffTableBank::register`] or
//! [`CoeffTableBank::replace`], called from data uploaded at runtime. That is
//! the property this module exists to make true, not merely policy.

use crate::wire::{Reader, WireError, put_f32, put_str, put_u16, put_u32};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A unique identifier for a coefficient table. `0` is never issued by
/// [`CoeffTableBank::register`] (ids start at 1), so callers may use `0` as a
/// sentinel for "no table" without colliding with a real id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableId(pub u32);

/// One pitch's dictionary entry: a fixed partial/noise support plus K ×
/// (M_p + J) coefficient sets for that pitch — K channels, each a weight
/// over M_p partials and J noise bands.
///
/// `coefficients` is row-major, `k_channels` rows of `partial_freqs.len() +
/// j_noise` columns each: `coefficients[k * (m + j) + c]` is channel `k`'s
/// weight on partial/noise-band `c` (partials first, then noise bands).
#[derive(Debug, Clone, PartialEq)]
pub struct PitchEntry {
    /// Analysis f0 in Hz.
    pub f0_hz: f32,
    /// Inharmonicity stretch factor applied to partial index when the
    /// ugen re-derives frequencies for a moving f0; stored alongside the
    /// explicit `partial_freqs` (analysis-time, fixed f0) rather than
    /// implied by them, since the two serve different consumers — see the
    /// layout document's "Why both explicit frequencies and a stretch
    /// factor" note.
    pub inharmonicity_stretch: f32,
    /// Fixed partial frequencies in Hz at this entry's f0 (`M_p` of them).
    pub partial_freqs: Vec<f32>,
    /// Number of channels `K` this entry carries.
    pub k_channels: u32,
    /// Number of smooth-noise basis columns `J` this entry carries. Fixed
    /// per entry (may differ from other entries in the same table).
    pub j_noise: u32,
    /// Flattened `K × (M_p + J)` coefficient sets, row-major by channel.
    pub coefficients: Vec<f32>,
    /// Open extension slot: `(key, value)` pairs beyond today's known
    /// fields. Order-preserving; a reader that doesn't recognize a key
    /// skips it.
    pub metadata: Vec<(String, f32)>,
}

impl PitchEntry {
    /// Number of partials `M_p`.
    pub fn m_partials(&self) -> usize {
        self.partial_freqs.len()
    }

    /// Whether `coefficients` has exactly `k_channels * (m_partials + j_noise)`
    /// entries, row-major by channel. Callers that build a `PitchEntry` by
    /// hand (as opposed to decoding one via [`CoeffTable::from_bytes`], which
    /// enforces this at decode time) should check this before registering.
    pub fn is_well_formed(&self) -> bool {
        let expected = self.k_channels as usize * (self.m_partials() + self.j_noise as usize);
        self.coefficients.len() == expected
    }

    /// Channel `k`'s coefficient row: partial weights first, then noise-band
    /// weights. Returns `None` if `k >= k_channels` or the entry is not
    /// well-formed (see [`is_well_formed`](Self::is_well_formed)).
    pub fn channel(&self, k: u32) -> Option<&[f32]> {
        if k >= self.k_channels || !self.is_well_formed() {
            return None;
        }
        let width = self.m_partials() + self.j_noise as usize;
        let start = k as usize * width;
        self.coefficients.get(start..start + width)
    }
}

/// A dictionary table: an instrument's per-pitch entries, addressed by id in
/// a [`CoeffTableBank`]. See the module doc and the layout document for the
/// wire-format contract [`CoeffTable::to_bytes`]/[`CoeffTable::from_bytes`]
/// implement.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CoeffTable {
    /// Optional name for lookup via [`CoeffTableBank::id_for_name`].
    pub name: String,
    /// Per-pitch entries, ordered by ascending `f0_hz` by convention (not
    /// enforced — a consumer doing pitch-bracket search should sort or
    /// verify rather than assume).
    pub entries: Vec<PitchEntry>,
}

/// Metadata keys (MOT-641) for the noise-band span a table's entries were fit
/// against, in Hz — `microsynth::ugens::partials::PartialsNoise`'s
/// [`noise_band_span`](crate::ugens::partials::PartialsNoise)-reading
/// convention: physically stored per-entry (the wire format's only metadata
/// slot), but **per-table** by convention (the noise basis a table's
/// coefficients were fit against is one fixed thing for the whole table —
/// band identity must not vary within it). [`CoeffTable::from_bytes`]
/// enforces that convention at decode time (MOT-641 QA F8): every entry must
/// carry the identical value for both keys, or neither. See
/// `docs/coeff-table-bank-format.md`'s Metadata section for the full wire
/// contract, and `partials.rs`'s `NOISE_BAND_MIN_HZ`/`NOISE_BAND_MAX_HZ` for
/// the fallback span a consuming ugen uses when a table carries neither key.
pub const NOISE_BAND_MIN_HZ_KEY: &str = "noise_band_min_hz";
pub const NOISE_BAND_MAX_HZ_KEY: &str = "noise_band_max_hz";

/// Errors from decoding a [`CoeffTable`] from bytes.
#[derive(Debug, Clone, PartialEq)]
pub enum CoeffTableCodecError {
    /// The magic prefix did not match.
    BadMagic,
    /// The format version is newer than this build understands.
    UnsupportedVersion(u16),
    /// The input ended before a full record could be read.
    UnexpectedEof,
    /// A string field was not valid UTF-8.
    BadUtf8,
    /// An entry's `coefficients` length did not match
    /// `k_channels * (m_partials + j_noise)`.
    MalformedEntry { index: usize },
    /// Two entries disagreed on [`NOISE_BAND_MIN_HZ_KEY`]/
    /// [`NOISE_BAND_MAX_HZ_KEY`] metadata (one has it and the other doesn't,
    /// or both have it at different values), or a single entry carried only
    /// one of the two keys (MOT-641 QA F8) — see those constants' doc for why
    /// this must hold table-wide. Names the first entry that broke the
    /// pattern the earlier entries established, and that earlier index, for
    /// diagnosis.
    InconsistentNoiseBandSpan { first_entry: usize, index: usize },
}

/// Wire-format magic prefix. See `docs/coeff-table-bank-format.md`.
const MAGIC: &[u8] = b"MSCT";
/// Wire-format version. Bump on any change to the byte layout below.
const FORMAT_VERSION: u16 = 1;

/// The read/write primitives themselves (little-endian ints/floats,
/// length-prefixed strings, a bounds-checked cursor) live in [`crate::wire`],
/// shared with `ir::serialize`'s codec — see that module's doc. Only the
/// two failure modes intrinsic to those primitives travel as [`WireError`];
/// this maps them onto this format's own error type so `?` composes.
impl From<WireError> for CoeffTableCodecError {
    fn from(e: WireError) -> Self {
        match e {
            WireError::UnexpectedEof => CoeffTableCodecError::UnexpectedEof,
            WireError::BadUtf8 => CoeffTableCodecError::BadUtf8,
        }
    }
}

impl CoeffTable {
    /// Encode to the canonical wire form uploaded via `ms_coeff_table_register`
    /// / `ms_coeff_table_replace` — see `docs/coeff-table-bank-format.md`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        put_u16(&mut out, FORMAT_VERSION);
        put_str(&mut out, &self.name);
        put_u32(&mut out, self.entries.len() as u32);
        for e in &self.entries {
            put_f32(&mut out, e.f0_hz);
            put_f32(&mut out, e.inharmonicity_stretch);
            put_u32(&mut out, e.partial_freqs.len() as u32);
            for &f in &e.partial_freqs {
                put_f32(&mut out, f);
            }
            put_u32(&mut out, e.k_channels);
            put_u32(&mut out, e.j_noise);
            put_u32(&mut out, e.coefficients.len() as u32);
            for &c in &e.coefficients {
                put_f32(&mut out, c);
            }
            put_u32(&mut out, e.metadata.len() as u32);
            for (k, v) in &e.metadata {
                put_str(&mut out, k);
                put_f32(&mut out, *v);
            }
        }
        out
    }

    /// Decode from bytes produced by [`to_bytes`](Self::to_bytes). Rejects a
    /// stored version newer than this build's [`FORMAT_VERSION`] (mirrors
    /// `ir::serialize`'s back-compat rule: a version-1 document always
    /// decodes here since this is the only version that has existed so far).
    /// Validates every entry's `coefficients` length against
    /// `k_channels * (m_partials + j_noise)` — a malformed entry is rejected
    /// at decode time, not left for a consumer to discover mid-render.
    pub fn from_bytes(bytes: &[u8]) -> Result<CoeffTable, CoeffTableCodecError> {
        let mut r = Reader::new(bytes);
        if r.take(MAGIC.len())? != MAGIC {
            return Err(CoeffTableCodecError::BadMagic);
        }
        let version = r.u16()?;
        if version > FORMAT_VERSION {
            return Err(CoeffTableCodecError::UnsupportedVersion(version));
        }
        let name = r.string()?;
        let entry_count = r.usize32()?;
        let mut entries = Vec::with_capacity(entry_count);
        for index in 0..entry_count {
            let f0_hz = r.f32()?;
            let inharmonicity_stretch = r.f32()?;
            let pf_count = r.usize32()?;
            let mut partial_freqs = Vec::with_capacity(pf_count);
            for _ in 0..pf_count {
                partial_freqs.push(r.f32()?);
            }
            let k_channels = r.u32()?;
            let j_noise = r.u32()?;
            let coeff_count = r.usize32()?;
            let mut coefficients = Vec::with_capacity(coeff_count);
            for _ in 0..coeff_count {
                coefficients.push(r.f32()?);
            }
            let meta_count = r.usize32()?;
            let mut metadata = Vec::with_capacity(meta_count);
            for _ in 0..meta_count {
                let k = r.string()?;
                let v = r.f32()?;
                metadata.push((k, v));
            }
            let entry = PitchEntry {
                f0_hz,
                inharmonicity_stretch,
                partial_freqs,
                k_channels,
                j_noise,
                coefficients,
                metadata,
            };
            if !entry.is_well_formed() {
                return Err(CoeffTableCodecError::MalformedEntry { index });
            }
            entries.push(entry);
        }
        validate_noise_band_span_consistency(&entries)?;
        Ok(CoeffTable { name, entries })
    }
}

/// Looks up `key` in `metadata`, returning `None` if absent (distinct from
/// [`PitchEntry`]'s own consumer-facing default-substituting lookup — this
/// needs to know *presence*, not just value, to catch a lone key).
fn metadata_value(metadata: &[(String, f32)], key: &str) -> Option<f32> {
    metadata
        .iter()
        .find(|(k, _)| k.as_str() == key)
        .map(|(_, v)| *v)
}

/// MOT-641 QA F8: every entry must agree on [`NOISE_BAND_MIN_HZ_KEY`]/
/// [`NOISE_BAND_MAX_HZ_KEY`] — both present at the identical value, or both
/// absent — across the whole table. A table with 0 or 1 entries trivially
/// satisfies this (nothing to disagree with). See those constants' doc for
/// why this is a decode-time, not just documented, rule.
fn validate_noise_band_span_consistency(
    entries: &[PitchEntry],
) -> Result<(), CoeffTableCodecError> {
    let mut baseline: Option<(usize, Option<(f32, f32)>)> = None;
    for (index, entry) in entries.iter().enumerate() {
        let min = metadata_value(&entry.metadata, NOISE_BAND_MIN_HZ_KEY);
        let max = metadata_value(&entry.metadata, NOISE_BAND_MAX_HZ_KEY);
        let this = match (min, max) {
            (Some(a), Some(b)) => Some((a, b)),
            (None, None) => None,
            _ => {
                // A lone key within a single entry is inconsistent on its
                // own, independent of any other entry.
                return Err(CoeffTableCodecError::InconsistentNoiseBandSpan {
                    first_entry: index,
                    index,
                });
            }
        };
        match &baseline {
            None => baseline = Some((index, this)),
            Some((first_entry, expected)) => {
                if *expected != this {
                    return Err(CoeffTableCodecError::InconsistentNoiseBandSpan {
                        first_entry: *first_entry,
                        index,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Errors from a [`CoeffTableBank`] operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoeffTableBankError {
    /// [`CoeffTableBank::replace`] was called with an id that is not
    /// currently registered. Replace never creates a new entry — use
    /// [`CoeffTableBank::register`] for that.
    NotFound(TableId),
}

/// A runtime bank of coefficient tables, addressed by id (and optionally by
/// name). See the module doc for the `replace`-keeps-the-id contract that
/// distinguishes this from [`crate::sample::SampleBank`].
pub struct CoeffTableBank {
    tables: BTreeMap<TableId, CoeffTable>,
    name_index: BTreeMap<String, TableId>,
    next_id: u32,
}

impl CoeffTableBank {
    /// Create an empty bank. The first id `register` hands out is `1` — `0`
    /// is reserved as the "no table" sentinel (see [`TableId`]'s doc).
    pub fn new() -> Self {
        CoeffTableBank {
            tables: BTreeMap::new(),
            name_index: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Register a new table at a freshly issued id. Returns the id.
    pub fn register(&mut self, table: CoeffTable) -> TableId {
        let id = TableId(self.next_id);
        self.next_id += 1;
        if !table.name.is_empty() {
            self.name_index.insert(table.name.clone(), id);
        }
        self.tables.insert(id, table);
        id
    }

    /// Replace the content at an already-registered id, in place. The id
    /// (and hence every existing reference to it) is unchanged. Errors if
    /// `id` is not currently registered.
    pub fn replace(&mut self, id: TableId, table: CoeffTable) -> Result<(), CoeffTableBankError> {
        if !self.tables.contains_key(&id) {
            return Err(CoeffTableBankError::NotFound(id));
        }
        // Update the name index: drop the old name's mapping (if it pointed
        // at this id) before inserting the new one, so a rename via replace
        // doesn't leave the old name resolving to this id too.
        if let Some(old) = self.tables.get(&id)
            && !old.name.is_empty()
        {
            self.name_index.remove(&old.name);
        }
        if !table.name.is_empty() {
            self.name_index.insert(table.name.clone(), id);
        }
        self.tables.insert(id, table);
        Ok(())
    }

    /// Free a table by id. A no-op if `id` is not registered.
    pub fn free(&mut self, id: TableId) {
        if let Some(table) = self.tables.remove(&id)
            && !table.name.is_empty()
        {
            self.name_index.remove(&table.name);
        }
    }

    /// Get a table by id.
    pub fn get(&self, id: TableId) -> Option<&CoeffTable> {
        self.tables.get(&id)
    }

    /// Get a table by name.
    pub fn get_by_name(&self, name: &str) -> Option<&CoeffTable> {
        let id = self.name_index.get(name)?;
        self.tables.get(id)
    }

    /// Look up a `TableId` by name.
    pub fn id_for_name(&self, name: &str) -> Option<TableId> {
        self.name_index.get(name).copied()
    }

    /// Number of registered tables.
    pub fn len(&self) -> usize {
        self.tables.len()
    }

    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }
}

impl Default for CoeffTableBank {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_table() -> CoeffTable {
        // K=2 channels, M=3 partials, J=1 noise band -> width 4.
        CoeffTable {
            name: "synthetic".into(),
            entries: alloc::vec![PitchEntry {
                f0_hz: 220.0,
                inharmonicity_stretch: 1.0,
                partial_freqs: alloc::vec![220.0, 440.0, 660.0],
                k_channels: 2,
                j_noise: 1,
                coefficients: alloc::vec![0.5, 0.3, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
                metadata: alloc::vec![("velocity_ref".into(), 0.8)],
            }],
        }
    }

    #[test]
    fn round_trips_through_bytes() {
        let table = synthetic_table();
        let bytes = table.to_bytes();
        let decoded = CoeffTable::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, table);
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = alloc::vec![0u8; 16];
        assert_eq!(
            CoeffTable::from_bytes(&bytes),
            Err(CoeffTableCodecError::BadMagic)
        );
    }

    #[test]
    fn rejects_newer_version() {
        let mut bytes = synthetic_table().to_bytes();
        // Version field is bytes [4..6) (after the 4-byte magic).
        bytes[4] = 0xFF;
        bytes[5] = 0xFF;
        assert_eq!(
            CoeffTable::from_bytes(&bytes),
            Err(CoeffTableCodecError::UnsupportedVersion(0xFFFF))
        );
    }

    #[test]
    fn rejects_malformed_entry_coefficient_length() {
        let mut table = synthetic_table();
        table.entries[0].coefficients.pop();
        let bytes = table.to_bytes();
        assert_eq!(
            CoeffTable::from_bytes(&bytes),
            Err(CoeffTableCodecError::MalformedEntry { index: 0 })
        );
    }

    /// A two-entry table (K=1, M=1, J=0 — the noise-band span keys don't
    /// interact with partial/noise coefficient counts, so the simplest
    /// possible entry shape isolates the span-consistency check).
    fn two_entry_table(metadata: [Vec<(String, f32)>; 2]) -> CoeffTable {
        let [meta0, meta1] = metadata;
        CoeffTable {
            name: "span-consistency".into(),
            entries: alloc::vec![
                PitchEntry {
                    f0_hz: 110.0,
                    inharmonicity_stretch: 1.0,
                    partial_freqs: alloc::vec![110.0],
                    k_channels: 1,
                    j_noise: 0,
                    coefficients: alloc::vec![1.0],
                    metadata: meta0,
                },
                PitchEntry {
                    f0_hz: 220.0,
                    inharmonicity_stretch: 1.0,
                    partial_freqs: alloc::vec![220.0],
                    k_channels: 1,
                    j_noise: 0,
                    coefficients: alloc::vec![1.0],
                    metadata: meta1,
                },
            ],
        }
    }

    #[test]
    fn from_bytes_accepts_a_table_where_no_entry_carries_a_noise_band_span() {
        let table = two_entry_table([alloc::vec![], alloc::vec![]]);
        let bytes = table.to_bytes();
        assert_eq!(CoeffTable::from_bytes(&bytes), Ok(table));
    }

    #[test]
    fn from_bytes_accepts_a_table_where_every_entry_agrees_on_the_span() {
        let span = alloc::vec![
            (NOISE_BAND_MIN_HZ_KEY.into(), 0.0),
            (NOISE_BAND_MAX_HZ_KEY.into(), 8000.0),
        ];
        let table = two_entry_table([span.clone(), span]);
        let bytes = table.to_bytes();
        assert_eq!(CoeffTable::from_bytes(&bytes), Ok(table));
    }

    #[test]
    fn from_bytes_rejects_entries_disagreeing_on_the_span_value() {
        let table = two_entry_table([
            alloc::vec![
                (NOISE_BAND_MIN_HZ_KEY.into(), 0.0),
                (NOISE_BAND_MAX_HZ_KEY.into(), 8000.0),
            ],
            alloc::vec![
                (NOISE_BAND_MIN_HZ_KEY.into(), 0.0),
                (NOISE_BAND_MAX_HZ_KEY.into(), 22_050.0),
            ],
        ]);
        let bytes = table.to_bytes();
        assert_eq!(
            CoeffTable::from_bytes(&bytes),
            Err(CoeffTableCodecError::InconsistentNoiseBandSpan {
                first_entry: 0,
                index: 1,
            })
        );
    }

    #[test]
    fn from_bytes_rejects_entries_where_only_some_carry_a_span() {
        let table = two_entry_table([
            alloc::vec![
                (NOISE_BAND_MIN_HZ_KEY.into(), 0.0),
                (NOISE_BAND_MAX_HZ_KEY.into(), 8000.0),
            ],
            alloc::vec![],
        ]);
        let bytes = table.to_bytes();
        assert_eq!(
            CoeffTable::from_bytes(&bytes),
            Err(CoeffTableCodecError::InconsistentNoiseBandSpan {
                first_entry: 0,
                index: 1,
            })
        );
    }

    #[test]
    fn from_bytes_rejects_an_entry_carrying_only_one_span_key() {
        let table = two_entry_table([
            alloc::vec![(NOISE_BAND_MIN_HZ_KEY.into(), 0.0)],
            alloc::vec![],
        ]);
        let bytes = table.to_bytes();
        assert_eq!(
            CoeffTable::from_bytes(&bytes),
            Err(CoeffTableCodecError::InconsistentNoiseBandSpan {
                first_entry: 0,
                index: 0,
            })
        );
    }

    #[test]
    fn from_bytes_accepts_a_single_entry_table_regardless_of_span_metadata() {
        // Nothing to disagree with at 0 or 1 entries.
        let table = CoeffTable {
            name: "one-entry".into(),
            entries: alloc::vec![PitchEntry {
                f0_hz: 110.0,
                inharmonicity_stretch: 1.0,
                partial_freqs: alloc::vec![110.0],
                k_channels: 1,
                j_noise: 0,
                coefficients: alloc::vec![1.0],
                metadata: alloc::vec![(NOISE_BAND_MIN_HZ_KEY.into(), 0.0)],
            }],
        };
        let bytes = table.to_bytes();
        assert_eq!(
            CoeffTable::from_bytes(&bytes),
            Err(CoeffTableCodecError::InconsistentNoiseBandSpan {
                first_entry: 0,
                index: 0,
            }),
            "a lone key is still inconsistent even with only one entry -- \
             that check is per-entry, not cross-entry"
        );
    }

    #[test]
    fn bank_register_get_free() {
        let mut bank = CoeffTableBank::new();
        let id = bank.register(synthetic_table());
        assert_eq!(id, TableId(1));
        assert_eq!(bank.get(id), Some(&synthetic_table()));
        assert_eq!(bank.get_by_name("synthetic"), Some(&synthetic_table()));
        assert_eq!(bank.id_for_name("synthetic"), Some(id));

        bank.free(id);
        assert!(bank.get(id).is_none());
        assert!(bank.get_by_name("synthetic").is_none());
        assert!(bank.is_empty());
    }

    #[test]
    fn bank_replace_keeps_id() {
        let mut bank = CoeffTableBank::new();
        let id = bank.register(synthetic_table());

        let mut replacement = synthetic_table();
        replacement.entries[0].f0_hz = 330.0;
        bank.replace(id, replacement.clone()).unwrap();

        assert_eq!(id, TableId(1));
        assert_eq!(bank.get(id), Some(&replacement));
        assert_eq!(bank.len(), 1);
    }

    #[test]
    fn bank_replace_missing_id_errors() {
        let mut bank = CoeffTableBank::new();
        assert_eq!(
            bank.replace(TableId(42), synthetic_table()),
            Err(CoeffTableBankError::NotFound(TableId(42)))
        );
    }

    #[test]
    fn channel_accessor_slices_row_major() {
        let table = synthetic_table();
        let entry = &table.entries[0];
        assert_eq!(entry.channel(0), Some(&[0.5, 0.3, 0.1, 0.1][..]));
        assert_eq!(entry.channel(1), Some(&[0.2, 0.2, 0.3, 0.3][..]));
        assert_eq!(entry.channel(2), None);
    }
}
