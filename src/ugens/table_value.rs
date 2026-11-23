//! `tableValue`: a number given per pitch in a coefficient table, read at
//! `freq`.
//!
//! A def whose instrument changes with register (one def spanning several
//! instruments' ranges, its spectra merged into one table) also has
//! constants that change with register: vibrato rate and depth, envelope
//! times, filter corners. The table's entries can carry them in their open
//! metadata slot (`docs/coeff-table-bank-format.md`), under the keys
//! `value.0`, `value.1`, …; this node outputs entry metadata `value.<slot>`
//! interpolated at `freq` exactly as the partial banks interpolate their
//! coefficients: between the two entries bracketing `freq`, linear in f0,
//! clamped to the first and last entries outside the grid. Wherever the key
//! is missing (an entry without it, every entry, or a table with no entries)
//! the value there is `default`: an entry lacking the key counts as carrying
//! `default` in the interpolation, and a table where nothing carries it reads
//! `default` everywhere. Entries that fail the table format's checks
//! (`PitchEntry::is_well_formed`) are skipped, as the partial banks skip
//! them.
//!
//! Bound to the same table as the def's bank (two `table_bindings` naming
//! one table id), so the spectra and the constants of a register travel
//! together.
//!
//! **Cost.** Every `value.<n>` key is parsed once, when the node is built
//! (no allocation while rendering). The value is recomputed only when `freq`
//! changes from one sample to the next (a binary search over the entries);
//! driven by the note's `freq` param it is computed once per note.
//!
//! **Channels.** One output channel per `freq` channel, each from its own
//! channel of `freq` (with a cache per channel); `slot` and `default` are
//! read once, at the voice's first sample, from their first channel.
//!
//! **Done rule.** A source of a parameter value: it never reports done.
//!
//! Ports: `freq` (Hz, default 220), `slot` (default 0), `default` (default 0).

use crate::buffer::{AudioBuffer, read_input};
use crate::coeff_table::CoeffTable;
use crate::context::{ProcessContext, Rate};
use crate::dsl::compiler::{TableUGenFactory, UGenRegistry};
use crate::node::{InputSpec, OutputSpec, UGen, UGenCategory};
use crate::ugens::partials::DEFAULT_FREQ_HZ;
use alloc::boxed::Box;
use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;

/// Highest slot a def may name.
pub const MAX_TABLE_VALUE_SLOT: usize = 255;

/// Channels with their own cache; further channels recompute every sample.
const CACHED_CHANNELS: usize = 2;

/// The table-bound value node. See the module doc.
pub struct TableValue {
    /// The well-formed entries' f0, sorted ascending.
    f0: Vec<f32>,
    /// `(slot, values)`: one value per entry in `f0`'s order, `NaN` where the
    /// entry lacks the key. Sorted by slot.
    slots: Vec<(usize, Vec<f32>)>,
    /// The latched slot's index into `slots` (`None`: no entry carries it).
    slot: Option<usize>,
    default: f32,
    started: bool,
    /// Last `freq` and value per cached channel.
    cache: [(f32, f32); CACHED_CHANNELS],
}

impl TableValue {
    /// Construct bound to `table`: parse every `value.<n>` key.
    pub fn new(table: Arc<CoeffTable>) -> Self {
        let mut entries: Vec<_> = table
            .entries
            .iter()
            .filter(|e| e.is_well_formed() && e.f0_hz.is_finite())
            .collect();
        entries.sort_by(|a, b| a.f0_hz.total_cmp(&b.f0_hz));
        let f0: Vec<f32> = entries.iter().map(|e| e.f0_hz).collect();
        let mut slots: Vec<(usize, Vec<f32>)> = Vec::new();
        for (i, e) in entries.iter().enumerate() {
            for (key, v) in &e.metadata {
                let Some(n) = key
                    .strip_prefix("value.")
                    .and_then(|n| n.parse::<usize>().ok())
                else {
                    continue;
                };
                if n > MAX_TABLE_VALUE_SLOT {
                    continue;
                }
                let at = match slots.binary_search_by_key(&n, |s| s.0) {
                    Ok(at) => at,
                    Err(at) => {
                        slots.insert(at, (n, alloc::vec![f32::NAN; entries.len()]));
                        at
                    }
                };
                slots[at].1[i] = *v;
            }
        }
        TableValue {
            f0,
            slots,
            slot: None,
            default: 0.0,
            started: false,
            cache: [(f32::NAN, 0.0); CACHED_CHANNELS],
        }
    }

    /// The metadata key slot `slot` reads.
    pub fn key(slot: usize) -> alloc::string::String {
        format!("value.{slot}")
    }

    /// The value of slot `slot` at `f0`, `default` where the key is missing
    /// (see the module doc).
    pub fn value_at(&self, slot: usize, f0: f32, default: f32) -> f32 {
        match self.slots.binary_search_by_key(&slot, |s| s.0) {
            Ok(at) => self.lookup(&self.slots[at].1, f0, default),
            Err(_) => default,
        }
    }

    fn lookup(&self, values: &[f32], f0: f32, default: f32) -> f32 {
        let v = |i: usize| {
            let x = values[i];
            if x.is_nan() { default } else { x }
        };
        let p = &self.f0;
        let n = p.len();
        if n == 0 {
            return default;
        }
        if n == 1 || f0 <= p[0] || !f0.is_finite() {
            return v(0);
        }
        let last = n - 1;
        if f0 >= p[last] {
            return v(last);
        }
        let (mut lo, mut hi) = (0usize, last);
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if p[mid] <= f0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (f_lo, f_hi) = (p[lo], p[hi]);
        let t = if f_hi > f_lo {
            (f0 - f_lo) / (f_hi - f_lo)
        } else {
            0.0
        };
        v(lo) + t * (v(hi) - v(lo))
    }

    fn current(&self, f0: f32) -> f32 {
        match self.slot {
            Some(at) => self.lookup(&self.slots[at].1, f0, self.default),
            None => self.default,
        }
    }
}

static INPUTS: [InputSpec; 3] = [
    InputSpec {
        name: "freq",
        rate: Rate::Audio,
        required: false,
    },
    InputSpec {
        name: "slot",
        rate: Rate::Audio,
        required: false,
    },
    InputSpec {
        name: "default",
        rate: Rate::Audio,
        required: false,
    },
];

impl UGen for TableValue {
    fn spec(&self) -> crate::node::UGenSpec {
        crate::node::UGenSpec {
            name: "TableValue",
            category: UGenCategory::Utility,
            inputs: &INPUTS,
            outputs: &[OutputSpec {
                name: "out",
                rate: Rate::Audio,
            }],
        }
    }

    fn init(&mut self, _context: &ProcessContext) {}

    fn reset(&mut self) {
        self.started = false;
        self.slot = None;
        self.default = 0.0;
        self.cache = [(f32::NAN, 0.0); CACHED_CHANNELS];
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied().flatten();
        if !self.started {
            self.started = true;
            let d = read_input(inputs.get(2).copied().flatten(), 0, 0, 0.0);
            self.default = if d.is_finite() { d } else { 0.0 };
            let s = read_input(inputs.get(1).copied().flatten(), 0, 0, 0.0);
            let slot = if s.is_finite() {
                (s.round().max(0.0) as usize).min(MAX_TABLE_VALUE_SLOT)
            } else {
                0
            };
            self.slot = self.slots.binary_search_by_key(&slot, |s| s.0).ok();
        }
        for ch in 0..output.num_channels() {
            let mut cache = self.cache.get(ch).copied().unwrap_or((f32::NAN, 0.0));
            let out = output.channel_mut(ch).samples_mut();
            for (i, o) in out.iter_mut().enumerate() {
                let f = read_input(freq_buf, ch, i, DEFAULT_FREQ_HZ);
                // NaN never equals itself, so the first sample always computes
                if f != cache.0 {
                    cache = (f, self.current(f));
                }
                *o = cache.1;
            }
            if let Some(c) = self.cache.get_mut(ch) {
                *c = cache;
            }
        }
    }
}

/// Register the `tableValue` table-bound kind.
pub fn register(reg: &mut UGenRegistry) {
    let factory: TableUGenFactory = Arc::new(|table| Box::new(TableValue::new(table)));
    reg.register_table_bound(
        "tableValue",
        factory,
        UGenCategory::Utility,
        &INPUTS,
        &[OutputSpec {
            name: "out",
            rate: Rate::Audio,
        }],
    );
}
