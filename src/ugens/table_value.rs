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
//! clamped to the first and last entries outside the grid. An entry without
//! the key reads `default`.
//!
//! Bound to the same table as the def's bank (two `table_bindings` naming
//! one table id), so the spectra and the constants of a register travel
//! together.
//!
//! **Cost.** The value is recomputed only when `freq` changes from one
//! sample to the next (a binary search over the entries); driven by the
//! note's `freq` param it is computed once per note.
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
    table: Arc<CoeffTable>,
    /// `(f0, value)` of the entries, sorted by f0, for the latched slot.
    points: Vec<(f32, f32)>,
    started: bool,
    /// Last `freq` and value per cached channel.
    cache: [(f32, f32); CACHED_CHANNELS],
}

impl TableValue {
    /// Construct bound to `table`.
    pub fn new(table: Arc<CoeffTable>) -> Self {
        TableValue {
            table,
            points: Vec::new(),
            started: false,
            cache: [(f32::NAN, 0.0); CACHED_CHANNELS],
        }
    }

    /// The metadata key slot `slot` reads.
    pub fn key(slot: usize) -> alloc::string::String {
        format!("value.{slot}")
    }

    fn latch(&mut self, slot: usize, default: f32) {
        let key = Self::key(slot);
        let mut points: Vec<(f32, f32)> = self
            .table
            .entries
            .iter()
            .filter(|e| e.f0_hz.is_finite())
            .map(|e| {
                let v = e
                    .metadata
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map_or(default, |&(_, v)| v);
                (e.f0_hz, v)
            })
            .collect();
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.points = points;
    }

    /// The value at `f0` (see the module doc).
    pub fn value_at(&self, f0: f32, default: f32) -> f32 {
        let p = &self.points;
        let n = p.len();
        if n == 0 {
            return default;
        }
        if n == 1 || f0 <= p[0].0 || !f0.is_finite() {
            return p[0].1;
        }
        let last = n - 1;
        if f0 >= p[last].0 {
            return p[last].1;
        }
        let (mut lo, mut hi) = (0usize, last);
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if p[mid].0 <= f0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (f_lo, f_hi) = (p[lo].0, p[hi].0);
        let t = if f_hi > f_lo {
            (f0 - f_lo) / (f_hi - f_lo)
        } else {
            0.0
        };
        p[lo].1 + t * (p[hi].1 - p[lo].1)
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
        self.cache = [(f32::NAN, 0.0); CACHED_CHANNELS];
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied().flatten();
        let default = {
            let d = read_input(inputs.get(2).copied().flatten(), 0, 0, 0.0);
            if d.is_finite() { d } else { 0.0 }
        };
        if !self.started {
            self.started = true;
            let s = read_input(inputs.get(1).copied().flatten(), 0, 0, 0.0);
            let slot = if s.is_finite() {
                (s.round().max(0.0) as usize).min(MAX_TABLE_VALUE_SLOT)
            } else {
                0
            };
            self.latch(slot, default);
        }
        for ch in 0..output.num_channels() {
            let mut cache = self.cache.get(ch).copied().unwrap_or((f32::NAN, 0.0));
            let out = output.channel_mut(ch).samples_mut();
            for (i, o) in out.iter_mut().enumerate() {
                let f = read_input(freq_buf, ch, i, DEFAULT_FREQ_HZ);
                // NaN never equals itself, so the first sample always computes
                if f != cache.0 {
                    cache = (f, self.value_at(f, default));
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
