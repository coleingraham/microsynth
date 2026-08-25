//! Physical modeling oscillators.
//!
//! - [`Pluck`]: Karplus-Strong plucked string synthesis
//! - [`Bowed`]: Digital waveguide bowed string model

use super::rng::Rng;
use crate::buffer::{AudioBuffer, MAX_BLOCK_SIZE, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use alloc::vec::Vec;

/// Minimum supported frequency (determines max buffer size).
const MIN_FREQ: f32 = 20.0;

// --- Pluck (Karplus-Strong) ---

/// Karplus-Strong plucked string synthesis.
///
/// On trigger, fills an internal delay line with a noise burst, then
/// recirculates through a one-pole lowpass filter with decay feedback.
/// Signals `is_done()` when energy drops below threshold.
///
/// Inputs: freq (Hz), decay (feedback 0-1, default 0.99), trig (trigger on
/// positive-going edge; default 1.0, so an unconnected trig auto-plucks once at
/// the start of the render — the natural one-shot behaviour, and audible rather
/// than silent when the port is left at its default).
pub struct Pluck {
    buffer: Vec<f32>,
    buf_len: usize,
    write_pos: usize,
    sample_rate: f32,
    filter_state: f32,
    energy: f32,
    rng: Rng,
    initialized: bool,
    prev_trig: f32,
}

impl Default for Pluck {
    fn default() -> Self {
        Self::new()
    }
}

impl Pluck {
    pub fn new() -> Self {
        Pluck {
            buffer: Vec::new(),
            buf_len: 0,
            write_pos: 0,
            sample_rate: 44100.0,
            filter_state: 0.0,
            energy: 0.0,
            rng: Rng::new(0xBEEF_CAFE),
            initialized: false,
            prev_trig: 0.0,
        }
    }

    fn trigger(&mut self, freq: f32) {
        let period = (self.sample_rate / freq.max(MIN_FREQ)).round() as usize;
        self.buf_len = period.max(2).min(self.buffer.len());
        // Fill delay line with noise burst
        for i in 0..self.buf_len {
            self.buffer[i] = self.rng.next_bipolar();
        }
        self.write_pos = 0;
        self.filter_state = 0.0;
        self.energy = 1.0;
        self.initialized = true;
    }
}

impl UGen for Pluck {
    ugen_spec!(
        "Pluck",
        category = Physical,
        inputs = [],
        optional_inputs = ["freq", "decay", "trig"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (context.sample_rate / MIN_FREQ) as usize + 2;
        self.buffer.resize(max_samples, 0.0);
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.buf_len = 0;
        self.write_pos = 0;
        self.filter_state = 0.0;
        self.energy = 0.0;
        self.initialized = false;
        self.prev_trig = 0.0;
    }

    fn is_done(&self) -> bool {
        self.initialized && self.energy < 1e-5
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied().flatten();
        let decay_buf = inputs.get(1).copied().flatten();
        let trig_buf = inputs.get(2).copied().flatten();

        // Pluck models a single shared string: `self.buffer` (the KS delay
        // line) and `self.rng` (consumed by `trigger()`'s noise burst) are
        // genuinely shared, mutable resources, not per-channel state. The
        // old per-channel loop below re-read `self.write_pos` etc. fresh on
        // each channel iteration (the same read-back-inside-loop bug fixed
        // throughout this crate — see filters::OnePole's process()
        // comment), but ALSO called `self.trigger()` — which mutates
        // `self.buffer` and advances `self.rng` — once per output channel
        // for the very same trigger edge, decorrelating channel 1's noise
        // burst from channel 0's. Both bugs are fixed together by running
        // the recurrence exactly once per sample (into a stack-allocated
        // mono scratch buffer) and copying that single result to every
        // output channel, instead of re-deriving it once per channel.
        let block_size = output.block_size();
        let mut mono = [0.0f32; MAX_BLOCK_SIZE];

        let mut write_pos = self.write_pos;
        let mut filter_state = self.filter_state;
        let mut energy = self.energy;
        let mut prev_trig = self.prev_trig;

        for (i, mono_sample) in mono[..block_size].iter_mut().enumerate() {
            let freq = read_input(freq_buf, 0, i, 440.0);
            let decay = read_input(decay_buf, 0, i, 0.99).clamp(0.0, 0.999);
            let trig = read_input(trig_buf, 0, i, 1.0);

            // Trigger detection (positive-going zero crossing)
            if trig > 0.0 && prev_trig <= 0.0 {
                self.trigger(freq);
                write_pos = self.write_pos;
                filter_state = self.filter_state;
                energy = self.energy;
            }
            prev_trig = trig;

            if self.buf_len < 2 {
                *mono_sample = 0.0;
                continue;
            }

            // Read from delay line
            let read_pos = write_pos;
            let delayed = self.buffer[read_pos];

            // One-pole lowpass (classic KS averaging filter)
            // Average current and previous: simple but effective damping
            let next_pos = (read_pos + 1) % self.buf_len;
            let next_sample = self.buffer[next_pos];
            filter_state = 0.5 * (delayed + next_sample);

            // Write back with decay
            self.buffer[write_pos] = filter_state * decay;
            *mono_sample = filter_state;

            write_pos = (write_pos + 1) % self.buf_len;

            // Track energy (exponential follower)
            energy = 0.999 * energy + 0.001 * filter_state.abs();
        }

        self.write_pos = write_pos;
        self.filter_state = filter_state;
        self.energy = energy;
        self.prev_trig = prev_trig;

        for ch in 0..output.num_channels() {
            output.channel_mut(ch).samples_mut()[..block_size].copy_from_slice(&mono[..block_size]);
        }
    }
}

// --- Bowed String ---

/// Digital waveguide bowed string model.
///
/// Uses two delay lines (nut-side and bridge-side) with a nonlinear
/// bow-string interaction at the bow point. One-pole lowpass filters
/// at each termination model frequency-dependent losses.
///
/// Inputs: freq (Hz), pressure (bow pressure 0-1, default 0.5),
///         position (bow position on string 0-1, default 0.13).
pub struct Bowed {
    nut_delay: Vec<f32>,
    nut_write: usize,
    bridge_delay: Vec<f32>,
    bridge_write: usize,
    nut_filter: f32,
    bridge_filter: f32,
    sample_rate: f32,
}

impl Default for Bowed {
    fn default() -> Self {
        Self::new()
    }
}

impl Bowed {
    pub fn new() -> Self {
        Bowed {
            nut_delay: Vec::new(),
            nut_write: 0,
            bridge_delay: Vec::new(),
            bridge_write: 0,
            nut_filter: 0.0,
            bridge_filter: 0.0,
            sample_rate: 44100.0,
        }
    }
}

/// Bow friction table: maps relative velocity to friction force.
/// Semi-circular curve scaled by bow pressure.
#[inline]
fn bow_table(delta_v: f32, pressure: f32) -> f32 {
    let x = delta_v * pressure.max(0.01) * 5.0;
    let val = 1.0 - x * x;
    if val > 0.0 { val.sqrt() } else { 0.0 }
}

impl UGen for Bowed {
    ugen_spec!(
        "Bowed",
        category = Physical,
        inputs = [],
        optional_inputs = ["freq", "pressure", "position"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
        let max_samples = (context.sample_rate / MIN_FREQ) as usize + 2;
        self.nut_delay.resize(max_samples, 0.0);
        self.bridge_delay.resize(max_samples, 0.0);
    }

    fn reset(&mut self) {
        self.nut_delay.fill(0.0);
        self.bridge_delay.fill(0.0);
        self.nut_write = 0;
        self.bridge_write = 0;
        self.nut_filter = 0.0;
        self.bridge_filter = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied().flatten();
        let pressure_buf = inputs.get(1).copied().flatten();
        let position_buf = inputs.get(2).copied().flatten();
        let max_len = self.nut_delay.len();
        if max_len == 0 {
            return;
        }

        // Bowed models a single shared string: `self.nut_delay` and
        // `self.bridge_delay` are the waveguide's own delay-line memory —
        // genuinely shared, mutable resources, not per-channel state (same
        // class of bug as `Pluck`, see its process() comment). The old
        // per-channel loop wrote `self.nut_delay[nut_write]` /
        // `self.bridge_delay[bridge_write]` directly from inside a loop
        // that reran once per output channel: channel 1's pass wrote (and
        // read back) delay-line contents that channel 0's pass, running
        // first over the *entire* block, had already advanced a full block
        // ahead. Fixed by running the recurrence exactly once per sample
        // (into a stack-allocated mono scratch buffer) and copying that
        // single result to every output channel.
        let block_size = output.block_size();
        let mut mono = [0.0f32; MAX_BLOCK_SIZE];

        let mut nut_write = self.nut_write;
        let mut bridge_write = self.bridge_write;
        let mut nut_filter = self.nut_filter;
        let mut bridge_filter = self.bridge_filter;

        for (i, mono_sample) in mono[..block_size].iter_mut().enumerate() {
            let freq = read_input(freq_buf, 0, i, 220.0).clamp(MIN_FREQ, self.sample_rate * 0.45);
            let pressure = read_input(pressure_buf, 0, i, 0.5).clamp(0.0, 1.0);
            let position = read_input(position_buf, 0, i, 0.13).clamp(0.02, 0.98);

            // Compute delay lengths from frequency and bow position
            let total_delay = self.sample_rate / freq;
            let nut_len = ((total_delay * position) as usize).clamp(1, max_len - 1);
            let bridge_len = ((total_delay * (1.0 - position)) as usize).clamp(1, max_len - 1);

            // Read returning waves from delay lines (arrived at terminations)
            let nut_read = (nut_write + max_len - nut_len) % max_len;
            let nut_out = self.nut_delay[nut_read];
            let bridge_read = (bridge_write + max_len - bridge_len) % max_len;
            let bridge_out = self.bridge_delay[bridge_read];

            // Reflections at terminations: inversion + one-pole lowpass (loss model)
            nut_filter = nut_filter * 0.55 + (-nut_out) * 0.45;
            bridge_filter = bridge_filter * 0.55 + (-bridge_out) * 0.45;

            // String velocity at bow point (sum of incoming waves from both sides)
            let v_string = nut_filter + bridge_filter;
            let v_bow = 0.3 * pressure;
            let delta_v = v_bow - v_string;

            // Bow-string interaction
            let force = bow_table(delta_v, pressure) * pressure * 0.3;

            // Cross-couple: reflected wave from each side passes through
            // the bow point and enters the opposite delay line
            self.nut_delay[nut_write] = bridge_filter + force;
            self.bridge_delay[bridge_write] = nut_filter + force;

            // Output from bridge side (pickup position)
            *mono_sample = bridge_out.clamp(-1.0, 1.0);

            nut_write = (nut_write + 1) % max_len;
            bridge_write = (bridge_write + 1) % max_len;
        }

        self.nut_write = nut_write;
        self.bridge_write = bridge_write;
        self.nut_filter = nut_filter;
        self.bridge_filter = bridge_filter;

        for ch in 0..output.num_channels() {
            output.channel_mut(ch).samples_mut()[..block_size].copy_from_slice(&mono[..block_size]);
        }
    }
}
