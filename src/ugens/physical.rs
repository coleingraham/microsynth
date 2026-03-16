//! Physical modeling oscillators.
//!
//! - [`Pluck`]: Karplus-Strong plucked string synthesis
//! - [`Bowed`]: Digital waveguide bowed string model

use crate::buffer::AudioBuffer;
use crate::context::{ProcessContext, Rate};
use crate::node::{InputSpec, OutputSpec, UGen, UGenSpec};
use super::rng::Rng;
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
/// Inputs: freq (Hz), decay (feedback 0-1, default 0.99), trig (trigger on positive edge).
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

static PLUCK_INPUTS: [InputSpec; 3] = [
    InputSpec { name: "freq", rate: Rate::Audio },
    InputSpec { name: "decay", rate: Rate::Audio },
    InputSpec { name: "trig", rate: Rate::Audio },
];
static PLUCK_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Pluck {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Pluck", inputs: &PLUCK_INPUTS, outputs: &PLUCK_OUTPUTS }
    }

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
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied();
        let decay_buf = inputs.get(1).copied();
        let trig_buf = inputs.get(2).copied();

        for ch in 0..output.num_channels() {
            let mut write_pos = self.write_pos;
            let mut filter_state = self.filter_state;
            let mut energy = self.energy;
            let mut prev_trig = self.prev_trig;
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(440.0);
                let decay = decay_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.99)
                    .clamp(0.0, 0.999);
                let trig = trig_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.0);

                // Trigger detection (positive-going zero crossing)
                if trig > 0.0 && prev_trig <= 0.0 {
                    self.trigger(freq);
                    write_pos = self.write_pos;
                    filter_state = self.filter_state;
                    energy = self.energy;
                }
                prev_trig = trig;

                if self.buf_len < 2 {
                    out[i] = 0.0;
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
                out[i] = filter_state;

                write_pos = (write_pos + 1) % self.buf_len;

                // Track energy (exponential follower)
                energy = 0.999 * energy + 0.001 * filter_state.abs();
            }

            if ch == 0 {
                self.write_pos = write_pos;
                self.filter_state = filter_state;
                self.energy = energy;
                self.prev_trig = prev_trig;
            }
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

static BOWED_INPUTS: [InputSpec; 3] = [
    InputSpec { name: "freq", rate: Rate::Audio },
    InputSpec { name: "pressure", rate: Rate::Audio },
    InputSpec { name: "position", rate: Rate::Audio },
];
static BOWED_OUTPUTS: [OutputSpec; 1] = [OutputSpec { name: "out", rate: Rate::Audio }];

impl UGen for Bowed {
    fn spec(&self) -> UGenSpec {
        UGenSpec { name: "Bowed", inputs: &BOWED_INPUTS, outputs: &BOWED_OUTPUTS }
    }

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
        inputs: &[&AudioBuffer],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied();
        let pressure_buf = inputs.get(1).copied();
        let position_buf = inputs.get(2).copied();
        let max_len = self.nut_delay.len();
        if max_len == 0 {
            return;
        }

        for ch in 0..output.num_channels() {
            let mut nut_write = self.nut_write;
            let mut bridge_write = self.bridge_write;
            let mut nut_filter = self.nut_filter;
            let mut bridge_filter = self.bridge_filter;
            let out = output.channel_mut(ch).samples_mut();

            for i in 0..out.len() {
                let freq = freq_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(220.0)
                    .clamp(MIN_FREQ, self.sample_rate * 0.45);
                let pressure = pressure_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                let position = position_buf
                    .map(|b| b.channel(ch % b.num_channels()).samples()[i])
                    .unwrap_or(0.13)
                    .clamp(0.02, 0.98);

                // Compute delay lengths from frequency and bow position
                let total_delay = self.sample_rate / freq;
                let nut_len = ((total_delay * position) as usize).clamp(1, max_len - 1);
                let bridge_len = ((total_delay * (1.0 - position)) as usize).clamp(1, max_len - 1);

                // Read returning waves from delay lines
                let nut_read = (nut_write + max_len - nut_len) % max_len;
                let nut_out = self.nut_delay[nut_read];
                let bridge_read = (bridge_write + max_len - bridge_len) % max_len;
                let bridge_out = self.bridge_delay[bridge_read];

                // Reflections with inversion and lowpass filtering (loss model)
                // Nut: rigid termination with loss
                nut_filter = 0.95 * nut_filter + 0.05 * (-nut_out);
                // Bridge: less lossy
                bridge_filter = 0.97 * bridge_filter + 0.03 * (-bridge_out);

                // String velocity at bow point
                let v_string = nut_filter + bridge_filter;
                // Bow velocity (normalized by pressure)
                let v_bow = 0.3 * pressure;
                let delta_v = v_bow - v_string;

                // Bow-string interaction
                let force = bow_table(delta_v, pressure) * pressure * 0.3;

                // Inject force into both delay lines
                self.nut_delay[nut_write] = nut_filter + force;
                self.bridge_delay[bridge_write] = bridge_filter + force;

                // Output from bridge side (pickup position)
                out[i] = (bridge_filter + force).clamp(-1.0, 1.0);

                nut_write = (nut_write + 1) % max_len;
                bridge_write = (bridge_write + 1) % max_len;
            }

            if ch == 0 {
                self.nut_write = nut_write;
                self.bridge_write = bridge_write;
                self.nut_filter = nut_filter;
                self.bridge_filter = bridge_filter;
            }
        }
    }
}
