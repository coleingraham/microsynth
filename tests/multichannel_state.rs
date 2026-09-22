//! Registry-wide regression coverage for the multichannel state-writeback
//! bug fixed across `src/ugens/*.rs`.
//!
//! **The bug.** Stateful UGens process a multichannel `AudioBuffer` with a
//! per-channel loop that snapshots persisted state, processes that
//! channel's block, and (for channel 0 only) writes the result back —
//! `filters::OnePole::process` is the canonical example, see its doc
//! comment. When the snapshot was re-read from `self` *inside* the channel
//! loop rather than taken once *before* it, the write-back for channel 0
//! happened before channel 1's snapshot was taken — so channel 1 started
//! from channel 0's END-of-block state, a full block ahead in time. On a
//! stereo bus with identical L/R input (the normal case: a mono voice
//! expanded to a stereo group/master bus), this produced a discontinuity
//! in the right channel at every block boundary that the left channel
//! never had.
//!
//! A second, related shape of the same underlying defect: a UGen whose
//! state includes a genuinely shared, singular *resource* (an RNG stream,
//! or delay-line memory indexed directly rather than through a per-channel
//! local copy) mutated that resource once per output channel instead of
//! once per sample — `Pluck`/`Bowed` (shared delay-line memory + RNG) and
//! `WhiteNoise`/`PinkNoise`/`VinylCrackle` (shared RNG stream, no
//! per-channel snapshot attempted at all) — producing channels that
//! differ for the whole render, not just at block boundaries. See each
//! UGen's own `process()` comment for its specific fix.
//!
//! **This test.** For every bare (non-table-bound) kind in the standard
//! `UGenRegistry` (`ugens::register_builtins`), construct it, feed 8
//! consecutive 128-sample blocks of a 2-channel buffer whose channels are
//! byte-identical (driving every *required* input port with the same
//! signal; optional ports are left unconnected so each UGen's own default
//! applies), and assert channel 1 == channel 0 for every sample of every
//! block. Three categories of kind are explicitly exempted (see the
//! constants below), each with the specific reason named per kind.

use microsynth::dsl::compiler::UGenRegistry;
use microsynth::node::UGen;
use microsynth::{AudioBuffer, ProcessContext, register_builtins};

const SAMPLE_RATE: f32 = 44100.0;
const BLOCK_SIZE: usize = 128;
const NUM_BLOCKS: usize = 8;
/// Exact equality is expected for every fixed kind (the fix makes channel 1
/// literally re-derive or copy channel 0's computation), but leave a tiny
/// float-noise allowance for kinds whose channel-1 arithmetic path isn't
/// bit-identical (e.g. `PartialsNoise`, which still `.clone()`s Vec/struct
/// state rather than reusing a single computed buffer).
const TOLERANCE: f32 = 1e-6;

/// Kinds that are stereo **by design** — their own doc comments say the two
/// channels are computed differently on purpose — exempted from the
/// channel-identity check, one reason per kind so a future reader doesn't
/// have to go re-derive why.
const STEREO_BY_DESIGN: &[(&str, &str)] = &[
    (
        "pan2",
        "equal-power stereo panner: left = cos(theta)*in, right = sin(theta)*in \
         — different trig functions of the same input by construction",
    ),
    (
        "gverb",
        "stereo reverb: combs_l/combs_r and allpasses_l/allpasses_r use different \
         delay-tap lengths (STEREO_SPREAD) for left/right decorrelation, per its \
         own doc comment",
    ),
    (
        "chorus",
        "stereo chorus: left/right delay taps use LFO phases offset by 90 degrees \
         for stereo decorrelation, per its own doc comment",
    ),
    (
        "stereoWidth",
        "Haas-effect stereo widener: left = dry, right = delayed+blended, per its \
         own doc comment",
    ),
    (
        "pingPongDelay",
        "ping-pong delay: alternating left/right taps that bounce the signal \
         between channels by construction, per its own doc comment",
    ),
];

/// Kinds with a fixed output channel count below 2 — there is no "channel
/// 1" to compare. Forcing a 2-channel buffer on these would just compare
/// real output (channel 0) against an untouched leftover buffer (channel
/// 1), which isn't a multichannel-identity property at all.
const FIXED_SUB_STEREO_OUTPUT: &[(&str, &str)] = &[(
    "mix",
    "Mix::output_channels() always returns 1 (mono down-mix); process() only \
     ever writes channel 0",
)];

/// Kinds exempted for a **different**, separately-documented bug (see the
/// module doc on `ugens::spectral`): each holds one shared `StftProcessor`
/// (or, for `convolution`, one shared ring-buffer set) driven once per
/// channel per sample by a single processor instance, so channel 1's frame
/// timing can diverge from channel 0's even for byte-identical input —
/// not the read-back-inside-loop state-snapshot bug this ticket fixes, and
/// not fixable by the same mechanism. Left broken pending a follow-up
/// ticket; asserting on them here would just pin the wrong bug.
const SHARED_PROCESSOR_BUG: &[(&str, &str)] = &[
    ("spectralFreeze", "see ugens::spectral module doc"),
    ("pitchShift", "see ugens::spectral module doc"),
    ("spectralFilter", "see ugens::spectral module doc"),
    ("spectralGate", "see ugens::spectral module doc"),
    ("spectralBlur", "see ugens::spectral module doc"),
    ("convolution", "see ugens::spectral module doc"),
];

fn exemption_reason(name: &str) -> Option<&'static str> {
    for &(n, reason) in STEREO_BY_DESIGN
        .iter()
        .chain(FIXED_SUB_STEREO_OUTPUT)
        .chain(SHARED_PROCESSOR_BUG)
    {
        if n == name {
            return Some(reason);
        }
    }
    None
}

/// A decaying 47 Hz sine plus a periodic click: exercises phase
/// accumulators, filter memory, envelope/trigger-edge detection, and RNG
/// consumption all in one signal. Computed from a *global* sample index
/// (not reset per block) so the driver is continuous across the whole
/// 8-block render — block-boundary state bugs only show up if the signal
/// (and the state driven by it) actually crosses a boundary coherently.
fn driver_sample(global_i: usize) -> f32 {
    let t = global_i as f32 / SAMPLE_RATE;
    let decay = (-1.5 * t).exp();
    let sine = decay * (2.0 * core::f32::consts::PI * 47.0 * t).sin();
    let click = if global_i.is_multiple_of(500) {
        0.8
    } else {
        0.0
    };
    sine + click
}

/// Build one 2-channel driver block, byte-identical on both channels.
fn make_driver_block(block_idx: usize) -> AudioBuffer {
    let mut buf = AudioBuffer::new(2, BLOCK_SIZE);
    for ch in 0..2 {
        let samples = buf.channel_mut(ch).samples_mut();
        for (i, s) in samples.iter_mut().enumerate() {
            *s = driver_sample(block_idx * BLOCK_SIZE + i);
        }
    }
    buf
}

/// Drive one UGen kind for `NUM_BLOCKS` blocks and assert channel 1 matches
/// channel 0 throughout. Returns the first few mismatches found (empty on
/// success) so the caller can build one readable failure message per kind.
fn find_channel_divergences(
    name: &str,
    factory: fn() -> Box<dyn UGen>,
    input_required: &[bool],
) -> Vec<String> {
    let mut ugen = factory();
    let ctx = ProcessContext::new(SAMPLE_RATE, BLOCK_SIZE);
    ugen.init(&ctx);

    let mut divergences = Vec::new();

    for block_idx in 0..NUM_BLOCKS {
        let driver = make_driver_block(block_idx);
        let inputs: Vec<Option<&AudioBuffer>> = input_required
            .iter()
            .map(|&required| if required { Some(&driver) } else { None })
            .collect();

        let mut output = AudioBuffer::new(2, BLOCK_SIZE);
        ugen.process(&ctx, &inputs, &mut output);

        let ch0 = output.channel(0).samples();
        let ch1 = output.channel(1).samples();
        for i in 0..BLOCK_SIZE {
            let diff = (ch0[i] - ch1[i]).abs();
            if diff > TOLERANCE {
                divergences.push(format!(
                    "{name}: block {block_idx} sample {i}: ch0={} ch1={} |diff|={}",
                    ch0[i], ch1[i], diff
                ));
                if divergences.len() >= 5 {
                    return divergences;
                }
            }
        }
    }

    divergences
}

#[test]
fn every_registered_ugen_produces_identical_channels_from_identical_input() {
    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let mut failures: Vec<String> = Vec::new();
    let mut tested = 0usize;
    let mut exempted = 0usize;

    for (name, entry) in reg.iter() {
        if let Some(reason) = exemption_reason(name) {
            exempted += 1;
            let _ = reason; // documented in the exemption tables above
            continue;
        }

        tested += 1;
        let divergences = find_channel_divergences(name, entry.factory, &entry.required);
        if !divergences.is_empty() {
            failures.push(format!(
                "{name}: {} divergent sample(s) found, first ones:\n  {}",
                divergences.len(),
                divergences.join("\n  ")
            ));
        }
    }

    // Sanity: the exemption lists should exactly match real registrations
    // (a typo'd name in an exemption table would silently exempt nothing
    // and this loop would never notice) — checked by cross-referencing
    // against `tested + exempted == reg.iter().count()` implicitly (every
    // name is either tested or exempted, never neither) and by each
    // exemption table's names being asserted registered below.
    for &(n, _) in STEREO_BY_DESIGN
        .iter()
        .chain(FIXED_SUB_STEREO_OUTPUT)
        .chain(SHARED_PROCESSOR_BUG)
    {
        assert!(
            reg.entry(n).is_some(),
            "exemption table names {n:?}, which is not a registered kind — stale exemption?"
        );
    }

    assert!(
        tested >= 40,
        "expected to exercise most of the ~60 registered kinds, only tested {tested} \
         (exempted {exempted}) — did registration or exemption bookkeeping break?"
    );

    assert!(
        failures.is_empty(),
        "{} kind(s) produced divergent channels from identical input:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// `PlayBuf` isn't in the DSL registry (no `register_spec` call for it — see
/// `ugens::mod`'s doc list vs. its `register_builtins`), so it's outside the
/// loop above; drive it directly with a **mono** sample so
/// `sample.read_interpolated(ch, position)` reads the same source channel
/// for every output channel (a stereo sample would legitimately differ per
/// channel by design — see `PlayBuf::process`'s comment), isolating the
/// transport-state fix (position/playing/done/prev_trig) under test.
#[test]
fn playbuf_produces_identical_channels_from_identical_input_with_a_mono_sample() {
    use microsynth::Sample;
    use microsynth::ugens::PlayBuf;
    use std::sync::Arc;

    let mono_samples: Vec<f32> = (0..SAMPLE_RATE as usize).map(driver_sample).collect();
    let sample = Arc::new(Sample::from_mono(&mono_samples, SAMPLE_RATE));

    let mut ugen = PlayBuf::new().with_sample(sample).with_loop(true);
    let ctx = ProcessContext::new(SAMPLE_RATE, BLOCK_SIZE);
    ugen.init(&ctx);

    let mut divergences = Vec::new();
    for block_idx in 0..NUM_BLOCKS {
        // rate/trig left unconnected (optional) so PlayBuf's own defaults
        // (rate=1.0, no re-trigger) apply; PlayBuf has no required inputs.
        let mut output = AudioBuffer::new(2, BLOCK_SIZE);
        ugen.process(&ctx, &[None, None], &mut output);

        let ch0 = output.channel(0).samples();
        let ch1 = output.channel(1).samples();
        for i in 0..BLOCK_SIZE {
            let diff = (ch0[i] - ch1[i]).abs();
            if diff > TOLERANCE {
                divergences.push(format!(
                    "block {block_idx} sample {i}: ch0={} ch1={} |diff|={}",
                    ch0[i], ch1[i], diff
                ));
            }
        }
    }

    assert!(
        divergences.is_empty(),
        "PlayBuf: {} divergent sample(s), first ones:\n{}",
        divergences.len(),
        divergences
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// `partialsNoise` (`ugens::partials::PartialsNoise`) is table-bound
/// (`UGenRegistry::register_table_bound`, MOT-634/636) — its factory needs a
/// resolved `CoeffTable` at construction time, which the bare-kind loop
/// above can't supply, so it's out of this test's reach. Its
/// read-back-inside-loop fix (the same class as every other site in this
/// file) was reviewed directly in `ugens::partials::PartialsNoise::process`
/// instead; skipped here per this ticket's own scope note ("kinds that need
/// a table you may skip").
#[test]
fn partials_noise_is_intentionally_skipped_table_bound() {}

// --- Channel independence: differing input channels ---
//
// The identity test above feeds both channels the same signal, which hides a
// second defect: one state shared across channels. A filter whose memory is
// shared makes channel 1 restart every block from channel 0's memory, and a
// delay line replayed per channel lets channel 1's writes leak into channel
// 0's reads. Both are silent while the channels are identical and audible
// the moment anything upstream makes them differ. This test drives the two
// channels with different signals and asserts that each output channel
// depends only on its own input channel.

/// Blocks per render. A `delay` at its default 0.1 s and a `feedbackDelay`
/// at its default 0.25 s first read back what they wrote at blocks 34 and
/// 86, so shorter renders would pass them without exercising the line.
const INDEPENDENCE_BLOCKS: usize = 96;

/// Three distinct drivers, each continuous across block boundaries. `0` is
/// the identity test's driver; `1` and `2` differ from it and from each
/// other in frequency, click period, and shape.
fn independence_driver(kind: u8, global_i: usize) -> f32 {
    let t = global_i as f32 / SAMPLE_RATE;
    let tau = core::f32::consts::TAU;
    match kind {
        0 => driver_sample(global_i),
        1 => {
            0.6 * (tau * 311.0 * t).sin()
                + if global_i.is_multiple_of(733) {
                    -0.7
                } else {
                    0.0
                }
        }
        _ => {
            let ramp = ((global_i % 257) as f32 / 257.0) * 2.0 - 1.0;
            0.5 * ramp + 0.3 * (tau * 1733.0 * t).sin()
        }
    }
}

/// Settings that make otherwise-neutral kinds do something: a shelf or
/// peaking EQ at 0 dB, a bitcrusher that does not downsample, or a delay
/// longer than the render all pass audio through untouched and would hide
/// the defect. Matched by port name; `None` leaves the port unconnected.
fn non_neutral_value(port: &str) -> Option<f32> {
    let p = port.to_ascii_lowercase();
    if p.contains("gain") {
        Some(6.0)
    } else if p == "bits" {
        Some(8.0)
    } else if p == "downsample" {
        Some(4.0)
    } else if p == "time" {
        Some(0.01)
    } else if p == "feedback" {
        Some(0.5)
    } else {
        None
    }
}

/// Render one kind for `INDEPENDENCE_BLOCKS` blocks with driver `left` on
/// input channel 0 and `right` on channel 1 of every required port. With
/// `non_neutral`, optional ports named by `non_neutral_value` get a mono
/// constant; otherwise every optional port is left unconnected.
fn render_stereo(
    factory: fn() -> Box<dyn UGen>,
    required: &[bool],
    left: u8,
    right: u8,
    non_neutral: bool,
) -> [Vec<f32>; 2] {
    let mut ugen = factory();
    let port_names: Vec<&'static str> = ugen.spec().inputs.iter().map(|s| s.name).collect();
    let ctx = ProcessContext::new(SAMPLE_RATE, BLOCK_SIZE);
    ugen.init(&ctx);

    let constants: Vec<Option<AudioBuffer>> = required
        .iter()
        .enumerate()
        .map(|(k, &is_required)| {
            if is_required || !non_neutral {
                return None;
            }
            port_names
                .get(k)
                .and_then(|n| non_neutral_value(n))
                .map(|v| {
                    let mut b = AudioBuffer::new(1, BLOCK_SIZE);
                    b.channel_mut(0).samples_mut().fill(v);
                    b
                })
        })
        .collect();

    let mut out = [Vec::new(), Vec::new()];
    for block_idx in 0..INDEPENDENCE_BLOCKS {
        let mut driver = AudioBuffer::new(2, BLOCK_SIZE);
        for i in 0..BLOCK_SIZE {
            let global_i = block_idx * BLOCK_SIZE + i;
            driver.channel_mut(0).samples_mut()[i] = independence_driver(left, global_i);
            driver.channel_mut(1).samples_mut()[i] = independence_driver(right, global_i);
        }
        let inputs: Vec<Option<&AudioBuffer>> = required
            .iter()
            .enumerate()
            .map(|(k, &is_required)| {
                if is_required {
                    Some(&driver)
                } else {
                    constants[k].as_ref()
                }
            })
            .collect();

        let mut output = AudioBuffer::new(2, BLOCK_SIZE);
        ugen.process(&ctx, &inputs, &mut output);
        out[0].extend_from_slice(output.channel(0).samples());
        out[1].extend_from_slice(output.channel(1).samples());
    }
    out
}

/// Index of the first block where `a` and `b` differ bit for bit.
fn first_divergent_block(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x.to_bits() != y.to_bits())
        .map(|i| i / BLOCK_SIZE)
}

/// Which optional ports `non_neutral_value` connects for a kind, for the
/// failure message.
fn non_neutral_ports(factory: fn() -> Box<dyn UGen>, required: &[bool]) -> Vec<String> {
    factory()
        .spec()
        .inputs
        .iter()
        .enumerate()
        .filter(|(k, _)| !required.get(*k).copied().unwrap_or(false))
        .filter_map(|(_, s)| non_neutral_value(s.name).map(|v| format!("{}={v}", s.name)))
        .collect()
}

#[test]
fn every_registered_processor_keeps_its_channels_independent() {
    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let mut failures: Vec<String> = Vec::new();
    let mut tested = 0usize;

    for (name, entry) in reg.iter() {
        if exemption_reason(name).is_some() || !entry.required.iter().any(|&r| r) {
            continue;
        }
        tested += 1;

        for non_neutral in [false, true] {
            let mode = if non_neutral {
                format!(
                    "non-neutral ({})",
                    non_neutral_ports(entry.factory, &entry.required).join(" ")
                )
            } else {
                "defaults".to_string()
            };
            let base = render_stereo(entry.factory, &entry.required, 0, 1, non_neutral);
            let left_changed = render_stereo(entry.factory, &entry.required, 2, 1, non_neutral);
            let right_changed = render_stereo(entry.factory, &entry.required, 0, 2, non_neutral);

            if let Some(block) = first_divergent_block(&base[1], &left_changed[1]) {
                failures.push(format!(
                    "{name} [{mode}]: right output changed when only the left input changed \
                     (from block {block}): one state shared across channels"
                ));
            }
            if let Some(block) = first_divergent_block(&base[0], &right_changed[0]) {
                failures.push(format!(
                    "{name} [{mode}]: left output changed when only the right input changed \
                     (from block {block}): one delay line replayed per channel"
                ));
            }
        }
    }

    assert!(
        tested >= 20,
        "expected to exercise most processor kinds, only tested {tested}"
    );
    assert!(
        failures.is_empty(),
        "{} case(s) where an output channel depends on the other channel's input:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// The stereo-by-design kinds compute both output channels from input
/// channel 0 alone, so their outputs must not depend on input channel 1 at
/// all. This is the audit of those kinds for the shared-state defect: with
/// one input channel there is nothing for a second channel's state to
/// corrupt, and this assertion pins that reading of their code.
#[test]
fn every_stereo_by_design_kind_reads_only_input_channel_0() {
    let mut reg = UGenRegistry::new();
    register_builtins(&mut reg);

    let mut failures: Vec<String> = Vec::new();
    for &(name, _) in STEREO_BY_DESIGN {
        let entry = reg
            .entry(name)
            .expect("stereo-by-design kind is registered");
        for non_neutral in [false, true] {
            let base = render_stereo(entry.factory, &entry.required, 0, 1, non_neutral);
            let right_changed = render_stereo(entry.factory, &entry.required, 0, 2, non_neutral);
            for ch in 0..2 {
                if let Some(block) = first_divergent_block(&base[ch], &right_changed[ch]) {
                    failures.push(format!(
                        "{name}: output channel {ch} changed when only input channel 1 \
                         changed (from block {block})"
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} stereo-by-design case(s) read input channel 1:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
