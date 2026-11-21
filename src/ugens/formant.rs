//! Formant synthesis: the phase-aligned formant (PAF) generator.
//!
//! - [`Paf`]: one formant, a band of harmonics of `freq` centred on `center`
//!   with bandwidth `bw`, both continuous controls. A vowel is four or five
//!   `paf` nodes on the same `freq`, summed.

use crate::buffer::{AudioBuffer, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use crate::ugens::turns::sin_cos_turns;

/// Largest envelope ratio `r` between neighbouring harmonics (see [`Paf`]).
/// It bounds the output's peak at `amp · (1 + r) / (1 − r)` = 199 · `amp`
/// and is reached once `bw` exceeds about 138 · `freq`.
pub const PAF_MAX_RATIO: f64 = 0.99;

/// `freq` is clamped to this range (Hz) before use.
const F0_MIN: f64 = 0.01;

/// `center` and `freq` are clamped below this fraction of the sample rate.
const NYQUIST_GUARD: f64 = 0.49;

/// Phase-aligned formant generator (Puckette 1995): one formant of a
/// harmonic tone, with the formant's centre and bandwidth as continuous,
/// audio-rate controls and no filter in the signal path.
///
/// With `θ` the phase of the fundamental (`freq`), `c = center / freq`,
/// `k = ⌊c⌋`, `α = c − k`:
///
/// ```text
/// out = amp · [ P_r(θ) · ((1 − α)·cos kθ + α·cos (k+1)θ) − DC ]
/// P_r(θ) = (1 − r²) / (1 − 2r·cos θ + r²) = 1 + 2 Σ_{n≥1} rⁿ cos nθ
/// r = 2^(−2·freq / bw)
/// ```
///
/// `P_r` is the *modulator*: a pulse with one peak per period whose
/// harmonics fall geometrically, by `r` per harmonic. It is Puckette's
/// waveshaped half-frequency sinusoid, `1 / (1 + b² sin²(θ/2))` up to a
/// constant, written in closed form (the Cauchy waveshaper
/// `f(x) = 1/(1 + x²)`; `b² = 4r / (1 − r)²`). Multiplying it by a cosine at
/// harmonic `k` moves that comb of decaying harmonics so it is centred on
/// harmonic `k`: line `k ± n` has amplitude `rⁿ`. The two carriers are
/// harmonics of the *same* phase `θ`, so every component of the output is an
/// exact harmonic of `freq` wherever `center` lies: a centre between two
/// harmonics is a crossfade of the two neighbouring combs, not a component
/// at `center` (which would beat against the harmonics). As `center` crosses
/// harmonic `k + 1`, `α` reaches 1 and the output is `P_r · cos (k+1)θ`,
/// which is exactly where it restarts with the new `k` at `α = 0`: the
/// sweep is continuous in value and in spectrum.
///
/// **Levels.** The line at the centre harmonic has amplitude `amp` (when
/// `center` is a harmonic), so `amp` is the formant's spectral peak, and the
/// spectral envelope is `amp · r^(|f − center| / freq)`: −6 dB at
/// `center ± bw/2`. `bw` is therefore the **−6 dB** bandwidth. (A two-pole
/// resonance of −3 dB bandwidth `B` is −6 dB wide at `√3 · B`; matching the
/// two at −6 dB, this envelope falls faster than the resonance's 6 dB/octave
/// skirts further out.) With `center` halfway between harmonics both lines
/// are `amp · (1 + r)/2`, and the formant's power dips by
/// `(1 + r)² / (2(1 + r²))`: under 1 dB once `bw` ≥ 1.25 · `freq`, but a
/// formant much narrower than the harmonic spacing loses up to 3 dB between
/// harmonics, as a real narrow resonance does (it is why sopranos tune F1
/// to a harmonic). The peak of the output is at most
/// `amp · (1 + r) / (1 − r)`, at most 199 · `amp` ([`PAF_MAX_RATIO`]); its
/// RMS is about `amp · sqrt((1 + r²) / (2(1 − r²)))`.
///
/// **Fold-over at 0 Hz.** Lines below harmonic 0 (`n > k`) fold onto
/// positive harmonics with the same sign (cos is even), so a formant whose
/// envelope reaches 0 Hz (`center` within a few `bw` of 0) is fuller on its
/// low side than the formula says, as in any PAF or FOF. The 0 Hz line
/// itself, `(1 − α)·r^k + α·r^(k+1)`, is subtracted: the output has no DC.
///
/// **Aliasing.** The lines do not stop at Nyquist; the first one past it is
/// about `20·log10(r) · (fs/2 − center) / freq` dB re the peak, i.e.
/// `−12 dB · (fs/2 − center) / bw`. For speech-like formants (`bw` up to a
/// few hundred Hz, `center` ≤ 8 kHz at 44.1/48 kHz) that is below −150 dB;
/// it reaches −40 dB only for bandwidths of about a third of the distance
/// from `center` to Nyquist. Folded lines are harmonics of `freq` folded at
/// Nyquist, so they are inharmonic unless `fs / freq` is an integer.
///
/// Inputs (all optional, all audio-rate):
/// - `freq`: the fundamental in Hz (default 220), clamped to
///   0.01 Hz … 0.49 · fs. At `freq ≤ 0` the output is 0 and the phase
///   holds, rather than a full-bandwidth tone at `center`.
/// - `center`: formant centre in Hz (default 800), clamped to 0 … 0.49 · fs;
///   audio-rate, so a vowel can morph or a formant sweep.
/// - `bw`: −6 dB bandwidth in Hz (default 100); `bw ≤ 0` leaves the bare
///   carriers (a sine at a harmonic `center`, two harmonics between).
/// - `amp`: the formant's peak level (default 1). Meant for an envelope or a
///   per-formant level; a non-finite `amp` reads as 0.
/// - `phase_reset`: on a rising edge (previous sample `<= 0`, this one
///   `> 0`) the fundamental's phase restarts at 0.
///
/// Other non-finite inputs read as their defaults.
pub struct Paf {
    phase: f64,
    prev_reset: f32,
    sample_rate: f32,
}

impl Default for Paf {
    fn default() -> Self {
        Self::new()
    }
}

impl Paf {
    pub fn new() -> Self {
        Paf {
            phase: 0.0,
            prev_reset: 0.0,
            sample_rate: 44100.0,
        }
    }
}

/// The geometric ratio between neighbouring lines for a −6 dB bandwidth `bw`
/// at fundamental `f0` (both Hz): `2^(−2·f0/bw)`, capped at [`PAF_MAX_RATIO`].
#[inline]
pub fn paf_ratio(f0: f64, bw: f64) -> f64 {
    if bw > 0.0 && f0 > 0.0 {
        (-2.0 * f0 / bw).exp2().min(PAF_MAX_RATIO)
    } else {
        0.0
    }
}

/// `r^k` for a whole-number exponent, by binary powering: one squaring per
/// bit of `k` and one multiply per set bit, at most 64 of each, against the
/// general logarithm-and-exponential path of `powf` (which the 0 Hz line
/// paid on every sample once vibrato moves `r`).
///
/// `pow_whole(r, 0)` is 1 (`0⁰` included), `pow_whole(0, k)` is 0 for
/// `k > 0`. For `0 ≤ r ≤ 1`, the only range the node uses, nothing
/// overflows, and every partial product and squared base is at least the
/// result, so while the result is a normal number nothing before it has
/// lost precision; once it would fall below the smallest subnormal it is 0,
/// as `powf` gives.
///
/// Accuracy: a rounding made while forming `r^(2^j)` is raised to the power
/// of every later squaring, so the result carries at most `k − 1` roundings
/// and its relative error is at most `γ(k−1) = (k−1)·u / (1 − (k−1)·u)`
/// (`u` = 2⁻⁵³) while it is a normal number; measured, it reaches about
/// 0.9 of that (`powf` stays within about 1 ulp). For the node the absolute
/// error is what counts: `r^k` ≤ 1 enters the output as the subtracted
/// 0 Hz line, and `(k−1)·rᵏ` is largest at `k ≈ 1 / ln(1/r)`, so for
/// `r` ≤ [`PAF_MAX_RATIO`] the error is below `36.6·u` ≈ 4.1e-15 for every
/// `k` (measured worst about 1.7e-15). Tests pin both against values
/// computed at high precision.
///
/// In the node `k` is at most 0.49·fs / 0.01 Hz (9.4 million at 192 kHz,
/// 24 bits), so the loop runs at most 24 times.
///
/// Written out rather than `f64::powi`: `powi`'s rounding is unspecified
/// and may differ between targets and compiler versions (on wasm32 it calls
/// compiler-builtins' `__powidf2`, natively the platform's), and its
/// exponent is an `i32`. This loop is the same arithmetic on every target,
/// so native and wasm renders agree bit for bit.
#[inline]
fn pow_whole(r: f64, k: u64) -> f64 {
    let (mut acc, mut base, mut e) = (1.0f64, r, k);
    while e != 0 {
        if e & 1 != 0 {
            acc *= base;
        }
        e >>= 1;
        if e != 0 {
            base *= base;
        }
    }
    acc
}

#[inline]
fn finite_or(x: f32, default: f32) -> f64 {
    if x.is_finite() {
        x as f64
    } else {
        default as f64
    }
}

impl UGen for Paf {
    ugen_spec!(
        "Paf",
        category = Oscillator,
        inputs = [],
        optional_inputs = ["freq", "center", "bw", "amp", "phase_reset"],
        outputs = ["out"]
    );

    fn init(&mut self, context: &ProcessContext) {
        self.sample_rate = context.sample_rate;
    }

    fn reset(&mut self) {
        self.phase = 0.0;
        self.prev_reset = 0.0;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let freq_buf = inputs.first().copied().flatten();
        let center_buf = inputs.get(1).copied().flatten();
        let bw_buf = inputs.get(2).copied().flatten();
        let amp_buf = inputs.get(3).copied().flatten();
        let reset_buf = inputs.get(4).copied().flatten();
        let sr = self.sample_rate as f64;
        let inv_sr = 1.0 / sr;
        let top = NYQUIST_GUARD * sr;

        // Every channel starts from the same block-start state (see FmOsc).
        let (phase0, reset0) = (self.phase, self.prev_reset);

        for ch in 0..output.num_channels() {
            let (mut phase, mut prev_reset) = (phase0, reset0);
            // the envelope ratio only changes when freq or bw do
            let (mut last_f0, mut last_bw, mut r) = (f64::NAN, f64::NAN, 0.0f64);
            // and r^k only when r or k do
            let (mut last_k, mut last_r, mut rk) = (f64::NAN, f64::NAN, 0.0f64);
            let out = output.channel_mut(ch).samples_mut();

            for (i, out_sample) in out.iter_mut().enumerate() {
                let f0 = finite_or(read_input(freq_buf, ch, i, 220.0), 220.0);
                let center = finite_or(read_input(center_buf, ch, i, 800.0), 800.0).clamp(0.0, top);
                let bw = finite_or(read_input(bw_buf, ch, i, 100.0), 100.0);
                // a VCA port: a non-finite level silences rather than opens
                let amp = finite_or(read_input(amp_buf, ch, i, 1.0), 0.0);
                let reset = read_input(reset_buf, ch, i, 0.0);

                if reset > 0.0 && prev_reset <= 0.0 {
                    phase = 0.0;
                }
                prev_reset = reset;

                if f0 <= 0.0 {
                    *out_sample = 0.0;
                    continue;
                }
                let f0 = f0.clamp(F0_MIN, top);

                if f0 != last_f0 || bw != last_bw {
                    r = paf_ratio(f0, bw);
                    last_f0 = f0;
                    last_bw = bw;
                }

                let c = center / f0;
                let k = c.floor();
                let alpha = c - k;
                // θ = 2π·phase; both sin_cos take the phase in turns
                let (s1, c1) = sin_cos_turns(phase);
                // cos kθ and cos (k+1)θ from one sin_cos of kθ (k·phase is
                // reduced to one cycle first, for precision at high k)
                let kp = k * phase;
                let (sk, ck) = sin_cos_turns(kp - kp.floor());
                let ck1 = ck * c1 - sk * s1;
                let carrier = (1.0 - alpha) * ck + alpha * ck1;
                let modulator = (1.0 - r * r) / (1.0 - 2.0 * r * c1 + r * r);
                // the 0 Hz line: the mean of P_r · cos nθ is rⁿ
                if k != last_k || r != last_r {
                    // k is a whole number in 0 … 0.49·fs / F0_MIN
                    rk = pow_whole(r, k as u64);
                    last_k = k;
                    last_r = r;
                }
                let dc = (1.0 - alpha) * rk + alpha * rk * r;
                let y = amp * (modulator * carrier - dc);
                *out_sample = if y.is_finite() { y as f32 } else { 0.0 };

                phase += f0 * inv_sr;
                phase -= phase.floor();
            }

            if ch == 0 {
                self.phase = phase;
                self.prev_reset = prev_reset;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const U: f64 = f64::EPSILON / 2.0;

    /// `γ(n) = n·u / (1 − n·u)`: the relative error bound for `n` roundings.
    fn gamma(n: u64) -> f64 {
        let nu = n as f64 * U;
        nu / (1.0 - nu)
    }

    /// `(hi, lo)` with `hi + lo` the product of two double-doubles to about
    /// 2⁻¹⁰⁴ relative (the low parts' product is dropped).
    fn dd_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        let p = a.0 * b.0;
        let e = a.0.mul_add(b.0, -p) + (a.0 * b.1 + a.1 * b.0);
        let h = p + e;
        (h, e - (h - p))
    }

    /// `r^k` in double-double by the same binary powering: relative error
    /// about `k·2⁻¹⁰⁴` (under 1e-24 for every `k` used here), far below
    /// `pow_whole`'s, while the result is above about 1e-290 (below that
    /// the low part underflows).
    fn dd_pow(r: f64, k: u64) -> (f64, f64) {
        let (mut acc, mut base, mut e) = ((1.0, 0.0), (r, 0.0), k);
        while e != 0 {
            if e & 1 != 0 {
                acc = dd_mul(acc, base);
            }
            e >>= 1;
            if e != 0 {
                base = dd_mul(base, base);
            }
        }
        acc
    }

    /// xorshift64, for a reproducible sweep
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// `(r bits, k, r^k)`: every `r` is an exact f64 (given by its bit
    /// pattern: 0.99, the cap, and `paf_ratio` of the noted `f0`/`bw`), and
    /// `r^k` was computed once at 60 significant digits (mpmath) and rounded
    /// correctly to f64; each literal is the shortest decimal that reads
    /// back as that f64. 9 408 000 is the largest `k` the node can reach
    /// (0.49 · 192 kHz / 0.01 Hz).
    #[rustfmt::skip]
    const TRUE_VALUES: [(u64, u64, f64); 27] = [
        (0x3fef_ae14_7ae1_47ae,         1, 0.99),                    // r cap
        (0x3fef_ae14_7ae1_47ae,         2, 0.9801),
        (0x3fef_ae14_7ae1_47ae,         9, 0.9135172474836408),
        (0x3fef_ae14_7ae1_47ae,        37, 0.6894490858690777),
        (0x3fef_ae14_7ae1_47ae,       100, 0.3660323412732292),
        (0x3fef_ae14_7ae1_47ae,       255, 0.07708584232989273),
        (0x3fef_ae14_7ae1_47ae,      1000, 4.317124741065786e-5),
        (0x3fef_ae14_7ae1_47ae,      4097, 1.3103649446228297e-18),
        (0x3fef_ae14_7ae1_47ae,     65000, 1.9393171619009236e-284),
        (0x3fef_ae14_7ae1_47ae,     70000, 2.908183650085128e-306),
        (0x3fef_ae14_7ae1_47ae,     73000, 2.33994e-319),           // subnormal
        (0x3fef_ae14_7ae1_47ae,     74200, 0.0),                     // below half the least subnormal
        (0x3fef_ae14_7ae1_47ae,   9408000, 0.0),
        (0x3fa8_4060_03b2_ae5b,         3, 1.0626837930860883e-4),   // f0 220, bw 100
        (0x3fa8_4060_03b2_ae5b,        10, 5.6843418860807895e-14),
        (0x3fa8_4060_03b2_ae5b,        50, 5.934729841099812e-67),
        (0x3fa8_4060_03b2_ae5b,       230, 2.2784756311112652e-305),
        (0x3feb_795e_38df_3c5e,         9, 0.2534898699475072),      // f0 110, bw 1000
        (0x3feb_795e_38df_3c5e,        17, 0.074842419038683),
        (0x3feb_795e_38df_3c5e,        96, 4.3877966433757393e-7),
        (0x3feb_795e_38df_3c5e,       400, 3.2311742677852177e-27),
        (0x3fea_2774_d0d1_3e74,        34, 1.0501904279912916e-3),   // f0 87.31, bw 600
        (0x3fea_2774_d0d1_3e74,       255, 4.565732012293232e-23),
        (0x3feb_adb9_e6fa_ee69,         5, 0.48414127495419923),     // f0 523.25, bw 5000
        (0x3feb_adb9_e6fa_ee69,        16, 0.09815462101082906),
        (0x3fef_628a_a657_7f3c,        30, 0.5586435690361106),      // f0 98, bw 7000
        (0x3fef_628a_a657_7f3c,      2047, 5.573971817143062e-18),
    ];

    #[test]
    fn within_gamma_k_minus_1_of_the_true_values() {
        for (bits, k, t) in TRUE_VALUES {
            let r = f64::from_bits(bits);
            let got = pow_whole(r, k);
            if t == 0.0 {
                assert_eq!(got, 0.0, "r {r}, k {k}");
            } else if t < f64::MIN_POSITIVE {
                // subnormal: absolute, a few of the least subnormal's steps
                // per rounding at most
                assert!((got - t).abs() <= 1e-300, "r {r}, k {k}: {got:e} vs {t:e}");
            } else {
                // the table value is itself within half an ulp of the truth
                let bound = gamma(k - 1) * t + 0.5 * f64::EPSILON * t;
                let d = (got - t).abs();
                assert!(d <= bound, "r {r}, k {k}: off {d:e}, bound {bound:e}");
            }
        }
    }

    #[test]
    fn within_gamma_k_minus_1_of_a_double_double_reference_and_4e_15_absolute() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut worst_rel = 0.0f64; // as a fraction of γ(k−1)
        let mut worst_abs = 0.0f64;
        let mut check = |r: f64, k: u64| {
            let got = pow_whole(r, k);
            let (th, tl) = dd_pow(r, k);
            let d = ((got - th) - tl).abs();
            worst_abs = worst_abs.max(d);
            if th > 1e-290 && k > 1 {
                worst_rel = worst_rel.max(d / th / gamma(k - 1));
            } else if k <= 1 {
                assert_eq!(got, if k == 0 { 1.0 } else { r });
            }
        };
        // k in bands up to the node's largest; r uniform in (0, 0.99] and
        // crowded towards the cap, where r^k stays large longest
        let bands: [(u64, u64); 5] = [
            (1, 16),
            (17, 256),
            (257, 4096),
            (4097, 65536),
            (65537, 9_408_000),
        ];
        for (lo, hi) in bands {
            for i in 0..100_000u64 {
                let r = if i % 2 == 0 {
                    PAF_MAX_RATIO * (1.0 - rng.unit())
                } else {
                    (1.0 - 10f64.powf(-(0.004_37 + 4.0 * rng.unit()))).min(PAF_MAX_RATIO)
                };
                check(r, lo + rng.next() % (hi - lo + 1));
            }
        }
        // every k through the underflow at the cap and a few other ratios
        for r in [PAF_MAX_RATIO, 0.98, 0.95, 0.9, 0.5, 0.1, 1e-3] {
            for k in 0..=80_000u64 {
                check(r, k);
            }
        }
        // measured: about 0.86 of γ(k−1) and 1.7e-15
        assert!(worst_rel <= 1.0, "worst {worst_rel} of γ(k−1)");
        assert!(
            worst_abs <= 36.6 * U * (1.0 + 1e-9),
            "worst abs {worst_abs:e}"
        );
    }

    #[test]
    fn edge_cases() {
        // 0⁰ = 1, as the node needs for a centre below the first harmonic
        assert_eq!(pow_whole(0.0, 0), 1.0);
        assert_eq!(pow_whole(PAF_MAX_RATIO, 0), 1.0);
        assert_eq!(pow_whole(0.0, 1), 0.0);
        assert_eq!(pow_whole(0.0, 9_408_000), 0.0);
        assert_eq!(pow_whole(0.37, 1), 0.37);
        assert_eq!(pow_whole(1.0, u64::MAX), 1.0);
        // the whole exponent range terminates (64 passes) and underflows to 0
        assert_eq!(pow_whole(PAF_MAX_RATIO, u64::MAX), 0.0);
        assert_eq!(pow_whole(0.5, 1074), f64::from_bits(1));
        assert_eq!(pow_whole(0.5, 1075), 0.0);
        // powers of two are exact all the way down
        for k in 0..=1074u64 {
            let exact = if k <= 1022 {
                f64::from_bits((1023 - k) << 52)
            } else {
                f64::from_bits(1 << (1074 - k))
            };
            assert_eq!(pow_whole(0.5, k), exact, "k {k}");
        }
    }

    #[test]
    fn the_node_clamps_keep_k_within_24_bits_at_192k() {
        let k_max = (NYQUIST_GUARD * 192_000.0 / F0_MIN).floor();
        assert_eq!(k_max, 9_408_000.0);
        assert!((k_max as u64) < 1 << 24);
    }
}
