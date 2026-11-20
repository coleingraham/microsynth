//! Sine and cosine of a phase given in turns (fractions of a cycle).
//!
//! An oscillator that keeps its phase in turns can reduce it exactly: the
//! quadrant is `round(4·t)`, and `t − quadrant/4` is computed without
//! rounding. That leaves an argument of at most π/4 in magnitude for two
//! short polynomials, with none of the large-argument reduction a general
//! `sin_cos` of `2π·t` has to do.
//!
//! Accuracy: for `t` in `[0, 1)` both results are within about 1.75e-16
//! (at most about 1.8 ulp) of the true `sin 2πt` and `cos 2πt`; a test pins
//! this against a table of values computed at high precision. The
//! library's `sin_cos(2π·t)` is itself up to about 6.9e-16 from the true
//! value, because the product `2π·t` is rounded before it is taken, so this
//! function and the library differ by up to about 7e-16; the tests that
//! compare with the library therefore bound the difference at 1e-15.
//!
//! The results are exactly 0 or ±1 at 0, 1/4, 1/2 and 3/4 turns. The
//! symmetries `sin(1 − t) = −sin t` and `cos(1 − t) = cos t` hold bit for
//! bit everywhere except at the odd eighths of a turn, which are the
//! round-half ties of the quadrant: at 1/8 the sine comes from the cosine
//! polynomial and at 7/8 from the sine polynomial, so the two agree only to
//! the last bit or two there.
//!
//! This function is used on every target, native included, although the
//! system libm is faster on some (aarch64 among them): the same arithmetic
//! everywhere is what makes wasm and native renders agree bit for bit, and
//! a per-architecture switch to libm would split them.

use core::f64::consts::TAU;

// Taylor coefficients 1/n! with alternating signs. On |x| ≤ π/4 the first
// omitted terms are x¹⁷/17! ≈ 4.6e-17 (sine) and x¹⁸/18! ≈ 2e-18 (cosine).
const S3: f64 = -1.0 / 6.0;
const S5: f64 = 1.0 / 120.0;
const S7: f64 = -1.0 / 5040.0;
const S9: f64 = 1.0 / 362_880.0;
const S11: f64 = -1.0 / 39_916_800.0;
const S13: f64 = 1.0 / 6_227_020_800.0;
const S15: f64 = -1.0 / 1_307_674_368_000.0;

const C4: f64 = 1.0 / 24.0;
const C6: f64 = -1.0 / 720.0;
const C8: f64 = 1.0 / 40_320.0;
const C10: f64 = -1.0 / 3_628_800.0;
const C12: f64 = 1.0 / 479_001_600.0;
const C14: f64 = -1.0 / 87_178_291_200.0;
const C16: f64 = 1.0 / 20_922_789_888_000.0;

/// `(sin 2πt, cos 2πt)` for a finite phase `t` in turns. Any finite `t` is
/// accepted (its fractional part is used); accuracy is best, and the
/// reduction exact, for `t` in `[0, 1)`, where an oscillator keeps it.
#[inline]
pub fn sin_cos_turns(t: f64) -> (f64, f64) {
    let t = t - t.floor();
    // nearest quarter turn; t − q/4 is exact (both are multiples of t's ulp
    // and the difference is at most 1/8 in magnitude)
    let q = (4.0 * t + 0.5).floor();
    let x = TAU * (t - 0.25 * q);
    let z = x * x;
    let s = x + x * z * (S3 + z * (S5 + z * (S7 + z * (S9 + z * (S11 + z * (S13 + z * S15))))));
    // 1 − z/2 rounds by up to half an ulp of 1; carry that rounding error
    // into the small terms (exact by Fast2Sum, since 1 ≥ z/2) so that
    // summing the polynomial adds only about half an ulp and the result
    // stays monotonic ulp by ulp. The total error against the true cosine
    // is larger, up to about 1.6 ulp near |x| = π/4, and comes mostly from
    // rounding the argument x = 2π·(t − q/4), not from this sum.
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    let c = w
        + (((1.0 - w) - hz)
            + z * z * (C4 + z * (C6 + z * (C8 + z * (C10 + z * (C12 + z * (C14 + z * C16)))))));
    match (q as i64) & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(t: f64) -> (f64, f64) {
        let (s, c) = sin_cos_turns(t);
        let (ls, lc) = (TAU * t).sin_cos();
        ((s - ls).abs(), (c - lc).abs())
    }

    #[test]
    fn matches_libm_over_a_dense_sweep() {
        // 2^22 points over [0, 1) with an irrational step, plus the regular
        // grid j / 2^20
        let n = 1 << 22;
        let mut worst = 0.0f64;
        for j in 0..n {
            let t = ((j as f64) * 0.618_033_988_749_894_9 / n as f64 * 7.0).fract();
            let (es, ec) = err(t);
            worst = worst.max(es).max(ec);
        }
        for j in 0..(1 << 20) {
            let (es, ec) = err(j as f64 / (1 << 20) as f64);
            worst = worst.max(es).max(ec);
        }
        // libm's own argument 2π·t carries an error of about 1e-16 at t ≈ 1
        assert!(worst < 1e-15, "worst abs error {worst:e}");
    }

    #[test]
    fn matches_libm_next_to_every_quadrant_and_octant_boundary() {
        let mut worst = 0.0f64;
        for k in 0..=8 {
            let b = k as f64 / 8.0;
            for d in [0.0, 1e-300, 1e-17, 1e-16, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3] {
                for t in [b - d, b + d] {
                    if (0.0..1.0).contains(&t) {
                        let (es, ec) = err(t);
                        worst = worst.max(es).max(ec);
                    }
                }
            }
            // ulp steps either side of the boundary
            let mut lo = b;
            let mut hi = b;
            for _ in 0..64 {
                lo = f64::from_bits(lo.to_bits().saturating_sub(1));
                hi = f64::from_bits(hi.to_bits() + 1);
                for t in [lo, hi] {
                    if (0.0..1.0).contains(&t) {
                        let (es, ec) = err(t);
                        worst = worst.max(es).max(ec);
                    }
                }
            }
        }
        assert!(worst < 1e-15, "worst abs error {worst:e}");
    }

    /// `(phase bits, sin 2πt, cos 2πt)`: every phase is an exact f64 (given
    /// by its bit pattern), and the expected values were computed once at
    /// 50 significant digits (mpmath) and rounded correctly to f64; each
    /// literal is the shortest decimal that reads back as that f64. Values
    /// near √½ are the true neighbours' values, not `FRAC_1_SQRT_2`.
    #[allow(clippy::approx_constant)]
    #[rustfmt::skip]
    const TRUE_VALUES: [(u64, f64, f64); 29] = [
        (0x3fbf_ffff_ffff_ffff, 0.7071067811865475,     0.7071067811865476),   // 1/8 − 1 ulp
        (0x3fc0_0000_0000_0001, 0.7071067811865477,     0.7071067811865474),   // 1/8 + 1 ulp
        (0x3fd7_ffff_ffff_ffff, 0.7071067811865478,     -0.7071067811865472),  // 3/8 − 1 ulp
        (0x3fd8_0000_0000_0001, 0.7071067811865472,     -0.7071067811865478),  // 3/8 + 1 ulp
        (0x3fe3_ffff_ffff_ffff, -0.707106781186547,     -0.707106781186548),   // 5/8 − 1 ulp
        (0x3fe4_0000_0000_0001, -0.707106781186548,     -0.707106781186547),   // 5/8 + 1 ulp
        (0x3feb_ffff_ffff_ffff, -0.707106781186548,     0.707106781186547),    // 7/8 − 1 ulp
        (0x3fec_0000_0000_0001, -0.707106781186547,     0.707106781186548),    // 7/8 + 1 ulp
        (0x3fbf_ffff_fc00_0000, 0.7071067770487903,     0.7071067853243047),   // 1/8 − 2^-30
        (0x3fec_0000_0080_0000, -0.7071067770487903,    0.7071067853243047),   // 7/8 + 2^-30
        (0x3fef_ffff_ffff_ffff, -6.975736996017264e-16, 1.0),                  // 1 − 2^-53
        (0x3fef_ffff_ffff_e000, -5.714523747137342e-12, 1.0),                  // 1 − 2^-40
        (0x3fef_fe00_0000_0000, -0.0015339801862847657, 0.9999988234517019),   // 1 − 2^-12
        (0x3c30_0000_0000_0000, 5.449794528138487e-18,  1.0),                  // 2^-60
        (0x3e10_0000_0000_0000, 5.8516723170686385e-9,  1.0),                  // 2^-30
        (0x3f30_0000_0000_0000, 0.0015339801862847657,  0.9999988234517019),   // 2^-12
        (0x3fb0_0000_0000_0000, 0.3826834323650898,     0.9238795325112867),   // 1/16
        (0x3fc8_0000_0000_0000, 0.9238795325112867,     0.3826834323650898),   // 3/16
        (0x3fd4_0000_0000_0000, 0.9238795325112867,     -0.3826834323650898),  // 5/16
        (0x3fdc_0000_0000_0000, 0.3826834323650898,     -0.9238795325112867),  // 7/16
        (0x3fe2_0000_0000_0000, -0.3826834323650898,    -0.9238795325112867),  // 9/16
        (0x3fee_0000_0000_0000, -0.3826834323650898,    0.9238795325112867),   // 15/16
        (0x3fb8_0000_0000_0000, 0.5555702330196022,     0.8314696123025452),   // 3/32
        (0x3fda_0000_0000_0000, 0.5555702330196022,     -0.8314696123025452),  // 13/32
        (0x3fb9_9999_9999_999a, 0.5877852522924731,     0.8090169943749475),   // ≈ 0.1
        (0x3fd3_3333_3333_3333, 0.9510565162951536,     -0.30901699437494734), // ≈ 0.3
        (0x3fe3_c6ef_372f_e951, -0.6754902942615244,    -0.7373688780783192),  // ≈ 0.618
        (0x3fe6_a090_2de0_0d1b, -0.963891187513328,     -0.26629641123031733), // ≈ 0.7071
        (0x3fef_9add_3c1b_e9b8, -0.07749241997589885,   0.996992941221892),    // ≈ 0.98765
    ];

    #[test]
    fn within_3e_16_of_the_true_values() {
        // the expected values are themselves within half an ulp (≤ 5.6e-17)
        // of the truth, so 3e-16 leaves the function about 2.4e-16
        for (bits, es, ec) in TRUE_VALUES {
            let t = f64::from_bits(bits);
            let (s, c) = sin_cos_turns(t);
            let (ds, dc) = ((s - es).abs(), (c - ec).abs());
            assert!(
                ds <= 3e-16 && dc <= 3e-16,
                "t = {t:e}: sin off {ds:e}, cos off {dc:e}"
            );
        }
    }

    #[test]
    fn exact_at_the_quarter_turns() {
        assert_eq!(sin_cos_turns(0.0), (0.0, 1.0));
        assert_eq!(sin_cos_turns(0.25), (1.0, 0.0));
        assert_eq!(sin_cos_turns(0.5), (0.0, -1.0));
        assert_eq!(sin_cos_turns(0.75), (-1.0, 0.0));
        // and the eighths agree with √½ to the last bit or two
        let h = core::f64::consts::FRAC_1_SQRT_2;
        for (k, (es, ec)) in [(1, (h, h)), (3, (h, -h)), (5, (-h, -h)), (7, (-h, h))] {
            let (s, c) = sin_cos_turns(k as f64 / 8.0);
            assert!((s - es).abs() <= 2e-16 && (c - ec).abs() <= 2e-16, "{k}/8");
        }
    }

    #[test]
    fn whole_turns_and_negative_phases_reduce() {
        for t in [0.1, 0.3, 0.6, 0.9] {
            let (s, c) = sin_cos_turns(t);
            for n in [-3.0, -1.0, 1.0, 5.0] {
                let (s2, c2) = sin_cos_turns(t + n);
                assert!((s - s2).abs() < 1e-14 && (c - c2).abs() < 1e-14);
            }
        }
        assert_eq!(sin_cos_turns(1.0), (0.0, 1.0));
        assert_eq!(sin_cos_turns(-0.25), (-1.0, 0.0));
    }

    #[test]
    fn symmetric_about_half_a_turn_and_quarter_shifted() {
        // 1 − t is exact for t in [1/2, 1), so these hold bit for bit
        // (away from the round-half ties at odd eighths)
        let n = 1 << 16;
        for j in 1..n {
            let t = 0.5 + 0.5 * (j as f64 + 0.37) / n as f64;
            let (s, c) = sin_cos_turns(t);
            let (sm, cm) = sin_cos_turns(1.0 - t);
            assert_eq!(s, -sm, "sin at {t}");
            assert_eq!(c, cm, "cos at {t}");
            // a quarter turn later, sine becomes cosine (t + 1/4 is exact
            // for t in [1/2, 3/4) and t − 3/4 for t in [3/4, 1))
            let u = if t < 0.75 { t + 0.25 } else { t - 0.75 };
            let (su, cu) = sin_cos_turns(u);
            assert!(
                (su - c).abs() < 1e-16 && (cu + s).abs() < 1e-16,
                "shift at {t}"
            );
        }
    }

    #[test]
    fn monotonic_on_each_quarter_and_across_octant_seams() {
        // sine rises on [0, 1/4] and [3/4, 1), falls on [1/4, 3/4]
        let check = |a: f64, b: f64| {
            let (sa, ca) = sin_cos_turns(a);
            let (sb, cb) = sin_cos_turns(b);
            let rising_s = !(0.25..0.75).contains(&a);
            let rising_c = a >= 0.5;
            if rising_s {
                assert!(sb >= sa, "sin falls between {a} and {b}");
            } else {
                assert!(sb <= sa, "sin rises between {a} and {b}");
            }
            if rising_c {
                assert!(cb >= ca, "cos falls between {a} and {b}");
            } else {
                assert!(cb <= ca, "cos rises between {a} and {b}");
            }
        };
        let n = 1 << 20;
        for j in 0..n - 1 {
            check(j as f64 / n as f64, (j + 1) as f64 / n as f64);
        }
        // ulp by ulp through every eighth (where the polynomials hand over)
        for k in 1..8 {
            let mut t = k as f64 / 8.0;
            for _ in 0..2000 {
                t = f64::from_bits(t.to_bits() - 1);
            }
            for _ in 0..4000 {
                let next = f64::from_bits(t.to_bits() + 1);
                // the extrema themselves are flat: skip pairs that straddle
                // a quarter turn
                if (t < 0.25) == (next < 0.25)
                    && (t < 0.5) == (next < 0.5)
                    && (t < 0.75) == (next < 0.75)
                {
                    check(t, next);
                }
                t = next;
            }
        }
    }
}
