//! Interpolation shapes for parameter glides.
//!
//! A glide moves a parameter from a start value to a target value over some
//! duration. The *shape* controls how the blend fraction toward the target
//! progresses over that duration; the *space* controls which domain the
//! interpolation happens in (see [`GlideSpace`]).

/// The interpolation shape applied to a parameter glide.
///
/// `x` is the normalized phase through the glide, `x` in `[0, 1]`, and each
/// shape maps it to the fraction of the distance toward the target that
/// should be covered at that point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum GlideShape {
    /// Stay at the start value for the whole segment, then jump to the
    /// target right at the end — a step function.
    Hold,
    /// Constant-rate ramp from start to target.
    #[default]
    Linear,
    /// Raised-cosine ease-in-out.
    Sine,
    /// Exponential ease with a signed tension coefficient.
    ///
    /// A positive coefficient eases slow-to-fast, a negative one eases
    /// fast-to-slow, and zero is numerically equivalent to [`GlideShape::Linear`].
    /// The coefficient is sanitized on evaluation (see [`glide_fraction`]):
    /// non-finite values and out-of-range magnitudes can't produce a
    /// non-finite result.
    Exponential(f32),
}

/// Which domain a parameter glide interpolates in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlideSpace {
    /// Interpolate directly in the parameter's own units — a constant-rate
    /// ramp in whatever units the parameter is expressed in (e.g. Hz for a
    /// frequency parameter).
    #[default]
    Raw,
    /// Interpolate in pitch space: convert start and target to a 12-TET note
    /// number, ramp there, then convert back to the parameter's units.
    ///
    /// Produces an equal-ratio sweep (equal glide time gives equal perceived
    /// rate) instead of an equal-difference sweep, which is what a listener
    /// expects from a pitch slide.
    Pitch,
}

/// Upper bound (in either direction) on the exponential shape's tension
/// coefficient.
///
/// `f32::exp` overflows to `inf` once its argument exceeds `ln(f32::MAX)`
/// (`~88.7228`), and the reference formula below then divides `inf` by
/// `inf`, producing `NaN` — the largest exponential the formula evaluates is
/// `k.exp()` itself (at `x = 1`), so that's the only overflow to guard.
/// This bound is deliberately set close to that ceiling rather than to some
/// arbitrary smaller value: a lower bound would silently distort tensions
/// that are already perfectly finite and numerically meaningful (e.g.
/// `k = 50` produces its own valid, merely very steep, curve — clamping it
/// down to a smaller bound would force it through a different, wrong
/// curve). `80.0` leaves `e^8.7228 ≈ 6100`x of headroom below the true
/// overflow point (`80.0.exp() ≈ 5.5e34` vs. `f32::MAX ≈ 3.4e38`), which is
/// enough margin that the clamp itself can never be the thing that
/// overflows, while only reshaping tensions that were already about to
/// overflow unclamped. Tension is clamped to this range before the formula
/// runs, so no finite input can drive it to overflow.
const TENSION_BOUND: f32 = 80.0;

/// Below this magnitude, the exponential shape's tension is treated as
/// exactly zero (linear).
///
/// This isn't just a shortcut for the already-special-cased `k == 0.0`: an
/// `f32`'s precision near `1.0` means `k.exp() - 1.0` can itself round to
/// exactly `0.0` for sufficiently small nonzero `k` (below roughly `6e-8`),
/// which would divide zero by zero. This threshold sits comfortably above
/// that precision floor, and a tension this close to zero is in any case
/// perceptually identical to a true linear ramp.
const TENSION_ZERO_EPSILON: f32 = 1e-4;

/// Evaluate a glide shape at normalized phase `x` (clamped to `[0, 1]`),
/// returning the fraction of the distance to the target covered so far.
///
/// This is the single place the four shape formulas are implemented, and
/// the single place the exponential shape's tension is sanitized against
/// non-finite or extreme values (see [`TENSION_BOUND`] and
/// [`TENSION_ZERO_EPSILON`]); callers should not reimplement these curves,
/// or the sanitization, elsewhere. The result is always finite and in
/// `[0, 1]` regardless of the tension passed in.
pub fn glide_fraction(shape: GlideShape, x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    match shape {
        GlideShape::Hold => 0.0,
        GlideShape::Linear => x,
        GlideShape::Sine => (1.0 - (core::f32::consts::PI * x).cos()) / 2.0,
        GlideShape::Exponential(k) => {
            // NaN carries no direction to preserve, so it falls back to the
            // neutral (linear) case rather than an arbitrary sign choice.
            // `f32::clamp` would propagate a NaN `self` through unchanged,
            // so NaN must be handled before clamping; `clamp` on its own
            // already maps +-inf to +-TENSION_BOUND correctly (inf compares
            // greater than max, -inf less than min).
            let k = if k.is_nan() {
                0.0
            } else {
                k.clamp(-TENSION_BOUND, TENSION_BOUND)
            };
            if k.abs() < TENSION_ZERO_EPSILON {
                x
            } else {
                ((k * x).exp() - 1.0) / (k.exp() - 1.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_is_always_zero_until_clamped_end() {
        assert_eq!(glide_fraction(GlideShape::Hold, 0.0), 0.0);
        assert_eq!(glide_fraction(GlideShape::Hold, 0.5), 0.0);
        assert_eq!(glide_fraction(GlideShape::Hold, 0.999), 0.0);
    }

    #[test]
    fn linear_is_identity() {
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            assert_eq!(glide_fraction(GlideShape::Linear, x), x);
        }
    }

    #[test]
    fn sine_matches_raised_cosine_formula() {
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let expected = (1.0 - (core::f32::consts::PI * x).cos()) / 2.0;
            assert_eq!(glide_fraction(GlideShape::Sine, x), expected);
        }
        assert_eq!(glide_fraction(GlideShape::Sine, 0.0), 0.0);
        assert!((glide_fraction(GlideShape::Sine, 0.5) - 0.5).abs() < 1e-6);
        assert!((glide_fraction(GlideShape::Sine, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn exponential_zero_tension_matches_linear() {
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            assert_eq!(
                glide_fraction(GlideShape::Exponential(0.0), x),
                glide_fraction(GlideShape::Linear, x)
            );
        }
    }

    #[test]
    fn exponential_matches_reference_formula() {
        for &k in &[3.0f32, -3.0, 0.5, -0.5] {
            for i in 0..=10 {
                let x = i as f32 / 10.0;
                let expected = ((k * x).exp() - 1.0) / (k.exp() - 1.0);
                assert_eq!(glide_fraction(GlideShape::Exponential(k), x), expected);
            }
        }
    }

    #[test]
    fn exponential_steep_but_finite_tension_is_not_clamped() {
        // A tension well under the bound — even one as steep as 50 — is
        // already perfectly finite and numerically meaningful on its own,
        // and must be evaluated with its own formula rather than silently
        // reshaped to match some smaller clamped tension. This is what
        // distinguishes "sanitize hostile input" from "narrow the usable
        // range": only tensions that would actually overflow get clamped.
        for &k in &[50.0f32, -50.0, 70.0, -70.0] {
            assert!(
                k.abs() < TENSION_BOUND,
                "test setup: {k} should be under the clamp bound"
            );
            for i in 0..=10 {
                let x = i as f32 / 10.0;
                let expected = ((k * x).exp() - 1.0) / (k.exp() - 1.0);
                assert_eq!(
                    glide_fraction(GlideShape::Exponential(k), x),
                    expected,
                    "k={k}, x={x}: unclamped tension must match its own formula exactly"
                );
            }
        }
    }

    #[test]
    fn positive_tension_eases_slow_to_fast() {
        // Slow-to-fast: covers less than half the distance at the midpoint.
        let mid = glide_fraction(GlideShape::Exponential(4.0), 0.5);
        assert!(mid < 0.5, "expected slow-to-fast midpoint < 0.5, got {mid}");
    }

    #[test]
    fn negative_tension_eases_fast_to_slow() {
        // Fast-to-slow: covers more than half the distance at the midpoint.
        let mid = glide_fraction(GlideShape::Exponential(-4.0), 0.5);
        assert!(mid > 0.5, "expected fast-to-slow midpoint > 0.5, got {mid}");
    }

    fn sample_xs() -> impl Iterator<Item = f32> {
        (0..=10).map(|i| i as f32 / 10.0)
    }

    #[test]
    fn exponential_hostile_tension_is_always_finite_and_bounded() {
        let hostile_tensions = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1000.0,
            -1000.0,
            f32::MAX,
            f32::MIN,
            200.0,
            -200.0,
            89.0,
            -89.0,
            1e-10,
            -1e-10,
        ];
        for &k in &hostile_tensions {
            for x in sample_xs() {
                let frac = glide_fraction(GlideShape::Exponential(k), x);
                assert!(
                    frac.is_finite(),
                    "k={k}, x={x}: expected a finite result, got {frac}"
                );
                assert!(
                    (0.0..=1.0).contains(&frac),
                    "k={k}, x={x}: expected a result in [0, 1], got {frac}"
                );
            }
        }
    }

    #[test]
    fn exponential_nan_tension_falls_back_to_linear() {
        // NaN carries no direction to preserve, so it degrades to the
        // neutral (linear) case rather than an arbitrary sign choice.
        for x in sample_xs() {
            assert_eq!(glide_fraction(GlideShape::Exponential(f32::NAN), x), x);
        }
    }

    #[test]
    fn exponential_tiny_nonzero_tension_falls_back_to_linear() {
        // Below TENSION_ZERO_EPSILON, k.exp() - 1.0 can itself round to
        // exactly 0.0 in f32, which would otherwise divide zero by zero.
        for x in sample_xs() {
            assert_eq!(glide_fraction(GlideShape::Exponential(1e-10), x), x);
            assert_eq!(glide_fraction(GlideShape::Exponential(-1e-10), x), x);
        }
    }

    #[test]
    fn exponential_tension_is_clamped_to_the_documented_bound() {
        // Beyond the bound, more extreme magnitudes must not change the
        // result further — they're all clamped to the same curve.
        for x in sample_xs() {
            let at_bound = glide_fraction(GlideShape::Exponential(TENSION_BOUND), x);
            let past_bound = glide_fraction(GlideShape::Exponential(1000.0), x);
            assert_eq!(at_bound, past_bound, "x={x}: clamping should be exact");

            let at_neg_bound = glide_fraction(GlideShape::Exponential(-TENSION_BOUND), x);
            let past_neg_bound = glide_fraction(GlideShape::Exponential(-1000.0), x);
            assert_eq!(
                at_neg_bound, past_neg_bound,
                "x={x}: clamping should be exact"
            );
        }
    }

    #[test]
    fn exponential_zero_tension_special_case_is_still_exact() {
        // The fix must not have disturbed the pre-existing exact k=0 case.
        for x in sample_xs() {
            assert_eq!(glide_fraction(GlideShape::Exponential(0.0), x), x);
        }
    }
}
