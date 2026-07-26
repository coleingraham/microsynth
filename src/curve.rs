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

/// Evaluate a glide shape at normalized phase `x` (clamped to `[0, 1]`),
/// returning the fraction of the distance to the target covered so far.
///
/// This is the single place the four shape formulas are implemented;
/// callers should not reimplement these curves elsewhere.
pub fn glide_fraction(shape: GlideShape, x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    match shape {
        GlideShape::Hold => 0.0,
        GlideShape::Linear => x,
        GlideShape::Sine => (1.0 - (core::f32::consts::PI * x).cos()) / 2.0,
        GlideShape::Exponential(k) => {
            if k == 0.0 {
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
}
