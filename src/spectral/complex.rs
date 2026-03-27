//! Minimal complex number type for FFT operations.
//!
//! `no_std` compatible — uses only LLVM intrinsic math (sin, cos, sqrt, atan2).

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A complex number with `f32` components.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    /// Create a new complex number.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Complex zero.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Construct from polar form (magnitude, phase).
    #[inline]
    pub fn from_polar(mag: f32, phase: f32) -> Self {
        Self {
            re: mag * phase.cos(),
            im: mag * phase.sin(),
        }
    }

    /// Magnitude (absolute value).
    #[inline]
    pub fn mag(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Phase angle in radians.
    #[inline]
    pub fn phase(self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Squared magnitude (avoids sqrt).
    #[inline]
    pub fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Scale by a real number.
    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl SubAssign for Complex {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl MulAssign for Complex {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        self.re = re;
        self.im = im;
    }
}

impl Div for Complex {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl Neg for Complex {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn polar_roundtrip() {
        let c = Complex::new(3.0, 4.0);
        let p = Complex::from_polar(c.mag(), c.phase());
        assert!((p.re - c.re).abs() < EPS);
        assert!((p.im - c.im).abs() < EPS);
    }

    #[test]
    fn magnitude_and_phase() {
        let c = Complex::new(1.0, 0.0);
        assert!((c.mag() - 1.0).abs() < EPS);
        assert!(c.phase().abs() < EPS);

        let c = Complex::new(0.0, 1.0);
        assert!((c.mag() - 1.0).abs() < EPS);
        assert!((c.phase() - core::f32::consts::FRAC_PI_2).abs() < EPS);
    }

    #[test]
    fn arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let sum = a + b;
        assert!((sum.re - 4.0).abs() < EPS);
        assert!((sum.im - 6.0).abs() < EPS);

        let diff = a - b;
        assert!((diff.re - (-2.0)).abs() < EPS);
        assert!((diff.im - (-2.0)).abs() < EPS);

        // (1+2i)(3+4i) = 3+4i+6i+8i² = 3+10i-8 = -5+10i
        let prod = a * b;
        assert!((prod.re - (-5.0)).abs() < EPS);
        assert!((prod.im - 10.0).abs() < EPS);
    }

    #[test]
    fn conjugate() {
        let c = Complex::new(3.0, 4.0);
        let cc = c.conj();
        assert!((cc.re - 3.0).abs() < EPS);
        assert!((cc.im - (-4.0)).abs() < EPS);
    }
}
