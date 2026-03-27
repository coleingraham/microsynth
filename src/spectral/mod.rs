//! Spectral processing primitives for frequency-domain audio analysis and synthesis.
//!
//! This module provides:
//! - [`Complex`] — minimal complex number type for FFT operations
//! - [`fft`] / [`ifft`] — in-place radix-2 Cooley-Tukey FFT/IFFT
//! - [`WindowType`] / [`make_window`] — standard analysis/synthesis windows
//! - [`StftProcessor`] — overlap-add STFT framework bridging block-size to FFT-size
//! - [`griffin_lim`] — offline phase reconstruction from magnitude spectrograms (std only)

pub mod complex;
pub mod fft;
pub mod griffin_lim;
pub mod stft;
pub mod window;

pub use complex::Complex;
pub use fft::{fft, ifft};
pub use stft::StftProcessor;
pub use window::{make_window, WindowType};
