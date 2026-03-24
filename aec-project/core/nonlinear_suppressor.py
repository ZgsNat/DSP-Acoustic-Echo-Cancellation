"""
Nonlinear (Residual) Echo Suppressor

After NLMS cancels the linear echo component, a residual remains due to:
    1. Non-linear distortion in loudspeaker/microphone hardware
    2. RIR estimation error (filter hasn't fully converged)
    3. Rapid RIR changes (person or phone moves)

This module applies spectral subtraction on the residual signal,
using the echo estimate spectrum as a proxy for residual echo power.

Spectral Subtraction formula (power domain):
    |E_clean(f)|² = max(|E(f)|² - alpha * |Y(f)|², beta * |E(f)|²)

Where:
    E(f)  : Residual echo spectrum (output of NLMS)
    Y(f)  : Echo estimate spectrum (NLMS output before subtraction)
    alpha : Over-subtraction factor (1.0–2.0). Higher = more aggressive.
    beta  : Spectral floor (0.01–0.1). Prevents over-subtraction artifacts
            (so-called "musical noise").

The reconstructed signal uses the original residual phase (phase of E(f))
since phase estimation is difficult and not worth the complexity here.
"""

import numpy as np


class NonlinearSuppressor:
    """
    Spectral subtraction-based residual echo suppressor.

    Operates frame-by-frame using overlap-add to avoid block artifacts.
    """

    def __init__(
        self,
        frame_size: int = 1024,
        overlap: int = 512,
        alpha: float = 1.5,
        beta: float = 0.05,
        smooth_alpha: float = 0.85,
    ) -> None:
        """
        Args:
            frame_size   : FFT frame size. Must be power of 2.
            overlap      : Kept for API compatibility (unused — no windowing).
            alpha        : Over-subtraction factor. Higher removes more residual
                           echo but risks distorting speech.
            beta         : Spectral floor. Prevents complete zeroing of bins
                           which causes "musical noise" artifacts.
            smooth_alpha : Temporal smoothing for echo power estimate (0–1).
                           Higher = more smoothing → less musical noise.
        """
        self.frame_size = frame_size
        self.overlap = overlap
        self.alpha = alpha
        self.beta = beta
        self.smooth_alpha = smooth_alpha

        # Temporally-smoothed echo power spectrum (across frames)
        # Initialized lazily on first process() call
        self._echo_power_smooth: np.ndarray | None = None

    def process(
        self,
        residual_frame: np.ndarray,
        echo_estimate_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Apply spectral subtraction to one frame.

        Args:
            residual_frame      : Output of NLMS filter e(n) — still has some echo
            echo_estimate_frame : Echo estimate y(n) from NLMS — proxy for residual power

        Returns:
            Suppressed output frame, same length as input.
        """
        assert len(residual_frame) == len(echo_estimate_frame)

        orig_len = len(residual_frame)
        e = residual_frame.astype(np.float64)
        y = echo_estimate_frame.astype(np.float64)

        n = self.frame_size

        # Pad to frame_size if necessary
        if len(e) < n:
            e = np.pad(e, (0, n - len(e)))
            y = np.pad(y, (0, n - len(y)))

        # FFT — no windowing to avoid per-frame amplitude dropouts at edges.
        # A Hann-windowed frame without proper overlap-add zeros out every frame
        # boundary, creating rhythmic chopping artifacts ("soạt xoẹt").
        E = np.fft.rfft(e[:n])
        Y = np.fft.rfft(y[:n])

        # Power spectra
        E_power = np.abs(E) ** 2
        Y_power = np.abs(Y) ** 2

        # Temporal smoothing of echo power estimate across frames.
        # Raw Y_power fluctuates wildly frame-to-frame, causing "musical noise"
        # (random tonal artifacts). Smoothing stabilises the suppression mask.
        if self._echo_power_smooth is None:
            self._echo_power_smooth = Y_power.copy()
        else:
            self._echo_power_smooth = (
                self.smooth_alpha * self._echo_power_smooth
                + (1.0 - self.smooth_alpha) * Y_power
            )

        # Spectral subtraction using smoothed echo estimate
        E_clean_power = np.maximum(
            E_power - self.alpha * self._echo_power_smooth,
            self.beta * E_power,    # Floor: never suppress below beta * original
        )

        # Reconstruct: keep original phase, use suppressed magnitude
        E_mag = np.sqrt(E_clean_power)
        E_phase = np.angle(E)
        E_clean = E_mag * np.exp(1j * E_phase)

        # Back to time domain
        e_clean = np.fft.irfft(E_clean)

        return e_clean[:orig_len].astype(np.float32)

    def reset(self) -> None:
        """Reset smoothing state. Call at start of new session."""
        self._echo_power_smooth = None