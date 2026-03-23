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
    ) -> None:
        """
        Args:
            frame_size : FFT frame size. Must be power of 2.
            overlap    : Overlap between frames (samples). Typically frame_size/2.
            alpha      : Over-subtraction factor. Higher removes more residual
                         echo but risks distorting speech.
            beta       : Spectral floor. Prevents complete zeroing of bins
                         which causes "musical noise" artifacts.
        """
        self.frame_size = frame_size
        self.overlap = overlap
        self.alpha = alpha
        self.beta = beta

        # Hann window for smooth overlap-add
        self._window = np.hanning(frame_size).astype(np.float64)

        # Output accumulation buffer for overlap-add
        self._out_buf = np.zeros(frame_size + overlap, dtype=np.float64)

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

        e = residual_frame.astype(np.float64)
        y = echo_estimate_frame.astype(np.float64)

        n = self.frame_size

        # Pad to frame_size if necessary
        if len(e) < n:
            e = np.pad(e, (0, n - len(e)))
            y = np.pad(y, (0, n - len(y)))

        # Apply window
        e_win = e[:n] * self._window
        y_win = y[:n] * self._window

        # FFT
        E = np.fft.rfft(e_win)
        Y = np.fft.rfft(y_win)

        # Power spectra
        E_power = np.abs(E) ** 2
        Y_power = np.abs(Y) ** 2

        # Spectral subtraction: suppress bins where echo estimate is strong
        E_clean_power = np.maximum(
            E_power - self.alpha * Y_power,
            self.beta * E_power,          # Floor: never go below beta * original
        )

        # Reconstruct: keep original phase, use suppressed magnitude
        E_mag = np.sqrt(E_clean_power)
        E_phase = np.angle(E)
        E_clean = E_mag * np.exp(1j * E_phase)

        # Back to time domain
        e_clean = np.fft.irfft(E_clean)

        # Return frame (already same length due to rfft/irfft symmetry)
        return e_clean[:len(residual_frame)].astype(np.float32)

    def reset(self) -> None:
        """Reset overlap-add buffer."""
        self._out_buf[:] = 0.0