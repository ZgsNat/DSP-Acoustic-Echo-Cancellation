"""
NLMS Adaptive Filter — Core of AEC

NLMS (Normalized Least Mean Squares) estimates the Room Impulse Response (RIR)
between the loudspeaker and microphone, then subtracts the echo estimate from
the microphone signal.

Update rule:
    w(n+1) = w(n) + (mu / (||x(n)||^2 + eps)) * e(n) * x(n)

Where:
    w(n)  : filter weight vector, length L (estimates RIR)
    x(n)  : reference signal vector (last L samples from speaker)
    d(n)  : microphone input (near-end speech + echo)
    y(n)  : echo estimate = w(n)^T * x(n)
    e(n)  : error / residual = d(n) - y(n)  ← this is our output
    mu    : step size (convergence speed vs stability tradeoff)
    eps   : small constant to prevent division by zero
    L     : filter length (determines how long an echo we can cancel)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class NLMSConfig:
    """Configuration for NLMS filter."""

    # Number of filter taps.
    # At 16kHz, L=512 → 32ms echo window. Cover most room echoes.
    # Longer L = captures longer reverb, but: slower convergence + more RAM.
    filter_length: int = 512

    # Step size mu in (0, 2) for stability.
    # mu=0.1: slow but stable; mu=0.5: fast but noisy steady state.
    # Typical choice for AEC: 0.1–0.3
    mu: float = 0.1

    # Regularization to avoid division by zero during silence.
    eps: float = 1e-6


class NLMSFilter:
    """
    Single-channel NLMS adaptive filter for Acoustic Echo Cancellation.

    Usage:
        filt = NLMSFilter(NLMSConfig())
        for frame in audio_frames:
            e = filt.process(mic_frame, ref_frame, update=True)
            # e is the echo-cancelled output (residual)
    """

    def __init__(self, config: NLMSConfig = NLMSConfig()) -> None:
        self.cfg = config
        L = config.filter_length

        # Filter weights — start at zero (assumes no echo initially)
        self.w: np.ndarray = np.zeros(L, dtype=np.float64)

        # Circular buffer holding last L samples of reference signal
        self._ref_buf: np.ndarray = np.zeros(L, dtype=np.float64)
        self._buf_idx: int = 0  # Next write position in circular buffer

    def process(
        self,
        mic_frame: np.ndarray,
        ref_frame: np.ndarray,
        update: bool = True,
    ) -> np.ndarray:
        """
        Process one frame of audio.

        Args:
            mic_frame : Microphone input d(n), shape (N,), float32/64
            ref_frame : Reference (speaker) signal x(n), shape (N,), float32/64
                        Must be delay-aligned before calling this method.
            update    : If False, freeze weights (used by DTD during double-talk)

        Returns:
            e_frame : Residual after echo subtraction, shape (N,), float64
                      This is what gets sent over the network.
        """
        assert mic_frame.shape == ref_frame.shape, "Frame size mismatch"

        N = len(mic_frame)
        L = self.cfg.filter_length
        mu = self.cfg.mu
        eps = self.cfg.eps

        # Work in float64 for numerical stability
        mic = mic_frame.astype(np.float64)
        ref = ref_frame.astype(np.float64)

        e_frame = np.empty(N, dtype=np.float64)

        for n in range(N):
            # --- Step 1: Write new reference sample into circular buffer ---
            self._ref_buf[self._buf_idx] = ref[n]
            self._buf_idx = (self._buf_idx + 1) % L

            # --- Step 2: Extract x_vec = [x(n), x(n-1), ..., x(n-L+1)] ---
            # Circular buffer read: most recent sample first
            # idx_start is the position of x(n) (just written)
            idx = (self._buf_idx - 1) % L
            # Unroll circular buffer into contiguous vector
            x_vec = np.concatenate([
                self._ref_buf[idx::-1],          # x(n) down to buffer start
                self._ref_buf[L-1:idx:-1]         # wrap around to buffer end
            ])
            # x_vec[0] = x(n), x_vec[1] = x(n-1), ..., x_vec[L-1] = x(n-L+1)

            # --- Step 3: Echo estimate ---
            y_n = np.dot(self.w, x_vec)          # y(n) = w^T * x_vec

            # --- Step 4: Residual (this is our output for this sample) ---
            e_n = mic[n] - y_n                   # e(n) = d(n) - y(n)
            e_frame[n] = e_n

            # --- Step 5: NLMS weight update (only if not double-talk) ---
            if update:
                norm = np.dot(x_vec, x_vec) + eps
                # Normalized step: adapts to signal level changes
                self.w += (mu / norm) * e_n * x_vec

        return e_frame

    def reset(self) -> None:
        """Reset filter state. Call when starting a new call session."""
        self.w[:] = 0.0
        self._ref_buf[:] = 0.0
        self._buf_idx = 0

    @property
    def weight_norm(self) -> float:
        """L2 norm of current weight vector. Useful for monitoring convergence."""
        return float(np.linalg.norm(self.w))