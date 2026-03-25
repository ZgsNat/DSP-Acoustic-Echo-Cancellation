# ==========================================
# FILE: aec_pipeline.py
# ==========================================
"""
AEC Pipeline — Main Entry Point

Composes all DSP blocks into a single callable interface:
    1. Delay Estimator (GCC-PHAT) — align reference to mic
    2. Double-Talk Detector (Geigel) — protect near-end speech
    3. NLMS Adaptive Filter — cancel linear echo
    4. Nonlinear Suppressor — clean residual echo

Usage in desktop app:
    pipeline = AECPipeline()
    # Feed frames in real-time loop:
    clean_frame = pipeline.process(mic_frame, ref_frame)

The pipeline is stateful — it must NOT be reset between frames.
Call reset() only when starting a new call session.
"""

import numpy as np
from dataclasses import dataclass

from .nlms_filter import NLMSFilter, NLMSConfig
# --- SỬA LỖI 1: Import DelayLine thay cho apply_delay ---
from .delay_estimator import DelayEstimator, DelayLine
from .double_talk_detector import GeigelhDTD
from .nonlinear_suppressor import NonlinearSuppressor


@dataclass
class AECConfig:
    """Aggregate config for the full AEC pipeline."""

    sample_rate: int = 16000
    frame_size: int = 1024

    # NLMS
    # filter_length=768: covers delay (up to ~40ms @ 16kHz = 640 samples) + RIR tail
    # mu=0.3: faster convergence; validated to reach ERLE ~28dB on synthetic echo
    filter_length: int = 768
    mu: float = 0.3
    eps: float = 1e-6

    # Delay estimation
    max_delay_ms: float = 150.0

    # Double-talk detector
    dtd_threshold: float = 0.8
    dtd_hangover_ms: float = 100.0

    # Nonlinear suppressor
    nls_alpha: float = 1.5    # Over-subtraction factor
    nls_beta: float = 0.05    # Spectral floor


class AECPipeline:
    """
    Full Acoustic Echo Cancellation pipeline.

    Thread-safety: NOT thread-safe. Call from a single audio thread.
    """

    def __init__(self, config: AECConfig = AECConfig()) -> None:
        self.cfg = config

        # Block 1: Delay estimator
        self._delay_est = DelayEstimator(
            sample_rate=config.sample_rate,
            max_delay_ms=config.max_delay_ms,
        )
        # --- SỬA LỖI 1: Khởi tạo DelayLine Buffer liên tục ---
        # Chứa tối đa 3 giây lịch sử tín hiệu ở tần số 16kHz (48000 samples)
        self._delay_line = DelayLine(max_delay_samples=48000)

        # Block 2: Double-talk detector
        self._dtd = GeigelhDTD(
            sample_rate=config.sample_rate,
            threshold=config.dtd_threshold,
            hangover_ms=config.dtd_hangover_ms,
        )

        # Block 3: NLMS adaptive filter
        self._nlms = NLMSFilter(NLMSConfig(
            filter_length=config.filter_length,
            mu=config.mu,
            eps=config.eps,
        ))

        # Block 4: Nonlinear (residual) echo suppressor
        self._nls = NonlinearSuppressor(
            frame_size=config.frame_size,
            alpha=config.nls_alpha,
            beta=config.nls_beta,
        )

        # Metrics accumulators (reset each call to get_metrics())
        self._mic_power_acc: list[float] = []
        self._out_power_acc: list[float] = []
        self._dt_count: int = 0
        self._frame_count: int = 0

    def process(
        self,
        mic_frame: np.ndarray,
        ref_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Process one frame through the full AEC pipeline.

        Args:
            mic_frame : Raw microphone input d(n), shape (frame_size,)
                        Contains: near-end speech + echo from far-end speaker
            ref_frame : Reference signal x(n), shape (frame_size,)
                        The audio currently being played through the speaker.
                        Must be captured BEFORE D/A conversion (from playback buffer).

        Returns:
            clean_frame : Echo-cancelled signal, shape (frame_size,), float32
                          Ready to encode and send over network.
        """
        mic = mic_frame.astype(np.float32)
        ref = ref_frame.astype(np.float32)

        # Accumulate mic power for ERLE computation
        self._mic_power_acc.append(float(np.mean(mic ** 2)))
        self._frame_count += 1

        # --- Block 1: Delay estimation & alignment ---
        # Update delay estimate every few frames (amortized cost)
        delay = self._delay_est.update(ref, mic)
        
        # --- SỬA LỖI 1: Align reference sử dụng DelayLine liên tục ---
        # Không còn bị mất dữ liệu ở đuôi hay chèn số 0 ở đầu mỗi frame riêng lẻ
        ref_aligned = self._delay_line.process(ref, delay)

        # --- Block 2: Double-talk detection ---
        # Uses aligned reference for fair comparison
        is_dt = self._dtd.detect(mic, ref_aligned)
        if is_dt:
            self._dt_count += 1

        # --- Block 3: NLMS adaptive filter ---
        # update=False during double-talk → freeze weights
        residual = self._nlms.process(mic, ref_aligned, update=not is_dt)
        # residual = e(n) = d(n) - w^T * x_aligned(n)
        # At this point, linear echo is mostly removed

        # --- Block 4: Nonlinear suppressor ---
        # Echo estimate y(n) = mic(n) - residual(n) = w^T * x_aligned(n)
        echo_estimate = mic.astype(np.float64) - residual
        clean = self._nls.process(residual, echo_estimate)

        # Accumulate output power for ERLE
        self._out_power_acc.append(float(np.mean(clean.astype(np.float32) ** 2)))

        return clean.astype(np.float32)

    def get_metrics(self) -> dict:
        """
        Compute and reset accumulated performance metrics.

        Returns dict with:
            erle_db       : Echo Return Loss Enhancement in dB
                            10 * log10(mic_power / output_power)
                            Higher = better. Target: ≥ 15 dB.
            double_talk_ratio : Fraction of frames with double-talk detected
            frame_count   : Total frames processed since last reset
            delay_ms      : Current estimated delay in ms
            filter_norm   : L2 norm of NLMS weights (proxy for convergence)
        """
        mic_power = np.mean(self._mic_power_acc) if self._mic_power_acc else 1e-10
        out_power = np.mean(self._out_power_acc) if self._out_power_acc else 1e-10

        erle_db = 10.0 * np.log10(mic_power / (out_power + 1e-10))

        metrics = {
            "erle_db": float(erle_db),
            "double_talk_ratio": self._dt_count / max(self._frame_count, 1),
            "frame_count": self._frame_count,
            "delay_ms": self._delay_est.current_delay_ms,
            "filter_norm": self._nlms.weight_norm,
        }

        # Reset accumulators
        self._mic_power_acc.clear()
        self._out_power_acc.clear()
        self._dt_count = 0
        self._frame_count = 0

        return metrics

    def reset(self) -> None:
        """Reset all internal state. Call at start of new session."""
        self._delay_line.reset() # --- SỬA LỖI 1: Reset DelayLine ---
        self._nlms.reset()
        self._dtd.reset()
        self._nls.reset()
        self._mic_power_acc.clear()
        self._out_power_acc.clear()
        self._dt_count = 0
        self._frame_count = 0