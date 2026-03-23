"""
Double-Talk Detector (DTD) — Geigel Algorithm

Problem:
    When both near-end user (person at mic) and far-end user (person at speaker)
    speak simultaneously (double-talk), NLMS makes a critical mistake:
    - It sees large error e(n) = d(n) - y(n)  (because d now contains voice too)
    - It interprets this as "echo estimate is wrong" and updates aggressively
    - Result: weights diverge, near-end voice is partially suppressed → distortion

Solution:
    Detect double-talk and FREEZE NLMS weight updates during that period.
    Conservative but safe: weights may be slightly stale, but voice is preserved.

Geigel Algorithm:
    Compare the maximum absolute value of mic signal over a window
    against the maximum absolute value of reference signal over the same window.

    if max(|d(n-k)|, k=0..K-1) > threshold * max(|x(n-k)|, k=0..K-1):
        double_talk = True

    Intuition: If mic is significantly louder than speaker, near-end person is talking.

Limitations:
    - False positives during loud echo (but that's safe — just slows adaptation)
    - Cannot distinguish double-talk from background noise spike
    - Geigel is simple but works well for its purpose in an AEC pipeline
"""

import numpy as np
from collections import deque


class GeigelhDTD:
    """
    Geigel Double-Talk Detector.

    Returns a boolean flag per frame: True = double-talk detected → freeze NLMS.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_ms: float = 20.0,
        threshold: float = 0.5,
        hangover_ms: float = 100.0,
    ) -> None:
        """
        Args:
            sample_rate  : Audio sample rate in Hz
            window_ms    : Sliding window length for max comparison (ms).
                           Longer window = more robust, but slower response.
            threshold    : Geigel threshold. Lower = more sensitive (more false positives).
                           Typical: 0.5 (mic must be > 50% of speaker peak to trigger)
            hangover_ms  : After double-talk ends, keep flag True for this long.
                           Prevents NLMS from updating on transient tail.
        """
        self.threshold = threshold

        win_samples = int(window_ms * sample_rate / 1000)
        hangover_samples = int(hangover_ms * sample_rate / 1000)

        # Sliding window buffers for max tracking
        self._mic_buf: deque = deque(maxlen=win_samples)
        self._ref_buf: deque = deque(maxlen=win_samples)

        # Hangover counter: stay in DT state for hangover_samples after DT ends
        self._hangover_count: int = 0
        self._hangover_max: int = hangover_samples

        # State
        self._in_double_talk: bool = False

    def detect(
        self, mic_frame: np.ndarray, ref_frame: np.ndarray
    ) -> bool:
        """
        Detect double-talk for this frame.

        Args:
            mic_frame : Microphone signal (near-end + echo)
            ref_frame : Reference signal (far-end / speaker output)

        Returns:
            True if double-talk detected (NLMS should freeze)
            False if single-talk only (NLMS can update)
        """
        # Update sliding window with peak absolute values from this frame
        mic_peak = float(np.max(np.abs(mic_frame)))
        ref_peak = float(np.max(np.abs(ref_frame)))

        self._mic_buf.append(mic_peak)
        self._ref_buf.append(ref_peak)

        # Max over sliding window
        max_mic = max(self._mic_buf) if self._mic_buf else 0.0
        max_ref = max(self._ref_buf) if self._ref_buf else 0.0

        # Geigel criterion: near-end louder than threshold * far-end
        raw_dt = max_mic > self.threshold * (max_ref + 1e-10)

        # Apply hangover: once triggered, stay True for hangover_ms
        if raw_dt:
            self._hangover_count = self._hangover_max
            self._in_double_talk = True
        elif self._hangover_count > 0:
            self._hangover_count -= len(mic_frame)
            self._in_double_talk = True
        else:
            self._in_double_talk = False

        return self._in_double_talk

    def reset(self) -> None:
        """Reset all state. Call at start of new session."""
        self._mic_buf.clear()
        self._ref_buf.clear()
        self._hangover_count = 0
        self._in_double_talk = False

    @property
    def is_double_talk(self) -> bool:
        """Current double-talk state (result of last detect() call)."""
        return self._in_double_talk