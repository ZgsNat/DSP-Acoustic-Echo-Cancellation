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

# ==========================================
# FILE 4: double_talk_detector.py
# ==========================================
"""
Double-Talk Detector (DTD) — Geigel Algorithm
(Đã sửa logic bộ đếm Hangover để trừ số khung (frames) thay vì số mẫu (samples))
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
        self.sr = sample_rate
        self.threshold = threshold
        self.hangover_ms = hangover_ms

        win_samples = int(window_ms * sample_rate / 1000)

        self._mic_buf: deque = deque(maxlen=win_samples)
        self._ref_buf: deque = deque(maxlen=win_samples)

        # --- SỬA LỖI 4: Quản lý số Frames duy trì Hangover ---
        self._hangover_frames_left: int = 0
        self._in_double_talk: bool = False

    def detect(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> bool:
        mic_peak = float(np.max(np.abs(mic_frame)))
        ref_peak = float(np.max(np.abs(ref_frame)))

        self._mic_buf.append(mic_peak)
        self._ref_buf.append(ref_peak)

        # Max over sliding window
        max_mic = max(self._mic_buf) if self._mic_buf else 0.0
        max_ref = max(self._ref_buf) if self._ref_buf else 0.0

        # Geigel criterion: near-end louder than threshold * far-end
        raw_dt = max_mic > self.threshold * (max_ref + 1e-10)

        # --- SỬA LỖI 4: Tính chính xác số lượng Frames cần duy trì Hangover ---
        frame_ms = (len(mic_frame) / self.sr) * 1000.0
        hangover_frames_max = int(np.ceil(self.hangover_ms / frame_ms))

        if raw_dt:
            # Nếu có Double-Talk thực tế, nạp lại bộ đếm số lượng frame
            self._hangover_frames_left = hangover_frames_max
            self._in_double_talk = True
        elif self._hangover_frames_left > 0:
            # Nếu bộ đếm vẫn còn > 0, giảm 1 frame cho mỗi lần đi qua
            self._hangover_frames_left -= 1
            self._in_double_talk = True
        else:
            self._in_double_talk = False

        return self._in_double_talk

    def reset(self) -> None:
        self._mic_buf.clear()
        self._ref_buf.clear()
        self._hangover_frames_left = 0
        self._in_double_talk = False

    @property
    def is_double_talk(self) -> bool:
        """Current double-talk state (result of last detect() call)."""
        return self._in_double_talk