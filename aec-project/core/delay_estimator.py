"""
Delay Estimator — GCC-PHAT (Generalized Cross-Correlation with Phase Transform)

Problem:
    The reference signal x(n) (recorded from speaker output buffer) and the
    actual echo in the microphone d(n) are NOT time-aligned. There is an
    unknown delay D caused by:
        - Audio driver buffer latency (~20–50ms)
        - Acoustic travel time from speaker to microphone (1ms per 34cm)

    If NLMS receives unaligned signals, it will never converge — the echo
    window x_vec(n) will never overlap with the echo in d(n).

Solution:
    GCC-PHAT estimates D by finding the lag that maximizes cross-correlation
    between x and d, using phase transform to sharpen the peak.

Algorithm:
    1. X(f) = FFT(x),  D(f) = FFT(d)
    2. GCC(f) = X(f) * conj(D(f)) / |X(f) * conj(D(f))|   ← PHAT weighting
    3. gcc_time = IFFT(GCC(f))
    4. D = argmax(gcc_time)  within search range [0, max_delay_samples]
"""

# ==========================================
# FILE 1: delay_estimator.py
# ==========================================
"""
Delay Estimator — GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
(Đã sửa lỗi đứt gãy tín hiệu bằng DelayLine Ring Buffer)
"""

import numpy as np

class DelayEstimator:
    """
    Estimates the delay between reference (speaker) and microphone signals
    using GCC-PHAT computed over a sliding window of frames.

    The estimated delay is smoothed over time to avoid abrupt jumps.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_delay_ms: float = 150.0,
        smooth_alpha: float = 0.9,
        fft_size: int = 4096,
    ) -> None:
        """
        Args:
            sample_rate   : Audio sample rate in Hz
            max_delay_ms  : Maximum expected delay in milliseconds.
                            Set generously (150ms covers buffer + acoustic delay).
            smooth_alpha  : Exponential smoothing factor for delay estimate.
                            0.9 = heavy smoothing (stable but slow to adapt)
                            0.5 = faster adaptation (less stable)
            fft_size      : FFT block size. Larger = better frequency resolution.
        """
        self.sr = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        self.alpha = smooth_alpha
        self.fft_size = fft_size

        # Smoothed delay estimate (fractional samples, rounded on use)
        self._delay_smooth: float = 0.0

        # Accumulation buffers for GCC-PHAT over multiple frames
        self._ref_acc: list[np.ndarray] = []
        self._mic_acc: list[np.ndarray] = []
        self._acc_max = 4  # Number of frames to accumulate before estimating

    def update(
            self, ref_frame: np.ndarray, mic_frame: np.ndarray
    ) -> int:
        """
        Feed one frame and get the current delay estimate in samples.

        Estimation runs every `_acc_max` frames to amortize FFT cost.

        Args:
            ref_frame : Reference signal (from speaker buffer)
            mic_frame : Microphone signal
        Returns:
            Estimated delay D in samples (integer, ≥ 0)
        """
        self._ref_acc.append(ref_frame.astype(np.float64))
        self._mic_acc.append(mic_frame.astype(np.float64))

        if len(self._ref_acc) >= self._acc_max:
            # Concatenate accumulated frames
            x = np.concatenate(self._ref_acc)
            d = np.concatenate(self._mic_acc)
            self._ref_acc.clear()
            self._mic_acc.clear()

            raw_delay = self._gcc_phat(x, d)
            self._delay_smooth = (
                self.alpha * self._delay_smooth + (1 - self.alpha) * raw_delay
            )

        return int(round(self._delay_smooth))

    def _gcc_phat(self, x: np.ndarray, d: np.ndarray) -> int:
        """
        Core GCC-PHAT computation.

        Returns estimated delay in samples.
        """
        n = self.fft_size

        # Zero-pad to fft_size
        X = np.fft.rfft(x, n=n)
        D = np.fft.rfft(d, n=n)

        # Cross-power spectrum
        R = X * np.conj(D)

        # PHAT weighting: normalize by magnitude → whitens the spectrum
        # This sharpens the GCC peak, making it more robust to colored noise
        magnitude = np.abs(R)
        magnitude = np.maximum(magnitude, 1e-10)
        R_phat = R / magnitude

        # Back to time domain
        gcc = np.fft.irfft(R_phat, n=n)

        # Search only in [0, max_delay_samples] range
        # We assume echo is always AFTER the reference (causal system)
        search_range = gcc[:self.max_delay_samples]
        delay = int(np.argmax(search_range))

        return delay

    @property
    def current_delay_samples(self) -> int:
        """Current smoothed delay estimate."""
        return int(round(self._delay_smooth))

    @property
    def current_delay_ms(self) -> float:
        """Current delay estimate in milliseconds."""
        return self._delay_smooth * 1000.0 / self.sr


# --- SỬA LỖI 1: THAY THẾ HÀM apply_delay BẰNG CLASS DelayLine ---
# Sử dụng cơ chế Ring Buffer để đảm bảo tín hiệu reference liên tục qua các frame
# Không bị chèn 0 và cắt xén ở mỗi frame riêng lẻ.
class DelayLine:
    """
    Continuous Delay Line (Ring Buffer) for aligning signals.
    """
    def __init__(self, max_delay_samples: int = 48000) -> None:
        # Bộ đệm đủ lớn để chứa lịch sử tín hiệu (mặc định 3s ở 16kHz)
        self.max_delay = max_delay_samples
        self.buffer = np.zeros(self.max_delay, dtype=np.float64)
        self.write_idx = 0

    def process(self, frame: np.ndarray, delay: int) -> np.ndarray:
        """
        Đẩy frame mới vào Ring Buffer và trích xuất frame trễ.
        
        Args:
            frame: Tín hiệu gốc cần làm trễ (reference signal)
            delay: Số lượng sample cần làm trễ (D >= 0)
        """
        N = len(frame)
        out = np.zeros(N, dtype=np.float64)
        delay = min(delay, self.max_delay - 1)
        
        for i in range(N):
            # Ghi sample mới vào buffer
            self.buffer[self.write_idx] = frame[i]
            # Tính chỉ số đọc (lùi lại 'delay' mẫu)
            read_idx = (self.write_idx - delay) % self.max_delay
            # Lấy mẫu đã làm trễ ra ngoài
            out[i] = self.buffer[read_idx]
            # Cập nhật con trỏ ghi
            self.write_idx = (self.write_idx + 1) % self.max_delay
            
        return out

    def reset(self) -> None:
        self.buffer[:] = 0.0
        self.write_idx = 0