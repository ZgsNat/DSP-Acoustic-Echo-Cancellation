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

# ==========================================
# FILE 2: nlms_filter.py
# ==========================================
"""
NLMS Adaptive Filter — Core of AEC
(Đã sửa nút thắt cổ chai bằng cách Vectorize mảng trượt)
"""

import numpy as np
from dataclasses import dataclass

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
        
        # --- SỬA LỖI 2: Dùng buffer tuyến tính thay vì circular buffer thủ công ---
        # Chỉ lưu lịch sử L-1 mẫu của frame trước đó
        self._history: np.ndarray = np.zeros(L - 1, dtype=np.float64)

    def process(
        self,
        mic_frame: np.ndarray,
        ref_frame: np.ndarray,
        update: bool = True,
    ) -> np.ndarray:
        N = len(mic_frame)
        L = self.cfg.filter_length
        mu = self.cfg.mu
        eps = self.cfg.eps

        # Work in float64 for numerical stability
        mic = mic_frame.astype(np.float64)
        ref = ref_frame.astype(np.float64)
        e_frame = np.empty(N, dtype=np.float64)

        # --- SỬA LỖI 2: Tối ưu hoá bằng strided windows ---
        # 1. Ghép lịch sử frame trước với frame hiện tại: Độ dài = (L-1) + N
        full_ref = np.concatenate([self._history, ref])
        
        # Cập nhật lịch sử cho frame tiếp theo
        self._history[:] = full_ref[-(L - 1):]

        # 2. Tạo views trượt qua mảng (không tốn thêm RAM, không cấp phát mảng con)
        # stride_tricks trả về các cửa sổ đi tới: [x[n-L+1] ... x[n]]
        # lật [::-1] ở chiều thứ 2 để có dạng: [x[n], x[n-1] ... x[n-L+1]]
        X_mat = np.lib.stride_tricks.sliding_window_view(full_ref, L)[:, ::-1]

        # Vòng lặp bây giờ thuần tuý là phép toán vector, triệt tiêu lệnh cấp phát/concatenate bên trong
        for n in range(N):
            x_vec = X_mat[n]                     # Lấy sẵn view trực tiếp O(1)
            y_n = np.dot(self.w, x_vec)          # y(n) = w^T * x_vec
            e_n = mic[n] - y_n                   # e(n) = d(n) - y(n)
            e_frame[n] = e_n

            if update:
                norm = np.dot(x_vec, x_vec) + eps
                self.w += (mu / norm) * e_n * x_vec

        return e_frame

    @property
    def weight_norm(self) -> float:
        """L2 norm of filter weights — proxy for convergence."""
        return float(np.linalg.norm(self.w))

    def reset(self) -> None:
        self.w[:] = 0.0
        self._history[:] = 0.0