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

# ==========================================
# FILE 3: nonlinear_suppressor.py
# ==========================================
"""
Nonlinear (Residual) Echo Suppressor
(Đã sửa lỗi nhiễu âm thanh bằng Windowing và Overlap-Add)
"""

import numpy as np

class NonlinearSuppressor:
    """
    Spectral subtraction-based residual echo suppressor.
    """

    def __init__(
        self,
        frame_size: int = 1024,
        alpha: float = 1.5,
        beta: float = 0.05,
        smooth_alpha: float = 0.85,
    ) -> None:
        self.hop_size = frame_size
        
        # --- SỬA LỖI 3: Khởi tạo biến cho Overlap-Add (OLA) ---
        # Phân tích FFT trên cửa sổ lớn gấp đôi (Overlap 50%)
        self.window_size = frame_size * 2
        self.alpha = alpha
        self.beta = beta
        self.smooth_alpha = smooth_alpha

        # Tạo Hann Window để tránh lỗi tích chập vòng ở biên
        self.window = np.hanning(self.window_size)
        
        # Bộ đệm để ghép nối frame cũ và mới (Overlap 50%)
        self._e_buffer = np.zeros(self.window_size, dtype=np.float64)
        self._y_buffer = np.zeros(self.window_size, dtype=np.float64)
        self._ola_buffer = np.zeros(self.window_size, dtype=np.float64)
        
        self._echo_power_smooth: np.ndarray | None = None

    def process(
        self,
        residual_frame: np.ndarray,
        echo_estimate_frame: np.ndarray,
    ) -> np.ndarray:
        assert len(residual_frame) == self.hop_size
        
        e = residual_frame.astype(np.float64)
        y = echo_estimate_frame.astype(np.float64)

        # --- SỬA LỖI 3: Cập nhật bộ đệm đầu vào (Trượt dữ liệu cũ, thêm dữ liệu mới) ---
        self._e_buffer[:-self.hop_size] = self._e_buffer[self.hop_size:]
        self._e_buffer[-self.hop_size:] = e
        
        self._y_buffer[:-self.hop_size] = self._y_buffer[self.hop_size:]
        self._y_buffer[-self.hop_size:] = y

        # Nhân cửa sổ Hann
        e_win = self._e_buffer * self.window
        y_win = self._y_buffer * self.window

        # FFT trên window_size (2048)
        E = np.fft.rfft(e_win)
        Y = np.fft.rfft(y_win)

        E_power = np.abs(E) ** 2
        Y_power = np.abs(Y) ** 2

        if self._echo_power_smooth is None:
            self._echo_power_smooth = Y_power.copy()
        else:
            self._echo_power_smooth = (
                self.smooth_alpha * self._echo_power_smooth
                + (1.0 - self.smooth_alpha) * Y_power
            )

        # Spectral Subtraction
        E_clean_power = np.maximum(
            E_power - self.alpha * self._echo_power_smooth,
            self.beta * E_power,
        )

        E_mag = np.sqrt(E_clean_power)
        E_phase = np.angle(E)
        E_clean = E_mag * np.exp(1j * E_phase)

        # IFFT quay về miền thời gian
        e_clean_win = np.fft.irfft(E_clean, n=self.window_size)

        # --- SỬA LỖI 3: Cộng dồn Overlap-Add (OLA) ---
        self._ola_buffer += e_clean_win
        
        # Trích xuất đoạn âm thanh (hop_size) đã xử lý hoàn chỉnh để xuất ra
        out_frame = self._ola_buffer[:self.hop_size].copy()
        
        # Dịch bộ đệm OLA cho frame tiếp theo (Shift left)
        self._ola_buffer[:-self.hop_size] = self._ola_buffer[self.hop_size:]
        self._ola_buffer[-self.hop_size:] = 0.0

        return out_frame.astype(np.float32)

    def reset(self) -> None:
        self._echo_power_smooth = None
        self._e_buffer[:] = 0.0
        self._y_buffer[:] = 0.0
        self._ola_buffer[:] = 0.0