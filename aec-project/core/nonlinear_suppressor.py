"""
nonlinear_suppressor.py - Nen echo du bang Spectral Subtraction voi OLA

Sau khi NLMS khu thanh phan echo tuyen tinh, van con echo du (residual) do:
  1. Me phi tuyen cua loa/mic (loa gia lam meo tin hieu, NLMS tuyen tinh
     khong mo hinh hoa duoc phan phi tuyen nay)
  2. Bo loc chua hoi tu het (misadjustment > 0)
  3. RIR thay doi khi nguoi dung di chuyen

Module nay dung Spectral Subtraction (tru pho) de giam echo du:

    |E_clean(f)|^2 = max( |E(f)|^2 - alpha * |Y_smooth(f)|^2,
                          beta  * |E(f)|^2 )

Trong do:
    E(f)        : Pho cua tin hieu sau NLMS (con echo du)
    Y_smooth(f) : Uoc luong cong suat echo (lam min tu echo estimate)
    alpha       : He so over-subtraction. Lon hon -> nen manh hon.
                  Mac dinh 1.5, tuned len 3.0 trong AECConfig.
    beta        : San pho (spectral floor). Ngan tru am thanh thanh 0
                  tuyet doi -> tranh musical noise. Mac dinh 0.05,
                  tuned xuong 0.002 trong AECConfig.

Pha cua E(f) duoc giu nguyen (khong xu ly pha). Day la cach tieu chuan
vi uoc luong pha rat kho va khong cai thien nhieu chat luong.

Overlap-Add (OLA):
  FFT yeu cau cua so co bien am (Hann window), nhung windowing lam mat
  tin hieu o bien frame. OLA giai quyet bang cach chong 50% frame:
    - Window size = 2 * frame_size (2048 voi frame 1024)
    - Hop size = frame_size (1024)
    - Do chong = 50%
  Khi cong 2 cua so Hann chong 50%, tong = 1.0 (hoan hao).
  Khong co OLA, am thanh se bi nhieu click o bien moi frame.

Lich su tinh chinh:
  - alpha 1.5 -> 3.0: nen manh hon, ERLE tang tu ~25dB len ~45dB
  - beta 0.05 -> 0.002: san thap hon, echo du giam ro ret
  - Ket hop voi safety clamp trong pipeline de tranh OLA burst
"""

import numpy as np


class NonlinearSuppressor:
    """Spectral Subtraction voi Hann windowing va Overlap-Add.

    Nhan residual frame (sau NLMS) va echo estimate frame, tra ve
    tin hieu da nen echo du. Hoat dong trong mien tan so.
    """

    def __init__(
        self,
        frame_size: int = 1024,
        alpha: float = 1.5,
        beta: float = 0.05,
        smooth_alpha: float = 0.85,
    ) -> None:
        self.hop_size = frame_size

        # FFT window lon gap doi frame (overlap 50%) de OLA hoat dong dung
        self.window_size = frame_size * 2
        self.alpha = alpha
        self.beta = beta

        # He so lam min cong suat echo. Gia tri cao (0.85) giup uoc luong
        # echo on dinh, tranh spectral subtraction nhay theo tung frame.
        self.smooth_alpha = smooth_alpha

        # Cua so Hann: bien do giam ve 0 o 2 dau frame.
        # 2 cua so Hann chong 50% cong lai = 1.0 (perfect reconstruction).
        self.window = np.hanning(self.window_size)

        # Buffer dau vao: giu frame cu (nua truoc) + frame moi (nua sau)
        # de tao cua so 2048 mau voi overlap 50%.
        self._e_buffer = np.zeros(self.window_size, dtype=np.float64)
        self._y_buffer = np.zeros(self.window_size, dtype=np.float64)

        # Buffer OLA: tich luy cac frame da IFFT, chong len nhau
        self._ola_buffer = np.zeros(self.window_size, dtype=np.float64)

        # Uoc luong cong suat echo da lam min (duy tri qua cac frame)
        self._echo_power_smooth: np.ndarray | None = None

    def process(
        self,
        residual_frame: np.ndarray,
        echo_estimate_frame: np.ndarray,
    ) -> np.ndarray:
        """Nen echo du cho 1 frame.

        Args:
            residual_frame: Tin hieu sau NLMS, shape (hop_size,)
            echo_estimate_frame: Uoc luong echo tu NLMS, shape (hop_size,)

        Returns:
            out_frame: Tin hieu da nen echo du, shape (hop_size,), float32
        """
        assert len(residual_frame) == self.hop_size

        e = residual_frame.astype(np.float64)
        y = echo_estimate_frame.astype(np.float64)

        # Truot buffer: bo nua cu (ben trai), them frame moi (ben phai)
        # Ket qua: _e_buffer = [frame cu | frame moi] (overlap 50%)
        self._e_buffer[:-self.hop_size] = self._e_buffer[self.hop_size:]
        self._e_buffer[-self.hop_size:] = e

        self._y_buffer[:-self.hop_size] = self._y_buffer[self.hop_size:]
        self._y_buffer[-self.hop_size:] = y

        # Nhan cua so Hann truoc FFT
        e_win = self._e_buffer * self.window
        y_win = self._y_buffer * self.window

        # FFT tren window_size (2048 diem)
        E = np.fft.rfft(e_win)
        Y = np.fft.rfft(y_win)

        E_power = np.abs(E) ** 2
        Y_power = np.abs(Y) ** 2

        # Lam min cong suat echo theo thoi gian (exponential moving average)
        if self._echo_power_smooth is None:
            self._echo_power_smooth = Y_power.copy()
        else:
            self._echo_power_smooth = (
                self.smooth_alpha * self._echo_power_smooth
                + (1.0 - self.smooth_alpha) * Y_power
            )

        # Spectral Subtraction: tru cong suat echo khoi tin hieu
        # max(..., beta * E_power) dam bao khong bao gio am -> tranh musical noise
        E_clean_power = np.maximum(
            E_power - self.alpha * self._echo_power_smooth,
            self.beta * E_power,
        )

        # Tai tao tin hieu: bien do moi + pha cu
        E_mag = np.sqrt(E_clean_power)
        E_phase = np.angle(E)
        E_clean = E_mag * np.exp(1j * E_phase)

        # IFFT quay ve mien thoi gian
        e_clean_win = np.fft.irfft(E_clean, n=self.window_size)

        # Overlap-Add: cong frame hien tai vao buffer OLA
        self._ola_buffer += e_clean_win

        # Trich xuat hop_size mau dau (da duoc chong du 2 lan) lam output
        out_frame = self._ola_buffer[:self.hop_size].copy()

        # Dich OLA buffer sang trai, xoa nua sau de chuan bi cho frame tiep
        self._ola_buffer[:-self.hop_size] = self._ola_buffer[self.hop_size:]
        self._ola_buffer[-self.hop_size:] = 0.0

        return out_frame.astype(np.float32)

    def reset(self) -> None:
        """Xoa toan bo buffer va trang thai. Goi khi bat dau phien moi."""
        self._echo_power_smooth = None
        self._e_buffer[:] = 0.0
        self._y_buffer[:] = 0.0
        self._ola_buffer[:] = 0.0