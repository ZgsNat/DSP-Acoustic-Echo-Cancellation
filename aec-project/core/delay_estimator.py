"""
delay_estimator.py - Uoc luong do tre va can chinh tin hieu reference

Module nay giai quyet bai toan: tin hieu reference x(n) (lay tu buffer loa)
va echo trong mic d(n) co do tre D mau khong biet truoc, do:
  - Buffer latency cua driver am thanh (~20-50ms)
  - Thoi gian truyen am tu loa den mic (khoang 1ms / 34cm)

Neu khong uoc luong va bu D, NLMS se khong hoi tu duoc vi cua so bo loc
khong khop voi vi tri echo thuc te trong tin hieu mic.

Gom 2 class:
  1. DelayEstimator: Dung GCC-PHAT de tinh D tu tuong quan cheo.
  2. DelayLine: Ring buffer de dich cham tin hieu reference di D mau.

GCC-PHAT (Generalized Cross-Correlation with Phase Transform):
    R(f) = X(f) * conj(D(f))
    R_phat(f) = R(f) / |R(f)|        <-- chi giu thong tin pha, bo bien do
    gcc(tau) = IFFT(R_phat)
    delay = argmax(gcc(tau))

    PHAT weighting chi giu pha nen peak rat nhon, chinh xac hon
    cross-correlation thong thuong (khong bi anh huong boi bien do tin hieu).

Ky thuat toi uu ap dung:
  - Tich luy nhieu frame (acc_frames=4) truoc khi tinh GCC -> giam jitter
  - Lam min pho GCC-PHAT (smooth_alpha=0.9) -> delay on dinh sau hoi tu
  - Zero-padding (2*N) -> tranh nhieu circular correlation
"""

import numpy as np


class DelayEstimator:
    """Uoc luong do tre bang GCC-PHAT.

    Tich luy acc_frames frame truoc khi tinh, sau do lam min pho GCC-PHAT
    theo thoi gian. Ket qua la delay on dinh, khong bi jitter.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_delay_ms: float = 250.0,
        smooth_alpha: float = 0.9,
        acc_frames: int = 4,
    ) -> None:
        self.sr = sample_rate

        # Gioi han tim kiem: chi xet delay trong khoang [0, max_delay_samples].
        # 250ms o 16kHz = 4000 mau. Du cho hau het he thong desktop.
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)

        # He so lam min pho GCC-PHAT. Gia tri cao (0.9) -> on dinh nhung
        # phan ung cham voi thay doi. Gia tri thap -> nhay nhung nhieu jitter.
        self.alpha = smooth_alpha

        # So frame tich luy truoc khi thuc hien GCC-PHAT.
        # Tich luy nhieu frame -> tin hieu dai hon -> peak GCC ro rang hon.
        self.acc_frames = acc_frames

        # Buffer tich luy tin hieu reference va mic
        self._ref_acc: list[np.ndarray] = []
        self._mic_acc: list[np.ndarray] = []

        # Pho GCC-PHAT da lam min (duy tri qua cac lan goi)
        self._R_phat_smooth = None

        # Ket qua delay hien tai (so mau)
        self._current_delay = 0

    def update(self, ref_frame: np.ndarray, mic_frame: np.ndarray) -> int:
        """Cap nhat uoc luong delay voi cap frame moi.

        Tich luy du acc_frames frame roi moi tinh GCC-PHAT.
        Giua cac lan tinh, tra ve delay cu (amortized cost).

        Args:
            ref_frame: Tin hieu reference shape (N,)
            mic_frame: Tin hieu mic shape (N,)

        Returns:
            delay: So mau tre uoc luong (>= 0)
        """
        self._ref_acc.append(ref_frame.astype(np.float64))
        self._mic_acc.append(mic_frame.astype(np.float64))

        if len(self._ref_acc) >= self.acc_frames:
            # Noi tat ca frame da tich luy thanh tin hieu dai
            x = np.concatenate(self._ref_acc)
            d = np.concatenate(self._mic_acc)
            self._ref_acc.clear()
            self._mic_acc.clear()

            # Zero-padding (2*N): bat buoc de tranh nhieu do circular correlation.
            # Khi FFT khong co zero-pad, tuong quan vong (circular) co the tao
            # peak gia vi tin hieu "cuon vong" dau-cuoi.
            n_fft = 2 * len(x)
            X = np.fft.rfft(x, n=n_fft)
            D = np.fft.rfft(d, n=n_fft)

            # Tinh cross-spectrum va ap dung PHAT weighting (chi giu pha)
            R = X * np.conj(D)
            magnitude = np.abs(R)
            magnitude = np.maximum(magnitude, 1e-10)  # Tranh chia 0
            R_phat = R / magnitude

            # Lam min pho GCC-PHAT qua thoi gian.
            # Khac voi cach lam min gia tri delay (bi nhay lien tuc),
            # lam min truc tiep tren pho giup peak GCC dan sac net hon,
            # va delay hoi tu ve gia tri on dinh tuyet doi (khong jitter).
            if self._R_phat_smooth is None:
                self._R_phat_smooth = R_phat
            else:
                self._R_phat_smooth = (
                    self.alpha * self._R_phat_smooth
                    + (1.0 - self.alpha) * R_phat
                )

            # IFFT de chuyen ve mien thoi gian -> tim peak
            gcc = np.fft.irfft(self._R_phat_smooth, n=n_fft)

            # Chi tim trong khoang [0, max_delay_samples]
            search_range = gcc[:self.max_delay_samples + 1]
            self._current_delay = int(np.argmax(search_range))

        return self._current_delay

    @property
    def current_delay_samples(self) -> int:
        """Delay hien tai tinh bang so mau."""
        return self._current_delay

    @property
    def current_delay_ms(self) -> float:
        """Delay hien tai tinh bang mili-giay."""
        return self._current_delay * 1000.0 / self.sr


class DelayLine:
    """Ring buffer de dich cham (delay) tin hieu reference.

    Khi DelayEstimator tinh duoc delay = D mau, DelayLine se tra ve
    x(n - D) thay vi x(n). Cach lam: ghi mau moi vao buffer tai write_idx,
    doc mau cu tai (write_idx - D), ca hai quay vong trong mang co dinh.

    Uu diem so voi np.roll hay concatenate:
      - Khong cap phat bo nho moi moi frame
      - Delay co the thay doi bat ky luc nao ma khong mat du lieu
      - Hoat dong lien tuc qua cac frame (stateful)
    """

    def __init__(self, max_delay_samples: int = 48000) -> None:
        # max_delay = 48000 mau = 3 giay o 16kHz. Du cho moi kich ban thuc te.
        self.max_delay = max_delay_samples
        self.buffer = np.zeros(self.max_delay, dtype=np.float64)
        self.write_idx = 0

    def process(self, frame: np.ndarray, delay: int) -> np.ndarray:
        """Dich cham frame di delay mau.

        Args:
            frame: Tin hieu dau vao shape (N,)
            delay: So mau can tre (>=0)

        Returns:
            out: Tin hieu da duoc tre delay mau, shape (N,)
        """
        N = len(frame)
        out = np.zeros(N, dtype=np.float64)

        # Clamp delay de tranh truy cap ngoai vung buffer
        delay = int(np.clip(delay, 0, self.max_delay - 1))

        for i in range(N):
            # Ghi mau moi vao vi tri hien tai
            self.buffer[self.write_idx] = frame[i]
            # Doc mau cu cach delay buoc ve truoc (modulo de quay vong)
            read_idx = (self.write_idx - delay) % self.max_delay
            out[i] = self.buffer[read_idx]
            self.write_idx = (self.write_idx + 1) % self.max_delay

        return out

    def reset(self) -> None:
        """Xoa toan bo buffer. Goi khi bat dau phien moi."""
        self.buffer[:] = 0.0
        self.write_idx = 0