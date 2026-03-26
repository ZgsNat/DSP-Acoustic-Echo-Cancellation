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
    """Uoc luong do tre bang GCC-PHAT voi co che on dinh.

    Tich luy acc_frames frame truoc khi tinh, sau do lam min pho GCC-PHAT
    theo thoi gian. Them hysteresis de tranh nhay delay lien tuc:
      - Chi chap nhan delay moi khi GCC peak du manh (confidence > min_confidence)
      - Chi thay doi khi delay moi xuat hien lien tiep (confirm_count lan)
      - Khi da lock delay, chi unlock khi peak tai delay cu yeu di ro ret
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_delay_ms: float = 250.0,
        smooth_alpha: float = 0.9,
        acc_frames: int = 4,
        min_confidence: float = 0.15,
        confirm_count: int = 3,
    ) -> None:
        self.sr = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        self.alpha = smooth_alpha
        self.acc_frames = acc_frames

        # Hysteresis: delay moi phai dat min_confidence va xuat hien
        # lien tiep confirm_count lan truoc khi duoc chap nhan.
        self.min_confidence = min_confidence
        self.confirm_count = confirm_count

        self._ref_acc: list[np.ndarray] = []
        self._mic_acc: list[np.ndarray] = []
        self._R_phat_smooth = None
        self._current_delay = 0

        # Hysteresis state
        self._candidate_delay = 0
        self._candidate_count = 0
        self._locked = False
        self._lock_confidence = 0.0

    def update(self, ref_frame: np.ndarray, mic_frame: np.ndarray) -> int:
        """Cap nhat uoc luong delay voi cap frame moi.

        Tich luy du acc_frames frame roi moi tinh GCC-PHAT.
        Ap dung hysteresis de on dinh delay.

        Args:
            ref_frame: Tin hieu reference shape (N,)
            mic_frame: Tin hieu mic shape (N,)

        Returns:
            delay: So mau tre uoc luong (>= 0)
        """
        self._ref_acc.append(ref_frame.astype(np.float64))
        self._mic_acc.append(mic_frame.astype(np.float64))

        if len(self._ref_acc) >= self.acc_frames:
            x = np.concatenate(self._ref_acc)
            d = np.concatenate(self._mic_acc)
            self._ref_acc.clear()
            self._mic_acc.clear()

            # Kiem tra ca ref va mic co tin hieu khong
            # Neu mot trong hai im lang, GCC-PHAT cho ket qua vo nghia
            ref_energy = float(np.mean(x ** 2))
            mic_energy = float(np.mean(d ** 2))
            if ref_energy < 1e-7 or mic_energy < 1e-7:
                # Khong du tin hieu de uoc luong -> giu delay cu
                return self._current_delay

            n_fft = 2 * len(x)
            X = np.fft.rfft(x, n=n_fft)
            D = np.fft.rfft(d, n=n_fft)

            R = X * np.conj(D)
            magnitude = np.abs(R)
            magnitude = np.maximum(magnitude, 1e-10)
            R_phat = R / magnitude

            if self._R_phat_smooth is None:
                self._R_phat_smooth = R_phat
            else:
                self._R_phat_smooth = (
                    self.alpha * self._R_phat_smooth
                    + (1.0 - self.alpha) * R_phat
                )

            gcc = np.fft.irfft(self._R_phat_smooth, n=n_fft)

            search_range = gcc[:self.max_delay_samples + 1]
            raw_delay = int(np.argmax(search_range))
            peak_val = float(search_range[raw_delay])

            # Confidence: peak so voi mean. Cao = peak ro rang.
            mean_val = float(np.mean(np.abs(search_range)))
            confidence = peak_val / (mean_val + 1e-10)

            # Hysteresis logic
            if confidence < self.min_confidence:
                # Peak qua yeu, khong tin cay -> giu delay cu
                pass
            elif not self._locked:
                # Chua lock: chap nhan delay moi neu lien tiep confirm_count lan
                if abs(raw_delay - self._candidate_delay) <= 2:
                    self._candidate_count += 1
                else:
                    self._candidate_delay = raw_delay
                    self._candidate_count = 1

                if self._candidate_count >= self.confirm_count:
                    self._current_delay = self._candidate_delay
                    self._locked = True
                    self._lock_confidence = confidence
            else:
                # Da lock: chi thay doi khi delay moi lien tiep VA
                # peak tai delay cu yeu di (echo path thay doi that su)
                current_gcc_at_lock = float(search_range[
                    min(self._current_delay, len(search_range) - 1)
                ])
                lock_still_good = current_gcc_at_lock > 0.5 * peak_val

                if lock_still_good:
                    # Delay cu van tot -> giu nguyen
                    pass
                else:
                    # Delay cu yeu -> xem xet delay moi
                    if abs(raw_delay - self._candidate_delay) <= 2:
                        self._candidate_count += 1
                    else:
                        self._candidate_delay = raw_delay
                        self._candidate_count = 1

                    if self._candidate_count >= self.confirm_count:
                        self._current_delay = self._candidate_delay
                        self._lock_confidence = confidence
                        self._candidate_count = 0

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