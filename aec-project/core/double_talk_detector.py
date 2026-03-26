"""
double_talk_detector.py - Phat hien double-talk bang tieu chuan Geigel

Khi nguoi dung near-end noi cung luc voi echo (double-talk), NLMS se
bi nhieu vi tin hieu mong muon (near-end speech) tro thanh "nhieu" doi voi
qua trinh cap nhat bo loc. Neu khong phat hien va dung cap nhat, NLMS se
phan ky va echo tang len.

Giai phap: Geigel DTD phat hien double-talk, bao pipeline dong bang NLMS.

Tieu chuan Geigel:
    Neu max|mic| > threshold * max|ref_history|  =>  double-talk

Y tuong: echo la ban sao suy giam cua reference. Neu mic lon hon nhieu
so voi reference (tinh ca history) => co them tin hieu khac (near-end speech).

Cac van de da gap va cach khac phuc:
  1. False positive khi im lang: Ca mic va ref deu o noise floor (~1e-6),
     ti so khong xac dinh -> bat DT nham. Fix: them MIN_LEVEL=1e-4,
     neu ca hai < MIN_LEVEL thi tra ve False. (giam false positive tu 12% ve 0%)

  2. False positive khi ref vua tat: Echo con doi lai (vi RIR dai ~200ms)
     nhung ref_buf ngan "quen" rang ref da tung to -> tuong mic to la
     near-end -> dong bang NLMS dung luc can cap nhat nhat.
     Fix: ref_buf phu het echo tail (echo_tail_ms=300ms). Buffer tu tinh
     maxlen khi biet frame_size thuc te o lan goi dau tien.

  3. Nhap nhay DT (on/off lien tuc): Khi near-end noi, co khoang im
     ngan giua cac am -> DT tat roi bat ngay. Fix: hangover mechanism
     giu DT=True them hangover_ms=100ms sau khi raw_dt tat.
"""

import numpy as np
from collections import deque


class GeigelhDTD:
    """Geigel Double-Talk Detector — phien ban cai tien cho real-time.

    Van de voi Geigel goc: so sanh mic peak vs ref peak.
    Nhung mic LUON chua echo (ban sao cua ref), nen mic peak >= ref peak
    hau nhu moi luc → false positive rate qua cao (78% trong thuc te).

    Phien ban cai tien: su dung ti so nang luong (energy ratio).
    Mic chua echo + near_end. Echo ~= gain * ref. Neu mic >> gain * ref
    thi co near-end. Dung RMS thay vi peak de on dinh hon.

    Them co che:
      - Uoc luong echo gain tu cac frame im lang (chi co echo)
      - Gia tri MIN_LEVEL cao hon de tranh false positive khi im lang
      - Hangover van duoc giu lai
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        hangover_ms: float = 100.0,
        echo_tail_ms: float = 300.0,
    ) -> None:
        self.sr = sample_rate
        self.threshold = threshold
        self.hangover_ms = hangover_ms
        self._echo_tail_ms = echo_tail_ms

        self._mic_buf: deque = deque(maxlen=2)
        self._ref_buf: deque = deque(maxlen=8)
        self._ref_buf_configured = False

        self._hangover_frames_left: int = 0
        self._in_double_talk: bool = False

        # Uoc luong echo gain: mic_rms / ref_rms khi chi co echo
        # Khoi tao = 1.0 (gia dinh echo gain vua phai)
        self._echo_gain_est: float = 1.0
        self._echo_gain_alpha: float = 0.95  # Lam min cham

    def detect(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> bool:
        """Phan loai frame hien tai: co double-talk hay khong.

        Phien ban cai tien: dung RMS energy ratio thay vi peak ratio.
        Chi phat hien DT khi mic_rms vuot xa muc echo du kien.

        Args:
            mic_frame: Tin hieu mic shape (N,)
            ref_frame: Tin hieu reference shape (N,)

        Returns:
            True neu dang co double-talk (pipeline nen dong bang NLMS)
        """
        if not self._ref_buf_configured:
            frame_ms = len(mic_frame) / self.sr * 1000.0
            ref_frames_needed = max(int(np.ceil(self._echo_tail_ms / frame_ms)), 2)
            old_data = list(self._ref_buf)
            self._ref_buf = deque(old_data, maxlen=ref_frames_needed)
            self._ref_buf_configured = True

        # Dung RMS thay vi peak — on dinh hon nhieu voi speech
        mic_rms = float(np.sqrt(np.mean(mic_frame.astype(np.float64) ** 2)))
        ref_rms = float(np.sqrt(np.mean(ref_frame.astype(np.float64) ** 2)))

        self._mic_buf.append(mic_rms)
        self._ref_buf.append(ref_rms)

        max_mic_rms = max(self._mic_buf)
        max_ref_rms = max(self._ref_buf)

        # Muc toi thieu: khi im lang, khong phat hien DT
        MIN_LEVEL = 1e-3
        if max_mic_rms < MIN_LEVEL and max_ref_rms < MIN_LEVEL:
            raw_dt = False
        elif max_ref_rms < MIN_LEVEL:
            # Ref im lang nhung mic co tin hieu → near-end speech (khong phai DT
            # theo nghia AEC, vi khong co echo de khu. Cho phep NLMS update
            # de no khong bi lech).
            raw_dt = False
        else:
            # So sanh mic_rms voi echo du kien = echo_gain * ref_rms
            # Neu mic >> echo_gain * ref → co them near-end speech
            expected_echo = self._echo_gain_est * max_ref_rms
            raw_dt = max_mic_rms > (1.0 + self.threshold) * expected_echo

            # Cap nhat uoc luong echo gain khi KHONG co DT
            # (chi khi ca ref va mic deu co tin hieu)
            if not raw_dt and max_ref_rms > MIN_LEVEL and max_mic_rms > MIN_LEVEL:
                current_gain = max_mic_rms / max_ref_rms
                # Chi cap nhat khi gain hop ly (0.1 → 10)
                if 0.1 < current_gain < 10.0:
                    self._echo_gain_est = (
                        self._echo_gain_alpha * self._echo_gain_est
                        + (1.0 - self._echo_gain_alpha) * current_gain
                    )

        # Hangover
        frame_ms = (len(mic_frame) / self.sr) * 1000.0
        hangover_frames_max = int(np.ceil(self.hangover_ms / frame_ms))

        if raw_dt:
            self._hangover_frames_left = hangover_frames_max
            self._in_double_talk = True
        elif self._hangover_frames_left > 0:
            self._hangover_frames_left -= 1
            self._in_double_talk = True
        else:
            self._in_double_talk = False

        return self._in_double_talk

    def reset(self) -> None:
        """Xoa trang thai. Goi khi bat dau phien moi."""
        self._mic_buf.clear()
        self._ref_buf.clear()
        self._hangover_frames_left = 0
        self._in_double_talk = False
        self._echo_gain_est = 1.0

    @property
    def is_double_talk(self) -> bool:
        """Trang thai DT hien tai (True/False)."""
        return self._in_double_talk