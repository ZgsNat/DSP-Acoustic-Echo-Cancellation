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
    """Geigel Double-Talk Detector.

    So sanh peak mic voi peak ref (tich luy theo thoi gian) de phat hien
    near-end speech. Ket hop voi hangover de tranh nhap nhay.

    Tham so mac dinh:
      - threshold=0.5: DT khi mic > 50% ref_max. Gia tri 0.8 duoc
        dung trong AECConfig (cho phep nhieu hon truoc khi bat DT).
      - echo_tail_ms=300: ref_buf phu 300ms, du cho RIR test dai nhat.
      - hangover_ms=100: giu DT them 100ms sau khi raw_dt tat.
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

        # mic_buf: chi can vai frame gan nhat (2 frame ~ 128ms o 1024@16kHz)
        self._mic_buf: deque = deque(maxlen=2)

        # ref_buf: phai phu het echo tail. Tam dat maxlen=8 (~500ms),
        # se tu dong tinh lai chinh xac o lan detect() dau tien khi biet
        # frame_size thuc te.
        self._ref_buf: deque = deque(maxlen=8)
        self._ref_buf_configured = False

        self._hangover_frames_left: int = 0
        self._in_double_talk: bool = False

    def detect(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> bool:
        """Phan loai frame hien tai: co double-talk hay khong.

        Args:
            mic_frame: Tin hieu mic shape (N,)
            ref_frame: Tin hieu reference shape (N,)

        Returns:
            True neu dang co double-talk (pipeline nen dong bang NLMS)
        """
        # Lan dau: tinh lai maxlen cua ref_buf dua tren frame_size thuc te
        if not self._ref_buf_configured:
            frame_ms = len(mic_frame) / self.sr * 1000.0
            ref_frames_needed = max(int(np.ceil(self._echo_tail_ms / frame_ms)), 2)
            old_data = list(self._ref_buf)
            self._ref_buf = deque(old_data, maxlen=ref_frames_needed)
            self._ref_buf_configured = True

        mic_peak = float(np.max(np.abs(mic_frame)))
        ref_peak = float(np.max(np.abs(ref_frame)))

        self._mic_buf.append(mic_peak)
        self._ref_buf.append(ref_peak)

        max_mic = max(self._mic_buf)
        max_ref = max(self._ref_buf)

        # Kiem tra muc toi thieu: khi im lang (ca hai < MIN_LEVEL),
        # ti so peak khong co y nghia -> luon tra ve False.
        # Khong co buoc nay, ti le false positive khi im lang len toi 12%.
        MIN_LEVEL = 1e-4
        if max_mic < MIN_LEVEL and max_ref < MIN_LEVEL:
            raw_dt = False
        else:
            # Tieu chuan Geigel: mic lon hon threshold * ref_history
            # thi co near-end speech. (+1e-10 de tranh chia 0)
            raw_dt = max_mic > self.threshold * (max_ref + 1e-10)

        # Hangover: giu DT them vai frame sau khi raw_dt tat,
        # tranh NLMS cap nhat ngay lap tuc khi near-end chua noi xong.
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

    @property
    def is_double_talk(self) -> bool:
        """Trang thai DT hien tai (True/False)."""
        return self._in_double_talk