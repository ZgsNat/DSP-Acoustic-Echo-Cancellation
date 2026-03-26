"""
aec_pipeline.py - Pipeline AEC hoan chinh, diem vao chinh cua he thong

Ghep noi 4 khoi DSP thanh mot giao dien duy nhat:
    1. Delay Estimator (GCC-PHAT) - Uoc luong va bu do tre reference
    2. Double-Talk Detector (Geigel) - Bao ve bo loc khi near-end noi
    3. NLMS Adaptive Filter - Khu echo tuyen tinh
    4. Nonlinear Suppressor (Spectral Subtraction) - Nen echo du

Luong tin hieu:
    ref(n) --> [Delay Est.] --> ref_aligned(n) --> [NLMS] --> residual(n)
    mic(n) --+---> [DTD] --(freeze)--> [NLMS]        |
             |                                         v
             +----> echo_est = mic - residual --> [NLS] --> clean(n)

    Trong do:
    - mic(n) = near_end(n) + echo(n) + noise(n)
    - echo(n) = h(n) * ref(n-D)   (h = RIR, D = bulk delay)
    - residual = mic - w^T * ref_aligned   (w = NLMS weights)
    - clean = NLS(residual, echo_estimate)

Su dung trong desktop app:
    pipeline = AECPipeline()
    clean = pipeline.process(mic_frame, ref_frame)

Pipeline co trang thai (stateful) - KHONG duoc reset giua cac frame.
Chi goi reset() khi bat dau phien goi moi.

Lich su tinh chinh (thong so mac dinh trong AECConfig):
    filter_length: 512 -> 4096 (phu het RIR 256ms o 16kHz)
    mu:           0.1 -> 0.7  (hoi tu nhanh, ERLE tang tu ~15dB len 48dB)
    nls_alpha:    1.5 -> 3.0  (nen echo du manh hon)
    nls_beta:     0.05 -> 0.002 (san pho thap, output sach hon)
    dtd_threshold: 0.5 -> 0.8 (it bat DT nham, cho NLMS cap nhat nhieu hon)
"""

import numpy as np
from dataclasses import dataclass

from .nlms_filter import NLMSFilter, NLMSConfig
from .delay_estimator import DelayEstimator, DelayLine
from .double_talk_detector import GeigelhDTD
from .nonlinear_suppressor import NonlinearSuppressor


@dataclass
class AECConfig:
    """Cau hinh toan bo pipeline AEC.

    Cac gia tri mac dinh la ket qua cua qua trinh tinh chinh tren
    tin hieu speech voi RIR mo phong (4 reflection, dai 250ms).
    """

    sample_rate: int = 16000
    frame_size: int = 1024

    # NLMS: filter_length phai phu het RIR.
    # 4096 taps o 16kHz = 256ms, du cho hau het phong desktop/phong hop.
    filter_length: int = 4096

    # mu (step size): Gia tri lon -> hoi tu nhanh nhung misadjustment cao.
    # 0.7 la diem can bang tot nhat cho speech: ERLE=48.5dB (echo-only),
    # 45.9dB (truoc double-talk), 26.0dB (trong double-talk).
    mu: float = 0.7
    eps: float = 1e-6

    # Delay estimation: gioi han tim kiem GCC-PHAT
    max_delay_ms: float = 300.0

    # DTD: threshold 0.8 nghia la mic phai lon gap 80% ref moi bat DT.
    # Gia tri nay cho phep NLMS cap nhat nhieu hon, hoi tu tot hon.
    dtd_threshold: float = 0.8
    dtd_hangover_ms: float = 100.0

    # NLS: alpha=3.0 over-subtraction manh, beta=0.002 san rat thap.
    # Ket hop cho echo du gan bang 0 trong doan echo-only.
    nls_alpha: float = 3.0
    nls_beta: float = 0.002


class AECPipeline:
    """Pipeline AEC hoan chinh.

    Khong thread-safe. Goi tu mot audio thread duy nhat.
    """

    def __init__(self, config: AECConfig = AECConfig()) -> None:
        self.cfg = config

        # Khoi 1: Uoc luong do tre (GCC-PHAT)
        self._delay_est = DelayEstimator(
            sample_rate=config.sample_rate,
            max_delay_ms=config.max_delay_ms,
        )
        # Ring buffer de dich cham reference. Chua toi da 3 giay (48000 mau).
        self._delay_line = DelayLine(max_delay_samples=48000)

        # Khoi 2: Phat hien double-talk (Geigel)
        self._dtd = GeigelhDTD(
            sample_rate=config.sample_rate,
            threshold=config.dtd_threshold,
            hangover_ms=config.dtd_hangover_ms,
        )

        # Khoi 3: Bo loc thich nghi NLMS
        self._nlms = NLMSFilter(NLMSConfig(
            filter_length=config.filter_length,
            mu=config.mu,
            eps=config.eps,
        ))

        # Khoi 4: Nen echo du (Spectral Subtraction + OLA)
        self._nls = NonlinearSuppressor(
            frame_size=config.frame_size,
            alpha=config.nls_alpha,
            beta=config.nls_beta,
        )

        # Bo tich luy chi so hieu suat (reset khi goi get_metrics)
        self._mic_power_acc: list[float] = []
        self._out_power_acc: list[float] = []
        self._dt_count: int = 0
        self._frame_count: int = 0

    def process(
        self,
        mic_frame: np.ndarray,
        ref_frame: np.ndarray,
    ) -> np.ndarray:
        """Xu ly 1 frame qua toan bo pipeline.

        Args:
            mic_frame: Tin hieu mic d(n), shape (frame_size,).
                       Chua: near-end speech + echo + noise.
            ref_frame: Tin hieu reference x(n), shape (frame_size,).
                       Audio dang phat qua loa, lay tu playback buffer
                       TRUOC khi qua D/A converter.

        Returns:
            clean_frame: Tin hieu da khu echo, shape (frame_size,), float32.
                         San sang de encode va gui qua mang.
        """
        mic = mic_frame.astype(np.float32)
        ref = ref_frame.astype(np.float32)

        # Tich luy cong suat mic de tinh ERLE sau
        self._mic_power_acc.append(float(np.mean(mic ** 2)))
        self._frame_count += 1

        # Khoi 1: Uoc luong delay va can chinh reference
        delay = self._delay_est.update(ref, mic)
        ref_aligned = self._delay_line.process(ref, delay)

        # Khoi 2: Phat hien double-talk
        # Dung ref_aligned (da can chinh) de so sanh cong bang voi mic
        is_dt = self._dtd.detect(mic, ref_aligned)
        if is_dt:
            self._dt_count += 1

        # Khoi 3: NLMS - khu echo tuyen tinh
        # update=False khi double-talk -> dong bang trong so bo loc
        residual = self._nlms.process(mic, ref_aligned, update=not is_dt)
        # residual = e(n) = d(n) - w^T * x_aligned(n)
        self._last_residual = residual.copy()

        # Khoi 4: NLS - nen echo du
        # echo_estimate = mic - residual = w^T * x_aligned (phan echo uoc luong)
        echo_estimate = mic.astype(np.float64) - residual
        clean = self._nls.process(residual, echo_estimate)

        # Safety clamp: NLS khong duoc phep khuech dai vuot NLMS output.
        # Van de: OLA windowing co the tao burst nhat thoi tai chuyen tiep
        # tin hieu (vi du khi near-end ngung noi dot ngot). Burst nay lam
        # nang luong NLS output > NLMS output -> nghe thay tieng "pop".
        # Giai phap: neu NLS energy > NLMS energy -> dung thang NLMS output.
        nls_energy = float(np.mean(clean.astype(np.float64) ** 2))
        nlms_energy = float(np.mean(residual.astype(np.float64) ** 2))
        if nls_energy > nlms_energy and nlms_energy > 1e-10:
            clean = residual.copy()

        # Tich luy cong suat output de tinh ERLE
        self._out_power_acc.append(float(np.mean(clean.astype(np.float32) ** 2)))

        return clean.astype(np.float32)

    def get_metrics(self) -> dict:
        """Tinh va tra ve chi so hieu suat, dong thoi reset bo tich luy.

        Returns:
            dict voi cac key:
                erle_db: ERLE tinh bang dB. 10*log10(mic_power/out_power).
                         Cao hon = tot hon. Muc tieu >= 15dB (dat 48.5dB).
                double_talk_ratio: Ti le frame co double-talk.
                frame_count: Tong so frame da xu ly.
                delay_ms: Do tre uoc luong hien tai (ms).
                filter_norm: Chuan L2 cua trong so NLMS (do hoi tu).
        """
        mic_power = np.mean(self._mic_power_acc) if self._mic_power_acc else 1e-10
        out_power = np.mean(self._out_power_acc) if self._out_power_acc else 1e-10

        erle_db = 10.0 * np.log10(mic_power / (out_power + 1e-10))

        metrics = {
            "erle_db": float(erle_db),
            "double_talk_ratio": self._dt_count / max(self._frame_count, 1),
            "frame_count": self._frame_count,
            "delay_ms": self._delay_est.current_delay_ms,
            "filter_norm": self._nlms.weight_norm,
        }

        # Reset bo tich luy
        self._mic_power_acc.clear()
        self._out_power_acc.clear()
        self._dt_count = 0
        self._frame_count = 0

        return metrics

    def reset(self) -> None:
        """Reset toan bo trang thai. Goi khi bat dau phien goi moi."""
        self._delay_line.reset()
        self._nlms.reset()
        self._dtd.reset()
        self._nls.reset()
        self._mic_power_acc.clear()
        self._out_power_acc.clear()
        self._dt_count = 0
        self._frame_count = 0