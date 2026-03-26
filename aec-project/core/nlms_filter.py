"""
nlms_filter.py - Bo loc thich nghi NLMS (Normalized Least Mean Squares)

Day la thanh phan cot loi cua pipeline AEC. NLMS uoc luong Room Impulse
Response (RIR) giua loa va mic, sau do tru echo estimate khoi tin hieu mic.

Cong thuc cap nhat trong so:
    w(n+1) = w(n) + (mu / (||x(n)||^2 + eps)) * e(n) * x(n)

Trong do:
    w(n)  : vector trong so bo loc, do dai L (uoc luong RIR)
    x(n)  : vector tin hieu reference (L mau gan nhat tu loa)
    d(n)  : tin hieu mic (near-end speech + echo)
    y(n)  : echo estimate = w(n)^T * x(n)
    e(n)  : sai so / residual = d(n) - y(n), day chinh la output
    mu    : buoc nhay (trade-off toc do hoi tu vs do on dinh)
    eps   : hang so nho tranh chia 0
    L     : do dai bo loc (xac dinh do dai echo toi da co the khu)

So sanh voi LMS co ban:
    LMS:  w(n+1) = w(n) + mu * e(n) * x(n)
    NLMS: w(n+1) = w(n) + (mu / ||x(n)||^2) * e(n) * x(n)

    Diem khac biet quan trong: NLMS chia cho nang luong tin hieu x(n),
    nen khi nguoi noi to (||x|| lon) -> step size tu dong giam (on dinh),
    khi nguoi noi nho (||x|| nho) -> step size tu dong tang (responsive).
    Day la ly do NLMS phu hop cho speech hon LMS.

Toi uu hieu suat:
    Phien ban nay su dung sliding_window_view (numpy stride_tricks) de tao
    cac cua so truot ma khong can cap phat bo nho moi. Thay vi vong lap
    thua tuan tu cat/noi mang, ta tao truoc ma tran X_mat chua tat ca cac
    cua so, roi chi viec truy cap X_mat[n] voi do phuc tap O(1).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class NLMSConfig:
    """Cau hinh cho bo loc NLMS.

    Cac gia tri mac dinh la gia tri an toan chung. Khi dung trong AECPipeline,
    cac gia tri nay se duoc ghi de boi AECConfig (vd: filter_length=4096, mu=0.7).
    """

    # So luong tap (trong so) cua bo loc.
    # O tan so 16kHz: L=4096 tap = 256ms, du de phu hau het RIR phong desktop.
    # L lon hon -> bat duoc echo xa hon (reverb dai), nhung hoi tu cham hon va ton RAM.
    filter_length: int = 512

    # Buoc nhay mu, phai nam trong khoang (0, 2) de dam bao on dinh.
    # mu=0.1: cham nhung on dinh; mu=0.7: nhanh nhung steady-state error cao hon.
    # Sau nhieu lan tinh chinh, mu=0.7 cho ket qua tot nhat voi speech 5 giay,
    # dat ERLE 48.5dB trong test echo-only.
    mu: float = 0.1

    # Hang so regularization tranh chia 0 khi silence.
    eps: float = 1e-6


class NLMSFilter:
    """Bo loc thich nghi NLMS don kenh cho Acoustic Echo Cancellation.

    Cach su dung:
        filt = NLMSFilter(NLMSConfig(filter_length=4096, mu=0.7))
        for frame in audio_frames:
            residual = filt.process(mic_frame, ref_frame, update=True)
            # residual la tin hieu da khu echo (error signal e(n))
    """

    def __init__(self, config: NLMSConfig = NLMSConfig()) -> None:
        self.cfg = config
        L = config.filter_length

        self.w: np.ndarray = np.zeros(L, dtype=np.float64)
        self._history: np.ndarray = np.zeros(L - 1, dtype=np.float64)

        # Divergence detection: theo doi residual/mic ratio qua cac frame
        self._diverge_count: int = 0
        self._diverge_max: int = 5  # Reset sau 5 frame phan ky lien tiep

    def process(
        self,
        mic_frame: np.ndarray,
        ref_frame: np.ndarray,
        update: bool = True,
    ) -> np.ndarray:
        """Xu ly mot frame qua bo loc NLMS.

        Args:
            mic_frame: Tin hieu mic d(n), shape (N,). Chua echo + near-end.
            ref_frame: Tin hieu reference x(n), shape (N,). Tin hieu dang phat qua loa.
            update: True = cap nhat trong so (che do binh thuong).
                    False = dong bang trong so (khi co double-talk, DTD bat).

        Returns:
            e_frame: Residual signal e(n) = d(n) - w^T*x(n), shape (N,).
                     Khi hoi tu tot, e(n) chi con near-end speech, echo da bi khu.
        """
        N = len(mic_frame)
        L = self.cfg.filter_length
        mu = self.cfg.mu
        eps = self.cfg.eps

        # Chuyen sang float64 de dam bao do chinh xac so hoc.
        # Su dung float32 co the gay tich luy sai so sau nhieu lan cap nhat.
        mic = mic_frame.astype(np.float64)
        ref = ref_frame.astype(np.float64)
        e_frame = np.empty(N, dtype=np.float64)

        # Ghep lich su frame truoc voi frame hien tai.
        full_ref = np.concatenate([self._history, ref])

        # Luu L-1 mau cuoi cung lam lich su cho frame tiep theo
        self._history[:] = full_ref[-(L - 1):]

        # Tao ma tran cac cua so truot bang stride_tricks.
        X_mat = np.lib.stride_tricks.sliding_window_view(full_ref, L)[:, ::-1]

        for n in range(N):
            x_vec = X_mat[n]
            y_n = np.dot(self.w, x_vec)
            e_n = mic[n] - y_n
            e_frame[n] = e_n

            if update:
                norm = np.dot(x_vec, x_vec) + eps
                self.w += (mu / norm) * e_n * x_vec

        # Divergence detection: neu residual energy > mic energy →
        # NLMS dang lam xau tin hieu thay vi cai thien
        mic_energy = float(np.dot(mic, mic))
        res_energy = float(np.dot(e_frame, e_frame))

        if mic_energy > 1e-8 and res_energy > 1.5 * mic_energy:
            self._diverge_count += 1
            if self._diverge_count >= self._diverge_max:
                # Filter da phan ky nghiem trong → reset trong so
                self.w[:] = 0.0
                self._diverge_count = 0
                # Tra ve mic goc thay vi residual xau
                e_frame[:] = mic
        else:
            self._diverge_count = max(0, self._diverge_count - 1)

        return e_frame

    @property
    def weight_norm(self) -> float:
        """Chuan L2 cua vector trong so - dung de theo doi qua trinh hoi tu.
        Gia tri tang dan khi filter hoc RIR, on dinh khi da hoi tu."""
        return float(np.linalg.norm(self.w))

    def reset(self) -> None:
        """Dat lai toan bo trang thai. Goi khi bat dau phien moi."""
        self.w[:] = 0.0
        self._history[:] = 0.0
        self._diverge_count = 0