"""
nonlinear_suppressor.py - Nen echo du bang Wiener Filter Gain voi OLA

Sau khi NLMS khu thanh phan echo tuyen tinh, van con echo du (residual) do:
  1. Me phi tuyen cua loa/mic (loa gia lam meo tin hieu, NLMS tuyen tinh
     khong mo hinh hoa duoc phan phi tuyen nay)
  2. Bo loc chua hoi tu het (misadjustment > 0)
  3. RIR thay doi khi nguoi dung di chuyen

Module nay dung Wiener Filter Gain de giam echo du:

    G(f) = clamp(1 - alpha * S_echo(f) / (S_residual(f) + eps), beta, 1.0)
    E_clean(f) = G(f) * E(f)

Trong do:
    E(f)        : Pho cua tin hieu sau NLMS (con echo du)
    S_echo(f)   : Cong suat echo uoc luong (lam min theo thoi gian)
    alpha       : He so over-subtraction. Lon hon -> nen manh hon.
    beta        : Gain floor (muc gain toi thieu). Mac dinh 0.005 = -46dB.
    G(f)        : Gain nhan (0..1), khong phai tru cong suat.

So sanh voi Spectral Subtraction (phien ban cu):
  Spectral Subtraction:  |E_clean|^2 = max(|E|^2 - alpha*|Y|^2, beta*|E|^2)
    → Bat/tat tung bin doc lap → musical noise (tieng re/twinkling)
    → Alpha cao = re nhieu, alpha thap = khong khu du echo

  Wiener Filter Gain:    E_clean = G(f) * E(f),  G muot theo thoi gian + tan so
    → Gain thay doi lien tuc, muot → KHONG musical noise
    → Asymmetric smoothing: nhanh khi echo xuat hien, cham khi echo mat
    → Co the dung alpha cao (2.5-4.0) ma khong re
    → Frequency smoothing ngan hieu ung "tonal" (tang/giam dot ngot 1 bin)

Pha cua E(f) duoc giu nguyen (chi thay doi bien do qua gain).

Overlap-Add (OLA):
  Hann window voi 50% overlap, window_size = 2 * frame_size.
  2 cua so Hann chong 50% cong lai = 1.0 (perfect reconstruction).
"""

import numpy as np


class NonlinearSuppressor:
    """Wiener Filter Gain voi Hann windowing va Overlap-Add.

    Nhan residual frame (sau NLMS) va echo estimate frame, tra ve
    tin hieu da nen echo du. Hoat dong trong mien tan so.

    Uu diem chinh so voi Spectral Subtraction:
      - Khong musical noise (tieng re) nho gain smoothing
      - Co the dung alpha cao ma van muot
      - Tu dong adapt: khi NLMS chua hoi tu (echo_est≈0) → gain≈1 (pass through)
                        khi NLMS da hoi tu (echo_est lon) → gain→beta (suppress manh)
      - Reference-based backup: dung tin hieu reference (loa) de uoc luong echo
        ngay tu frame dau tien, KHONG can doi NLMS hoi tu.
    """

    def __init__(
        self,
        frame_size: int = 1024,
        alpha: float = 2.5,
        beta: float = 0.005,
        smooth_alpha: float = 0.85,
    ) -> None:
        self.hop_size = frame_size
        self.window_size = frame_size * 2
        self.alpha = alpha
        self.beta = beta
        self.smooth_alpha = smooth_alpha

        self.window = np.hanning(self.window_size)

        self._e_buffer = np.zeros(self.window_size, dtype=np.float64)
        self._y_buffer = np.zeros(self.window_size, dtype=np.float64)
        self._r_buffer = np.zeros(self.window_size, dtype=np.float64)
        self._ola_buffer = np.zeros(self.window_size, dtype=np.float64)

        self._echo_power_smooth: np.ndarray | None = None
        self._ref_power_smooth: np.ndarray | None = None
        self._gain_smooth: np.ndarray | None = None

        # Uoc luong ti so echo_power / ref_power (adapt online)
        # Khoi tao = 0.5 (gia dinh echo ~ 50% ref power, hop ly cho hau het phong)
        self._echo_ref_ratio: float = 0.5

    def process(
        self,
        residual_frame: np.ndarray,
        echo_estimate_frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
        is_double_talk: bool = False,
    ) -> np.ndarray:
        """Nen echo du cho 1 frame bang Wiener Filter Gain.

        Args:
            residual_frame: Tin hieu sau NLMS, shape (hop_size,)
            echo_estimate_frame: Uoc luong echo tu NLMS, shape (hop_size,)
            ref_frame: Tin hieu reference (loa), shape (hop_size,), optional.
                       Khi co: dung lam backup echo estimate cho frame dau
                       (truoc khi NLMS hoi tu). Tu dong adapt ratio online.
            is_double_talk: True khi DTD phat hien co near-end speech.
                       Khi True: tat ref-based backup va tang gain floor
                       de bao ve near-end speech khong bi suppress.

        Returns:
            out_frame: Tin hieu da nen echo du, shape (hop_size,), float32
        """
        assert len(residual_frame) == self.hop_size

        e = residual_frame.astype(np.float64)
        y = echo_estimate_frame.astype(np.float64)

        # Truot buffer: bo nua cu (ben trai), them frame moi (ben phai)
        self._e_buffer[:-self.hop_size] = self._e_buffer[self.hop_size:]
        self._e_buffer[-self.hop_size:] = e
        self._y_buffer[:-self.hop_size] = self._y_buffer[self.hop_size:]
        self._y_buffer[-self.hop_size:] = y

        # ── Bypass NLS hoan toan khi double-talk ──
        # Ly do: NLMS (dong bang) van khu echo tuyen tinh tot (filter da hoi tu).
        # Residual = near_end + residual_echo_nho. NLS khong can thiet.
        # Van de: OLA overlap tu frame truoc (da suppress) se lam mat dau beep.
        # Giai phap: tra residual truc tiep, cap nhat state de post-DT muot.
        if is_double_talk:
            # Cap nhat echo power tracking (Y tu NLMS van dung khi filter dong bang)
            y_win = self._y_buffer * self.window
            Y = np.fft.rfft(y_win)
            Y_power = np.abs(Y) ** 2
            if self._echo_power_smooth is None:
                self._echo_power_smooth = Y_power.copy()
            else:
                self._echo_power_smooth = (
                    self.smooth_alpha * self._echo_power_smooth
                    + (1.0 - self.smooth_alpha) * Y_power
                )

            # Cap nhat ref buffer (giu dong bo cho post-DT)
            if ref_frame is not None:
                r = ref_frame.astype(np.float64)
                self._r_buffer[:-self.hop_size] = self._r_buffer[self.hop_size:]
                self._r_buffer[-self.hop_size:] = r
                r_win = self._r_buffer * self.window
                R = np.fft.rfft(r_win)
                R_power = np.abs(R) ** 2
                if self._ref_power_smooth is None:
                    self._ref_power_smooth = R_power.copy()
                else:
                    self._ref_power_smooth = (
                        self.smooth_alpha * self._ref_power_smooth
                        + (1.0 - self.smooth_alpha) * R_power
                    )
                # KHONG cap nhat echo_ref_ratio khi DT (NLMS dong bang, ratio khong chinh xac)

            # Xoa OLA buffer de echo da suppress khong tran sang post-DT
            self._ola_buffer[:] = 0.0
            # Reset gain smooth de post-DT dung instant gain (khong stale)
            self._gain_smooth = None

            return residual_frame.astype(np.float32)

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

        # ── Reference-based echo power (backup cho khi NLMS chua hoi tu) ──
        # Y dua vao NLMS echo estimate → khi NLMS chua hoi tu, Y≈0 → NLS khong
        # suppress duoc gi. Giai phap: dung ref_power * ratio lam backup.
        echo_power_effective = self._echo_power_smooth
        if ref_frame is not None:
            r = ref_frame.astype(np.float64)
            self._r_buffer[:-self.hop_size] = self._r_buffer[self.hop_size:]
            self._r_buffer[-self.hop_size:] = r
            r_win = self._r_buffer * self.window
            R = np.fft.rfft(r_win)
            R_power = np.abs(R) ** 2

            if self._ref_power_smooth is None:
                self._ref_power_smooth = R_power.copy()
            else:
                self._ref_power_smooth = (
                    self.smooth_alpha * self._ref_power_smooth
                    + (1.0 - self.smooth_alpha) * R_power
                )

            # Uoc luong echo power tu reference: echo ≈ ratio * ref
            echo_power_from_ref = self._echo_ref_ratio * self._ref_power_smooth

            # Lay MAX cua NLMS-based va ref-based → luon co echo estimate
            echo_power_effective = np.maximum(
                self._echo_power_smooth,
                echo_power_from_ref,
            )

            # Adapt ratio: khi NLMS cho echo estimate tot (Y_power lon),
            # cap nhat ratio = Y_power / R_power de chinh xac hon
            total_y = float(np.sum(Y_power))
            total_r = float(np.sum(self._ref_power_smooth))
            if total_r > 1e-6 and total_y > 1e-6:
                current_ratio = np.clip(total_y / total_r, 0.01, 5.0)
                self._echo_ref_ratio = (
                    0.95 * self._echo_ref_ratio + 0.05 * current_ratio
                )

        # ── Wiener Filter Gain ──
        # G(f) = 1 - alpha * echo_power_effective / (residual_power + eps)
        # echo_power_effective = max(NLMS echo est, ref-based echo est)
        # → Suppress echo ngay tu frame dau tien (nho ref-based backup)
        # Khi DT: chi dung NLMS echo estimate (ref-based da tat)
        # → Wiener filter tu phan biet: echo bins (gain thap) vs near-end bins (gain cao)
        eps = 1e-10
        gain = 1.0 - self.alpha * echo_power_effective / (E_power + eps)
        gain = np.clip(gain, self.beta, 1.0)

        # ── Temporal gain smoothing (asymmetric) ──
        # Fast attack: gain giam nhanh khi echo xuat hien (phan ung nhanh)
        # Slow release: gain tang cham khi echo mat (tranh gain pumping / musical noise)
        # Luu y: DT da bypass toan bo truoc do → code nay chi chay khi KHONG co DT.
        if self._gain_smooth is None:
            self._gain_smooth = gain.copy()
        else:
            ATTACK = 0.1   # gain giam: 90% tu gain moi → ~1 frame (~64ms)
            RELEASE = 0.85  # gain tang: 15% tu gain moi → ~7 frame (~450ms)
            alpha_t = np.where(gain < self._gain_smooth, ATTACK, RELEASE)
            self._gain_smooth = alpha_t * self._gain_smooth + (1.0 - alpha_t) * gain

        # ── Frequency smoothing (3-bin weighted average) ──
        # Ngan chan gain nhay dot ngot giua cac bin ke nhau → tonal artifact
        g = self._gain_smooth
        g_smooth = g.copy()
        if len(g) > 2:
            g_smooth[1:-1] = 0.25 * g[:-2] + 0.5 * g[1:-1] + 0.25 * g[2:]
        g_smooth = np.clip(g_smooth, self.beta, 1.0)

        # Ap dung gain vao pho (nhan, khong tru → muot hon)
        # Pha duoc giu nguyen vi gain la so thuc
        E_clean = E * g_smooth

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
        self._ref_power_smooth = None
        self._gain_smooth = None
        self._echo_ref_ratio = 0.5
        self._e_buffer[:] = 0.0
        self._y_buffer[:] = 0.0
        self._r_buffer[:] = 0.0
        self._ola_buffer[:] = 0.0