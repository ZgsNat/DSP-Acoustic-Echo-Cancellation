"""
diagnostic_logger.py - Ghi chi tiet per-frame metrics de debug AEC real-time

Ghi ra file CSV voi moi dong la 1 frame, gom cac cot:
    frame_idx, timestamp, mic_rms, ref_rms, ref_is_silence,
    delay_samples, delay_ms, is_double_talk,
    nlms_residual_rms, echo_est_rms, nls_output_rms, nls_bypassed,
    erle_instant_db, filter_norm, process_time_ms

Sau khi ket thuc call, goi print_summary() de in bao cao tong hop.
"""

import csv
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class FrameMetrics:
    """Metrics cho 1 frame xu ly."""
    frame_idx: int = 0
    timestamp: float = 0.0

    # Input levels
    mic_rms: float = 0.0
    mic_peak: float = 0.0
    ref_rms: float = 0.0
    ref_peak: float = 0.0
    ref_is_silence: bool = False  # ref_queue was empty

    # Delay estimation
    delay_samples: int = 0
    delay_ms: float = 0.0

    # Double-talk
    is_double_talk: bool = False

    # NLMS output
    nlms_residual_rms: float = 0.0
    echo_est_rms: float = 0.0

    # NLS output
    nls_output_rms: float = 0.0
    nls_bypassed: bool = False

    # Overall
    erle_instant_db: float = 0.0
    filter_norm: float = 0.0
    process_time_ms: float = 0.0


CSV_COLUMNS = [
    "frame_idx", "timestamp",
    "mic_rms", "mic_peak", "ref_rms", "ref_peak", "ref_is_silence",
    "delay_samples", "delay_ms",
    "is_double_talk",
    "nlms_residual_rms", "echo_est_rms",
    "nls_output_rms", "nls_bypassed",
    "erle_instant_db", "filter_norm", "process_time_ms",
]


class DiagnosticLogger:
    """Ghi per-frame AEC metrics ra CSV file.

    Usage:
        logger = DiagnosticLogger("aec_debug.csv")
        # Trong vong lap xu ly:
        m = FrameMetrics(...)
        logger.log(m)
        # Ket thuc:
        logger.print_summary()
        logger.close()
    """

    def __init__(self, filepath: str = "aec_debug.csv") -> None:
        self._filepath = filepath
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._records: list[FrameMetrics] = []
        self._start_time = time.time()

        # Mo file va ghi header
        self._file = open(filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()

    def log(self, m: FrameMetrics) -> None:
        """Ghi 1 frame metrics."""
        if self._writer is None:
            return

        row = {
            "frame_idx":        m.frame_idx,
            "timestamp":        f"{m.timestamp:.4f}",
            "mic_rms":          f"{m.mic_rms:.8f}",
            "mic_peak":         f"{m.mic_peak:.6f}",
            "ref_rms":          f"{m.ref_rms:.8f}",
            "ref_peak":         f"{m.ref_peak:.6f}",
            "ref_is_silence":   int(m.ref_is_silence),
            "delay_samples":    m.delay_samples,
            "delay_ms":         f"{m.delay_ms:.2f}",
            "is_double_talk":   int(m.is_double_talk),
            "nlms_residual_rms": f"{m.nlms_residual_rms:.8f}",
            "echo_est_rms":     f"{m.echo_est_rms:.8f}",
            "nls_output_rms":   f"{m.nls_output_rms:.8f}",
            "nls_bypassed":     int(m.nls_bypassed),
            "erle_instant_db":  f"{m.erle_instant_db:.2f}",
            "filter_norm":      f"{m.filter_norm:.6f}",
            "process_time_ms":  f"{m.process_time_ms:.3f}",
        }
        self._writer.writerow(row)
        self._records.append(m)

        # Flush moi 50 frame (~3.2s o 1024@16kHz) de khong mat data neu crash
        if self._file and len(self._records) % 50 == 0:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    def print_summary(self) -> None:
        """In bao cao tong hop de debug."""
        if not self._records:
            print("[DiagLog] No frames recorded.")
            return

        n = len(self._records)
        duration = self._records[-1].timestamp - self._records[0].timestamp

        mic_rms_arr = np.array([m.mic_rms for m in self._records])
        ref_rms_arr = np.array([m.ref_rms for m in self._records])
        residual_rms_arr = np.array([m.nlms_residual_rms for m in self._records])
        nls_rms_arr = np.array([m.nls_output_rms for m in self._records])
        erle_arr = np.array([m.erle_instant_db for m in self._records])
        delay_arr = np.array([m.delay_samples for m in self._records])
        proc_time_arr = np.array([m.process_time_ms for m in self._records])

        ref_silence_count = sum(1 for m in self._records if m.ref_is_silence)
        dt_count = sum(1 for m in self._records if m.is_double_talk)
        nls_bypass_count = sum(1 for m in self._records if m.nls_bypassed)

        # Phan loai frame theo muc tin hieu
        active_mic = mic_rms_arr > 0.001  # mic co tin hieu
        active_ref = ref_rms_arr > 0.001  # ref co tin hieu

        print()
        print("=" * 70)
        print("          AEC DIAGNOSTIC SUMMARY")
        print("=" * 70)
        print(f"  File:         {self._filepath}")
        print(f"  Frames:       {n} ({duration:.1f}s)")
        print()

        # --- Input levels ---
        print("  [INPUT LEVELS]")
        print(f"    Mic  RMS:   mean={np.mean(mic_rms_arr):.6f}  "
              f"max={np.max(mic_rms_arr):.6f}  "
              f"active={np.sum(active_mic)}/{n} ({100*np.mean(active_mic):.0f}%)")
        print(f"    Ref  RMS:   mean={np.mean(ref_rms_arr):.6f}  "
              f"max={np.max(ref_rms_arr):.6f}  "
              f"active={np.sum(active_ref)}/{n} ({100*np.mean(active_ref):.0f}%)")
        print(f"    Ref EMPTY:  {ref_silence_count}/{n} "
              f"({100*ref_silence_count/n:.1f}%) ← ref_queue was empty")
        print()

        # --- Delay ---
        print("  [DELAY ESTIMATION]")
        unique_delays = np.unique(delay_arr)
        print(f"    Delay values seen: {unique_delays[:10]} samples")
        if len(delay_arr) > 0:
            final_delay = delay_arr[-1]
            print(f"    Final delay: {final_delay} samples "
                  f"({final_delay * 1000 / 16000:.1f}ms)")
            # Delay stability: how many times it changed
            changes = np.sum(np.diff(delay_arr) != 0)
            print(f"    Delay changes: {changes} times")
        print()

        # --- Double-talk ---
        print("  [DOUBLE-TALK DETECTOR]")
        print(f"    DT frames:  {dt_count}/{n} ({100*dt_count/n:.1f}%)")
        if dt_count > n * 0.5:
            print(f"    ⚠️  WARNING: DTD active >50% of time! "
                  f"NLMS barely updating → echo won't converge")
        print()

        # --- NLMS performance ---
        print("  [NLMS FILTER]")
        print(f"    Filter norm: {self._records[-1].filter_norm:.4f} "
              f"(0=not converged)")
        # ERLE only when both mic and ref are active
        both_active = active_mic & active_ref
        if np.any(both_active):
            erle_active = erle_arr[both_active]
            print(f"    ERLE (active): mean={np.mean(erle_active):.1f}dB  "
                  f"median={np.median(erle_active):.1f}dB  "
                  f"min={np.min(erle_active):.1f}dB  "
                  f"max={np.max(erle_active):.1f}dB")
        else:
            print(f"    ERLE: no frames with both mic+ref active")
        print()

        # --- NLS behavior ---
        print("  [NONLINEAR SUPPRESSOR]")
        print(f"    NLS bypassed: {nls_bypass_count}/{n} "
              f"({100*nls_bypass_count/n:.1f}%) ← safety clamp triggered")
        if nls_bypass_count > n * 0.3:
            print(f"    ⚠️  WARNING: NLS bypassed >30%! NLS may be creating artifacts")
        # Compare NLS output vs NLMS residual
        if np.any(both_active):
            nls_active = nls_rms_arr[both_active]
            res_active = residual_rms_arr[both_active]
            ratio = np.mean(nls_active) / (np.mean(res_active) + 1e-10)
            print(f"    NLS/NLMS ratio: {ratio:.3f} "
                  f"(<1 means NLS reducing, >1 means NLS amplifying)")
        print()

        # --- Processing time ---
        print("  [PERFORMANCE]")
        frame_budget_ms = 1024 / 16000 * 1000  # 64ms at 16kHz
        print(f"    Process time: mean={np.mean(proc_time_arr):.2f}ms  "
              f"max={np.max(proc_time_arr):.2f}ms  "
              f"budget={frame_budget_ms:.0f}ms")
        overruns = np.sum(proc_time_arr > frame_budget_ms)
        if overruns > 0:
            print(f"    ⚠️  WARNING: {overruns} frames exceeded real-time budget!")
        print()

        # --- Timeline: per-second summary ---
        print("  [TIMELINE — per second]")
        print(f"    {'Sec':>4s}  {'mic_rms':>8s}  {'ref_rms':>8s}  "
              f"{'ERLE_dB':>7s}  {'delay':>6s}  {'DT%':>5s}  {'ref_empty%':>10s}")
        fps = 16000 / 1024  # ~15.6 frames/sec
        sec = 0
        while sec * fps < n:
            s_idx = int(sec * fps)
            e_idx = min(int((sec + 1) * fps), n)
            if s_idx >= n:
                break

            chunk_mic = mic_rms_arr[s_idx:e_idx]
            chunk_ref = ref_rms_arr[s_idx:e_idx]
            chunk_erle = erle_arr[s_idx:e_idx]
            chunk_delay = delay_arr[s_idx:e_idx]
            chunk_dt = sum(1 for m in self._records[s_idx:e_idx] if m.is_double_talk)
            chunk_empty = sum(1 for m in self._records[s_idx:e_idx] if m.ref_is_silence)
            chunk_n = e_idx - s_idx

            print(f"    {sec:>4d}  {np.mean(chunk_mic):>8.5f}  "
                  f"{np.mean(chunk_ref):>8.5f}  "
                  f"{np.mean(chunk_erle):>7.1f}  "
                  f"{int(np.median(chunk_delay)):>6d}  "
                  f"{100*chunk_dt/chunk_n:>5.0f}  "
                  f"{100*chunk_empty/chunk_n:>10.0f}")
            sec += 1

        print("=" * 70)
        print(f"  CSV saved to: {os.path.abspath(self._filepath)}")
        print("=" * 70)
        print()
