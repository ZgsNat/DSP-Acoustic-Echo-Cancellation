"""
test_loopback.py - Mo phong real-time AEC truc tiep tren 1 may

Muc dich:
    Test AEC pipeline trong dieu kien giong cuoc goi that (queue, timing,
    silence gaps, double-talk) ma KHONG can 2 laptop.

Cach chay:
    python test_loopback.py                       # Chay mac dinh, co report
    python test_loopback.py --diag loopback.csv   # Ghi CSV de phan tich sau
    python test_loopback.py --scenario all         # Chay tat ca scenarios

Scenarios:
    1. echo_only     - Far-end noi lien tuc, near-end im lang
    2. intermittent  - Far-end noi ngat quang (co silence gaps nhu speech that)
    3. double_talk   - Ca 2 dau noi cung luc
    4. ref_silence   - Ref queue empty (mo phong mat goi / jitter)
    5. all           - Chay tat ca scenarios tuan tu

Tin hieu:
    - Near-end & far-end: dung speech-like signal (AM-modulated noise, co
      pause ngau nhien giong speech pattern that).
    - RIR: Mo phong phong desktop (direct path + 4 reflections, RT60 ~200ms).
    - Noise: Additive white noise -40dB (giong mic noise that).

Output:
    - In ERLE summary va cac chi so quan trong
    - Neu co --diag: ghi CSV cho tung frame
"""

import argparse
import sys
import os
import time
import queue
import numpy as np
from dataclasses import dataclass

# Them duong dan project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.aec_pipeline import AECPipeline, AECConfig
from core.diagnostic_logger import DiagnosticLogger


# ===========================================================================
# Mo phong tin hieu
# ===========================================================================

SAMPLE_RATE = 16000
FRAME_SIZE = 1024


def generate_rir(length: int = 4096, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Tao Room Impulse Response mo phong phong desktop.

    Direct path + 4 early reflections + exponential decay tail.
    RT60 ~ 200ms, tuong duong phong 4x3m.
    """
    rir = np.zeros(length, dtype=np.float64)

    # Direct path (delay ~5ms = 80 samples o 16kHz)
    direct_delay = int(0.005 * sr)
    rir[direct_delay] = 1.0

    # Early reflections (tuong, san, tran)
    reflections = [
        (int(0.012 * sr), 0.6),   # tuong gan: 12ms, gain 0.6
        (int(0.020 * sr), 0.4),   # tuong xa:  20ms, gain 0.4
        (int(0.035 * sr), 0.25),  # tran:      35ms, gain 0.25
        (int(0.050 * sr), 0.15),  # san:       50ms, gain 0.15
    ]
    for delay, gain in reflections:
        if delay < length:
            rir[delay] = gain

    # Late reverb tail: exponential decay noise
    decay_start = int(0.060 * sr)
    if decay_start < length:
        t = np.arange(length - decay_start) / sr
        rt60 = 0.2  # 200ms
        decay = np.exp(-6.9 * t / rt60)  # -60dB at RT60
        np.random.seed(42)
        noise = np.random.randn(length - decay_start) * 0.05
        rir[decay_start:] = noise * decay

    # Normalize
    rir /= np.max(np.abs(rir)) + 1e-10
    return rir


def generate_speech_signal(duration_s: float, sr: int = SAMPLE_RATE,
                           pause_ratio: float = 0.3,
                           seed: int = 0) -> np.ndarray:
    """Tao tin hieu giong speech: AM-modulated noise voi cac khoang im lang.

    Args:
        duration_s: Tong thoi gian (giay)
        sr: Sample rate
        pause_ratio: Ti le thoi gian im lang (0.3 = 30% thoi gian la im)
        seed: Random seed de co the lap lai
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_s * sr)

    # Bandpass noise (300-3400Hz - dai tan speech)
    noise = rng.randn(n_samples)
    from scipy.signal import butter, lfilter
    b, h = butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype='bandpass') # type: ignore
    speech_noise = lfilter(b, h, noise)

    # AM modulation: tao envelope giong speech (3-8Hz)
    t = np.arange(n_samples) / sr
    f_am = 4.0 + rng.rand() * 4.0  # 4-8Hz
    envelope = 0.5 * (1 + np.sin(2 * np.pi * f_am * t + rng.rand() * 2 * np.pi))
    speech_noise *= envelope

    # Tao cac khoang im lang (pause) ngau nhien
    if pause_ratio > 0:
        segment_ms_range = (200, 800)  # Moi doan noi/im dai 200-800ms
        pos = 0
        is_speaking = True
        while pos < n_samples:
            seg_len = int(rng.randint(segment_ms_range[0], segment_ms_range[1]) * sr / 1000)
            seg_len = min(seg_len, n_samples - pos)
            if not is_speaking:
                speech_noise[pos:pos + seg_len] = 0.0
            pos += seg_len
            # Chuyen doi noi/im voi xac suat phu hop
            if is_speaking:
                is_speaking = rng.rand() > pause_ratio
            else:
                is_speaking = rng.rand() > 0.4  # 60% chance bat dau noi lai

    # Normalize to reasonable level (giong muc mic that ~ peak 0.3)
    peak = np.max(np.abs(speech_noise)) + 1e-10
    speech_noise = speech_noise * (0.3 / peak)

    return speech_noise.astype(np.float32)


def convolve_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolution tin hieu voi RIR de tao echo."""
    echo = np.convolve(signal, rir)[:len(signal)]
    return echo.astype(np.float32)


# ===========================================================================
# Scenarios
# ===========================================================================

@dataclass
class ScenarioResult:
    name: str
    duration_s: float
    total_frames: int
    erle_db: float
    dt_ratio: float
    max_filter_norm: float
    final_filter_norm: float
    avg_process_ms: float
    residual_to_mic_ratio: float  # <1 = good, >1 = AEC making it worse
    passed: bool
    notes: str = ""


def run_scenario(
    name: str,
    far_end: np.ndarray,     # Tin hieu far-end (reference goc)
    near_end: np.ndarray,    # Tin hieu near-end (nguoi noi phia mic)
    rir: np.ndarray,          # Room Impulse Response
    noise_level: float = 1e-4,  # Mic noise level (linear)
    ref_drop_rate: float = 0.0,  # Ti le frame mat ref (mo phong jitter)
    diag_path: str | None = None,
    config: AECConfig | None = None,
) -> ScenarioResult:
    """Chay 1 scenario, tra ve ket qua tong hop.

    Mo phong real-time:
    - Mic nhan: near_end + convolve(far_end, rir) + noise
    - Ref nhan: far_end (co the bi mat voi ref_drop_rate)
    - AEC xu ly frame-by-frame nhu trong desktop app
    """
    cfg = config or AECConfig()
    pipeline = AECPipeline(cfg, diagnostic_path=diag_path)

    n_samples = min(len(far_end), len(near_end))
    n_frames = n_samples // FRAME_SIZE

    # Tao echo bang convolution
    echo = convolve_rir(far_end[:n_samples], rir)

    # Cong mic signal: near_end + echo + noise
    rng = np.random.RandomState(99)
    noise = (rng.randn(n_samples) * noise_level).astype(np.float32)
    mic_full = near_end[:n_samples] + echo[:n_samples] + noise

    # Track metrics
    mic_powers = []
    out_powers = []
    filter_norms = []
    process_times = []
    ref_silence_count = 0

    for i in range(n_frames):
        start = i * FRAME_SIZE
        end = start + FRAME_SIZE

        mic_frame = mic_full[start:end].astype(np.float32)
        ref_frame = far_end[start:end].astype(np.float32)

        # Mo phong mat ref frame (jitter / network loss)
        if ref_drop_rate > 0 and rng.rand() < ref_drop_rate:
            ref_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
            pipeline.mark_ref_silence(True)
            ref_silence_count += 1
        else:
            pipeline.mark_ref_silence(False)

        # Process
        t0 = time.perf_counter()
        output = pipeline.process(mic_frame, ref_frame)
        t1 = time.perf_counter()
        process_times.append((t1 - t0) * 1000)

        # Metrics
        mic_p = float(np.mean(mic_frame.astype(np.float64) ** 2))
        out_p = float(np.mean(output.astype(np.float64) ** 2))
        mic_powers.append(mic_p)
        out_powers.append(out_p)
        filter_norms.append(pipeline._nlms.weight_norm)

    # Tinh ERLE toan cuc
    avg_mic_power = np.mean(mic_powers) if mic_powers else 1e-10
    avg_out_power = np.mean(out_powers) if out_powers else 1e-10
    erle_db = 10.0 * np.log10(avg_mic_power / (avg_out_power + 1e-10))

    # Residual/mic ratio (trung binh)
    ratios = [np.sqrt(o / (m + 1e-10)) for m, o in zip(mic_powers, out_powers)]
    avg_ratio = np.mean(ratios)

    # DT ratio
    metrics = pipeline.get_metrics()

    max_fnorm = max(filter_norms) if filter_norms else 0.0
    final_fnorm = filter_norms[-1] if filter_norms else 0.0

    # Dieu kien PASS:
    # 1. ERLE > 5dB (AEC co giam echo)
    # 2. Avg residual < mic (ratio < 1.0) — AEC khong lam xau hon
    # 3. Filter norm khong bung no (< 50)
    notes_parts = []
    passed = True

    if erle_db < 5.0:
        passed = False
        notes_parts.append(f"ERLE qua thap ({erle_db:.1f}dB < 5dB)")
    if avg_ratio > 1.0:
        passed = False
        notes_parts.append(f"AEC lam xau tin hieu (ratio={avg_ratio:.2f} > 1.0)")
    if max_fnorm > 50.0:
        passed = False
        notes_parts.append(f"Filter norm bung no (max={max_fnorm:.1f} > 50)")

    # Close diagnostic
    if diag_path:
        diag = pipeline.diagnostic_logger
        if diag:
            diag.close()

    return ScenarioResult(
        name=name,
        duration_s=n_samples / SAMPLE_RATE,
        total_frames=n_frames,
        erle_db=float(erle_db),
        dt_ratio=metrics.get("double_talk_ratio", 0.0),
        max_filter_norm=max_fnorm,
        final_filter_norm=final_fnorm,
        avg_process_ms=float(np.mean(process_times)),
        residual_to_mic_ratio=float(avg_ratio),
        passed=passed,
        notes="; ".join(notes_parts) if notes_parts else "OK",
    )


# ===========================================================================
# Cac scenario cu the
# ===========================================================================

def scenario_echo_only(diag_path: str | None = None) -> ScenarioResult:
    """Far-end noi lien tuc, near-end im lang. Kiem tra NLMS convergence."""
    print("\n[1/5] Echo Only - Far-end lien tuc, near-end im...")
    rir = generate_rir()
    far_end = generate_speech_signal(10.0, pause_ratio=0.0, seed=1)
    near_end = np.zeros_like(far_end)

    return run_scenario(
        name="echo_only",
        far_end=far_end, near_end=near_end, rir=rir,
        diag_path=diag_path,
    )


def scenario_intermittent(diag_path: str | None = None) -> ScenarioResult:
    """Far-end noi ngat quang (30% im). Kiem tra stability khi ref co gaps."""
    print("\n[2/5] Intermittent Far-end - Co silence gaps...")
    rir = generate_rir()
    far_end = generate_speech_signal(10.0, pause_ratio=0.3, seed=2)
    near_end = np.zeros_like(far_end)

    return run_scenario(
        name="intermittent",
        far_end=far_end, near_end=near_end, rir=rir,
        diag_path=diag_path,
    )


def scenario_double_talk(diag_path: str | None = None) -> ScenarioResult:
    """Ca 2 dau noi. Kiem tra DTD bao ve NLMS."""
    print("\n[3/5] Double Talk - Ca 2 dau noi cung luc...")
    rir = generate_rir()
    far_end = generate_speech_signal(10.0, pause_ratio=0.2, seed=3)
    near_end = generate_speech_signal(10.0, pause_ratio=0.3, seed=4)

    return run_scenario(
        name="double_talk",
        far_end=far_end, near_end=near_end, rir=rir,
        diag_path=diag_path,
    )


def scenario_ref_silence(diag_path: str | None = None) -> ScenarioResult:
    """30% frame mat ref (mo phong network jitter). Kiem tra NLMS khong bung."""
    print("\n[4/5] Ref Silence - 30% frame mat ref (jitter)...")
    rir = generate_rir()
    far_end = generate_speech_signal(10.0, pause_ratio=0.0, seed=5)
    near_end = np.zeros_like(far_end)

    return run_scenario(
        name="ref_silence",
        far_end=far_end, near_end=near_end, rir=rir,
        ref_drop_rate=0.3,
        diag_path=diag_path,
    )


def scenario_worst_case(diag_path: str | None = None) -> ScenarioResult:
    """Ket hop: double talk + 20% ref drop + noise cao. Stress test."""
    print("\n[5/5] Worst Case - Double talk + jitter + noise...")
    rir = generate_rir()
    far_end = generate_speech_signal(10.0, pause_ratio=0.2, seed=6)
    near_end = generate_speech_signal(10.0, pause_ratio=0.4, seed=7)

    return run_scenario(
        name="worst_case",
        far_end=far_end, near_end=near_end, rir=rir,
        noise_level=5e-4,   # Noise cao hon (~ -26dB)
        ref_drop_rate=0.2,
        diag_path=diag_path,
    )


# ===========================================================================
# Main
# ===========================================================================

SCENARIOS = {
    "echo_only": scenario_echo_only,
    "intermittent": scenario_intermittent,
    "double_talk": scenario_double_talk,
    "ref_silence": scenario_ref_silence,
    "worst_case": scenario_worst_case,
    "all": None,  # Special: chay tat ca
}


def print_results(results: list[ScenarioResult]) -> None:
    """In bang ket qua tong hop."""
    print("\n" + "=" * 80)
    print("KET QUA LOOPBACK TEST")
    print("=" * 80)
    print(f"{'Scenario':<16} {'ERLE(dB)':>9} {'DT%':>6} {'Ratio':>7} "
          f"{'FNorm':>8} {'MaxFN':>8} {'ms/fr':>7} {'Result':>8}")
    print("-" * 80)

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name:<16} {r.erle_db:>9.1f} {r.dt_ratio*100:>5.1f}% "
              f"{r.residual_to_mic_ratio:>7.3f} "
              f"{r.final_filter_norm:>8.2f} {r.max_filter_norm:>8.2f} "
              f"{r.avg_process_ms:>7.2f} {status:>8}")
        if not r.passed:
            all_passed = False
            print(f"  └─ {r.notes}")

    print("-" * 80)
    print(f"Tong: {sum(1 for r in results if r.passed)}/{len(results)} PASSED")

    if all_passed:
        print("\n✓ Tat ca scenarios deu PASS. AEC san sang test real-time.")
    else:
        print("\n✗ Co scenario FAIL. Can kiem tra truoc khi test real-time.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Loopback test - Mo phong AEC real-time tren 1 may")
    parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()),
        default="all",
        help="Chon scenario (mac dinh: all)")
    parser.add_argument(
        "--diag", "-d",
        default=None,
        help="Duong dan CSV de ghi diagnostic (vd: loopback.csv)")
    parser.add_argument(
        "--mu", type=float, default=None,
        help="Override mu (step size)")
    args = parser.parse_args()

    # Config overrides
    cfg = AECConfig()
    if args.mu is not None:
        cfg.mu = args.mu

    print("=" * 60)
    print("AEC LOOPBACK TEST")
    print(f"  Sample rate:    {SAMPLE_RATE} Hz")
    print(f"  Frame size:     {FRAME_SIZE} samples ({FRAME_SIZE/SAMPLE_RATE*1000:.0f}ms)")
    print(f"  Filter length:  {cfg.filter_length} taps ({cfg.filter_length/SAMPLE_RATE*1000:.0f}ms)")
    print(f"  mu (step size): {cfg.mu}")
    print("=" * 60)

    results = []

    if args.scenario == "all":
        for name, fn in SCENARIOS.items():
            if fn is not None:
                diag = f"{name}_{args.diag}" if args.diag else None
                results.append(fn(diag_path=diag))
    else:
        fn = SCENARIOS[args.scenario]
        results.append(fn(diag_path=args.diag))

    print_results(results)

    # Print diagnostic summary if available
    if args.diag and len(results) == 1:
        # Single scenario with diag — in chi tiet
        from core.diagnostic_logger import DiagnosticLogger
        logger = DiagnosticLogger(args.diag)
        # Logger da duoc close trong run_scenario, chi can print summary
        # Load lai tu CSV
        print("\nDiagnostic file saved to:", args.diag)


if __name__ == "__main__":
    main()
