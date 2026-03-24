"""
Offline Metrics Evaluation

Computes ERLE and SNR from recorded WAV files.
Used by Member 3 to evaluate AEC performance in real room conditions.

Usage:
    python metrics_evaluation.py --mic mic.wav --ref ref.wav --output output.wav
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load WAV file as float32 normalized to [-1, 1]."""
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2**31
    return data.astype(np.float32), sr


def compute_erle(mic: np.ndarray, output: np.ndarray, frame_size: int = 1024) -> dict:
    """
    Compute ERLE frame-by-frame.
    Returns mean ERLE over all frames (excluding silence frames).
    """
    erle_frames = []
    for i in range(0, len(mic) - frame_size, frame_size):
        m = mic[i:i+frame_size]
        o = output[i:i+frame_size]
        mic_pow = np.mean(m ** 2)
        out_pow = np.mean(o ** 2)
        if mic_pow > 1e-8:  # Skip silence
            erle = 10.0 * np.log10(mic_pow / (out_pow + 1e-10))
            erle_frames.append(erle)

    if not erle_frames:
        return {"erle_mean": 0.0, "erle_max": 0.0, "erle_min": 0.0, "frames": 0}

    return {
        "erle_mean": float(np.mean(erle_frames)),
        "erle_max":  float(np.max(erle_frames)),
        "erle_min":  float(np.min(erle_frames)),
        "frames":    len(erle_frames),
    }


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """SNR in dB. signal = clean speech, noise = distortion."""
    sig_pow   = np.mean(signal ** 2)
    noise_pow = np.mean(noise ** 2)
    return 10.0 * np.log10(sig_pow / (noise_pow + 1e-10))


def run_aec_on_file(mic_path: str, ref_path: str) -> np.ndarray:
    """Run AEC pipeline on WAV files and return output array."""
    from core.aec_pipeline import AECPipeline, AECConfig

    mic, sr_m = load_wav(mic_path)
    ref, sr_r = load_wav(ref_path)

    assert sr_m == sr_r, f"Sample rate mismatch: mic={sr_m}, ref={sr_r}"

    # Trim to same length
    length = min(len(mic), len(ref))
    mic = mic[:length]
    ref = ref[:length]

    pipeline = AECPipeline(AECConfig())
    frame_size = 1024
    outputs = []

    for i in range(0, length - frame_size, frame_size):
        out = pipeline.process(mic[i:i+frame_size], ref[i:i+frame_size])
        outputs.append(out)

    return np.concatenate(outputs) if outputs else np.array([], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="AEC Offline Metrics Evaluation")
    parser.add_argument("--mic",    required=True, help="Mic recording (with echo)")
    parser.add_argument("--ref",    required=True, help="Reference (speaker) recording")
    parser.add_argument("--output", help="Optional: pre-processed output WAV (skip AEC)")
    args = parser.parse_args()

    mic, sr = load_wav(args.mic)
    print(f"Mic: {args.mic} | {len(mic)/sr:.1f}s | {sr}Hz")

    if args.output:
        output, _ = load_wav(args.output)
        print(f"Output: {args.output}")
    else:
        print("Running AEC pipeline on files...")
        output = run_aec_on_file(args.mic, args.ref)

    length = min(len(mic), len(output))
    mic    = mic[:length]
    output = output[:length]

    erle = compute_erle(mic, output)
    print("\n=== ERLE Results ===")
    print(f"  Mean ERLE : {erle['erle_mean']:.1f} dB")
    print(f"  Max  ERLE : {erle['erle_max']:.1f} dB")
    print(f"  Min  ERLE : {erle['erle_min']:.1f} dB")
    print(f"  Frames    : {erle['frames']}")

    if erle['erle_mean'] >= 15:
        print("\n✅ PASS — ERLE ≥ 15 dB (target achieved)")
    elif erle['erle_mean'] >= 8:
        print("\n⚠️  PARTIAL — ERLE 8–15 dB (acceptable for demo)")
    else:
        print("\n❌ FAIL — ERLE < 8 dB (check delay estimation + filter length)")


if __name__ == "__main__":
    main()