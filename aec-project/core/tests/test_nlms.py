"""
Unit Tests: NLMS Filter + Full AEC Pipeline

Run: python -m pytest core/tests/test_nlms.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pytest
from core.nlms_filter import NLMSFilter, NLMSConfig
from core.aec_pipeline import AECPipeline, AECConfig
from core.tests.generate_synthetic_echo import create_echo_scenario


SAMPLE_RATE = 16000
FRAME_SIZE = 1024


def compute_erle(mic: np.ndarray, output: np.ndarray) -> float:
    """ERLE in dB. Higher = better echo cancellation."""
    mic_power = np.mean(mic.astype(np.float64) ** 2)
    out_power = np.mean(output.astype(np.float64) ** 2)
    return 10.0 * np.log10(mic_power / (out_power + 1e-10))


class TestNLMSBasic:
    """Unit tests for the NLMS filter in isolation."""

    def test_output_shape(self):
        """Output must have same shape as input."""
        filt = NLMSFilter()
        mic = np.random.randn(FRAME_SIZE).astype(np.float32)
        ref = np.random.randn(FRAME_SIZE).astype(np.float32)
        out = filt.process(mic, ref)
        assert out.shape == mic.shape

    def test_identity_when_no_echo(self):
        """When mic = near-end only (no echo), output should remain close to mic."""
        filt = NLMSFilter()
        # Pure near-end speech, no correlated reference
        mic = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.1
        ref = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.01  # Very low reference
        out = filt.process(mic, ref)
        # Output should be approximately mic (not distorted)
        assert np.allclose(out, mic, atol=0.05), "Filter distorts near-end when no echo"

    def test_convergence_on_synthetic_echo(self):
        """
        NLMS must converge on a pure echo scenario.
        After ~3 seconds of adaptation, ERLE should exceed 10dB.
        """
        scenario = create_echo_scenario(duration_s=5.0, echo_delay_ms=5.0)
        ref = scenario["reference"]
        mic = scenario["mic_signal"]  # Pure echo, no near-end

        cfg = NLMSConfig(filter_length=512, mu=0.1)
        filt = NLMSFilter(cfg)

        outputs = []
        n = FRAME_SIZE
        # Skip first 2 seconds (adaptation period)
        skip_samples = 2 * SAMPLE_RATE

        for i in range(0, len(mic) - n, n):
            out = filt.process(mic[i:i+n], ref[i:i+n], update=True)
            if i >= skip_samples:
                outputs.append(out)

        if outputs:
            all_output = np.concatenate(outputs)
            all_mic = mic[skip_samples:skip_samples + len(all_output)]
            erle = compute_erle(all_mic, all_output)
            print(f"\nNLMS ERLE after 2s adaptation: {erle:.1f} dB")
            assert erle >= 10.0, f"NLMS ERLE too low: {erle:.1f} dB (expected ≥ 10 dB)"

    def test_reset_clears_state(self):
        """After reset(), filter should behave as if newly created."""
        filt = NLMSFilter()
        ref = np.random.randn(FRAME_SIZE).astype(np.float32)
        mic = ref * 0.5  # Simple scaled echo

        # Run for a while to build up weights
        for _ in range(50):
            filt.process(mic, ref)

        assert filt.weight_norm > 0.01, "Weights should have adapted"

        filt.reset()
        assert filt.weight_norm == 0.0, "Reset should zero weights"


class TestAECPipeline:
    """Integration tests for the full AEC pipeline."""

    def test_pipeline_runs_without_crash(self):
        """Pipeline must not crash on 5 seconds of random audio."""
        pipeline = AECPipeline()
        for _ in range(5 * SAMPLE_RATE // FRAME_SIZE):
            mic = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.1
            ref = np.random.randn(FRAME_SIZE).astype(np.float32) * 0.1
            out = pipeline.process(mic, ref)
            assert out.shape == (FRAME_SIZE,)

    def test_erle_on_synthetic_echo(self):
        """
        Full pipeline ERLE must be ≥ 15 dB on synthetic echo after convergence.
        This is the main acceptance criterion.
        """
        scenario = create_echo_scenario(
            duration_s=8.0,
            echo_delay_ms=20.0,      # 20ms delay (realistic buffer latency)
            echo_attenuation_db=6.0,  # Echo at -6dB vs reference
        )
        ref = scenario["reference"]
        mic = scenario["mic_signal"]

        pipeline = AECPipeline(AECConfig(filter_length=512, mu=0.1))

        outputs = []
        n = FRAME_SIZE
        warmup_samples = 3 * SAMPLE_RATE  # Skip first 3 seconds

        for i in range(0, len(mic) - n, n):
            out = pipeline.process(mic[i:i+n], ref[i:i+n])
            if i >= warmup_samples:
                outputs.append(out)

        if outputs:
            all_output = np.concatenate(outputs)
            all_mic = mic[warmup_samples:warmup_samples + len(all_output)]
            erle = compute_erle(all_mic, all_output)
            print(f"\nFull pipeline ERLE: {erle:.1f} dB")
            assert erle >= 15.0, f"Pipeline ERLE too low: {erle:.1f} dB (expected ≥ 15 dB)"

    def test_metrics_structure(self):
        """get_metrics() must return expected keys."""
        pipeline = AECPipeline()
        mic = np.random.randn(FRAME_SIZE).astype(np.float32)
        ref = np.random.randn(FRAME_SIZE).astype(np.float32)
        pipeline.process(mic, ref)
        metrics = pipeline.get_metrics()
        for key in ["erle_db", "double_talk_ratio", "frame_count", "delay_ms", "filter_norm"]:
            assert key in metrics, f"Missing metric: {key}"


if __name__ == "__main__":
    # Quick smoke test without pytest
    t = TestAECPipeline()
    t.test_erle_on_synthetic_echo()
    print("All smoke tests passed.")