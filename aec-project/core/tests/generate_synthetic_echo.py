"""
Synthetic Echo Generator for Testing

Creates a controlled test scenario:
    - Clean speech signal (sine sweeps or loaded WAV)
    - Synthetic Room Impulse Response (short FIR filter)
    - Echo = clean_speech convolved with RIR, added to mic with controlled SNR
    - Optional: Add near-end speech for double-talk testing
"""

import numpy as np


def generate_rir(
    sample_rate: int = 16000,
    room_size_ms: float = 30.0,
    decay_factor: float = 0.7,
) -> np.ndarray:
    """Synthetic Room Impulse Response as short FIR filter."""
    length = int(room_size_ms * sample_rate / 1000)
    h = np.zeros(length, dtype=np.float64)

    # Direct path
    direct_delay = int(0.5 * sample_rate / 1000)
    h[direct_delay] = 1.0

    # Early reflections with exponential decay
    reflection_times_ms = [5.0, 12.0, 20.0, 28.0]
    rng = np.random.default_rng(0)
    for i, t_ms in enumerate(reflection_times_ms):
        tap = int(t_ms * sample_rate / 1000)
        if tap < length:
            amp = (decay_factor ** (i + 1)) * (0.5 + 0.5 * rng.random())
            h[tap] = amp

    h /= (np.max(np.abs(h)) + 1e-10)
    return h


def generate_speech_like_signal(
    duration_s: float = 5.0,
    sample_rate: int = 16000,
    seed: int = 42,
) -> np.ndarray:
    """Bandpass filtered noise with syllable-rate envelope. Approximates speech spectrum."""
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    noise = rng.standard_normal(n).astype(np.float64)
    b, a = butter(4, [300 / (sample_rate / 2), 3400 / (sample_rate / 2)], btype='band') # type: ignore
    speech = lfilter(b, a, noise)
    t = np.arange(n) / sample_rate
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    speech *= envelope
    speech /= (np.max(np.abs(speech)) + 1e-10)
    speech *= 0.5
    return speech.astype(np.float32)


def create_echo_scenario(
    duration_s: float = 5.0,
    sample_rate: int = 16000,
    echo_delay_ms: float = 20.0,
    echo_attenuation_db: float = 6.0,
    near_end_db: float = None, # type: ignore
) -> dict:
    """
    Create complete synthetic echo test scenario.

    Returns dict: reference, mic_signal, echo_only, rir, near_end, delay_samples
    """
    far_end = generate_speech_like_signal(duration_s, sample_rate, seed=1).astype(np.float64)
    rir = generate_rir(sample_rate)

    delay_samples = int(echo_delay_ms * sample_rate / 1000)
    echo = np.convolve(far_end, rir, mode='full')[:len(far_end)]
    echo = np.roll(echo, delay_samples)
    echo[:delay_samples] = 0.0

    attenuation = 10 ** (-echo_attenuation_db / 20.0)
    echo *= attenuation

    mic = echo.copy()
    near_end = None
    if near_end_db is not None:
        near_end = generate_speech_like_signal(duration_s, sample_rate, seed=99).astype(np.float64)
        near_end_amp = 10 ** (near_end_db / 20.0) * np.std(echo)
        near_end = near_end / (np.std(near_end) + 1e-10) * near_end_amp
        mic = mic + near_end

    return {
        "reference": far_end.astype(np.float32),
        "mic_signal": mic.astype(np.float32),
        "near_end": near_end.astype(np.float32) if near_end is not None else None,
        "rir": rir,
        "echo_only": echo.astype(np.float32),
        "delay_samples": delay_samples,
    }