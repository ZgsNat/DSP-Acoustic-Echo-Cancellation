# Core DSP — Planning
**Owner: Member 1 (Lead)**
**Deliverable: Pure Python DSP modules, no external AEC libs**

---

## Task List

- [ ] `nlms_filter.py` — NLMS adaptive filter class
- [ ] `delay_estimator.py` — GCC-PHAT delay estimation
- [ ] `double_talk_detector.py` — Geigel DTD
- [ ] `nonlinear_suppressor.py` — Spectral subtraction post-filter
- [ ] `aec_pipeline.py` — Compose all blocks into single callable
- [ ] `tests/generate_synthetic_echo.py` — Test signal generator
- [ ] `tests/test_nlms.py` — Unit tests: ERLE ≥ 15dB on synthetic echo
- [ ] `tests/test_dtd.py` — Unit tests: DTD correctly freezes on double-talk

## Signal Contract

All modules operate on:
- numpy float32 arrays
- Sample rate: 16000 Hz
- Frame size: 1024 samples (64ms)
- Mono channel

## Acceptance Criteria

- ERLE ≥ 15 dB on synthetic echo (single-talk scenario)
- No crash on 60s continuous audio
- Double-talk: output SNR degradation < 3dB vs reference