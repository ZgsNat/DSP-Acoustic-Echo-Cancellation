"""
Audio Processor — Bridge between capture queues and AEC pipeline.

Runs in its own thread. Reads mic + ref frames, optionally applies AEC,
then puts processed frames into send_queue for the network sender.

AEC can be toggled on/off at runtime without restarting the thread.
When toggled off, raw mic frames bypass the pipeline entirely.
"""

import queue
import threading
import time
import numpy as np
import sys
import os

# Adjust path so we can import core from desktop-app context
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from core.aec_pipeline import AECPipeline, AECConfig
from core.diagnostic_logger import DiagnosticLogger


class AudioProcessor:
    """
    Runs a background thread that:
        1. Reads mic_frame from mic_queue
        2. Reads ref_frame from ref_queue (what speaker is playing)
        3. If AEC enabled: passes both through AECPipeline
        4. Puts output frame into send_queue

    Metrics are collected and exposed for the UI to display.
    """

    def __init__(
        self,
        mic_queue:  queue.Queue,
        ref_queue:  queue.Queue,
        send_queue: queue.Queue,
        aec_config: AECConfig | None = None,
        diagnostic_path: str | None = None,
    ) -> None:
        self.mic_queue  = mic_queue
        self.ref_queue  = ref_queue
        self.send_queue = send_queue

        self._pipeline = AECPipeline(
            aec_config or AECConfig(),
            diagnostic_path=diagnostic_path,
        )
        self._aec_enabled = False  # Off by default — toggled by UI
        self._lock = threading.Lock()

        # Metrics shared with UI
        self.latest_metrics: dict = {}
        self._metrics_interval = 2.0  # Seconds between metric updates
        self._last_metrics_time = time.time()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AudioProcessor")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def set_aec_enabled(self, enabled: bool) -> None:
        """Thread-safe toggle. UI calls this on button click."""
        with self._lock:
            self._aec_enabled = enabled
            if enabled:
                # Reset pipeline state when re-enabling
                # (avoids stale filter state from a paused session)
                self._pipeline.reset()

    @property
    def aec_enabled(self) -> bool:
        with self._lock:
            return self._aec_enabled

    def _run(self) -> None:
        """Processing loop. Runs until stop() is called."""
        # Silence frame used when ref_queue is empty (speaker playing nothing)
        silence = np.zeros(1024, dtype=np.float32)

        while not self._stop_event.is_set():
            # Block waiting for mic frame (with timeout to check stop_event)
            try:
                mic_frame = self.mic_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Get reference frame — non-blocking, use silence if unavailable
            ref_is_silence = False
            try:
                ref_frame = self.ref_queue.get_nowait()
            except queue.Empty:
                ref_frame = silence
                ref_is_silence = True

            # Apply AEC or bypass
            with self._lock:
                aec_on = self._aec_enabled

            if aec_on:
                self._pipeline.mark_ref_silence(ref_is_silence)
                output_frame = self._pipeline.process(mic_frame, ref_frame)
            else:
                output_frame = mic_frame

            # Push processed frame to network sender
            try:
                self.send_queue.put_nowait(output_frame)
            except queue.Full:
                pass  # Drop frame rather than block (real-time constraint)

            # Periodically update metrics for UI display
            now = time.time()
            if aec_on and (now - self._last_metrics_time) >= self._metrics_interval:
                self.latest_metrics = self._pipeline.get_metrics()
                self._last_metrics_time = now

    def print_diagnostic_summary(self) -> None:
        """Print diagnostic summary. Call after stopping."""
        diag = self._pipeline.diagnostic_logger
        if diag is not None:
            diag.print_summary()
            diag.close()