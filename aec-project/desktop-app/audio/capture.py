"""
Audio Capture and Playback — PyAudio Interface

CRITICAL DESIGN: Reference signal is captured from playback queue BEFORE D/A
conversion, not from the microphone. This gives AEC a clean acoustic reference.

Threading model:
    PyAudio callback threads → thread-safe queues → processing thread
"""

import queue
import numpy as np
import pyaudio

SAMPLE_RATE = 16000
FRAME_SIZE  = 1024
CHANNELS    = 1
FORMAT      = pyaudio.paInt16
DTYPE       = np.int16
MAX_QUEUE   = 16   # frames (~1 second buffer)


class AudioCapture:
    """Non-blocking mic capture. Pushes float32 frames to mic_queue."""

    def __init__(self) -> None:
        self.mic_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUE)
        self._pa = pyaudio.PyAudio()
        self._stream = None

    def start(self) -> None:
        self._stream = self._pa.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
            input=True, frames_per_buffer=FRAME_SIZE,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        frame = np.frombuffer(in_data, dtype=DTYPE).astype(np.float32) / 32768.0
        try:
            self.mic_queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest to avoid unbounded growth
            try:
                self.mic_queue.get_nowait()
                self.mic_queue.put_nowait(frame)
            except queue.Empty:
                pass
        return (None, pyaudio.paContinue)

    def stop(self) -> None:
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()


class AudioPlayback:
    """
    Non-blocking speaker playback.

    For each frame played:
      1. Plays audio through speaker (D/A).
      2. Copies same frame to ref_queue BEFORE D/A — this is the AEC reference.

    When play_queue is empty (packet loss / jitter), plays silence.
    """

    def __init__(self) -> None:
        self.play_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUE)
        self.ref_queue:  queue.Queue[np.ndarray] = queue.Queue(maxsize=MAX_QUEUE)
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._silence = np.zeros(FRAME_SIZE, dtype=np.float32)

    def start(self) -> None:
        self._stream = self._pa.open(
            format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
            output=True, frames_per_buffer=FRAME_SIZE,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            frame = self.play_queue.get_nowait()
        except queue.Empty:
            frame = self._silence

        # Push to ref_queue BEFORE converting to PCM bytes
        # This is the pre-D/A reference for AEC
        try:
            self.ref_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # AEC degrades gracefully without latest ref frame

        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(DTYPE)
        return (pcm.tobytes(), pyaudio.paContinue)

    def stop(self) -> None:
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()