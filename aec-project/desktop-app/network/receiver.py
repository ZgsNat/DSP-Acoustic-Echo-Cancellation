"""
UDP Audio Receiver

Listens on a UDP port, parses incoming audio packets, and puts decoded
float32 frames into play_queue for AudioPlayback to consume.

Handles:
    - Packet parsing (header + PCM payload)
    - Out-of-order detection (sequence number tracking)
    - Silence insertion for lost packets (jitter concealment)
"""

import socket
import struct
import queue
import threading
import numpy as np

DTYPE         = np.int16
HEADER_FORMAT = "!III"
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)
FRAME_SIZE    = 1024
MAX_PACKET    = HEADER_SIZE + FRAME_SIZE * 2 + 64  # bytes, with headroom


class AudioReceiver:
    """
    Background thread that listens on (host, port) for UDP audio packets
    and pushes decoded float32 frames to play_queue.
    """

    def __init__(
        self,
        host:       str,
        port:       int,
        play_queue: queue.Queue,
    ) -> None:
        self.host       = host
        self.port       = port
        self.play_queue = play_queue

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)  # Allow checking stop_event

        self._expected_seq = None  # None = haven't seen first packet yet
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Stats
        self.packets_received = 0
        self.packets_lost     = 0

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AudioReceiver")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()

    def _run(self) -> None:
        silence = np.zeros(FRAME_SIZE, dtype=np.float32)

        while not self._stop_event.is_set():
            try:
                data, addr = self._sock.recvfrom(MAX_PACKET)
            except socket.timeout:
                continue
            except OSError:
                break

            frame = self._parse_packet(data, silence)
            if frame is not None:
                try:
                    self.play_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Playback can't keep up — drop

    def _parse_packet(
        self, data: bytes, silence: np.ndarray
    ) -> np.ndarray | None:
        """
        Parse UDP packet → float32 frame.
        Returns silence frame if packet is malformed or out of order.
        Returns None if header parse fails entirely.
        """
        if len(data) < HEADER_SIZE:
            return None

        seq, timestamp_ms, payload_len = struct.unpack(
            HEADER_FORMAT, data[:HEADER_SIZE]
        )

        payload = data[HEADER_SIZE:]
        if len(payload) != payload_len:
            return None  # Truncated packet

        # Sequence number gap detection
        if self._expected_seq is None:
            self._expected_seq = seq  # First packet, no gap possible

        gap = (seq - self._expected_seq) & 0xFFFFFFFF
        if gap > 0 and gap < 100:
            # Sequence gap: some packets were lost
            # Insert silence frames for each lost packet (basic PLC)
            self.packets_lost += gap
            for _ in range(min(gap, 4)):  # Cap at 4 silence frames
                try:
                    self.play_queue.put_nowait(silence.copy())
                except queue.Full:
                    break

        self._expected_seq = (seq + 1) & 0xFFFFFFFF
        self.packets_received += 1

        # Decode int16 PCM → float32
        pcm = np.frombuffer(payload, dtype=DTYPE).astype(np.float32) / 32768.0
        return pcm