"""
UDP Audio Sender

Packet format (12-byte header + PCM payload):
    [4 bytes] sequence_number  : uint32, wraps at 2^32
    [4 bytes] timestamp_ms     : uint32, milliseconds since epoch mod 2^32
    [4 bytes] payload_length   : uint32, bytes of PCM that follow
    [N bytes] PCM payload      : int16 little-endian samples

Why UDP?
    - TCP adds head-of-line blocking: a lost packet stalls all subsequent packets
    - For voice, a dropped 64ms frame is better than 200ms of frozen audio
    - LAN packet loss is typically < 0.1%, negligible for voice quality
"""

import socket
import struct
import queue
import threading
import time
import numpy as np

DTYPE = np.int16
HEADER_FORMAT = "!III"   # network byte order: seq, timestamp_ms, payload_len
HEADER_SIZE   = struct.calcsize(HEADER_FORMAT)   # 12 bytes


class AudioSender:
    """
    Background thread that reads float32 frames from send_queue
    and transmits them as UDP packets to (peer_host, peer_port).
    """

    def __init__(
        self,
        send_queue: queue.Queue,
        peer_host:  str,
        peer_port:  int,
    ) -> None:
        self.send_queue = send_queue
        self.peer_host  = peer_host
        self.peer_port  = peer_port

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Stats for UI
        self.packets_sent = 0

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AudioSender")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self.send_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._send_frame(frame)

    def _send_frame(self, frame: np.ndarray) -> None:
        """Encode frame as int16 PCM and send as UDP packet."""
        # float32 [-1, 1] → int16 [-32767, 32767]
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(DTYPE)
        payload = pcm.tobytes()

        timestamp_ms = int(time.time() * 1000) & 0xFFFFFFFF  # 32-bit wrap

        header = struct.pack(HEADER_FORMAT, self._seq, timestamp_ms, len(payload))
        packet = header + payload

        try:
            self._sock.sendto(packet, (self.peer_host, self.peer_port))
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            self.packets_sent += 1
        except OSError:
            pass  # Socket closed during shutdown