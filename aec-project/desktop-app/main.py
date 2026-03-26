"""
Desktop App — Entry Point

Wires together:
    AudioCapture → AudioProcessor (AEC) → AudioSender → [network] → AudioReceiver → AudioPlayback
    UI → AEC toggle → AudioProcessor

Usage:
    # Device A
    python main.py --local-port 5005 --peer-host 192.168.1.101 --peer-port 5005

    # Device B
    python main.py --local-port 5005 --peer-host 192.168.1.100 --peer-port 5005

Both devices run identical code. "Peer" is the other machine.
"""

import argparse
import queue
import signal
import sys
import os
import time
import threading

# Allow running from desktop-app/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audio.capture   import AudioCapture, AudioPlayback
from audio.processor import AudioProcessor
from network.sender  import AudioSender
from network.receiver import AudioReceiver
from ui.app_window   import AppWindow
from core.aec_pipeline import AECConfig


def parse_args():
    p = argparse.ArgumentParser(description="AEC Voice Call Demo")
    p.add_argument("--local-port",  type=int,   default=5005,
                   help="UDP port to listen on (default: 5005)")
    p.add_argument("--peer-host",   type=str,   required=True,
                   help="Peer IP address (e.g. 192.168.1.101)")
    p.add_argument("--peer-port",   type=int,   default=5005,
                   help="Peer UDP port (default: 5005)")
    p.add_argument("--no-ui",       action="store_true",
                   help="Headless mode (no tkinter, for testing)")
    p.add_argument("--filter-len",  type=int,   default=4096,
                   help="NLMS filter length (default: 4096)")
    p.add_argument("--mu",          type=float, default=0.7,
                   help="NLMS step size (default: 0.7)")
    p.add_argument("--diag",        type=str,   default=None,
                   help="Enable diagnostic logging to CSV file (e.g. --diag aec_debug.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Queues — all inter-thread communication
    # ------------------------------------------------------------------ #
    mic_queue   = queue.Queue(maxsize=16)   # Capture → Processor
    ref_queue   = queue.Queue(maxsize=16)   # Playback → Processor (AEC reference)
    send_queue  = queue.Queue(maxsize=16)   # Processor → Sender
    play_queue  = queue.Queue(maxsize=32)   # Receiver → Playback
    metrics_q   = queue.Queue(maxsize=8)    # Processor → UI

    # ------------------------------------------------------------------ #
    # Components
    # ------------------------------------------------------------------ #
    capture  = AudioCapture()
    playback = AudioPlayback()

    # Share mic_queue and ref_queue with their respective capture objects
    # AudioCapture.mic_queue and AudioPlayback.ref_queue are already the queues
    # we want, so we pass them directly to the processor below.

    aec_config = AECConfig(filter_length=args.filter_len, mu=args.mu)
    processor = AudioProcessor(
        mic_queue  = capture.mic_queue,
        ref_queue  = playback.ref_queue,
        send_queue = send_queue,
        aec_config = aec_config,
        diagnostic_path = args.diag,
    )

    sender   = AudioSender(send_queue, args.peer_host, args.peer_port)
    receiver = AudioReceiver("0.0.0.0", args.local_port, playback.play_queue)

    # Metrics relay: processor → metrics_q (for UI)
    # Run a small relay thread that polls processor.latest_metrics
    def _metrics_relay():
        while not _shutdown.is_set():
            if processor.latest_metrics:
                try:
                    metrics_q.put_nowait(processor.latest_metrics.copy())
                except queue.Full:
                    pass
            time.sleep(0.5)

    _shutdown = threading.Event()
    relay_thread = threading.Thread(target=_metrics_relay, daemon=True)

    # ------------------------------------------------------------------ #
    # Startup sequence
    # ------------------------------------------------------------------ #
    print(f"[AEC] Starting — local port: {args.local_port}, peer: {args.peer_host}:{args.peer_port}")
    print(f"[AEC] NLMS filter_length={args.filter_len}, mu={args.mu}")    
    if args.diag:
        print(f"[AEC] Diagnostic logging ENABLED \u2192 {args.diag}")
    receiver.start()   # Start listening before anything else
    playback.start()
    capture.start()
    processor.start()
    sender.start()
    relay_thread.start()

    print("[AEC] All components running. Waiting for peer...")

    # ------------------------------------------------------------------ #
    # UI (or headless loop)
    # ------------------------------------------------------------------ #
    def on_aec_toggle(enabled: bool):
        processor.set_aec_enabled(enabled)
        state = "ON" if enabled else "OFF"
        print(f"[AEC] AEC toggled {state}")

    def shutdown():
        print("\n[AEC] Shutting down...")
        _shutdown.set()
        sender.stop()
        receiver.stop()
        processor.stop()
        # Print diagnostic summary before closing audio devices
        processor.print_diagnostic_summary()
        capture.stop()
        playback.stop()
        print("[AEC] Done.")

    if args.no_ui:
        # Headless: just run until Ctrl+C
        print("[AEC] Headless mode. Press Ctrl+C to stop.")
        signal.signal(signal.SIGINT, lambda s, f: (shutdown(), sys.exit(0)))
        while True:
            time.sleep(1)
            if receiver.packets_received > 0:
                print(f"[AEC] RX: {receiver.packets_received} pkts | "
                      f"TX: {sender.packets_sent} pkts | "
                      f"Lost: {receiver.packets_lost} pkts")
    else:
        peer_str = f"{args.peer_host}:{args.peer_port}"
        window = AppWindow(
            peer_addr     = peer_str,
            local_port    = args.local_port,
            on_aec_toggle = on_aec_toggle,
            metrics_queue = metrics_q,
        )

        # Run UI on main thread (required by tkinter)
        try:
            window.run()
        finally:
            shutdown()


if __name__ == "__main__":
    main()