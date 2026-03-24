# Desktop App — Planning
**Owner: Member 1 (Lead)**
**Language: Python**
**Goal: Two-machine LAN voice call with AEC toggle**

---

## Task List

- [ ] `audio/capture.py` — PyAudio mic capture + speaker playback
- [ ] `audio/processor.py` — Bridge between audio capture and AEC pipeline
- [ ] `network/sender.py` — UDP audio sender (encode + packetize)
- [ ] `network/receiver.py` — UDP audio receiver (depacketize + decode + play)
- [ ] `ui/app_window.py` — tkinter UI with AEC on/off toggle + ERLE display
- [ ] `main.py` — Wires everything together, CLI args for IP/port

## Architecture

```
[Capture Thread]          [Send Thread]           [Recv Thread]
MicCapture                AudioSender             AudioReceiver
    │                          │                       │
    │ mic_frame                │ encoded_bytes          │ decoded_frame
    ▼                          ▼                       ▼
AudioProcessor ──────► UDP Socket ─────► Network ─► Speaker playback
(AEC if enabled)
```

All threads communicate via thread-safe queues (queue.Queue).

## Key Technical Decisions

- **UDP not TCP**: Lower latency. 1-2% packet loss is acceptable for voice.
- **No compression**: Raw int16 PCM. ~256kbps but LAN bandwidth is not a constraint.
- **Sequence number in header**: Detect packet loss, discard out-of-order.
- **PyAudio callback mode**: Non-blocking audio I/O to avoid latency spikes.
- **Reference signal**: Captured from playback buffer BEFORE D/A, not from mic.
  This is critical — capturing from mic would include room acoustics.

## Dependencies

pip install pyaudio numpy scipy

## Acceptance Criteria

- Two instances of main.py on same/different LAN machines can talk
- Latency ≤ 150ms (network + buffer)
- AEC toggle works in real-time without restarting
- ERLE displayed in UI updates every ~2 seconds