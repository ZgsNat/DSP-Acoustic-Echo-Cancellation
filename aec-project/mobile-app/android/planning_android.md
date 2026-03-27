# Android AEC Planning (Java + Reuse core)

**Owner: Member 2 (Mobile)**  
**Goal: Xây app Android thoại thời gian thực, dùng lại core AEC hiện có để khử echo**

---

## 1) Mục tiêu kỹ thuật

- Nền tảng: Android app viết Java.
- Luồng thoại: thu mic -> xử lý AEC -> gửi UDP; nhận UDP -> phát loa.
- AEC engine: dùng lại `aec-project/core/` (NLMS + DelayEstimator + DTD + NonlinearSuppressor), không dùng thư viện AEC đóng.
- Audio contract thống nhất với core:
	- Sample rate: 16000 Hz
	- Mono
	- Frame size: 1024 samples
	- Dữ liệu nội bộ: float32 trong đoạn [-1, 1]

---

## 2) Kiến trúc tích hợp core vào Android

### Phương án chọn (khuyến nghị): Java + Python bridge (Chaquopy)

Lý do:
- Core đã viết bằng Python, có thể tái sử dụng ngay logic AEC.
- Giảm rủi ro sai khác thuật toán khi port sang Java.
- Dễ đồng bộ kết quả ERLE với desktop.

Kiến trúc runtime:
1. Java Audio Thread đọc frame mic và ref.
2. Java gọi Python bridge `AECPipeline.process(mic, ref)` theo frame.
3. Python trả frame clean về Java.
4. Java gửi frame clean qua UDP.

Module mới cần tạo trong Android app:
- `AecBridge.java`: API Java gọi sang Python.
- `AudioEngine.java`: điều phối capture/playback/processor.
- `UdpSender.java`, `UdpReceiver.java`: socket tương thích desktop header 12 byte.
- `python/aec_android_bridge.py`: wrapper cho `core.aec_pipeline`.

### Phương án dự phòng: Port core sang Java

Chỉ dùng khi bridge Python không đạt hiệu năng/độ ổn định.
Port theo đúng contract và công thức từ core để giữ cùng hành vi.

---

## 3) Kế hoạch triển khai theo giai đoạn

## Phase 0 - Chuẩn bị project (0.5 ngày)

- [ ] Cập nhật Gradle để hỗ trợ Python bridge (Chaquopy).
- [ ] Tạo cấu trúc package:
	- `app/src/main/java/com/example/dsp/audio/`
	- `app/src/main/java/com/example/dsp/network/`
	- `app/src/main/java/com/example/dsp/aec/`
	- `app/src/main/python/`
- [ ] Chuẩn hóa constants dùng chung: sample rate, frame size, packet header.

Deliverable:
- App build được sau khi thêm plugin/dependency.

## Phase 1 - Raw audio call không AEC (1 ngày)

- [ ] Implement `AudioRecord` capture 16kHz mono, frame 1024.
- [ ] Implement `AudioTrack` playback 16kHz mono.
- [ ] Implement UDP sender/receiver theo format desktop:
	- Header `!III`: seq, timestamp_ms, payload_len
	- Payload: PCM int16 little-endian
- [ ] UI tối thiểu:
	- nhập peer IP/port
	- Start/Stop Call
	- trạng thái network (sent/recv/loss)

Deliverable:
- Android <-> Desktop truyền thoại 2 chiều được, chưa AEC.

## Phase 2 - Tích hợp core AEC qua bridge (1.5 ngày)

- [ ] Copy hoặc mount `core/` vào runtime Python của app.
- [ ] Viết `aec_android_bridge.py`:
	- init pipeline với `AECConfig`
	- hàm `process_frame(mic_bytes_or_list, ref_bytes_or_list)`
	- hàm `get_metrics()` và `reset()`
- [ ] Viết `AecBridge.java`:
	- lifecycle: init/reset/release
	- convert `short[] <-> float[]`
	- gọi Python theo frame
- [ ] Thêm cờ `aecEnabled` trong luồng xử lý:
	- ON: mic -> AEC -> send
	- OFF: mic raw -> send

Deliverable:
- Android chạy call với AEC bật/tắt được realtime.

## Phase 3 - Đồng bộ reference và timing (1 ngày)

- [ ] Bắt reference từ playback buffer trước D/A (bắt buộc cho AEC).
- [ ] Tạo queue `refQueue` song song với `playQueue` (giống desktop design).
- [ ] Xử lý underflow:
	- thiếu ref frame -> fill zero frame
	- queue full -> drop frame cũ, ưu tiên real-time
- [ ] Tách thread an toàn:
	- Audio I/O thread
	- AEC processing thread
	- Network thread

Deliverable:
- Echo giảm rõ, không giật audio kéo dài khi mạng dao động nhẹ.

## Phase 4 - Đo lường, tuning, hardening (1 ngày)

- [ ] Thu metrics từ core:
	- ERLE dB
	- delay_ms
	- double_talk_ratio
- [ ] Logging ring-buffer (không log quá dày trên UI thread).
- [ ] Tuning tham số AECConfig cho mobile:
	- `filter_length`, `mu`, `dtd_threshold`, `nls_alpha/beta`
- [ ] Kiểm thử 3 kịch bản:
	- near-end only
	- far-end only (đo ERLE)
	- double-talk

Deliverable:
- Đạt tiêu chí acceptance (mục 5).

---

## 4) Mapping công việc vào code hiện tại

File Android hiện có cần mở rộng:
- `DSP/app/src/main/java/com/example/dsp/MainActivity.java`
	- thêm UI control Start/Stop + AEC toggle + metrics.
- `DSP/app/src/main/res/layout/activity_main.xml`
	- thêm input IP/port, nút điều khiển, vùng hiển thị stats.

File mới đề xuất:
- `DSP/app/src/main/java/com/example/dsp/audio/AudioCapture.java`
- `DSP/app/src/main/java/com/example/dsp/audio/AudioPlayback.java`
- `DSP/app/src/main/java/com/example/dsp/audio/AudioProcessor.java`
- `DSP/app/src/main/java/com/example/dsp/network/UdpSender.java`
- `DSP/app/src/main/java/com/example/dsp/network/UdpReceiver.java`
- `DSP/app/src/main/java/com/example/dsp/aec/AecBridge.java`
- `DSP/app/src/main/python/aec_android_bridge.py`

---

## 5) Acceptance criteria (Android)

- [ ] Call 2 chiều Android <-> Desktop chạy liên tục 10 phút không crash.
- [ ] AEC ON làm giảm echo nghe thấy rõ rệt so với OFF.
- [ ] Trong far-end only: ERLE trung bình >= 12 dB trên Android (mục tiêu stretch: >= 15 dB).
- [ ] Double-talk: giọng near-end không bị méo nặng (SNR degradation < 3 dB so baseline).
- [ ] Độ trễ end-to-end chấp nhận được cho demo (< 300 ms LAN).

---

## 6) Rủi ro và phương án giảm thiểu

- Rủi ro: Python bridge gây overhead CPU.
	- Giảm thiểu: xử lý theo frame lớn 1024, tránh copy mảng thừa, ưu tiên mảng primitive.
- Rủi ro: lệch clock capture/playback làm drift.
	- Giảm thiểu: queue giới hạn, chèn zero frame khi thiếu, bỏ frame khi tràn.
- Rủi ro: thiết bị Android có AEC phần cứng mặc định.
	- Giảm thiểu: tắt AcousticEchoCanceler của OS khi chạy AEC core để tránh "double AEC".
- Rủi ro: khác định dạng PCM với desktop.
	- Giảm thiểu: giữ đúng int16 little-endian + header 12 bytes như desktop.

---

## 7) Timeline đề xuất (4-5 ngày)

- Day 1: Phase 0 + Phase 1 (raw call).
- Day 2: Phase 2 (bridge core vào Android).
- Day 3: Phase 3 (reference/timing/thread safety).
- Day 4: Phase 4 (metrics + tuning + test).
- Day 5: buffer fix bug + chuẩn bị demo.

---

## 8) Definition of Done

- [ ] Demo trực tiếp Android gọi với Desktop, bật/tắt AEC realtime.
- [ ] Có log metrics ERLE/delay/double-talk cho phần trình bày.
- [ ] Code Android tách module rõ: audio, network, aec, ui.
- [ ] Kịch bản demo đã chạy thử tối thiểu 3 lần liên tiếp.
