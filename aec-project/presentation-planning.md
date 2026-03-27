# Dự án AEC - Tài liệu chuẩn bị bảo vệ
**Mọi thành viên phải đọc kỹ trước buổi bảo vệ**

---

## 1. Vấn đề là gì? (Problem Statement)

### Bối cảnh thực tế
Khi hai người dùng A và B gọi video/voice qua LAN:
- Loa của A phát âm thanh -> phòng của B phản xạ (reflect) âm thanh đó -> Mic của B thu lại.
- Mic B gửi cả giọng B lẫn tiếng vọng (echo) từ loa A ngược về cho A.
- Kết quả: A nghe thấy tiếng vọng của chính mình -> trải nghiệm tệ.

```
[Thiết bị A]                                 [Thiết bị B]
   Loa A -- phát -- âm thanh A
                                     B nghe được A
                         Mic B thu: giọng B + echo(âm A)
   A nghe echo                  -- gửi về --> A
   của chính mình <---------------------------------
```

### Điều AEC cần làm
Trước khi Mic B gửi âm thanh (audio) lên mạng, loại bỏ phần "echo từ loa A" ra khỏi tín hiệu -- chỉ gửi giọng B sạch.

---

## 2. Tại sao khó? (Technical Challenges)

### 2.1 Đáp ứng xung phòng (Room Impulse Response - RIR) - không biết và thay đổi liên tục
Âm thanh từ loa đến mic không phải bản sao thẳng. Nó bị:
- **Phản xạ (reflection)**: dội lại từ tường, bàn, trần nhà.
- **Hấp thụ (absorption)**: vật liệu mềm hấp thụ một phần.
- **Nhiễu xạ (diffraction)**: âm thanh bẻ cong quanh vật cản.

RIR được biểu diễn dưới dạng tích chập (convolution):
$$mic\_echo(n) = speaker(n) * h(n)$$
*(Trong đó $h(n)$ là RIR, dài khoảng 20-250ms)*

Vấn đề: $h(n)$ là ẩn số (unknown) và thay đổi khi người di chuyển -> phải dùng bộ lọc thích nghi (adaptive filter).

### 2.2 Tính phi tuyến (Non-linearity)
Loa và mic đều có độ méo (distortion) phi tuyến. Bộ lọc tuyến tính (linear filter) chỉ xử lý được khoảng 70-80% echo, phần còn lại cần hậu xử lý riêng (NLS).

### 2.3 Nói đồng thời (Double-Talk)
Khi cả hai người nói cùng lúc:
- NLMS không phân biệt được đâu là "echo cần xóa" so với "giọng người dùng cần giữ".
- Nếu không có bộ phát hiện Double-Talk (DTD): bộ lọc sẽ cố gắng xóa luôn giọng người dùng -> gây méo tiếng (distortion).

### 2.4 Trễ (Delay) không biết trước
Tín hiệu tham chiếu (từ loa) và tín hiệu mic không đồng bộ -- có độ trễ $D$ do:
- Trễ bộ đệm (Buffer latency) (~20-50ms).
- Thời gian truyền âm từ loa đến mic (1ms/34cm).

Nếu không ước lượng được $D$ -> NLMS sẽ không thể hội tụ (converge).

---

## 3. Giải pháp: Quy trình (Pipeline) AEC 4 khối

```
Tham chiếu x(n) -->[Ước lượng trễ GCC-PHAT]---------------->|
                                                            | x(n-D) đã khớp
Mic d(n) ---------------------------------------------------->[Bộ lọc NLMS]--> e(n)
                                                            |               (dư ảnh)
                           Bộ phát hiện Double-Talk ---------+                |
                           (Thuật toán Geigel)                                |
                                                                              v
                                                                    [Bộ triệt phi tuyến]
                                                                    (Wiener Filter Gain)
                                                                              |
                                                                              v
                                                                         Đầu ra sạch
```

---

## 4. Thuật toán cốt lõi: Bộ lọc thích nghi NLMS

### Tại sao chọn NLMS thay vì LMS?

**LMS (Least Mean Squares)** - cập nhật cơ bản:
$$w(n+1) = w(n) + \mu \cdot e(n) \cdot x(n)$$
Vấn đề: hệ số $\mu$ cố định -> nếu năng lượng đầu vào thay đổi (người nói to/nhỏ), bộ lọc sẽ:
- $\mu$ quá lớn: bị phân kỳ (diverge).
- $\mu$ quá nhỏ: hội tụ chậm vô ích.

**NLMS (Normalized LMS)** - chuẩn hóa theo năng lượng đầu vào:
$$w(n+1) = w(n) + \left( \frac{\mu}{\|x(n)\|^2 + \epsilon} \right) \cdot e(n) \cdot x(n)$$
- $\|x(n)\|^2$: tổng bình phương $N$ mẫu gần nhất.
- Khi tín hiệu mạnh: bước nhảy (step size) tự động giảm -> ổn định.
- Khi tín hiệu yếu: bước nhảy tự động tăng -> nhạy bén.
- $\epsilon = 10^{-6}$: tránh chia cho 0 khi im lặng.

### Cách hoạt động từng bước
Tại mỗi mẫu $n$:
1. Lấy $x\_vec(n) = [x(n), x(n-1), ..., x(n-L+1)]$ (L mẫu gần nhất từ tham chiếu).
2. Tính toán echo ước lượng: $y(n) = w(n)^T \cdot x\_vec(n)$.
3. Tính sai số (residual): $e(n) = d(n) - y(n)$ (với $d(n)$ là đầu vào mic).
4. Tính chuẩn (norm): $norm = x\_vec(n)^T \cdot x\_vec(n) + \epsilon$.
5. Cập nhật trọng số: $w(n+1) = w(n) + (\mu / norm) \cdot e(n) \cdot x\_vec(n)$.

### Tham số và kết quả tinh chỉnh

| Tham số | Giá trị mặc định | Giá trị đã tinh chỉnh | Ghi chú |
|---------|-------------------|----------------------|---------|
| `filter_length` L | 512 (32ms) | **4096 (256ms)** | Phủ hết RIR dài 250ms. 512 quá ngắn, bỏ sót echo xa. |
| `mu` | 0.1 | **0.5** (real-time) / **0.7** (offline test) | 0.5 cân bằng hội tụ nhanh và ổn định cho real-time. 0.7 cho offline test đạt ERLE 48.5dB. |
| `eps` | $10^{-6}$ | $10^{-6}$ | Không thay đổi, chỉ để tránh chia cho 0. |

**Lý do tăng mu từ 0.1 lên 0.5-0.7**: Giọng nói là tín hiệu không dừng (non-stationary), thay đổi nhanh. $\mu$ nhỏ (0.1) hội tụ quá chậm, chưa kịp bắt kịp RIR thì giọng nói đã thay đổi. $\mu = 0.5$ (real-time) hội tụ trong khoảng 80ms, đủ nhanh và ổn định cho cuộc gọi thực. $\mu = 0.7$ (offline) hội tụ nhanh hơn (~50ms) nhưng steady-state error cao hơn, phù hợp cho demo ngắn.

**Tối ưu hóa (stride_tricks)**: Thay vì dùng vòng lặp `for` chạy từng mẫu (chậm trong Python), chúng tôi dùng `np.lib.stride_tricks.as_strided` để tạo ma trận Toeplitz rồi nhân ma trận. Tốc độ tăng gấp ~50 lần, đủ nhanh cho xử lý thời gian thực ở 16kHz.

---

## 5. Bộ phát hiện Double-Talk (DTD): Thuật toán Geigel

### Ý tưởng
Nếu tín hiệu mic mạnh hơn tín hiệu tham chiếu một ngưỡng (threshold) nhất định -> khả năng cao là người dùng đang nói (double-talk).

```
Nếu max(|mic_history|) > threshold * max(|ref_history|):
    double_talk = True  -> Đóng băng cập nhật NLMS (giữ nguyên w)
Ngược lại:
    double_talk = False -> Cho phép NLMS cập nhật bình thường
```

### Các vấn đề đã gặp và cách khắc phục
1. **Dương tính giả khi im lặng**: Khi cả mic và tham chiếu ở mức nhiễu nền (~$10^{-6}$), tỉ số đỉnh không có ý nghĩa.
   - **Sửa**: Thêm `MIN_LEVEL = 10^{-4}`. Nếu cả hai nhỏ hơn mức này thì luôn trả về False.
2. **Dương tính giả khi tham chiếu vừa tắt**: Echo vẫn còn dội lại (RIR dài ~200ms) nhưng bộ đệm tham chiếu ngắn đã "quên" mức năng lượng cũ.
   - **Sửa**: `ref_buf` tự động tính chiều dài theo `echo_tail_ms = 300ms`.
3. **Nhấp nháy DTD**: Giọng nói người dùng có khoảng nghỉ ngắn.
   - **Sửa**: Dùng `hangover = 100ms` để giữ trạng thái DTD thêm một khoảng thời gian.

---

## 6. Ước lượng trễ (Delay Estimation): GCC-PHAT

### Tại sao cần?
Tín hiệu tham chiếu $x(n)$ (ghi lại từ loa) đến mic sớm hơn echo một khoảng $D$ mẫu. Nếu NLMS dùng $x(n)$ không được căn chỉnh (align), cửa sổ bộ lọc sẽ không bao giờ khớp với echo thực tế.

### GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
$$R = FFT(x) \cdot conj(FFT(d))$$
$$R_{phat} = \frac{R}{|R|}$$
$$gcc(\tau) = IFFT(R_{phat})$$
$$delay = argmax(gcc(\tau))$$

Trọng số PHAT làm cho đỉnh (peak) nhọn hơn so với tương quan chéo thông thường, giúp kết quả chính xác hơn.

---

## 7. Bộ triệt phi tuyến (Nonlinear Suppressor - Post-filter)

Sau NLMS, vẫn còn echo dư (residual) do méo phi tuyến hoặc sai số ước lượng. Chúng tôi dùng phương pháp **Wiener Filter Gain** kết hợp với **Overlap-Add (OLA)**.

### Tại sao đổi từ Spectral Subtraction sang Wiener Filter Gain?
Spectral Subtraction ban đầu gây **musical noise** (tiếng rè, tiếng kim loại) do các bin tần số dao động giữa 0 và giá trị lớn. Wiener Filter Gain khắc phục bằng cách dùng hệ số gain mượt thay vì trừ phổ trực tiếp.

### Công thức Wiener Filter Gain
$$G(f) = max\left(1 - \alpha \cdot \frac{|Echo_{smooth}(f)|^2}{|E(f)|^2 + \epsilon}, \beta\right)$$
$$E_{clean}(f) = G(f) \cdot E(f)$$
- $\alpha = 2.5$: Hệ số over-subtraction (triệt mạnh hơn echo ước lượng).
- $\beta = 0.005$: Gain floor = $-46$dB, NLS có thể suppress thêm ~46dB trên NLMS.
- $G(f)$ luôn trong khoảng $[\beta, 1]$ → **chỉ giảm, không bao giờ khuếch đại** → không gây musical noise.

### Kỹ thuật làm mượt Gain
- **Gain smoothing bất đối xứng**: Attack nhanh ($\alpha_{attack} = 0.1$), release chậm ($\alpha_{release} = 0.85$). Echo xuất hiện → suppress ngay lập tức; echo biến mất → mở gain từ từ, tránh tiếng lách tách.
- **Frequency smoothing**: Trung bình có trọng số 3-bin liền kề, tránh gain nhảy giữa hai bin cạnh nhau.
- **Double-talk bypass**: Khi DTD phát hiện double-talk, NLS **bypass hoàn toàn** (trả về residual từ NLMS trực tiếp, không qua OLA). Buffer OLA được reset để tránh tràn năng lượng khi chuyển trạng thái.

**Cửa sổ Overlap-Add (OLA)**: Sử dụng cửa sổ Hann và chồng lấp 50% giữa các khung hình để tránh hiện tượng tiếng lách tách (click/pop) ở biên mỗi khung.

---

## 8. Kết quả đo được

| Chỉ số | Mục tiêu | Đạt được | Đánh giá |
|--------|----------|----------|----------|
| ERLE (Chỉ có echo, offline) | >= 15 dB | **48.5 dB** (mu=0.7) | Vượt xa mục tiêu |
| ERLE (Echo-only, real-time pipeline) | >= 15 dB | ~**40+ dB** (mu=0.5) | Đủ tốt cho cuộc gọi |
| Triệt echo (Khi nói đồng thời) | >= 10 dB | **26.0 dB** | Tốt |
| Độ méo giọng người dùng | < 3 dB | **0.6 dB** | Rất tốt |

---

## 9. Câu hỏi thường gặp khi bảo vệ

### "Tại sao echo không thể bằng 0 tuyệt đối?"
3 lý do chính:
1. **Bộ lọc hữu hạn**: NLMS dùng L=4096, nhưng RIR thực tế có thể dài hơn hoặc có thành phần ngoài cửa sổ.
2. **Sai số hội tụ (Misadjustment)**: NLMS hội tụ về điểm cân bằng, không phải tối ưu tuyệt đối.
3. **Đặc tính giọng nói**: Giọng nói không phải nhiễu trắng (white noise). Các tần số có năng lượng khác nhau nên tốc độ hội tụ không đều.

### "Tại sao dùng NLMS mà không dùng RLS?"
- RLS (Recursive Least Squares) hội tụ nhanh hơn nhưng độ phức tạp là $O(L^2)$. Với L=4096, RLS cần ~16 triệu phép tính mỗi mẫu, không khả thi cho thời gian thực.
- NLMS có độ phức tạp $O(L)$, kết hợp với hậu xử lý NLS là đã đủ tốt.

### "Chốt chặn an toàn (Safety clamp) hoạt động như thế nào?"
Khi năng lượng đầu ra sau NLS lớn hơn năng lượng **mic gốc** (do OLA tràn năng lượng từ frame trước hoặc NLMS phân kỳ sau double-talk), hệ thống tự động scale output xuống bằng mức mic. Nguyên tắc: pipeline **chỉ suppress (giảm), không bao giờ amplify (khuếch đại)** so với tín hiệu mic đầu vào.

---

## 10. Luồng toàn hệ thống (E2E Flow)

### 10.1 Kiến trúc đa luồng (Threading Architecture)
Mỗi Node (máy tính) chạy 5 luồng độc lập, giao tiếp qua hàng đợi (Queue):

```
Luồng 1: UDP Receiver ──→ play_queue ──→ Luồng 2: Playback Callback ──→ Loa
                                              │
                                              └──→ ref_queue (tham chiếu cho AEC)
                                                        │
Luồng 3: Mic Capture Callback ──→ mic_queue ──→ Luồng 4: AEC Processor ──→ send_queue ──→ Luồng 5: UDP Sender
```

### 10.2 Xử lý tín hiệu tham chiếu (Reference Signal Flow)
Điểm then chốt: Playback callback và Mic callback là hai PyAudio stream độc lập. Do scheduling jitter, số lượng ref frame và mic frame có thể không khớp 1:1 tại mỗi chu kỳ xử lý.

**Giải pháp: Intermediate Reference Feeding**
- Processor thu thập **tất cả** ref frames có trong queue (thường 1, đôi khi 0 hoặc 2+).
- Nếu có nhiều hơn 1 ref frame: các frame trung gian (intermediate) được **nạp qua `feed_reference()`** vào DelayLine (ring buffer) và NLMS history, đảm bảo chuỗi reference liên tục.
- Frame cuối cùng được dùng để chạy `pipeline.process()` cùng mic frame.
- Nếu không có ref frame nào: dùng **ref frame gần nhất** (không dùng silence, vì echo từ loa vẫn đang dội trong phòng).

**Tại sao cần thiết?** Nếu bỏ qua ref frame trung gian:
- DelayLine (ring buffer) có lỗ hổng → `ref_aligned` sai lệch.
- NLMS `_history` mất mẫu → tương quan chéo bị phá → bộ lọc **không thể hội tụ** → echo hoàn toàn không bị triệt.

### 10.3 Pipeline xử lý tín hiệu (mỗi frame)
1. **Loa phát**: Âm thanh từ đầu xa được phát ra loa. Playback callback đồng thời đẩy bản sao vào `ref_queue` (tham chiếu trước D/A, sạch hơn thu lại từ mic).
2. **Mic thu**: Thu hỗn hợp giọng người nói gần và tiếng vọng.
3. **Căn chỉnh**: GCC-PHAT tìm độ trễ $D$ và căn khớp tín hiệu tham chiếu với mic qua DelayLine (ring buffer).
4. **DTD**: Kiểm tra xem người dùng tại chỗ có đang nói không. Nếu có, đóng băng cập nhật bộ lọc NLMS.
5. **NLMS**: Ước lượng và trừ đi thành phần echo tuyến tính.
6. **NLS (Wiener Filter Gain)**: Triệt tiêu các thành phần phi tuyến và echo dư còn sót lại. Bypass hoàn toàn khi double-talk.
7. **Safety clamp**: Đảm bảo output không to hơn mic gốc.
8. **Đầu ra**: Tín hiệu sạch được mã hóa int16 và gửi qua UDP tới Node kia.

---

## 11. Ghi chú cho Demo

Kịch bản demo chuẩn:
1. Kết nối hai máy tính qua mạng LAN.
2. **Tắt AEC**: Người dùng A nói, người dùng B bật loa to -> A sẽ nghe thấy tiếng mình vọng lại cực kỳ khó chịu.
3. **Bật AEC**: Tiếng vọng biến mất ngay lập tức (sau khoảng 50-100ms hội tụ), giọng người dùng B vẫn rõ ràng.
4. **Nói đồng thời**: Cả hai cùng nói, kiểm tra xem tiếng vọng có bị lọt vào không và giọng nói có bị méo không.
5. Hiển thị thông số **ERLE** trên màn hình để minh chứng bằng con số cụ thể.