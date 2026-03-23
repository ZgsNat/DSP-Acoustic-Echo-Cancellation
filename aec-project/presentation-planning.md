# AEC Project — Presentation & Professor Q&A Guide
**Tài liệu nội bộ nhóm — mọi thành viên phải đọc kỹ trước buổi bảo vệ**

---

## 1. Vấn đề là gì? (Problem Statement)

### Bối cảnh thực tế
Khi hai người dùng A và B gọi video/voice qua LAN:
- Loa của A phát âm thanh → phòng của B reflect âm thanh đó → Mic của B thu lại
- Mic B gửi cả giọng B **lẫn echo từ loa A** ngược về cho A
- Kết quả: A nghe thấy tiếng vọng của chính mình → trải nghiệm tệ, giao tiếp khó khăn

```
[Device A]                              [Device B]
   Loa A ──phát── âm thanh A
                                 B nghe được A
                         Mic B thu: giọng B + echo(âm A)
   A nghe echo                  ──gửi về──► A
   của chính mình ◄──────────────────────────
```

### Điều AEC cần làm
Trước khi Mic B gửi audio lên mạng, loại bỏ phần "echo từ loa A" ra khỏi tín hiệu — chỉ gửi giọng B sạch.

---

## 2. Tại sao khó? (Technical Challenges)

### 2.1 Room Impulse Response (RIR) — không biết và thay đổi liên tục
Âm thanh từ loa đến mic không phải là bản copy thẳng. Nó bị:
- **Phản xạ (reflection)**: dội lại từ tường, bàn, trần nhà
- **Hấp thụ (absorption)**: vật liệu mềm hấp thụ một phần
- **Nhiễu xạ (diffraction)**: âm thanh bẻ cong quanh vật cản

RIR được biểu diễn dưới dạng convolution:
```
mic_echo(n) = speaker(n) * h(n)   ← h(n) là RIR, dài 20–200ms
```

**Vấn đề**: h(n) là unknown và thay đổi khi người di chuyển → phải dùng adaptive filter.

### 2.2 Non-linearity
Loa và mic đều có distortion phi tuyến. Linear filter chỉ xử lý được ~70-80% echo, phần còn lại cần post-processing riêng.

### 2.3 Double-Talk
Khi cả hai người nói cùng lúc:
- NLMS không phân biệt được đâu là "echo cần xóa" vs "giọng người dùng cần giữ"
- Nếu không có DTD: filter sẽ cố gắng xóa luôn giọng người dùng → distortion

### 2.4 Delay không biết trước
Reference signal (từ loa) và mic signal không sync — có delay D do:
- Buffer latency (~20-50ms)
- Travel time từ loa đến mic (1ms/34cm)

Nếu không ước lượng D → NLMS sẽ không converge được.

---

## 3. Giải pháp: 3-Stage AEC Pipeline

```
Reference x(n) ──►[Delay Est. GCC-PHAT]──────────────────►┐
                                                            │ x(n-D) aligned
Mic d(n) ──────────────────────────────────────────────────►[NLMS Filter]──► e(n)
                                                            │              (residual)
                           Double-Talk Detector ────────────┘                │
                           (Geigel Algorithm)                                 │
                                                                              ▼
                                                                    [Nonlinear Suppressor]
                                                                    (Spectral Subtraction)
                                                                              │
                                                                              ▼
                                                                         Clean Output
```

---

## 4. Core Algorithm: NLMS Adaptive Filter

### Tại sao NLMS thay vì LMS?

LMS (Least Mean Squares) — update cơ bản:
```
w(n+1) = w(n) + μ · e(n) · x(n)
```
Vấn đề: μ cố định → nếu input power thay đổi (người nói to/nhỏ), filter sẽ:
- μ quá lớn: diverge
- μ quá nhỏ: converge chậm vô ích

NLMS (Normalized LMS) — normalize theo input power:
```
w(n+1) = w(n) + (μ / (||x(n)||² + ε)) · e(n) · x(n)
```
- `||x(n)||²` = sum của bình phương N samples gần nhất
- Khi signal mạnh: step size tự động giảm → stable
- Khi signal yếu: step size tự động tăng → responsive
- ε ≈ 1e-6 tránh chia 0 khi silence

### Cách hoạt động từng bước

```
Tại mỗi sample n:

1. Lấy x_vec(n) = [x(n), x(n-1), ..., x(n-L+1)]  ← L samples gần nhất từ reference
2. Tính echo estimate: y(n) = w(n)ᵀ · x_vec(n)
3. Tính error (residual): e(n) = d(n) - y(n)      ← d(n) là mic input
4. Tính norm: norm = x_vec(n)ᵀ · x_vec(n) + ε
5. Update weights: w(n+1) = w(n) + (μ / norm) · e(n) · x_vec(n)
```

### Tham số và trade-offs

| Tham số | Giá trị | Effect |
|---------|---------|--------|
| `filter_length` L | 512–1024 | Dài hơn = bắt echo xa hơn (reverb dài), nhưng tốn RAM, converge chậm hơn |
| `mu` μ | 0.1–0.3 | Lớn = converge nhanh nhưng steady-state error cao; nhỏ = ổn định hơn |
| `eps` ε | 1e-6 | Chỉ để tránh chia 0, không ảnh hưởng nhiều |

Tại sao L = 512 tại 16kHz? → 512/16000 = 32ms → đủ để bắt echo trong phòng thông thường (RIR thường < 30ms).

---

## 5. Double-Talk Detection: Geigel Algorithm

### Ý tưởng
Nếu mic signal **mạnh hơn** reference signal một ngưỡng threshold → khả năng cao là người dùng đang nói (double-talk).

```python
# Geigel DTD
if max(|d(n-k)|) > threshold * max(|x(n-k)|):
    double_talk = True  → freeze NLMS update (w không đổi)
else:
    double_talk = False → cho phép NLMS update bình thường
```

### Tại sao quan trọng
Khi double-talk xảy ra mà không có DTD:
- NLMS thấy `e(n)` lớn (vì có giọng người dùng thêm vào)
- NLMS nghĩ echo estimate sai → update mạnh → diverge
- Kết quả: giọng người dùng bị distort nặng

---

## 6. Delay Estimation: GCC-PHAT

### Tại sao cần
Reference signal x(n) (ghi lại từ loa) đến mic sớm hơn echo một khoảng D samples.
Nếu NLMS dùng x(n) không-aligned: filter window sẽ không bao giờ khớp với echo thực tế.

### GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
```python
# Tính cross-correlation trong frequency domain
X = FFT(x), D = FFT(d)
GCC = IFFT(X * conj(D) / |X * conj(D)|)  ← PHAT weighting
delay = argmax(GCC)
```

Sau khi có D: dùng `x(n-D)` làm reference thay vì `x(n)`.

---

## 7. Nonlinear Suppressor (Post-filter)

Sau NLMS, vẫn còn residual echo do:
- Non-linearity của loa/mic
- RIR estimation error

Spectral Subtraction:
```python
|E_clean(f)|² = max(|E(f)|² - α · |Y(f)|², β · |E(f)|²)
```
- α = over-subtraction factor (1.0–2.0)
- β = spectral floor (0.01–0.1) — tránh musical noise

---

## 8. Metrics: Làm sao biết AEC hoạt động tốt?

### ERLE — Echo Return Loss Enhancement (quan trọng nhất)
```python
ERLE = 10 * log10(E[d(n)²] / E[e(n)²])
```
- d(n) = mic signal (có echo)
- e(n) = output signal (sau AEC)
- **Giá trị tốt: 20–40 dB** trong production
- Project này target: ≥ 15 dB là acceptable

Hiểu đơn giản: ERLE = 20dB nghĩa là echo bị giảm 100 lần về power.

### SNR — Signal-to-Noise Ratio
```python
SNR = 10 * log10(P_signal / P_noise)
```
Kiểm tra xem giọng người dùng có bị distort không.

### PESQ — Perceptual Evaluation of Speech Quality
Đo chất lượng thoại cảm nhận theo chuẩn ITU-T P.862. Score 1–4.5.
Khó implement manually → dùng thư viện `pesq`.

---

## 9. Luồng toàn hệ thống (E2E Flow)

```
[Device B — gọi video]

Speaker B đang phát audio từ A:
  speaker_out(n) ──────────────────────────────────► Loa
                 └──► [Buffer] ──► AECPipeline.ref  ─────────┐

Mic B thu:                                                    │
  mic_in(n) ──────────────────────────────────────────────────┤
                                                              │
                    [AEC Pipeline khi bật]:                   │
                    ┌─────────────────────────────────────────┘
                    │
                    ├─► [Delay Estimator] → align reference
                    ├─► [DTD] → freeze/unfreeze NLMS
                    ├─► [NLMS Filter] → subtract echo estimate
                    └─► [NL Suppressor] → clean residual
                                │
                                ▼
                         clean_signal(n)
                                │
                         [UDP encode] ──► Network ──► Device A
```

---

## 10. Câu hỏi thầy có thể hỏi — và câu trả lời

**Q: Tại sao chọn NLMS mà không dùng RLS?**
A: RLS converge nhanh hơn và optimal về mặt toán học (minimize least squares toàn bộ history), nhưng complexity O(L²) mỗi sample vs NLMS là O(L). Với filter length 512 và real-time constraint, RLS quá nặng cho CPU. NLMS là tradeoff hợp lý nhất cho bài toán này.

**Q: Filter length bao nhiêu là đủ?**
A: Cần đủ để cover RIR của phòng. Ở 16kHz, 512 samples = 32ms. Phòng thông thường có reverberation time T60 ~ 200-500ms, nhưng echo chính (direct path + vài reflection đầu) thường trong 30-50ms. 512 là số hợp lý: đủ để cover, không quá nặng.

**Q: Khi double-talk, NLMS làm gì?**
A: Freeze. Weight vector w(n) không được update. Echo estimate giữ nguyên từ lần cuối không có double-talk. Đây là conservative choice — tốt hơn là distort giọng người dùng.

**Q: Non-linearity được xử lý như thế nào?**
A: Linear NLMS chỉ triệt được linear component của echo. Phần residual (do harmonic distortion của loa/mic) được xử lý bởi nonlinear suppressor — dùng spectral subtraction trên power spectrum của residual signal.

**Q: Làm sao test không cần hai thiết bị thật?**
A: Generate synthetic echo: lấy clean speech signal, convolve với RIR giả (FIR filter), add vào mic signal với SNR nhất định. ERLE đo trên synthetic echo cho kết quả reproducible.

**Q: Delay estimation sai thì sao?**
A: NLMS filter không converge. Reference window x_vec(n) không align với echo trong d(n) → error e(n) luôn lớn → weights oscillate không có nghĩa. Đây là lý do delay estimation là bước critical nhất, thường bị bỏ qua.

---

## 11. Limitations của Implementation

1. **Stationary echo only**: RIR thay đổi nhanh (người di chuyển mạnh) → NLMS cần thời gian re-converge
2. **No far-end noise suppression**: Chỉ xử lý echo, không suppress background noise từ loa
3. **Single-channel**: Không dùng microphone array → không có beam-forming
4. **Float32 processing**: Production systems dùng fixed-point để tiết kiệm CPU trên mobile

---

## 12. Ghi chú cho Demo

Kịch bản demo chuẩn:
1. Bật app Desktop A và Desktop B kết nối LAN
2. **AEC OFF**: B nói → A nghe thấy giọng B; bật loa A lớn → B nghe rõ tiếng vọng
3. **AEC ON trên B**: Bật toggle → echo biến mất, giọng B vẫn clear
4. Show ERLE metric tăng (từ ~0 dB lên ≥15 dB)
5. Demo double-talk: cả A và B cùng nói → NLMS freeze, không distort