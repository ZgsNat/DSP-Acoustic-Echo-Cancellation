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
                                                                    (Spectral Subtraction)
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
| `mu` | 0.1 | **0.7** | Hội tụ nhanh trên giọng nói. Kết quả: ERLE 48.5dB (chỉ có echo). |
| `eps` | $10^{-6}$ | $10^{-6}$ | Không thay đổi, chỉ để tránh chia cho 0. |

**Lý do tăng mu từ 0.1 lên 0.7**: Giọng nói là tín hiệu không dừng (non-stationary), thay đổi nhanh. $\mu$ nhỏ (0.1) hội tụ quá chậm, chưa kịp bắt kịp RIR thì giọng nói đã thay đổi. $\mu = 0.7$ hội tụ trong khoảng 50ms, đủ nhanh để bắt kịp mọi âm tiết.

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

Sau NLMS, vẫn còn echo dư (residual) do méo phi tuyến hoặc sai số ước lượng. Chúng tôi dùng phương pháp **Spectral Subtraction** kết hợp với **Overlap-Add (OLA)**.

### Công thức trừ phổ
$$|E_{clean}(f)|^2 = max(|E(f)|^2 - \alpha \cdot |Y_{smooth}(f)|^2, \beta \cdot |E(f)|^2)$$
- $\alpha$: Hệ số trừ quá mức (càng lớn triệt càng mạnh).
- $\beta$: Sàn phổ (spectral floor) để tránh nhiễu âm thanh (musical noise).

**Cửa sổ Overlap-Add (OLA)**: Sử dụng cửa sổ Hann và chồng lấp 50% giữa các khung hình để tránh hiện tượng tiếng lách tách (click/pop) ở biên mỗi khung.

---

## 8. Kết quả đo được

| Chỉ số | Mục tiêu | Đạt được | Đánh giá |
|--------|----------|----------|----------|
| ERLE (Chỉ có echo) | >= 15 dB | **48.5 dB** | Vượt xa mục tiêu |
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
Khi năng lượng đầu ra của bộ NLS lớn hơn năng lượng sau NLMS (do lỗi nhất thời của OLA), hệ thống tự động sử dụng kết quả từ NLMS để đảm bảo không bao giờ khuếch đại tín hiệu sai lệch.

---

## 10. Luồng toàn hệ thống (E2E Flow)

1. **Loa phát**: Âm thanh từ đầu xa được phát ra loa và đồng thời đưa vào bộ đệm tham chiếu.
2. **Mic thu**: Thu hỗn hợp giọng người nói gần và tiếng vọng.
3. **Căn chỉnh**: GCC-PHAT tìm độ trễ và căn khớp tín hiệu tham chiếu với mic.
4. **DTD**: Kiểm tra xem người dùng tại chỗ có đang nói không. Nếu có, dừng cập nhật bộ lọc.
5. **NLMS**: Ước lượng và trừ đi thành phần echo tuyến tính.
6. **NLS**: Triệt tiêu các thành phần phi tuyến và echo dư còn sót lại.
7. **Đầu ra**: Tín hiệu sạch được mã hóa và gửi qua mạng.

---

## 11. Ghi chú cho Demo

Kịch bản demo chuẩn:
1. Kết nối hai máy tính qua mạng LAN.
2. **Tắt AEC**: Người dùng A nói, người dùng B bật loa to -> A sẽ nghe thấy tiếng mình vọng lại cực kỳ khó chịu.
3. **Bật AEC**: Tiếng vọng biến mất ngay lập tức (sau khoảng 50-100ms hội tụ), giọng người dùng B vẫn rõ ràng.
4. **Nói đồng thời**: Cả hai cùng nói, kiểm tra xem tiếng vọng có bị lọt vào không và giọng nói có bị méo không.
5. Hiển thị thông số **ERLE** trên màn hình để minh chứng bằng con số cụ thể.