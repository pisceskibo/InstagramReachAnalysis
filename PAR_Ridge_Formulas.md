# Công thức toán học: PAR và PAR + Ridge Regularization

## 1. Mô hình hồi quy tuyến tính

Cho vector đặc trưng $x_t$ = [Likes, Saves, Comments, Shares, ProfileVisits, Follows] và biến mục tiêu $y_t$ = Impressions.

Dự đoán:

$$\hat{y_t} = w_t^T x_t + b$$

trong đó:
- $x_t$: vector đặc trưng của mẫu thứ $t$
- $y_t$: giá trị thực tế
- $\hat{y_t}$: giá trị dự đoán
- $w_t$: vector trọng số tại thời điểm $t$
- $b$: hệ số chặn (bias)

---

## 2. Passive Aggressive Regressor (PAR)

### 2.1. Sai số và hàm mất mát epsilon-insensitive

Sai số:

$$e_t = y_t - \hat{y_t}$$

Hàm mất mát trên một mẫu:

$$L_t(w, x_t, y_t) = \max(0, |y_t - \hat{y_t}| - \epsilon)$$

### 2.2. Cơ chế Passive và Aggressive

- Nếu $|y_t - \hat{y_t}| \leq \epsilon$ thì $L_t = 0$ → **Passive** (không cập nhật trọng số)
- Nếu $|y_t - \hat{y_t}| > \epsilon$ thì $L_t > 0$ → **Aggressive** (cập nhật trọng số)

### 2.3. Bài toán tối ưu tại mỗi bước (PA-II)

$$w_{t+1} = \argmin_w \frac{1}{2} \|w - w_t\|^2 + C \cdot L_t(w, x_t, y_t)$$

### 2.4. Nghiệm: hệ số cập nhật

$$\tau_t = \frac{L_t(w_t, x_t, y_t)}{\|x_t\|^2 + \dfrac{1}{2C}}$$

### 2.5. Quy tắc cập nhật trọng số

$$w_{t+1} = w_t + \tau_t \cdot \mathrm{sgn}(y_t - w_t^T x_t) \cdot x_t$$

$$b_{t+1} = b_t + \tau_t \cdot \mathrm{sgn}(y_t - w_t^T x_t)$$

### 2.6. Hàm mục tiêu theo dõi (trung bình trên tập huấn luyện)

$$J_{\text{PAR}}(w) = \frac{1}{n} \sum_{i=1}^{n} \max(0, |y_i - w^T x_i - b| - \epsilon)$$

---

## 3. PAR + Ridge Regularization

### 3.1. Bài toán tối ưu tại mỗi bước (thêm L2 penalty)

$$w_{t+1} = \argmin_w \frac{1}{2} \|w - w_t\|^2 + C \cdot L_t(w, x_t, y_t) + \frac{\alpha}{2} \|w\|^2$$

trong đó $\alpha > 0$ là hệ số Ridge (L2 regularization).

### 3.2. Hệ số cập nhật (ổn định)

$$\tau_t = \frac{L_t}{\dfrac{\|x_t\|^2}{1 + \alpha} + \dfrac{1}{2C}}$$

### 3.3. Quy tắc cập nhật trọng số (khi $L_t > 0$)

$$w_{t+1} = \frac{w_t + \tau_t \cdot \mathrm{sgn}(e_t) \cdot x_t}{1 + \alpha}$$

$$b_{t+1} = b_t + \tau_t \cdot \mathrm{sgn}(e_t)$$

### 3.4. Hàm mục tiêu theo dõi (có Ridge penalty)

$$J_{\text{ridge}}(w) = \frac{1}{n} \sum_{i=1}^{n} \max(0, |y_i - w^T x_i - b| - \epsilon) + \frac{\alpha}{2} \|w\|^2$$

---

## 4. Thông số đánh giá mô hình

### 4.1. Coefficient of Determination ($R^2$)

$$R^2 = 1 - \frac{\displaystyle \sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\displaystyle \sum_{i=1}^{n} (y_i - \bar{y})^2}$$

### 4.2. Mean Absolute Error (MAE)

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|$$

---

## 5. Các siêu tham số

| Tham số    | Ý nghĩa                                                                 |
| ---------- | ------------------------------------------------------------------------ |
| $C$        | Độ lớn bước cập nhật (aggressiveness); $C$ lớn → cập nhật mạnh          |
| $\epsilon$ | Ngưỡng sai số cho phép (vùng passive); $\epsilon$ lớn → ít cập nhật     |
| $\alpha$   | Hệ số Ridge (L2); $\alpha$ lớn → co trọng số mạnh, ổn định hơn         |
| max\_iter  | Số epoch huấn luyện; cần đủ lớn để objective hội tụ                     |

---

## 6. So sánh PAR và PAR + Ridge

| Tiêu chí               | PAR                                                        | PAR + Ridge                                                               |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Bài toán tối ưu**    | $\frac{1}{2}\|w-w_t\|^2 + C \cdot L_t$                    | $\frac{1}{2}\|w-w_t\|^2 + C \cdot L_t + \frac{\alpha}{2}\|w\|^2$         |
| **Regularization**     | Không có                                                   | L2 (Ridge) trên vector trọng số                                          |
| **Hiệu ứng**          | Trọng số có thể lớn tùy ý                                 | Trọng số bị co lại mỗi bước → ổn định, giảm overfitting                  |
| **Tham số thêm**       | Không                                                      | $\alpha$ (mức co trọng số)                                               |
| **Khi nào nên dùng**   | Dữ liệu ít nhiễu, cần mô hình đơn giản                   | Dữ liệu nhiều nhiễu, cần kiểm soát độ lớn trọng số                      |
