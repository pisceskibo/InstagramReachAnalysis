# InstagramReachAnalysis
> Thuật toán Passive Aggressive Regressor cho bài toán tối ưu hóa mô hình dự đoán chiến lược truyền thông Instagram

## 1. Giới thiệu bài toán:
+ Instagram là một trong những ứng dụng mạng xã hội phổ biến ngày nay và được sử dụng:
    + Quảng bá hình ảnh doanh nghiệp
    + Xây dựng thương hiệu cá nhân, blogging
    + Sáng tạo và tham khảo ra các thể loại content

+ Vấn đề: instagram thay đổi liên tục để phù hợp với mọi người nên nó ảnh hưởng đến phạm vi tiếp cận các bài viết lâu dài → nhà sáng tạo nội dung cần theo dõi dữ liệu tiếp cận của mình.

+ Hướng tiếp cận: thu thập dữ liệu thủ công về mức độ tiếp cận bài đăng.

+ Mục tiêu: ứng dụng thuật toán học máy tối ưu trực tuyến Passive Aggressive Regressor để tối ưu hóa sai số dự đoán lượt tiếp cận theo thời gian thực. Từ đó tìm ra các chỉ số tương tác cốt lõi nhằm tăng chiến lược phân phối nội dung trên mạng xã hội.


## 2. Giới thiệu bộ dữ liệu:
Bài toán giúp dự đoán tổng lượt Impressions của bài đăng Instagram dựa trên các chỉ số tương tác đầu vào (Likes, Saves, Comments, Shares, Follows, Profile Visits, ...). Đây là bài toán Hồi quy (Regression) với dữ liệu mạng xã hội biến động liên tục.

| Cột              | Ý nghĩa                              |
| ---------------- | ------------------------------------ |
| `Impressions`    | Tổng số lượt hiển thị của bài đăng (biến mục tiêu)  |
| `From Home`      | Lượt hiển thị đến từ trang Home/Feed |
| `From Hashtags`  | Lượt hiển thị đến từ Hashtags        |
| `From Explore`   | Lượt hiển thị đến từ Explore         |
| `From Other`     | Lượt hiển thị đến từ các nguồn khác  |
| `Saves`          | Số lần bài đăng được lưu             |
| `Comments`       | Số lượt bình luận                    |
| `Shares`         | Số lượt chia sẻ                      |
| `Likes`          | Số lượt thích                        |
| `Profile Visits` | Số lượt truy cập trang cá nhân       |
| `Follows`        | Số lượt follow mới                   |
| `Caption`        | Nội dung caption của bài đăng        |
| `Hashtags`       | Các hashtag được sử dụng             |

Bộ dữ liệu: dữ liệu Fashionista's Instagram cá nhân do Aman Kharwal sử dụng và công bố trong bài viết “[Instagram Reach Analysis using Python](https://www.kaggle.com/datasets/bhanupratapbiswas/instagram-reach-analysis-case-study/data)” đăng ngày 22/03/2022.


## 3. Phân tích dữ liệu:
+ Nhóm phạm vi tiếp cận: `From Home`* , `From Hashtags`* , `From Explore`, `From Other`
+ Nhóm từ văn bản: `Caption`, `Hashtags`
+ Nhóm tương tác: `Impressions` (biến mục tiêu), `Likes`* , `Comments`, `Shares`* , `Saves`*
+ Trong số những người vào trang `Profile Visits`, có bao nhiêu phân trăm người quyết định `Follows` tài khoản này: 

$$Conversion Rate = \frac{Follows}{Profile Visits} \times 100\%$$


## 4. Thuật toán Passive Aggressive Regressor (PAR):
### 4.1. Mô hình hồi quy tuyến tính Online Learning:
+ Online Linear Regression là cách xây dựng mô hình hồi quy tuyến tính theo kiểu học tuần tự: mô hình nhận từng mẫu dữ liệu mới, dự đoán và tính sai số rồi cập nhật trọng số ngay lập tức thay vì huấn luyện toàn bộ mô hình từ đầu.
+ So sánh Online Linear Regression với Linear Regression:
    + Linear Regression: thu thập toàn bộ tập dữ liệu và tìm các trọng số w sao cho tổng bình phương sai số đạt GTNN: $$\displaystyle \min_w \sum_{i = 1} (y_i - \hat{y_i})^2$$ với $w_i = (X^T X)^{-1} X^T y_i$
    + Online Linear Regression: thu thập tuần tự dữ liệu mới thông qua bộ dữ liệu.

### 4.2. Thuật toán Passive Aggressive Regressor:
Passive Aggressive Regressor (PAR) là một thuật toán hồi quy tuyến tính theo phương pháp Online Learning, được thiết kế để học dữ liệu tuần tự từng mẫu thay vì phải sử dụng toàn bộ tập dữ liệu cùng một lúc.
| Tiêu chí                       | Hồi quy OLS (Linear Regression)               | Passive Aggressive Regressor (PAR)              |
| ------------------------------ | --------------------------------------------- | ----------------------------------------------- |
| **Độ phức tạp**                | $O(nd^2)$                                     | O(d) mỗi mẫu                                  |
| **Cách xử lý dữ liệu**         | Xử lý toàn bộ tập dữ liệu                     | Xử lý từng mẫu tuần tự                          |
| **Bộ nhớ**                     | Cần lưu ma trận thiết kế $(n \times d)$         | Chủ yếu chỉ lưu vector trọng số (w)             |
| **Cập nhật mô hình**           | Thường phải tính toán lại trên tập dữ liệu    | Cập nhật (w) ngay sau mỗi mẫu                   |
| **Dữ liệu lớn**                | Có thể tốn nhiều thời gian/bộ nhớ khi (n) lớn | Phù hợp với dữ liệu rất lớn                     |
| **Mục tiêu**               | Tối thiểu hóa tổng bình phương sai số         | Cập nhật mạnh khi sai số vượt ngưỡng $\epsilon$ |

### 4.2.1. Mô hình hóa bài toán:
Cho x = [Likes, Saves, Comments, Shares, ProfileVisits, Follows] và y = Impressions với n mẫu dữ liệu. Khi đó, ta có mô hình hóa sau:

$$\hat{y_t} = w_t^T x_t + b$$

trong đó:
+ $x_t$: vector đặc trưng của mẫu thứ t
+ $y_t$: giá trị thực tế 
+ $\hat{y_t}$: giá trị dự đoán
+ $w_t$: vector trọng số tại thời điểm t

Do đó:

```math
\hat{\text{Impressions}} = w_t \begin{pmatrix} 
\text{Likes} \\ 
\text{Saves} \\ 
\text{Comments} \\ 
\text{Shares} \\ 
\text{ProfileVisits} \\ 
\text{Follows} 
\end{pmatrix} + b
```

### 4.2.2. Sai số và hàm mất mát:
+ Sai số: $e_t = y_t - \hat{y_t}$
+ Hàm mất mát: 
$$L_t(w, x_t, y_t) = max(0, |y_t - \hat{y_t}| - \epsilon)$$

### 4.2.3. Cơ chế Passive và Aggressive:
+ Nếu $|y_t - \hat{y_t}| \leq \epsilon$ thì $L_t = 0$ => Mô hình dự đoán đủ tốt và không cần cập nhật trọng số (Passive)
+ Nếu $|y_t - \hat{y_t}| > \epsilon$ thì $L_t > 0$ => Mô hình chưa đủ chính xác và cần cập nhật lại trọng số (Aggressive)

### 4.2.4. Cập nhật trọng số (Aggressive):
Bài toán tối ưu tại mỗi bước thỏa mãn:
+ Cho mẫu thứ t có $\hat{y_t} = w_t^T x_t + b$ và hàm mất mát $L_t(w, x_t, y_t) = max(0, |y_t - \hat{y_t}| - \epsilon)$
+ Cập nhật vector trọng số với siêu tham số C > 0:

$$w_{t + 1} = \argmin_w \frac{1}{2} ||w - w_t||^2 + CL_{t}(w, x_t, y_t)$$

+ Cập nhật hệ số: 

$$\tau_t = \frac{L_t(w_t, x_t, y_t)}{||x_t||^2 + \frac{1}{2C}}$$

+ Quy tắc cập nhật trọng số (bước cuối cùng): 

$$w_{t + 1} = w_t + \tau_t \cdot sgn(y_t - w_t^T x_t)x_t$$

## 5. Thông số đánh giá mô hình:
### 5.1. Coefficient of Determination:
Mô hình dữ liệu tốt đến đâu, khi $R^2$ càng cao thì mô hình càng tốt. Giúp mô hình giải thích và dự đoán sự biến thiên của Instagram Impressions tốt hơn.

$$R^2 = 1 - \frac{\displaystyle \sum_{i = 1}^n (y_i - \hat{y_i})^2}{\displaystyle \sum_{i = 1}^n (y_i - \overline{y_i})^2}$$

### 5.2. Mean Absolute Error:
Trung bình dự đoán sai lệch bao nhiêu, sai lệch càng thấp thì mô hình càng tốt.

$$MSE = \frac{1}{n} \displaystyle \sum_{i = 1}^n (y_i - \hat{y_i})^2$$

### 5.3. So sánh thông số:
| Mô hình                          | R² Score ↑ |       MSE ↓ |
| -------------------------------- | ---------: | ----------: |
| **Passive Aggressive Regressor** |     0.8789 | **1004.62** |
| **Linear Regression**            | **0.8895** |     1058.48 |

## 6. Tài liệu tham khảo:
Tài liệu tham khảo: https://thecleverprogrammer.com/2022/03/22/instagram-reach-analysis-using-python/

## 7. So sánh PAR và PAR + Ridge:
Notebook [`compare_par_ridge.ipynb`](compare_par_ridge.ipynb) và script [`compare_par_ridge.py`](compare_par_ridge.py) triển khai:
- Custom **Passive Aggressive Regressor (PA-II)** và **PAR + Ridge** bằng NumPy/Numba
- Thử nghiệm nhiều tham số `C`, `epsilon`, `alpha`, `max_iter`
- Biểu đồ Objective vs Iterations / Time (so sánh setup tốt nhất)
- Đối chiếu hàm mục tiêu + thời gian với scikit-learn (default)

Chạy nhanh:
```bash
python compare_par_ridge.py
```
Biểu đồ lưu trong thư mục `plots_par_ridge/`.

## 8. PAR thủ công và PAR thư viện
Script [`compare_instagram_predict.py`](compare_instagram_predict.py) có hai nhánh huấn luyện trên cùng một tập train và cùng bước chuẩn hóa:

- **PAR thủ công**: dùng NumPy, duyệt từng mẫu theo PA-I và tự cập nhật `w`, `b`, `tau`; không gọi `fit` hoặc `predict` của scikit-learn.
- **PAR scikit-learn**: dùng `PassiveAggressiveRegressor` làm mốc đối chiếu.

Với sai số $e = y - (w^Tx+b)$, bản thủ công dùng:

$$L = max(0, |e| - \epsilon), \qquad \tau = min\left(C, \frac{L}{||x||^2 + 1}\right)$$

Nếu $L > 0$ thì cập nhật:

$$w \leftarrow w + \tau\,sgn(e)x, \qquad b \leftarrow b + \tau\,sgn(e)$$

Chạy phép so sánh:

```bash
python compare_instagram_predict.py
```

Kết quả in ra gồm `R²`, `MSE` và thời gian chạy của hai cách.

## 9. So sánh PAR với các thuật toán tối ưu
Script [`compare_optimization_methods.py`](compare_optimization_methods.py) so sánh:

- **PAR**: cập nhật online theo từng mẫu; một epoch là một lần quét toàn bộ tập train.
- **GD**: Gradient Descent theo batch.
- **Nesterov**: Gradient Descent có momentum và gradient tại điểm nhìn trước.

### Cách so sánh

Mỗi vòng lặp của các thuật toán được xem là một lần xử lý toàn bộ tập train. Sau mỗi vòng, chương trình ghi:

- epsilon-insensitive loss trên train;
- `MSE` và `R²` trên test;
- thời gian tích lũy;
- số bước lặp/epoch.

Để tránh thang đo `Impressions` lớn làm hỏng bước học, mục tiêu được chuẩn hóa trong lúc tối ưu; `MSE` và `R²` được đổi lại và báo cáo theo đơn vị Impressions.

Epsilon-insensitive loss không khả vi tại biên `epsilon`. Vì vậy GD và Nesterov dùng bản loss được làm trơn để tính đạo hàm; PAR vẫn dùng loss gốc. Đây là lý do cần xem đồng thời đường cong loss chung và các metric test, không chỉ so sánh một con số cuối.

Chạy:

```bash
python compare_optimization_methods.py
```

Các file được lưu trong `plots_optimizers/`:

- `objective_vs_iteration.png`: objective theo số vòng lặp;
- `objective_vs_time.png`: objective theo thời gian;
- `mse_vs_iteration.png`: MSE test theo vòng lặp;
- `r2_vs_iteration.png`: R² test theo vòng lặp;
- `final_metrics.png`: MSE, R² và thời gian cuối;
- `optimizer_history.csv`, `optimizer_summary.csv`: dữ liệu để lập bảng báo cáo.

Ngoài bộ mặc định 100 bước, chương trình tạo thêm bộ 1000 bước để quan sát hội tụ dài hơn:

- `objective_vs_iteration_1000.png`;
- `mse_vs_iteration_1000.png`;
- `r2_vs_iteration_1000.png`;
- `objective_vs_time_1000.png`;
- `final_metrics_1000.png`;
- `optimizer_history_1000.csv`, `optimizer_summary_1000.csv`.

Bảng so sánh riêng train/test được lưu tại:

- `comparison_table.csv`, `comparison_table.png` cho 100 bước;
- `comparison_table_1000.csv`, `comparison_table_1000.png` cho 1000 bước.

Các cột gồm `Train time (ms)`, `Train MSE`, `Train R2`, `Test time (ms)`, `Test MSE` và `Test R2`.

### Đo công bằng hơn: batch và streaming

Benchmark cũng ghi thêm `samples_processed`, `time_ms` và `time_per_sample_ms` để không chỉ dựa vào số epoch. Trước khi đo PAR, kernel Numba được gọi warm-up một lần; thời gian biên dịch lần đầu không được tính vào thời gian huấn luyện chính.

Trong chế độ streaming, dữ liệu train được đưa vào theo batch. PAR giữ nguyên trọng số và cập nhật đúng một lần trên batch mới. GD và Nesterov phải huấn luyện lại trên toàn bộ dữ liệu đã xuất hiện sau mỗi batch. Vì vậy chương trình lưu riêng:

- `streaming_objective_vs_samples.png`: objective theo số mẫu đã xử lý;
- `streaming_objective_vs_time.png`: objective theo thời gian;
- `streaming_time_per_sample.png`: thời gian trung bình trên mỗi mẫu;
- `streaming_history.csv`, `streaming_summary.csv`: dữ liệu chi tiết và dòng tổng kết.

Trong bảng streaming, `samples_processed` của GD/Nesterov có thể lớn hơn số mẫu thật của tập train vì mỗi lần xuất hiện batch mới, hai thuật toán phải quét lại dữ liệu tích lũy nhiều epoch. Đây là thước đo trực tiếp chi phí của việc retrain khi dữ liệu đến liên tục.
