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

> Tài liệu tham khảo: https://thecleverprogrammer.com/2022/03/22/instagram-reach-analysis-using-python/