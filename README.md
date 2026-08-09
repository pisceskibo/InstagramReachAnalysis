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


## 4. Mô hình Instagram Reach Prediction Model:


> Tài liệu tham khảo: https://thecleverprogrammer.com/2022/03/22/instagram-reach-analysis-using-python/