# Scoring (prototype)

Folder riêng, tách khỏi `Lexicon_based/` và `Seed_set_Prepare/`, dùng để thử
nghiệm công thức tính điểm sentiment cho từng bài báo - lai giữa cách đếm
nhị phân của LIWC/LM gốc và cách cộng dồn có trọng số của VADER.

## Công thức

Cho từng category (negative, positive, uncertainty, litigious, strong_modal,
weak_modal, constraining), từng bài báo:

```
S_cat = ( sum_i  weight(w_i) * sign(w_i) )  /  N_total
```

- `w_i`: mỗi lần match 1 term trong dictionary (1/2/3-gram) thuộc category đó.
- `weight(w_i)`: term thuộc `final_seed` gốc = 1.0; term duyệt qua bootstrap
  (round 1-4) = `score_centered` của nó.
- `sign(w_i) = -1` nếu trong 4 token trước đó (cùng 1 câu) có 1 cue-word phủ
  định (`negation_cue_words.txt`), ngược lại `+1`.
- `N_total`: tổng số token của bài báo (chuẩn hóa theo độ dài).

`net_sentiment_score = positive_score - negative_score`. 5 category còn lại
(uncertainty, litigious, strong_modal, weak_modal, constraining) giữ điểm
riêng, không gộp chung.

## File

- `build_weighted_dictionary.py` -> `data/weighted_dictionary.csv` (1767
  term, lấy danh sách từ `seed_round4` - đã là bản gộp final_seed + 4 vòng
  bootstrap - rồi tra trọng số `score_centered` từ `review_round1..4.csv`).
- `score_articles.py` -> `data/article_scores.parquet` (điểm đầy đủ 126,576
  bài) + `data/article_scores_sample.csv` (45 bài mẫu: 15 tích cực nhất, 15
  tiêu cực nhất, 15 ngẫu nhiên, để review nhanh không cần mở file parquet).

## Đã kiểm tra sơ bộ

Top 5 bài `net_sentiment_score` cao nhất đều là tin lợi nhuận/tăng trưởng
vượt kế hoạch ("Vinamilk: dự kiến lãi ròng cả năm 9,310 tỷ đồng, vượt 13% kế
hoạch"...) - hợp lý.

## Giới hạn đã biết (prototype, chưa phải bản chính thức)

1. **Chưa có từ nhấn mạnh (degree modifier)** - VD "rất tăng trưởng mạnh"
   vs "tăng trưởng mạnh" hiện tính điểm như nhau. Đang xây dựng riêng, xem
   `../../Intensifier_words/`.
2. Match n-gram **không loại overlap** - 1 chuỗi token có thể vừa khớp
   1-gram vừa khớp 3-gram chứa nó, có thể double-count.
3. `NEGATION_WINDOW = 4` token là giá trị mặc định, **chưa hiệu chỉnh thực
   nghiệm** riêng cho tiếng Việt / văn phong tin tài chính.
4. `negation_cue_words.txt` đang được kiểm tra lại nguồn xác thực (xem
   `../../Intensifier_words/SOURCES.md`).
