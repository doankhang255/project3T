# Tổng kết project Equity Sentiment Index

## 1. Mục tiêu project

Project này xây dựng một pipeline dữ liệu để đo lường sentiment của tin tức thị trường chứng khoán Việt Nam, sau đó phân tích mối quan hệ giữa sentiment và VN-Index.

Mục tiêu chính:

- Làm sạch dữ liệu giá lịch sử và dữ liệu tin tức tài chính/chứng khoán.
- Tạo bộ dữ liệu `equity_news` sạch, có nội dung đã được tokenize.
- Xây dựng candidate terms và dictionary sentiment tài chính.
- Gán sentiment score và sentiment label cho từng bài báo.
- Tạo Equity Sentiment Index theo ngày, tuần và tháng.
- Ghép weekly sentiment với weekly return của VN-Index.
- Chạy correlation, abnormal return và predictive regression để đánh giá tác động của sentiment.

## 2. Cấu trúc project

```text
project3T/
├── Historical_price/          # EDA, clean và feature engineering cho giá lịch sử
├── News/                      # Clean tin tức, tokenize, sentiment dictionary/model/index
├── News_Vnindex/              # Merge sentiment với VN-Index, correlation/regression
├── event_stock/               # Xử lý và gán sentiment cho event stock
├── internship_report/         # Báo cáo cuối kỳ bằng Quarto
├── REF/                       # Tài liệu tham khảo/báo cáo gốc
└── requirements.txt           # Thư viện Python cần dùng
```

Lưu ý: trong workspace hiện tại không thấy các thư mục dữ liệu/output như `data`, `data_News`, `data_Histo`. Các file này có thể đang nằm ngoài repo hoặc bị ignore. Các số liệu bên dưới được lấy từ `internship_report/final_report.qmd`.

## 3. Những việc đã làm được

### 3.1. Xử lý dữ liệu giá lịch sử

Đã xây dựng pipeline trong `Historical_price/` để:

- Đọc dữ liệu giá lịch sử từ parquet/csv.
- Chuẩn hóa kiểu dữ liệu cho `symbol`, `date`, `year`, OHLCV và các cột giao dịch.
- Lọc danh sách mã chứng khoán cần phân tích.
- Kiểm tra và tách các dòng lỗi: duplicate theo `symbol-date`, sai lệch năm, OHLC không hợp lệ, volume bằng 0, mismatch thành phần volume, dữ liệu foreign/proprietary trading bị thiếu.
- Làm sạch OHLC bằng cách loại các dòng có OHLC bằng 0.
- Chuẩn hóa trường hợp volume bằng 0 nhưng OHLC thay đổi bằng `basic_price`.
- Tạo feature dữ liệu giá: return hiện tại, future return 1/3/5 ngày, log volume, abnormal volume, intraday range, rolling volatility và các lag feature.
- Tách riêng dữ liệu VNINDEX và tạo file VNINDEX clean để tính weekly return.

Script chính:

- `Historical_price/EDA_raw_historical_price.py`
- `Historical_price/EDA_ticker_required.py`
- `Historical_price/clean_EDA.py`
- `Historical_price/EDA_Gen_feature.py`
- `Historical_price/VNindex_EDA.py`

### 3.2. Xử lý dữ liệu tin tức

Đã xây dựng pipeline trong `News/EDA_clean/` để:

- Đọc dữ liệu tin tức raw.
- Chuẩn hóa các cột text: link, domain, category, title, description, keywords, author.
- Parse `publication_date`, loại ngày không hợp lệ và các record trước năm 2010.
- Kiểm tra missing values, duplicate theo `link + publication_date`, thống kê category/domain.
- Chuẩn hóa domain về dạng ngắn gọn, bỏ `http`, `https`, `www`, path/query/hash.
- Parse `keywords` dạng dictionary-like thành danh sách keyword.
- Lọc category/domain liên quan đến thị trường cổ phiếu để tạo `equity_news`.
- Làm sạch nội dung bài viết: loại nội dung quảng cáo, cắt bỏ phần "có thể bạn quan tâm", xóa title/description bị lặp trong content, thay content quá ngắn bằng description.

Script chính:

- `News/EDA_clean/EDA_raw_news.py`
- `News/EDA_clean/news_clean.py`
- `News/EDA_clean/to_equity_news.py`
- `News/EDA_clean/clean_content.py`

### 3.3. Tokenize tiếng Việt và tạo term/document features

Đã làm hai hướng tokenize:

- `underthesea` trong `News/prepare_data_model/tokenize_underthesea.py`.
- `VNCoreNLP` trong `News/prepare_data_model/used_for_v1/tokenize_VNCoreNLP.py`.

Pipeline VNCoreNLP có thêm xử lý theo câu:

- Tách câu bằng dấu chấm.
- Loại số trước khi tách câu.
- Word segmentation bằng VNCoreNLP.
- Làm sạch token: lowercase, bỏ ký tự đặc biệt, bỏ token quá ngắn, bỏ token bắt đầu bằng số.
- Tạo `Tokenize_content`, `Tokenize_content_sentences`, `total_tokenizer`.

Sau tokenize, project đã tạo ma trận sparse term-document bằng `CountVectorizer`:

- `News/prepare_data_model/build_CSR_matrix.py`

Và lọc candidate terms theo:

- Term frequency.
- Document frequency.
- `df_ratio`.
- Stopword tiếng Việt.
- Loại token có số hoặc shape không hợp lệ.

Script liên quan:

- `News/prepare_data_model/candidate_term_non_ngram.py`
- `News/prepare_data_model/used_for_v1/build_ngram_terms.py`
- `News/prepare_data_model/used_for_v1/candidate_term_ngram.py`

### 3.4. Xây dựng dictionary sentiment tài chính

Đã xây dựng dictionary sentiment từ candidate n-gram terms:

- Input là `candidate_ngram_terms.parquet`.
- Chỉ giữ unigram và bigram.
- Dùng model PhoBERT sentiment local: `wonrax/phobert-base-vietnamese-sentiment`.
- Dự đoán nhãn `positive`, `negative`, `neutral` và confidence/probability cho từng term.
- Xuất dictionary ra parquet và csv.

Script chính:

- `News/Process_sentiment_label/build_sentiment_dictionary.py`

Kết quả theo báo cáo:

- `candidate_ngram_terms.parquet`: 43.840 candidate terms.
- `candidate_ngram_terms_dictionary.parquet`: 43.840 terms đã có nhãn sentiment.

### 3.5. Gán sentiment cho từng bài báo

Đã xây dựng pipeline gán sentiment cho nội dung bài báo:

- Load dictionary sentiment.
- Match unigram/bigram trong từng câu/bài viết.
- Ưu tiên bigram trước unigram.
- Đếm số token/term positive, negative, neutral.
- Tính `pos_ratio`, `neg_ratio`, `neutral_ratio`, `sentiment_coverage_ratio`.
- Tính `sentiment_score = (pos_count - neg_count) / (pos_count + neg_count)`.
- Gán label:
  - `positive` nếu score >= 0.30 và có ít nhất 2 polarity hits.
  - `negative` nếu score <= -0.30 và có ít nhất 2 polarity hits.
  - Còn lại là `neutral`.

Script chính:

- `News/Process_sentiment_label/build_sentiment_label_content.py`

Kết quả theo báo cáo:

- `equity_news_content_sentiment_ratios.parquet`: 126.576 bài báo có sentiment score và label.

### 3.6. Fine-tune model PhoBERT sentiment

Đã có script fine-tune model sentiment trên dữ liệu đã gán nhãn:

- Base model: `wonrax/phobert-base-vietnamese-sentiment`.
- Train/validation split 80/20, stratify theo label.
- Label mapping: negative = 0, positive = 1, neutral = 2.
- Dùng class weight trong CrossEntropyLoss để xử lý imbalance.
- Metric: accuracy, macro F1, weighted F1.
- Lưu best model theo `f1_macro`.

Script chính:

- `News/Process_sentiment_label/model_sentiment_v2.py`
- `News/Train_model/model_sentiment_v1.py`

Kết quả theo báo cáo:

- Accuracy khoảng 85,5%.
- Macro F1 khoảng 82,0%.

### 3.7. Xây dựng Equity Sentiment Index

Đã tạo sentiment index theo 3 tần suất:

- Daily index.
- Weekly index.
- Monthly index.

Công thức chính:

- Gom nhóm bài báo theo ngày/tuần/tháng.
- `sentiment_index` = trung bình `sentiment_score`.
- Đếm số bài positive/negative/neutral.
- Chuẩn hóa `sentiment_index_z = (sentiment_index - mean) / std`.

Script chính:

- `News/Build_sentiment_index/build_sentiment_index_daily.py`
- `News/Build_sentiment_index/build_sentiment_index_weekly.py`
- `News/Build_sentiment_index/build_sentiment_index_monthly.py`

Kết quả theo báo cáo:

- Daily sentiment index: 5.037 quan sát.
- Weekly sentiment index: 731 quan sát.
- Monthly sentiment index: đã có script tạo index theo tháng.

### 3.8. Xử lý VN-Index và merge với sentiment

Đã xây dựng pipeline trong `News_Vnindex/` để:

- Lấy dữ liệu VNINDEX đã clean.
- Gom theo tuần `W-SUN`.
- Tính weekly OHLCV, weekly return, future return 1 tuần và 4 tuần.
- Tính lag return, volatility 12 tuần và log volume.
- Merge weekly VNINDEX với weekly sentiment theo `week_end`.
- Thêm `log_article_count`.

Script chính:

- `News_Vnindex/build_vnindex_weekly_return.py`
- `News_Vnindex/merge_vnindex_weekly_with_sentiment.py`

Kết quả theo báo cáo:

- `vnindex_weekly_return.parquet`: 793 tuần.
- `vnindex_weekly_sentiment_merged.parquet`: 722 tuần.
- Có 9 tuần nghỉ lễ Tết chưa được xử lý nên số quan sát sau merge nhỏ hơn weekly sentiment.

### 3.9. Abnormal return, correlation và predictive regression

Đã tạo abnormal return theo hai cách:

- Rolling expected return 26 tuần, tối thiểu 12 tuần.
- Rolling AR(1) expected return 52 tuần, tối thiểu 26 quan sát.

Đã tính:

- `abnormal_return_rolling_1w`
- `future_abnormal_rolling_ret_1w`
- `future_abnormal_rolling_ret_4w`
- `abnormal_return_ar1_1w`
- `future_abnormal_ar1_ret_1w`
- `future_abnormal_ar1_ret_4w`

Đã chạy correlation:

- Pearson và Spearman.
- Giữa `sentiment_index_z`, `net_positive_article_ratio` với future return/abnormal return 1w và 4w.

Đã chạy predictive regression:

- Target: future return 1w/4w và future abnormal return 1w/4w.
- Predictor: `sentiment_index_z`, `return_lag_1w`, `volatility_12w`, `log_article_count`.
- Sai số chuẩn Newey-West/HAC theo horizon.

Script chính:

- `News_Vnindex/build_vnindex_weekly_abnormal_return.py`
- `News_Vnindex/vnindex_weekly_correlation.py`
- `News_Vnindex/vnindex_weekly_predictive_regression.py`
- `News_Vnindex/run_vnindex_weekly_correlation.py`

Kết luận theo báo cáo:

- Sentiment có quan hệ yếu với return tuần kế tiếp.
- Tín hiệu âm rõ hơn ở horizon 4 tuần, đặc biệt với future abnormal return.

### 3.10. Xử lý event stock

Đã có module riêng trong `event_stock/` để EDA và gán nhãn sentiment cho event của cổ phiếu:

- Chuẩn hóa text và sửa typo trong event title.
- Dùng keyword map positive/negative/neutral.
- Dùng `event_code` làm prior/fallback.
- Ưu tiên positive/negative hơn neutral khi title match nhiều nhóm.

Script chính:

- `event_stock/EDA_event_stoick_fix.py`
- `event_stock/EDA_event_stock.ipynb`

## 4. Thứ tự các bước từ clean đến bước cuối

Đây là luồng chạy logic của project, từ bước clean dữ liệu đến bước phân tích cuối cùng.

### Bước 1. Clean và EDA dữ liệu giá lịch sử

Chạy các script trong `Historical_price/`:

```bash
python Historical_price/EDA_raw_historical_price.py
python Historical_price/EDA_ticker_required.py
python Historical_price/clean_EDA.py
python Historical_price/EDA_Gen_feature.py
python Historical_price/VNindex_EDA.py
```

Output kỳ vọng:

- Dữ liệu giá đã chuẩn hóa.
- Dữ liệu ticker required đã clean.
- Feature giá lịch sử.
- File VNINDEX clean, thường là `data_Histo/vnindex_eda_output.csv`.

### Bước 2. Clean metadata tin tức raw

Chạy:

```bash
python News/EDA_clean/EDA_raw_news.py
python News/EDA_clean/news_clean.py
```

Output kỳ vọng:

- Tin tức đã parse ngày.
- Domain/category/keyword đã chuẩn hóa.
- Bỏ record ngày lỗi, trước năm 2010, description quá ngắn, duplicate.

### Bước 3. Lọc equity news

Chạy:

```bash
python News/EDA_clean/to_equity_news.py
```

Output kỳ vọng:

- `equity_news.parquet`: chỉ giữ tin liên quan đến thị trường cổ phiếu dựa trên danh sách category/domain hợp lệ.

### Bước 4. Clean content bài báo

Chạy:

```bash
python News/EDA_clean/clean_content.py
```

Output kỳ vọng:

- `equity_news_clean_content.parquet`: nội dung bài báo đã loại nhiễu, title/description lặp, quảng cáo và phần liên quan không cần thiết.

### Bước 5. Tokenize tiếng Việt

Có thể dùng underthesea:

```bash
python News/prepare_data_model/tokenize_underthesea.py
```

Hoặc VNCoreNLP:

```bash
python News/prepare_data_model/used_for_v1/tokenize_VNCoreNLP.py
```

Output kỳ vọng:

- `equity_news_tokenized.parquet` hoặc `equity_news_tokenized_vncorenlp.parquet`.
- Các cột token: `Tokenize_content`, `Tokenize_content_sentences`, `total_tokenizer`.

### Bước 6. Tạo term-document matrix và candidate terms

Chạy:

```bash
python News/prepare_data_model/build_CSR_matrix.py
python News/prepare_data_model/used_for_v1/build_ngram_terms.py
python News/prepare_data_model/used_for_v1/candidate_term_ngram.py
```

Output kỳ vọng:

- Term statistics.
- Candidate unigram/bigram terms.
- `candidate_ngram_terms.parquet`.

### Bước 7. Xây dựng sentiment dictionary

Chạy:

```bash
python News/Process_sentiment_label/build_sentiment_dictionary.py
```

Output kỳ vọng:

- `candidate_ngram_terms_dictionary.parquet`.
- `candidate_ngram_terms_dictionary.csv`.

### Bước 8. Gán sentiment score/label cho từng bài báo

Chạy:

```bash
python News/Process_sentiment_label/build_sentiment_label_content.py
```

Output kỳ vọng:

- `equity_news_content_sentiment_ratios.parquet`.
- Mỗi bài báo có pos/neg/neutral count, ratio, coverage, score và label.

### Bước 9. Fine-tune model sentiment

Chạy nếu cần train model:

```bash
python News/Process_sentiment_label/model_sentiment_v2.py
```

Output kỳ vọng:

- Model PhoBERT sentiment fine-tuned cho ngữ cảnh tài chính.
- Evaluation report gồm accuracy, macro F1, weighted F1.

### Bước 10. Tạo sentiment index theo ngày/tuần/tháng

Chạy:

```bash
python News/Build_sentiment_index/build_sentiment_index_daily.py
python News/Build_sentiment_index/build_sentiment_index_weekly.py
python News/Build_sentiment_index/build_sentiment_index_monthly.py
```

Output kỳ vọng:

- `market_sentiment_index_daily.parquet`
- `market_sentiment_index_weekly.parquet`
- `market_sentiment_index_monthly.parquet`

### Bước 11. Tạo weekly return VN-Index

Chạy:

```bash
python News_Vnindex/build_vnindex_weekly_return.py
```

Output kỳ vọng:

- `vnindex_weekly_return.parquet`
- Weekly return, future return 1w/4w, lag return, volatility 12w.

### Bước 12. Merge weekly sentiment với weekly VN-Index

Chạy:

```bash
python News_Vnindex/merge_vnindex_weekly_with_sentiment.py
```

Output kỳ vọng:

- `vnindex_weekly_sentiment_merged.parquet`
- Dataset chung theo `week_end`.

### Bước 13. Tính abnormal return

Chạy:

```bash
python News_Vnindex/build_vnindex_weekly_abnormal_return.py
```

Output kỳ vọng:

- `vnindex_weekly_sentiment_abnormal_return.parquet`
- Future abnormal return 1w/4w theo rolling mean và AR(1).

### Bước 14. Chạy correlation

Chạy:

```bash
python News_Vnindex/vnindex_weekly_correlation.py
```

Hoặc script tổng hợp:

```bash
python News_Vnindex/run_vnindex_weekly_correlation.py
```

Output kỳ vọng:

- `vnindex_weekly_correlation.parquet`
- `vnindex_weekly_correlation.csv`

### Bước 15. Chạy predictive regression

Chạy:

```bash
python News_Vnindex/vnindex_weekly_predictive_regression.py
```

Output kỳ vọng:

- `vnindex_weekly_predictive_regression.parquet`
- `vnindex_weekly_predictive_regression.csv`
- Bảng hệ số, Newey-West standard error, t-stat, p-value, R-squared.

### Bước 16. Tổng hợp báo cáo

Báo cáo chính nằm ở:

```text
internship_report/final_report.qmd
```

Nội dung báo cáo đã tổng hợp:

- Bối cảnh và mục tiêu thực tập.
- Quy trình xử lý giá lịch sử và tin tức.
- Dictionary sentiment và model PhoBERT.
- Equity Sentiment Index.
- Correlation, abnormal return và predictive regression với VN-Index.
- Kết luận và hướng mở rộng.

## 5. Kết quả trung gian quan trọng

Theo `internship_report/final_report.qmd`, các mốc output chính:

| File | Số dòng | Ghi chú |
|---|---:|---|
| `data_News/dataset_news.parquet` | 719.603 | Dữ liệu tin tức ban đầu |
| `data_News/clean_news.parquet` | 678.418 | Tin tức sau clean cơ bản |
| `data_News/equity_news.parquet` | 130.545 | Tin liên quan thị trường cổ phiếu |
| `data_News/equity_news_clean_content.parquet` | 126.576 | Nội dung bài báo sau clean |
| `data_News/equity_news_tokenized_vncorenlp.parquet` | 126.576 | Nội dung đã tokenize |
| `data_News/candidate_ngram_terms.parquet` | 43.840 | Candidate terms |
| `data_News/candidate_ngram_terms_dictionary.parquet` | 43.840 | Dictionary sentiment |
| `data_News/equity_news_content_sentiment_ratios.parquet` | 126.576 | Bài báo có sentiment |
| `data_News/market_sentiment_index_daily.parquet` | 5.037 | Daily sentiment index |
| `data_News/market_sentiment_index_weekly.parquet` | 731 | Weekly sentiment index |
| `data_Histo/historical_price_all.parquet` | 4.307.791 | Dữ liệu giá lịch sử |
| `data_Histo/vnindex_weekly_return.parquet` | 793 | Weekly return VN-Index |
| `data_News/vnindex_weekly_sentiment_merged.parquet` | 722 | Weekly sentiment + VN-Index |

Khoảng thời gian chính:

- Tin tức cổ phiếu sau clean: 01/01/2010 đến 31/12/2023.
- Dữ liệu giá lịch sử: 04/01/2010 đến 12/05/2025.
- Daily sentiment index: 01/01/2010 đến 31/12/2023.
- Weekly sentiment/VN-Index merged: giai đoạn 2010 đến 2023.

## 6. Trạng thái hiện tại và việc có thể làm tiếp

Project đã có pipeline khá đầy đủ từ raw data đến phân tích định lượng. Các phần đã hoàn thành gồm clean dữ liệu, tokenize, tạo dictionary, gán sentiment, tạo index, merge VNINDEX, correlation và regression.

Những điểm nên tiếp tục hoàn thiện:

- Chuẩn hóa lại đường dẫn giữa `data` và `data_News`/`data_Histo` vì trong code hiện đang có sự khác nhau giữa các script.
- Thêm một file runner hoặc Makefile để chạy pipeline theo đúng thứ tự.
- Lưu log kết quả mỗi bước ra file thay vì chỉ print ra console.
- Thêm unit test cho các hàm clean quan trọng: parse ngày, normalize domain, clean content, sentiment score.
- Xử lý rõ hơn các tuần nghỉ lễ/Tết khi merge weekly sentiment với VNINDEX.
- Thêm backtesting hoặc mô hình dự báo ngoài mẫu nếu muốn biến sentiment index thành signal đầu tư.
