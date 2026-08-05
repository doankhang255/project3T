# Kế Hoạch Các Phương Pháp Xây Dựng Sentiment

## Mục Tiêu Chung

Tất cả các phương pháp bên dưới cùng giải quyết một nhiệm vụ:

```text
Tin tức tài chính
-> Phân loại sentiment từng bài: positive / negative / neutral
-> Tạo sentiment score từng bài
-> Tổng hợp theo ngày / tuần / doanh nghiệp
-> Xây dựng sentiment index
-> So sánh tác động với VN-Index / return / abnormal return
```

## 1. Phương Pháp Lexicon-Based

Tên gọi:

```text
Lexicon-based sentiment classification
Dictionary-based sentiment scoring
LIWC / VADER / Loughran-McDonald style
```=

Ý tưởng:

```text
Từ / cụm từ -> nhãn sentiment
Bài báo -> đếm positive / negative / neutral terms
Bài báo -> tính sentiment_score bằng công thức
```

Thành phần cần làm:

```text
Sentiment dictionary
Tokenization / n-gram matching
Term counting
Formula-based sentiment_score
Article-level sentiment_label
Daily / weekly sentiment index
```

Trạng thái trong project:

```text
Đã làm phần lớn.
File chính: equity_news_content_sentiment_ratios.parquet
Lưu ý: đây là weak label / formula label, không phải output của model fine-tuned.
```

## 2. Phương Pháp Traditional Machine Learning

Tên gọi:

```text
TF-IDF + Machine Learning
Bag of Words + Supervised ML
N-gram features + ML classifier
```

Ý tưởng:

```text
Tin tức đã gán ground truth
-> Bag of Words / TF-IDF / n-gram matrix
-> Train classifier
-> Predict sentiment từng bài
```

Model có thể dùng:

```text
Logistic Regression
Naive Bayes
Linear SVM
Random Forest
XGBoost / LightGBM
```

Thành phần cần làm:

```text
Ground truth labels
Train / validation / test split
TF-IDF vectorizer
ML classifier
Prediction probability
sentiment_score = P(positive) - P(negative)
Daily / weekly sentiment index
```

Trạng thái trong project:

```text
Chưa thấy triển khai thành một nhánh độc lập.
Project đã có tokenized text, n-gram terms, CSR matrix, nhưng chưa có pipeline ML supervised rõ ràng.
```

## 3. Phương Pháp Transfer Learning

Tên gọi:

```text
Transformer fine-tuned sentiment classifier
PhoBERT / BERT / RoBERTa fine-tuning
Contextual embedding-based sentiment classification
```

Ý tưởng:

```text
Pretrained Vietnamese language model
-> Fine-tune cho sentiment tài chính
-> Predict positive / negative / neutral cho từng bài
```

Model có thể dùng:

```text
PhoBERT
Vietnamese BERT
RoBERTa-style model
XLNet-style model
```

Thành phần cần làm:

```text
Input text
Label source: weak label hoặc ground truth
Tokenizer
Transformer classifier
Fine-tuning
Evaluation trên ground truth
sentiment_score = P(positive) - P(negative)
Daily / weekly sentiment index
```

Trạng thái trong project:

```text
Đã có fine-tune PhoBERT từ weak label.
File train chính: model_sentiment_v2.py
Model output folder: phobert_financial_sentiment_model_v2_class_weight
Cần bổ sung bước inference trên ground_truth_label.csv để đánh giá thật.
```

## 4. Phương Pháp PCA-Based Sentiment Index

Tên gọi:

```text
PCA-based sentiment index
PCA-based pessimism index
Factor-based media sentiment index
```

Ý tưởng:

```text
Nhiều biến sentiment / proxy
-> Chuẩn hóa biến
-> PCA
-> Lấy PC1 làm chỉ số sentiment hoặc pessimism index
```

Biến đầu vào có thể dùng:

```text
positive_article_ratio
negative_article_ratio
neutral_article_ratio
net_positive_article_ratio
avg_pos_ratio
avg_neg_ratio
sentiment_coverage_ratio
article_count
log_article_count
uncertainty / risk / loss term ratios nếu có
```

Thành phần cần làm:

```text
Aggregate article-level signals theo ngày / tuần
Build feature matrix
Standardize variables
Run PCA
Interpret loadings
Orient sign để PC1 đại diện cho pessimism
Compare PCA index với formula index
```

Trạng thái trong project:

```text
Chưa thấy triển khai thành một nhánh độc lập.
Nên xem PCA là phương pháp tạo index tổng hợp, không phải classifier cấp bài báo giống ML / Transformer.
Nếu muốn đưa vào như một phương pháp độc lập, PCA nên dùng sau khi có các proxy sentiment cấp bài hoặc cấp ngày/tuần.
```

## Phân Biệt Score Và Index

```text
Sentiment score:
  Điểm sentiment ở cấp từng bài báo.

Sentiment index:
  Chỉ số tổng hợp sentiment theo ngày / tuần / doanh nghiệp / thị trường.
```

## Cấu Trúc So Sánh Đề Xuất

```text
Method 1: Lexicon
-> article_sentiment_score_lexicon
-> daily_weekly_index_lexicon

Method 2: TF-IDF + ML
-> article_sentiment_score_ml
-> daily_weekly_index_ml

Method 3: Transformer fine-tuned
-> article_sentiment_score_transformer
-> daily_weekly_index_transformer

Method 4: PCA-based index
-> pca_sentiment_index / pca_pessimism_index
```

## Nhận Xét Về Project Hiện Tại

```text
Project hiện tại đang trộn một số phương pháp:

1. Lexicon:
   Đã dùng dictionary / n-gram / công thức để tạo sentiment_score và sentiment_label.

2. Transfer learning:
   Đã fine-tune PhoBERT trên label sinh từ phương pháp lexicon.

3. Traditional ML:
   Chưa có nhánh độc lập dùng TF-IDF + ML với ground truth.

4. PCA:
   Chưa có nhánh độc lập tạo PCA sentiment / pessimism index.
```

## Hướng Làm Tiếp Theo

```text
1. Giữ Lexicon làm baseline.
2. Dùng ground_truth_label.csv để evaluate Lexicon.
3. Xây TF-IDF + ML trên ground truth.
4. Evaluate Transformer fine-tuned trên ground truth.
5. Tạo PCA index từ các proxy sentiment aggregate.
6. So sánh 4 kết quả bằng:
   - accuracy
   - macro F1
   - per-class F1
   - confusion matrix
   - correlation / regression với VN-Index
```

## Ghi Chú Về Macro News Trong Equity News

Hiện tại `equity_news` đã được lọc từ dữ liệu tin tức, nhưng vẫn có thể lẫn một phần `macro news`. Điều này khó tránh hoàn toàn vì nhiều bài viết vừa nói về thị trường cổ phiếu, vừa nói về yếu tố vĩ mô như lãi suất, tỷ giá, lạm phát, chính sách tiền tệ hoặc kinh tế thế giới.

Các cách hạn chế ảnh hưởng của macro news:

```text
1. Tạo bộ từ khóa macro:
   lãi suất, lạm phát, GDP, CPI, tỷ giá, USD, NHNN,
   chính sách tiền tệ, Fed, giá dầu, giá vàng, FDI,...

2. Tạo thêm các biến nhận diện:
   macro_keyword_count
   ticker_count
   has_ticker
   news_scope

3. Phân loại phạm vi tin:
   pure_equity_news
   mixed_equity_macro_news
   macro_news
   other_or_uncertain

4. Gán trọng số khi xây index:
   pure_equity_news: weight = 1.0
   mixed_equity_macro_news: weight = 0.5
   macro_news: weight = 0.0 hoặc loại khỏi equity index

5. Tính nhiều phiên bản index để kiểm tra độ bền:
   index_all_news
   index_no_macro
   index_weighted_macro
   index_pure_equity_only

6. Nếu cần phân tích sâu hơn:
   tạo thêm macro_sentiment_index và dùng làm biến kiểm soát
   trong hồi quy với VN-Index.
```

Ý nghĩa:

```text
Không cần loại bỏ macro news tuyệt đối.
Quan trọng là nhận diện, giảm trọng số hoặc kiểm soát ảnh hưởng của nó.
Sau đó so sánh các phiên bản index để xem kết quả có ổn định hay không.
```
