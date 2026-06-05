# Project 3T - Phân tích Sentiment Tin tức và Tác động đến VN-Index

**Dữ liệu sử dụng trong dự án này được lấy từ công ty 3T.** Đây là nguồn dữ liệu đầu vào quan trọng cho toàn bộ quá trình xử lý, bao gồm dữ liệu tin tức tài chính, dữ liệu giá lịch sử và các bảng trung gian phục vụ phân tích thị trường chứng khoán Việt Nam.

## Giới thiệu

Dự án này xây dựng pipeline xử lý tin tức tài chính bằng NLP, gắn nhãn sentiment cho nội dung tin tức, tạo chỉ số **Equity Sentiment Index**, sau đó phân tích mối quan hệ giữa sentiment thị trường và biến động **VN-Index**.

Mục tiêu chính:

* Làm sạch và chuẩn hóa dữ liệu tin tức từ nguồn dữ liệu 3T.
* Lọc các tin liên quan đến cổ phiếu và thị trường chứng khoán.
* Tách từ, tạo từ điển/cụm từ sentiment và gắn nhãn cảm xúc cho tin tức.
* Tổng hợp sentiment thành chỉ số theo ngày và theo tuần.
* Kết hợp sentiment index với dữ liệu VN-Index để phân tích tương quan, hồi quy dự báo và tác động thị trường.
* Tạo báo cáo thực tập và các biểu đồ minh họa kết quả.

## Cấu trúc thư mục

```text
project3T/
|-- data_News/              # Dữ liệu tin tức, dữ liệu sentiment và kết quả phân tích tin tức
|-- data_Histo/             # Dữ liệu giá lịch sử và VN-Index đã xử lý
|-- News/                   # Code xử lý tin tức, NLP, sentiment label và sentiment index
|-- News_Vnindex/           # Code merge sentiment với VN-Index, correlation và regression
|-- Historical_price/       # Code EDA và xử lý dữ liệu giá lịch sử
|-- event_stock/            # Phân tích event study liên quan đến cổ phiếu
|-- REF/                    # Tài liệu tham khảo và paper nền tảng
|-- internship_report/      # Báo cáo thực tập, file Quarto, PDF và slide trình bày
|-- requirements.txt        # Danh sách thư viện Python cần cài đặt
`-- test.py                 # Script thử nghiệm trong quá trình phát triển
```

## Pipeline chính

### 1. Xử lý tin tức

Code nằm trong `News/EDA_clean/` và `News/prepare_data_model/`.

Quá trình xử lý gồm:

* Đọc và kiểm tra dữ liệu tin tức gốc.
* Làm sạch nội dung bài viết.
* Chuẩn hóa domain, category và các trường metadata.
* Lọc tin liên quan đến cổ phiếu/thị trường chứng khoán.
* Tokenize tiếng Việt bằng `underthesea` hoặc `VNCoreNLP`.
* Tạo candidate terms, n-gram terms và ma trận đặc trưng phục vụ sentiment model.

Một số file dữ liệu đầu ra quan trọng:

* `data_News/clean_news.parquet`
* `data_News/equity_news.parquet`
* `data_News/equity_news_clean_content.parquet`
* `data_News/equity_news_tokenized.parquet`
* `data_News/equity_news_tokenized_vncorenlp.parquet`

### 2. Gắn nhãn Sentiment

Code nằm trong `News/Process_sentiment_label/` và `News/Train_model/`.

Thành phần chính:

* Xây dựng từ điển sentiment.
* Gắn nhãn positive, negative, neutral cho nội dung tin tức.
* Tạo các tỷ lệ sentiment theo bài viết.
* Thử nghiệm model sentiment cho dữ liệu tiếng Việt.

Kết quả đầu ra tiêu biểu:

* `data_News/equity_news_content_sentiment_ratios.parquet`
* `data_News/equity_news_content_positive.csv`
* `data_News/equity_news_content_negative.csv`
* `data_News/equity_news_content_neutral.csv`

### 3. Xây dựng Equity Sentiment Index

Code nằm trong `News/Build_sentiment_index/`.

Chỉ số sentiment được tổng hợp theo:

* Ngày: `market_sentiment_index_daily.parquet`
* Tuần: `market_sentiment_index_weekly.parquet`

Một số cột quan trọng:

* `article_count`: số lượng bài viết.
* `sentiment_index`: điểm sentiment trung bình.
* `sentiment_index_z`: điểm sentiment đã chuẩn hóa z-score.
* `positive_article_count`, `negative_article_count`, `neutral_article_count`: số bài theo từng nhóm sentiment.

### 4. Xử lý dữ liệu giá và VN-Index

Code nằm trong `Historical_price/` và `News_Vnindex/`.

Dữ liệu giá lịch sử được dùng để:

* Kiểm tra và làm sạch giá cổ phiếu/VN-Index.
* Tạo biến return theo tuần.
* Tạo biến future return và các biến control như volatility, volume, lag return.

Kết quả đầu ra tiêu biểu:

* `data_Histo/historical_price_all.parquet`
* `data_Histo/vnindex_weekly_return.parquet`
* `data_Histo/vnindex_weekly_return.csv`

### 5. Phân tích tác động của Sentiment đến VN-Index

Code nằm trong `News_Vnindex/`.

Phân tích gồm:

* Merge sentiment weekly với VN-Index weekly return.
* Tính tương quan giữa sentiment và return.
* Chạy predictive regression để kiểm tra sentiment có giải thích/dự báo biến động VN-Index hay không.
* Vẽ biểu đồ sentiment index, VN-Index return, cumulative return và tác động hồi quy.

Kết quả đầu ra:

* `data_News/vnindex_weekly_sentiment_merged.parquet`
* `data_News/vnindex_weekly_sentiment_merged.csv`
* `data_News/vnindex_weekly_correlation.csv`
* `data_News/vnindex_weekly_predictive_regression.csv`
* `data_News/figures/`

## Cài đặt môi trường

Yêu cầu Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Cách chạy một số bước chính

Tạo VN-Index weekly return:

```powershell
python News_Vnindex\build_vnindex_weekly_return.py
```

Merge VN-Index weekly với sentiment weekly:

```powershell
python News_Vnindex\merge_vnindex_weekly_with_sentiment.py
```

Chạy correlation:

```powershell
python News_Vnindex\run_vnindex_weekly_correlation.py
```

Chạy predictive regression:

```powershell
python News_Vnindex\vnindex_weekly_predictive_regression.py
```

Vẽ biểu đồ phân tích:

```powershell
python News_Vnindex\plot_sentiment_vs_vnindex_timeseries.py
python News_Vnindex\plot_predictive_regression_impact.py
```

## Báo cáo và Slide

Thư mục `internship_report/` chứa báo cáo thực tập dưới dạng Quarto và file PDF đã render.

Render lại báo cáo:

```powershell
cd internship_report
quarto render final_report.qmd --to pdf
```

File kết quả nằm trong:

```text
internship_report/_output/final_report.pdf
```

## Tài liệu tham khảo

Thư mục `REF/` gồm các paper và tài liệu nền tảng về:

* Media sentiment và thị trường tài chính.
* Từ điển sentiment trong tài chính.
* Tác động của tin tức đến lợi suất cổ phiếu.
* Phương pháp event study và predictive regression.

## Lưu ý về dữ liệu

Vì dữ liệu được lấy từ công ty 3T, cần đảm bảo việc chia sẻ, sao chép hoặc công bố dữ liệu từ repo này tuân thủ quy định nội bộ và các thỏa thuận liên quan đến bảo mật dữ liệu.
