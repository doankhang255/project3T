# Project 3T - Phan Tich Sentiment Tin Tuc Va Tac Dong Den VN-Index

**Du lieu su dung trong du an nay duoc lay tu cong ty 3T.** Day la nguon du lieu dau vao quan trong cho toan bo qua trinh xu ly, gom du lieu tin tuc tai chinh, du lieu gia lich su va cac bang trung gian phuc vu phan tich thi truong chung khoan Viet Nam.

## Gioi Thieu

Du an nay xay dung pipeline xu ly tin tuc tai chinh bang NLP, gan nhan sentiment cho noi dung tin tuc, tao chi so **Equity Sentiment Index**, sau do phan tich moi quan he giua sentiment thi truong va bien dong **VN-Index**.

Muc tieu chinh:

- Lam sach va chuan hoa du lieu tin tuc tu nguon du lieu 3T.
- Loc cac tin lien quan den co phieu va thi truong chung khoan.
- Tach tu, tao tu dien/cum tu sentiment va gan nhan cam xuc cho tin tuc.
- Tong hop sentiment thanh chi so theo ngay va theo tuan.
- Ket hop sentiment index voi du lieu VN-Index de phan tich tuong quan, hoi quy du bao va tac dong thi truong.
- Tao bao cao thuc tap va cac bieu do minh hoa ket qua.

## Cau Truc Thu Muc

```text
project3T/
|-- data_News/              # Du lieu tin tuc, du lieu sentiment va ket qua phan tich tin tuc
|-- data_Histo/             # Du lieu gia lich su va VN-Index da xu ly
|-- News/                   # Code xu ly tin tuc, NLP, sentiment label va sentiment index
|-- News_Vnindex/           # Code merge sentiment voi VN-Index, correlation va regression
|-- Historical_price/       # Code EDA va xu ly du lieu gia lich su
|-- event_stock/            # Phan tich event study lien quan den co phieu
|-- REF/                    # Tai lieu tham khao va paper nen tang
|-- internship_report/      # Bao cao thuc tap, file Quarto, PDF va slide trinh bay
|-- requirements.txt        # Danh sach thu vien Python can cai dat
`-- test.py                 # Script thu nghiem trong qua trinh phat trien
```

## Pipeline Chinh

### 1. Xu Ly Tin Tuc

Code nam trong `News/EDA_clean/` va `News/prepare_data_model/`.

Qua trinh xu ly gom:

- Doc va kiem tra du lieu tin tuc goc.
- Lam sach noi dung bai viet.
- Chuan hoa domain, category va cac truong metadata.
- Loc tin lien quan den co phieu/thi truong.
- Tokenize tieng Viet bang `underthesea` hoac `VNCoreNLP`.
- Tao candidate terms, n-gram terms va ma tran dac trung phuc vu sentiment model.

Mot so file du lieu dau ra quan trong:

- `data_News/clean_news.parquet`
- `data_News/equity_news.parquet`
- `data_News/equity_news_clean_content.parquet`
- `data_News/equity_news_tokenized.parquet`
- `data_News/equity_news_tokenized_vncorenlp.parquet`

### 2. Gan Nhan Sentiment

Code nam trong `News/Process_sentiment_label/` va `News/Train_model/`.

Thanh phan chinh:

- Xay dung tu dien sentiment.
- Gan nhan positive, negative, neutral cho noi dung tin tuc.
- Tao cac ti le sentiment theo bai viet.
- Thu nghiem model sentiment cho du lieu tieng Viet.

Ket qua dau ra tieu bieu:

- `data_News/equity_news_content_sentiment_ratios.parquet`
- `data_News/equity_news_content_positive.csv`
- `data_News/equity_news_content_negative.csv`
- `data_News/equity_news_content_neutral.csv`

### 3. Xay Dung Equity Sentiment Index

Code nam trong `News/Build_sentiment_index/`.

Chi so sentiment duoc tong hop theo:

- Ngay: `market_sentiment_index_daily.parquet`
- Tuan: `market_sentiment_index_weekly.parquet`

Mot so cot quan trong:

- `article_count`: so luong bai viet.
- `sentiment_index`: diem sentiment trung binh.
- `sentiment_index_z`: diem sentiment da chuan hoa z-score.
- `positive_article_count`, `negative_article_count`, `neutral_article_count`: so bai theo tung nhom sentiment.

### 4. Xu Ly Du Lieu Gia Va VN-Index

Code nam trong `Historical_price/` va `News_Vnindex/`.

Du lieu gia lich su duoc dung de:

- Kiem tra va lam sach gia co phieu/VN-Index.
- Tao bien return theo tuan.
- Tao bien future return va cac bien control nhu volatility, volume, lag return.

Ket qua dau ra tieu bieu:

- `data_Histo/historical_price_all.parquet`
- `data_Histo/vnindex_weekly_return.parquet`
- `data_Histo/vnindex_weekly_return.csv`

### 5. Phan Tich Tac Dong Sentiment Den VN-Index

Code nam trong `News_Vnindex/`.

Phan tich gom:

- Merge sentiment weekly voi VN-Index weekly return.
- Tinh tuong quan giua sentiment va return.
- Chay predictive regression de kiem tra sentiment co giai thich/du bao bien dong VN-Index hay khong.
- Ve bieu do sentiment index, VN-Index return, cumulative return va tac dong hoi quy.

Ket qua dau ra:

- `data_News/vnindex_weekly_sentiment_merged.parquet`
- `data_News/vnindex_weekly_sentiment_merged.csv`
- `data_News/vnindex_weekly_correlation.csv`
- `data_News/vnindex_weekly_predictive_regression.csv`
- `data_News/figures/`

## Cai Dat Moi Truong

Yeu cau Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Cach Chay Mot So Buoc Chinh

Tao VN-Index weekly return:

```powershell
python News_Vnindex\build_vnindex_weekly_return.py
```

Merge VN-Index weekly voi sentiment weekly:

```powershell
python News_Vnindex\merge_vnindex_weekly_with_sentiment.py
```

Chay correlation:

```powershell
python News_Vnindex\run_vnindex_weekly_correlation.py
```

Chay predictive regression:

```powershell
python News_Vnindex\vnindex_weekly_predictive_regression.py
```

Ve bieu do phan tich:

```powershell
python News_Vnindex\plot_sentiment_vs_vnindex_timeseries.py
python News_Vnindex\plot_predictive_regression_impact.py
```

## Bao Cao Va Slide

Thu muc `internship_report/` chua bao cao thuc tap duoi dang Quarto va file PDF da render.

Render lai bao cao:

```powershell
cd internship_report
quarto render final_report.qmd --to pdf
```

File ket qua nam trong:

```text
internship_report/_output/final_report.pdf
```

## Tai Lieu Tham Khao

Thu muc `REF/` gom cac paper va tai lieu nen tang ve:

- Media sentiment va thi truong tai chinh.
- Tu dien sentiment trong tai chinh.
- Tac dong cua tin tuc den loi suat co phieu.
- Phuong phap event study va predictive regression.

## Luu Y Ve Du Lieu

Vi du lieu duoc lay tu cong ty 3T, can dam bao viec chia se, sao chep hoac cong bo du lieu tu repo nay tuan thu quy dinh noi bo va cac thoa thuan lien quan den bao mat du lieu.
