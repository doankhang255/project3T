# Huong Dan Phan Tich Tac Dong Equity Sentiment Index Den VN-Index

File nay huong dan cach xu ly phan tich tac dong sau khi da co:

- `market_sentiment_index_daily.parquet`
- `market_sentiment_index_weekly.parquet`

Muc tieu la tra loi cau hoi:

> Equity Sentiment Index co giai thich hoac du bao duoc bien dong VN-Index trong tuong lai hay khong?

## 1. Du Lieu Dau Vao Can Co

### 1.1. Sentiment index daily

File sentiment daily nen co cac cot:

```text
date
article_count
sentiment_index
sentiment_index_z
positive_article_count
negative_article_count
neutral_article_count
```

Trong do:

- `sentiment_index`: diem sentiment trung binh cua cac bai bao trong ngay.
- `sentiment_index_z`: diem sentiment da chuan hoa z-score.
- `article_count`: so luong bai bao trong ngay.

Nen uu tien dung `sentiment_index_z` khi phan tich vi no da dua sentiment ve scale co y nghia:

```text
z = 0   : muc sentiment binh thuong
z > 0   : sentiment tich cuc hon binh thuong
z < 0   : sentiment tieu cuc hon binh thuong
z > 1   : tich cuc ro hon binh thuong
z < -1  : tieu cuc ro hon binh thuong
```

### 1.2. Sentiment index weekly

File sentiment weekly nen co cac cot:

```text
week_start
week_end
article_count
sentiment_index
sentiment_index_z
positive_article_count
negative_article_count
neutral_article_count
```

Weekly index thuong it nhieu hon daily, phu hop de phan tich tac dong trung han.

### 1.3. Du lieu VN-Index

Can co file gia VN-Index voi cac cot toi thieu:

```text
date
close
volume
```

Neu co them `open`, `high`, `low` thi cang tot, nhung chua bat buoc.

Luu y: `close` chu yeu dung de tinh return. Khi phan tich tac dong trong tai chinh, nen tap trung vao `return`, `future_return`, cumulative return, hoac abnormal return thay vi chi nhin truc tiep vao level gia/chi so.

## 2. Tinh Return Cho VN-Index

Nen dung log return:

```text
return_t = log(close_t / close_t-1)
```

Sau do tao cac bien future return:

```text
future_ret_1d  = log(close_t+1  / close_t)
future_ret_5d  = log(close_t+5  / close_t)
future_ret_20d = log(close_t+20 / close_t)
```

Y nghia:

- `future_ret_1d`: VN-Index tang/giam trong ngay giao dich tiep theo.
- `future_ret_5d`: VN-Index tang/giam trong 5 ngay giao dich tiep theo.
- `future_ret_20d`: VN-Index tang/giam trong 20 ngay giao dich tiep theo.

Code mau:

```python
import numpy as np
import pandas as pd

vnindex = pd.read_parquet("data1/vnindex_price.parquet")
vnindex["date"] = pd.to_datetime(vnindex["date"]).dt.normalize()
vnindex = vnindex.sort_values("date").reset_index(drop=True)

vnindex["return_1d"] = np.log(vnindex["close"] / vnindex["close"].shift(1))
vnindex["future_ret_1d"] = np.log(vnindex["close"].shift(-1) / vnindex["close"])
vnindex["future_ret_5d"] = np.log(vnindex["close"].shift(-5) / vnindex["close"])
vnindex["future_ret_20d"] = np.log(vnindex["close"].shift(-20) / vnindex["close"])

vnindex["volatility_20d"] = vnindex["return_1d"].rolling(20).std()
vnindex["return_lag_1d"] = vnindex["return_1d"].shift(1)
```

## 3. Xu Ly Ngay Giao Dich Va Ngay Tin Tuc

Sentiment daily cua ban co the co ca ngay khong giao dich, vi bao van dang tin vao cuoi tuan hoac ngay nghi.

Co 2 cach xu ly:

### Cach 1. Chi merge nhung ngay trung voi ngay giao dich

Cach nay don gian:

```python
df = vnindex.merge(sentiment_daily, on="date", how="inner")
```

Uu diem:

- De lam.
- It gay tranh cai.

Nhuoc diem:

- Bo qua sentiment vao cuoi tuan hoac ngay nghi.

### Cach 2. Day sentiment ngay nghi sang ngay giao dich ke tiep

Cach nay phu hop hon neu ban muon do tac dong cua tin tuc len phien giao dich tiep theo.

Y tuong:

- Tin thu Bay, Chu Nhat duoc gan vao thu Hai.
- Tin ngay nghi le duoc gan vao phien giao dich tiep theo.

Code mau:

```python
sentiment_daily = sentiment_daily.sort_values("date")
trading_calendar = vnindex[["date"]].sort_values("date")

sentiment_daily = pd.merge_asof(
    sentiment_daily,
    trading_calendar.rename(columns={"date": "trading_date"}),
    left_on="date",
    right_on="trading_date",
    direction="forward",
)

sentiment_daily = (
    sentiment_daily
    .groupby("trading_date", as_index=False)
    .agg(
        article_count=("article_count", "sum"),
        sentiment_index=("sentiment_index", "mean"),
        sentiment_index_z=("sentiment_index_z", "mean"),
        positive_article_count=("positive_article_count", "sum"),
        negative_article_count=("negative_article_count", "sum"),
        neutral_article_count=("neutral_article_count", "sum"),
    )
    .rename(columns={"trading_date": "date"})
)
```

Neu du lieu tin tuc khong co gio dang bai, cach an toan la dung sentiment ngay `t` de du bao return tuong lai `t+1`, `t+5`, `t+20`.

## 4. Tao Bien Sentiment De Phan Tich

Nen tao them cac bien:

```text
sentiment_z
sentiment_z_lag_1d
sentiment_z_lag_2d
sentiment_z_ma_5d
high_sentiment_dummy
low_sentiment_dummy
```

Code mau:

```python
df["sentiment_z"] = df["sentiment_index_z"]
df["sentiment_z_lag_1d"] = df["sentiment_z"].shift(1)
df["sentiment_z_lag_2d"] = df["sentiment_z"].shift(2)
df["sentiment_z_ma_5d"] = df["sentiment_z"].rolling(5).mean()

df["high_sentiment"] = (df["sentiment_z"] > 1).astype(int)
df["low_sentiment"] = (df["sentiment_z"] < -1).astype(int)
```

Goi y:

- Dung `sentiment_z` de do tac dong truc tiep cua sentiment ngay hien tai.
- Dung `sentiment_z_lag_1d` neu muon chac chan khong co look-ahead bias.
- Dung `sentiment_z_ma_5d` neu sentiment daily nhieu nhieu.

## 5. Phan Tich Mo Ta Truoc Khi Chay Model

Truoc khi hoi quy, nen kiem tra:

```python
print(df[["sentiment_z", "future_ret_1d", "future_ret_5d", "future_ret_20d"]].corr())
```

Nen ve chart:

- `sentiment_z` theo thoi gian.
- VN-Index `return_1d` theo thoi gian.
- VN-Index cumulative return theo thoi gian.
- `sentiment_z` va `future_ret_5d`.
- `sentiment_z` va abnormal/excess return neu co.
- Rolling mean cua sentiment 20 ngay.

Goi y dien giai:

- Neu `sentiment_z` cao thuong di kem `future_ret_5d` cao, sentiment co the co kha nang du bao tich cuc.
- Neu `sentiment_z` cao nhung `future_ret_5d` thap, co the sentiment dang phan anh su hung phan qua muc.

## 6. Regression Tac Dong Den VN-Index

Mo hinh co ban:

```text
future_return_t+h = alpha
                  + beta * sentiment_index_z_t
                  + gamma1 * return_lag_1d
                  + gamma2 * volatility_20d
                  + gamma3 * log_article_count
                  + error
```

Trong do:

- `beta > 0`: sentiment tich cuc hon binh thuong co lien quan den return tuong lai cao hon.
- `beta < 0`: sentiment tich cuc hon binh thuong co the bao hieu thi truong sap dieu chinh.
- `h`: horizon du bao, vi du 1 ngay, 5 ngay, 20 ngay.

Code mau:

```python
import numpy as np
import statsmodels.api as sm

df["log_article_count"] = np.log1p(df["article_count"])

reg_df = df[
    [
        "future_ret_5d",
        "sentiment_z",
        "return_lag_1d",
        "volatility_20d",
        "log_article_count",
    ]
].dropna()

y = reg_df["future_ret_5d"]
X = reg_df[
    [
        "sentiment_z",
        "return_lag_1d",
        "volatility_20d",
        "log_article_count",
    ]
]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
print(model.summary())
```

Nen dung `HAC/Newey-West standard errors` vi du lieu time-series thuong co:

- Tu tuong quan theo thoi gian.
- Phuong sai thay doi.

Voi `future_ret_1d`, co the dung:

```python
cov_kwds={"maxlags": 1}
```

Voi `future_ret_5d`, co the dung:

```python
cov_kwds={"maxlags": 5}
```

Voi `future_ret_20d`, co the dung:

```python
cov_kwds={"maxlags": 20}
```

## 7. Co Nen Dung Abnormal Return Cho VN-Index Khong?

Co the dung, nhung can hieu dung vi VN-Index chinh la chi so thi truong.

Voi co phieu rieng le:

```text
abnormal_return_i,t = stock_return_i,t - vnindex_return_t
```

Con voi VN-Index, khong the lay VN-Index tru chinh no. Neu muon tao abnormal return cho VN-Index, co the dung mot trong cac cach:

### Cach 1. Excess return so voi lai suat phi rui ro

```text
vnindex_excess_return_t = vnindex_return_t - risk_free_rate_t
```

Phu hop neu ban co lai suat tin phieu, lai suat lien ngan hang, hoac proxy lai suat phi rui ro.

### Cach 2. Unexpected return so voi rolling mean

```text
vnindex_abnormal_return_t = vnindex_return_t - rolling_mean_return_60d
```

Code mau:

```python
df["expected_return_60d"] = df["return_1d"].rolling(60).mean().shift(1)
df["abnormal_return_1d"] = df["return_1d"] - df["expected_return_60d"]
```

Luu y `.shift(1)` de khong dung du lieu ngay hien tai vao expected return.

### Cach 3. Expected return tu mo hinh AR

Mo hinh don gian:

```text
return_t = alpha + beta * return_t-1 + error_t
```

Sau do:

```text
abnormal_return_t = actual_return_t - expected_return_t
```

Voi project hien tai, nen bat dau bang:

```text
future_ret_1d
future_ret_5d
future_ret_20d
```

Sau do lam them robustness voi:

```text
abnormal_return_1d = return_1d - rolling_mean_return_60d
```

## 8. Event Study Theo Sentiment Cao/Thap

Event study giup dien giai rat truc quan.

Dinh nghia event:

```text
High sentiment event: sentiment_z > 1
Low sentiment event : sentiment_z < -1
```

Hoac dung percentile:

```text
High sentiment event: top 20% sentiment_z
Low sentiment event : bottom 20% sentiment_z
```

Sau do tinh return trung binh sau event:

```text
CAR_1d  = future_ret_1d
CAR_5d  = future_ret_5d
CAR_20d = future_ret_20d
```

Bang can tao:

```text
event_group       avg_future_ret_1d   avg_future_ret_5d   avg_future_ret_20d   event_count
high_sentiment
normal_sentiment
low_sentiment
```

Code mau:

```python
conditions = [
    df["sentiment_z"] > 1,
    df["sentiment_z"] < -1,
]
choices = ["high_sentiment", "low_sentiment"]

df["sentiment_event_group"] = np.select(
    conditions,
    choices,
    default="normal_sentiment",
)

event_summary = (
    df.groupby("sentiment_event_group")
    .agg(
        event_count=("date", "size"),
        avg_future_ret_1d=("future_ret_1d", "mean"),
        avg_future_ret_5d=("future_ret_5d", "mean"),
        avg_future_ret_20d=("future_ret_20d", "mean"),
    )
    .reset_index()
)

print(event_summary)
```

Neu `high_sentiment` co `avg_future_ret_5d` lon hon `low_sentiment`, sentiment co the co tac dong tich cuc den thi truong.

Neu `high_sentiment` co `avg_future_ret_5d` thap hon, co the sentiment cao dang phan anh su hung phan qua muc va thi truong co xu huong dieu chinh.

## 9. Weekly Analysis

Weekly analysis tuong tu daily, nhung dung du lieu theo tuan.

Can tao weekly return:

```python
vnindex_weekly = (
    vnindex
    .set_index("date")
    .resample("W-SUN")
    .agg(
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    .dropna()
    .reset_index()
)

vnindex_weekly["week_end"] = vnindex_weekly["date"].dt.normalize()
vnindex_weekly["weekly_return"] = np.log(
    vnindex_weekly["close"] / vnindex_weekly["close"].shift(1)
)
vnindex_weekly["future_ret_1w"] = np.log(
    vnindex_weekly["close"].shift(-1) / vnindex_weekly["close"]
)
vnindex_weekly["future_ret_4w"] = np.log(
    vnindex_weekly["close"].shift(-4) / vnindex_weekly["close"]
)
```

Merge voi weekly sentiment:

```python
weekly_df = vnindex_weekly.merge(
    sentiment_weekly,
    on="week_end",
    how="inner",
)
```

Regression weekly:

```text
future_ret_1w = alpha + beta * sentiment_index_z + controls + error
future_ret_4w = alpha + beta * sentiment_index_z + controls + error
```

Weekly thuong tot hon daily neu:

- Tin tuc daily qua nhieu nhieu.
- Article count tung ngay khong deu.
- Ban muon xem tac dong trung han.

## 10. Robustness Check Nen Lam

Nen kiem tra ket qua co ben vung khong bang cac cach:

1. Dung `sentiment_index` thay cho `sentiment_index_z`.
2. Dung `sentiment_z_lag_1d` thay cho `sentiment_z`.
3. Loc ngay co `article_count >= 10`.
4. Chay rieng daily va weekly.
5. Thu cac horizon khac nhau: 1d, 5d, 20d.
6. Winsorize return o muc 1% va 99% de giam anh huong outlier.
7. Them bien control: volatility, volume change, past return.

Code winsorize mau:

```python
def winsorize_series(series, lower=0.01, upper=0.99):
    lower_value = series.quantile(lower)
    upper_value = series.quantile(upper)
    return series.clip(lower_value, upper_value)

df["future_ret_5d_w"] = winsorize_series(df["future_ret_5d"])
```

## 11. Cach Dien Giai Ket Qua

Khi doc regression, tap trung vao:

```text
coef cua sentiment_z
p-value cua sentiment_z
R-squared
so quan sat
```

Vi `sentiment_z` la z-score, nen:

> He so beta cho biet khi sentiment tang 1 do lech chuan so voi muc binh thuong, future return cua VN-Index thay doi bao nhieu.

Vi du:

```text
beta = 0.003
```

Nghia la khi sentiment tang 1 standard deviation, `future_ret_5d` trung binh tang khoang:

```text
0.3%
```

Neu:

```text
p-value < 0.05
```

thi ket qua co y nghia thong ke o muc 5%.

Neu:

```text
p-value > 0.10
```

thi chua co bang chung thong ke ro rang.

## 12. Nhung Loi Can Tranh

### 12.1. Look-ahead bias

Khong nen dung sentiment cua tuong lai de giai thich return hien tai.

An toan nhat:

```text
sentiment_t -> future_return_t+1:t+h
```

### 12.2. Dung same-day return khi khong co gio dang bai

Neu khong co gio dang bai, khong chac tin xuat hien truoc hay sau gio giao dich.

Nen uu tien:

```text
future_ret_1d
future_ret_5d
future_ret_20d
```

### 12.3. Ngay co qua it bai bao

Ngay chi co 1 hoac 2 bai co the lam sentiment index cuc doan.

Nen thu them filter:

```text
article_count >= 10
```

### 12.4. Chi nhin accuracy cua model sentiment

Model sentiment tot chua chac sentiment index co kha nang du bao thi truong.

Phan tich tac dong can kiem tra rieng bang:

- Correlation.
- Regression.
- Event study.
- Robustness check.

## 13. Thu Tu Lam Khuyen Nghi

Nen lam theo thu tu:

1. Chuan bi VN-Index price va tinh return.
2. Merge daily sentiment voi VN-Index.
3. Phan tich correlation.
4. Chay regression cho `future_ret_1d`, `future_ret_5d`, `future_ret_20d`.
5. Lam event study voi `sentiment_z > 1` va `sentiment_z < -1`.
6. Lap lai voi weekly sentiment.
7. Chay robustness check.
8. Viet ket luan ve dau, do lon va y nghia thong ke cua tac dong.

## 14. Output Nen Luu

Trong bao cao, nen trinh bay toi thieu:

- Chart sentiment index z-score theo thoi gian.
- Chart VN-Index return/cumulative return va sentiment index.
- Bang correlation.
- Bang regression.
- Bang event study high/low sentiment.
- Ket luan daily va weekly co dong nhat hay khong.
