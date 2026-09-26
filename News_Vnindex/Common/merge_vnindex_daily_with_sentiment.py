"""Merge chỉ số sentiment cấp-ngày (schema: date, article_count,
sentiment_index, sentiment_index_z - output của
News/Build_sentiment_index/build_sentiment_index_*.py) với dữ liệu VN-Index
cấp ngày. Logic dùng chung cho MỌI phương pháp sentiment: mỗi thư mục method
(News_Vnindex/{Lexicon,Transfer_Learning,Traditional_ML}/daily/) chỉ có 1
script mỏng khai báo {method: file chỉ số ngày} của riêng nó rồi gọi
``run_merge`` ở đây.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VNINDEX_DAILY_PATH = PROJECT_ROOT / "data_Histo" / "vnindex_eda_output.csv"
SENTIMENT_INDEX_DIR = PROJECT_ROOT / "News" / "Build_sentiment_index" / "data"
OUTPUT_DIR = PROJECT_ROOT / "data_News"

SYMBOL_COLUMN = "symbol"
DATE_COLUMN = "date"
CLOSE_COLUMN = "close_price"
OPEN_COLUMN = "open_price"
HIGH_COLUMN = "high_price"
LOW_COLUMN = "low_price"
VOLUME_COLUMN = "vol_total"
VALUE_COLUMN = "val_total"
BUY_VOL_FOREIGN_COLUMN = "buy_vol_foreign"
SELL_VOL_FOREIGN_COLUMN = "sell_vol_foreign"

SENTIMENT_SCORE_COLUMN = "sentiment_index"
ARTICLE_COUNT_COLUMN = "article_count"


def standardize_series(values: pd.Series) -> pd.Series:
    standard_deviation = values.std()
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / standard_deviation


def prepare_vnindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if SYMBOL_COLUMN in out.columns:
        out[SYMBOL_COLUMN] = out[SYMBOL_COLUMN].astype("string").str.strip().str.upper()
        out = out.loc[out[SYMBOL_COLUMN].eq("VNINDEX")].copy()

    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    for column in [
        OPEN_COLUMN,
        HIGH_COLUMN,
        LOW_COLUMN,
        CLOSE_COLUMN,
        VOLUME_COLUMN,
        VALUE_COLUMN,
        BUY_VOL_FOREIGN_COLUMN,
        SELL_VOL_FOREIGN_COLUMN,
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.loc[
        out[DATE_COLUMN].notna()
        & out[CLOSE_COLUMN].notna()
        & out[CLOSE_COLUMN].gt(0)
    ].copy()
    out = out.sort_values(DATE_COLUMN, kind="mergesort")
    out = out.drop_duplicates(subset=[DATE_COLUMN], keep="last").reset_index(drop=True)

    out["daily_return"] = np.log(out[CLOSE_COLUMN] / out[CLOSE_COLUMN].shift(1))
    out["future_ret_1d"] = np.log(out[CLOSE_COLUMN].shift(-1) / out[CLOSE_COLUMN])
    out["future_ret_5d"] = np.log(out[CLOSE_COLUMN].shift(-5) / out[CLOSE_COLUMN])
    out["future_ret_20d"] = np.log(out[CLOSE_COLUMN].shift(-20) / out[CLOSE_COLUMN])
    out["return_lag_1d"] = out["daily_return"].shift(1)
    # .shift(1): Exog phải là thông tin QUÁ KHỨ (đúng Exog_{t-1} trong
    # Tetlock) - không .shift thì cửa sổ rolling 20 ngày tính đến CẢ ngày t,
    # nghĩa là dùng 1 phần chính return ngày t (biến đang muốn dự báo) làm
    # biến kiểm soát cho chính nó - look-ahead nhẹ, đã phát hiện khi đối
    # chiếu lại REF/Tetlock_Media_Sentiment_JF.pdf.
    out["volatility_20d"] = out["daily_return"].rolling(20).std().shift(1)
    out["log_vol_total"] = np.log1p(out[VOLUME_COLUMN])
    out["log_val_total"] = np.log1p(out[VALUE_COLUMN])

    # Khối ngoại (buy_vol_foreign/sell_vol_foreign) - có sẵn trong dữ liệu
    # gốc nhưng trước giờ chưa được đưa vào pipeline merge/regression. Khối
    # nội = tổng volume trừ đi khối ngoại (xấp xỉ, không phải cột gốc).
    has_foreign_columns = BUY_VOL_FOREIGN_COLUMN in out.columns and SELL_VOL_FOREIGN_COLUMN in out.columns
    if has_foreign_columns:
        out["foreign_vol_total"] = out[BUY_VOL_FOREIGN_COLUMN] + out[SELL_VOL_FOREIGN_COLUMN]
        out["foreign_vol_net"] = out[BUY_VOL_FOREIGN_COLUMN] - out[SELL_VOL_FOREIGN_COLUMN]
        out["domestic_vol_total"] = (out[VOLUME_COLUMN] - out["foreign_vol_total"]).clip(lower=0)
        out["log_foreign_vol_total"] = np.log1p(out["foreign_vol_total"])
        out["log_domestic_vol_total"] = np.log1p(out["domestic_vol_total"])

    output_columns = [
        DATE_COLUMN,
        OPEN_COLUMN,
        HIGH_COLUMN,
        LOW_COLUMN,
        CLOSE_COLUMN,
        VOLUME_COLUMN,
        VALUE_COLUMN,
        "daily_return",
        "future_ret_1d",
        "future_ret_5d",
        "future_ret_20d",
        "return_lag_1d",
        "volatility_20d",
        "log_vol_total",
        "log_val_total",
    ]
    if has_foreign_columns:
        output_columns += [
            "foreign_vol_total",
            "foreign_vol_net",
            "domestic_vol_total",
            "log_foreign_vol_total",
            "log_domestic_vol_total",
        ]
    return out[output_columns]


def map_to_next_trading_date(
    sentiment_dates: pd.Series,
    trading_dates: pd.Series,
) -> pd.Series:
    trading_date_values = pd.to_datetime(trading_dates).sort_values().to_numpy(
        dtype="datetime64[ns]"
    )
    sentiment_date_values = pd.to_datetime(sentiment_dates).to_numpy(dtype="datetime64[ns]")

    # Strictly next trading date: news on day t is used from the next market session.
    positions = np.searchsorted(
        trading_date_values,
        sentiment_date_values,
        side="right",
    )

    mapped_dates = np.full(len(sentiment_date_values), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_positions = positions < len(trading_date_values)
    mapped_dates[valid_positions] = trading_date_values[positions[valid_positions]]
    return pd.Series(pd.to_datetime(mapped_dates), index=sentiment_dates.index)


def prepare_effective_sentiment(
    sentiment_df: pd.DataFrame,
    trading_dates: pd.Series,
) -> pd.DataFrame:
    out = sentiment_df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    out[ARTICLE_COUNT_COLUMN] = pd.to_numeric(out[ARTICLE_COUNT_COLUMN], errors="coerce")
    out[SENTIMENT_SCORE_COLUMN] = pd.to_numeric(out[SENTIMENT_SCORE_COLUMN], errors="coerce")

    out = out.loc[
        out[DATE_COLUMN].notna()
        & out[ARTICLE_COUNT_COLUMN].notna()
        & out[ARTICLE_COUNT_COLUMN].gt(0)
        & out[SENTIMENT_SCORE_COLUMN].notna()
    ].copy()

    out["effective_trading_date"] = map_to_next_trading_date(
        out[DATE_COLUMN],
        trading_dates,
    )
    out = out.loc[out["effective_trading_date"].notna()].copy()
    out["weighted_sentiment"] = out[SENTIMENT_SCORE_COLUMN] * out[ARTICLE_COUNT_COLUMN]

    effective_sentiment = out.groupby("effective_trading_date", sort=True).agg(
        sentiment_calendar_start=(DATE_COLUMN, "min"),
        sentiment_calendar_end=(DATE_COLUMN, "max"),
        source_calendar_day_count=(DATE_COLUMN, "size"),
        article_count=(ARTICLE_COUNT_COLUMN, "sum"),
        weighted_sentiment=("weighted_sentiment", "sum"),
    )
    effective_sentiment = effective_sentiment.reset_index()
    effective_sentiment["sentiment_index"] = (
        effective_sentiment["weighted_sentiment"] / effective_sentiment["article_count"]
    )
    effective_sentiment["sentiment_index_z"] = standardize_series(
        effective_sentiment["sentiment_index"]
    )
    effective_sentiment["log_article_count"] = np.log1p(
        effective_sentiment["article_count"]
    )

    output_columns = [
        "effective_trading_date",
        "sentiment_calendar_start",
        "sentiment_calendar_end",
        "source_calendar_day_count",
        "article_count",
        "sentiment_index",
        "sentiment_index_z",
        "log_article_count",
    ]
    return effective_sentiment[output_columns]


def merge_vnindex_daily_with_sentiment(
    vnindex_daily_df: pd.DataFrame,
    sentiment_daily_df: pd.DataFrame,
) -> pd.DataFrame:
    vnindex_daily = prepare_vnindex_daily(vnindex_daily_df)
    effective_sentiment = prepare_effective_sentiment(
        sentiment_daily_df,
        vnindex_daily[DATE_COLUMN],
    )

    merged_df = effective_sentiment.merge(
        vnindex_daily,
        left_on="effective_trading_date",
        right_on=DATE_COLUMN,
        how="inner",
    )
    merged_df = merged_df.drop(columns=[DATE_COLUMN])
    merged_df = merged_df.rename(columns={"effective_trading_date": DATE_COLUMN})

    ordered_columns = [
        DATE_COLUMN,
        "sentiment_calendar_start",
        "sentiment_calendar_end",
        "source_calendar_day_count",
        "article_count",
        "sentiment_index",
        "sentiment_index_z",
        "log_article_count",
        OPEN_COLUMN,
        HIGH_COLUMN,
        LOW_COLUMN,
        CLOSE_COLUMN,
        VOLUME_COLUMN,
        VALUE_COLUMN,
        "daily_return",
        "future_ret_1d",
        "future_ret_5d",
        "future_ret_20d",
        "return_lag_1d",
        "volatility_20d",
        "log_vol_total",
        "log_val_total",
        "foreign_vol_total",
        "foreign_vol_net",
        "domestic_vol_total",
        "log_foreign_vol_total",
        "log_domestic_vol_total",
    ]
    ordered_columns = [column for column in ordered_columns if column in merged_df.columns]
    merged_df = merged_df[ordered_columns]
    return merged_df.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)


def run_merge(sentiment_daily_paths_by_method: dict[str, Path]) -> None:
    """``{method_suffix: daily sentiment index parquet}`` -> ghi
    data_News/vnindex_daily_sentiment_merged_{method_suffix}.{parquet,csv}."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vnindex_daily_df = pd.read_csv(VNINDEX_DAILY_PATH, encoding="utf-8-sig")

    for method_suffix, sentiment_daily_path in sentiment_daily_paths_by_method.items():
        sentiment_daily_df = pd.read_parquet(sentiment_daily_path)
        merged_df = merge_vnindex_daily_with_sentiment(vnindex_daily_df, sentiment_daily_df)

        output_parquet_path = OUTPUT_DIR / f"vnindex_daily_sentiment_merged_{method_suffix}.parquet"
        output_csv_path = OUTPUT_DIR / f"vnindex_daily_sentiment_merged_{method_suffix}.csv"
        merged_df.to_parquet(output_parquet_path, index=False)
        merged_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

        print(f"--- {method_suffix} ---")
        print("Sentiment daily input:", sentiment_daily_path)
        print("Output parquet:", output_parquet_path)
        print("Sentiment daily rows:", len(sentiment_daily_df))
        print("Merged rows:", len(merged_df))
        print(merged_df.head(10).to_string(index=False))
        print()
