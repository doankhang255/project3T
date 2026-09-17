"""Bản sao cho phương pháp "phobert" của
News_Vnindex/Lexicon/daily/build_vnindex_daily_abnormal_return.py - cùng
logic tính abnormal return (rolling mean 120 ngày ~26 tuần, và AR(1) rolling
252 ngày ~52 tuần), chỉ đổi input/output sang bản merge của Transfer_Learning.
Xem docstring của bản Lexicon để biết lý do chọn 2 cách ước lượng expected
return này.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data_News" / "vnindex_daily_sentiment_merged_phobert.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data_News"

RETURN_COLUMN = "daily_return"
ROLLING_EXPECTED_WINDOW = 120  # ~26 tuần * 5 ngày giao dịch/tuần
ROLLING_EXPECTED_MIN_PERIODS = 60
AR_WINDOW = 252  # ~52 tuần * 5 ngày giao dịch/tuần
AR_MIN_OBS = 120


def sum_future_values(values: pd.Series, horizon: int) -> pd.Series:
    future_sum = values.shift(-1)
    for step in range(2, horizon + 1):
        future_sum = future_sum + values.shift(-step)
    return future_sum


def compute_rolling_ar1_expected_return(returns: pd.Series, window: int, min_obs: int) -> pd.Series:
    """Ước lượng expected return tại mỗi ngày t bằng hồi quy AR(1)
    (return_t ~ a + b*return_{t-1}) fit trên `window` ngày trước đó (không
    dùng dữ liệu tương lai, tránh look-ahead bias)."""
    returns = pd.to_numeric(returns, errors="coerce")
    lagged_returns = returns.shift(1)
    expected_values = pd.Series(np.nan, index=returns.index, dtype="float64")

    for index_position in range(len(returns)):
        start_position = max(0, index_position - window)
        train_df = pd.DataFrame(
            {
                "return": returns.iloc[start_position:index_position],
                "return_lag": lagged_returns.iloc[start_position:index_position],
            }
        ).dropna()

        current_lag = lagged_returns.iloc[index_position]
        if len(train_df) < min_obs or pd.isna(current_lag):
            continue

        x = np.column_stack(
            [np.ones(len(train_df)), train_df["return_lag"].to_numpy(dtype="float64")]
        )
        y = train_df["return"].to_numpy(dtype="float64")
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        expected_values.iloc[index_position] = beta[0] + beta[1] * current_lag

    return expected_values


def add_daily_abnormal_return(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("date", kind="mergesort").reset_index(drop=True)
    out[RETURN_COLUMN] = pd.to_numeric(out[RETURN_COLUMN], errors="coerce")

    out["expected_return_rolling"] = (
        out[RETURN_COLUMN]
        .rolling(ROLLING_EXPECTED_WINDOW, min_periods=ROLLING_EXPECTED_MIN_PERIODS)
        .mean()
        .shift(1)
    )
    out["abnormal_return_rolling_1d"] = out[RETURN_COLUMN] - out["expected_return_rolling"]
    out["future_abnormal_rolling_ret_1d"] = out["abnormal_return_rolling_1d"].shift(-1)
    out["future_abnormal_rolling_ret_5d"] = sum_future_values(out["abnormal_return_rolling_1d"], horizon=5)

    out["expected_return_ar1"] = compute_rolling_ar1_expected_return(
        out[RETURN_COLUMN], window=AR_WINDOW, min_obs=AR_MIN_OBS
    )
    out["abnormal_return_ar1_1d"] = out[RETURN_COLUMN] - out["expected_return_ar1"]
    out["future_abnormal_ar1_ret_1d"] = out["abnormal_return_ar1_1d"].shift(-1)
    out["future_abnormal_ar1_ret_5d"] = sum_future_values(out["abnormal_return_ar1_1d"], horizon=5)

    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_df = pd.read_parquet(INPUT_PATH)
    abnormal_df = add_daily_abnormal_return(merged_df)

    output_parquet_path = OUTPUT_DIR / "vnindex_daily_sentiment_abnormal_return_phobert.parquet"
    output_csv_path = OUTPUT_DIR / "vnindex_daily_sentiment_abnormal_return_phobert.csv"
    abnormal_df.to_parquet(output_parquet_path, index=False)
    abnormal_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print("Input:", INPUT_PATH)
    print("Output parquet:", output_parquet_path)
    print("Rows:", len(abnormal_df))
    print(
        abnormal_df[
            [
                "date",
                "daily_return",
                "expected_return_rolling",
                "abnormal_return_rolling_1d",
                "expected_return_ar1",
                "abnormal_return_ar1_1d",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
