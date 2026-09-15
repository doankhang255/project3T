"""
Cửa sổ (ROLLING/AR) quy đổi tương đương bản tuần (26 tuần/52 tuần) sang số
ngày giao dịch (~5 ngày/tuần): 26 tuần ~ 120 ngày, 52 tuần ~ 252 ngày.  
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATHS_BY_METHOD = {
    "pmi": PROJECT_ROOT / "data_News" / "vnindex_daily_sentiment_merged_pmi.parquet",
    "intensity": PROJECT_ROOT / "data_News" / "vnindex_daily_sentiment_merged_intensity.parquet",
    "pca_pmi": PROJECT_ROOT / "data_News" / "vnindex_daily_sentiment_merged_pca_pmi.parquet",
    "pca_intensity": PROJECT_ROOT / "data_News" / "vnindex_daily_sentiment_merged_pca_intensity.parquet",
}
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
    for suffix, input_path in INPUT_PATHS_BY_METHOD.items():
        merged_df = pd.read_parquet(input_path)
        abnormal_df = add_daily_abnormal_return(merged_df)

        output_parquet_path = OUTPUT_DIR / f"vnindex_daily_sentiment_abnormal_return_{suffix}.parquet"
        output_csv_path = OUTPUT_DIR / f"vnindex_daily_sentiment_abnormal_return_{suffix}.csv"
        abnormal_df.to_parquet(output_parquet_path, index=False)
        abnormal_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

        print(f"--- {suffix} ---")
        print("Input:", input_path)
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
        print()


if __name__ == "__main__":
    main()
