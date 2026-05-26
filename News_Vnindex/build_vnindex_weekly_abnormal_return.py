from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_merged.parquet"
OUTPUT_PARQUET_PATH = PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_abnormal_return.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_abnormal_return.csv"

RETURN_COLUMN = "weekly_return"
ROLLING_EXPECTED_WINDOW = 26
ROLLING_EXPECTED_MIN_PERIODS = 12
AR_WINDOW = 52
AR_MIN_OBS = 26

 
def sum_future_values(values: pd.Series, horizon: int) -> pd.Series:
    future_sum = values.shift(-1)
    for step in range(2, horizon + 1):
        future_sum = future_sum + values.shift(-step)
    return future_sum


def compute_rolling_ar1_expected_return(returns: pd.Series, window: int = AR_WINDOW, min_obs: int = AR_MIN_OBS,) -> pd.Series:
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
            [
                np.ones(len(train_df)),
                train_df["return_lag"].to_numpy(dtype="float64"),
            ]
        )
        y = train_df["return"].to_numpy(dtype="float64")
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        expected_values.iloc[index_position] = beta[0] + beta[1] * current_lag

    return expected_values


def add_weekly_abnormal_return(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("week_end", kind="mergesort").reset_index(drop=True)
    out[RETURN_COLUMN] = pd.to_numeric(out[RETURN_COLUMN], errors="coerce")

    out["expected_return_26w"] = (
        out[RETURN_COLUMN]
        .rolling(ROLLING_EXPECTED_WINDOW, min_periods=ROLLING_EXPECTED_MIN_PERIODS,)
        .mean()
        .shift(1)
    )
    out["abnormal_return_rolling_1w"] = (out[RETURN_COLUMN] - out["expected_return_26w"])

    out["future_abnormal_rolling_ret_1w"] = out["abnormal_return_rolling_1w"].shift(-1)

    out["future_abnormal_rolling_ret_4w"] = sum_future_values(out["abnormal_return_rolling_1w"], horizon=4,)

    out["expected_return_ar1_52w"] = compute_rolling_ar1_expected_return(
        out[RETURN_COLUMN],
        window=AR_WINDOW,
        min_obs=AR_MIN_OBS,
    )

    out["abnormal_return_ar1_1w"] = (out[RETURN_COLUMN] - out["expected_return_ar1_52w"])

    out["future_abnormal_ar1_ret_1w"] = out["abnormal_return_ar1_1w"].shift(-1)

    out["future_abnormal_ar1_ret_4w"] = sum_future_values(out["abnormal_return_ar1_1w"], horizon=4,)

    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    merged_df = pd.read_parquet(INPUT_PATH)
    abnormal_df = add_weekly_abnormal_return(merged_df)

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    abnormal_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    abnormal_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input:", INPUT_PATH)
    print("Output parquet:", OUTPUT_PARQUET_PATH)
    print("Input rows:", len(merged_df))
    print("Output rows:", len(abnormal_df))
    print(
        abnormal_df[
            [
                "week_end",
                "weekly_return",
                "expected_return_26w",
                "abnormal_return_rolling_1w",
                "expected_return_ar1_52w",
                "abnormal_return_ar1_1w",
                "future_abnormal_rolling_ret_4w",
                "future_abnormal_ar1_ret_4w",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
