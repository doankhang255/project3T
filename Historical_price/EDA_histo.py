from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clean_EDA import build_clean_eda_data

FEATURE_DATA_PATH = Path("historical_price_feature.parquet")


def add_rolling_return_volatility(
    df: pd.DataFrame,
    window: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    out = df.copy()
    out["rolling_std_ret_20"] = (
        out.groupby("ticker", dropna=False)["ret_0"]
        .transform(lambda s: s.rolling(window, min_periods=min_periods).std())
    )
    return out


def add_lag_features(df: pd.DataFrame, lag_columns: list[str],  max_lag: int = 5) -> pd.DataFrame:
    out = df.copy()
    ticker_group = out.groupby("ticker", dropna=False)

    for column in lag_columns:
        for lag in range(1, max_lag + 1):
            out[f"{column}_lag{lag}"] = ticker_group[column].shift(lag)
    return out


def build_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    ticker_group = out.groupby("ticker", dropna=False)

    previous_close = ticker_group["close_price"].shift(1)
    next_close_1 = ticker_group["close_price"].shift(-1)
    next_close_3 = ticker_group["close_price"].shift(-3)
    next_close_5 = ticker_group["close_price"].shift(-5)

    out["ret_0"] = np.log(out["close_price"] / previous_close)
    out["ret_1"] = np.log(next_close_1 / out["close_price"])
    out["ret_3"] = np.log(next_close_3 / out["close_price"])
    out["ret_5"] = np.log(next_close_5 / out["close_price"])

    out["log_vol_total"] = np.log1p(out["vol_total"].clip(lower=0))
    rolling_log_vol_mean = ticker_group["log_vol_total"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).mean()
    )
    out["abn_vol"] = out["log_vol_total"] - rolling_log_vol_mean
    out["intraday_range_ratio"] = (out["high_price"] - out["low_price"]) / out["close_price"]
    out = add_rolling_return_volatility(out, window=20, min_periods=10)
    out = add_lag_features(
        out,
        lag_columns=["ret_0", "log_vol_total", "rolling_std_ret_20"],
        max_lag=5,
    )
    return out[
        [
            "ticker",
            "date",
            "close_price",
            "ret_0",
            "ret_1",
            "ret_3",
            "ret_5",
            "rolling_std_ret_20",
            "log_vol_total",
            "abn_vol",
            "intraday_range_ratio",
            "ret_0_lag1",
            "ret_0_lag2",
            "ret_0_lag3",
            "ret_0_lag4",
            "ret_0_lag5",
            "log_vol_total_lag1",
            "log_vol_total_lag2",
            "log_vol_total_lag3",
            "log_vol_total_lag4",
            "log_vol_total_lag5",
            "rolling_std_ret_20_lag1",
            "rolling_std_ret_20_lag2",
            "rolling_std_ret_20_lag3",
            "rolling_std_ret_20_lag4",
            "rolling_std_ret_20_lag5",
        ]
    ].copy()


def build_feature_data() -> pd.DataFrame:
    result = build_clean_eda_data()
    clean_df = result["clean_eda_data"].copy()
    return build_feature(clean_df)


def main() -> None:
    feature_df = build_feature_data()
    feature_df.to_parquet(FEATURE_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
