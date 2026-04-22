from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clean_historical_price import load_tabular_data, prepare_historical_price_data

PRICE_DATA_PATH = "parquet_all.parquet"


def load_price_data_for_eda(path: str | Path) -> pd.DataFrame:
    df = load_tabular_data(path)
    if "ticker" not in df.columns:
        out = prepare_historical_price_data(df)
    else:
        out = df.copy()
        out["symbol"] = out["symbol"].fillna("").astype(str).str.strip().str.upper()
        out["ticker"] = out["ticker"].fillna("").astype(str).str.strip().str.upper()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def build_missing_ratio(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "number of missing rows": df.isna().sum().values,
            "missing pct": (df.isna().mean() * 100).round(2).values,
        }
    )


def build_duplicate_ticker_date_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.loc[df.duplicated(subset=["ticker", "date"], keep=False)]
        .sort_values(["ticker", "date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_symbol_date_panel(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["symbol", "ticker", "date"], kind="mergesort").reset_index(drop=True)


def build_ticker_row_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("ticker", dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values("ticker", kind="mergesort")
        .reset_index(drop=True)
    )


def build_universe_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["symbol", "ticker"], dropna=False)
        .agg(
            row_count=("symbol", "size"),
            min_date=("date", "min"),
            max_date=("date", "max"),
            zero_volume_days=("vol_total", lambda s: int(s.fillna(0).eq(0).sum())),
        )
        .reset_index()
        .sort_values(["row_count", "symbol"], ascending=[False, True], kind="mergesort")
    )
    return summary


def build_coverage_by_year(df: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        df.groupby("year", dropna=False)
        .agg(
            row_count=("symbol", "size"),
            unique_symbols=("symbol", "nunique"),
            unique_tickers=("ticker", "nunique"),
            zero_volume_days=("vol_total", lambda s: int(s.fillna(0).eq(0).sum())),
        )
        .reset_index()
        .sort_values("year", kind="mergesort")
    )
    return coverage


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


def add_lag_features(
    df: pd.DataFrame,
    lag_columns: list[str],
    max_lag: int = 5,
) -> pd.DataFrame:
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


def print_preview(title: str, df: pd.DataFrame, rows: int = 10) -> None:
    print(f"\n{title}:")
    if df.empty:
        print("Empty")
        return
    print(df.head(rows).to_string(index=False))


def main() -> None:
    raw_df = load_tabular_data(PRICE_DATA_PATH)
    raw_columns = list(raw_df.columns)
    df = load_price_data_for_eda(PRICE_DATA_PATH)

    missing_ratio = build_missing_ratio(df[raw_columns])
    duplicate_ticker_date_rows = build_duplicate_ticker_date_rows(df)
    symbol_date_panel = build_symbol_date_panel(df)
    ticker_row_counts = build_ticker_row_counts(df)
    universe_by_symbol = build_universe_by_symbol(df)
    coverage_by_year = build_coverage_by_year(df)
    feature = build_feature(df)

    print("Rows:", len(df))
    print("Unique symbols:", df["symbol"].nunique(dropna=True))
    print("Unique tickers:", df["ticker"].replace("", pd.NA).nunique(dropna=True))
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Duplicate ticker-date rows:", len(duplicate_ticker_date_rows))

    print("Missing ratio (%)", missing_ratio)
    print_preview("Duplicate ticker-date rows", duplicate_ticker_date_rows)
    print_preview(
        "Symbol-date panel (sorted by symbol then date)",
        symbol_date_panel[
            [
                "symbol",
                "ticker",
                "date",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "vol_total",
                "val_total",
            ]
        ],
    )
    print_preview("Row count by ticker", ticker_row_counts)
    # print_preview("Universe by symbol", universe_by_symbol)
    # print_preview("Coverage by year", coverage_by_year)
    # print_preview("Feature preview", feature)


if __name__ == "__main__":
    main()
