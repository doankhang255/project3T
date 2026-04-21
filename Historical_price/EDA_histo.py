from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clean_historical_price import load_tabular_data, prepare_historical_price_data

# Point this to the raw historical price file.
PRICE_DATA_PATH = ""


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
    missing_ratio = (df.isna().mean() * 100).round(2)
    return (
        missing_ratio.rename("missing_ratio_pct")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_ratio_pct", ascending=False)
        .reset_index(drop=True)
    )


def build_duplicate_ticker_date_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.loc[df.duplicated(subset=["ticker", "date"], keep=False)]
        .sort_values(["ticker", "date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_universe_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["symbol", "ticker"], dropna=False)
        .agg(
            row_count=("symbol", "size"),
            min_date=("date", "min"),
            max_date=("date", "max"),
            avg_close_price=("close_price", "mean"),
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
            avg_daily_volume=("vol_total", "mean"),
        )
        .reset_index()
        .sort_values("year", kind="mergesort")
    )
    return coverage


def build_liquidity_summary(df: pd.DataFrame) -> pd.DataFrame:
    liquidity_columns = [
        "vol_total",
        "vol_deal",
        "val_total",
        "buy_count",
        "buy_vol",
        "sell_count",
        "sell_vol",
    ]
    summary = (
        df[liquidity_columns]
        .agg(["count", "mean", "median", "min", "max"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "column"})
    )
    return summary


def build_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for column in ["close_price", "vol_total", "val_total"]:
        series = df[column].dropna()
        if series.empty:
            metrics.append(
                {
                    "column": column,
                    "p95": np.nan,
                    "p99": np.nan,
                    "rows_gt_p99": 0,
                }
            )
            continue

        p95 = float(series.quantile(0.95))
        p99 = float(series.quantile(0.99))
        metrics.append(
            {
                "column": column,
                "p95": p95,
                "p99": p99,
                "rows_gt_p99": int((df[column] > p99).sum()),
            }
    )
    return pd.DataFrame(metrics)


def build_feature_preview(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    symbol_group = out.groupby("symbol", dropna=False)
    buy_val_column = "buy_val_foreign" if "buy_val_foreign" in out.columns else "buy_val_foreigh"
    sell_val_column = "sell_val_foreign" if "sell_val_foreign" in out.columns else "sel_val_foreign"

    previous_close = symbol_group["close_price"].shift(1)
    next_close_1 = symbol_group["close_price"].shift(-1)
    next_close_3 = symbol_group["close_price"].shift(-3)
    next_close_5 = symbol_group["close_price"].shift(-5)

    out["ret_0"] = np.log(out["close_price"] / previous_close)
    out["ret_1"] = np.log(next_close_1 / out["close_price"])
    out["ret_3"] = np.log(next_close_3 / out["close_price"])
    out["ret_5"] = np.log(next_close_5 / out["close_price"])

    out["log_vol_total"] = np.log1p(out["vol_total"].clip(lower=0))
    out["intraday_range_ratio"] = (out["high_price"] - out["low_price"]) / out["close_price"]
    out["foreign_net_vol"] = out["buy_vol_foreign"].fillna(0) - out["sell_vol_foreign"].fillna(0)
    out["foreign_net_value"] = out[buy_val_column].fillna(0) - out[sell_val_column].fillna(0)
    out["foreign_net_ratio"] = out["foreign_net_vol"] / out["vol_total"].replace({0: np.nan})
    return out


def print_preview(title: str, df: pd.DataFrame, rows: int = 10) -> None:
    print(f"\n{title}:")
    if df.empty:
        print("Empty")
        return
    print(df.head(rows).to_string(index=False))


def main() -> None:
    df = load_price_data_for_eda(PRICE_DATA_PATH)

    missing_ratio = build_missing_ratio(df)
    duplicate_ticker_date_rows = build_duplicate_ticker_date_rows(df)
    universe_by_symbol = build_universe_by_symbol(df)
    coverage_by_year = build_coverage_by_year(df)
    liquidity_summary = build_liquidity_summary(df)
    outlier_summary = build_outlier_summary(df)
    feature_preview = build_feature_preview(df)

    print("Historical price EDA checks:")
    print("Rows:", len(df))
    print("Unique symbols:", df["symbol"].nunique(dropna=True))
    print("Unique tickers:", df["ticker"].replace("", pd.NA).nunique(dropna=True))
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Duplicate ticker-date rows:", len(duplicate_ticker_date_rows))

    print_preview("Missing ratio (%)", missing_ratio)
    print_preview("Duplicate ticker-date rows", duplicate_ticker_date_rows)
    print_preview("Universe by symbol", universe_by_symbol)
    print_preview("Coverage by year", coverage_by_year)
    print_preview("Liquidity summary", liquidity_summary)
    print_preview("Outlier summary", outlier_summary)
    print_preview(
        "Feature preview",
        feature_preview[
            [
                "symbol",
                "ticker",
                "date",
                "close_price",
                "ret_0",
                "ret_1",
                "ret_3",
                "ret_5",
                "log_vol_total",
                "intraday_range_ratio",
                "foreign_net_ratio",
            ]
        ],
    )


if __name__ == "__main__":
    main()
