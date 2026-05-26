from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data_Histo" / "vnindex_eda_output.csv"
OUTPUT_PARQUET_PATH = PROJECT_ROOT / "data_Histo" / "vnindex_weekly_return.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data_Histo" / "vnindex_weekly_return.csv"

SYMBOL_COLUMN = "symbol"
DATE_COLUMN = "date"
CLOSE_COLUMN = "close_price"
OPEN_COLUMN = "open_price"
HIGH_COLUMN = "high_price"
LOW_COLUMN = "low_price"
VOLUME_COLUMN = "vol_total"
VALUE_COLUMN = "val_total"


def prepare_vnindex_price(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if SYMBOL_COLUMN in out.columns:
        out[SYMBOL_COLUMN] = out[SYMBOL_COLUMN].astype("string").str.strip().str.upper()
        out = out.loc[out[SYMBOL_COLUMN].eq("VNINDEX")].copy()

    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
    for column in [
        OPEN_COLUMN,
        HIGH_COLUMN,
        LOW_COLUMN,
        CLOSE_COLUMN,
        VOLUME_COLUMN,
        VALUE_COLUMN,
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

    weekly_period = out[DATE_COLUMN].dt.to_period("W-SUN")
    out["week_start"] = weekly_period.apply(lambda period: period.start_time).dt.normalize()
    out["week_end"] = weekly_period.apply(lambda period: period.end_time).dt.normalize()

    return out


def build_vnindex_weekly_return(df: pd.DataFrame) -> pd.DataFrame:
    vnindex_df = prepare_vnindex_price(df)

    weekly_return = vnindex_df.groupby(["week_start", "week_end"], sort=True).agg(
        first_trading_date=(DATE_COLUMN, "min"),
        last_trading_date=(DATE_COLUMN, "max"),
        trading_day_count=(DATE_COLUMN, "size"),
        open_price=(OPEN_COLUMN, "first"),
        high_price=(HIGH_COLUMN, "max"),
        low_price=(LOW_COLUMN, "min"),
        close_price=(CLOSE_COLUMN, "last"),
        vol_total=(VOLUME_COLUMN, "sum"),
        val_total=(VALUE_COLUMN, "sum"),
    )
    weekly_return = weekly_return.reset_index()

    weekly_return["weekly_return"] = np.log(weekly_return["close_price"] / weekly_return["close_price"].shift(1))

    weekly_return["future_ret_1w"] = np.log(weekly_return["close_price"].shift(-1) / weekly_return["close_price"])
    weekly_return["future_ret_4w"] = np.log(weekly_return["close_price"].shift(-4) / weekly_return["close_price"]
                                            )
    weekly_return["return_lag_1w"] = weekly_return["weekly_return"].shift(1)
    weekly_return["volatility_12w"] = weekly_return["weekly_return"].rolling(12).std()
    weekly_return["log_vol_total"] = np.log1p(weekly_return["vol_total"])

    weekly_return = weekly_return[
        [
            "week_start",
            "week_end",
            "first_trading_date",
            "last_trading_date",
            "trading_day_count",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "vol_total",
            "val_total",
            "weekly_return",
            "future_ret_1w",
            "future_ret_4w",
            "return_lag_1w",
            "volatility_12w",
            "log_vol_total",
        ]
    ]
    return weekly_return


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    vnindex_price_df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    weekly_return = build_vnindex_weekly_return(vnindex_price_df)

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    weekly_return.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    weekly_return.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input:", INPUT_PATH)
    print("Output parquet:", OUTPUT_PARQUET_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("Input rows:", len(vnindex_price_df))
    print("Weekly return rows:", len(weekly_return))
    print(weekly_return.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
