from __future__ import annotations

from pathlib import Path

import pandas as pd


from EDA_ticker_required import (
        HISTORICAL_PRICE_PATH,
        REQUIRED_SYMBOLS_PATH,
        load_ticker_required_data,
)

OHLC_COLUMNS = ["open_price", "high_price", "low_price", "close_price"]


def drop_zero_ohlc_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    zero_ohlc_mask = pd.Series(False, index=out.index)
    for column in OHLC_COLUMNS:
        zero_ohlc_mask |= out[column].notna() & out[column].eq(0)

    zero_ohlc_rows = out.loc[zero_ohlc_mask].copy()
    cleaned_df = out.loc[~zero_ohlc_mask].copy()
    return cleaned_df, zero_ohlc_rows


def normalize_zero_volume_ohlc_to_basic_price(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    complete_ohlc_mask = out[OHLC_COLUMNS].notna().all(axis=1)
    rounded_ohlc = out.loc[complete_ohlc_mask, OHLC_COLUMNS].round(3)
    zero_volume_ohlc_changed_mask = pd.Series(False, index=out.index)
    zero_volume_ohlc_changed_mask.loc[complete_ohlc_mask] = (
        out.loc[complete_ohlc_mask, "vol_total"].fillna(0).eq(0)
        & rounded_ohlc.max(axis=1).gt(rounded_ohlc.min(axis=1))
    )
    fix_mask = zero_volume_ohlc_changed_mask & out["basic_price"].notna()

    for column in OHLC_COLUMNS:
        out.loc[fix_mask, column] = out.loc[fix_mask, "basic_price"]

    normalized_rows = out.loc[fix_mask].copy()
    return out, normalized_rows


def build_clean_eda_data(
    path: str | Path = HISTORICAL_PRICE_PATH,
    required_symbols_path: str | Path = REQUIRED_SYMBOLS_PATH,
) -> dict[str, pd.DataFrame]:
    ticker_required_df = load_ticker_required_data(path, required_symbols_path)
    if "Ticker" in ticker_required_df.columns:
        ticker_required_df = ticker_required_df.rename(columns={"Ticker": "ticker"})
    without_zero_ohlc_df, zero_ohlc_rows = drop_zero_ohlc_rows(ticker_required_df)
    clean_df, normalized_zero_volume_ohlc_rows = normalize_zero_volume_ohlc_to_basic_price(
        without_zero_ohlc_df
    )

    sort_columns = ["ticker", "date"]
    clean_df = clean_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    zero_ohlc_rows = zero_ohlc_rows.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    normalized_zero_volume_ohlc_rows = normalized_zero_volume_ohlc_rows.sort_values(
        sort_columns,
        kind="mergesort",
    ).reset_index(drop=True)

    return {
        "ticker_required_data": ticker_required_df,
        "clean_eda_data": clean_df,
        "dropped_zero_ohlc_rows": zero_ohlc_rows,
        "normalized_zero_volume_ohlc_rows": normalized_zero_volume_ohlc_rows,
    }


def main() -> None:
    result = build_clean_eda_data()
    ticker_required_df = result["ticker_required_data"]
    clean_df = result["clean_eda_data"]
    dropped_zero_ohlc_rows = result["dropped_zero_ohlc_rows"]
    normalized_zero_volume_ohlc_rows = result["normalized_zero_volume_ohlc_rows"]

    print("Ticker-required input rows:", len(ticker_required_df))
    print("Rows dropped because OHLC = 0:", len(dropped_zero_ohlc_rows))
    print("Rows after OHLC cleaning:", len(clean_df))


if __name__ == "__main__":
    main()
