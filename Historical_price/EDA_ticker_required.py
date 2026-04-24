from __future__ import annotations
from pathlib import Path
import pandas as pd

try:
    from Historical_price.EDA_raw_historical_price import (
        HISTORICAL_PRICE_PATH as RAW_HISTORICAL_PRICE_PATH,
        REQUIRED_SYMBOLS_PATH as RAW_REQUIRED_SYMBOLS_PATH,
        collect_data_quality_issue_rows,
        filter_required_symbols,
        load_tabular_data,
        mark_duplicate_symbol_date,
        prepare_required_symbols,
        reconcile_year_from_date,
        remove_invalid_ohlc_rows,
        standardize_types,
    )
except ModuleNotFoundError:
    from EDA_raw_historical_price import (
        HISTORICAL_PRICE_PATH as RAW_HISTORICAL_PRICE_PATH,
        REQUIRED_SYMBOLS_PATH as RAW_REQUIRED_SYMBOLS_PATH,
        collect_data_quality_issue_rows,
        filter_required_symbols,
        load_tabular_data,
        mark_duplicate_symbol_date,
        prepare_required_symbols,
        reconcile_year_from_date,
        remove_invalid_ohlc_rows,
        standardize_types,
    )

HISTORICAL_PRICE_PATH = RAW_HISTORICAL_PRICE_PATH
REQUIRED_SYMBOLS_PATH = RAW_REQUIRED_SYMBOLS_PATH

def rename_symbol_column_to_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "symbol" in out.columns:
        out = out.rename(columns={"symbol": "Ticker"})
    return out


def build_required_ticker_eda(df: pd.DataFrame,required_symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    prepared_df = standardize_types(df)
    if required_symbols is None:
        required_symbols = prepare_required_symbols(REQUIRED_SYMBOLS_PATH)

    _, required_rows, _ = filter_required_symbols(prepared_df, required_symbols)
    required_rows, year_mismatch_rows = reconcile_year_from_date(required_rows)
    required_rows, duplicate_rows = mark_duplicate_symbol_date(required_rows)
    required_rows, invalid_ohlc_rows = remove_invalid_ohlc_rows(required_rows)
    quality_issue_data = collect_data_quality_issue_rows(required_rows, required_symbols)

    tickers = (
        required_rows["symbol"]
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
        .to_frame(name="Ticker")
    )

    sort_columns = ["symbol", "date"]
    required_rows = required_rows.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    duplicate_rows = duplicate_rows.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    invalid_ohlc_rows = invalid_ohlc_rows.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    year_mismatch_rows = year_mismatch_rows.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    output_frames: dict[str, pd.DataFrame] = {
        "ticker_required_data": rename_symbol_column_to_ticker(required_rows),
        "tickers": tickers,
        "duplicate_ticker_date_rows": rename_symbol_column_to_ticker(duplicate_rows),
        "invalid_ohlc_rows": rename_symbol_column_to_ticker(invalid_ohlc_rows),
        "year_mismatch_rows": rename_symbol_column_to_ticker(year_mismatch_rows),
    }

    for key, issue_df in quality_issue_data.items():
        sorted_issue_df = issue_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
        output_frames[key] = rename_symbol_column_to_ticker(sorted_issue_df)

    return output_frames


def load_ticker_required_data(
    path: str | Path = HISTORICAL_PRICE_PATH,
    required_symbols_path: str | Path = REQUIRED_SYMBOLS_PATH,
) -> pd.DataFrame:
    required_symbols = prepare_required_symbols(required_symbols_path)
    raw_df = load_tabular_data(path)
    return build_required_ticker_eda(raw_df, required_symbols)["ticker_required_data"].copy()


def main() -> None:
    required_symbols = prepare_required_symbols(REQUIRED_SYMBOLS_PATH)
    raw_df = load_tabular_data(HISTORICAL_PRICE_PATH)
    result = build_required_ticker_eda(raw_df, required_symbols)

    ticker_required_df = result["ticker_required_data"]
    tickers = result["tickers"]
    duplicate_ticker_date_rows = result["duplicate_ticker_date_rows"]
    invalid_ohlc_rows = result["invalid_ohlc_rows"]
    year_mismatch_rows = result["year_mismatch_rows"]
    zero_vol_total_rows = result["zero_vol_total_rows"]
    adj_ratio_not_one_rows = result["adj_ratio_not_one_rows"]
    vol_total_components_mismatch_rows = result["vol_total_components_mismatch_rows"]
    zero_vol_total_but_ohlc_changed_rows = result["zero_vol_total_but_ohlc_changed_rows"]
    foreign_flow_all_missing_rows = result["foreign_flow_all_missing_rows"]
    prop_trading_all_missing_rows = result["prop_trading_all_missing_rows"]
    parsed_dates = pd.to_datetime(ticker_required_df["date"], errors="coerce")

    print("Input rows:", len(ticker_required_df))
    print("Symbols required from symbols.csv:", len(required_symbols))
    print("Tickers:", len(tickers))
    print("Duplicate ticker-date rows:", len(duplicate_ticker_date_rows))
    print("Rows with invalid OHLC:", len(invalid_ohlc_rows))
    print("Min date:", parsed_dates.min())
    print("Max date:", parsed_dates.max())
    print("Rows with year mismatch:", len(year_mismatch_rows))
    print("Rows with vol_total = 0:", len(zero_vol_total_rows))
    print("Rows with adj_ratio != 1:", len(adj_ratio_not_one_rows))
    print("Rows with vol_total != vol_deal + vol_putth:", len(vol_total_components_mismatch_rows))
    print("Rows with vol_total = 0 but OHLC changed:", len(zero_vol_total_but_ohlc_changed_rows))
    print("Rows with all foreign flow fields missing:", len(foreign_flow_all_missing_rows))
    print("Rows with all proprietary trading fields missing:", len(prop_trading_all_missing_rows))

if __name__ == "__main__":
    main()
