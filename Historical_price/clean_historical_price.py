from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import pandas as pd

HISTORICAL_PRICE_PATH = "historical_price_2894.parquet" 

FLOAT_COLUMNS = [
    "high_price",
    "low_price",
    "open_price",
    "average_price",
    "close_price",
    "basic_price",
    "adj_ratio",
    "unit",
    "vol_total",
    "vol_deal",
    "vol_putth",
    "val_total",
    "val_putth",
    "buy_vol_foreign",
    "buy_val_foreigh",
    "sell_vol_foreign",
    "sel_val_foreign",
    "buy_count",
    "buy_vol",
    "sell_count",
    "sell_vol",
    "foreign_room",
    "prop_trading_deal",
    "prop_trading_putth",
    "prop_trading_net",
]

CW_SYMBOL_PATTERN = re.compile(r"^C(?P<ticker>[A-Z]{2,})(?P<date_code>\d{2,})$")
ALPHA_DIGIT_SYMBOL_PATTERN = re.compile(r"^(?P<ticker>[A-Z]{2,})(?P<date_code>\d{2,})$")
THREE_CHAR_TICKER_PATTERN = re.compile(r"^(?P<ticker>[A-Z0-9]{3})$")
PLAIN_TICKER_PATTERN = re.compile(r"^(?P<ticker>[A-Z]{2,10})$")


def load_tabular_data(path: str | Path) -> pd.DataFrame:
    source_path = Path(path)
    if not str(source_path).strip():
        raise ValueError("Set HISTORICAL_PRICE_PATH before running clean_historical_price.py.")
    if not source_path.exists():
        raise FileNotFoundError(f"Historical price file not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(source_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(source_path)
    return df


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = out["symbol"].fillna("").astype("string").str.strip().str.upper()
    out["date"] = out["date"].fillna("").astype("string").str.strip()
    for column in FLOAT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Float64")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def parse_symbol_components(symbol: object) -> Dict[str, object]:
    raw_symbol = str(symbol).strip().upper()
    if raw_symbol == "" or raw_symbol.lower() == "nan":
        return {
            "ticker": pd.NA,
            "instrument_type": "unknown",
            "symbol_prefix": pd.NA,
            "symbol_date_code": pd.NA,
            "symbol_parse_rule": "empty_symbol",
        }

    match = CW_SYMBOL_PATTERN.match(raw_symbol)
    if match:
        return {
            "ticker": match.group("ticker"),
            "instrument_type": "c_prefixed_symbol",
            "symbol_prefix": "C",
            "symbol_date_code": match.group("date_code"),
            "symbol_parse_rule": "C + ticker + date_code",
        }

    match = THREE_CHAR_TICKER_PATTERN.match(raw_symbol)
    if match:
        return {
            "ticker": match.group("ticker"),
            "instrument_type": "three_char_ticker",
            "symbol_prefix": pd.NA,
            "symbol_date_code": pd.NA,
            "symbol_parse_rule": "three_char_alphanumeric_ticker",
        }

    match = ALPHA_DIGIT_SYMBOL_PATTERN.match(raw_symbol)
    if match:
        return {
            "ticker": match.group("ticker"),
            "instrument_type": "alpha_digit_symbol",
            "symbol_prefix": pd.NA,
            "symbol_date_code": match.group("date_code"),
            "symbol_parse_rule": "ticker + date_code",
        }

    match = PLAIN_TICKER_PATTERN.match(raw_symbol)
    if match:
        return {
            "ticker": match.group("ticker"),
            "instrument_type": "plain_ticker",
            "symbol_prefix": pd.NA,
            "symbol_date_code": pd.NA,
            "symbol_parse_rule": "plain_ticker",
        }

    alpha_chunks = re.findall(r"[A-Z]+", raw_symbol)
    if alpha_chunks:
        return {
            "ticker": alpha_chunks[-1],
            "instrument_type": "fallback_alpha_extract",
            "symbol_prefix": raw_symbol[:1],
            "symbol_date_code": pd.NA,
            "symbol_parse_rule": "fallback_last_alpha_chunk",
        }

    return {
        "ticker": pd.NA,
        "instrument_type": "unknown",
        "symbol_prefix": pd.NA,
        "symbol_date_code": pd.NA,
        "symbol_parse_rule": "unparsed",
    }


def add_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    parsed_symbol = df["symbol"].apply(parse_symbol_components).apply(pd.Series)
    out = pd.concat([df.copy(), parsed_symbol], axis=1)
    out["ticker"] = out["ticker"].astype("string").str.strip().str.upper()
    return out


def reconcile_year_from_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed_date = pd.to_datetime(out["date"], errors="coerce")
    out["year_from_date"] = parsed_date.dt.year.astype("Int64")
    out["year_raw"] = out["year"]
    out["year_mismatch_flag"] = (
        out["year"].notna()
        & out["year_from_date"].notna()
        & out["year"].ne(out["year_from_date"])
    )
    out["year"] = out["year_from_date"].where(out["year_from_date"].notna(), out["year"])
    return out


def mark_duplicate_symbol_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    duplicate_rows = out[out.duplicated(subset=["symbol", "date"], keep="first")].copy()
    deduplicated = out.drop_duplicates(subset=["symbol", "date"], keep="first").copy()
    return deduplicated, duplicate_rows


def remove_negative_ohlc_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    negative_price_mask = (
        (out["open_price"].notna() & out["open_price"].lt(0))
        | (out["high_price"].notna() & out["high_price"].lt(0))
        | (out["low_price"].notna() & out["low_price"].lt(0))
        | (out["close_price"].notna() & out["close_price"].lt(0))
    )
    negative_rows = out.loc[negative_price_mask].copy()
    cleaned_df = out.loc[~negative_price_mask].copy()
    return cleaned_df, negative_rows


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    ohlc_columns = ["open_price", "high_price", "low_price", "close_price"]
    foreign_flow_columns = [ "buy_vol_foreign", "buy_val_foreigh", "sell_vol_foreign", "sel_val_foreign",]
    prop_columns = ["prop_trading_deal", "prop_trading_putth", "prop_trading_net"]

    out["missing_symbol_flag"] = out["symbol"].eq("")
    out["missing_ticker_flag"] = out["ticker"].isna() | out["ticker"].eq("")
    out["missing_date_flag"] = out["date"].isna()
    out["missing_core_price_flag"] = out[ohlc_columns].isna().any(axis=1)

    out["invalid_ohlc_flag"] = False
    complete_ohlc = out[ohlc_columns].notna().all(axis=1)
    rounded_ohlc = out.loc[complete_ohlc, ohlc_columns].round(3)
    out.loc[complete_ohlc, "invalid_ohlc_flag"] = (
        rounded_ohlc["high_price"].lt(rounded_ohlc["low_price"])
        | rounded_ohlc["high_price"].lt(
            rounded_ohlc[["open_price", "close_price"]].max(axis=1)
        )
        | rounded_ohlc["low_price"].gt(
            rounded_ohlc[["open_price", "close_price"]].min(axis=1)
        )
    )

    out["zero_volume_flag"] = (
        out["vol_total"].fillna(0).eq(0) | out["val_total"].fillna(0).eq(0)
    )
    out["adj_ratio_not_one_flag"] = out["adj_ratio"].notna() & out["adj_ratio"].ne(1)
    expected_vol_total = out["vol_deal"].fillna(0) + out["vol_putth"].fillna(0)
    out["vol_total_components_mismatch_flag"] = (
        out["vol_total"].notna()
        & out["vol_total"].round(3).ne(expected_vol_total.round(3))
    )
    out["zero_volume_but_ohlc_changed_flag"] = False
    out.loc[complete_ohlc, "zero_volume_but_ohlc_changed_flag"] = (
        out.loc[complete_ohlc, "vol_total"].fillna(0).eq(0)
        & rounded_ohlc.max(axis=1).gt(rounded_ohlc.min(axis=1))
    )
    out["foreign_flow_all_missing_flag"] = out[foreign_flow_columns].isna().all(axis=1)
    out["prop_trading_all_missing_flag"] = out[prop_columns].isna().all(axis=1)
    out["review_reason"] = ""

    out.loc[out["missing_symbol_flag"], "review_reason"] += "missing_symbol;"
    out.loc[out["missing_ticker_flag"], "review_reason"] += "missing_ticker;"
    out.loc[out["missing_date_flag"], "review_reason"] += "missing_date;"
    out.loc[out["missing_core_price_flag"], "review_reason"] += "missing_core_ohlc;"
    out.loc[out["invalid_ohlc_flag"], "review_reason"] += "invalid_ohlc;"
    out["review_reason"] = out["review_reason"].str.rstrip(";")

    out["severe_issue_flag"] = (
        out["missing_symbol_flag"]
        | out["missing_ticker_flag"]
        | out["missing_date_flag"]
        | out["missing_core_price_flag"]
        | out["invalid_ohlc_flag"]
    )
    return out


def prepare_historical_price_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = standardize_types(out)
    out = add_symbol_features(out)
    out = reconcile_year_from_date(out)
    return out


def find_market_index_rows(df: pd.DataFrame) -> pd.DataFrame:
    known_market_symbols = {
        "VNINDEX",
        "HNXINDEX",
        "UPCOMINDEX",
        "VN30",
        "HNX30",
        "VN100",
        "VNALLSHARE",
    }
    symbol_text = df["symbol"].fillna("").astype("string").str.upper()
    ticker_text = df["ticker"].fillna("").astype("string").str.upper()
    market_index_mask = (
        symbol_text.isin(known_market_symbols)
        | ticker_text.isin(known_market_symbols)
        | symbol_text.str.contains("INDEX", na=False)
        | ticker_text.str.contains("INDEX", na=False)
    )
    return df.loc[market_index_mask].copy()


def clean_historical_price_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    prepared_df = prepare_historical_price_data(df)
    deduplicated_df, duplicate_rows = mark_duplicate_symbol_date(prepared_df)
    deduplicated_df, negative_price_rows = remove_negative_ohlc_rows(deduplicated_df)
    deduplicated_df = add_quality_flags(deduplicated_df)

    clean_df = deduplicated_df.loc[~deduplicated_df["severe_issue_flag"]].copy()
    review_df = deduplicated_df.loc[deduplicated_df["severe_issue_flag"]].copy()

    sort_columns = ["ticker", "symbol", "date"]
    clean_df = clean_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    review_df = review_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    duplicate_rows = duplicate_rows.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    negative_price_rows = negative_price_rows.sort_values(
        sort_columns,
        kind="mergesort",
    ).reset_index(drop=True)

    return {
        "prepared_data": prepared_df,
        "deduplicated_data": deduplicated_df,
        "clean_data": clean_df,
        "review_rows": review_df,
        "duplicate_rows": duplicate_rows,
        "negative_price_rows": negative_price_rows,
    }


def main() -> None:
    raw_df = load_tabular_data(HISTORICAL_PRICE_PATH)
    result = clean_historical_price_dataset(raw_df)
    prepared_df = result["prepared_data"]
    deduplicated_df = result["deduplicated_data"]
    negative_price_rows = result["negative_price_rows"]
    market_index_rows = find_market_index_rows(deduplicated_df)
    market_index_symbols = (
        market_index_rows[["symbol", "ticker"]]
        .drop_duplicates()
        .sort_values(["symbol", "ticker"], kind="mergesort")
        .reset_index(drop=True)
    )
    zero_volume_but_ohlc_changed_rows = deduplicated_df.loc[
        deduplicated_df["zero_volume_but_ohlc_changed_flag"]
    ].copy()
    parsed_dates = pd.to_datetime(deduplicated_df["date"], errors="coerce")

    print("Input rows:", len(raw_df))
    print("Duplicate symbol-date rows:", len(result["duplicate_rows"]))
    print("Rows with negative OHLC price (< 0):", len(negative_price_rows))
    print("Rows after symbol-date dedup:", len(deduplicated_df))
    print("Clean rows:", len(result["clean_data"]))
    print("Review rows:", len(result["review_rows"]))
    print("Rows with market-index-like symbols/tickers:", len(market_index_rows))
    print("Unique market-index-like symbols:", len(market_index_symbols))
    print("Unique symbols:", deduplicated_df["symbol"].nunique(dropna=True))
    print("Unique tickers:", deduplicated_df["ticker"].nunique(dropna=True))
    print("Min date:", parsed_dates.min())
    print("Max date:", parsed_dates.max())
    print("Rows with year mismatch:", int(deduplicated_df["year_mismatch_flag"].sum()))
    print("Rows with missing symbol:", int(deduplicated_df["missing_symbol_flag"].sum()))
    print("Rows with missing ticker:", int(deduplicated_df["missing_ticker_flag"].sum()))
    print("Rows with missing date:", int(deduplicated_df["missing_date_flag"].sum()))
    print("Rows with missing core OHLC price:", int(deduplicated_df["missing_core_price_flag"].sum()))
    print("Rows with invalid OHLC:", int(deduplicated_df["invalid_ohlc_flag"].sum()))
    print("Rows with zero volume/value:", int(deduplicated_df["zero_volume_flag"].sum()))
    print("Rows with adj_ratio != 1:", int(deduplicated_df["adj_ratio_not_one_flag"].sum()))
    print(
        "Rows with vol_total != vol_deal + vol_putth:",
        int(deduplicated_df["vol_total_components_mismatch_flag"].sum()),
    )
    print(
        "Rows with vol_total = 0 but OHLC changed:",
        int(deduplicated_df["zero_volume_but_ohlc_changed_flag"].sum()),
    )
    print(
        "Rows with all foreign flow fields missing:",
        int(deduplicated_df["foreign_flow_all_missing_flag"].sum()),
    )
    print(
        "Rows with all proprietary trading fields missing:",
        int(deduplicated_df["prop_trading_all_missing_flag"].sum()),
    )
    if not negative_price_rows.empty:
        print("\nRows with negative OHLC price that were removed:")
        print(
            negative_price_rows[
                ["symbol", "ticker", "date", "open_price", "high_price", "low_price", "close_price"]
            ].to_string(index=False)
        )
    if not zero_volume_but_ohlc_changed_rows.empty:
        print("\nRows with vol_total = 0 but OHLC changed:")
        print(
            zero_volume_but_ohlc_changed_rows[
                [
                    "symbol",
                    "ticker",
                    "date",
                    "vol_total",
                    "val_total",
                    "open_price",
                    "high_price",
                    "low_price",
                    "close_price",
                ]
            ].to_string(index=False)
        )
    if not market_index_symbols.empty:
        print("\nMarket-index-like symbols/tickers:")
        print(market_index_symbols.to_string(index=False))

if __name__ == "__main__":
    main()
