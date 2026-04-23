from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_PRICE_PATH = PROJECT_ROOT / "historical_price_all1.parquet"
REQUIRED_SYMBOLS_PATH = PROJECT_ROOT / "symbols.csv"

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


def load_required_symbols(path: str | Path) -> pd.Index:
    source_path = Path(path)
    if not str(source_path).strip():
        raise ValueError("Set REQUIRED_SYMBOLS_PATH before running clean_historical_price.py.")
    if not source_path.exists():
        raise FileNotFoundError(f"Symbols file not found: {source_path}")

    symbols_df = pd.read_csv(
        source_path,
        header=None,
        usecols=[0],
        names=["symbol"],
        dtype="string",
    )
    required_symbols = (
        symbols_df["symbol"]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.upper()
    )
    required_symbols = required_symbols[
        required_symbols.ne("") & ~required_symbols.isin({"SYMBOL", "TICKER"})
    ]
    required_symbols = required_symbols.drop_duplicates().tolist()
    return pd.Index(required_symbols, dtype="string", name="ticker")


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = out["symbol"].fillna("").astype("string").str.strip().str.upper()
    out["date"] = out["date"].fillna("").astype("string").str.strip()
    for column in FLOAT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Float64")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def normalize_required_tickers(required_tickers: pd.Index) -> list[str]:
    normalized_tickers = (
        pd.Series(required_tickers, dtype="string")
        .fillna("")
        .astype("string")
        .str.strip()
        .str.upper()
    )
    normalized_tickers = normalized_tickers[normalized_tickers.ne("")].drop_duplicates().tolist()
    if not normalized_tickers:
        raise ValueError("symbols.csv does not contain any valid ticker.")
    return sorted(normalized_tickers, key=lambda ticker: (-len(ticker), ticker))


def build_required_ticker_pattern(required_tickers: pd.Index) -> re.Pattern[str]:
    normalized_tickers = normalize_required_tickers(required_tickers)
    escaped_tickers = [re.escape(ticker) for ticker in normalized_tickers]
    return re.compile(f"({'|'.join(escaped_tickers)})")


def build_symbol_match_lookup(
    symbols: pd.Series,
    required_tickers: pd.Index,
) -> pd.DataFrame:
    normalized_tickers = normalize_required_tickers(required_tickers)
    unique_symbols = (
        symbols.fillna("")
        .astype("string")
        .str.strip()
        .str.upper()
        .drop_duplicates()
        .tolist()
    )

    lookup_rows = []
    for symbol in unique_symbols:
        matches = [ticker for ticker in normalized_tickers if ticker in symbol]
        primary_ticker = matches[0] if matches else pd.NA
        lookup_rows.append(
            {
                "symbol": symbol,
                "ticker": primary_ticker,
                "ticker_extracted": primary_ticker,
                "matched_ticker_substrings": ",".join(matches) if matches else pd.NA,
                "ticker_match_count": len(matches),
            }
        )

    match_lookup = pd.DataFrame(lookup_rows)
    match_lookup["ticker"] = match_lookup["ticker"].astype("string").str.strip().str.upper()
    match_lookup["ticker_extracted"] = (
        match_lookup["ticker_extracted"].astype("string").str.strip().str.upper()
    )
    match_lookup["matched_ticker_substrings"] = (
        match_lookup["matched_ticker_substrings"].astype("string").str.strip().str.upper()
    )
    match_lookup["ticker_match_count"] = (
        pd.to_numeric(match_lookup["ticker_match_count"], errors="coerce")
        .fillna(0)
        .astype("Int64")
    )
    match_lookup["ticker_multi_match_flag"] = match_lookup["ticker_match_count"].ge(2)
    return match_lookup


def add_symbol_features(
    df: pd.DataFrame,
    required_tickers: pd.Index,
) -> pd.DataFrame:
    out = df.copy()
    symbol_match_lookup = build_symbol_match_lookup(out["symbol"], required_tickers)
    out = out.merge(
        symbol_match_lookup,
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    out["ticker"] = out["ticker"].astype("string").str.strip().str.upper()
    out["ticker_extracted"] = out["ticker_extracted"].astype("string").str.strip().str.upper()
    out["matched_ticker_substrings"] = (
        out["matched_ticker_substrings"].astype("string").str.strip().str.upper()
    )
    out["ticker_match_count"] = (
        pd.to_numeric(out["ticker_match_count"], errors="coerce")
        .fillna(0)
        .astype("Int64")
    )
    out["ticker_multi_match_flag"] = out["ticker_match_count"].ge(2)
    return out


def filter_required_tickers(
    df: pd.DataFrame,
    required_tickers: pd.Index,
) -> tuple[pd.DataFrame, pd.Index]:
    out = df.copy()
    required_ticker_set = set(required_tickers.astype(str).tolist())
    out["ticker_in_symbols_flag"] = out["ticker"].isin(required_ticker_set)
    filtered_df = out.loc[out["ticker_in_symbols_flag"]].copy()
    matched_tickers = (
        filtered_df["ticker"]
        .dropna()
        .astype("string")
        .str.strip()
        .str.upper()
    )
    matched_tickers = matched_tickers[matched_tickers.ne("")]
    matched_tickers = matched_tickers.drop_duplicates().sort_values().tolist()
    return filtered_df, pd.Index(matched_tickers, dtype="string", name="ticker")


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


def prepare_historical_price_data(
    df: pd.DataFrame,
    required_tickers: pd.Index | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out = standardize_types(out)
    if required_tickers is None:
        required_tickers = load_required_symbols(REQUIRED_SYMBOLS_PATH)
    out = add_symbol_features(out, required_tickers)
    out = reconcile_year_from_date(out)
    return out


def clean_historical_price_dataset(
    df: pd.DataFrame,
    required_tickers: pd.Index,
) -> Dict[str, pd.DataFrame]:
    prepared_df = prepare_historical_price_data(df, required_tickers=required_tickers)
    filtered_prepared_df, matched_tickers = filter_required_tickers(
        prepared_df,
        required_tickers,
    )
    deduplicated_df, duplicate_rows = mark_duplicate_symbol_date(filtered_prepared_df)
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
        "filtered_prepared_data": filtered_prepared_df,
        "deduplicated_data": deduplicated_df,
        "clean_data": clean_df,
        "review_rows": review_df,
        "duplicate_rows": duplicate_rows,
        "negative_price_rows": negative_price_rows,
        "matched_tickers": pd.DataFrame({"ticker": matched_tickers}),
    }


def count_ticker_records(df: pd.DataFrame) -> int:
    ticker_present = df["ticker"].notna() & df["ticker"].ne("")
    return int(ticker_present.sum())


def count_unique_tickers(df: pd.DataFrame) -> int:
    return int(df["ticker"].replace("", pd.NA).nunique(dropna=True))


def count_unmatched_symbol_records(df: pd.DataFrame) -> int:
    ticker_missing = df["ticker"].isna() | df["ticker"].eq("")
    return int(ticker_missing.sum())


def main() -> None:
    required_tickers = load_required_symbols(REQUIRED_SYMBOLS_PATH)
    raw_df = load_tabular_data(HISTORICAL_PRICE_PATH)
    result = clean_historical_price_dataset(raw_df, required_tickers)
    prepared_df = result["prepared_data"]
    filtered_prepared_df = result["filtered_prepared_data"]
    deduplicated_df = result["deduplicated_data"]
    negative_price_rows = result["negative_price_rows"]
    matched_tickers = result["matched_tickers"]
    zero_volume_but_ohlc_changed_rows = deduplicated_df.loc[
        deduplicated_df["zero_volume_but_ohlc_changed_flag"]
    ].copy()
    parsed_dates = pd.to_datetime(deduplicated_df["date"], errors="coerce")
    unique_symbols_in_dataset = prepared_df["symbol"].replace("", pd.NA).nunique(dropna=True)
    ticker_record_count = count_ticker_records(prepared_df)
    records_with_multi_ticker_match = int(prepared_df["ticker_multi_match_flag"].sum())

    print("Input rows:", len(raw_df))
    print("Symbols required from symbols.csv:", len(required_tickers))
    print("Unique symbols in dataset:", unique_symbols_in_dataset)
    print("Matched tickers:", len(matched_tickers))
    print("Ticker records in column ticker:", ticker_record_count)
    print(
        "Rows with symbols matching >= 2 ticker substrings:",
        records_with_multi_ticker_match,
    )
    print("Rows after required ticker filter:", len(filtered_prepared_df))
    print("Duplicate symbol-date rows:", len(result["duplicate_rows"]))
    print("Rows with negative OHLC price (< 0):", len(negative_price_rows))
    print("Rows after symbol-date dedup:", len(deduplicated_df))
    print("Clean rows:", len(result["clean_data"]))
    print("Review rows:", len(result["review_rows"]))
    print("Unique symbols after filter:", deduplicated_df["symbol"].replace("", pd.NA).nunique(dropna=True))
    print("Unique tickers after filter:", deduplicated_df["ticker"].replace("", pd.NA).nunique(dropna=True))
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
    # print(
    #     "Rows with vol_total != vol_deal + vol_putth:",
    #     int(deduplicated_df["vol_total_components_mismatch_flag"].sum()),
    # )
    # print(
    #     "Rows with vol_total = 0 but OHLC changed:",
    #     int(deduplicated_df["zero_volume_but_ohlc_changed_flag"].sum()),
    # )
    # print(
    #     "Rows with all foreign flow fields missing:",
    #     int(deduplicated_df["foreign_flow_all_missing_flag"].sum()),
    # )
    # print(
    #     "Rows with all proprietary trading fields missing:",
    #     int(deduplicated_df["prop_trading_all_missing_flag"].sum()),
    # )
    # if not negative_price_rows.empty:
    #     print("\nRows with negative OHLC price that were removed:")
    #     print(
    #         negative_price_rows[
    #             ["symbol", "ticker", "date", "open_price", "high_price", "low_price", "close_price"]
    #         ].to_string(index=False)
    #     )
    # if not zero_volume_but_ohlc_changed_rows.empty:
    #     print("\nRows with vol_total = 0 but OHLC changed:")

if __name__ == "__main__":
    main()
