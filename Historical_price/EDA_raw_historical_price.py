from __future__ import annotations

from pathlib import Path

import pandas as pd

HISTORICAL_PRICE_PATH = "price_all.parquet"
REQUIRED_SYMBOLS_PATH = "symbols.csv"

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
    
    df = pd.read_parquet(source_path)
    return df


def prepare_required_symbols(path: str | Path) -> list[str]:
    source_path = Path(path)
    if not str(source_path).strip():
        raise ValueError("Set REQUIRED_SYMBOLS_PATH before running clean_historical_price.py.")
    if not source_path.exists():
        raise FileNotFoundError(f"Symbols file not found: {source_path}")

    symbols_df = pd.read_csv(
        source_path,
        header=None,
        usecols=[0],
        names=["ticker_required"],
        dtype="string",
    )
    normalized_symbols = (
        symbols_df["ticker_required"]
        .fillna("")
        .astype("string")
        .str.strip()
        .str.upper()
    )
    normalized_symbols = (
        normalized_symbols[normalized_symbols.ne("")]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not normalized_symbols:
        raise ValueError("symbols.csv does not contain any valid symbol.")
    return normalized_symbols


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = out["symbol"].fillna("").astype("string").str.strip().str.upper()
    out["date"] = out["date"].fillna("").astype("string").str.strip()
    for column in FLOAT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Float64")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def filter_required_symbols(df: pd.DataFrame, required_symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    required_symbol_set = set(required_symbols)
    symbol_in_required_mask = out["symbol"].isin(required_symbol_set)
    exact_match_rows = out.loc[symbol_in_required_mask].copy()
    unmatched_rows = out.loc[~symbol_in_required_mask].copy()
    return out, exact_match_rows, unmatched_rows


def reconcile_year_from_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    parsed_date = pd.to_datetime(out["date"], errors="coerce")
    year_from_date = parsed_date.dt.year.astype("Int64")
    year_mismatch_mask = (
        out["year"].notna()
        & year_from_date.notna()
        & out["year"].ne(year_from_date)
    )
    year_mismatch_rows = out.loc[year_mismatch_mask].copy()
    return out, year_mismatch_rows


def mark_duplicate_symbol_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    duplicate_rows = out[out.duplicated(subset=["symbol", "date"], keep="first")].copy()
    return out, duplicate_rows


def remove_invalid_ohlc_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    ohlc_columns = ["open_price", "high_price", "low_price", "close_price"]
    complete_ohlc_mask = out[ohlc_columns].notna().all(axis=1)
    missing_ohlc_mask = ~complete_ohlc_mask
    nonpositive_ohlc_mask = (
        (out["open_price"].notna() & out["open_price"].le(0))
        | (out["high_price"].notna() & out["high_price"].le(0))
        | (out["low_price"].notna() & out["low_price"].le(0))
        | (out["close_price"].notna() & out["close_price"].le(0))
    )
    invalid_logic_mask = pd.Series(False, index=out.index)
    rounded_ohlc = out.loc[complete_ohlc_mask, ohlc_columns].round(3)
    invalid_logic_mask.loc[complete_ohlc_mask] = (
        rounded_ohlc["high_price"].lt(rounded_ohlc["low_price"])
        | rounded_ohlc["high_price"].lt(
            rounded_ohlc[["open_price", "close_price"]].max(axis=1)
        )
        | rounded_ohlc["low_price"].gt(
            rounded_ohlc[["open_price", "close_price"]].min(axis=1)
        )
    )
    invalid_ohlc_mask = missing_ohlc_mask | nonpositive_ohlc_mask | invalid_logic_mask
    invalid_ohlc_rows = out.loc[invalid_ohlc_mask].copy()
    invalid_ohlc_rows["ohlc_issue_reason"] = ""
    invalid_ohlc_rows.loc[missing_ohlc_mask, "ohlc_issue_reason"] += "missing_ohlc;"
    invalid_ohlc_rows.loc[nonpositive_ohlc_mask, "ohlc_issue_reason"] += "nonpositive_ohlc;"
    invalid_ohlc_rows.loc[invalid_logic_mask, "ohlc_issue_reason"] += "invalid_ohlc_logic;"
    invalid_ohlc_rows["ohlc_issue_reason"] = invalid_ohlc_rows["ohlc_issue_reason"].str.rstrip(";")
    return out, invalid_ohlc_rows


def collect_data_quality_issue_rows(df: pd.DataFrame, required_symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = df.copy()
    required_symbol_set = set(required_symbols)
    missing_symbol_mask = out["symbol"].eq("")
    missing_date_mask = out["date"].isna() | out["date"].eq("")
    symbol_not_in_required_mask = (
        out["symbol"].ne("")
        & ~out["symbol"].isin(required_symbol_set)
    )
    required_issue_mask = (
        missing_symbol_mask | missing_date_mask | symbol_not_in_required_mask
    )

    zero_vol_total_mask = out["vol_total"].notna() & out["vol_total"].eq(0)
    adj_ratio_not_one_mask = out["adj_ratio"].notna() & out["adj_ratio"].ne(1)
    expected_vol_total = out["vol_deal"].fillna(0) + out["vol_putth"].fillna(0)
    vol_total_components_mismatch_mask = (
        out["vol_total"].notna()
        & out["vol_total"].round(3).ne(expected_vol_total.round(3))
    )

    ohlc_columns = ["open_price", "high_price", "low_price", "close_price"]
    complete_ohlc_mask = out[ohlc_columns].notna().all(axis=1)
    rounded_ohlc = out.loc[complete_ohlc_mask, ohlc_columns].round(3)
    zero_vol_total_but_ohlc_changed_mask = pd.Series(False, index=out.index)
    zero_vol_total_but_ohlc_changed_mask.loc[complete_ohlc_mask] = (
        out.loc[complete_ohlc_mask, "vol_total"].fillna(0).eq(0)
        & rounded_ohlc.max(axis=1).gt(rounded_ohlc.min(axis=1))
    )

    foreign_flow_columns = [
        "buy_vol_foreign",
        "buy_val_foreigh",
        "sell_vol_foreign",
        "sel_val_foreign",
    ]
    prop_columns = ["prop_trading_deal", "prop_trading_putth", "prop_trading_net"]
    foreign_flow_all_missing_mask = out[foreign_flow_columns].isna().all(axis=1)
    prop_trading_all_missing_mask = out[prop_columns].isna().all(axis=1)

    clean_df = out.loc[~required_issue_mask].copy()
    return {
        "clean_data": clean_df,
        "missing_symbol_rows": out.loc[missing_symbol_mask].copy(),
        "missing_date_rows": out.loc[missing_date_mask].copy(),
        "symbol_not_in_required_rows": out.loc[symbol_not_in_required_mask].copy(),
        "zero_vol_total_rows": out.loc[zero_vol_total_mask].copy(),
        "adj_ratio_not_one_rows": out.loc[adj_ratio_not_one_mask].copy(),
        "vol_total_components_mismatch_rows": out.loc[vol_total_components_mismatch_mask].copy(),
        "zero_vol_total_but_ohlc_changed_rows": out.loc[zero_vol_total_but_ohlc_changed_mask].copy(),
        "foreign_flow_all_missing_rows": out.loc[foreign_flow_all_missing_mask].copy(),
        "prop_trading_all_missing_rows": out.loc[prop_trading_all_missing_mask].copy(),
    }


def clean_historical_price_dataset(df: pd.DataFrame, required_symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    prepared_df = df.copy()
    prepared_df = standardize_types(prepared_df)
    if required_symbols is None:
        required_symbols = prepare_required_symbols(REQUIRED_SYMBOLS_PATH)
    prepared_df, year_mismatch_rows = reconcile_year_from_date(prepared_df)
    prepared_df, exact_match_rows, unmatched_rows = filter_required_symbols(prepared_df, required_symbols)
    unmatched_symbols = (
        unmatched_rows["symbol"]
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    unmatched_symbols = pd.DataFrame({"symbol": unmatched_symbols})
    deduplicated_df, duplicate_rows = mark_duplicate_symbol_date(prepared_df)
    deduplicated_df, invalid_ohlc_rows = remove_invalid_ohlc_rows(deduplicated_df)
    quality_issue_data = collect_data_quality_issue_rows(deduplicated_df, required_symbols)
    clean_df = quality_issue_data["clean_data"]

    sort_columns = ["symbol", "date"]
    clean_df = clean_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    duplicate_rows = duplicate_rows.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    invalid_ohlc_rows = invalid_ohlc_rows.sort_values(sort_columns, kind="mergesort",).reset_index(drop=True)
    year_mismatch_rows = year_mismatch_rows.sort_values(["symbol", "date"], kind="mergesort",).reset_index(drop=True)

    inspection_frames = {
        "filtered_prepared_data": exact_match_rows,
        "unmatched_symbol_rows": unmatched_rows,
        **quality_issue_data,
    }
    for key, issue_df in inspection_frames.items():
        if key == "clean_data":
            continue
        inspection_frames[key] = issue_df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

    return {
        "prepared_data": prepared_df,
        "filtered_prepared_data": inspection_frames["filtered_prepared_data"],
        "deduplicated_data": deduplicated_df,
        "clean_data": clean_df,
        "duplicate_rows": duplicate_rows,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "year_mismatch_rows": year_mismatch_rows,
        "unmatched_symbol_rows": inspection_frames["unmatched_symbol_rows"],
        "unmatched_symbols": unmatched_symbols,
        **{
            key: value
            for key, value in inspection_frames.items()
            if key not in {"clean_data", "filtered_prepared_data", "unmatched_symbol_rows"}
        },
    }


def main() -> None:
    required_symbols = prepare_required_symbols(REQUIRED_SYMBOLS_PATH)
    raw_df = load_tabular_data(HISTORICAL_PRICE_PATH)
    result = clean_historical_price_dataset(raw_df, required_symbols)
    prepared_df = result["prepared_data"]
    filtered_prepared_df = result["filtered_prepared_data"]
    deduplicated_df = result["deduplicated_data"]
    invalid_ohlc_rows = result["invalid_ohlc_rows"]
    year_mismatch_rows = result["year_mismatch_rows"]
    missing_symbol_rows = result["missing_symbol_rows"]
    missing_date_rows = result["missing_date_rows"]
    symbol_not_in_required_rows = result["symbol_not_in_required_rows"]
    zero_vol_total_rows = result["zero_vol_total_rows"]
    adj_ratio_not_one_rows = result["adj_ratio_not_one_rows"]
    vol_total_components_mismatch_rows = result["vol_total_components_mismatch_rows"]
    zero_vol_total_but_ohlc_changed_rows = result["zero_vol_total_but_ohlc_changed_rows"]
    foreign_flow_all_missing_rows = result["foreign_flow_all_missing_rows"]
    prop_trading_all_missing_rows = result["prop_trading_all_missing_rows"]
    unmatched_symbols = result["unmatched_symbols"]
    parsed_dates = pd.to_datetime(deduplicated_df["date"], errors="coerce")
    unique_symbols_in_dataset = prepared_df["symbol"].replace("", pd.NA).nunique(dropna=True)
    matched_symbols = filtered_prepared_df["symbol"].replace("", pd.NA).nunique(dropna=True)

    print("Input rows:", len(raw_df))
    print("Symbols required from symbols.csv:", len(required_symbols))
    print("Unique symbols in dataset:", unique_symbols_in_dataset)
    print("Required symbols found in dataset:", matched_symbols)
    print("Rows with symbol in symbols.csv:", len(filtered_prepared_df))
    print("Rows with symbol not in symbols.csv:", len(symbol_not_in_required_rows))
    print("Duplicate symbol-date rows:", len(result["duplicate_rows"]))
    print("Rows with invalid OHLC:", len(invalid_ohlc_rows))
    print("Clean rows:", len(result["clean_data"]))
    print("Min date:", parsed_dates.min())
    print("Max date:", parsed_dates.max())
    print("Rows with year mismatch:", len(year_mismatch_rows))
    print("Rows with missing symbol:", len(missing_symbol_rows))
    print("Unmatched symbols:", len(unmatched_symbols))
    print("Rows with missing date:", len(missing_date_rows))
    print("Rows with vol_total = 0:", len(zero_vol_total_rows))
    print("Rows with adj_ratio != 1:", len(adj_ratio_not_one_rows))
    print("Rows with vol_total != vol_deal + vol_putth:", len(vol_total_components_mismatch_rows))
    print("Rows with vol_total = 0 but OHLC changed:", len(zero_vol_total_but_ohlc_changed_rows))
    print("Rows with all foreign flow fields missing:", len(foreign_flow_all_missing_rows))
    print("Rows with all proprietary trading fields missing:", len(prop_trading_all_missing_rows))
if __name__ == "__main__":
    main()
