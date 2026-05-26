from __future__ import annotations
from pathlib import Path
import pandas as pd

from EDA_raw_historical_price import (
        HISTORICAL_PRICE_PATH,
        load_tabular_data,
        mark_duplicate_symbol_date,
        reconcile_year_from_date,
        remove_invalid_ohlc_rows,
        standardize_types,
)


def filter_vnindex_records(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["symbol"].eq("VNINDEX")].copy()


def collect_vnindex_issue_rows(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = df.copy()

    zero_vol_total_mask = out["vol_total"].notna() & out["vol_total"].eq(0)
    adj_ratio_not_one_mask = out["adj_ratio"].notna() & out["adj_ratio"].ne(1)
    adj_ratio_equal_one_mask = out["adj_ratio"].notna() & out["adj_ratio"].eq(1)

    ohlc_columns = ["open_price", "high_price", "low_price", "close_price"]
    complete_ohlc_mask = out[ohlc_columns].notna().all(axis=1)
    rounded_ohlc = out.loc[complete_ohlc_mask, ohlc_columns].round(3)
    zero_vol_total_but_ohlc_changed_mask = pd.Series(False, index=out.index)
    zero_vol_total_but_ohlc_changed_mask.loc[complete_ohlc_mask] = (
        out.loc[complete_ohlc_mask, "vol_total"].fillna(0).eq(0)
        & rounded_ohlc.max(axis=1).gt(rounded_ohlc.min(axis=1))
    )

    return {
        "zero_vol_total_rows": out.loc[zero_vol_total_mask].copy(),
        "adj_ratio_not_one_rows": out.loc[adj_ratio_not_one_mask].copy(),
        "adj_ratio_equal_one_rows": out.loc[adj_ratio_equal_one_mask].copy(),
        "zero_vol_total_but_ohlc_changed_rows": out.loc[
            zero_vol_total_but_ohlc_changed_mask
        ].copy(),
    }


def build_vnindex_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prepared_df = standardize_types(df)
    vnindex_df = filter_vnindex_records(prepared_df)
    vnindex_df, year_mismatch_rows = reconcile_year_from_date(vnindex_df)
    vnindex_df, duplicate_rows = mark_duplicate_symbol_date(vnindex_df)
    vnindex_df, invalid_ohlc_rows = remove_invalid_ohlc_rows(vnindex_df)
    issue_rows = collect_vnindex_issue_rows(vnindex_df)

    sort_columns = ["symbol", "date"]
    vnindex_df = vnindex_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    year_mismatch_rows = year_mismatch_rows.sort_values(
        sort_columns,
        kind="mergesort",
    ).reset_index(drop=True)
    duplicate_rows = duplicate_rows.sort_values(
        sort_columns,
        kind="mergesort",
    ).reset_index(drop=True)
    invalid_ohlc_rows = invalid_ohlc_rows.sort_values(
        sort_columns,
        kind="mergesort",
    ).reset_index(drop=True)
    for key, issue_df in issue_rows.items():
        issue_rows[key] = issue_df.sort_values(
            sort_columns,
            kind="mergesort",
        ).reset_index(drop=True)

    return {
        "vnindex_data": vnindex_df,
        "year_mismatch_rows": year_mismatch_rows,
        "duplicate_rows": duplicate_rows,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        **issue_rows,
    }


def load_vnindex_eda(path: str | Path = HISTORICAL_PRICE_PATH) -> dict[str, pd.DataFrame]:
    raw_df = load_tabular_data(path)
    return build_vnindex_eda(raw_df)


def main() -> None:
    result = load_vnindex_eda()
    vnindex_df = result["vnindex_data"]
    duplicate_rows = result["duplicate_rows"]
    year_mismatch_rows = result["year_mismatch_rows"]
    invalid_ohlc_rows = result["invalid_ohlc_rows"]
    zero_vol_total_but_ohlc_changed_rows = result["zero_vol_total_but_ohlc_changed_rows"]
    zero_vol_total_rows = result["zero_vol_total_rows"]
    adj_ratio_not_one_rows = result["adj_ratio_not_one_rows"]
    adj_ratio_equal_one_rows = result["adj_ratio_equal_one_rows"]
    parsed_dates = pd.to_datetime(vnindex_df["date"], errors="coerce")
    output = vnindex_df.to_csv("data_Histo/vnindex_eda_output.csv", index=False)

    print("Input records:", len(vnindex_df))
    print("Min Date:", parsed_dates.min())
    print("Max Date:", parsed_dates.max())
    print("Duplicate records by symbol-date:", len(duplicate_rows))
    print("Rows with year mismatch:", len(year_mismatch_rows))
    print("Rows with invalid OHLC:", len(invalid_ohlc_rows))
    print("Rows with vol_total = 0 but OHLC changed:", len(zero_vol_total_but_ohlc_changed_rows))
    print("Rows with vol_total = 0:", len(zero_vol_total_rows))
    print("Rows with adj_ratio != 1:", len(adj_ratio_not_one_rows))
    print("Rows with adj_ratio = 1:", len(adj_ratio_equal_one_rows))

if __name__ == "__main__":
    main()
