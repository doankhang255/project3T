from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VNINDEX_WEEKLY_PATH = PROJECT_ROOT / "data_Histo" / "vnindex_weekly_return.parquet"
SENTIMENT_WEEKLY_PATHS_BY_METHOD = {
    "pmi": PROJECT_ROOT / "News" / "Build_sentiment_index" / "data" / "market_sentiment_index_weekly_pmi.parquet",
    "intensity": PROJECT_ROOT
    / "News"
    / "Build_sentiment_index"
    / "data"
    / "market_sentiment_index_weekly_intensity.parquet",
    "pca_pmi": PROJECT_ROOT
    / "News"
    / "Build_sentiment_index"
    / "data"
    / "market_sentiment_index_weekly_pca_pmi.parquet",
    "pca_intensity": PROJECT_ROOT
    / "News"
    / "Build_sentiment_index"
    / "data"
    / "market_sentiment_index_weekly_pca_intensity.parquet",
}
OUTPUT_DIR = PROJECT_ROOT / "data_News"


def prepare_week_key(df: pd.DataFrame, column: str = "week_end") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    out = out.loc[out[column].notna()].copy()
    return out


def merge_vnindex_weekly_with_sentiment(
    vnindex_weekly_df: pd.DataFrame,
    sentiment_weekly_df: pd.DataFrame,
) -> pd.DataFrame:
    vnindex_weekly_df = prepare_week_key(vnindex_weekly_df)
    sentiment_weekly_df = prepare_week_key(sentiment_weekly_df)

    merged_df = sentiment_weekly_df.merge(
        vnindex_weekly_df,
        on="week_end",
        how="inner",
        suffixes=("_sentiment", "_vnindex"),
    )

    if "week_start_sentiment" in merged_df.columns:
        merged_df = merged_df.rename(columns={"week_start_sentiment": "week_start"})
    if "week_start_vnindex" in merged_df.columns:
        merged_df = merged_df.drop(columns=["week_start_vnindex"])

    merged_df["log_article_count"] = np.log1p(merged_df["article_count"])

    ordered_columns = [
        "week_start",
        "week_end",
        "article_count",
        "sentiment_index",
        "sentiment_index_z",
        "positive_article_count",
        "negative_article_count",
        "neutral_article_count",
        "log_article_count",
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
    existing_columns = [column for column in ordered_columns if column in merged_df.columns]
    remaining_columns = [column for column in merged_df.columns if column not in existing_columns]
    merged_df = merged_df[existing_columns + remaining_columns]

    return merged_df.sort_values("week_end", kind="mergesort").reset_index(drop=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vnindex_weekly_df = pd.read_parquet(VNINDEX_WEEKLY_PATH)

    for method_suffix, sentiment_weekly_path in SENTIMENT_WEEKLY_PATHS_BY_METHOD.items():
        sentiment_weekly_df = pd.read_parquet(sentiment_weekly_path)
        merged_df = merge_vnindex_weekly_with_sentiment(vnindex_weekly_df, sentiment_weekly_df)

        output_parquet_path = OUTPUT_DIR / f"vnindex_weekly_sentiment_merged_{method_suffix}.parquet"
        output_csv_path = OUTPUT_DIR / f"vnindex_weekly_sentiment_merged_{method_suffix}.csv"
        merged_df.to_parquet(output_parquet_path, index=False)
        merged_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

        print(f"--- {method_suffix} ---")
        print("Sentiment weekly input:", sentiment_weekly_path)
        print("Output parquet:", output_parquet_path)
        print("VN-Index weekly rows:", len(vnindex_weekly_df))
        print("Sentiment weekly rows:", len(sentiment_weekly_df))
        print("Merged rows:", len(merged_df))
        print(merged_df.head(10).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
