from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_INPUT_PATH = PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_merged.parquet"
ABNORMAL_INPUT_PATH = (
    PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_abnormal_return.parquet"
)
OUTPUT_PARQUET_PATH = PROJECT_ROOT / "data_News" / "vnindex_weekly_correlation.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data_News" / "vnindex_weekly_correlation.csv"

CORRELATION_PAIRS = [
    ("sentiment_index_z", "future_ret_1w"),
    ("sentiment_index_z", "future_ret_4w"),
    ("sentiment_index_z", "future_abnormal_rolling_ret_1w"),
    ("sentiment_index_z", "future_abnormal_rolling_ret_4w"),
    ("sentiment_index_z", "future_abnormal_ar1_ret_1w"),
    ("sentiment_index_z", "future_abnormal_ar1_ret_4w"),
    ("net_positive_article_ratio", "future_ret_1w"),
    ("net_positive_article_ratio", "future_ret_4w"),
    ("net_positive_article_ratio", "future_abnormal_rolling_ret_1w"),
    ("net_positive_article_ratio", "future_abnormal_rolling_ret_4w"),
    ("net_positive_article_ratio", "future_abnormal_ar1_ret_1w"),
    ("net_positive_article_ratio", "future_abnormal_ar1_ret_4w"),
]


def add_sentiment_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    article_count = pd.to_numeric(out["article_count"], errors="coerce")
    article_count = article_count.replace(0, np.nan)

    out["positive_article_ratio"] = out["positive_article_count"] / article_count
    out["negative_article_ratio"] = out["negative_article_count"] / article_count
    out["neutral_article_ratio"] = out["neutral_article_count"] / article_count
    out["net_positive_article_ratio"] = (
        out["positive_article_count"] - out["negative_article_count"]
    ) / article_count

    return out


def build_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    out = add_sentiment_ratios(df)
    rows = []

    for sentiment_column, return_column in CORRELATION_PAIRS:
        if sentiment_column not in out.columns:
            continue

        if return_column not in out.columns:
            continue

        pair_df = out[[sentiment_column, return_column]].dropna()
        if len(pair_df) < 3:
            continue

        rows.append(
            {
                "sentiment_variable": sentiment_column,
                "return_variable": return_column,
                "method": "pearson",
                "correlation": pair_df[sentiment_column].corr(
                    pair_df[return_column],
                    method="pearson",
                ),
                "observation_count": len(pair_df),
            }
        )
        rows.append(
            {
                "sentiment_variable": sentiment_column,
                "return_variable": return_column,
                "method": "spearman",
                "correlation": pair_df[sentiment_column].corr(
                    pair_df[return_column],
                    method="spearman",
                ),
                "observation_count": len(pair_df),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["return_variable", "method", "sentiment_variable"],
        kind="mergesort",
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_path = ABNORMAL_INPUT_PATH if ABNORMAL_INPUT_PATH.exists() else BASE_INPUT_PATH
    merged_df = pd.read_parquet(input_path)
    correlation_df = build_correlation_table(merged_df)

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    correlation_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    correlation_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input:", input_path)
    print("Output parquet:", OUTPUT_PARQUET_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("Input rows:", len(merged_df))
    print("Correlation rows:", len(correlation_df))
    print(correlation_df.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
