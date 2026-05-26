from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data_News" / "equity_news_content_sentiment_ratios.parquet"
OUTPUT_PARQUET_PATH = PROJECT_ROOT / "data_News" / "market_sentiment_index_monthly.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data_News" / "market_sentiment_index_monthly.csv"

DATE_COLUMN = "publication_date"
SENTIMENT_SCORE_COLUMN = "sentiment_score"
SENTIMENT_LABEL_COLUMN = "sentiment_label"


def prepare_article_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
    out[SENTIMENT_SCORE_COLUMN] = pd.to_numeric(out[SENTIMENT_SCORE_COLUMN], errors="coerce")
    out[SENTIMENT_LABEL_COLUMN] = (
        out[SENTIMENT_LABEL_COLUMN].astype("string").str.strip().str.casefold()
    )

    out = out.loc[
        out[DATE_COLUMN].notna()
        & out[SENTIMENT_SCORE_COLUMN].notna()
        & out[SENTIMENT_LABEL_COLUMN].notna()
    ].copy()

    monthly_period = out[DATE_COLUMN].dt.to_period("M")
    out["month"] = monthly_period.astype(str)
    out["month_start"] = monthly_period.apply(lambda period: period.start_time).dt.normalize()
    out["month_end"] = monthly_period.apply(lambda period: period.end_time).dt.normalize()

    return out


def standardize_series(values: pd.Series) -> pd.Series:
    standard_deviation = values.std()
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=values.index)

    return (values - values.mean()) / standard_deviation


def build_monthly_market_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    article_df = prepare_article_sentiment(df)

    monthly_index = article_df.groupby(["month", "month_start", "month_end"], sort=True).agg(
        article_count=(SENTIMENT_SCORE_COLUMN, "size"),
        sentiment_index=(SENTIMENT_SCORE_COLUMN, "mean"),
        positive_article_count=(
            SENTIMENT_LABEL_COLUMN,
            lambda values: values.eq("positive").sum(),
        ),
        negative_article_count=(
            SENTIMENT_LABEL_COLUMN,
            lambda values: values.eq("negative").sum(),
        ),
        neutral_article_count=(
            SENTIMENT_LABEL_COLUMN,
            lambda values: values.eq("neutral").sum(),
        ),
    )

    monthly_index = monthly_index.reset_index()
    monthly_index["sentiment_index_z"] = standardize_series(
        monthly_index["sentiment_index"],
    )
    monthly_index = monthly_index[
        [
            "month",
            "month_start",
            "month_end",
            "article_count",
            "sentiment_index",
            "sentiment_index_z",
            "positive_article_count",
            "negative_article_count",
            "neutral_article_count",
        ]
    ]
    return monthly_index


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    article_sentiment_df = pd.read_parquet(INPUT_PATH)

    monthly_index = build_monthly_market_sentiment_index(article_sentiment_df)

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly_index.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    monthly_index.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input:", INPUT_PATH)
    print("Output parquet:", OUTPUT_PARQUET_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("Input rows:", len(article_sentiment_df))
    print("Monthly index rows:", len(monthly_index))
    print(monthly_index.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
