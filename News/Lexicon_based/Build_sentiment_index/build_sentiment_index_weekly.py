from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEXICON_DATA_DIR = PROJECT_ROOT / "News" / "Lexicon_based" / "data"
INPUT_PATH = LEXICON_DATA_DIR / "equity_news_content_sentiment_ratios.parquet"
OUTPUT_PARQUET_PATH = LEXICON_DATA_DIR / "market_sentiment_index_weekly.parquet"
OUTPUT_CSV_PATH = LEXICON_DATA_DIR / "market_sentiment_index_weekly.csv"

DATE_COLUMN = "publication_date"
SENTIMENT_SCORE_COLUMN = "sentiment_score"
SENTIMENT_LABEL_COLUMN = "sentiment_label"

def prepare_article_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
    out[SENTIMENT_SCORE_COLUMN] = pd.to_numeric(out[SENTIMENT_SCORE_COLUMN], errors="coerce",)
    
    out[SENTIMENT_LABEL_COLUMN] = (out[SENTIMENT_LABEL_COLUMN].astype("string").str.strip().str.casefold())

    out = out.loc[
        out[DATE_COLUMN].notna()
        & out[SENTIMENT_SCORE_COLUMN].notna()
        & out[SENTIMENT_LABEL_COLUMN].notna()
    ].copy()

    weekly_period = out[DATE_COLUMN].dt.to_period("W-SUN")
    out["week_start"] = weekly_period.apply(lambda period: period.start_time).dt.normalize()
    out["week_end"] = weekly_period.apply(lambda period: period.end_time).dt.normalize()

    return out


def standardize_series(values: pd.Series) -> pd.Series:
    standard_deviation = values.std()
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=values.index)

    return (values - values.mean()) / standard_deviation


def build_weekly_market_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    article_df = prepare_article_sentiment(df)

    weekly_index = article_df.groupby(["week_start", "week_end"], sort=True).agg(
        article_count=(SENTIMENT_SCORE_COLUMN, "size"),
        sentiment_index=(SENTIMENT_SCORE_COLUMN, "mean"),

        positive_article_count=(SENTIMENT_LABEL_COLUMN, lambda values: values.eq("positive").sum(),),
        negative_article_count=(SENTIMENT_LABEL_COLUMN, lambda values: values.eq("negative").sum(),),
        neutral_article_count=(SENTIMENT_LABEL_COLUMN, lambda values: values.eq("neutral").sum(),),
    )
    weekly_index = weekly_index.reset_index()
    weekly_index["sentiment_index_z"] = standardize_series(
        weekly_index["sentiment_index"],
    )
    weekly_index = weekly_index[
        [
            "week_start",
            "week_end",
            "article_count",
            "sentiment_index",
            "sentiment_index_z",
            "positive_article_count",
            "negative_article_count",
            "neutral_article_count",
        ]
    ]
    return weekly_index


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    article_sentiment_df = pd.read_parquet(INPUT_PATH)

    weekly_index = build_weekly_market_sentiment_index(article_sentiment_df)

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    weekly_index.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    weekly_index.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input:", INPUT_PATH)
    print("Output parquet:", OUTPUT_PARQUET_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("Input rows:", len(article_sentiment_df))
    print("Weekly index rows:", len(weekly_index))
    print(weekly_index.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
