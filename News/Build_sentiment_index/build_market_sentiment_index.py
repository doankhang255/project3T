from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data1" / "equity_news_content_sentiment_ratios.parquet"
OUTPUT_PARQUET_PATH = PROJECT_ROOT / "data1" / "market_sentiment_index_daily.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data1" / "market_sentiment_index_daily.csv"

DATE_COLUMN = "publication_date"
SENTIMENT_SCORE_COLUMN = "sentiment_score"
SENTIMENT_LABEL_COLUMN = "sentiment_label"

REQUIRED_COLUMNS = {
    DATE_COLUMN,
    SENTIMENT_SCORE_COLUMN,
    SENTIMENT_LABEL_COLUMN,
}


def read_input_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    

def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing columns: {sorted(missing_columns)}")


def prepare_article_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
    out[SENTIMENT_SCORE_COLUMN] = pd.to_numeric(
        out[SENTIMENT_SCORE_COLUMN],
        errors="coerce",
    )
    out[SENTIMENT_LABEL_COLUMN] = (
        out[SENTIMENT_LABEL_COLUMN].astype("string").str.strip().str.casefold()
    )

    out = out.loc[
        out[DATE_COLUMN].notna()
        & out[SENTIMENT_SCORE_COLUMN].notna()
        & out[SENTIMENT_LABEL_COLUMN].notna()
    ].copy()
    out["date"] = out[DATE_COLUMN].dt.normalize()

    return out


def standardize_series(values: pd.Series) -> pd.Series:
    standard_deviation = values.std()
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=values.index)

    return (values - values.mean()) / standard_deviation


def build_daily_market_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    article_df = prepare_article_sentiment(df)

    daily_index = article_df.groupby("date", sort=True).agg(
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
    daily_index = daily_index.reset_index()
    daily_index["sentiment_index_z"] = standardize_series(
        daily_index["sentiment_index"],
    )
    daily_index = daily_index[
        [
            "date",
            "article_count",
            "sentiment_index",
            "sentiment_index_z",
            "positive_article_count",
            "negative_article_count",
            "neutral_article_count",
        ]
    ]
    return daily_index


def save_daily_market_sentiment_index(
    daily_index: pd.DataFrame,
    output_parquet_path: Path = OUTPUT_PARQUET_PATH,
    output_csv_path: Path = OUTPUT_CSV_PATH,
) -> None:
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    daily_index.to_parquet(output_parquet_path, index=False)
    daily_index.to_csv(output_csv_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build daily market sentiment index from article-level sentiment.",
    )
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-parquet-path", type=Path, default=OUTPUT_PARQUET_PATH)
    parser.add_argument("--output-csv-path", type=Path, default=OUTPUT_CSV_PATH)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    article_sentiment_df = read_input_dataframe(args.input_path)
    daily_index = build_daily_market_sentiment_index(article_sentiment_df)
    save_daily_market_sentiment_index(
        daily_index,
        output_parquet_path=args.output_parquet_path,
        output_csv_path=args.output_csv_path,
    )

    print("Input:", args.input_path)
    print("Output parquet:", args.output_parquet_path)
    print("Output csv:", args.output_csv_path)
    print("Input rows:", len(article_sentiment_df))
    print("Daily index rows:", len(daily_index))
    print(daily_index.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
