"""Gộp điểm sentiment cấp-bài thành chỉ số cấp-TUẦN trên toàn bộ corpus - bản
song song với build_sentiment_index_daily.py (đã sửa cùng bug path cũ, cùng
lý do bỏ predicted_label - xem file đó để biết chi tiết).

Đọc THẲNG article_scores.parquet/article_scores_intensity.parquet (không cần
predicted_label), gộp theo tuần (W-SUN, khớp quy ước week_start/week_end đã
dùng ở build_vnindex_weekly_return.py).

Output:
    data/market_sentiment_index_weekly_pmi.parquet
    data/market_sentiment_index_weekly_intensity.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEXICON_BASED_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Lexicon_based"
INPUT_PATHS_BY_METHOD = {
    "pmi": LEXICON_BASED_DIR / "Scoring" / "data" / "article_scores.parquet",
    "intensity": LEXICON_BASED_DIR / "Scoring_Intensity" / "data" / "article_scores_intensity.parquet",
}
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

DATE_COLUMN = "publication_date"
SENTIMENT_SCORE_COLUMN = "net_sentiment_score"


def prepare_article_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
    out[SENTIMENT_SCORE_COLUMN] = pd.to_numeric(out[SENTIMENT_SCORE_COLUMN], errors="coerce")
    out = out.loc[out[DATE_COLUMN].notna() & out[SENTIMENT_SCORE_COLUMN].notna()].copy()

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
    )
    weekly_index = weekly_index.reset_index()
    weekly_index["sentiment_index_z"] = standardize_series(weekly_index["sentiment_index"])
    return weekly_index


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, input_path in INPUT_PATHS_BY_METHOD.items():
        print(f"Đọc ({suffix}):", input_path)
        article_sentiment_df = pd.read_parquet(input_path, columns=[DATE_COLUMN, SENTIMENT_SCORE_COLUMN])
        weekly_index = build_weekly_market_sentiment_index(article_sentiment_df)

        output_parquet_path = OUTPUT_DIR / f"market_sentiment_index_weekly_{suffix}.parquet"
        output_csv_path = OUTPUT_DIR / f"market_sentiment_index_weekly_{suffix}.csv"
        weekly_index.to_parquet(output_parquet_path, index=False)
        weekly_index.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

        print("Output parquet:", output_parquet_path)
        print("Weekly index rows:", len(weekly_index))
        print(weekly_index.head(5).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
