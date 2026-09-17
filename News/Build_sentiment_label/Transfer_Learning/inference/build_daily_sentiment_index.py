"""Gộp điểm sentiment cấp-bài (article_scores_phobert.parquet, output của
score_corpus_phobert.py) thành chỉ số sentiment thị trường cấp NGÀY.

Cùng công thức và schema đầu ra (date, article_count, sentiment_index,
sentiment_index_z) với News/Build_sentiment_index/build_sentiment_index_daily.py
(nhánh Lexicon) - để News_Vnindex/Common/merge_vnindex_daily_with_sentiment.py
dùng lại được nguyên hàm merge, không cần sửa. Giữ script riêng ở đây (không
sửa thẳng file của Lexicon) để nhánh Transfer_Learning tự chứa, đúng quy ước
các nhánh khác trong repo (mỗi nhánh sở hữu code scoring/đánh giá riêng).

Output:
    data/market_sentiment_index_daily_phobert.parquet
    data/market_sentiment_index_daily_phobert.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "data" / "article_scores_phobert.parquet"
OUTPUT_DIR = SCRIPT_DIR / "data"

DATE_COLUMN = "publication_date"
SENTIMENT_SCORE_COLUMN = "net_sentiment_score"


def prepare_article_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
    out[SENTIMENT_SCORE_COLUMN] = pd.to_numeric(out[SENTIMENT_SCORE_COLUMN], errors="coerce")

    out = out.loc[out[DATE_COLUMN].notna() & out[SENTIMENT_SCORE_COLUMN].notna()].copy()
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
    )
    daily_index = daily_index.reset_index()
    daily_index["sentiment_index_z"] = standardize_series(daily_index["sentiment_index"])
    return daily_index


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Đọc:", INPUT_PATH)
    article_sentiment_df = pd.read_parquet(INPUT_PATH, columns=[DATE_COLUMN, SENTIMENT_SCORE_COLUMN])
    daily_index = build_daily_market_sentiment_index(article_sentiment_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_parquet_path = OUTPUT_DIR / "market_sentiment_index_daily_phobert.parquet"
    output_csv_path = OUTPUT_DIR / "market_sentiment_index_daily_phobert.csv"
    daily_index.to_parquet(output_parquet_path, index=False)
    daily_index.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print("Output parquet:", output_parquet_path)
    print("Daily index rows:", len(daily_index))
    print(daily_index.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
