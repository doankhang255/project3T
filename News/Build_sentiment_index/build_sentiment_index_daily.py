"""Gộp điểm sentiment cấp-bài thành chỉ số cấp-ngày trên toàn bộ corpus -
dùng chung cho MỌI phương pháp chấm điểm (mỗi nhánh chỉ tự chấm điểm 126k bài
bằng model của nó, rồi đưa file điểm vào đây).

Đọc THẲNG file điểm cấp-bài của từng phương pháp - chỉ cần
`publication_date` + `net_sentiment_score` (nhãn 3 lớp chỉ phục vụ đối chiếu
ground truth, không liên quan đến việc tính chỉ số ngày này).

Nguồn:
    - Cách 1 (PMI):        Lexicon_based/Scoring/data/article_scores.parquet
    - Cách 2 (intensity):  Lexicon_based/Scoring_Intensity/data/article_scores_intensity.parquet
    - PhoBERT (E3):        Transfer_Learning/inference/data/article_scores_phobert.parquet
    - Random Forest:       Traditional_ML/inference/data/article_scores_random_forest.parquet

Xuất RIÊNG 1 file/phương pháp - đúng schema (date, article_count,
sentiment_index, sentiment_index_z) mà
News_Vnindex/Common/merge_vnindex_daily_with_sentiment.py cần.

Output:
    data/market_sentiment_index_daily_{pmi,intensity,phobert,random_forest}.{parquet,csv}
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENTIMENT_LABEL_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label"
LEXICON_BASED_DIR = SENTIMENT_LABEL_DIR / "Lexicon_based"
INPUT_PATHS_BY_METHOD = {
    "pmi": LEXICON_BASED_DIR / "Scoring" / "data" / "article_scores.parquet",
    "intensity": LEXICON_BASED_DIR / "Scoring_Intensity" / "data" / "article_scores_intensity.parquet",
    "phobert": SENTIMENT_LABEL_DIR / "Transfer_Learning" / "inference" / "data" / "article_scores_phobert.parquet",
    "random_forest": SENTIMENT_LABEL_DIR
    / "Traditional_ML"
    / "inference"
    / "data"
    / "article_scores_random_forest.parquet",
}
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, input_path in INPUT_PATHS_BY_METHOD.items():
        print(f"Đọc ({suffix}):", input_path)
        article_sentiment_df = pd.read_parquet(input_path, columns=[DATE_COLUMN, SENTIMENT_SCORE_COLUMN])
        daily_index = build_daily_market_sentiment_index(article_sentiment_df)

        output_parquet_path = OUTPUT_DIR / f"market_sentiment_index_daily_{suffix}.parquet"
        output_csv_path = OUTPUT_DIR / f"market_sentiment_index_daily_{suffix}.csv"
        daily_index.to_parquet(output_parquet_path, index=False)
        daily_index.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

        print("Output parquet:", output_parquet_path)
        print("Daily index rows:", len(daily_index))
        print(daily_index.head(5).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
