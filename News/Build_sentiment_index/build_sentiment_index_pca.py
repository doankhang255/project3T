"""Chỉ số sentiment kiểu Tetlock THẬT - PCA trên nhiều category, không phải
trung bình đơn giản positive-negative như build_sentiment_index_daily.py.

Tetlock (2007) dùng PCA trên 77 category GI, lấy principal component 1 (eigen-
vector ứng với eigenvalue lớn nhất) làm "pessimism factor" - lý do: 1 chỉ số
tổng hợp tối ưu (theo nghĩa giữ được nhiều phương sai nhất) từ NHIỀU khía
cạnh ngữ nghĩa, thay vì chỉ dùng riêng 1 cặp positive/negative.

Ở đây dùng PCA trên 7 category LM đã có sẵn (negative/positive/uncertainty/
litigious/strong_modal/weak_modal/constraining_score trong article_scores*.parquet)
thay vì 77 category GI của Tetlock.

Quy trình:
    1. Gộp 7 category score theo NGÀY (hoặc TUẦN) - mean mỗi category/kỳ.
    2. Chuẩn hóa (z-score) từng category riêng - PCA cần cùng thang đo.
    3. PCA trên ma trận tương quan 7x7 (np.linalg.eigh, đối xứng) - lấy
       eigenvector ứng eigenvalue LỚN NHẤT (component 1).
    4. Định hướng dấu: cho tương quan DƯƠNG với (positive_score-negative_score)
       - để "factor cao = thiên tích cực", cùng chiều net_sentiment_score đã
       dùng ở các file khác cho dễ so sánh (PCA tự thân không xác định dấu).

GIỚI HẠN so với bản gốc Tetlock (ghi rõ để không hiểu nhầm):
    - Tetlock fit loadings bằng dữ liệu năm t-1, áp dụng cho năm t (tránh
      look-ahead). Ở đây fit 1 lần trên TOÀN BỘ mẫu (đơn giản hơn, có
      look-ahead nhẹ - loadings "biết" cả tương lai) - làm bản đơn giản
      trước, có thể nâng cấp sau nếu factor này cho tín hiệu tốt.
    - Tetlock re-center category theo thứ-trong-tuần trước khi PCA (khử
      seasonality khỏi factor). Chưa làm ở đây.

Output (khớp đúng schema build_sentiment_index_daily.py/_weekly.py để cắm
thẳng vào merge_vnindex_daily_with_sentiment.py / merge_vnindex_weekly_with_sentiment.py
bằng cách thêm entry "pca_pmi"/"pca_intensity" vào *_PATHS_BY_METHOD):
    data/market_sentiment_index_daily_pca_pmi.parquet
    data/market_sentiment_index_daily_pca_intensity.parquet
    data/market_sentiment_index_weekly_pca_pmi.parquet
    data/market_sentiment_index_weekly_pca_intensity.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEXICON_BASED_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Lexicon_based"
INPUT_PATHS_BY_METHOD = {
    "pca_pmi": LEXICON_BASED_DIR / "Scoring" / "data" / "article_scores.parquet",
    "pca_intensity": LEXICON_BASED_DIR / "Scoring_Intensity" / "data" / "article_scores_intensity.parquet",
}
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

DATE_COLUMN = "publication_date"
CATEGORY_SCORE_COLUMNS = [
    "negative_score",
    "positive_score",
    "uncertainty_score",
    "litigious_score",
    "strong_modal_score",
    "weak_modal_score",
    "constraining_score",
]


def standardize_series(values: pd.Series) -> pd.Series:
    standard_deviation = values.std()
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / standard_deviation


def compute_pca_factor(period_category_df: pd.DataFrame) -> tuple[pd.Series, dict, float]:
    """PCA trên 7 category (đã chuẩn hóa) của bảng đã gộp theo kỳ (ngày/tuần).
    Trả về (factor thô theo từng dòng, loadings {category: hệ số}, tỉ lệ
    phương sai component 1 giải thích được)."""
    standardized = period_category_df[CATEGORY_SCORE_COLUMNS].apply(standardize_series)
    correlation_matrix = standardized.to_numpy().T @ standardized.to_numpy() / (len(standardized) - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)  # tăng dần
    top_index = int(np.argmax(eigenvalues))
    loadings = eigenvectors[:, top_index]

    raw_factor = standardized.to_numpy() @ loadings

    net_sentiment = period_category_df["positive_score"] - period_category_df["negative_score"]
    if np.corrcoef(raw_factor, net_sentiment)[0, 1] < 0:
        loadings = -loadings
        raw_factor = -raw_factor

    explained_variance_ratio = float(eigenvalues[top_index] / eigenvalues.sum())
    loadings_dict = dict(zip(CATEGORY_SCORE_COLUMNS, loadings))
    return pd.Series(raw_factor, index=period_category_df.index), loadings_dict, explained_variance_ratio


def prepare_article_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
    for column in CATEGORY_SCORE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.loc[out[DATE_COLUMN].notna()].copy()
    return out


def build_daily_pca_index(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, float]:
    out = prepare_article_scores(df)
    out["date"] = out[DATE_COLUMN].dt.normalize()

    daily_categories = out.groupby("date", sort=True)[CATEGORY_SCORE_COLUMNS].mean()
    daily_categories["article_count"] = out.groupby("date", sort=True).size()
    daily_categories = daily_categories.reset_index()

    factor, loadings, explained_variance_ratio = compute_pca_factor(daily_categories)
    daily_categories["sentiment_index"] = factor
    daily_categories["sentiment_index_z"] = standardize_series(factor)

    return daily_categories[["date", "article_count", "sentiment_index", "sentiment_index_z"]], loadings, explained_variance_ratio


def build_weekly_pca_index(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, float]:
    out = prepare_article_scores(df)
    weekly_period = out[DATE_COLUMN].dt.to_period("W-SUN")
    out["week_start"] = weekly_period.apply(lambda period: period.start_time).dt.normalize()
    out["week_end"] = weekly_period.apply(lambda period: period.end_time).dt.normalize()

    weekly_categories = out.groupby(["week_start", "week_end"], sort=True)[CATEGORY_SCORE_COLUMNS].mean()
    weekly_categories["article_count"] = out.groupby(["week_start", "week_end"], sort=True).size()
    weekly_categories = weekly_categories.reset_index()

    factor, loadings, explained_variance_ratio = compute_pca_factor(weekly_categories)
    weekly_categories["sentiment_index"] = factor
    weekly_categories["sentiment_index_z"] = standardize_series(factor)

    return (
        weekly_categories[["week_start", "week_end", "article_count", "sentiment_index", "sentiment_index_z"]],
        loadings,
        explained_variance_ratio,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, input_path in INPUT_PATHS_BY_METHOD.items():
        print(f"=== {suffix} ===")
        article_df = pd.read_parquet(input_path, columns=[DATE_COLUMN] + CATEGORY_SCORE_COLUMNS)

        daily_df, daily_loadings, daily_evr = build_daily_pca_index(article_df)
        daily_df.to_parquet(OUTPUT_DIR / f"market_sentiment_index_daily_{suffix}.parquet", index=False)
        daily_df.to_csv(OUTPUT_DIR / f"market_sentiment_index_daily_{suffix}.csv", index=False, encoding="utf-8-sig")

        weekly_df, weekly_loadings, weekly_evr = build_weekly_pca_index(article_df)
        weekly_df.to_parquet(OUTPUT_DIR / f"market_sentiment_index_weekly_{suffix}.parquet", index=False)
        weekly_df.to_csv(OUTPUT_DIR / f"market_sentiment_index_weekly_{suffix}.csv", index=False, encoding="utf-8-sig")

        print(f"Daily rows: {len(daily_df)}  |  component 1 giải thích {daily_evr*100:.1f}% phương sai (7 category)")
        print("Loadings (daily):")
        for category, loading in sorted(daily_loadings.items(), key=lambda item: -abs(item[1])):
            print(f"  {category:>20}: {loading:+.3f}")
        print()
        print(f"Weekly rows: {len(weekly_df)}  |  component 1 giải thích {weekly_evr*100:.1f}% phương sai")
        print("Loadings (weekly):")
        for category, loading in sorted(weekly_loadings.items(), key=lambda item: -abs(item[1])):
            print(f"  {category:>20}: {loading:+.3f}")
        print()


if __name__ == "__main__":
    main()
