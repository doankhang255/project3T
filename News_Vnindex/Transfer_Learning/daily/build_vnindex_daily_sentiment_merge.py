"""Merge chỉ số sentiment cấp-ngày của PhoBERT fine-tune (E3,
Transfer_Learning/inference/build_daily_sentiment_index.py) với dữ liệu
VN-Index cấp ngày.

Không viết lại logic merge - import thẳng hàm thuần
``merge_vnindex_daily_with_sentiment`` từ News_Vnindex/Common (đã tổng quát
theo schema (date, article_count, sentiment_index, sentiment_index_z), đúng
những gì build_daily_sentiment_index.py xuất ra). Chỉ nối dây (wiring) input/
output riêng cho phương pháp "phobert" ở đây, không sửa file Common - cùng
nguyên tắc self-contained mà model/repeated_cv_tune_holdout.py đã theo.

Output (chung thư mục data_News/ với các bản Lexicon, phân biệt bằng hậu tố
"_phobert" - để so sánh trực tiếp các phương pháp trên cùng 1 chỗ):
    data_News/vnindex_daily_sentiment_merged_phobert.parquet
    data_News/vnindex_daily_sentiment_merged_phobert.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News_Vnindex.Common.merge_vnindex_daily_with_sentiment import (  # noqa: E402
    merge_vnindex_daily_with_sentiment,
)

VNINDEX_DAILY_PATH = PROJECT_ROOT / "data_Histo" / "vnindex_eda_output.csv"
SENTIMENT_DAILY_PATH = (
    PROJECT_ROOT
    / "News"
    / "Build_sentiment_label"
    / "Transfer_Learning"
    / "inference"
    / "data"
    / "market_sentiment_index_daily_phobert.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "data_News"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vnindex_daily_df = pd.read_csv(VNINDEX_DAILY_PATH, encoding="utf-8-sig")
    sentiment_daily_df = pd.read_parquet(SENTIMENT_DAILY_PATH)

    merged_df = merge_vnindex_daily_with_sentiment(vnindex_daily_df, sentiment_daily_df)

    output_parquet_path = OUTPUT_DIR / "vnindex_daily_sentiment_merged_phobert.parquet"
    output_csv_path = OUTPUT_DIR / "vnindex_daily_sentiment_merged_phobert.csv"
    merged_df.to_parquet(output_parquet_path, index=False)
    merged_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print("Sentiment daily input:", SENTIMENT_DAILY_PATH)
    print("Output parquet:", output_parquet_path)
    print("Sentiment daily rows:", len(sentiment_daily_df))
    print("Merged rows:", len(merged_df))
    print(merged_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
