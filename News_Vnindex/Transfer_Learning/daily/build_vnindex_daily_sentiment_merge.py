"""Bước 1/3 VN-Index - Transfer_Learning (PhoBERT E3 fine-tune): merge chỉ số sentiment cấp ngày
(News/Build_sentiment_index/data/market_sentiment_index_daily_*.parquet) với
VN-Index cấp ngày. Logic ở News_Vnindex/Common/merge_vnindex_daily_with_sentiment.py;
file này chỉ khai báo input của method này.

Output: data_News/vnindex_daily_sentiment_merged_phobert.{parquet,csv}
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News_Vnindex.Common.merge_vnindex_daily_with_sentiment import (  # noqa: E402
    SENTIMENT_INDEX_DIR,
    run_merge,
)

SENTIMENT_DAILY_PATHS_BY_METHOD = {
    "phobert": SENTIMENT_INDEX_DIR / "market_sentiment_index_daily_phobert.parquet",
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_merge(SENTIMENT_DAILY_PATHS_BY_METHOD)


if __name__ == "__main__":
    main()
