"""Bước 2/3 VN-Index - Lexicon (PMI / Intensity): abnormal return cấp ngày
(rolling mean 120 ngày + AR(1) rolling 252 ngày). Logic ở
News_Vnindex/Common/vnindex_daily_abnormal_return.py; file này chỉ khai báo
input của method này.

Output: data_News/vnindex_daily_sentiment_abnormal_return_{pmi, intensity}.{parquet,csv}
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News_Vnindex.Common.vnindex_daily_abnormal_return import (  # noqa: E402
    OUTPUT_DIR,
    run_abnormal_return,
)

INPUT_PATHS_BY_METHOD = {
    "pmi": OUTPUT_DIR / "vnindex_daily_sentiment_merged_pmi.parquet",
    "intensity": OUTPUT_DIR / "vnindex_daily_sentiment_merged_intensity.parquet",
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_abnormal_return(INPUT_PATHS_BY_METHOD)


if __name__ == "__main__":
    main()
