"""Bước 3/3 VN-Index - Lexicon (PMI / Intensity): hồi quy dự báo kiểu Tetlock (2007)
cấp ngày trên abnormal return, Newey-West HAC 5 lag, kiểm định tổng lag 2-5.
Đặc tả + logic ở News_Vnindex/Common/vnindex_daily_predictive_regression.py
(dùng chung cho mọi method); file này chỉ khai báo input của method này.

Output: data_News/vnindex_daily_predictive_regression.csv
        data_News/vnindex_daily_regression_lag_sum_test.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News_Vnindex.Common.vnindex_daily_predictive_regression import (  # noqa: E402
    OUTPUT_DIR,
    run_predictive_regression,
)

MERGED_DATA_PATHS_BY_METHOD = {
    "Cach1_PMI": OUTPUT_DIR / "vnindex_daily_sentiment_abnormal_return_pmi.parquet",
    "Cach2_Intensity": OUTPUT_DIR / "vnindex_daily_sentiment_abnormal_return_intensity.parquet",
}

# Hậu tố tên file output của nhánh này - dùng chung với run_diagnostics.py.
OUTPUT_SUFFIX = ""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_predictive_regression(MERGED_DATA_PATHS_BY_METHOD, output_suffix=OUTPUT_SUFFIX)


if __name__ == "__main__":
    main()
