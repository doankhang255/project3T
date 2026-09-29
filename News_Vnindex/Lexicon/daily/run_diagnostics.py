"""Chẩn đoán + kiểm định bổ sung VN-Index - Lexicon (PMI / Intensity).
Chạy SAU 3 bước chính (merge -> abnormal return -> vnindex_daily_predictive_regression.py).
Logic ở News_Vnindex/Common/ (dùng chung cho mọi nhánh); file này chỉ gọi
chúng với method + hậu tố output của nhánh này (lấy từ
vnindex_daily_predictive_regression.py cùng thư mục).

Output trong data_News/ (hậu tố ""):
    model_r_squared_diagnosis.png              - R² theo nhóm biến + F-test 5 lag sentiment
    predictor_correlation_heatmap.png, predictor_vif.csv - cộng tuyến (VIF)
    verify_abnormal_return.png                 - kiểm tra abnormal return tính đúng
    vnindex_daily_volume_regression_*.csv      - sentiment -> khối lượng giao dịch
    vnindex_daily_foreign_volume_regression_*.csv - khối ngoại vs khối nội (year FE)
    ols_fit_vs_actual.png                      - thực tế vs OLS dự báo
    sentiment_lag_coefficients.png             - hệ số từng lag + CI 95%
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News_Vnindex.Common.check_predictor_multicollinearity import run_multicollinearity_check  # noqa: E402
from News_Vnindex.Common.diagnose_model_r_squared import run_r_squared_diagnosis  # noqa: E402
from News_Vnindex.Common.plot_ols_fit_vs_actual import run_ols_fit_plot  # noqa: E402
from News_Vnindex.Common.plot_sentiment_lag_coefficients import run_sentiment_lag_plot  # noqa: E402
from News_Vnindex.Common.verify_abnormal_return import run_abnormal_return_verification  # noqa: E402
from News_Vnindex.Common.vnindex_daily_foreign_volume_regression import (  # noqa: E402
    run_foreign_volume_regression,
)
from News_Vnindex.Common.vnindex_daily_sentiment_reverse_causality import (  # noqa: E402
    run_reverse_causality,
)
from News_Vnindex.Common.vnindex_daily_volume_regression import run_volume_regression  # noqa: E402
from vnindex_daily_predictive_regression import (  # noqa: E402  (cùng thư mục)
    MERGED_DATA_PATHS_BY_METHOD,
    OUTPUT_SUFFIX,
)

DIAGNOSTICS = [
    ("R² theo nhóm biến + F-test", run_r_squared_diagnosis),
    ("Cộng tuyến (VIF)", run_multicollinearity_check),
    ("Kiểm tra abnormal return", run_abnormal_return_verification),
    ("Sentiment -> khối lượng giao dịch (co kiem soat article_count)", run_volume_regression),
    ("Khối ngoại vs khối nội (year FE, co kiem soat article_count)", run_foreign_volume_regression),
    ("Kiem nhan qua nguoc: sentiment ~ return tre", run_reverse_causality),
    ("Thực tế vs OLS dự báo", run_ols_fit_plot),
    ("Hệ số sentiment theo lag", run_sentiment_lag_plot),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for title, run in DIAGNOSTICS:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
        run(MERGED_DATA_PATHS_BY_METHOD, output_suffix=OUTPUT_SUFFIX)


if __name__ == "__main__":
    main()
