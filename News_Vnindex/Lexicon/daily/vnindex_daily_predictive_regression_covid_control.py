"""Robustness check RIÊNG (KHÔNG sửa vnindex_daily_predictive_regression.py
gốc): thêm dummy giai đoạn khủng hoảng COVID-19 vào predictor - đúng ý tưởng
Tetlock thêm dummy sập thị trường 19/10/1987 (Black Monday) để đảm bảo kết
quả không bị 1 giai đoạn biến động cực đoan duy nhất chi phối.

Import lại các hàm ĐÃ CÓ từ file gốc (add_regression_features,
fit_with_lag_sum_test, newey_west_covariance, MERGED_DATA_PATHS_BY_METHOD,
TARGET_COLUMNS) - không viết lại logic hồi quy/lag/dummy đã có, chỉ thêm 1
cột `covid_crash` vào predictor rồi fit lại.

Giai đoạn COVID chọn (thủ công, có thể điều chỉnh):
    2020-01-23 -> 2020-04-30
    - 2020-01-23: Việt Nam công bố ca nhiễm COVID-19 đầu tiên.
    - 2020-04-30: ~3 tháng, phủ hết giai đoạn giảm sâu + hồi phục còn biến
      động mạnh. VN-Index giảm từ ~991 điểm (23/1/2020, ngay trước khi đóng
      cửa nghỉ Tết) xuống đáy ~662 điểm (24/3/2020) - giảm ~33% - rồi hồi
      phục dần nhưng vẫn biến động mạnh hết tháng 4.

Output (tên khác file gốc, không ghi đè):
    data_News/vnindex_daily_predictive_regression_covid_control.csv
    data_News/vnindex_daily_regression_lag_sum_test_covid_control.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnindex_daily_predictive_regression import (  # noqa: E402
    MERGED_DATA_PATHS_BY_METHOD,
    SENTIMENT_COLUMN,
    TARGET_COLUMNS,
    add_regression_features,
    fit_with_lag_sum_test,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data_News"

COVID_START = pd.Timestamp("2020-01-23")
COVID_END = pd.Timestamp("2020-04-30")


def add_covid_dummy(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    return ((dates >= COVID_START) & (dates <= COVID_END)).astype(float)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    coef_frames = []
    lag_sum_rows = []
    for method_name, merged_data_path in MERGED_DATA_PATHS_BY_METHOD.items():
        merged_df = pd.read_parquet(merged_data_path)
        for target_column in TARGET_COLUMNS:
            featured_df, sentiment_lag_columns, predictor_columns = add_regression_features(
                merged_df, target_column
            )
            featured_df["covid_crash"] = add_covid_dummy(featured_df["date"])
            predictor_columns_with_covid = predictor_columns + ["covid_crash"]

            coef_df, lag_sum_result = fit_with_lag_sum_test(
                featured_df, target_column, predictor_columns_with_covid, sentiment_lag_columns
            )
            coef_df.insert(0, "target", target_column)
            coef_df.insert(0, "method", method_name)
            coef_frames.append(coef_df)
            lag_sum_rows.append({"method": method_name, "target": target_column, **lag_sum_result})

    coef_result_df = pd.concat(coef_frames, ignore_index=True)
    lag_sum_df = pd.DataFrame(lag_sum_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coef_result_df.to_csv(
        OUTPUT_DIR / "vnindex_daily_predictive_regression_covid_control.csv", index=False, encoding="utf-8-sig"
    )
    lag_sum_df.to_csv(
        OUTPUT_DIR / "vnindex_daily_regression_lag_sum_test_covid_control.csv", index=False, encoding="utf-8-sig"
    )

    n_covid_days = int(
        add_covid_dummy(pd.read_parquet(next(iter(MERGED_DATA_PATHS_BY_METHOD.values())))["date"]).sum()
    )
    print(f"Số ngày rơi vào giai đoạn COVID ({COVID_START.date()} -> {COVID_END.date()}): {n_covid_days}")
    print()

    print("== Hệ số covid_crash (có ý nghĩa lớn/lệch không?) ==")
    covid_rows = coef_result_df.loc[coef_result_df["predictor_variable"] == "covid_crash"]
    print(
        covid_rows[["method", "target", "coefficient", "std_error_newey_west", "t_stat", "p_value"]].to_string(
            index=False
        )
    )

    print()
    print("== 5 lag của sentiment SAU KHI kiểm soát COVID (so với bản gốc không có covid_crash) ==")
    for method_name in MERGED_DATA_PATHS_BY_METHOD:
        for target_column in TARGET_COLUMNS:
            sub = coef_result_df.loc[
                (coef_result_df["method"] == method_name)
                & (coef_result_df["target"] == target_column)
                & coef_result_df["predictor_variable"].str.startswith(f"{SENTIMENT_COLUMN}_lag")
            ]
            print(f"\n--- {method_name} / {target_column} ---")
            print(
                sub[["predictor_variable", "coefficient", "std_error_newey_west", "t_stat", "p_value"]].to_string(
                    index=False
                )
            )

    print()
    print("== Kiểm định tổng lag2-5 SAU KHI kiểm soát COVID ==")
    print(lag_sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
