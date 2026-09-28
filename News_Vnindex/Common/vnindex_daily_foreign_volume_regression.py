"""Sentiment có dự báo được volume KHỐI NGOẠI khác với KHỐI NỘI không?

Dữ liệu khối ngoại (buy_vol_foreign/sell_vol_foreign) có sẵn trong
data_Histo/vnindex_eda_output.csv nhưng trước giờ chưa được đưa vào pipeline
- đã bổ sung vào merge_vnindex_daily_with_sentiment.py (foreign_vol_total,
foreign_vol_net, domestic_vol_total).

QUAN TRỌNG - bài học rút ra từ vnindex_daily_volume_regression.py (bản
volume tổng): log_vol_total có tương quan RẤT MẠNH với thời gian (corr với
row-order = 0.93 - thị trường VN tăng trưởng volume tự nhiên qua 14 năm),
sentiment_index_z cũng trôi nhẹ theo năm (corr = 0.34) - 5 lag (5 ngày)
KHÔNG đủ hấp thụ xu hướng nhiều năm này, khiến hệ số sentiment ban đầu
"có ý nghĩa" chỉ vì cả 2 cùng tăng theo thời gian (spurious). Sau khi thêm
year fixed effect, tín hiệu ở volume tổng chỉ còn sống ở Cách 1 PMI/Cách 2
Intensity, MẤT HẲN ở PCA - không đạt chuẩn nhất quán 4 phương pháp.

-> File này BẮT BUỘC year dummy ngay từ đầu (không phải tùy chọn thêm sau),
để không lặp lại sai lầm tương tự.

Target: log_foreign_vol_total và log_domestic_vol_total (cùng ngày) - so
sánh xem sentiment dự báo khối nào rõ hơn.

KIỂM SOÁT ATTENTION (mentor feedback): thêm `log_article_count` ở đúng các
lag mà sentiment_index_z dùng - xem docstring vnindex_daily_volume_regression.py
để biết lý do (sentiment/hướng tin dễ lẫn với attention/lượng tin).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from News_Vnindex.Common.vnindex_daily_predictive_regression import (
    EXTRA_EXOG_COLUMNS,
    N_LAGS,
    OUTPUT_DIR,
    SENTIMENT_COLUMN,
    fit_with_lag_sum_test,
    is_near_tet,
)
TARGET_COLUMNS = ["log_foreign_vol_total", "log_domestic_vol_total"]


def add_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.sort_values("date", kind="mergesort").reset_index(drop=True).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = out["date"].dt.year

    day_of_week = out["date"].dt.dayofweek
    dow_dummy_columns = []
    for day_index, day_name in zip([1, 2, 3, 4], ["tue", "wed", "thu", "fri"]):
        column_name = f"dow_{day_name}"
        out[column_name] = (day_of_week == day_index).astype(float)
        dow_dummy_columns.append(column_name)
    out["near_tet"] = is_near_tet(out["date"])

    year_dummies = pd.get_dummies(out["year"], prefix="year", drop_first=True).astype(float)
    year_columns = list(year_dummies.columns)
    out = pd.concat([out, year_dummies], axis=1)

    sentiment_lag_columns = []
    all_lag_columns = []
    for lag in range(1, N_LAGS + 1):
        sentiment_lag_column = f"{SENTIMENT_COLUMN}_lag{lag}"
        target_lag_column = f"{target_column}_lag{lag}"
        article_count_lag_column = f"log_article_count_lag{lag}"

        out[sentiment_lag_column] = out[SENTIMENT_COLUMN].shift(lag)
        out[target_lag_column] = out[target_column].shift(lag)
        out[article_count_lag_column] = out["log_article_count"].shift(lag)

        sentiment_lag_columns.append(sentiment_lag_column)
        all_lag_columns += [sentiment_lag_column, target_lag_column, article_count_lag_column]

    predictor_columns = (
        all_lag_columns + dow_dummy_columns + ["near_tet"] + EXTRA_EXOG_COLUMNS + year_columns
    )
    return out, sentiment_lag_columns, predictor_columns


def run_foreign_volume_regression(paths_by_method: dict[str, Path], output_suffix: str = "") -> None:
    """Hồi quy cho mọi method trong ``paths_by_method``. Output:
    data_News/vnindex_daily_foreign_volume_regression_coefficients{output_suffix}.csv
    data_News/vnindex_daily_foreign_volume_regression_lag_sum_test{output_suffix}.csv"""

    coef_frames = []
    lag_sum_rows = []
    for target_column in TARGET_COLUMNS:
        for method_name, merged_data_path in paths_by_method.items():
            merged_df = pd.read_parquet(merged_data_path)
            featured_df, sentiment_lag_columns, predictor_columns = add_features(merged_df, target_column)
            coef_df, lag_sum_result = fit_with_lag_sum_test(
                featured_df, target_column, predictor_columns, sentiment_lag_columns
            )
            coef_df.insert(0, "target", target_column)
            coef_df.insert(0, "method", method_name)
            coef_frames.append(coef_df)
            lag_sum_result["method"] = method_name
            lag_sum_result["target"] = target_column
            lag_sum_rows.append(lag_sum_result)

            print(f"\n--- {method_name} / target={target_column} (co year FE) ---")
            sentiment_rows = coef_df.loc[coef_df["predictor_variable"].isin(sentiment_lag_columns)]
            print(sentiment_rows[["predictor_variable", "coefficient", "std_error_newey_west", "t_stat", "p_value"]].to_string(index=False))
            print(f"R^2 = {coef_df['r_squared'].iloc[0]:.5f}  (n={coef_df['observation_count'].iloc[0]})")

    lag_sum_df = pd.DataFrame(lag_sum_rows)[
        [
            "method",
            "target",
            "immediate_lag1_coefficient",
            "reversal_lag2_5_sum_coefficient",
            "reversal_lag2_5_sum_se",
            "reversal_lag2_5_sum_t",
            "reversal_lag2_5_sum_p",
            "observation_count",
        ]
    ]
    print("\n== Kiểm định tổng (đã control year FE) ==")
    print(lag_sum_df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(coef_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / f"vnindex_daily_foreign_volume_regression_coefficients{output_suffix}.csv", index=False, encoding="utf-8-sig"
    )
    lag_sum_df.to_csv(
        OUTPUT_DIR / f"vnindex_daily_foreign_volume_regression_lag_sum_test{output_suffix}.csv", index=False, encoding="utf-8-sig"
    )
    print("\nĐã lưu 2 file vào:", OUTPUT_DIR)

