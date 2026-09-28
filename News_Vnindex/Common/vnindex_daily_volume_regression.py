"""Hồi quy dự báo KHỐI LƯỢNG GIAO DỊCH (không phải lợi suất) kiểu Tetlock
(2007) - target thứ 2 trong bài gốc bên cạnh lợi suất (xem phương trình
volume trong REF/Tetlock_Media_Sentiment_JF.pdf): pessimism cao -> khối
lượng giao dịch tăng sau đó (dấu hiệu nhà đầu tư bất đồng quan điểm/giao
dịch nhiều hơn khi có tin bi quan) - target chưa từng test trong project
này (trước giờ chỉ test lợi suất/abnormal return).

Thiết kế mirror ĐÚNG file vnindex_daily_predictive_regression.py (cùng
N_LAGS=5, cùng Newey-West, cùng dow/near_tet dummy) - chỉ đổi:
    target = log_vol_total (thay vì abnormal_return_*) - tức khối lượng
    giao dịch NGÀY t được hồi quy trên 5 lag của chính nó (tự tương quan,
    volume rất persistent) + 5 lag sentiment + dow dummy + near_tet +
    volatility_20d.

KIỂM SOÁT ATTENTION (mentor feedback, sau phát hiện volume khối nội): thêm
`log_article_count` ở ĐÚNG các lag mà sentiment_index_z dùng - sentiment
(hướng tin) và số lượng bài/ngày (attention/mức độ được chú ý) rất dễ lẫn
nhau (ngày khủng hoảng vừa nhiều bài vừa sentiment xấu cùng lúc); nếu không
kiểm soát, hệ số sentiment "có ý nghĩa" có thể chỉ phản ánh LƯỢNG tin chứ
không phải HƯỚNG tin.

Import lại is_near_tet/fit_with_lag_sum_test từ
Common/vnindex_daily_predictive_regression.py - tránh chép lại hàm tiện ích.
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


def add_volume_regression_features(
    df: pd.DataFrame, target_column: str, min_lag: int
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Giống add_regression_features() bản gốc, nhưng target = volume (tự
    hồi quy trên lag của chính nó thay vì 5 lag return + 1 lag volume).

    `target_column`:
        - "log_vol_total" (volume CÙNG NGÀY t): predictor chỉ được dùng
          lag1..lag5 (t-1..t-5) - ngày t chính là ngày đang dự báo nên không
          có "lag0" để dùng. Dummy lịch (dow/near_tet) lấy theo ngày t (ngày
          đang dự báo).
        - "future_log_vol_1d" (volume ngày t+1, = log_vol_total.shift(-1)):
          predictor dùng được CẢ lag0 (chính ngày t, vì t luôn xảy ra TRƯỚC
          t+1) đến lag4 (t-4) - tổng vẫn 5 mốc thời gian như bản cùng ngày,
          chỉ dịch tất cả tới gần "hiện tại" hơn 1 bước. Dummy lịch phải lấy
          theo ngày t+1 (ngày đang dự báo), không phải ngày t - dùng
          `.shift(-1)` trên chính cột dummy đã tính theo ngày t.
    `min_lag` = 1 (bản cùng ngày) hoặc 0 (bản tương lai).
    """
    out = df.sort_values("date", kind="mergesort").reset_index(drop=True).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["future_log_vol_1d"] = out["log_vol_total"].shift(-1)

    day_of_week = out["date"].dt.dayofweek
    dow_dummy_columns = []
    for day_index, day_name in zip([1, 2, 3, 4], ["tue", "wed", "thu", "fri"]):
        column_name = f"dow_{day_name}"
        out[column_name] = (day_of_week == day_index).astype(float)
        dow_dummy_columns.append(column_name)
    out["near_tet"] = is_near_tet(out["date"])

    if target_column == "future_log_vol_1d":
        # Dummy lich phai phan anh ngay DANG DU BAO (t+1), khong phai ngay t.
        for column_name in dow_dummy_columns + ["near_tet"]:
            out[column_name] = out[column_name].shift(-1)

    sentiment_lag_columns = []
    all_lag_columns = []
    for lag in range(min_lag, N_LAGS):
        offset = lag if lag > 0 else 0
        suffix = f"lag{lag}" if lag > 0 else "lag0"
        sentiment_lag_column = f"{SENTIMENT_COLUMN}_{suffix}"
        target_lag_column = f"log_vol_total_{suffix}"
        article_count_lag_column = f"log_article_count_{suffix}"

        out[sentiment_lag_column] = out[SENTIMENT_COLUMN].shift(offset)
        out[target_lag_column] = out["log_vol_total"].shift(offset)
        out[article_count_lag_column] = out["log_article_count"].shift(offset)

        sentiment_lag_columns.append(sentiment_lag_column)
        all_lag_columns += [sentiment_lag_column, target_lag_column, article_count_lag_column]

    predictor_columns = all_lag_columns + dow_dummy_columns + ["near_tet"] + EXTRA_EXOG_COLUMNS
    return out, sentiment_lag_columns, predictor_columns


# (target_column, min_lag): "log_vol_total" dùng lag1..lag5 (volume CÙNG
# NGÀY đang dự báo, không có lag0 khả dụng); "future_log_vol_1d" dùng
# lag0..lag4 (volume NGÀY MAI, lag0 = chính hôm nay hợp lệ vì luôn xảy ra
# trước ngày mai).
TARGET_SPECS = [("log_vol_total", 1), ("future_log_vol_1d", 0)]


def run_volume_regression(paths_by_method: dict[str, Path], output_suffix: str = "") -> None:
    """Hồi quy cho mọi method trong ``paths_by_method``. Output:
    data_News/vnindex_daily_volume_regression_coefficients{output_suffix}.csv
    data_News/vnindex_daily_volume_regression_lag_sum_test{output_suffix}.csv"""

    coef_frames = []
    lag_sum_rows = []
    for target_column, min_lag in TARGET_SPECS:
        for method_name, merged_data_path in paths_by_method.items():
            merged_df = pd.read_parquet(merged_data_path)
            featured_df, sentiment_lag_columns, predictor_columns = add_volume_regression_features(
                merged_df, target_column, min_lag
            )
            coef_df, lag_sum_result = fit_with_lag_sum_test(
                featured_df, target_column, predictor_columns, sentiment_lag_columns
            )
            coef_df.insert(0, "target", target_column)
            coef_df.insert(0, "method", method_name)
            coef_frames.append(coef_df)
            lag_sum_result["method"] = method_name
            lag_sum_result["target"] = target_column
            lag_sum_rows.append(lag_sum_result)

            print(f"\n--- {method_name} / target={target_column} ---")
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
    print("\n== Kiểm định tổng: sentiment lag đầu tiên (tức thời) vs tổng các lag còn lại ==")
    print("(immediate_lag1_coefficient = lag1 cho target log_vol_total, = lag0 cho target future_log_vol_1d)")
    print(lag_sum_df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(coef_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / f"vnindex_daily_volume_regression_coefficients{output_suffix}.csv", index=False, encoding="utf-8-sig"
    )
    lag_sum_df.to_csv(
        OUTPUT_DIR / f"vnindex_daily_volume_regression_lag_sum_test{output_suffix}.csv", index=False, encoding="utf-8-sig"
    )
    print("\nĐã lưu 2 file vào:", OUTPUT_DIR)

