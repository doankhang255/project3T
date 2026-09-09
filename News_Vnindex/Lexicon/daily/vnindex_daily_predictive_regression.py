"""Hồi quy dự báo kiểu Tetlock (2007) ở cấp NGÀY - điểm 4 feedback mentor
(kiểm định gián tiếp không cần chờ mở rộng ground truth, xem
News/Build_sentiment_label/MENTOR_FEEDBACK_PLAN.md mục C).

Dựa theo phương trình (1) trong REF/Tetlock_Media_Sentiment_JF.pdf:
    Dow_t = a + b*L5(Dow_t) + g*L5(BdNws_t) + d*L5(Vlm_t) + Exog_{t-1} + e_t
- Target = daily_return NGÀY t - dự báo bằng 5 lag (t-1..t-5) của chính nó và
  của sentiment. Volume chỉ dùng 1 lag (khác bản gốc dùng L5(Vlm) - đã kiểm
  tra bằng VIF (check_predictor_multicollinearity.py): 5 lag của
  log_vol_total cộng tuyến rất nặng (VIF 25-31, do volume tự tương quan cao
  ngày-qua-ngày), 5 lag gần như đo trùng nhau nên rút gọn còn 1 lag).
- Exog: dummy thứ-trong-tuần (Thứ 3-6, Thứ 2 làm baseline) + dummy gần Tết
  (`near_tet`, thay cho dummy tháng 1 của bản gốc - "January effect" là hiện
  tượng riêng của Mỹ, không có căn cứ cho VN-Index; hiện tượng mùa vụ đáng
  chú ý ở VN là Tết Âm lịch, không cố định vào 1 tháng dương lịch nên phải
  tra bảng ngày Tết từng năm, xem TET_DATES) + volatility_20d.
- Newey-West HAC 5 lag - hàm `newey_west_covariance` tự viết riêng cho file
  này (không import từ vnindex_weekly_predictive_regression.py), để file tự
  chứa (self-contained), không phụ thuộc thư mục cha.
- Chạy RIÊNG cho Cách 1 (PMI) và Cách 2 (intensity), input là 2 file
  `data_News/vnindex_daily_sentiment_merged_{pmi,intensity}.parquet` do
  Common/merge_vnindex_daily_with_sentiment.py xuất ra - để so sánh khách
  quan 2 cách chấm điểm.
- Ngoài hệ số từng lag, tính kiểm định tổng hệ số lag 2-5 của sentiment
  (= "hiệu ứng đảo chiều", phát hiện trung tâm của Tetlock) bằng đúng ma
  trận hiệp phương sai Newey-West, không suy từ p-value từng lag riêng lẻ.

Target = ABNORMAL RETURN (build_vnindex_daily_abnormal_return.py), KHÔNG
phải raw daily_return - trừ đi phần lợi suất "dự đoán được" từ chính lịch sử
của nó (rolling mean / AR(1)) trước khi hồi quy với sentiment, để sentiment
không "ăn ké" vào phần vốn đã dự đoán được từ giá quá khứ. Chạy CẢ 2 biến
thể abnormal return (rolling và AR(1)) cho mỗi method.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MERGED_DATA_PATHS_BY_METHOD = {
    "Cach1_PMI": PROJECT_ROOT / "data_News" / "vnindex_daily_sentiment_abnormal_return_pmi.parquet",
    "Cach2_Intensity": PROJECT_ROOT / "data_News" / "vnindex_daily_sentiment_abnormal_return_intensity.parquet",
    "PCA_PMI": PROJECT_ROOT / "data_News" / "vnindex_daily_sentiment_abnormal_return_pca_pmi.parquet",
    "PCA_Intensity": PROJECT_ROOT / "data_News" / "vnindex_daily_sentiment_abnormal_return_pca_intensity.parquet",
}
OUTPUT_DIR = PROJECT_ROOT / "data_News"

N_LAGS = 5
TARGET_COLUMNS = ["abnormal_return_rolling_1d", "abnormal_return_ar1_1d"]
SENTIMENT_COLUMN = "sentiment_index_z"
EXTRA_EXOG_COLUMNS = ["volatility_20d"]

# Ngày mùng 1 Tết Âm lịch (dương lịch) - thay cho month_january của bản gốc
# Tetlock (US "January effect", không có căn cứ cho VN-Index). Tết không cố
# định vào 1 tháng dương lịch (có năm rơi tháng 1, có năm tháng 2) nên cần
# tra bảng theo từng năm, không dùng dummy theo tháng được. Ngày tự tổng hợp
# thủ công (không phải tính lịch âm tự động) - kiểm tra lại nếu mở rộng dữ
# liệu quá năm 2025.
TET_DATES = pd.to_datetime(
    [
        "2010-02-14", "2011-02-03", "2012-01-23", "2013-02-10", "2014-01-31",
        "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
        "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
        "2025-01-29",
    ]
)
NEAR_TET_WINDOW_DAYS = 10  # +-10 ngày dương lịch quanh mùng 1 Tết


def is_near_tet(dates: pd.Series, window_days: int = NEAR_TET_WINDOW_DAYS) -> pd.Series:
    """Dummy = 1 nếu ngày nằm trong `window_days` ngày dương lịch quanh mùng 1
    Tết GẦN NHẤT (trước hoặc sau) - dùng searchsorted tìm vị trí chèn vào
    TET_DATES đã sắp xếp, so khoảng cách tới 2 mốc Tết liền kề (trước/sau),
    lấy khoảng cách nhỏ hơn."""
    date_values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    tet_values = TET_DATES.to_numpy(dtype="datetime64[ns]")

    insert_positions = np.searchsorted(tet_values, date_values, side="left")
    # Giá trị khởi tạo lớn (10 năm - dư sức lớn hơn khoảng cách thực tế tới
    # Tết) nhưng KHÔNG được quá lớn: timedelta64[ns] dùng int64, quy đổi ra ns
    # chỉ chịu được ~292 năm - dùng số quá lớn (từng thử 10**9 ngày) sẽ TRÀN
    # SỐ (overflow) ra giá trị âm vô nghĩa (đã gặp lỗi này khi debug).
    distances = np.full(len(date_values), np.timedelta64(3650, "D")).astype("timedelta64[ns]")

    for offset in (-1, 0):
        neighbor_positions = insert_positions + offset
        valid = (neighbor_positions >= 0) & (neighbor_positions < len(tet_values))
        neighbor_dates = np.where(
            valid, tet_values[np.clip(neighbor_positions, 0, len(tet_values) - 1)], np.datetime64("NaT")
        )
        candidate_distances = np.abs(date_values - neighbor_dates)
        distances = np.where(valid & (candidate_distances < distances), candidate_distances, distances)

    within_window = distances <= np.timedelta64(window_days, "D")
    return pd.Series(within_window.astype(float), index=dates.index)


def add_regression_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Thêm 5 lag của sentiment/target + dummy thứ-trong-tuần/gần Tết.
    Trả về (df đã thêm cột, tên cột lag sentiment (để test tổng lag 2-5),
    toàn bộ tên cột predictor)."""
    out = df.sort_values("date", kind="mergesort").reset_index(drop=True).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    day_of_week = out["date"].dt.dayofweek
    dow_dummy_columns = []
    for day_index, day_name in zip([1, 2, 3, 4], ["tue", "wed", "thu", "fri"]):
        column_name = f"dow_{day_name}"
        out[column_name] = (day_of_week == day_index).astype(float)
        dow_dummy_columns.append(column_name)
    out["near_tet"] = is_near_tet(out["date"])

    out["log_vol_total_lag1"] = out["log_vol_total"].shift(1)

    sentiment_lag_columns = []
    all_lag_columns = []
    for lag in range(1, N_LAGS + 1):
        sentiment_lag_column = f"{SENTIMENT_COLUMN}_lag{lag}"
        target_lag_column = f"{target_column}_lag{lag}"

        out[sentiment_lag_column] = out[SENTIMENT_COLUMN].shift(lag)
        out[target_lag_column] = out[target_column].shift(lag)

        sentiment_lag_columns.append(sentiment_lag_column)
        all_lag_columns += [sentiment_lag_column, target_lag_column]

    predictor_columns = (
        all_lag_columns + ["log_vol_total_lag1"] + dow_dummy_columns + ["near_tet"] + EXTRA_EXOG_COLUMNS
    )
    return out, sentiment_lag_columns, predictor_columns


def newey_west_covariance(x: np.ndarray, residuals: np.ndarray, max_lag: int) -> np.ndarray:
    """Ma trận hiệp phương sai Newey-West HAC (Heteroskedasticity- and
    Autocorrelation-Consistent) cho hệ số OLS - hàm tự viết riêng cho file
    này."""
    n_obs = x.shape[0]
    x_residual = x * residuals[:, None]
    s_matrix = x_residual.T @ x_residual

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = x_residual[lag:].T @ x_residual[:-lag]
        s_matrix = s_matrix + weight * (gamma + gamma.T)

    x_tx_inverse = np.linalg.pinv(x.T @ x)
    covariance = x_tx_inverse @ s_matrix @ x_tx_inverse
    return covariance * n_obs / max(n_obs - x.shape[1], 1)


def fit_with_lag_sum_test(
    df: pd.DataFrame,
    target_column: str,
    predictor_columns: list[str],
    sentiment_lag_columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Hồi quy OLS + Newey-West HAC, trả về (bảng hệ số từng biến, kết quả
    kiểm định tổng hệ số lag 2-5 của sentiment) - tự tính SE cho tổ hợp
    tuyến tính bằng ma trận hiệp phương sai đầy đủ, không suy từ p-value
    từng hệ số riêng lẻ."""
    model_df = df[[target_column] + predictor_columns].dropna().copy()

    y = model_df[target_column].to_numpy(dtype="float64")
    x_without_constant = model_df[predictor_columns].to_numpy(dtype="float64")
    x = np.column_stack([np.ones(len(model_df)), x_without_constant])
    variable_names = ["const"] + predictor_columns

    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    n_obs = len(model_df)
    n_params = x.shape[1]

    covariance = newey_west_covariance(x, residuals, max_lag=N_LAGS)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_stats = coefficients / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=max(n_obs - n_params, 1))

    rss = float(np.sum(residuals**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - rss / tss if tss != 0 else np.nan

    coef_df = pd.DataFrame(
        {
            "predictor_variable": variable_names,
            "coefficient": coefficients,
            "std_error_newey_west": standard_errors,
            "t_stat": t_stats,
            "p_value": p_values,
            "r_squared": r_squared,
            "observation_count": n_obs,
        }
    )

    # Kiểm định tổng hệ số lag 2-5 của sentiment (bỏ lag 1 = hiệu ứng tức
    # thời) bằng ma trận hiệp phương sai đầy đủ - đúng cách làm thống kê cho
    # tổ hợp tuyến tính, không phải cộng dồn p-value từng lag.
    reversal_lag_columns = sentiment_lag_columns[1:]
    weight_vector = np.zeros(len(variable_names))
    for column in reversal_lag_columns:
        weight_vector[variable_names.index(column)] = 1.0

    sum_coefficient = float(weight_vector @ coefficients)
    sum_variance = float(weight_vector @ covariance @ weight_vector)
    sum_se = float(np.sqrt(max(sum_variance, 0)))
    sum_t = sum_coefficient / sum_se if sum_se > 0 else np.nan
    sum_p = (
        2.0 * stats.t.sf(np.abs(sum_t), df=max(n_obs - n_params, 1)) if not np.isnan(sum_t) else np.nan
    )

    lag_sum_result = {
        "immediate_lag1_coefficient": float(coefficients[variable_names.index(sentiment_lag_columns[0])]),
        "reversal_lag2_5_sum_coefficient": sum_coefficient,
        "reversal_lag2_5_sum_se": sum_se,
        "reversal_lag2_5_sum_t": sum_t,
        "reversal_lag2_5_sum_p": sum_p,
        "observation_count": n_obs,
    }
    return coef_df, lag_sum_result


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
            coef_df, lag_sum_result = fit_with_lag_sum_test(
                featured_df, target_column, predictor_columns, sentiment_lag_columns
            )

            coef_df.insert(0, "target", target_column)
            coef_df.insert(0, "method", method_name)
            coef_frames.append(coef_df)
            lag_sum_rows.append({"method": method_name, "target": target_column, **lag_sum_result})

    coef_result_df = pd.concat(coef_frames, ignore_index=True)
    lag_sum_df = pd.DataFrame(lag_sum_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coef_result_df.to_csv(OUTPUT_DIR / "vnindex_daily_predictive_regression.csv", index=False, encoding="utf-8-sig")
    lag_sum_df.to_csv(OUTPUT_DIR / "vnindex_daily_regression_lag_sum_test.csv", index=False, encoding="utf-8-sig")

    print("== 5 lag của sentiment (từng method x target) - xem pattern giảm-rồi-đảo-chiều ==")
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
    print("== Kiểm định tổng: lag1 (tức thời) vs tổng lag2-5 (đảo chiều) ==")
    print(lag_sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
