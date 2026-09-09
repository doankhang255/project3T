"""BẢN NÂNG CẤP multi-lag cho weekly - mirror đúng
News_Vnindex/Lexicon/daily/vnindex_daily_predictive_regression.py, thay
vnindex_weekly_predictive_regression.py (bản đơn giản 1-lag, dùng target =
future_ret_Nw) đang có sẵn ở cùng thư mục.

Theo đúng phương trình (1) trong REF/Tetlock_Media_Sentiment_JF.pdf:
    Dow_t = a + b*L5(Dow_t) + g*L5(BdNws_t) + d*L5(Vlm_t) + Exog_{t-1} + e_t
- Target = abnormal_return_{rolling,ar1}_1w TUẦN t (không phải future_ret) -
  dự báo bằng 5 lag (t-1..t-5) của chính nó và của sentiment. Volume chỉ 1
  lag (giống daily - đã kiểm tra VIF ở daily thấy 5 lag volume cộng tuyến
  nặng do tự tương quan cao, áp dụng luôn quy tắc đó ở đây không cần kiểm
  tra lại).
- Exog: dummy gần Tết (near_tet, theo tuần) + volatility_12w (đã sửa
  .shift(1) ở build_vnindex_weekly_return.py - tránh look-ahead).
- Newey-West HAC 5 lag - IMPORT lại `newey_west_covariance` từ
  vnindex_weekly_predictive_regression.py (bản đơn giản, CÙNG THƯ MỤC weekly/)
  - không viết lại, khác với daily/ (không import xuyên thư mục
  daily<->weekly theo yêu cầu trước đó, nhưng trong CÙNG thư mục weekly/ thì
  vẫn dùng lại được).
- Kiểm định tổng hệ số lag2-5 (hiệu ứng đảo chiều) bằng ma trận hiệp phương
  sai đầy đủ, giống hệt daily.
- Chạy cho cả 4 method: PMI, Intensity, PCA_PMI, PCA_Intensity.

Output:
    data_News/vnindex_weekly_predictive_regression_multilag.csv
    data_News/vnindex_weekly_regression_lag_sum_test_multilag.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vnindex_weekly_predictive_regression import newey_west_covariance  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MERGED_DATA_PATHS_BY_METHOD = {
    "Cach1_PMI": PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_abnormal_return_pmi.parquet",
    "Cach2_Intensity": PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_abnormal_return_intensity.parquet",
    "PCA_PMI": PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_abnormal_return_pca_pmi.parquet",
    "PCA_Intensity": PROJECT_ROOT / "data_News" / "vnindex_weekly_sentiment_abnormal_return_pca_intensity.parquet",
}
OUTPUT_DIR = PROJECT_ROOT / "data_News"

N_LAGS = 5
TARGET_COLUMNS = ["abnormal_return_rolling_1w", "abnormal_return_ar1_1w"]
SENTIMENT_COLUMN = "sentiment_index_z"
EXTRA_EXOG_COLUMNS = ["volatility_12w"]

# Cùng bảng ngày Tết đã dùng ở daily (News_Vnindex/Lexicon/daily/vnindex_daily_predictive_regression.py)
TET_DATES = pd.to_datetime(
    [
        "2010-02-14", "2011-02-03", "2012-01-23", "2013-02-10", "2014-01-31",
        "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
        "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
        "2025-01-29",
    ]
)
NEAR_TET_WINDOW_DAYS = 10


def is_near_tet(dates: pd.Series, window_days: int = NEAR_TET_WINDOW_DAYS) -> pd.Series:
    """Giống hệt logic ở daily (đã kiểm chứng đúng), áp cho week_end - dummy
    = 1 nếu week_end nằm trong window_days ngày quanh mùng 1 Tết gần nhất."""
    date_values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    tet_values = TET_DATES.to_numpy(dtype="datetime64[ns]")

    insert_positions = np.searchsorted(tet_values, date_values, side="left")
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
    out = df.sort_values("week_end", kind="mergesort").reset_index(drop=True).copy()
    out["week_end"] = pd.to_datetime(out["week_end"], errors="coerce")

    out["near_tet"] = is_near_tet(out["week_end"])
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

    predictor_columns = all_lag_columns + ["log_vol_total_lag1", "near_tet"] + EXTRA_EXOG_COLUMNS
    return out, sentiment_lag_columns, predictor_columns


def fit_with_lag_sum_test(
    df: pd.DataFrame, target_column: str, predictor_columns: list[str], sentiment_lag_columns: list[str]
) -> tuple[pd.DataFrame, dict]:
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

    reversal_lag_columns = sentiment_lag_columns[1:]
    weight_vector = np.zeros(len(variable_names))
    for column in reversal_lag_columns:
        weight_vector[variable_names.index(column)] = 1.0

    sum_coefficient = float(weight_vector @ coefficients)
    sum_variance = float(weight_vector @ covariance @ weight_vector)
    sum_se = float(np.sqrt(max(sum_variance, 0)))
    sum_t = sum_coefficient / sum_se if sum_se > 0 else np.nan
    sum_p = 2.0 * stats.t.sf(np.abs(sum_t), df=max(n_obs - n_params, 1)) if not np.isnan(sum_t) else np.nan

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
    coef_result_df.to_csv(
        OUTPUT_DIR / "vnindex_weekly_predictive_regression_multilag.csv", index=False, encoding="utf-8-sig"
    )
    lag_sum_df.to_csv(
        OUTPUT_DIR / "vnindex_weekly_regression_lag_sum_test_multilag.csv", index=False, encoding="utf-8-sig"
    )

    print("== 5 lag của sentiment (từng method x target) ==")
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
