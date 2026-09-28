"""Kiem nhan qua nguoc (mentor feedback, sau phat hien tin hieu volume khoi
noi): hoi quy sentiment_index_z(t) tren cac lag cua LOI SUAT THUC (daily_return,
khong phai abnormal return) - de kiem tra tin dang PHAN ANH thi truong (return
qua khu du bao duoc sentiment hom nay) hay DAN DAT no (khong co quan he nay).

Neu return_lag1 (hom qua) du bao duoc sentiment hom nay mot cach co y nghia,
do la bang chung bao chi dang viet lai nhung gi gia da lam (phan anh), lam
yeu di cach dien giai "attention/sentiment dan dat giao dich khoi noi" - thay
vao do, sentiment co the chi la PHAN UNG voi bien dong gia da xay ra.

Dac ta mirror vnindex_daily_predictive_regression.py (Newey-West HAC 5 lag,
kiem dinh tong lag2-5) nhung DAO NGUOC vai tro: target = sentiment_index_z,
predictor chinh = 5 lag cua daily_return (loi suat THUC, vi cau hoi la "gia
da di chuyen chua" - khong phai phan loi suat bat thuong sau khi tru ky
vong), kiem soat them 5 lag cua chinh sentiment (tu tuong quan) + dow/near_tet
+ volatility_20d + log_vol_total_lag1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from News_Vnindex.Common.vnindex_daily_predictive_regression import (
    N_LAGS,
    OUTPUT_DIR,
    SENTIMENT_COLUMN,
    is_near_tet,
    newey_west_covariance,
)

RETURN_COLUMN = "daily_return"
EXTRA_EXOG_COLUMNS = ["volatility_20d"]


def add_reverse_causality_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
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

    return_lag_columns = []
    sentiment_lag_columns = []
    for lag in range(1, N_LAGS + 1):
        return_lag_column = f"{RETURN_COLUMN}_lag{lag}"
        sentiment_lag_column = f"{SENTIMENT_COLUMN}_lag{lag}"
        out[return_lag_column] = out[RETURN_COLUMN].shift(lag)
        out[sentiment_lag_column] = out[SENTIMENT_COLUMN].shift(lag)
        return_lag_columns.append(return_lag_column)
        sentiment_lag_columns.append(sentiment_lag_column)

    predictor_columns = (
        return_lag_columns
        + sentiment_lag_columns
        + ["log_vol_total_lag1"]
        + dow_dummy_columns
        + ["near_tet"]
        + EXTRA_EXOG_COLUMNS
    )
    return out, return_lag_columns, predictor_columns


def fit_reverse_causality(
    df: pd.DataFrame, predictor_columns: list[str], return_lag_columns: list[str]
) -> tuple[pd.DataFrame, dict]:
    model_df = df[[SENTIMENT_COLUMN] + predictor_columns].dropna().copy()
    y = model_df[SENTIMENT_COLUMN].to_numpy(dtype="float64")
    x_without_const = model_df[predictor_columns].to_numpy(dtype="float64")
    x = np.column_stack([np.ones(len(model_df)), x_without_const])
    variable_names = ["const"] + predictor_columns

    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    n_obs = len(model_df)
    n_params = x.shape[1]

    covariance = newey_west_covariance(x, residuals, max_lag=N_LAGS)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_stats = coefficients / standard_errors
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=max(n_obs - n_params, 1))

    coef_df = pd.DataFrame(
        {
            "predictor_variable": variable_names,
            "coefficient": coefficients,
            "std_error_newey_west": standard_errors,
            "t_stat": t_stats,
            "p_value": p_values,
            "observation_count": n_obs,
        }
    )

    reversal_columns = return_lag_columns[1:]
    weight_vector = np.zeros(len(variable_names))
    for column in reversal_columns:
        weight_vector[variable_names.index(column)] = 1.0
    sum_coefficient = float(weight_vector @ coefficients)
    sum_variance = float(weight_vector @ covariance @ weight_vector)
    sum_se = float(np.sqrt(max(sum_variance, 0)))
    sum_t = sum_coefficient / sum_se if sum_se > 0 else np.nan
    sum_p = (
        2.0 * stats.t.sf(np.abs(sum_t), df=max(n_obs - n_params, 1)) if not np.isnan(sum_t) else np.nan
    )

    lag1_column = return_lag_columns[0]
    summary = {
        "return_lag1_coefficient": float(coefficients[variable_names.index(lag1_column)]),
        "return_lag1_p": float(p_values[variable_names.index(lag1_column)]),
        "return_lag2_5_sum_coefficient": sum_coefficient,
        "return_lag2_5_sum_se": sum_se,
        "return_lag2_5_sum_p": sum_p,
        "observation_count": n_obs,
    }
    return coef_df, summary


def run_reverse_causality(paths_by_method: dict[str, Path], output_suffix: str = "") -> None:
    """``{method_suffix: merged/abnormal-return parquet}`` -> hoi quy
    sentiment_index_z(t) tren lag cua daily_return. Ghi:
    data_News/vnindex_daily_sentiment_reverse_causality_coefficients{output_suffix}.csv
    data_News/vnindex_daily_sentiment_reverse_causality_summary{output_suffix}.csv"""
    coef_frames = []
    summary_rows = []
    for method_name, merged_data_path in paths_by_method.items():
        merged_df = pd.read_parquet(merged_data_path)
        featured_df, return_lag_columns, predictor_columns = add_reverse_causality_features(merged_df)
        coef_df, summary = fit_reverse_causality(featured_df, predictor_columns, return_lag_columns)
        coef_df.insert(0, "method", method_name)
        coef_frames.append(coef_df)
        summary_rows.append({"method": method_name, **summary})

        print(f"\n--- {method_name} (sentiment(t) ~ daily_return lag - kiem nhan qua nguoc) ---")
        lag_rows = coef_df.loc[coef_df["predictor_variable"].str.startswith(f"{RETURN_COLUMN}_lag")]
        print(
            lag_rows[["predictor_variable", "coefficient", "std_error_newey_west", "t_stat", "p_value"]]
            .to_string(index=False)
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\n== Kiem dinh tong: return_lag1 (tuc thoi, gia hom qua) vs tong return_lag2-5 (tre hon) ==")
    print(summary_df.to_string(index=False))
    print(
        "\n(return_lag1 co y nghia -> bao chi co the dang PHAN ANH gia da di chuyen, "
        "khong ung ho cach dien giai sentiment DAN DAT giao dich khoi noi)"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(coef_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / f"vnindex_daily_sentiment_reverse_causality_coefficients{output_suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        OUTPUT_DIR / f"vnindex_daily_sentiment_reverse_causality_summary{output_suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nDa luu 2 file vao:", OUTPUT_DIR)
