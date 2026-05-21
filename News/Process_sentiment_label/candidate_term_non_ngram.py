from __future__ import annotations
from pathlib import Path
import math
import re
import sys

import pandas as pd

try:
    from News.Process_sentiment_label.build_CSR_matrix import INPUT_PATH
    from News.Process_sentiment_label.build_CSR_matrix import TOKENIZED_COLUMN
    from News.Process_sentiment_label.build_CSR_matrix import build_term_document_matrix
    from News.Process_sentiment_label.build_CSR_matrix import build_tf_df_dataframe
except ImportError:
    from News.Process_sentiment_label.build_CSR_matrix import INPUT_PATH
    from News.Process_sentiment_label.build_CSR_matrix import TOKENIZED_COLUMN
    from News.Process_sentiment_label.build_CSR_matrix import build_term_document_matrix
    from News.Process_sentiment_label.build_CSR_matrix import build_tf_df_dataframe


LETTER_PATTERN = re.compile(r"[^\W\d_]", flags=re.UNICODE)

VIETNAMESE_STOPWORDS = {
    "ra",
    "ai",
    "anh",
    "bà",
    "bài",
    "bạn",
    "bằng",
    "bị",
    "bộ",
    "bởi",
    "cả",
    "các",
    "cái",
    "cần",
    "càng",
    "chỉ",
    "cho",
    "chưa",
    "có",
    "còn",
    "cùng",
    "của",
    "cũng",
    "đã",
    "đang",
    "đây",
    "đến",
    "để",
    "đều",
    "điều",
    "đó",
    "được",
    "gì",
    "hơn",
    "khi",
    "không",
    "là",
    "lại",
    "lên",
    "mà",
    "một",
    "một_số",
    "này",
    "năm",
    "nếu",
    "ngày",
    "nhiều",
    "những",
    "sẽ",
    "sự",
    "số",
    "sở",
    "sau",
    "theo",
    "thì",
    "thông_qua",
    "tại",
    "từ",
    "trong",
    "trên",
    "trước",
    "và",
    "vào",
    "về",
    "vì",
    "việc",
    "với",
}


def add_term_statistics(tf_df: pd.DataFrame, total_documents: int) -> pd.DataFrame:
    out = tf_df.copy()
    out["df_ratio"] = out["df"] / total_documents
    out["avg_tf_per_doc"] = out["tf"] / out["df"]
    out["candidate_score"] = out["tf"] * out["df"].apply(
        lambda df: math.log((total_documents + 1) / (df + 1))
    )
    return out


def check_term_valid(term: str) -> str | None:
    if not isinstance(term, str):
        return "invalid_shape"
    if len(term) <= 1:
        return "invalid_shape"
    if not LETTER_PATTERN.search(term):
        return "invalid_shape"
    if any(character.isdigit() for character in term):
        return "has_digit"
    if term in VIETNAMESE_STOPWORDS:
        return "stopword"
    return None


def choose_candidate_terms(
    tf_df: pd.DataFrame,
    total_documents: int,
    min_df_ratio: float = 0.03,
    max_df_ratio: float = 0.50,
    return_groups: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    term_stats = add_term_statistics(tf_df, total_documents)
    term_stats["rejection_reason"] = term_stats["term"].apply(check_term_valid)

    digit_mask = term_stats["rejection_reason"].eq("has_digit")
    no_digit_df, terms_with_digits_df = split_by_mask(term_stats, digit_mask)

    invalid_shape_mask = no_digit_df["rejection_reason"].eq("invalid_shape")
    valid_shape_df, invalid_shape_df = split_by_mask(no_digit_df, invalid_shape_mask)

    stopword_mask = valid_shape_df["rejection_reason"].eq("stopword")
    non_stopword_df, stopword_terms_df = split_by_mask(valid_shape_df, stopword_mask)

    below_min_df_ratio_mask = non_stopword_df["df_ratio"].lt(min_df_ratio)
    df_ratio_pass_df, below_min_df_ratio_df = split_by_mask(
        non_stopword_df,
        below_min_df_ratio_mask,
    )

    above_max_df_ratio_mask = df_ratio_pass_df["df_ratio"].gt(max_df_ratio)
    candidate_terms_df, above_max_df_ratio_df = split_by_mask(
        df_ratio_pass_df,
        above_max_df_ratio_mask,
    )

    candidate_terms_df = candidate_terms_df.drop(columns=["rejection_reason"])
    
    candidate_terms_df = candidate_terms_df.sort_values(
        by=["candidate_score", "tf", "df", "term"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    candidate_terms_df = candidate_terms_df.reset_index(drop=True)

    if not return_groups:
        return candidate_terms_df

    return {
        "all_terms_df": term_stats.reset_index(drop=True),
        "terms_with_digits_df": terms_with_digits_df,
        "invalid_shape_df": invalid_shape_df,
        "stopword_terms_df": stopword_terms_df,
        "below_min_df_ratio_df": below_min_df_ratio_df,
        "above_max_df_ratio_df": above_max_df_ratio_df,
        "candidate_terms_df": candidate_terms_df,
    }


def split_by_mask(df: pd.DataFrame, mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    failed_df = df.loc[mask].copy().reset_index(drop=True)
    passed_df = df.loc[~mask].copy().reset_index(drop=True)
    return passed_df, failed_df


def build_candidate_terms(
    path: Path = INPUT_PATH,
    text_column: str = TOKENIZED_COLUMN,
    vectorizer_min_df: int = 2,
    min_df_ratio: float = 0.03,
    max_df_ratio: float = 0.50,
) -> dict[str, pd.DataFrame]:
    term_document_matrix, terms, _ = build_term_document_matrix(path=path, text_column=text_column, min_df=vectorizer_min_df)
    total_documents = term_document_matrix.shape[0]
    tf_df = build_tf_df_dataframe(term_document_matrix, terms)
    return choose_candidate_terms(
        tf_df=tf_df,
        total_documents=total_documents,
        min_df_ratio=min_df_ratio,
        max_df_ratio=max_df_ratio,
        return_groups=True,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    result = build_candidate_terms()
    candidate_terms_df = result["candidate_terms_df"]

    print("Terms removed because they contain digits:", len(result["terms_with_digits_df"]))
    print("Terms removed because shape is invalid:", len(result["invalid_shape_df"]))
    print("Terms removed because they are stopwords:", len(result["stopword_terms_df"]))
    print("Terms removed because df_ratio < min_df_ratio:", len(result["below_min_df_ratio_df"]))
    print("Terms removed because df_ratio > max_df_ratio:", len(result["above_max_df_ratio_df"]))
    print("Candidate terms:", len(candidate_terms_df))
    print(candidate_terms_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
