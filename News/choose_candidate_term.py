from __future__ import annotations

from pathlib import Path
import math
import re
import sys

import pandas as pd

try:
    from News.build_CSR_matrix import INPUT_PATH
    from News.build_CSR_matrix import TOKENIZED_COLUMN
    from News.build_CSR_matrix import build_term_document_matrix
    from News.build_CSR_matrix import build_tf_df_dataframe
except ImportError:
    from build_CSR_matrix import INPUT_PATH
    from build_CSR_matrix import TOKENIZED_COLUMN
    from build_CSR_matrix import build_term_document_matrix
    from build_CSR_matrix import build_tf_df_dataframe


OUTPUT_PATH = Path(__file__).with_name("candidate_terms.csv")
LETTER_PATTERN = re.compile(r"[^\W\d_]", flags=re.UNICODE)

VIETNAMESE_STOPWORDS = {
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


def add_term_statistics(
    tf_df: pd.DataFrame,
    total_documents: int,
) -> pd.DataFrame:
    out = tf_df.copy()
    out["df_ratio"] = out["df"] / total_documents
    out["avg_tf_per_doc"] = out["tf"] / out["df"]
    out["candidate_score"] = out["tf"] * out["df"].apply(
        lambda df: math.log((total_documents + 1) / (df + 1))
    )
    return out


def is_clean_candidate_term(term: str) -> bool:
    if not isinstance(term, str):
        return False
    if len(term) <= 1:
        return False
    if term in VIETNAMESE_STOPWORDS:
        return False
    if any(character.isdigit() for character in term):
        return False
    if not LETTER_PATTERN.search(term):
        return False
    return True


def choose_candidate_terms(
    tf_df: pd.DataFrame,
    total_documents: int,
    min_tf: int = 10,
    min_df: int = 5,
    max_df_ratio: float = 0.50,
) -> pd.DataFrame:
    term_stats = add_term_statistics(tf_df, total_documents)
    candidate_mask = (
        term_stats["tf"].ge(min_tf)
        & term_stats["df"].ge(min_df)
        & term_stats["df_ratio"].le(max_df_ratio)
        & term_stats["term"].apply(is_clean_candidate_term)
    )
    candidate_terms = term_stats.loc[candidate_mask].copy()
    candidate_terms = candidate_terms.sort_values(
        by=["candidate_score", "tf", "df", "term"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    candidate_terms = candidate_terms.reset_index(drop=True)
    return candidate_terms


def build_candidate_terms(
    path: Path = INPUT_PATH,
    text_column: str = TOKENIZED_COLUMN,
    vectorizer_min_df: int = 2,
    min_tf: int = 10,
    min_df: int = 5,
    max_df_ratio: float = 0.50,
) -> pd.DataFrame:
    term_document_matrix, terms, _ = build_term_document_matrix(
        path=path,
        text_column=text_column,
        min_df=vectorizer_min_df,
    )
    tf_df = build_tf_df_dataframe(term_document_matrix, terms)
    return choose_candidate_terms(
        tf_df=tf_df,
        total_documents=term_document_matrix.shape[0],
        min_tf=min_tf,
        min_df=min_df,
        max_df_ratio=max_df_ratio,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    candidate_terms = build_candidate_terms()
    candidate_terms.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved candidate terms to: {OUTPUT_PATH}")
    print("Candidate terms:", len(candidate_terms))
    print(candidate_terms.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
