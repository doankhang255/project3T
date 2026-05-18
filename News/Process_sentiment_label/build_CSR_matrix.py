from pathlib import Path
import sys
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


INPUT_PATH = Path("data/equity_news_tokenized.parquet")
TOKENIZED_COLUMN = "Tokenize_content"

#csr_matrix: sparse matrix in Compressed Sparse Row format, which is an efficient way to store and manipulate large sparse matrices.
def build_term_document_matrix(path: Path = INPUT_PATH, text_column: str = TOKENIZED_COLUMN, min_df: int = 2,) -> tuple[csr_matrix, list[str], CountVectorizer]:
    df = pd.read_parquet(path, columns=[text_column])

    vectorizer = CountVectorizer(
        analyzer=lambda tokens: tokens,
        lowercase=False,
        min_df=min_df,
        dtype="int32"
    )
    term_document_matrix = vectorizer.fit_transform(df[text_column])
    terms = vectorizer.get_feature_names_out().tolist()
    return term_document_matrix, terms, vectorizer


def build_tf_df_dataframe(term_document_matrix: csr_matrix, terms: list[str],) -> pd.DataFrame:
    tf = term_document_matrix.sum(axis=0).A1
    df = term_document_matrix.astype(bool).sum(axis=0).A1

    tf_df_dataframe = pd.DataFrame({"term": terms, "tf": tf, "df": df})

    tf_df_dataframe = tf_df_dataframe.sort_values(
        by=["tf", "df", "term"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    tf_df_dataframe = tf_df_dataframe.reset_index(drop=True)

    return tf_df_dataframe


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    term_document_matrix, terms, vectorizer = build_term_document_matrix()
    print(type(term_document_matrix))
    print(term_document_matrix.data)
    print(term_document_matrix.indices)
    print(term_document_matrix.indptr)
    print(term_document_matrix.shape)
    tf_df = build_tf_df_dataframe(term_document_matrix, terms)
    print(tf_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
