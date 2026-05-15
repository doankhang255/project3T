from pathlib import Path
import sys

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

try:
    from News.Process_sentiment_label.build_CSR_matrix import INPUT_PATH
    from News.Process_sentiment_label.build_CSR_matrix import TOKENIZED_COLUMN
    from News.Process_sentiment_label.build_CSR_matrix import build_term_document_matrix
except ImportError:
    from News.Process_sentiment_label.build_CSR_matrix import INPUT_PATH
    from News.Process_sentiment_label.build_CSR_matrix import TOKENIZED_COLUMN
    from News.Process_sentiment_label.build_CSR_matrix import build_term_document_matrix


def build_tfidf_matrix(
    path: Path = INPUT_PATH,
    text_column: str = TOKENIZED_COLUMN,
    min_df: int = 2,
    norm: str = "l2",
    smooth_idf: bool = True,
    sublinear_tf: bool = False,
) -> tuple[csr_matrix, list[str], CountVectorizer, TfidfTransformer]:
    term_document_matrix, terms, vectorizer = build_term_document_matrix(
        path=path,
        text_column=text_column,
        min_df=min_df,
    )
    transformer = TfidfTransformer(
        norm=norm,
        use_idf=True,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    )
    tfidf_matrix = transformer.fit_transform(term_document_matrix)
    return tfidf_matrix, terms, vectorizer, transformer


def build_idf_dataframe(terms: list[str], transformer: TfidfTransformer) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "term": terms,
                "idf": transformer.idf_,
            }
        )
        .sort_values(["idf", "term"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tfidf_matrix, terms, _, transformer = build_tfidf_matrix()
    idf_df = build_idf_dataframe(terms, transformer)

    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("First 20 terms:", terms[:20])
    print("Top 20 highest IDF terms:")
    print(idf_df.head(20).to_string(index=False))
    print("Preview first 5 documents x first 10 terms:")
    print(pd.DataFrame(tfidf_matrix[:5, :10].toarray(), columns=terms[:10]))


if __name__ == "__main__":
    main()
