from pathlib import Path
import re
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from underthesea import word_tokenize


INPUT_PATH = Path(__file__).with_name("equity_news_des_title.parquet")
TEXT_COLUMN = "des-title"
TOKENIZED_COLUMN = "Tokenize_des"
TOKEN_PATTERN = re.compile(r"[^\W\d_]", flags=re.UNICODE)


def is_valid_token(token: str) -> bool:
    token = token.strip()
    return len(token) > 1 and bool(TOKEN_PATTERN.search(token)) and not token.isnumeric()


def tokenize_vietnamese_text(text: object) -> list[str]:
    if pd.isna(text):
        return []

    tokens = word_tokenize(str(text).lower(), format=None)
    return [token for token in tokens if is_valid_token(token)]


def load_and_tokenize_equity_news(path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)

    out = df.copy()
    out[TOKENIZED_COLUMN] = out[TEXT_COLUMN].apply(tokenize_vietnamese_text)
    return out


def build_term_document_matrix(
    df: pd.DataFrame,
    text_column: str = TOKENIZED_COLUMN,
    min_df: int = 2,
) -> tuple[csr_matrix, list[str], CountVectorizer]:
    vectorizer = CountVectorizer(
        analyzer=lambda tokens: tokens,
        lowercase=False,
        min_df=min_df,
        dtype="int32",
    )
    term_document_matrix = vectorizer.fit_transform(df[text_column])
    terms = vectorizer.get_feature_names_out().tolist()
    return term_document_matrix, terms, vectorizer


def main() -> None:
    df = load_and_tokenize_equity_news()
    term_doc_matrix, terms, _ = build_term_document_matrix(df)
    print("Term-document matrix shape:", term_doc_matrix.shape)
    print("First 20 terms:", terms[:20])
    print(pd.DataFrame(term_doc_matrix[:5, :10].toarray(), columns=terms[:10]))
    print(df["Tokenize_des"].head())

if __name__ == "__main__":
    main()
