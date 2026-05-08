from pathlib import Path
import re
import pandas as pd
from underthesea import word_tokenize


INPUT_PATH = Path(__file__).with_name("equity_news_des_title.parquet")
OUTPUT_PATH = Path(__file__).with_name("equity_news_tokenized.parquet")
TEXT_COLUMN = "des-title"
TOKENIZED_COLUMN = "Tokenize_des"
TOTAL_TOKENIZER_COLUMN = "total_tokenizer"

TOKEN_PATTERN = re.compile(r"[^\W\d_]", flags=re.UNICODE)
SPECIAL_CHARACTER_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+", flags=re.UNICODE)


def clean_token(token: str) -> str | None:
    token = SPECIAL_CHARACTER_PATTERN.sub(" ", token.strip().lower())
    token = WHITESPACE_PATTERN.sub("_", token).strip("_")
    if len(token) <= 1:
        return None
    if not TOKEN_PATTERN.search(token):
        return None
    if token[0].isdigit():
        return None
    return token


def tokenize_vietnamese_text(text: object) -> list[str]:
    if pd.isna(text):
        return []

    tokens = word_tokenize(str(text), format=None)
    cleaned_tokens = [clean_token(token) for token in tokens]
    return [token for token in cleaned_tokens if token is not None]


def load_and_tokenize_equity_news(path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    out = df.copy()
    out[TOKENIZED_COLUMN] = out[TEXT_COLUMN].apply(tokenize_vietnamese_text)
    out[TOTAL_TOKENIZER_COLUMN] = out[TOKENIZED_COLUMN].apply(len)
    return out


def main() -> None:
    df = load_and_tokenize_equity_news(INPUT_PATH)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved tokenized dataframe to: {OUTPUT_PATH}")
    print("Rows:", len(df))


if __name__ == "__main__":
    main()
