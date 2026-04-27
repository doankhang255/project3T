from __future__ import annotations
import ast
import pandas as pd
from EDA_raw_news import build_year_before_2005_rows
from EDA_raw_news import build_unique_value_counts
from EDA_raw_news import INPUT_PATH
from EDA_raw_news import load_input_data
from EDA_raw_news import NULL_LIKE_VALUES
from EDA_raw_news import normalize_text_columns
from EDA_raw_news import TEXT_COLUMNS


def normalize_domain_column(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    normalize_domain = out.copy()
    domain = normalize_domain["domain"].astype("string").str.lower()
    domain = domain.mask(domain.isna() | domain.fillna("").isin(NULL_LIKE_VALUES))
    domain = domain.str.replace(r"^https?://", "", regex=True)
    domain = domain.str.replace(r"^www\.", "", regex=True)
    domain = domain.str.split("/", n=1).str[0]
    domain = domain.str.split("?", n=1).str[0]
    domain = domain.str.split("#", n=1).str[0]
    normalize_domain["domain"] = domain.mask(domain.fillna("").eq("")).astype("string")
    return out, normalize_domain


def parse_dict_like_keywords(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def parse_keyword_value(raw: object) -> object:
        if pd.isna(raw):
            return pd.NA

        if isinstance(raw, dict):
            keys = [str(key) for key in raw.keys() if str(key) != ""]
            return ", ".join(keys) if keys else pd.NA

        text = str(raw)
        if text == "" or text.lower() in NULL_LIKE_VALUES:
            return pd.NA
        if not (text.startswith("{") and text.endswith("}")):
            return text

        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return text

        if isinstance(parsed, dict):
            keys = [str(key) for key in parsed.keys() if str(key) != ""]
            return ", ".join(keys) if keys else pd.NA
        return text

    out["keywords"] = out["keywords"].apply(parse_keyword_value).astype("string")
    return out


def build_clean_news(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[out["publication_date"].notna()].copy()
    year_before_2005_rows = build_year_before_2005_rows(out)
    out = out.drop(index=year_before_2005_rows.index).copy()
    out = out.drop_duplicates(subset=["link", "publication_date"], keep="first").copy()
    out = out.sort_values(["publication_date", "link"], kind="mergesort").reset_index(drop=True)

    _, normalize_domain = normalize_domain_column(out)
    normalized_keywords = parse_dict_like_keywords(out)

    clean_news = pd.DataFrame(
        {
            "publication_date": out["publication_date"],
            "year": out["year"],
            "domain_norm": normalize_domain["domain"].astype("string"),
            "category": out["category"],
            "title": out["title"],
            "description": out["description"],
            "keywords_norm": normalized_keywords["keywords"].astype("string"),
        }
    )
    return clean_news.reset_index(drop=True)


def main() -> None:
    raw_df = load_input_data(INPUT_PATH)
    normalized_df = normalize_text_columns(raw_df, TEXT_COLUMNS)
    input_rows = len(raw_df)
    prepared_df = normalized_df.copy()
    category_counts = build_unique_value_counts(prepared_df, "category")
    domain_counts = build_unique_value_counts(prepared_df, "domain")
    clean_news_df = build_clean_news(prepared_df)

    print("Input rows:", input_rows)
    print("Clean news rows:", len(clean_news_df))

if __name__ == "__main__":
    main()
