from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

MASTER_DICTIONARY_PATH = DATA_DIR / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
OUTPUT_DIR = DATA_DIR / "categories"

# Cac cot flag trong Master Dictionary: gia tri > 0 la nam tu duoc gan vao
# danh muc do (khong phai co/khong dang boolean), = 0 la khong thuoc danh muc.
CATEGORY_COLUMNS = [
    "Negative",
    "Positive",
    "Uncertainty",
    "Litigious",
    "Strong_Modal",
    "Weak_Modal",
    "Constraining",
]


def load_master_dictionary(path: Path = MASTER_DICTIONARY_PATH) -> pd.DataFrame:
    # keep_default_na=False + na_values=[""]: file goc co tu that "NULL" ("null
    # and void") bi pandas hieu nham thanh NaN neu dung bo NA string mac dinh.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing_columns = set(CATEGORY_COLUMNS + ["Word"]).difference(df.columns)
    if missing_columns:
        raise ValueError(f"Master Dictionary thieu cot: {sorted(missing_columns)}")
    return df


def extract_category_words(df: pd.DataFrame, category: str) -> pd.DataFrame:
    matched = df.loc[df[category] > 0, ["Word", "Word Count", "Doc Count", category]].copy()
    matched = matched.rename(columns={category: f"{category}_year_added"})
    matched = matched.sort_values("Word Count", ascending=False).reset_index(drop=True)
    return matched


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_master_dictionary()
    print("Master Dictionary path:", MASTER_DICTIONARY_PATH)
    print("Total words:", len(df))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for category in CATEGORY_COLUMNS:
        category_df = extract_category_words(df, category)
        output_path = OUTPUT_DIR / f"{category.lower()}_master_dictionary.csv"
        category_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"\n{category}: {len(category_df)} tu")
        print("Top 15 theo Word Count:")
        print(category_df.head(15)[["Word", "Word Count", "Doc Count"]].to_string(index=False))
        print("Da luu:", output_path)


if __name__ == "__main__":
    main()
