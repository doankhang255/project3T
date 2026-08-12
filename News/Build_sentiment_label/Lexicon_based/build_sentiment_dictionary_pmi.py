from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Common.matrix_csr_utils import (
    DEFAULT_TOKENIZED_NEWS_PATH,
    NGRAM_SEPARATOR,
    TOKENIZED_SENTENCES_COLUMN,
    build_document_terms,
    build_ngram_terms_with_summary,
    load_tokenized_documents,
)


SCRIPT_DIR = Path(__file__).resolve().parent
LEXICON_DATA_DIR = SCRIPT_DIR / "data"
RESOURCES_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Resources"

INPUT_TOKENIZED_PATH = DEFAULT_TOKENIZED_NEWS_PATH
CANDIDATE_TERMS_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms.parquet"
POSITIVE_SEED_PATH = RESOURCES_DIR / "positive_word.txt"
NEGATIVE_SEED_PATH = RESOURCES_DIR / "negative_word.txt"

OUTPUT_DICTIONARY_PARQUET_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms_dictionary_pmi.parquet"
OUTPUT_DICTIONARY_CSV_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms_dictionary_pmi.csv"
OUTPUT_SEED_RESOLUTION_CSV_PATH = LEXICON_DATA_DIR / "pmi_seed_resolution.csv"

SEED_MIN_N = 1
SEED_MAX_N = 3
CANDIDATE_NGRAM_RANGE = (2, 3)

# Duoi nguong nay, seed bi coi la qua hiem de dong gop tin hieu PMI dang tin cay
# (dong bo voi LEXICON_MIN_DF_FLOOR ben select_lexicon_candidate_terms.py).
SEED_MIN_DF = 20

# So_score = mean(PMI toi seed positive) - mean(PMI toi seed negative).
# Nguong gan nhan duoc tinh theo do lech chuan cua so_score tren toan bo candidate
# (giong cach lam sentiment_index_z o buoc Build_sentiment_index), thay vi mot
# nguong tuyet doi co dinh, vi thang gia tri PMI phu thuoc vao corpus.
LABEL_Z_THRESHOLD = 0.5
MIN_SEED_MATCHES_EACH_SIDE = 1

# Duoi nguong nay, PMI duoc tinh tu qua it bai (df thap) nen do tin cay thap.
# Khong xoa term khoi ket qua - chi danh dau qua cot pmi_confidence de biet
# nhan nao dang tin, nhan nao can can trong (vi du chi dua tren 1 bai bao).
PMI_CONFIDENCE_MIN_DF = 20


def load_seed_words(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    items = [item.strip() for item in re.split(r"[,\n]", text)]
    return [item for item in items if item]


def normalize_for_seed_matching(term: str, separator: str = NGRAM_SEPARATOR) -> str:
    return term.replace(separator, "_")


def resolve_seed_real_forms(
    seed_words: list[str],
    corpus_ngram_terms_df: pd.DataFrame,
    min_df: int = SEED_MIN_DF,
) -> pd.DataFrame:
    """Tim dang thuc te (trong corpus da tokenize) cua tung seed viet tay.

    Mot seed co the ung voi nhieu dang thuc te khac nhau (do bien the tach tu cua
    tokenizer o cac cau khac nhau) - gop df/tf lai theo dung seed goc.
    """
    working_df = corpus_ngram_terms_df.copy()
    working_df["seed_key"] = working_df["term"].apply(normalize_for_seed_matching)

    # Chuan hoa ca seed doc tu file giong het cach chuan hoa term tu corpus, vi
    # file seed co the dang viet toan bo gach duoi ("vuot_ke_hoach") hoac dang
    # "that" theo tokenizer (khoang trang giua token, gach duoi trong tu ghep,
    # vd "vuot ke_hoach") - ca 2 deu phai quy ve cung 1 dang de so khop duoc.
    seed_set = {normalize_for_seed_matching(word) for word in seed_words}
    matched_df = working_df.loc[working_df["seed_key"].isin(seed_set)].copy()

    resolved_df = (
        matched_df.groupby("seed_key")
        .agg(
            df=("df", "sum"),
            tf=("tf", "sum"),
            real_forms=("term", lambda terms: sorted(set(terms))),
        )
        .reset_index()
        .rename(columns={"seed_key": "seed"})
    )

    missing_seeds = seed_set.difference(set(resolved_df["seed"]))
    if missing_seeds:
        print(
            f"[canh bao] {len(missing_seeds)} seed khong xuat hien trong corpus "
            f"(df=0), bi bo qua: {sorted(missing_seeds)}"
        )

    below_min_df = resolved_df.loc[resolved_df["df"].lt(min_df), "seed"].tolist()
    if below_min_df:
        print(
            f"[canh bao] {len(below_min_df)} seed co df < {min_df}, se bi loai "
            f"khoi buoc tinh PMI (van giu trong file goc): {sorted(below_min_df)}"
        )

    return resolved_df.loc[resolved_df["df"].ge(min_df)].reset_index(drop=True)


def load_candidate_terms(
    path: Path = CANDIDATE_TERMS_PATH,
    ngram_range: tuple[int, int] = CANDIDATE_NGRAM_RANGE,
    min_candidate_df: int | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required_columns = {"term", "ngram_n", "df", "tf"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Candidate terms file missing columns: {sorted(missing_columns)}")

    min_n, max_n = ngram_range
    out = df.loc[df["ngram_n"].between(min_n, max_n)]
    if min_candidate_df is not None:
        out = out.loc[out["df"].ge(min_candidate_df)]
    return out.reset_index(drop=True)


def build_binary_document_term_matrix(
    tokenized_documents,
    vocabulary_terms: list[str],
    min_n: int = SEED_MIN_N,
    max_n: int = CANDIDATE_NGRAM_RANGE[1],
):
    document_terms = [
        build_document_terms(raw_tokens, min_n=min_n, max_n=max_n)
        for raw_tokens in tokenized_documents
    ]
    vectorizer = CountVectorizer(
        analyzer=lambda terms: terms,
        lowercase=False,
        vocabulary=vocabulary_terms,
        binary=True,
        dtype="int32",
    )
    matrix = vectorizer.fit_transform(document_terms)
    return matrix


def compute_pmi(
    co_df: np.ndarray,
    candidate_df: np.ndarray,
    seed_df: np.ndarray,
    total_documents: int,
) -> np.ndarray:
    """PMI(term, seed) = log2( (co_df * N) / (df_term * df_seed) ).

    Tra ve NaN cho cap khong dong xuat hien lan nao (co_df = 0) thay vi -inf,
    de loai khoi trung binh thay vi coi la bang chung "rat tieu cuc".
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = co_df * total_documents
        denominator = candidate_df[:, None] * seed_df[None, :]
        pmi = np.log2(numerator / denominator)

    pmi = np.where(co_df > 0, pmi, np.nan)
    return pmi


def build_embedded_seed_exclusion_mask(
    candidate_terms: list[str],
    real_forms: list[str],
    separator: str = NGRAM_SEPARATOR,
) -> np.ndarray:
    """True o (i, j) neu real_forms[j] la 1 day token con lien tiep nam ben
    trong candidate_terms[i] (vd seed "giam" long trong candidate "giam lo").
    Dem 2 dau bang separator de bat dung ranh gioi token, tranh khop nham
    theo ky tu tho. Dung de loai tung cap (candidate, real_form) bi long khoi
    PMI, thay vi loai ca candidate hay ca seed - cac seed khac khong long
    trong candidate van duoc tinh PMI binh thuong.
    """
    padded_candidates = pd.Series(
        [f"{separator}{term}{separator}" for term in candidate_terms]
    )
    mask = np.zeros((len(candidate_terms), len(real_forms)), dtype=bool)
    for form_idx, form in enumerate(real_forms):
        padded_form = f"{separator}{form}{separator}"
        mask[:, form_idx] = padded_candidates.str.contains(
            padded_form, regex=False
        ).to_numpy()
    return mask


def compute_semantic_orientation(
    pmi_positive: np.ndarray,
    pmi_negative: np.ndarray,
) -> pd.DataFrame:
    with np.errstate(invalid="ignore"):
        pos_mean = np.nanmean(pmi_positive, axis=1)
        neg_mean = np.nanmean(pmi_negative, axis=1)

    pos_matches = np.sum(~np.isnan(pmi_positive), axis=1)
    neg_matches = np.sum(~np.isnan(pmi_negative), axis=1)

    so_score = np.where(
        (pos_matches >= MIN_SEED_MATCHES_EACH_SIDE) & (neg_matches >= MIN_SEED_MATCHES_EACH_SIDE),
        np.nan_to_num(pos_mean, nan=0.0) - np.nan_to_num(neg_mean, nan=0.0),
        np.nan,
    )

    return pd.DataFrame(
        {
            "pos_pmi_mean": pos_mean,
            "neg_pmi_mean": neg_mean,
            "pos_seed_matches": pos_matches,
            "neg_seed_matches": neg_matches,
            "so_score": so_score,
        }
    )


def assign_label_from_so_score(so_score: pd.Series, z_threshold: float = LABEL_Z_THRESHOLD) -> pd.Series:
    valid = so_score.dropna()
    mean = valid.mean() if len(valid) else 0.0
    std = valid.std(ddof=0) if len(valid) > 1 else 0.0

    if std == 0 or np.isnan(std):
        so_z = pd.Series(0.0, index=so_score.index)
    else:
        so_z = (so_score - mean) / std

    labels = pd.Series("neutral", index=so_score.index)
    labels = labels.mask(so_score.isna(), "neutral")
    labels = labels.mask(so_z.ge(z_threshold), "positive")
    labels = labels.mask(so_z.le(-z_threshold), "negative")
    return labels, so_z


def build_pmi_dictionary(
    tokenized_path: Path = INPUT_TOKENIZED_PATH,
    candidate_terms_path: Path = CANDIDATE_TERMS_PATH,
    positive_seed_path: Path = POSITIVE_SEED_PATH,
    negative_seed_path: Path = NEGATIVE_SEED_PATH,
    min_candidate_df: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    news_df, tokenized_column = load_tokenized_documents(
        path=tokenized_path,
        tokenized_column=TOKENIZED_SENTENCES_COLUMN,
    )
    tokenized_documents = news_df[tokenized_column].tolist()
    total_documents = len(tokenized_documents)

    corpus_ngram_terms_df, _ = build_ngram_terms_with_summary(
        path=tokenized_path,
        tokenized_column=tokenized_column,
        min_n=SEED_MIN_N,
        max_n=SEED_MAX_N,
    )

    positive_seed_words = load_seed_words(positive_seed_path)
    negative_seed_words = load_seed_words(negative_seed_path)

    positive_resolved_df = resolve_seed_real_forms(positive_seed_words, corpus_ngram_terms_df)
    negative_resolved_df = resolve_seed_real_forms(negative_seed_words, corpus_ngram_terms_df)

    positive_resolved_df["polarity"] = "positive"
    negative_resolved_df["polarity"] = "negative"
    seed_resolution_df = pd.concat([positive_resolved_df, negative_resolved_df], ignore_index=True)

    positive_real_forms = sorted(
        {form for forms in positive_resolved_df["real_forms"] for form in forms}
    )
    negative_real_forms = sorted(
        {form for forms in negative_resolved_df["real_forms"] for form in forms}
    )

    all_candidate_terms_df = load_candidate_terms(
        candidate_terms_path, min_candidate_df=min_candidate_df
    )

    # Mot candidate co the trung voi chinh 1 seed (vd "vuot ke_hoach" vua la seed
    # positive, vua lot vao candidate list vi ca 2 deu tach n-gram tu cung corpus).
    # Neu de PMI tinh binh thuong, no se tu so sanh voi chinh no (co_df = df cua
    # chinh no), sinh ra PMI = log2(N/df) - mot con so lon vo nghia, khong phan
    # anh sentiment. Cac term nay da duoc con nguoi xac nhan nhan qua seed roi nen
    # gan thang nhan theo seed, khong dua qua PMI nua.
    seed_terms_all = set(positive_real_forms) | set(negative_real_forms)
    overlap_mask = all_candidate_terms_df["term"].isin(seed_terms_all)
    seed_overlap_df = all_candidate_terms_df.loc[overlap_mask].reset_index(drop=True)
    candidate_terms_df = all_candidate_terms_df.loc[~overlap_mask].reset_index(drop=True)

    if not seed_overlap_df.empty:
        print(
            f"[canh bao] {len(seed_overlap_df)} candidate trung voi seed, "
            "gan thang nhan theo seed thay vi tinh PMI: "
            f"{sorted(seed_overlap_df['term'].tolist())}"
        )

    candidate_terms = candidate_terms_df["term"].tolist()

    combined_vocabulary = list(
        dict.fromkeys(candidate_terms + positive_real_forms + negative_real_forms)
    )

    matrix = build_binary_document_term_matrix(
        tokenized_documents,
        vocabulary_terms=combined_vocabulary,
        min_n=SEED_MIN_N,
        max_n=CANDIDATE_NGRAM_RANGE[1],
    )

    term_to_col = {term: idx for idx, term in enumerate(combined_vocabulary)}
    candidate_cols = np.array([term_to_col[term] for term in candidate_terms])
    positive_cols = np.array([term_to_col[term] for term in positive_real_forms]) if positive_real_forms else np.array([], dtype=int)
    negative_cols = np.array([term_to_col[term] for term in negative_real_forms]) if negative_real_forms else np.array([], dtype=int)

    matrix_df = np.asarray(matrix.sum(axis=0)).ravel()

    positive_form_to_seed: dict[str, str] = {}
    for _, row in positive_resolved_df.iterrows():
        for form in row["real_forms"]:
            positive_form_to_seed[form] = row["seed"]

    negative_form_to_seed: dict[str, str] = {}
    for _, row in negative_resolved_df.iterrows():
        for form in row["real_forms"]:
            negative_form_to_seed[form] = row["seed"]

    candidate_df_array = matrix_df[candidate_cols].astype(float)

    def build_seed_pmi(
        real_forms: list[str],
        cols: np.ndarray,
        form_to_seed: dict[str, str],
        resolved_df: pd.DataFrame,
    ) -> tuple[np.ndarray, int]:
        if not real_forms:
            return np.empty((len(candidate_terms), 0)), 0

        co_df_forms = (matrix[:, candidate_cols].T @ matrix[:, cols]).toarray().astype(float)
        form_df = matrix_df[cols].astype(float)

        pmi_forms = compute_pmi(co_df_forms, candidate_df_array, form_df, total_documents)

        # Loai tung cap (candidate, real_form) ma real_form bi long ben trong
        # candidate (vd seed "giam" long trong candidate "giam lo"): cap nay
        # dong xuat hien ~100% nen PMI se bi thoi phong vo nghia, khong phan
        # anh lien ket sentiment thuc. Cac seed khac khong long trong candidate
        # do van duoc giu de tinh PMI binh thuong.
        embedded_mask = build_embedded_seed_exclusion_mask(candidate_terms, real_forms)
        pmi_forms = np.where(embedded_mask, np.nan, pmi_forms)

        seed_ids = np.array([form_to_seed[form] for form in real_forms])
        unique_seeds = resolved_df["seed"].tolist()
        pmi_by_seed = np.full((len(candidate_terms), len(unique_seeds)), np.nan)
        for seed_idx, seed in enumerate(unique_seeds):
            form_mask = seed_ids == seed
            if not form_mask.any():
                continue
            with np.errstate(invalid="ignore"):
                pmi_by_seed[:, seed_idx] = np.nanmax(pmi_forms[:, form_mask], axis=1)
        return pmi_by_seed, int(embedded_mask.sum())

    pmi_positive, excluded_positive_pairs = build_seed_pmi(
        positive_real_forms, positive_cols, positive_form_to_seed, positive_resolved_df
    )
    pmi_negative, excluded_negative_pairs = build_seed_pmi(
        negative_real_forms, negative_cols, negative_form_to_seed, negative_resolved_df
    )
    print(
        "[thong ke] Cap (candidate, seed real-form) bi loai vi seed long trong "
        f"candidate: {excluded_positive_pairs} (positive) + "
        f"{excluded_negative_pairs} (negative)"
    )

    so_df = compute_semantic_orientation(pmi_positive, pmi_negative)
    labels, so_z = assign_label_from_so_score(so_df["so_score"])

    pmi_result_df = candidate_terms_df.copy()
    pmi_result_df = pd.concat([pmi_result_df.reset_index(drop=True), so_df.reset_index(drop=True)], axis=1)
    pmi_result_df["so_score_z"] = so_z.reset_index(drop=True)
    pmi_result_df["sentiment_label"] = labels.reset_index(drop=True)
    pmi_result_df["label_source"] = "pmi"
    # df thap -> PMI tinh tu qua it bai, do tin cay thap. Khong xoa term, chi
    # danh dau de biet nhan nao dang tin, nhan nao can can trong khi su dung.
    pmi_result_df["pmi_confidence"] = np.where(
        pmi_result_df["df"].ge(PMI_CONFIDENCE_MIN_DF), "reliable", "low"
    )

    seed_overlap_result_df = seed_overlap_df.copy()
    positive_real_forms_set = set(positive_real_forms)
    seed_overlap_result_df["sentiment_label"] = seed_overlap_result_df["term"].apply(
        lambda term: "positive" if term in positive_real_forms_set else "negative"
    )
    for column in ["pos_pmi_mean", "neg_pmi_mean", "pos_seed_matches", "neg_seed_matches", "so_score", "so_score_z"]:
        seed_overlap_result_df[column] = np.nan
    seed_overlap_result_df["label_source"] = "seed_direct"
    # Nhan gan thang tu seed, khong qua PMI, nen coi la dang tin cay.
    seed_overlap_result_df["pmi_confidence"] = "reliable"

    result_df = pd.concat([pmi_result_df, seed_overlap_result_df], ignore_index=True)

    return result_df, seed_resolution_df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dictionary_df, seed_resolution_df = build_pmi_dictionary()

    LEXICON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dictionary_df.to_parquet(OUTPUT_DICTIONARY_PARQUET_PATH, index=False)
    dictionary_df.to_csv(OUTPUT_DICTIONARY_CSV_PATH, index=False, encoding="utf-8-sig")
    seed_resolution_df.to_csv(OUTPUT_SEED_RESOLUTION_CSV_PATH, index=False, encoding="utf-8-sig")

    print("Input tokenized corpus:", INPUT_TOKENIZED_PATH)
    print("Candidate terms:", CANDIDATE_TERMS_PATH)
    print("Positive seed file:", POSITIVE_SEED_PATH)
    print("Negative seed file:", NEGATIVE_SEED_PATH)
    print("Output dictionary parquet:", OUTPUT_DICTIONARY_PARQUET_PATH)
    print("Output dictionary csv:", OUTPUT_DICTIONARY_CSV_PATH)
    print("Output seed resolution csv:", OUTPUT_SEED_RESOLUTION_CSV_PATH)
    print("Candidate terms scored:", len(dictionary_df))
    print("Label counts:")
    print(dictionary_df["sentiment_label"].value_counts(dropna=False).to_string())
    print("Confidence counts:")
    print(dictionary_df["pmi_confidence"].value_counts(dropna=False).to_string())
    print("\nSeed resolution:")
    print(seed_resolution_df.to_string(index=False))


if __name__ == "__main__":
    main()
