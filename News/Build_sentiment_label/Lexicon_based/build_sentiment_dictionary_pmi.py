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
    build_ngram_tf_df_dataframe,
    build_ngram_terms_with_summary,
    load_tokenized_documents,
    normalize_sentence_token_lists,
)


SCRIPT_DIR = Path(__file__).resolve().parent
LEXICON_DATA_DIR = SCRIPT_DIR / "data"
# Resources/ da duoc doi ten/chuyen vao Seed_set_Prepare/manual_seed/ (xem
# Seed_set_Prepare/lexicon_md_pipeline.ipynb) - cap nhat lai duong dan cho
# khop, giu nguyen y nghia goc: day la seed thu cong don le, KHONG phai
# final_seed/ (ban da gop voi Master Dictionary + loc df=0).
RESOURCES_DIR = PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare" / "manual_seed"

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

# ------------------------------------------------------------------
# Context window: don vi dung de dem dong-xuat-hien (co_df). "document" la
# hanh vi goc (nguyen ca bai). "sentence" tach moi bai thanh cac cau rieng
# le va coi moi cau la 1 don vi - hep hon nhieu nen dong-xuat-hien phan anh
# lien ket ngu nghia that hon la "tinh co cung xuat hien trong 1 bai dai".
# ------------------------------------------------------------------
CONTEXT_WINDOW_DOCUMENT = "document"
CONTEXT_WINDOW_SENTENCE = "sentence"

# ------------------------------------------------------------------
# 3 bien the cong thuc PMI (xem giai thich chi tiet trong compute_pmi va
# compute_cds_smoothed_seed_df). "plain" la cong thuc PMI goc, dung cho v1/v2.
# ------------------------------------------------------------------
PMI_VARIANT_PLAIN = "plain"
PMI_VARIANT_PMI_K = "pmi_k"
PMI_VARIANT_ADD_ALPHA = "add_alpha"
PMI_VARIANT_CDS = "cds"

PMI_K_DEFAULT = 3.0
SMOOTHING_ALPHA_DEFAULT = 1.0
CDS_BETA_DEFAULT = 0.7


def load_seed_words(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    items = [item.strip() for item in re.split(r"[,\n]", text)]
    return [item for item in items if item]


def normalize_for_seed_matching(term: str, separator: str = NGRAM_SEPARATOR) -> str:
    return term.replace(separator, "_")


def explode_documents_to_sentences(tokenized_documents: list) -> list[list[str]]:
    """Tach moi document thanh danh sach cac cau rieng le, dung khi
    context_window="sentence". Moi cau tro thanh 1 "don vi" doc lap de tinh
    df/tf/dong-xuat-hien, thay vi ca bai la 1 don vi.

    Dung chung normalize_sentence_token_lists de xu ly nhat quan voi cach
    corpus da duoc tach cau that (vd cot Tokenize_content_sentences cua file
    VNCoreNLP) - neu 1 "document" chi la list token phang (khong co cau that,
    vd file underthesea) thi ham nay tra ve dung 1 "cau" bang ca bai, khong
    tach duoc gi them.
    """
    sentences: list[list[str]] = []
    for raw_document_tokens in tokenized_documents:
        sentences.extend(normalize_sentence_token_lists(raw_document_tokens))
    return sentences


def resolve_seed_real_forms(
    seed_words: list[str],
    corpus_ngram_terms_df: pd.DataFrame,
    min_df: int = SEED_MIN_DF,
    total_units: int | None = None,
    max_df_ratio: float | None = None,
) -> pd.DataFrame:
    """Tim dang thuc te (trong corpus da tokenize) cua tung seed viet tay.

    Mot seed co the ung voi nhieu dang thuc te khac nhau (do bien the tach tu cua
    tokenizer o cac cau khac nhau) - gop df/tf lai theo dung seed goc.

    `total_units` + `max_df_ratio`: neu duoc truyen, loai them ca seed QUA PHO
    BIEN (df_ratio > max_df_ratio) khoi buoc tinh PMI - doi xung voi ngat
    duoi `min_df` da co san. Seed qua pho bien (vd xuat hien trong hon 10% so
    don vi) it co gia tri phan biet sentiment, tuong tu ly do can chan tran
    df_ratio cho candidate.
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

    kept_df = resolved_df.loc[resolved_df["df"].ge(min_df)]

    if max_df_ratio is not None and total_units:
        ratio = kept_df["df"] / total_units
        above_cap = kept_df.loc[ratio.gt(max_df_ratio), "seed"].tolist()
        if above_cap:
            print(
                f"[canh bao] {len(above_cap)} seed co df_ratio > {max_df_ratio} "
                f"(qua pho bien, it gia tri phan biet), bi loai khoi buoc tinh "
                f"PMI: {sorted(above_cap)}"
            )
        kept_df = kept_df.loc[ratio.le(max_df_ratio)]

    return kept_df.reset_index(drop=True)


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
    tokenized_units,
    vocabulary_terms: list[str],
    min_n: int = SEED_MIN_N,
    max_n: int = CANDIDATE_NGRAM_RANGE[1],
):
    """Xay ma tran nhi phan (co/khong xuat hien) theo tung "don vi" trong
    `tokenized_units`. Don vi la document (hanh vi goc) hoac sentence (sau khi
    da explode_documents_to_sentences) tuy context_window - ham nay khong can
    biet no dang xu ly don vi nao, chi nhan list token va tao n-gram ben trong.
    """
    unit_terms = [
        build_document_terms(raw_tokens, min_n=min_n, max_n=max_n)
        for raw_tokens in tokenized_units
    ]
    vectorizer = CountVectorizer(
        analyzer=lambda terms: terms,
        lowercase=False,
        vocabulary=vocabulary_terms,
        binary=True,
        dtype="int32",
    )
    matrix = vectorizer.fit_transform(unit_terms)
    return matrix


def compute_pmi(
    co_df: np.ndarray,
    candidate_df: np.ndarray,
    seed_df: np.ndarray,
    total_units: int,
    variant: str = PMI_VARIANT_PLAIN,
    k: float = PMI_K_DEFAULT,
    alpha: float = SMOOTHING_ALPHA_DEFAULT,
) -> np.ndarray:
    """PMI(term, seed) = log2( (co_df * N) / (df_term * df_seed) ).

    Tra ve NaN cho cap khong dong xuat hien lan nao (co_df = 0) thay vi -inf,
    de loai khoi trung binh thay vi coi la bang chung "rat tieu cuc".

    3 bien the (chon qua `variant`):
    - "plain": cong thuc goc o tren, dung cho build_sentiment_dictionary_pmi
      (v1) va _v2. Khong sua doi gi.
    - "pmi_k": PMI^k (Daille 1994) - cong them (k-1)*log2(P(x,y)) vao PMI goc.
      P(x,y) <= 1 nen log2(P(x,y)) <= 0 - cap co qua it lan dong xuat hien
      thuc te (P(x,y) rat nho) bi phat cang nang, du ty le co_df/df cao. Day
      la cach "discount tan suat" ma khong can nguong cat cung nhu SEED_MIN_DF.
    - "add_alpha": cong pseudo-count `alpha` vao ca 3 so dem (co_df, df_term,
      df_seed) truoc khi tinh ty le, kieu Laplace smoothing. Ket qua la KHONG
      con cap nao bi NaN nua (kha ca_df=0 cung ra 1 gia tri am huu han) - so
      voi "plain"/"pmi_k" (loai cap co_df=0 khoi trung binh), add_alpha coi
      "chua tung dong xuat hien" la bang chung sentiment TRAI CHIEU yeu, chu
      khong phai "thieu du lieu".
    - "cds": khong xu ly gi them o day - ham goi (build_seed_pmi) da tinh san
      `seed_df` theo phien ban lam min (xem compute_cds_smoothed_seed_df)
      truoc khi truyen vao, nen chi can dung lai cong thuc "plain" voi
      seed_df da duoc thay the.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if variant == PMI_VARIANT_ADD_ALPHA:
            co = co_df + alpha
            candidate_smoothed = candidate_df[:, None] + alpha
            seed_smoothed = seed_df[None, :] + alpha
            return np.log2((co * total_units) / (candidate_smoothed * seed_smoothed))

        numerator = co_df * total_units
        denominator = candidate_df[:, None] * seed_df[None, :]
        pmi = np.log2(numerator / denominator)
        pmi = np.where(co_df > 0, pmi, np.nan)

        if variant == PMI_VARIANT_PMI_K:
            p_xy = np.where(co_df > 0, co_df / total_units, np.nan)
            pmi = pmi + (k - 1) * np.log2(p_xy)

        return pmi


def compute_cds_smoothed_seed_df(
    resolved_df: pd.DataFrame,
    beta: float = CDS_BETA_DEFAULT,
) -> dict[str, float]:
    """Lam phang do lech tan suat GIUA CAC SEED trong CUNG 1 nhom cuc
    (positive hoac negative rieng), kieu context distribution smoothing cua
    Levy, Goldberg & Dagan (2015) ap dung cho word2vec: nang df cua seed len
    luy thua `beta` (< 1) truoc khi dung lam mau so trong PMI - seed hiem
    trong nhom duoc "nang" ty trong tuong doi len, seed pho bien trong nhom bi
    "ha" ty trong xuong, tranh 1 seed pho bien qua muc lan at ket qua trung
    binh cua ca nhom.

    Khac voi ban goc cua Levy et al. (lam phang tren TOAN BO tu vung dung de
    negative sampling), o day chi lam phang NOI BO trong 1 nhom cuc (positive
    hoac negative) vi seed set chi la vai chuc tu co dinh, khong phai tu vung
    day du. Gia tri tra ve duoc quy lai ve dung tong df goc cua nhom (khong
    chi la ty trong 0-1) de van dung truc tiep duoc trong cong thuc PMI chuan
    (thay cho df_seed goc) ma khong lam lech don vi/thang do.
    """
    seeds = resolved_df["seed"].tolist()
    raw_df = resolved_df["df"].to_numpy(dtype=float)
    smoothed = raw_df**beta
    share = smoothed / smoothed.sum()
    effective_df = share * raw_df.sum()
    return dict(zip(seeds, effective_df))


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
    context_window: str = CONTEXT_WINDOW_DOCUMENT,
    pmi_variant: str = PMI_VARIANT_PLAIN,
    pmi_k: float = PMI_K_DEFAULT,
    smoothing_alpha: float = SMOOTHING_ALPHA_DEFAULT,
    cds_beta: float = CDS_BETA_DEFAULT,
    max_df_ratio: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    news_df, tokenized_column = load_tokenized_documents(
        path=tokenized_path,
        tokenized_column=TOKENIZED_SENTENCES_COLUMN,
    )
    tokenized_documents = news_df[tokenized_column].tolist()

    # context_window quyet dinh "don vi" dung de dem dong-xuat-hien: nguyen
    # document (hanh vi goc, v1/v2) hoac tach nho thanh tung sentence rieng.
    if context_window == CONTEXT_WINDOW_SENTENCE:
        unit_documents = explode_documents_to_sentences(tokenized_documents)
    else:
        unit_documents = tokenized_documents
    total_units = len(unit_documents)

    if context_window == CONTEXT_WINDOW_SENTENCE:
        corpus_ngram_terms_df = build_ngram_tf_df_dataframe(
            tokenized_documents=unit_documents,
            total_documents=total_units,
            min_n=SEED_MIN_N,
            max_n=SEED_MAX_N,
        )
    else:
        corpus_ngram_terms_df, _ = build_ngram_terms_with_summary(
            path=tokenized_path,
            tokenized_column=tokenized_column,
            min_n=SEED_MIN_N,
            max_n=SEED_MAX_N,
        )

    positive_seed_words = load_seed_words(positive_seed_path)
    negative_seed_words = load_seed_words(negative_seed_path)

    positive_resolved_df = resolve_seed_real_forms(
        positive_seed_words, corpus_ngram_terms_df, total_units=total_units, max_df_ratio=max_df_ratio
    )
    negative_resolved_df = resolve_seed_real_forms(
        negative_seed_words, corpus_ngram_terms_df, total_units=total_units, max_df_ratio=max_df_ratio
    )

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
        unit_documents,
        vocabulary_terms=combined_vocabulary,
        min_n=SEED_MIN_N,
        max_n=CANDIDATE_NGRAM_RANGE[1],
    )

    term_to_col = {term: idx for idx, term in enumerate(combined_vocabulary)}
    candidate_cols = np.array([term_to_col[term] for term in candidate_terms])
    positive_cols = np.array([term_to_col[term] for term in positive_real_forms]) if positive_real_forms else np.array([], dtype=int)
    negative_cols = np.array([term_to_col[term] for term in negative_real_forms]) if negative_real_forms else np.array([], dtype=int)

    matrix_unit_df = np.asarray(matrix.sum(axis=0)).ravel()

    # Chan tran df_ratio cho candidate: candidate qua pho bien (xuat hien o
    # ty le don vi > max_df_ratio) it co gia tri phan biet sentiment - loai
    # khoi buoc tinh PMI (khong xoa khoi candidate_ngram_terms.parquet goc).
    candidate_df_all = matrix_unit_df[candidate_cols].astype(float)
    if max_df_ratio is not None:
        candidate_ratio_all = candidate_df_all / total_units
        keep_mask = candidate_ratio_all <= max_df_ratio
    else:
        keep_mask = np.ones_like(candidate_df_all, dtype=bool)

    excluded_high_df_df = candidate_terms_df.loc[~keep_mask].copy()
    if not excluded_high_df_df.empty:
        print(
            f"[canh bao] {len(excluded_high_df_df)} candidate co df_ratio > "
            f"{max_df_ratio} (qua pho bien), bi loai khoi buoc tinh PMI."
        )

    candidate_terms_df = candidate_terms_df.loc[keep_mask].reset_index(drop=True)
    candidate_terms = candidate_terms_df["term"].tolist()
    candidate_cols = candidate_cols[keep_mask]
    candidate_df_array = candidate_df_all[keep_mask]

    positive_form_to_seed: dict[str, str] = {}
    for _, row in positive_resolved_df.iterrows():
        for form in row["real_forms"]:
            positive_form_to_seed[form] = row["seed"]

    negative_form_to_seed: dict[str, str] = {}
    for _, row in negative_resolved_df.iterrows():
        for form in row["real_forms"]:
            negative_form_to_seed[form] = row["seed"]

    def build_seed_pmi(
        real_forms: list[str],
        cols: np.ndarray,
        form_to_seed: dict[str, str],
        resolved_df: pd.DataFrame,
    ) -> tuple[np.ndarray, int]:
        if not real_forms:
            return np.empty((len(candidate_terms), 0)), 0

        co_df_forms = (matrix[:, candidate_cols].T @ matrix[:, cols]).toarray().astype(float)
        raw_form_df = matrix_unit_df[cols].astype(float)

        if pmi_variant == PMI_VARIANT_CDS:
            cds_seed_df = compute_cds_smoothed_seed_df(resolved_df, beta=cds_beta)
            form_df = np.array([cds_seed_df[form_to_seed[form]] for form in real_forms], dtype=float)
        else:
            form_df = raw_form_df

        pmi_forms = compute_pmi(
            co_df_forms,
            candidate_df_array,
            form_df,
            total_units,
            variant=pmi_variant,
            k=pmi_k,
            alpha=smoothing_alpha,
        )

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
