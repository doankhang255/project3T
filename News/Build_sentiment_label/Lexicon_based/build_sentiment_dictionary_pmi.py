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

INPUT_TOKENIZED_PATH = DEFAULT_TOKENIZED_NEWS_PATH
CANDIDATE_TERMS_PATH = LEXICON_DATA_DIR / "candidate_ngram_terms.parquet"

# 7 danh muc cua Loughran-McDonald, khop voi ten file trong
# Seed_set_Prepare/final_seed/{category}_word.txt. KHONG con ghep cap doi lap
# (positive vs negative) nhu ban PMI nhi phan truoc - moi danh muc la 1 co
# doc lap, 1 candidate co the thuoc 0, 1 hoac nhieu danh muc cung luc, dung
# tinh than 7 danh sach doc lap cua Loughran-McDonald goc.
CATEGORY_NAMES = [
    "negative",
    "positive",
    "uncertainty",
    "litigious",
    "strong_modal",
    "weak_modal",
    "constraining",
]
FINAL_SEED_DIR = (
    PROJECT_ROOT / "News" / "Build_sentiment_label" / "Seed_set_Prepare" / "final_seed"
)

SEED_MIN_N = 1
SEED_MAX_N = 3
CANDIDATE_NGRAM_RANGE = (2, 3)

# Duoi nguong nay, seed bi coi la qua hiem de dong gop tin hieu PMI dang tin cay
# (dong bo voi LEXICON_MIN_DF_FLOOR ben select_lexicon_candidate_terms.py).
SEED_MIN_DF = 20

# Nguong z-score de gan mot candidate vao 1 danh muc (tren phan phoi diem cua
# CHINH danh muc do qua toan bo candidate), giong cach lam sentiment_index_z o
# buoc Build_sentiment_index, thay vi mot nguong tuyet doi co dinh.
LABEL_Z_THRESHOLD = 0.5

# Duoi nguong nay, PMI duoc tinh tu qua it bai/cau (df thap) nen do tin cay thap.
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
# compute_cds_smoothed_seed_df). "plain" la cong thuc PMI goc.
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
    duoi `min_df` da co san.
    """
    working_df = corpus_ngram_terms_df.copy()
    working_df["seed_key"] = working_df["term"].apply(normalize_for_seed_matching)

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
    `tokenized_units`. Don vi la document hoac sentence tuy context_window.
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
    - "plain": cong thuc goc o tren.
    - "pmi_k": PMI^k (Daille 1994) - cong them (k-1)*log2(P(x,y)) vao PMI goc,
      phat cap co qua it lan dong xuat hien thuc te du ty le co_df/df cao.
    - "add_alpha": cong pseudo-count `alpha` vao ca 3 so dem truoc khi tinh ty
      le, kieu Laplace smoothing. KHONG con cap nao bi NaN - ca cap co_df=0
      cung ra 1 gia tri am huu han, duoc coi la bang chung sentiment trai
      chieu YEU thay vi "thieu du lieu". Day la ly do add_alpha luon tinh
      trung binh tren TOAN BO seed (khong bi loai tru vi thieu match), it bi
      nhieu boi hien tuong phuong sai mau nho hon plain/pmi_k/cds.
    - "cds": khong xu ly gi them o day - ham goi da tinh san `seed_df` theo
      phien ban lam min (xem compute_cds_smoothed_seed_df) truoc khi truyen
      vao, nen chi can dung lai cong thuc "plain" voi seed_df da thay the.
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
    """Lam phang do lech tan suat GIUA CAC SEED trong CUNG 1 danh muc, kieu
    context distribution smoothing cua Levy, Goldberg & Dagan (2015) ap dung
    cho word2vec: nang df cua seed len luy thua `beta` (< 1) truoc khi dung
    lam mau so trong PMI. Gia tri tra ve duoc quy lai ve dung tong df goc cua
    danh muc de van dung truc tiep duoc trong cong thuc PMI chuan.
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
    Dung de loai tung cap (candidate, real_form) bi long khoi PMI, thay vi
    loai ca candidate hay ca seed.
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


# ============================================================
# PMI DA DANH MUC (multi-category, khong ghep cap doi lap)
# ============================================================
#
# Khac voi ban PMI nhi phan (positive vs negative, so_score = hieu 2 cuc),
# 7 danh muc Loughran-McDonald KHONG phai cac cap doi lap cua nhau - 1
# candidate co the vua "negative" vua "litigious" vua "uncertainty" cung luc.
# Vi vay moi danh muc duoc cham diem DOC LAP: category_score(candidate) =
# trung binh PMI(candidate, seed) tren toan bo seed CUA CHINH danh muc do
# (khong tru di danh muc nao khac), roi chuan hoa z-score tren toan bo
# candidate DE RIENG cho danh muc do.
#
# compute_multi_category_pmi() lam phan NANG (build ma tran, tinh PMI cho
# tung cap candidate-seed) CHI 1 LAN. aggregate_category_labels() lam phan
# NHE (gop theo min_seed_matches + z_threshold) - co the goi lai nhieu lan
# voi cac nguong khac nhau MA KHONG can tinh lai PMI, de tien thu nghiem.


def compute_multi_category_pmi(
    tokenized_path: Path = INPUT_TOKENIZED_PATH,
    candidate_terms_path: Path = CANDIDATE_TERMS_PATH,
    seed_dir: Path = FINAL_SEED_DIR,
    categories: list[str] = CATEGORY_NAMES,
    min_candidate_df: int | None = None,
    context_window: str = CONTEXT_WINDOW_SENTENCE,
    pmi_variant: str = PMI_VARIANT_ADD_ALPHA,
    pmi_k: float = PMI_K_DEFAULT,
    smoothing_alpha: float = SMOOTHING_ALPHA_DEFAULT,
    cds_beta: float = CDS_BETA_DEFAULT,
    max_df_ratio: float | None = None,
    seed_min_df: int = SEED_MIN_DF,
) -> dict:
    news_df, tokenized_column = load_tokenized_documents(
        path=tokenized_path,
        tokenized_column=TOKENIZED_SENTENCES_COLUMN,
    )
    tokenized_documents = news_df[tokenized_column].tolist()

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

    print(f"Tong don vi ({context_window}):", total_units)

    # Resolve seed cho tung danh muc rieng (moi danh muc 1 file seed rieng).
    category_resolved: dict[str, pd.DataFrame] = {}
    category_real_forms: dict[str, list[str]] = {}
    for category in categories:
        seed_words = load_seed_words(seed_dir / f"{category}_word.txt")
        resolved_df = resolve_seed_real_forms(
            seed_words,
            corpus_ngram_terms_df,
            min_df=seed_min_df,
            total_units=total_units,
            max_df_ratio=max_df_ratio,
        )
        category_resolved[category] = resolved_df
        real_forms = sorted({form for forms in resolved_df["real_forms"] for form in forms})
        category_real_forms[category] = real_forms
        print(f"[{category}] {len(resolved_df)}/{len(seed_words)} seed du dieu kien tinh PMI")

    all_candidate_terms_df = load_candidate_terms(
        candidate_terms_path, min_candidate_df=min_candidate_df
    )

    # Mot candidate co the trung voi chinh 1 seed CUA BAT KY danh muc nao.
    # Cung logic nhu ban nhi phan: gan thang nhan cho danh muc do (khong qua
    # PMI, tranh tu so sanh voi chinh no), loai khoi tap candidate tinh PMI.
    seed_terms_all: set[str] = set()
    term_to_seed_categories: dict[str, list[str]] = {}
    for category, real_forms in category_real_forms.items():
        for form in real_forms:
            seed_terms_all.add(form)
            term_to_seed_categories.setdefault(form, []).append(category)

    overlap_mask = all_candidate_terms_df["term"].isin(seed_terms_all)
    seed_overlap_df = all_candidate_terms_df.loc[overlap_mask].reset_index(drop=True)
    candidate_terms_df = all_candidate_terms_df.loc[~overlap_mask].reset_index(drop=True)

    if not seed_overlap_df.empty:
        print(
            f"[canh bao] {len(seed_overlap_df)} candidate trung voi seed cua it "
            "nhat 1 danh muc, gan thang nhan theo danh muc do thay vi tinh PMI."
        )

    candidate_terms = candidate_terms_df["term"].tolist()
    combined_vocabulary = list(
        dict.fromkeys(candidate_terms + sorted(seed_terms_all))
    )

    matrix = build_binary_document_term_matrix(
        unit_documents,
        vocabulary_terms=combined_vocabulary,
        min_n=SEED_MIN_N,
        max_n=CANDIDATE_NGRAM_RANGE[1],
    )
    term_to_col = {term: idx for idx, term in enumerate(combined_vocabulary)}
    matrix_unit_df = np.asarray(matrix.sum(axis=0)).ravel()

    candidate_cols_all = np.array([term_to_col[term] for term in candidate_terms])
    candidate_df_all = matrix_unit_df[candidate_cols_all].astype(float)

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
    candidate_cols = candidate_cols_all[keep_mask]
    candidate_df_array = candidate_df_all[keep_mask]

    # QUAN TRONG: candidate_df_array la df cua candidate O CAP DON VI DANG DUNG
    # (cau/tai lieu), KHAC voi cot "df" san co trong candidate_terms_df (do la
    # df cap TAI LIEU tu file candidate_ngram_terms.parquet, luon co san du
    # context_window la gi). Candidate hiem o cap don vi bi PMI (dac biet
    # add_alpha) thoi phong diem DEU tren moi danh muc - can gia tri nay o
    # buoc aggregate_category_labels() de loc rieng, khong dua vao cot "df" cu.
    candidate_terms_df = candidate_terms_df.copy()
    candidate_terms_df["candidate_unit_df"] = candidate_df_array

    category_pmi_by_seed: dict[str, tuple[np.ndarray, list[str]]] = {}
    for category in categories:
        resolved_df = category_resolved[category]
        real_forms = category_real_forms[category]
        unique_seeds = resolved_df["seed"].tolist()

        if not real_forms:
            category_pmi_by_seed[category] = (
                np.empty((len(candidate_terms), 0)),
                unique_seeds,
            )
            continue

        cols = np.array([term_to_col[form] for form in real_forms])
        co_df_forms = (matrix[:, candidate_cols].T @ matrix[:, cols]).toarray().astype(float)
        raw_form_df = matrix_unit_df[cols].astype(float)

        # Anh xa moi dang thuc te (real form) ve dung seed goc cua no - dung
        # vong lap tuong minh (khong dung comprehension + walrus) de tranh loi
        # pham vi bien kho hieu khi 1 ten bien duoc dung o nhieu comprehension
        # ke nhau trong cung 1 ham.
        form_to_seed: dict[str, str] = {}
        for _, row in resolved_df.iterrows():
            for form in row["real_forms"]:
                form_to_seed[form] = row["seed"]

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

        embedded_mask = build_embedded_seed_exclusion_mask(candidate_terms, real_forms)
        pmi_forms = np.where(embedded_mask, np.nan, pmi_forms)

        seed_ids = np.array([form_to_seed[form] for form in real_forms])
        pmi_by_seed = np.full((len(candidate_terms), len(unique_seeds)), np.nan)
        for seed_idx, seed in enumerate(unique_seeds):
            form_mask = seed_ids == seed
            if not form_mask.any():
                continue
            with np.errstate(invalid="ignore"):
                pmi_by_seed[:, seed_idx] = np.nanmax(pmi_forms[:, form_mask], axis=1)

        category_pmi_by_seed[category] = (pmi_by_seed, unique_seeds)

    return {
        "candidate_terms_df": candidate_terms_df,
        "category_pmi_by_seed": category_pmi_by_seed,
        "category_resolved": category_resolved,
        "seed_overlap_df": seed_overlap_df,
        "term_to_seed_categories": term_to_seed_categories,
        "categories": categories,
        "pmi_variant": pmi_variant,
        "context_window": context_window,
    }


def aggregate_category_labels(
    result: dict,
    min_seed_matches: int = 1,
    z_threshold: float = LABEL_Z_THRESHOLD,
    min_candidate_unit_df: int | None = None,
    center_on_candidate_mean: bool = True,
) -> pd.DataFrame:
    """Ap dung 1 bo nguong (min_seed_matches, z_threshold, min_candidate_unit_df)
    len ket qua PMI da tinh san trong `result` (tu compute_multi_category_pmi).
    Tach rieng khoi buoc tinh PMI (nang nhat) de thu nhieu nguong khac nhau
    nhanh, khong can build lai ma tran / tinh lai PMI moi lan doi nguong.

    `min_candidate_unit_df`: loai candidate qua hiem O CAP DON VI DANG DUNG
    (candidate_unit_df, xem compute_multi_category_pmi) truoc khi tinh z-score.
    Da kiem chung: chi loc theo df KHONG du de sua loi "1 candidate bi gan ca
    7 category" - ty le nay hau nhu khong doi du nang nguong (xem
    center_on_candidate_mean ben duoi de biet nguyen nhan that).

    `center_on_candidate_mean`: (mac dinh BAT) voi add_alpha, so hang
    1/(candidate_df+alpha) trong cong thuc PMI la CHUNG cho ca 7 category cua
    CUNG 1 candidate - candidate cang hiem thi diem RAW cang bi doi len DEU o
    moi category, khong lien quan gi den sentiment that. O ban PMI nhi phan
    (so_score = mean(PMI_pos) - mean(PMI_neg)) so hang nay tu trieu tieu vi bi
    tru; ban da category doc lap khong co phep tru nao nen bias lo ra. Da kiem
    chung tren du lieu that: tuong quan giua candidate_unit_df va diem RAW
    trung binh 7 category la -0,56 (rat manh). Cach sua: voi tung candidate,
    tru di DIEM TRUNG BINH CUA CHINH NO tren ca 7 category truoc khi chuan hoa
    z-score - mo phong dung phep tru da hoat dong tot o ban nhi phan, ap dung
    cho N category thay vi 2. Sau khi ap dung, candidate bi gan ca 7 (hoac 6)
    category cung luc bien mat hoan toan tren du lieu that (tu 20.712 candidate
    xuong 0).
    """
    candidate_terms_df = result["candidate_terms_df"]
    categories = result["categories"]

    if min_candidate_unit_df is not None:
        keep_mask = (candidate_terms_df["candidate_unit_df"] >= min_candidate_unit_df).to_numpy()
    else:
        keep_mask = np.ones(len(candidate_terms_df), dtype=bool)

    out = candidate_terms_df.loc[keep_mask].reset_index(drop=True)

    # Pass 1: tinh raw_score (+ matches) cho ca 7 category truoc, chua chuan
    # hoa z-score - can du 7 cot nay cung luc de tinh diem trung bien theo
    # HANG (theo candidate) o buoc center_on_candidate_mean ben duoi.
    raw_scores: dict[str, np.ndarray] = {}
    match_counts: dict[str, np.ndarray] = {}
    for category in categories:
        pmi_by_seed_full, unique_seeds = result["category_pmi_by_seed"][category]
        pmi_by_seed = pmi_by_seed_full[keep_mask]
        if pmi_by_seed.shape[1] == 0:
            raw_scores[category] = np.full(len(out), np.nan)
            match_counts[category] = np.zeros(len(out), dtype=int)
            continue

        matches = np.sum(~np.isnan(pmi_by_seed), axis=1)
        with np.errstate(invalid="ignore"):
            raw_mean = np.nanmean(pmi_by_seed, axis=1)
        raw_scores[category] = np.where(matches >= min_seed_matches, raw_mean, np.nan)
        match_counts[category] = matches

    if center_on_candidate_mean:
        score_matrix = np.column_stack([raw_scores[c] for c in categories])
        with np.errstate(invalid="ignore"):
            row_mean = np.nanmean(score_matrix, axis=1)
    else:
        row_mean = np.zeros(len(out))

    # Pass 2: tru di diem trung binh hang (neu bat), roi moi chuan hoa
    # z-score TREN COT (qua toan bo candidate) cho tung category.
    for category in categories:
        score = raw_scores[category] - row_mean

        valid = score[~np.isnan(score)]
        mean = valid.mean() if len(valid) else 0.0
        std = valid.std(ddof=0) if len(valid) > 1 else 0.0
        if std == 0 or np.isnan(std):
            z = np.zeros_like(score)
        else:
            z = (score - mean) / std
        z = np.where(np.isnan(score), np.nan, z)

        out[f"{category}_score"] = raw_scores[category]
        out[f"{category}_score_centered"] = score
        out[f"{category}_z"] = z
        out[f"{category}_matches"] = match_counts[category]
        out[f"{category}_flag"] = z >= z_threshold

    # candidate trung voi seed cua 1 hay nhieu danh muc: gan flag=True thang
    # cho danh muc do (khong qua PMI), cac danh muc khac de NaN/False vi
    # khong tinh PMI cho nhom nay.
    seed_overlap_df = result["seed_overlap_df"]
    term_to_seed_categories = result["term_to_seed_categories"]
    if not seed_overlap_df.empty:
        overlap_rows = seed_overlap_df.copy()
        for category in categories:
            overlap_rows[f"{category}_score"] = np.nan
            overlap_rows[f"{category}_z"] = np.nan
            overlap_rows[f"{category}_matches"] = 0
            overlap_rows[f"{category}_flag"] = False
        for idx, term in overlap_rows["term"].items():
            for category in term_to_seed_categories.get(term, []):
                overlap_rows.at[idx, f"{category}_flag"] = True
        overlap_rows["label_source"] = "seed_direct"
        out["label_source"] = "pmi"
        out = pd.concat([out, overlap_rows], ignore_index=True)
    else:
        out["label_source"] = "pmi"

    out["pmi_confidence"] = np.where(
        out["df"].ge(PMI_CONFIDENCE_MIN_DF), "reliable", "low"
    )
    return out
