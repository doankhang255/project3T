# Traditional_ML — Backlog cải tiến gán nhãn sentiment

> **Trạng thái:** ghi lại để làm sau. Hiện ground truth chỉ **152 dòng**
> (neg 49 / neu 68 / pos 35) nên mọi thay đổi bên dưới **chưa chắc tối ưu** —
> sai số CV ±0.02 macro-F1 lớn hơn phần lớn kỳ vọng cải thiện. Chỉ nên
> triển khai + đo lại khi GT đã lên **vài trăm → vài nghìn dòng** và cân bằng
> hơn giữa 3 lớp.
>
> **Cách thử an toàn:** làm từng ý thành thí nghiệm riêng trong
> `experiment_vocab/` (import hàm thuần, tự wiring, chỉ xuất bảng so sánh),
> **không sửa pipeline chính** cho tới khi có bằng chứng CV rõ ràng.

## Baseline hiện tại (đối chiếu khi thử cải tiến)

5-fold stratified CV, TF-IDF fit per-fold, tokenizer VNCoreNLP.

| model | macro F1 | accuracy |
|---|--:|--:|
| random_forest | 0.634 | 0.658 |
| naive_bayes | 0.605 | 0.632 |
| logistic_regression | 0.593 | 0.632 |
| svm | 0.562 | 0.638 |

Vocab (fit toàn bộ 152): 1.468 term (449 uni / 630 bi / 389 tri).
Lớp `positive` là lớp yếu nhất ở mọi model (F1 0.30–0.56).

---

## Mục lục

- [0. Những chỗ ĐANG ĐÚNG — đừng đụng](#0-những-chỗ-đang-đúng--đừng-đụng)
- [A. Feature theo 7 nhóm lexicon tài chính](#a-feature-theo-7-nhóm-lexicon-tài-chính)
- [B. Xử lý phủ định](#b-xử-lý-phủ-định)
- [B'. POS-restricted n-gram + chuẩn hoá số](#b-pos-restricted-n-gram--chuẩn-hoá-số)
- [C. Chọn feature / trọng số có giám sát (per-fold)](#c-chọn-feature--trọng-số-có-giám-sát-per-fold)
- [C'. Feature SO-PMI cấp bài](#c-feature-so-pmi-cấp-bài)
- [D. Model & methodology](#d-model--methodology)
- [E. Chất lượng nhãn](#e-chất-lượng-nhãn)
- [F. Weak / semi-supervised](#f-weak--semi-supervised)
- [Không nên tốn công](#không-nên-tốn-công)
- [Thứ tự ưu tiên](#thứ-tự-ưu-tiên)
- [Tham khảo (REF)](#tham-khảo-ref)

---

## 0. Những chỗ ĐANG ĐÚNG — đừng đụng

| Điểm | Căn cứ |
|---|---|
| Công thức TF-IDF `w = (1+log tf)/(1+log a_j) · log(N/df)` | **Chính là phương trình (1) của Loughran-McDonald 2011** cho văn bản tài chính. `1/(1+log a_j)` là document-length adjustment của họ. Không phải công thức "lạ" — giữ. |
| `max_df_ratio = 0.85` + idf | LM 2011: term weighting giảm attenuation bias từ high-frequency words. |
| Leak-free per-fold CV, stratified, macro-F1, `class_weight=balanced` | Chuẩn. |
| Tokenizer VNCoreNLP đồng bộ với Lexicon_based / Build_sentiment_index | Đúng. |

Nền tảng ổn — cải tiến nằm ở **feature** và **methodology**, không viết lại pipeline.

---

## A. Feature theo 7 nhóm lexicon tài chính

- [ ] **Trạng thái:** chưa làm — *ưu tiên #1*

**Ý tưởng.** Ghép cạnh ma trận TF-IDF một khối feature nhỏ (~16 cột) đếm hit
theo từng nhóm từ điển tài chính đã curate.

**Căn cứ.**
- Loughran-McDonald 2011: từ điển đa dụng (Harvard) phân loại sai ~74% từ
  trong văn bản tài chính; từ điển chuyên ngành giảm measurement error, tăng
  power. LM xét cả **tỷ lệ thô** lẫn **tf-idf weighted** của mỗi nhóm.
- Tetlock 2007: trong 77 category chỉ **Negative** và **Weak-modal** tải tín
  hiệu → kỳ vọng `neg_prop`, `weak_modal_prop` mạnh nhất.

**Nguồn dữ liệu (đã có).** `News/Build_sentiment_label/Seed_set_Prepare/final_seed/`:
`negative_word.txt`, `positive_word.txt`, `uncertainty_word.txt`,
`litigious_word.txt`, `strong_modal_word.txt`, `weak_modal_word.txt`,
`constraining_word.txt` (+ `../negation_cue_words.txt`).

**Cách làm.**
1. Hàm `load_lexicon_categories() -> dict[str, set[str]]` đọc 7 file trên
   (token underscore-joined, khớp định dạng token VNCoreNLP).
2. Với mỗi bài, từ `_document_terms` (hoặc `Tokenize_content`) tính cho mỗi
   nhóm `c`:
   - `hits_c` = số token thuộc nhóm c
   - `prop_c` = `hits_c / total_tokenizer`
   - `tfidf_c` = tổng trọng số tf-idf của các token thuộc nhóm c
3. Thêm: `net_polarity = prop_positive - prop_negative`,
   `coverage = (hits_pos + hits_neg) / total_tokenizer`.
4. `np.hstack([x_tfidf, x_lexicon])` **trong `run_cross_validation`**, sau
   `transform_tfidf`, trước `model.fit`. Tính hit trên train + val riêng
   (không fit gì → không rò rỉ, nhưng vẫn đặt trong fold cho nhất quán).
5. Thử 2 biến thể: (a) chỉ `prop_c`, (b) `prop_c` + `tfidf_c`.

**Rủi ro / lưu ý.** Với 152 dòng, 16 feature thêm vào 1.400 feature TF-IDF có
thể chìm nghỉm ở model tuyến tính; thử thêm biến thể **chỉ dùng 16 feature
lexicon** (bỏ TF-IDF) để xem trần của riêng khối này. NB (`MultinomialNB`)
cần input không âm — `net_polarity` âm được → để NB thì clip hoặc bỏ cột đó,
hoặc chuyển NB sang `ComplementNB` + `GaussianNB` cho khối lexicon.

**Kỳ vọng.** +1–3 macro-F1, chủ yếu ở lớp `negative`; giúp model tuyến tính
nhiều hơn RF.

---

## B. Xử lý phủ định

- [ ] **Trạng thái:** chưa làm — *ưu tiên #1 (đi kèm A)*

**Ý tưởng.** Lật / triệt tiêu hit **positive** khi có từ phủ định ngay trước.

**Căn cứ.** LM 2011, mục III: "simple negation = một trong
`{no, not, none, neither, never, nobody}` trong **3 từ trước** một từ
positive". LM **không** phủ định từ negative ("not terrible earnings" hầu như
không xuất hiện trong tin tài chính).

→ Cũng **giải thích lớp `positive` yếu**: LM, Tetlock, Engelberg đều thấy từ
positive "ít giá trị thông tin" vì hay bị phủ định. Một phần là bản chất, không
chỉ do ít data.

**Nguồn.** `News/Build_sentiment_label/Seed_set_Prepare/negation_cue_words.txt`.

**Cách làm.**
1. Khi tính `hits_positive` (mục A): với mỗi token positive tại vị trí `i`,
   nếu có cue phủ định trong `tokens[i-3:i]` thì **không đếm** (hoặc trừ vào
   `hits_negative`).
2. Tùy chọn: sinh unigram `NOT_<token>` cho token positive trong cửa sổ đó,
   cho vào vocab TF-IDF.
3. Đo riêng: có/không có bước phủ định.

**Kỳ vọng.** +0.5–2 F1 ở lớp `positive`.

---

## B'. POS-restricted n-gram + chuẩn hoá số

- [ ] **Trạng thái:** chưa làm — *ưu tiên #3*

**Ý tưởng.** (1) Chỉ giữ n-gram khớp mẫu POS mang sentiment, bỏ n-gram chứa
tên riêng. (2) Thay số / mã CK / % bằng placeholder trước khi tạo n-gram.

**Căn cứ.** Turney 2002, Table 1: chỉ trích cụm 2 từ khớp mẫu
adj+noun / adv+adj / adv+verb...; **tránh proper noun** để tên công ty không
rò rỉ vào phân loại. Thí nghiệm `experiment_vocab` cho thấy trigram df=2 gần
như toàn mảnh vụn số (`chiếm tỷ đồng`, `ghi_nhận âm tỷ`).

**Cách làm.**
1. VNCoreNLP có annotator `pos` — chạy lại tokenize GT kèm POS (mở rộng
   `prepare_ground_truth.py` để lấy cột POS từ corpus, **nếu** corpus
   vncorenlp có; nếu không, tokenize lại 152 bài với `annotators=["wseg","pos"]`).
2. `build_document_terms` mới: chỉ sinh n-gram khớp whitelist mẫu POS; drop
   token/gram có tag proper noun (`Np`).
3. Regex thay `\d[\d.,]*%?` → `<NUM>`, mã 3 ký tự in hoa → `<TICKER>` trước
   khi build n-gram.

**Rủi ro.** Cần POS tag cho 152 bài (chi phí một lần). Whitelist mẫu POS
tiếng Việt phải tự định nghĩa (không có sẵn như Penn Treebank).

**Kỳ vọng.** Vocab giảm ~40–60%, số CV gần như không đổi (như thí nghiệm
`experiment_vocab`), nhưng top-feature sạch hơn và ổn định hơn khi GT lớn.

---

## C. Chọn feature / trọng số có giám sát (per-fold)

- [ ] **Trạng thái:** chưa làm — *ưu tiên #3*

**Ý tưởng.** Thay bước chọn feature **không giám sát** hiện tại
(`select_top_features` xếp theo df + tổng weight) bằng bước **có giám sát**
(xếp theo mức liên quan với nhãn `y`), hoặc đổi trọng số term sang Delta-IDF.

**Căn cứ.**
- Yang & Pedersen 1997: chi² và information gain là bộ chọn feature tốt nhất
  cho text classification.
- Martineau & Finin 2009 "Delta TF-IDF": trọng số term theo độ phân biệt lớp,
  tăng sentiment trên data nhỏ.

**Cách làm.**
- Sửa `model/common.py::select_top_features`: thêm chế độ `method="chi2"` dùng
  `sklearn.feature_selection.chi2(x_train, y_train)` → giữ top-K theo p-value.
  Gọi **trong fold** với `x_train, y_train` (đã có sẵn ở
  `run_cross_validation`). Áp `selected_indices` lên `x_val` như hiện tại.
- Hoặc: sau `transform_tfidf`, nhân mỗi cột với Delta-IDF
  `|log((tp+0.5)/(fp+0.5)) − log((fn+0.5)/(tn+0.5))|` tính từ train fold.
- Giữ `MAX_FEATURES` như cap trên; thêm `K` (vd 300–800) cho chi².

**Rủi ro.** Với 121 dòng train/fold, ước lượng chi² per-term rất nhiễu → K
nhỏ dễ mất feature tốt do may rủi. Chỉ đáng làm khi GT ≥ vài trăm.

---

## C'. Feature SO-PMI cấp bài

- [ ] **Trạng thái:** chưa làm — *ưu tiên #3*

**Ý tưởng.** Thêm 2 cột: `mean_SO_PMI` (trung bình semantic orientation của
các cụm trong bài), `frac_positive_phrases`.

**Căn cứ.** Turney 2002: điểm bài = trung bình SO của các cụm. Bạn **đã** tính
SO-PMI trong `News/Build_sentiment_label/Lexicon_based/build_sentiment_dictionary_pmi.py`
— tái dùng bảng SO per-term/phrase, map vào 152 bài.

**Cách làm.** Load bảng SO đã có → với mỗi bài, tra SO của từng term khớp →
`mean`, `std`, `frac(SO>0)` → 3 cột ghép vào ma trận (mục A).

---

## D. Model & methodology

- [ ] **Trạng thái:** chưa làm — *ưu tiên #2 (rẻ, gần như free)*

| Việc | Cách làm | Căn cứ |
|---|---|---|
| **Nested-CV tuning** | Vòng CV trong: grid `C ∈ {0.01..10}` (LogReg/SVM), `alpha ∈ {0.1..2}` (NB), `min_samples_leaf`/`max_features` (RF). Vòng ngoài giữ nguyên 5-fold để báo cáo. Thêm hàm `tune_estimator(factory, param_grid, x_train, y_train)` gọi trong `run_cross_validation`. | Cawley & Talbot 2010: phải **nested** để không lạc quan hoá. |
| **ComplementNB** thay `MultinomialNB` | 1 dòng trong `model/naive_bayes.py::build_estimator`. | Rennie et al. 2003 — cho text mất cân bằng. |
| **Bỏ Platt calibration SVM** | `model/svm.py`: dùng `LinearSVC` + `decision_function`, hoặc bỏ hẳn SVM và chỉ dùng LogReg cho xác suất. | Platt trên inner-fold ~40 dòng rất nhiễu → nghi là lý do recall positive SVM sụp còn 0.20. |
| **Calibrate RF** | `CalibratedClassifierCV(rf, method="isotonic", cv=...)` nếu `sentiment_score_ml` dùng ở bước index. | Xác suất RF lệch calibration. |
| **Repeated stratified CV** | `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=...)`, báo cáo mean ± std. | 5-fold đơn trên 152 dòng → nhiễu ±0.02. |
| **McNemar / paired t-test** giữa 2 model | `statsmodels` McNemar trên out-of-fold predictions trước khi kết luận model nào thắng. | Dietterich 1998. |
| **Ensemble** | Trung bình xác suất (đã calibrate) của 4 model → argmax. Thêm `model/ensemble.py`. | Thường +1–2 macro-F1, gần như free. |

---

## E. Chất lượng nhãn

- [ ] **Trạng thái:** chưa làm — *cần trước khi kết luận model*

- **Inter-annotator agreement:** hiện chỉ 1 annotator (`annotator` = "1" trong
  `ground_truth_labeled.csv`). Cho annotator thứ 2 gán ~50 bài → **Cohen's κ**.
  Biết **trần con người**: nếu trần ~0.70 thì RF 0.63 đã sát trần, tinh chỉnh
  feature lời ít.
- **Anchoring bias:** CSV có `model_sentiment_label` / `model_sentiment_score`.
  Nếu annotator nhìn thấy dự đoán model cũ khi gán → nhãn bị kéo về phía đó.
  Kiểm tra setup Label Studio.
- **Neutral là "thùng rác"** (45%). Thử 2 tầng: (1) subjective vs
  factual/neutral, (2) pos vs neg trên phần subjective. (Wiebe, Hatzivassiloglou
  — subjectivity detection, Turney có trích.)
- **Ordinal:** neg < neutral < pos là thứ tự; lỗi neg↔pos tệ hơn neg↔neu.
  Ordinal logistic (Frank & Hall 2001) hoặc cost matrix trong loss.

---

## F. Weak / semi-supervised (không cần gán thêm tay)

- [ ] **Trạng thái:** chưa làm — *ưu tiên #5*

Nguyên liệu đã có: nhãn lexicon (`Lexicon_based`), weak label PhoBERT
(`Transfer_Learning`), 126k bài chưa nhãn (`equity_news_tokenized_vncorenlp.parquet`).

- **Self-training / EM:** train trên 152 → pseudo-label bài confidence cao
  trong 126k → train lại. Nigam et al. 2000 (NB + labeled/unlabeled cho text).
  Ngưỡng confidence + giữ tỉ lệ lớp để tránh confirmation bias.
- **Snorkel-style label model** (Ratner et al. 2017): gộp rule lexicon +
  PhoBERT + heuristic → tập train lớn có nhiễu → train ML trên đó, đánh giá
  trên 152.
- **Active learning** (Settles 2009): nếu có gán thêm, gán bài **uncertainty
  cao nhất** thay vì ngẫu nhiên → 2–3× F1 mỗi nhãn.

---

## Không nên tốn công

- Word embedding trung bình thành vector bài — không thêm ngữ cảnh (mất trật
  tự từ y như TF-IDF), trùng vai PhoBERT.
- XGBoost / LightGBM trên 152×1400 sparse — GBM hiếm khi thắng linear ở quy mô
  này.
- Char n-gram — tiếng Việt đã tách âm tiết sẵn.

---

## Thứ tự ưu tiên

1. **A + B** — feature 7 nhóm lexicon + phủ định. Nhiều lợi ích nhất, tái dùng
   `Seed_set_Prepare`, REF support mạnh (LM 2011, Tetlock 2007).
2. **D** — nested-CV tuning + ComplementNB + ensemble. Methodology, khả năng
   +2–4 F1, rẻ.
3. **B' + C + C'** — POS n-gram + placeholder số + chi² + SO-PMI. Xoá rác,
   nền tảng Turney 2002.
4. **E** — repeated CV + Cohen's κ. Để biết có đang sát trần không.
5. **F** — self-training / Snorkel. Tăng tín hiệu train mà không gán tay.

---

## Tham khảo (REF)

Trong `REF/`:

- **Loughran & McDonald (2011)**, "When Is a Liability Not a Liability? Textual
  Analysis, Dictionaries, and 10-Ks", *Journal of Finance*.
  `adg_cons2015_loughran-mcdonald-je-2011.pdf`.
  → công thức tf-idf (eq. 1), 7 nhóm từ điển tài chính, quy tắc phủ định,
  positive words yếu.
- **Turney (2002)**, "Thumbs Up or Thumbs Down? Semantic Orientation Applied to
  Unsupervised Classification of Reviews", *ACL*. `P02-1053.pdf`.
  → SO-PMI, mẫu POS trích cụm, tránh proper noun, điểm bài = trung bình SO.
- **Tetlock (2007)**, "Giving Content to Investor Sentiment: The Role of Media
  in the Stock Market", *Journal of Finance*. `Tetlock_Media_Sentiment_JF.pdf`.
  → PCA 77 category → pessimism factor; Negative + Weak-modal là 2 nhóm tải
  tín hiệu.
- **Allen, McAleer & Singh (2015)**, "Daily Market News Sentiment and Stock
  Prices". `15090.pdf`.
  → công thức gộp index có neutral ở mẫu số (liên quan bước `Build_sentiment_index`,
  không phải gán nhãn bài).
- Baker & Wurgler (cross-section), `wurgler_baker_cross_section.pdf` — index
  investor sentiment bằng PCA proxy (bước index, phương pháp #4).

Ngoài `REF/` (nên tra khi làm):

- Yang & Pedersen (1997), "A Comparative Study on Feature Selection in Text
  Categorization" — chi², IG.
- Martineau & Finin (2009), "Delta TFIDF: An Improved Feature Space for
  Sentiment Analysis".
- Rennie et al. (2003), "Tackling the Poor Assumptions of Naive Bayes Text
  Classifiers" — ComplementNB.
- Cawley & Talbot (2010), "On Over-fitting in Model Selection and Subsequent
  Selection Bias in Performance Evaluation" — nested CV.
- Dietterich (1998), "Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms" — McNemar.
- Nigam et al. (2000), "Text Classification from Labeled and Unlabeled
  Documents using EM".
- Ratner et al. (2017), "Snorkel: Rapid Training Data Creation with Weak
  Supervision".
- Settles (2009), "Active Learning Literature Survey".
- Frank & Hall (2001), "A Simple Approach to Ordinal Classification".
