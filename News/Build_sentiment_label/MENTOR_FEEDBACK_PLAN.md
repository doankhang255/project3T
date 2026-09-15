# Kế hoạch hành động sau feedback của mentor (Mike Nguyen)

Ngày ghi nhận feedback: 2026-09-05 (nhắn qua Slack). Tài liệu này áp lại 7 điểm
feedback thành việc cụ thể, theo từng nhánh (`Lexicon_based/`, `Traditional_ML/`,
`Transfer_Learning/`), để không làm tràn lan cùng lúc.

**Kết luận chung của mentor**: "Nhìn chung hướng làm đúng và tốt. Cứ tập trung
vào ground truth và kiểm định ở tầng chỉ số ngày, đừng tối ưu thêm hệ số trên
152 bài nữa."

---

## A. Sửa lỗi thống kê ở kết quả hiện có (điểm 1–2) — ĐÃ XONG tune/holdout

**Vấn đề mentor chỉ ra**: với n=152, sai số chuẩn của accuracy ~4 điểm %, nên
61,2% / 63,2% / 63,8% (các bước trong `Scoring/` và `Scoring_Intensity/`) về
mặt thống kê là **như nhau** — không nên kết luận "longest-match kém hơn"
hay "Cách 2 tốt hơn Cách 1". Nặng hơn: 125 tổ hợp hệ số (Cách 2),
`NEGATION_WINDOW`, neutral margin — đều được **chọn trên chính 152 bài dùng
để báo cáo kết quả** → 63,2% là con số in-sample, không phải ước lượng
khách quan.

### Đã làm

Ground truth hiện đã tăng lên **599 bài** (`ground_truth_combined.csv` =
`ground_truth_labeled.csv` 152 + `ground_truth_news.csv` 447, xác nhận 0
trùng `source_row_id`). Chia cố định (`data/ground_truth_tune_holdout_split.csv`,
stratified theo sentiment, `random_state=42`): **Tune 419 bài / Holdout 180
bài**. Dò lại TOÀN BỘ tham số từng bị chọn in-sample, đúng quy trình
tune/holdout (chọn trên Tune, chấm Holdout ĐÚNG 1 LẦN):

| Tham số | Tốt nhất trên Tune | Trên Holdout (tốt nhất vs hiện tại) | Kết luận |
|---|---|---|---|
| `negation_window` — Cách 1 | window=10 (61,6%) | 58,33% vs **58,89%** (window=4) | Giữ =4 |
| `negation_window` — Cách 2 | window=10 (61,6%) | 57,78% vs **58,89%** (window=4) | Giữ =4 |
| 125 tổ hợp hệ số — Cách 2 | production/marker=0/scale=0 (61,1%) | 55,56% = **55,56%** (bằng nhau) | Giữ production |
| `margin` (Neutral) — Cách 1 | 0,0037 (64,2%) | 57,22% vs **58,89%** (margin=0) | Giữ =0 (Cách 3 gốc) |
| `margin` (Neutral) — Cách 2 | 0,0324 (64,2%) | 57,22% vs **58,89%** (margin=0) | Giữ =0 (Cách 3 gốc) |

**Kết quả**: KHÔNG tham số nào cần đổi - mọi giá trị hiện tại đều thắng hoặc
bằng phương án "tối ưu" tìm được qua grid search khi kiểm chứng khách quan.
Mẫu hình lặp lại rất nhất quán qua cả 5 lần: bất kỳ tham số nào "làm đẹp" số
trên Tune đều MẤT ĐIỂM trên Holdout - bằng chứng thực nghiệm trực tiếp,
nhiều lần cho đúng cảnh báo của mentor.

Script: `Scoring/tune_negation_window.py`, `Scoring_Intensity/tune_negation_window.py`,
`Scoring/tune_neutral_margin.py`, `Scoring_Intensity/tune_neutral_margin.py`
(viết lại), `Scoring_Intensity/tune_intensity_coefficients.py` (viết lại).

### Việc còn lại

- [ ] Thêm caveat sai số chuẩn vào `METHODOLOGY_SUMMARY.qmd` mỗi chỗ so
      sánh % giữa các bước (giờ n=599, sai số chuẩn accuracy còn ~±2 điểm %,
      vẫn không nhỏ).
- [ ] Cân nhắc báo cáo lại accuracy chính thức bằng con số Holdout (58,89%)
      thay vì accuracy trên toàn bộ Tune+Holdout gộp, để nhất quán với tinh
      thần "không dùng cùng 1 tập để vừa tune vừa báo cáo" (dù ở đây không
      có gì bị tune thêm nữa, accuracy toàn tập 599 bài vẫn hợp lý dùng làm
      con số chính thức).

**Trạng thái**: tune/holdout cho toàn bộ tham số scoring-time đã xong.

---

## B. Mở rộng ground truth (điểm 3) — cần phối hợp người khác

**Mục tiêu**: 800–1000 bài (hiện có 152).

- [ ] Lấy mẫu **có chiến lược** thay vì ngẫu nhiên: phân tầng theo thời gian
      + theo ngành.
- [ ] Thêm active learning: ưu tiên gán các bài model đang "phân vân" nhất
      (VD `|positive_score − negative_score|` nhỏ, hoặc không match từ nào).
- [ ] Tuyển ≥2 người gán độc lập trên phần chồng lấn (mentor khuyến khích 4
      người), tính **Cohen's kappa** — biết trần (ceiling) bài toán là bao
      nhiêu. Nếu người gán chỉ đồng thuận ~70%, coi như accuracy 63% hiện tại
      đã gần trần.

**Trạng thái**: cần tự sắp xếp với Hải/Mến/lab mate. Có thể quay lại nhờ hỗ
trợ code (script lấy mẫu phân tầng, script tính kappa) khi đến bước đó.

---

## C. Kiểm định cấp-ngày kiểu Tetlock (điểm 4) — ĐÃ XONG bản đầu, xem kết luận

**Ý tưởng**: không cần chờ ground truth lớn — sai số cấp-bài triệt tiêu bớt
khi tổng hợp theo ngày trên toàn bộ 126.576 bài, rồi hồi quy với lợi suất
VN-Index (thiết kế Tetlock 2007, có kiểm soát lợi suất trễ + khối lượng giao
dịch). Nếu chỉ số ngày có sức giải thích, phương pháp coi như hoạt động dù
accuracy cấp-bài mới 63%. Đây cũng là cách so Cách 1 vs Cách 2 khách quan
hơn (cỡ mẫu lớn hơn nhiều so với 152 bài).

**Cấu trúc thư mục hiện tại** (đọc lại `REF/Tetlock_Media_Sentiment_JF.pdf`
để đối chiếu thiết kế trước khi làm — xem docstring từng file):
- `News/Build_sentiment_index/` — xây chỉ số sentiment theo kỳ (ngày/tuần),
  đọc thẳng `Lexicon_based/Scoring*/data/article_scores*.parquet` (không cần
  `predicted_label`, chỉ cần `net_sentiment_score`):
  - `build_sentiment_index_daily.py` / `build_sentiment_index_weekly.py` — trung bình đơn giản positive-negative (Cách 1 PMI, Cách 2 intensity).
  - `build_sentiment_index_pca.py` — **chỉ số PCA kiểu Tetlock thật** trên 7 category LM (xem mục "PCA factor" bên dưới).
- `News_Vnindex/Common/` — hàm merge dùng chung cho mọi method (`merge_vnindex_daily_with_sentiment.py` có `map_to_next_trading_date` map tin cuối tuần/lễ vào phiên kế tiếp; `merge_vnindex_weekly_with_sentiment.py` tương ứng cấp tuần).
- `News_Vnindex/Lexicon/daily/` — pipeline cấp ngày (tự chứa, không import từ `weekly/`): `build_vnindex_daily_abnormal_return.py`, `vnindex_daily_predictive_regression.py` (multi-lag + dummy `near_tet` + Newey-West), cộng các script chẩn đoán (`check_predictor_multicollinearity.py`, `visualize_coefficient_covariance.py`, `plot_sentiment_lag_coefficients.py`, `plot_ols_fit_vs_actual.py`, `verify_abnormal_return.py`, `diagnose_model_r_squared.py`, `vnindex_daily_predictive_regression_covid_control.py`).
- `News_Vnindex/Lexicon/weekly/` — pipeline cấp tuần (bản đơn giản 1-lag có sẵn từ trước, đã sửa path cho khớp Lexicon_based mới): `build_vnindex_weekly_return.py`, `build_vnindex_weekly_abnormal_return.py`, `vnindex_weekly_predictive_regression.py`.
- 4 method test song song ở cả 2 tần suất: `pmi`, `intensity`, `pca_pmi`, `pca_intensity`.

### Kết quả DAILY (multi-lag, Newey-West, đã dọn VIF/near_tet/volatility-lag)

**Không tìm thấy bằng chứng vững chắc** ở cả 4 method. Lag1 (tức thời) không
có ý nghĩa (p 0,25–0,64). Kiểm định tổng lag2-5 (đảo chiều) không có ý nghĩa
(p 0,21–0,86). Lag4 có ý nghĩa riêng lẻ (p<0,02) ở PMI/Intensity nhưng **biến
mất hoàn toàn với PCA_PMI/PCA_Intensity** (p 0,75–0,98) — bằng chứng khá rõ
lag4 là nhiễu multiple-testing, không phải tín hiệu thật (nếu thật, phải
xuất hiện lại với chỉ số sentiment khác). Robustness check thêm dummy
COVID-19 (23/1–30/4/2020) không đổi kết luận. R² toàn mô hình chỉ 1,36% —
đã tách theo nhóm biến (`diagnose_model_r_squared.py`): phần lớn (73%) đến
từ dummy mùa vụ/volatility, **không phải từ sentiment lẫn lịch sử return** —
cho thấy return cấp ngày VN-Index nói chung rất khó dự đoán từ bất kỳ nguồn
nào ở đây, không riêng sentiment.

### Kết quả WEEKLY — phát hiện đáng chú ý nhất

Ở horizon **4 tuần**, `future_ret_4w` (return thô, chưa điều chỉnh abnormal)
cho **p ≈ 0,05–0,06 nhất quán ở cả 4 method độc lập** (PMI 0,057, Intensity
0,051, PCA_PMI 0,052, PCA_Intensity 0,054) — mức nhất quán chưa từng thấy ở
các kiểm định khác trong toàn bộ quá trình (so với lag4 ở daily chỉ xuất
hiện 2/4 method). R² cũng cao hơn hẳn daily (~2% so với 1,36%). Tuy nhiên 2
biến thể abnormal-return ở 4 tuần (`future_abnormal_rolling_ret_4w`,
`future_abnormal_ar1_ret_4w`) chỉ có ý nghĩa borderline ở PMI/Intensity
(p 0,057–0,081) và **mất ý nghĩa với PCA** (p 0,18–0,28) — nên "future_ret_4w
thô" nhất quán nhưng chưa rõ có do sentiment thật hay do 1 phần chưa lọc hết
(bản weekly hiện tại vẫn là bản đơn giản 1-lag, chưa nâng cấp multi-lag +
dummy mùa vụ như bản daily).

### PCA factor (kiểu Tetlock thật, thay trung bình đơn giản)

`build_sentiment_index_pca.py` — PCA trên 7 category LM (thay vì chỉ
positive-negative). **Phát hiện quan trọng**: component 1 có `positive_score`
VÀ `negative_score` cùng hệ số DƯƠNG (không đối lập như Tetlock, nơi Positive
load âm, Negative load dương) — component 1 ở đây giống chỉ số "mật độ ngôn
từ tài chính nói chung" (weak_modal/strong_modal chi phối mạnh nhất) hơn là
trục "bi quan-lạc quan" sạch. Giải thích được 25-43% phương sai 7 category
(tùy ngày/tuần). Vẫn đưa vào test thực nghiệm — kết quả xem 2 mục trên.

### Weekly multi-lag (nâng cấp, `vnindex_weekly_predictive_regression_multilag.py`)

Nâng cấp weekly lên đúng đặc tả daily (5-lag sentiment/target + 1 lag volume
+ dummy `near_tet` + `volatility_12w` đã sửa `.shift(1)` tránh look-ahead,
Newey-West, kiểm định tổng lag2-5) — import lại `newey_west_covariance` từ
bản đơn giản CÙNG thư mục `weekly/` (không xuyên `daily/`↔`weekly/`).

**Kết quả: tín hiệu "future_ret_4w nhất quán 4 method" từ bản đơn giản
KHÔNG được xác nhận lại.** PMI/Intensity: không còn gì (mọi p 0,18–0,87).
PCA_PMI/PCA_Intensity: lag2 âm + lag3 dương có ý nghĩa/borderline riêng lẻ
(p 0,02–0,07) nhưng gần như **triệt tiêu lẫn nhau** khi cộng lại → kiểm
định tổng lag2-5 không có ý nghĩa (p 0,68–0,85). Pattern chỉ xuất hiện với
PCA, không với PMI/Intensity — cùng dấu hiệu kém tin cậy đã thấy với lag4 ở
daily (không nhất quán qua các cách tính sentiment khác nhau).

### Kết luận tổng thể cho mentor (sau khi thử daily rigorous + weekly simple
### + weekly multi-lag + PCA factor — 4 hướng kiểm định độc lập)

**Chưa tìm được bằng chứng vững chắc, nhất quán về liên hệ sentiment ↔ lợi
suất VN-Index** ở bất kỳ tần suất/cách tính nào khi kiểm tra đủ nghiêm ngặt
(multi-lag + dummy mùa vụ + Newey-West + nhiều cách tính sentiện độc lập).
Mọi tín hiệu "có vẻ có ý nghĩa" phát hiện dọc đường (lag4 daily, future_ret_4w
weekly đơn giản, lag2/3 weekly PCA) đều **biến mất hoặc không nhất quán**
khi kiểm tra chéo bằng cách tính sentiment khác — đúng dấu hiệu nhiễu do
kiểm định nhiều lần (đã chạy tổng cộng hàng trăm phép kiểm định qua cả quá
trình), không phải tín hiệu kinh tế thật ổn định.

**Điều này KHÔNG có nghĩa phương pháp Lexicon-Based thất bại** — chỉ có
nghĩa dữ liệu/kiểm định hiện tại (ground truth 152 bài, corpus tin thưa
2010-2013, sentiment cấp-vĩ-mô có thể bị pha loãng bởi tin công ty đơn lẻ)
chưa đủ mạnh để phát hiện tín hiệu, nếu có. Việc cần làm tiếp theo quan
trọng nhất vẫn là **mục B (ground truth 800-1000 bài)** — không nên đầu tư
thêm vào việc dò thêm biến thể hồi quy nữa cho tới khi có ground truth lớn
hơn để biết chính accuracy cấp-bài đang ở đâu so với trần con người.

- [x] Viết script mới: tổng hợp `article_scores_labeled.parquet` (Cách 1) và
      `article_scores_intensity_labeled.parquet` (Cách 2) theo ngày → sentiment
      index hàng ngày kiểu mới, KHÔNG đụng vào `market_sentiment_index_daily.parquet`
      cũ (folder mới `Lexicon_based/Market_Validation/`, self-contained).
- [x] Merge với `vnindex_eda_output.csv` theo ngày (3.468/5.037 ngày khớp
      được giao dịch thật).
- [x] Hồi quy lợi suất VN-Index ~ sentiment ngày (Newey-West HAC, import
      thẳng hàm từ `News_Vnindex/vnindex_weekly_predictive_regression.py`,
      không viết lại) — đã đọc lại `REF/Tetlock_Media_Sentiment_JF.pdf` để
      đối chiếu thiết kế, xem `Market_Validation/README.md` mục "Khác biệt".
- [x] Dùng kết quả này so sánh Cách 1 vs Cách 2 — **kết quả lần chạy đầu
      (bản đơn giản, 1 lag) chưa có ý nghĩa thống kê ở cả 4 tổ hợp** (p >
      0,18) — chưa kết luận được gì, cần nâng cấp lên multi-lag + dummy
      mùa vụ (đúng bản gốc Tetlock) trước khi kết luận không có tín hiệu.

**Trạng thái**: pipeline 4 bước đã chạy xong, ĐÃ nâng cấp lên bản multi-lag +
dummy mùa vụ (đúng đặc tả Tetlock hơn). Kết quả: **vẫn không tìm thấy bằng
chứng vững chắc** về liên hệ sentiment ↔ lợi suất VN-Index cấp ngày (lag 1
không có ý nghĩa, kiểm định tổng lag2-5 "đảo chiều" không có ý nghĩa ở cả 2
cách; xem `Market_Validation/README.md` để biết chi tiết + 4 giới hạn còn
lại chưa khắc phục — đặc biệt điểm 1: inner-join làm méo nhẹ cấu trúc lag ở
~9,4% ngày). Chưa nên kết luận "phương pháp không có tín hiệu" — cần khắc
phục giới hạn 1 (dùng lịch giao dịch liên tục thay vì inner-join) và/hoặc
chia mẫu theo giai đoạn trước khi kết luận chắc chắn.

(Ghi chú: `News/Build_sentiment_index/build_sentiment_index_daily.py` đã có
sẵn từ trước — sửa lại đường dẫn input + generalize đọc 2 nguồn Lexicon_based/
thay vì tạo file mới trùng lặp trong `Market_Validation/`.)

---

## D. Nhánh Traditional_ML (điểm 5) — `Traditional_ML/`

- [ ] Thêm ComplementNB.
- [ ] Thêm kiểm định McNemar so 2 model.
- [ ] Báo cáo repeated CV kèm độ lệch chuẩn thay vì 1 con số đơn lẻ; chưa
      chốt Random Forest là lựa chọn cuối cùng (mô hình tuyến tính có thể
      vượt lên khi dữ liệu tăng).

**Trạng thái**: chưa động vào trong phiên làm việc này — cần đọc lại
`run_pipeline.py`/`verify_pipeline.py` (xem [[traditional-ml-pipeline]]) trước
khi sửa.

---

## E. Nhánh Transfer_Learning / PhoBERT (điểm 6) — `Transfer_Learning/`

Thứ tự mentor đề xuất (không nhảy thẳng vào fine-tune):

1. [ ] **Domain-adaptive pretraining**: tiếp tục pretrain PhoBERT bằng MLM
       trên 126k bài **chưa nhãn** — không cần ground truth, chạy nền được
       ngay.
2. [ ] **Frozen-feature extraction**: PhoBERT đóng băng trích đặc trưng +
       logistic regression trên top — ổn định hơn fine-tune toàn bộ khi còn
       ít nhãn.
3. [ ] Chỉ fine-tune toàn phần khi có vài trăm nhãn trở lên (chờ kết quả B).

**Trạng thái**: chưa bắt đầu.

---

## F. Nhãn Neutral mập mờ (điểm 7) — ĐÃ XONG chẩn đoán + Mức 1, 2

Mentor: *"Luật gán nhãn hiện tại cho Neutral khi positive_score đúng bằng
negative_score. Trường hợp hai điểm khác 0 nhưng bằng nhau và trường hợp bài
hoàn toàn không có từ nào khớp là hai tình huống rất khác nhau, nên tách ra
để xem tỷ lệ mỗi loại."*

### Chẩn đoán (`diagnose_neutral_split.py`)

Tách `Neutral` thành `Neutral_balanced` (positive_score = negative_score > 0
— có tín hiệu, hòa nhau) và `Neutral_no_match` (positive_score =
negative_score = 0 — không khớp từ nào, đoán mặc định):

- Trên toàn corpus 126.576 bài: `Neutral_no_match` chiếm **~95% số bài bị gán
  Neutral** (ban đầu 44,1% tổng corpus, sau Mức 2 giảm còn 41,3%).
  `Neutral_balanced` chỉ 1,6-2,2% — không đáng kể.
- Trên ground truth: chỉ ~55-70% bài `Neutral_no_match` thực sự là Neutral
  thật (phần còn lại là Positive/Negative bị bỏ sót vì không match được từ
  nào) → **độ phủ từ điển (coverage) là vấn đề lớn hơn hướng chấm điểm.**

Kết luận: 3 mức xử lý, ưu tiên từ rẻ/an toàn nhất:
- **Mức 1** (thống kê, không đổi logic): báo cáo riêng tỷ lệ 2 loại Neutral
  thay vì gộp chung — đã có sẵn qua `diagnose_neutral_split.py`.
- **Mức 2** (sửa gốc — mở rộng độ phủ): xem bên dưới, **ĐÃ LÀM**.
- **Mức 3** (đổi luật gán nhãn khi không match): chưa làm, xem "Việc còn lại".

### Mức 2 — mở rộng độ phủ: ĐÃ LÀM, 2 phần

**(a) Sửa bug tính `ngram_n`** (`Scoring/build_weighted_dictionary.py`,
`Scoring_Intensity/build_intensity_dictionary.py`): code cũ tính
`ngram_n = term.count(" ") + 1` — đếm DẤU CÁCH, nhưng seed nối từ ghép bằng
GẠCH DƯỚI theo quy ước VNCoreNLP. Với cụm KHÔNG phải compound thật của
VNCoreNLP (VD `tăng_mạnh`, `giảm_mạnh`, `có_lãi`, `nợ_xấu` — tokenizer luôn
tách thành 2 token rời `tăng`+`mạnh`, không gộp), bug này gán nhầm
`ngram_n=1` khiến term **không bao giờ khớp được** (0 lần trên 126.576 bài,
đã kiểm chứng). Sửa: kiểm tra thật với tập token của corpus, term nào không
tồn tại như 1 token literal thì tách lại theo `_` và nối bằng dấu cách (đúng
định dạng candidate mà `score_sentence()` dùng khi tra n-gram > 1).

Kết quả đo được (chỉ sửa bug, không thêm/bớt từ nào):
| | Trước | Sau |
|---|---:|---:|
| Term "chết" (0 match/126.576 bài) | 209/1729 (12,1%) | 141/1729 (8,2%) |
| Neutral_no_match / toàn corpus | 44,1% | 41,3% |
| Accuracy Cách 1 PMI / new_250 (OOS) | 55,2% | 56,8% |
| Accuracy Cách 2 Intensity / new_250 (OOS) | 54,8% | 56,4% |

Cả 2 tập (in-sample lẫn out-of-sample) đều tăng → cải thiện thật, không phải
in-sample overfit (khác hẳn phép thử khôi phục 37 từ nhiễu ở mục A, nơi
old_152 tăng nhưng new_250 đứng yên/giảm nhẹ).

**(b) Đào ứng viên từ còn thiếu** (`diagnose_missing_terms.py`, output
`data/missing_term_candidates.csv`): lấy đúng các bài Neutral_no_match nhưng
nhãn thật là Positive/Negative trong ground truth, đếm tần suất từ CHƯA có
trong dictionary. Kết quả: phần lớn bài bị bỏ sót còn lại là **tin công bố
sự kiện thuần túy** (trả cổ tức, lãnh đạo/cổ đông lớn đăng ký mua/bán cổ
phiếu...) — sentiment được suy ra từ KIẾN THỨC TÀI CHÍNH (nhận cổ tức = tốt,
lãnh đạo bán ra = tín hiệu xấu), không nằm ở từ ngữ trong bài. Ứng viên đào
được (`cổ_đông`, `cổ_tức`, `bán`, `mua`, `đăng_ký`...) đều là danh từ thực
thể/từ chức năng/động từ giao dịch trung tính — rủi ro lặp lại lỗi
over-predict như `tăng`/`giảm` nếu thêm cứng. Người dùng đã tự review toàn
bộ danh sách và chỉ thêm 2 từ qua kiểm chứng riêng lẻ: `thấp` (mới, negative)
và `phát_triển` (negative→dùng lại 1 mình, positive - trước đó nằm trong 37
từ bị revert ở mục A nhưng test riêng lẻ cho kết quả khác, không regression).
Kết quả: Cách 2 Intensity tăng thêm ~0,5 điểm % (cả 2 tập), Cách 1 PMI không
đổi, không có regression ở tập nào.

**Kết luận Mức 2**: đã khai thác gần hết dư địa an toàn (sửa bug + vài từ
kiểm chứng riêng lẻ). Phần no-match còn lại (~41%) chủ yếu là giới hạn cấu
trúc của lexicon (không suy luận được "ai làm gì, theo hướng nào" từ tin sự
kiện) — nên ghi nhận là giới hạn đã biết, không cố nhét thêm từ đại trà.

### Việc còn lại

- [ ] **Mức 3** (đổi luật gán nhãn khi `Neutral_no_match`): cân nhắc coi
      "không khớp từ nào" là **không dự đoán được** (bỏ qua khi tính
      accuracy) thay vì mặc định Neutral — hiện đang default-Neutral nên
      pha loãng độ chính xác 1 cách không công bằng cho ~41% bài.
- [ ] Áp Mức 1 (báo cáo tách riêng 2 loại Neutral) vào
      `Scoring/classify_and_evaluate.py` chính thức (hiện chỉ có ở script
      chẩn đoán riêng).

**Trạng thái**: chẩn đoán + Mức 2 xong, có số liệu; Mức 1/3 còn lại là việc
nhỏ.

---

## Thứ tự ưu tiên đề xuất

1. **B** (ground truth) — mentor gọi là ưu tiên số 1; cần thời gian + người,
   nên khởi động sớm và chạy song song nền trong lúc làm các mục khác.
2. **C** (kiểm định cấp-ngày) — không cần chờ B, tận dụng hạ tầng có sẵn,
   cho tín hiệu đáng tin ngay ở cỡ mẫu lớn.
3. **A** (sửa thống kê/tune-holdout) — nhanh, nên làm **trước khi báo cáo
   thêm bất kỳ con số accuracy nào mới** cho mentor.
4. **F** (chi tiết nhỏ) — làm tiện tay lúc sửa A.
5. **D, E** (ML / PhoBERT) — mentor xếp ưu tiên 2, làm khi rảnh tay từ nhánh
   Lexicon-Based.
