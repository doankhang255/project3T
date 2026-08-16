from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
CATEGORIES_DIR = SCRIPT_DIR / "data" / "categories"
VI_SEEDS_DIR = SCRIPT_DIR / "vi_seeds"

# Moi entry: cum tu tieng Viet mang nghia tai chinh -> danh sach tu goc tieng
# Anh trong Master Dictionary (LM) ma no dich/gop lai. Duoc chon thu cong tu
# top-150-theo-Doc-Count cua tung danh muc, bo qua tu noi phap ly co (HEREIN,
# THEREOF, WHEREAS, HEREBY...) vi dich rieng le khong mang nghia tai chinh,
# va gop cac bien the cua cung 1 goc tu (VD REQUIRE/REQUIRES/REQUIRED) vao
# chung 1 seed Viet de tranh trung lap.
UNCERTAINTY_MAP: dict[str, list[str]] = {
    "xấp_xỉ": ["APPROXIMATELY", "APPROXIMATE", "APPROXIMATES", "APPROXIMATED", "APPROXIMATING", "ROUGHLY"],
    "rủi_ro": ["RISK", "RISKS"],
    "rủi_ro_cao": ["RISKY"],
    "cho_rằng": ["BELIEVE", "BELIEVES", "BELIEVED"],
    "khác_biệt_so_với_dự_kiến": ["DIFFER", "DIFFERS", "DIFFERING", "DIFFERED"],
    "giả_định": ["ASSUMPTIONS", "ASSUMPTION", "ASSUME", "ASSUMED", "ASSUMING", "ASSUMES"],
    "không_chắc_chắn": ["UNCERTAINTIES", "UNCERTAINTY", "UNCERTAIN"],
    "dự_kiến": ["ANTICIPATED", "ANTICIPATE", "ANTICIPATES", "ANTICIPATION", "ANTICIPATING"],
    "có_khả_năng_xảy_ra": ["POSSIBLE", "POSSIBILITY", "POSSIBILITIES", "PROBABLE", "PROBABILITY", "PROBABILITIES", "PROBABLY"],
    "tình_huống_bất_ngờ": ["CONTINGENCIES", "CONTINGENCY", "CONTINGENT", "CONTINGENTLY"],
    "đang_chờ_xử_lý": ["PENDING", "UNSETTLED"],
    "phụ_thuộc": ["DEPENDENT", "DEPENDENCE", "DEPENDENCY"],
    "biến_động": ["FLUCTUATIONS", "FLUCTUATE", "FLUCTUATION", "FLUCTUATING", "FLUCTUATED", "FLUCTUATES"],
    "mức_độ_rủi_ro_tiếp_xúc": ["EXPOSURE", "EXPOSURES"],
    "biến_động_mạnh": ["VOLATILITY", "VOLATILE", "VOLATILITIES"],
    "thay_đổi_thất_thường": ["VARIABLE", "VARY", "VARYING", "VARIES", "VARIED", "VARIATION", "VARIATIONS", "VARIABILITY", "VARIANCE", "VARIANCES", "VARIANTS"],
    "điều_chỉnh_lại": ["REVISED", "REVISE"],
    "dự_báo": ["PREDICT", "PREDICTED", "PREDICTS", "PREDICTING", "PREDICTION", "PREDICTIONS", "PREDICTIVE"],
    "khó_dự_đoán": ["UNPREDICTABLE", "UNPREDICTABILITY"],
    "còn_nghi_vấn": ["DOUBTFUL", "DOUBT", "DOUBTS"],
    "khả_năng_xảy_ra": ["LIKELIHOOD"],
    "chưa_xác_định": ["UNKNOWN", "UNIDENTIFIED", "UNSPECIFIED", "UNDESIGNATED"],
    "tài_sản_vô_hình": ["INTANGIBLE", "INTANGIBLES"],
    "sơ_bộ": ["PRELIMINARY", "PRELIMINARILY"],
    "chưa_được_quan_sát": ["UNOBSERVABLE"],
    "không_xác_định_thời_hạn": ["INDEFINITE", "INDEFINITELY"],
    "bất_ngờ": ["UNEXPECTED", "UNEXPECTEDLY", "SUDDEN"],
    "mang_tính_suy_đoán": ["SPECULATIVE", "SPECULATION"],
    "bất_thường": ["UNUSUAL", "UNUSUALLY"],
    "có_điều_kiện": ["CONDITIONAL", "CONDITIONALLY"],
    "bất_ổn": ["INSTABILITY"],
    "cần_làm_rõ": ["CLARIFICATION", "UNCLEAR"],
    "biện_pháp_phòng_ngừa": ["PRECAUTIONS"],
    "đánh_giá_lại": ["REASSESS", "REASSESSED", "REASSESSMENT", "RECONSIDER"],
    "ngoài_kế_hoạch": ["UNPLANNED"],
    "chưa_được_chứng_minh": ["UNPROVEN", "UNPROVED"],
    "không_thể_xác_định": ["INDETERMINATE"],
    "sai_lệch_so_với_kế_hoạch": ["DEVIATION", "DEVIATIONS", "DEVIATE"],
    "mơ_hồ": ["AMBIGUITY", "AMBIGUITIES"],
    "được_cho_là": ["PRESUMED", "PRESUMPTION"],
    "ít_có_khả_năng": ["IMPROBABLE"],
    "tạm_thời_chưa_chốt": ["TENTATIVE"],
    "không_thể_đánh_giá_được": ["NONASSESSABLE"],
    "thay_đổi_điều_khoản": ["ALTERATION", "ALTERATIONS"],
}

LITIGIOUS_MAP: dict[str, list[str]] = {
    "thẩm_quyền_pháp_lý": ["JURISDICTION", "JURISDICTIONS"],
    "pháp_lý": ["LEGAL", "LEGALLY"],
    "hợp_pháp": ["LAWFUL", "LAWFULLY"],
    "bất_hợp_pháp": ["UNLAWFUL"],
    "sửa_đổi": ["AMENDED", "AMENDMENT", "AMENDMENTS", "AMEND", "AMENDS", "AMENDING"],
    "quy_định": ["REGULATION", "REGULATIONS"],
    "cơ_quan_quản_lý": ["REGULATORY", "REGULATED", "REGULATE", "REGULATING", "REGULATES", "REGULATORS"],
    "luật_pháp": ["LAWS", "LAW"],
    "quy_định_theo_luật_định": ["STATUTORY", "STATUTE", "STATUTES"],
    "lập_pháp": ["LEGISLATION", "LEGISLATIVE"],
    "kiện_tụng": ["LITIGATION"],
    "vụ_kiện": ["LAWSUIT", "LAWSUITS"],
    "hợp_đồng": ["CONTRACTS", "CONTRACT", "CONTRACTUAL", "CONTRACTUALLY", "CONTRACTED", "CONTRACTING"],
    "dàn_xếp_giải_quyết": ["SETTLEMENT", "SETTLEMENTS"],
    "tòa_án": ["COURT", "COURTS"],
    "tư_pháp": ["JUDICIAL", "JUSTICE"],
    "khiếu_nại": ["CLAIMS", "CLAIM"],
    "chấp_thuận": ["CONSENT", "CONSENTS", "CONSENTED"],
    "luật_sư": ["COUNSEL", "ATTORNEY", "ATTORNEYS"],
    "vi_phạm_hợp_đồng": ["BREACH", "BREACHES", "BREACHED"],
    "ban_hành": ["PROMULGATED"],
    "bồi_thường_thiệt_hại": ["INDEMNIFICATION", "INDEMNIFY", "INDEMNIFIED", "INDEMNITY", "INDEMNITIES"],
    "trợ_cấp_thôi_việc": ["SEVERANCE"],
    "cáo_buộc": ["ALLEGED", "ALLEGING", "ALLEGATIONS", "ALLEGES", "ALLEGEDLY", "ALLEGE"],
    "làm_chứng": ["WITNESS"],
    "kháng_cáo": ["APPEAL", "APPEALS", "APPEALED"],
    "được_diễn_giải_theo": ["CONSTRUED"],
    "khắc_phục_hậu_quả_môi_trường": ["REMEDIATION", "REMEDIATE"],
    "mang_tính_bồi_thường": ["COMPENSATORY"],
    "có_hiệu_lực_thi_hành": ["ENFORCEABLE", "ENFORCEABILITY"],
    "không_thể_thi_hành": ["UNENFORCEABLE", "UNENFORCEABILITY"],
    "hình_sự": ["CRIMINAL"],
    "thay_thế_điều_khoản_trước": ["SUPERSEDES", "SUPERSEDE", "SUPERSEDED"],
    "quyền_truy_đòi": ["RECOURSE"],
    "bị_đơn": ["DEFENDANTS", "DEFENDANT"],
    "trọng_tài": ["ARBITRATION", "ARBITRATOR"],
    "nguyên_đơn": ["PLAINTIFFS", "PLAINTIFF"],
    "đơn_kiện": ["PETITION"],
    "không_thể_hủy_ngang": ["IRREVOCABLE", "IRREVOCABLY"],
    "bên_có_nghĩa_vụ_thanh_toán": ["OBLIGOR"],
    "tài_sản_bị_ràng_buộc": ["ENCUMBRANCES", "ENCUMBRANCE"],
    "bảo_lãnh": ["SURETY"],
    "lệnh_cấm": ["INJUNCTIVE", "INJUNCTION", "INJUNCTIONS"],
    "cố_ý_vi_phạm": ["WILLFUL"],
    "phán_quyết": ["RULING", "RULINGS"],
    "gây_bất_lợi": ["PREJUDICE"],
    "điều_khoản_có_thể_tách_biệt": ["SEVERABILITY"],
    "sắc_lệnh": ["DECREE"],
    "thu_hồi": ["REVOCATION"],
    "bồi_thẩm_đoàn": ["JURY"],
    "hành_vi_gây_thiệt_hại": ["TORT"],
    "hủy_bỏ_hợp_đồng": ["RESCISSION"],
    "thừa_nhận": ["ADMISSION"],
}

STRONG_MODAL_MAP: dict[str, list[str]] = {
    "chắc_chắn_sẽ": ["WILL"],
    "bắt_buộc_phải": ["MUST"],
    "tốt_nhất": ["BEST"],
    "cao_nhất": ["HIGHEST"],
    "thấp_nhất": ["LOWEST"],
    "không_bao_giờ": ["NEVER"],
    "luôn_luôn": ["ALWAYS"],
    "rõ_ràng": ["CLEARLY"],
    "mạnh_mẽ": ["STRONGLY"],
    "không_thể_tranh_cãi": ["UNDISPUTED"],
    "dứt_khoát": ["DEFINITIVELY"],
    "vô_song": ["UNPARALLELED"],
    "rõ_ràng_không_mập_mờ": ["UNAMBIGUOUSLY"],
    "chắc_chắn": ["DEFINITELY"],
    "không_nghi_ngờ_gì": ["UNDOUBTEDLY"],
    "chưa_từng_bị_vượt_qua": ["UNSURPASSED"],
    "kiên_định_không_khoan_nhượng": ["UNCOMPROMISING"],
    "minh_bạch_tuyệt_đối": ["UNEQUIVOCALLY", "UNEQUIVOCAL"],
}

WEAK_MODAL_MAP: dict[str, list[str]] = {
    "có_thể": ["MAY", "POSSIBLY", "MAYBE"],
    "có_thể_sẽ": ["COULD"],
    "có_lẽ_sẽ": ["MIGHT"],
    "có_khả_năng_xảy_ra": ["POSSIBLE", "CONCEIVABLE"],
    "tùy_thuộc_vào": ["DEPENDING", "DEPENDED"],
    "phụ_thuộc_vào": ["DEPEND", "DEPENDS"],
    "không_chắc_chắn": ["UNCERTAIN", "UNCERTAINLY"],
    "dường_như": ["APPEARS", "APPEARED"],
    "có_vẻ_như": ["APPEARING"],
    "gần_như": ["NEARLY", "ALMOST"],
    "đôi_khi": ["SOMETIMES"],
    "thỉnh_thoảng": ["OCCASIONALLY"],
    "phần_nào": ["SOMEWHAT"],
    "cho_thấy_khả_năng": ["SUGGEST"],
    "gợi_ý_rằng": ["SUGGESTS"],
    "có_lẽ": ["PERHAPS"],
    "hình_như": ["APPARENTLY"],
    "hiếm_khi": ["SELDOM", "SELDOMLY"],
}

CONSTRAINING_MAP: dict[str, list[str]] = {
    "yêu_cầu_bắt_buộc": ["REQUIREMENTS", "REQUIREMENT", "REQUIRED", "REQUIRES", "REQUIRE", "REQUIRING"],
    "nghĩa_vụ": ["OBLIGATIONS", "OBLIGATION", "OBLIGATED", "OBLIGATE", "OBLIGATES", "OBLIGATING", "OBLIGATORY", "OBLIGED"],
    "cam_kết": ["COMMITMENTS", "COMMITMENT", "COMMITTED", "COMMIT", "COMMITS", "COMMITTING"],
    "bị_hạn_chế": ["RESTRICTED", "RESTRICTIONS", "RESTRICT", "RESTRICTS", "RESTRICTING", "RESTRICTION", "RESTRICTIVE"],
    "được_phép": ["PERMITTED", "PERMITTING", "PERMISSIBLE", "PERMISSION", "PERMISSIONS"],
    "giới_hạn": ["LIMIT", "LIMITS", "LIMITING"],
    "tuân_thủ": ["COMPLY", "ABIDE"],
    "điều_khoản_giao_ước": ["COVENANTS", "COVENANT", "COVENANTED"],
    "ngăn_cản": ["PREVENT", "PREVENTED", "PREVENTING", "PREVENTS"],
    "áp_đặt": ["IMPOSED", "IMPOSE", "IMPOSES", "IMPOSING", "IMPOSITION", "IMPOSITIONS"],
    "cầm_cố": ["PLEDGED", "PLEDGE", "PLEDGES", "PLEDGING"],
    "bắt_buộc": ["MANDATORY", "MANDATED", "MANDATE", "MANDATES", "MANDATING", "COMPULSORY", "COMPULSION"],
    "cấm": ["PROHIBITED", "PROHIBIT", "PROHIBITS", "PROHIBITION", "PROHIBITING", "PROHIBITIONS", "PROHIBITIVE", "PROHIBITIVELY", "FORBIDDEN"],
    "ràng_buộc": ["CONSTRAINTS", "CONSTRAINED", "CONSTRAIN", "CONSTRAINT"],
    "bị_ràng_buộc": ["BOUND"],
    "không_thể_hủy_ngang": ["IRREVOCABLE", "IRREVOCABLY"],
    "loại_trừ_khả_năng": ["PRECLUDE", "PRECLUDED", "PRECLUDES", "PRECLUDING"],
    "nghiêm_ngặt": ["STRICT", "STRICTLY", "STRICTER", "STRICTEST"],
    "không_sẵn_có": ["UNAVAILABLE", "UNAVAILABILITY"],
    "tài_sản_bị_ràng_buộc": ["ENCUMBRANCES", "ENCUMBRANCE", "ENCUMBERED", "ENCUMBER", "ENCUMBERING", "ENCUMBERS"],
    "chỉ_thị_bắt_buộc": ["DIRECTIVE", "DIRECTIVES", "DICTATE", "DICTATED", "DICTATES"],
    "quy_định_rõ_trong_hợp_đồng": ["STIPULATED", "STIPULATE", "STIPULATES", "STIPULATION", "STIPULATIONS"],
    "kiềm_chế_không_thực_hiện": ["REFRAIN", "REFRAINING"],
    "kìm_hãm": ["INHIBIT", "INHIBITING", "INHIBITED", "INHIBITS"],
    "kéo_theo_nghĩa_vụ": ["ENTAIL", "ENTAILS"],
    "không_thể_hủy_bỏ": ["NONCANCELABLE", "NONCANCELLABLE"],
    "ký_quỹ": ["ESCROW", "ESCROWED", "ESCROWS"],
    "buộc_phải": ["COMPELLING", "COMPEL", "COMPELLED"],
    "đòi_hỏi_phải": ["NECESSITATE", "NECESSITATED", "NECESSITATING", "NECESSITATES"],
    "kiềm_chế_hành_động": ["RESTRAINING", "RESTRAIN", "RESTRAINED", "RESTRAINS", "RESTRAINT", "RESTRAINTS"],
    "khăng_khăng_yêu_cầu": ["INSIST"],
    "giới_hạn_trong_phạm_vi": ["CONFINED", "CONFINES"],
    "mắc_nợ": ["INDEBTED"],
    "điều_kiện_tiên_quyết": ["PRECONDITION"],
    "dành_riêng_cho_mục_đích_cụ_thể": ["EARMARKED"],
}

NEGATIVE_MAP: dict[str, list[str]] = {
    "thua_lỗ": ["LOSS", "LOSSES", "LOST", "LOSE"],
    "bất_lợi": ["ADVERSELY", "ADVERSE", "NEGATIVELY", "NEGATIVE", "UNFAVORABLE"],
    "gây_hiểu_lầm": ["MISLEADING"],
    "kiện_tụng": ["LITIGATION"],
    "gian_lận": ["FRAUD"],
    "bỏ_sót_thông_tin": ["OMIT", "OMITTED", "OMISSIONS"],
    "thiếu_sót": ["DEFICIENCIES", "DEFICIENCY", "INSUFFICIENT", "INADEQUATE", "INEFFECTIVE"],
    "điểm_yếu": ["WEAKNESSES", "WEAKNESS"],
    "suy_giảm_giá_trị": ["IMPAIRMENT", "IMPAIRED", "IMPAIR", "IMPAIRMENTS"],
    "thâm_hụt": ["DEFICIT"],
    "không_thể": ["UNABLE", "INABILITY"],
    "thất_bại": ["FAILURE", "FAIL", "FAILED", "FAILS", "FAILURES"],
    "trình_bày_lại": ["RESTATED", "RESTATEMENT"],
    "chấm_dứt_hợp_đồng": ["TERMINATION", "TERMINATED", "TERMINATE", "TERMINATES"],
    "hạn_chế_bất_lợi": ["LIMITATIONS", "LIMITATION"],
    "suy_giảm": ["DECLINE", "DECLINES", "DECLINED", "DECLINING"],
    "biến_động_mạnh": ["VOLATILITY", "VOLATILE"],
    "vỡ_nợ": ["DEFAULTS", "DEFAULT"],
    "khiếu_nại": ["CLAIMS"],
    "bồi_thường_thiệt_hại_kiện": ["DAMAGES", "DAMAGE"],
    "khó_khăn": ["DIFFICULT", "DIFFICULTIES", "DIFFICULTY"],
    "bị_phạt": ["PENALTIES", "PENALTY", "FINES"],
    "thanh_lý_bắt_buộc": ["LIQUIDATION", "LIQUIDATED"],
    "chưa_thanh_toán": ["UNPAID", "DELINQUENT", "ARREARS"],
    "nghi_ngờ_đáng_kể": ["DOUBTFUL", "DOUBT"],
    "vi_phạm": ["BREACH", "BREACHES", "VIOLATION", "VIOLATIONS", "VIOLATE", "VIOLATED"],
    "trì_hoãn": ["DELAY", "DELAYS", "DELAYED"],
    "thiếu_hụt": ["LACK", "ABSENCE"],
    "phá_sản": ["BANKRUPTCY", "INSOLVENCY", "DISSOLUTION"],
    "chịu_tổn_hại": ["EXPOSED", "EXPOSE", "HARM"],
    "lo_ngại": ["CONCERN", "CONCERNS", "CAUTIONARY", "CAUTIONED"],
    "ngừng_hoạt_động": ["CEASE", "CEASED", "DISCONTINUED", "DISCONTINUE", "SUSPENDED", "SUSPENSION", "SUSPEND"],
    "tái_cấu_trúc": ["RESTRUCTURING"],
    "báo_cáo_sai": ["MISSTATEMENT", "MISSTATEMENTS", "INACCURATE", "INCORRECT", "ERROR", "ERRORS"],
    "bị_cáo_buộc": ["ALLEGED", "ALLEGING", "ALLEGATIONS"],
    "tồi_tệ": ["BAD", "POOR", "SEVERE"],
    "bào_chữa": ["DEFEND", "DEFENDING", "DEFENDANT", "DEFENDANTS", "PLAINTIFF", "PLAINTIFFS"],
    "ngoài_dự_kiến": ["UNANTICIPATED", "UNEXPECTED", "UNFORESEEN"],
    "hủy_bỏ": ["CANCELLED", "CANCELED", "CANCELLATION", "CANCEL"],
    "bị_đe_dọa": ["THREATENED"],
    "gián_đoạn": ["DISRUPTIONS", "DISRUPTION", "INTERRUPTION", "INTERRUPTIONS"],
    "mất_khả_năng_thu_hồi": ["UNCOLLECTIBLE"],
    "xung_đột": ["CONFLICT", "CONFLICTS"],
    "tranh_chấp": ["DISPUTES", "DISPUTE"],
    "thách_thức": ["CHALLENGES", "CHALLENGE", "CHALLENGING"],
    "không_được_phép": ["UNAUTHORIZED", "INVALID"],
    "chịu_thiệt_hại": ["SUFFER", "SUFFERED", "INJURY"],
    "lỗi_thời": ["OBSOLETE"],
    "buộc_phải_từ_chức": ["RESIGNATION"],
    "tốn_kém": ["COSTLY"],
    "xuống_dốc": ["DOWNTURN", "SLOW"],
    "không_thành_công": ["UNSUCCESSFUL"],
    "sơ_suất": ["NEGLIGENCE"],
    "thảm_họa": ["DISASTERS", "HAZARDOUS"],
    "bị_bác_bỏ": ["DISMISSED", "DENIED"],
    "sai_phạm": ["MISCONDUCT"],
    "không_nhất_quán": ["INCONSISTENT"],
    "chưa_giải_quyết": ["UNRESOLVED"],
    "tịch_thu_tài_sản": ["FORFEITURE", "FORFEITED", "FORFEITURES"],
    "xiết_nợ": ["FORECLOSURE"],
    "xâm_phạm": ["INFRINGEMENT"],
    "bỏ_lỡ_thời_hạn": ["LAPSE"],
    "suy_thoái": ["DETERIORATE", "DETERIORATION"],
    "vấn_đề_phát_sinh": ["PROBLEM", "PROBLEMS"],
}

POSITIVE_MAP: dict[str, list[str]] = {
    "lợi_nhuận_tăng": ["GAIN", "GAINS", "GAINED", "GAINING"],
    "có_năng_lực": ["ABLE"],
    "tốt_nhất": ["BEST"],
    "cải_thiện": ["IMPROVEMENTS", "IMPROVEMENT", "IMPROVE", "IMPROVED", "IMPROVING", "IMPROVES"],
    "cơ_hội": ["OPPORTUNITIES", "OPPORTUNITY"],
    "thành_công": ["SUCCESSFUL", "SUCCESS", "SUCCESSFULLY", "SUCCEED", "SUCCEEDING", "SUCCEEDED"],
    "thuận_lợi": ["FAVORABLE", "FAVORABLY", "FAVORED", "FAVORING"],
    "đạt_được": ["ACHIEVE", "ACHIEVED", "ACHIEVING", "ACHIEVEMENT", "ACHIEVEMENTS", "ACHIEVES", "ACCOMPLISH", "ACCOMPLISHED", "ACCOMPLISHING", "ATTAIN", "ATTAINED", "ATTAINING", "ATTAINS", "ATTAINMENT"],
    "tiến_bộ": ["ADVANCES", "ADVANCING", "ADVANCEMENT", "ADVANCEMENTS", "PROGRESS", "PROGRESSES", "PROGRESSED"],
    "đáp_ứng": ["SATISFY", "SATISFIES", "SATISFIED", "SATISFACTION", "SATISFACTORY", "SATISFYING", "SATISFACTORILY"],
    "khả_năng_sinh_lời": ["PROFITABILITY", "PROFITABLE", "PROFITABLY"],
    "tốt": ["GOOD"],
    "tích_cực": ["POSITIVE", "POSITIVELY"],
    "độc_quyền": ["EXCLUSIVE", "EXCLUSIVELY", "EXCLUSIVITY"],
    "tạo_điều_kiện": ["ENABLE", "ENABLES", "ENABLING", "ENABLED"],
    "mạnh": ["STRONG", "STRONGER", "STRONGEST"],
    "tốt_hơn": ["BETTER"],
    "nâng_cao": ["ENHANCE", "ENHANCED", "ENHANCING", "ENHANCEMENT", "ENHANCEMENTS", "ENHANCES"],
    "dẫn_đầu": ["LEADING", "LEADERSHIP"],
    "lợi_thế": ["ADVANTAGE", "ADVANTAGES", "ADVANTAGEOUS"],
    "hài_lòng": ["SATISFIED", "PLEASED", "PLEASURE", "ENJOY", "ENJOYED", "ENJOYMENT"],
    "đảm_bảo": ["ASSURE", "ASSURED", "ASSURING"],
    "cao_nhất": ["HIGHEST"],
    "đầy_đủ": ["ADEQUATELY"],
    "vượt_trội": ["SUPERIOR", "EXCEPTIONAL", "EXCELLENT", "EXCELLENCE", "PREMIER", "EXEMPLARY"],
    "hiệu_quả": ["EFFICIENCY", "EFFICIENT", "EFFICIENTLY", "EFFICIENCIES"],
    "sức_mạnh": ["STRENGTH", "STRENGTHS", "STRENGTHEN", "STRENGTHENING", "STRENGTHENED", "STRENGTHENS"],
    "giá_trị": ["VALUABLE"],
    "mong_muốn": ["DESIRED", "DESIRABLE"],
    "hấp_dẫn": ["ATTRACTIVE", "ATTRACTIVENESS"],
    "ổn_định": ["STABLE", "STABILITY", "STABILIZE", "STABILIZATION", "STABILIZED", "STABILIZING"],
    "đổi_mới": ["INNOVATIVE", "INNOVATION", "INNOVATIONS", "INNOVATE"],
    "liêm_chính": ["INTEGRITY"],
    "minh_bạch": ["TRANSPARENCY"],
    "sáng_chế": ["INVENTIONS", "INVENTION", "INVENTOR"],
    "hợp_tác": ["ALLIANCES", "ALLIANCE", "COLLABORATION", "COLLABORATIVE", "COLLABORATIONS", "COLLABORATE", "COLLABORATING", "COLLABORATOR", "COLLABORATORS"],
    "được_hưởng_lợi": ["BENEFITED", "BENEFITING"],
    "dễ_dàng": ["EASILY", "EASY", "EASIER"],
    "vinh_dự": ["HONOR", "HONORED"],
    "từ_thiện": ["CHARITABLE"],
    "phần_thưởng": ["REWARD", "REWARDING"],
    "hoàn_hảo": ["PERFECT", "PERFECTED"],
    "được_ưa_chuộng": ["POPULAR", "POPULARITY"],
    "được_trao_quyền": ["EMPOWERED", "EMPOWER"],
    "mang_tính_xây_dựng": ["CONSTRUCTIVE"],
    "thân_thiện": ["FRIENDLY"],
    "sáng_tạo": ["CREATIVE"],
    "chủ_động": ["PROACTIVE", "PROACTIVELY", "DILIGENT", "DILIGENTLY"],
    "tự_tin": ["CONFIDENT"],
    "lạc_quan": ["OPTIMISTIC"],
    "chiến_thắng": ["WIN", "WINNING"],
    "lấy_lại": ["REGAIN"],
    "khác_biệt_nổi_bật": ["DISTINCTIVE", "DISTINCTION"],
    "đột_phá": ["BREAKTHROUGH"],
    "lý_tưởng": ["IDEAL"],
    "suôn_sẻ": ["SMOOTH"],
    "xứng_đáng": ["MERITORIOUS"],
}

CATEGORY_SEEDS: dict[str, dict[str, list[str]]] = {
    "negative": NEGATIVE_MAP,
    "positive": POSITIVE_MAP,
    "uncertainty": UNCERTAINTY_MAP,
    "litigious": LITIGIOUS_MAP,
    "strong_modal": STRONG_MODAL_MAP,
    "weak_modal": WEAK_MODAL_MAP,
    "constraining": CONSTRAINING_MAP,
}


def validate_against_master_dictionary(category: str, seed_map: dict[str, list[str]]) -> None:
    """Kiem tra moi tu tieng Anh trong seed_map co thuc su ton tai trong danh
    muc tuong ung cua Master Dictionary (tranh go nham chinh ta khi tra cuu
    thu cong tu cac danh sach in ra man hinh).
    """
    category_csv = CATEGORIES_DIR / f"{category}_master_dictionary.csv"
    known_words = set(pd.read_csv(category_csv)["Word"])
    all_mapped_words = {word for words in seed_map.values() for word in words}
    unknown_words = sorted(all_mapped_words.difference(known_words))
    if unknown_words:
        raise ValueError(
            f"[{category}] {len(unknown_words)} tu khong ton tai trong danh muc "
            f"Master Dictionary '{category}': {unknown_words}"
        )


def write_seed_txt(vi_terms: list[str], output_path: Path, width: int = 79) -> None:
    text = ", ".join(vi_terms)
    wrapped_lines = textwrap.wrap(
        text, width=width, break_long_words=False, break_on_hyphens=False
    )
    output_path.write_text("\n".join(wrapped_lines) + "\n", encoding="utf-8")


def write_mapping_csv(seed_map: dict[str, list[str]], output_path: Path) -> None:
    rows = [
        {"vietnamese_term": vi_term, "english_words": "; ".join(en_words)}
        for vi_term, en_words in seed_map.items()
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    VI_SEEDS_DIR.mkdir(parents=True, exist_ok=True)

    for category, seed_map in CATEGORY_SEEDS.items():
        validate_against_master_dictionary(category, seed_map)

        vi_terms = list(seed_map.keys())
        seed_txt_path = VI_SEEDS_DIR / f"{category}_word.txt"
        write_seed_txt(vi_terms, seed_txt_path)

        mapping_csv_path = CATEGORIES_DIR / f"{category}_vi_mapping.csv"
        write_mapping_csv(seed_map, mapping_csv_path)

        total_english_words = sum(len(words) for words in seed_map.values())
        print(
            f"{category}: {len(vi_terms)} seed tieng Viet, gop tu "
            f"{total_english_words} tu tieng Anh trong Master Dictionary"
        )
        print("  ->", seed_txt_path)
        print("  ->", mapping_csv_path)


if __name__ == "__main__":
    main()
