"""Inti prediksi risiko kredit yang dipakai bersama oleh API dan Streamlit app.

Modul ini memuat pipeline siap pakai (model/model.joblib), membangun baris input
dari data nasabah baru, menghitung probabilitas gagal bayar terkalibrasi (PD),
dan menentukan zona keputusan.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from main import GRADE_MAP, SEED, build_deployable, load_and_clean

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "credit_risk_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "model.joblib"

HOME_OPTIONS = ["MORTGAGE", "OTHER", "OWN", "RENT"]
INTENT_OPTIONS = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
                  "MEDICAL", "PERSONAL", "VENTURE"]
GRADE_OPTIONS = list(GRADE_MAP)
DEFAULT_OPTIONS = ["N", "Y"]

REQUIRED_FIELDS = ["person_age", "person_income", "person_home_ownership",
                   "loan_intent", "loan_grade", "loan_amnt",
                   "loan_percent_income", "cb_person_default_on_file",
                   "cb_person_cred_hist_length"]
OPTIONAL_FIELDS = ["person_emp_length", "loan_int_rate"]

_bundle = None


def load_bundle(force_reload=False):
    """Muat model.joblib; bila belum ada, latih otomatis dari dataset."""
    global _bundle
    if _bundle is not None and not force_reload:
        return _bundle
    if MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)
        return _bundle
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Model belum ada dan dataset tidak ditemukan: {DATA_PATH}")
    X, y = load_and_clean(str(DATA_PATH))
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y,
                                      random_state=SEED)
    pos_w = (ytr == 0).sum() / (ytr == 1).sum()
    _bundle = build_deployable(Xtr.reset_index(drop=True), ytr,
                               "Z-Score", "XGBoost", pos_w)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_bundle, MODEL_PATH)
    return _bundle


def build_row(features):
    """Ubah dict input (nama kolom dataset) menjadi DataFrame 1 baris."""
    missing = [f for f in REQUIRED_FIELDS if features.get(f) is None]
    if missing:
        raise ValueError(f"Field wajib belum diisi: {', '.join(missing)}")

    grade = features["loan_grade"]
    if grade not in GRADE_MAP:
        raise ValueError(f"loan_grade tidak valid: {grade}")

    flag = features["cb_person_default_on_file"]
    if flag not in DEFAULT_OPTIONS:
        raise ValueError("cb_person_default_on_file harus 'Y' atau 'N'")

    def optional(name):
        value = features.get(name)
        return np.nan if value is None else float(value)

    return pd.DataFrame([{
        "person_age": int(features["person_age"]),
        "person_income": float(features["person_income"]),
        "person_emp_length": optional("person_emp_length"),
        "loan_amnt": float(features["loan_amnt"]),
        "loan_int_rate": optional("loan_int_rate"),
        "loan_percent_income": float(features["loan_percent_income"]),
        "cb_person_cred_hist_length": int(features["cb_person_cred_hist_length"]),
        "loan_grade_ord": GRADE_MAP[grade],
        "cb_default_bin": int(flag == "Y"),
        "person_home_ownership": features["person_home_ownership"],
        "loan_intent": features["loan_intent"],
        "loan_grade": grade,
    }])


def calibrate(bundle, raw_score):
    """Ubah skor mentah model menjadi PD terkalibrasi (Platt scaling)."""
    p = float(np.clip(raw_score, 1e-6, 1 - 1e-6))
    logit = np.array([[np.log(p / (1 - p))]])
    return float(bundle["calibrator"].predict_proba(logit)[:, 1][0])


def _zona(pd_cal, t_low, t_high):
    if pd_cal < t_low:
        return "approve"
    if pd_cal < t_high:
        return "review"
    return "decline"


KEPUTUSAN = {"approve": "DISETUJUI", "review": "REVIEW_MANUAL",
             "decline": "DITOLAK"}
RISIKO = {"approve": "RENDAH", "review": "SEDANG", "decline": "TINGGI"}


def predict_one(features, t_low=None, t_high=None):
    """Prediksi satu nasabah.

    features : dict dengan nama kolom dataset (tanpa loan_status)
    t_low    : batas bawah zona review (default: nilai dari bundle)
    t_high   : batas atas zona review / batas tolak (default: nilai dari bundle)
    """
    bundle = load_bundle()
    sumber = "override" if (t_low is not None or t_high is not None) else "model"
    if t_low is None:
        t_low = bundle["t_low"]
    if t_high is None:
        t_high = bundle["t_high"]
    t_low, t_high = float(t_low), float(t_high)
    if not (0 < t_low < t_high < 1):
        raise ValueError("Threshold tidak valid: harus 0 < t_low < t_high < 1")

    row = build_row(features)
    raw = float(bundle["pipeline"].predict_proba(row)[:, 1][0])
    pd_cal = calibrate(bundle, raw)
    zona = _zona(pd_cal, t_low, t_high)

    catatan = []
    if np.isnan(row["person_emp_length"].iloc[0]):
        catatan.append("person_emp_length kosong -> diimputasi median data latih")
    if np.isnan(row["loan_int_rate"].iloc[0]):
        catatan.append("loan_int_rate kosong -> diimputasi median per loan_grade")

    peringatan = _cek_ood(row, bundle.get("train_ranges") or {})

    return {
        "probabilitas_gagal_bayar": pd_cal,
        "skor_mentah": raw,
        "keputusan": KEPUTUSAN[zona],
        "zona": zona,
        "risiko": RISIKO[zona],
        "threshold_dipakai": {"t_low": t_low, "t_high": t_high,
                              "sumber": sumber},
        "catatan": catatan,
        "peringatan": peringatan,
    }


def _fmt(v):
    v = float(v)
    return f"{int(v)}" if v.is_integer() else f"{v:.2f}"


def _cek_ood(row, train_ranges):
    """Bandingkan input dengan rentang data latih; nilai di luar = ekstrapolasi."""
    peringatan = []
    for kolom, rentang in train_ranges.items():
        if kolom not in row.columns:
            continue
        nilai = row[kolom].iloc[0]
        if pd.isna(nilai):
            continue
        if nilai < rentang["min"] or nilai > rentang["max"]:
            peringatan.append(
                f"{kolom} = {_fmt(nilai)} di luar rentang data latih "
                f"({_fmt(rentang['min'])} - {_fmt(rentang['max'])}) "
                f"- prediksi bersifat ekstrapolasi")
    return peringatan


def model_info():
    bundle = load_bundle()
    return {
        "model": bundle["model"],
        "normalisasi": bundle["scaler"],
        "kalibrasi": "Platt scaling (sigmoid)",
        "roc_auc_oof": round(float(bundle["oof_auc"]), 4),
        "pr_auc_oof": round(float(bundle["oof_pr_auc"]), 4),
        "base_rate_latih": round(float(bundle["base_rate"]), 4),
        "trained_at": bundle["trained_at"],
        "threshold_default": {"t_low": round(float(bundle["t_low"]), 2),
                              "t_high": round(float(bundle["t_high"]), 2)},
        "rentang_data_latih": bundle.get("train_ranges"),
        "input_wajib": REQUIRED_FIELDS,
        "input_opsional": OPTIONAL_FIELDS,
        "nilai_kategori": {
            "person_home_ownership": HOME_OPTIONS,
            "loan_intent": INTENT_OPTIONS,
            "loan_grade": GRADE_OPTIONS,
            "cb_person_default_on_file": DEFAULT_OPTIONS,
        },
    }
