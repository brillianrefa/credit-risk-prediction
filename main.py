"""
Perbandingan Metode ML - Sistem Evaluasi Risiko & Segmentasi Kredit Nasabah
============================================================================
Dataset : credit_risk_dataset.csv  (target: loan_status, 1 = gagal bayar -> "Ditolak")

Bagian yang dibandingkan
  1. Normalisasi : Min-Max, Z-Score, Decimal Scaling, Sigmoidal, Softmax
  2. Klasifikasi : Logistic Regression, Decision Tree, k-NN, Random Forest, XGBoost
  3. Validasi    : Holdout, Random Subsampling, K-Fold CV, Bootstrap
  4. Segmentasi  : K-Means (k = 2..8) dengan SSE, Silhouette, Davies-Bouldin

Cara pakai:
    pip install -r requirements.txt
    python main.py --data credit_risk_dataset.csv --out hasil
    streamlit run app.py     # aplikasi web untuk prediksi nasabah baru
    uvicorn api:app --port 8000   # REST API untuk tim backend

Model siap pakai (pipeline + kalibrasi + threshold) otomatis disimpan ke
model/model.joblib setelah training selesai.

Opsi penting:
    --k 5         jumlah fold untuk K-Fold CV
    --repeats 10  jumlah ulangan Random Subsampling
    --boot 20     jumlah resampling Bootstrap

Catatan: Leave-One-Out CV tidak diikutkan karena harus melatih satu model per baris data
(sekitar 26 ribu model) sehingga terlalu lambat.

Definisi normalisasi (dipelajari dari data latih saja, lalu dipakai ke data uji):
    Min-Max         : (x - min) / (max - min)                       -> [0, 1]
    Z-Score         : (x - mean) / std                              -> rata-rata 0, std 1
    Decimal Scaling : x / 10^j, j = bilangan bulat terkecil agar max|x'| < 1
    Sigmoidal       : (1 - e^-a) / (1 + e^-a), a = (x - mean) / std -> (-1, 1)
    Softmax         : 1 / (1 + e^-a), a = (x - mean) / (std / 2*pi) -> (0, 1)

Semua tabel disimpan sebagai CSV dan semua grafik sebagai PNG di folder --out.
"""
import argparse
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, davies_bouldin_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import (StratifiedKFold,
                                     StratifiedShuffleSplit, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
TARGET = "loan_status"
GRADE_MAP = {g: i for i, g in enumerate("ABCDEFG")}

NUM_COLS = ["person_age", "person_income", "person_emp_length", "loan_amnt",
            "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
            "loan_grade_ord"]
BIN_COLS = ["cb_default_bin"]
CAT_COLS = ["person_home_ownership", "loan_intent"]

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc", "pr_auc": "average_precision"}


# ----------------------------------------------------------------------------
# Metode normalisasi (implementasi sendiri agar sesuai definisi di materi)
# ----------------------------------------------------------------------------
class DecimalScaler(BaseEstimator, TransformerMixin):
    """Decimal scaling: x / 10^j dengan j terkecil sehingga max|x'| < 1."""

    def fit(self, X, y=None):
        max_abs = np.abs(np.asarray(X, dtype=float)).max(axis=0)
        j = np.where(max_abs > 0, np.floor(np.log10(np.maximum(max_abs, 1e-12))) + 1, 0)
        self.divisor_ = 10.0 ** j
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) / self.divisor_


class SigmoidalScaler(BaseEstimator, TransformerMixin):
    """Sigmoidal: (1 - e^-a)/(1 + e^-a) dengan a = (x - mean)/std  -> (-1, 1)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        return self

    def transform(self, X):
        a = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        return np.tanh(a / 2.0)          # identik dengan (1-e^-a)/(1+e^-a)


class SoftmaxScaler(BaseEstimator, TransformerMixin):
    """Softmax scaling: 1/(1+e^-a), a = (x - mean)/(lam * std / 2pi)  -> (0, 1)."""

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
        return self

    def transform(self, X):
        a = (np.asarray(X, dtype=float) - self.mean_) / (self.lam * self.std_ / (2 * np.pi))
        return expit(a)


SCALERS = {"Min-Max": MinMaxScaler, "Z-Score": StandardScaler,
           "Decimal Scaling": DecimalScaler, "Sigmoidal": SigmoidalScaler,
           "Softmax": SoftmaxScaler}


# ----------------------------------------------------------------------------
# 0. Load, cleaning, pipeline
# ----------------------------------------------------------------------------
def load_and_clean(path):
    df = pd.read_csv(path)
    n0 = len(df)
    df = df.drop_duplicates()
    # data tidak masuk akal: usia > 100, lama kerja > 60 tahun atau melebihi (usia - 14)
    df = df[(df["person_age"] <= 100)]
    df = df[(df["person_emp_length"].isna()) |
            ((df["person_emp_length"] <= 60) &
             (df["person_emp_length"] <= df["person_age"] - 14))]
    df = df.reset_index(drop=True)
    df["loan_grade_ord"] = df["loan_grade"].map(GRADE_MAP)
    df["cb_default_bin"] = (df["cb_person_default_on_file"] == "Y").astype(int)
    print(f"[Data] {n0} baris -> {len(df)} baris setelah pembersihan "
          f"| proporsi gagal bayar = {df[TARGET].mean():.3f}")
    X = df[NUM_COLS + BIN_COLS + CAT_COLS + ["loan_grade"]]
    y = df[TARGET].values
    return X, y


class GradeMedianImputer(BaseEstimator, TransformerMixin):
    """Imputasi di dalam pipeline (tanpa leakage):
    loan_int_rate -> median per loan_grade, person_emp_length -> median global."""

    def fit(self, X, y=None):
        self.grade_med_ = X.groupby("loan_grade")["loan_int_rate"].median()
        self.int_med_ = X["loan_int_rate"].median()
        self.emp_med_ = X["person_emp_length"].median()
        return self

    def transform(self, X):
        X = X.copy()
        fill = X["loan_grade"].map(self.grade_med_).fillna(self.int_med_)
        X["loan_int_rate"] = X["loan_int_rate"].fillna(fill)
        X["person_emp_length"] = X["person_emp_length"].fillna(self.emp_med_)
        return X


def make_preprocessor(scaler_name):
    ct = ColumnTransformer(
        [("num", SCALERS[scaler_name](), NUM_COLS),
         ("bin", "passthrough", BIN_COLS),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="drop")
    return Pipeline([("impute", GradeMedianImputer()), ("prep", ct)])


def make_model(name, pos_weight):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=30,
                                      class_weight="balanced", random_state=SEED)
    if name == "k-NN":
        return KNeighborsClassifier(n_neighbors=25, n_jobs=-1)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=SEED)
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                 subsample=0.9, colsample_bytree=0.9,
                                 scale_pos_weight=pos_weight, eval_metric="logloss",
                                 n_jobs=-1, random_state=SEED)
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            print("[Info] xgboost tidak terpasang -> memakai HistGradientBoosting")
            return HistGradientBoostingClassifier(class_weight="balanced",
                                                  random_state=SEED)
    raise ValueError(name)


def pipe(scaler_name, model_name, pos_weight):
    return Pipeline([("pre", make_preprocessor(scaler_name)),
                     ("clf", make_model(model_name, pos_weight))])


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summarize(res):
    out = {}
    for m in SCORING:
        out[m] = res[f"test_{m}"].mean()
        out[m + "_std"] = res[f"test_{m}"].std()
    out["fit_time_s"] = res["fit_time"].mean()
    return out


def save_table(df, out, name):
    df.to_csv(os.path.join(out, name + ".csv"))
    print(f"\n=== {name} ===")
    print(df.round(4).to_string())


# ----------------------------------------------------------------------------
# 1. Perbandingan normalisasi
# ----------------------------------------------------------------------------
def compare_normalization(Xtr, ytr, pos_w, out):
    scalers = list(SCALERS)
    models = ["Logistic Regression", "k-NN", "Decision Tree"]   # DT = kontrol
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for sc in scalers:
        for mn in models:
            res = cross_validate(pipe(sc, mn, pos_w), Xtr, ytr, cv=cv, scoring=SCORING)
            r = summarize(res)
            r.update(normalisasi=sc, model=mn)
            rows.append(r)
            print(f"  [Norm] {sc:16s} | {mn:20s} | AUC={r['roc_auc']:.4f} F1={r['f1']:.4f}")
    df = pd.DataFrame(rows).set_index(["model", "normalisasi"])
    save_table(df[["roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy"]],
               out, "1a_perbandingan_normalisasi")

    # peringkat: rata-rata AUC pada model yang sensitif skala (LR + k-NN)
    rank = (df.loc[["Logistic Regression", "k-NN"]]
            .groupby("normalisasi")[["roc_auc", "f1"]].mean()
            .sort_values("roc_auc", ascending=False))
    rank.insert(0, "peringkat", range(1, len(rank) + 1))
    save_table(rank, out, "1b_peringkat_normalisasi")

    piv = df["roc_auc"].unstack("model").loc[scalers]
    ax = piv.plot(kind="bar", figsize=(8.5, 4.6), rot=0)
    ax.set_ylabel("ROC-AUC (5-fold CV)")
    ax.set_title("Pengaruh metode normalisasi")
    ax.set_ylim(max(0.5, piv.min().min() - 0.03), 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "1_normalisasi.png"), dpi=140)
    plt.close()

    best = rank.index[0]
    print(f"\n[Pilihan] Normalisasi terbaik (rata-rata AUC LR & k-NN): {best}")
    return best


# ----------------------------------------------------------------------------
# 2. Perbandingan klasifikasi
# ----------------------------------------------------------------------------
def compare_classifiers(Xtr, ytr, Xte, yte, scaler, pos_w, out):
    names = ["Logistic Regression", "Decision Tree", "k-NN", "Random Forest", "XGBoost"]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_rows, test_rows, curves = [], [], {}
    for mn in names:
        p = pipe(scaler, mn, pos_w)
        res = cross_validate(p, Xtr, ytr, cv=cv, scoring=SCORING)
        cv_rows.append(pd.Series(summarize(res), name=mn))

        t0 = time.time()
        p.fit(Xtr, ytr)
        fit_t = time.time() - t0
        t0 = time.time()
        proba = p.predict_proba(Xte)[:, 1]
        pred_t = time.time() - t0
        pred = (proba >= 0.5).astype(int)
        test_rows.append(pd.Series({
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred),
            "recall": recall_score(yte, pred),
            "f1": f1_score(yte, pred),
            "roc_auc": roc_auc_score(yte, proba),
            "pr_auc": average_precision_score(yte, proba),
            "ks": ks_stat(yte, proba),
            "fit_time_s": fit_t,
            "predict_time_s": pred_t}, name=mn))
        curves[mn] = proba
        print(f"  [Clf] {mn:20s} | CV AUC={cv_rows[-1]['roc_auc']:.4f} "
              f"| Test AUC={test_rows[-1]['roc_auc']:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    save_table(cv_df[["roc_auc", "roc_auc_std", "pr_auc", "f1", "recall",
                      "precision", "accuracy", "fit_time_s"]],
               out, "2a_klasifikasi_cv_5fold_train")
    save_table(test_df, out, "2b_klasifikasi_test_holdout_threshold_0.5")
    save_table(test_df.sort_values("roc_auc", ascending=False)[["roc_auc", "f1", "recall", "precision"]]
               .assign(peringkat=range(1, len(test_df) + 1)),
               out, "2c_peringkat_klasifikasi")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for mn, pr in curves.items():
        fpr, tpr, _ = roc_curve(yte, pr)
        ax[0].plot(fpr, tpr, label=f"{mn} ({roc_auc_score(yte, pr):.3f})")
        pcs, rcl, _ = precision_recall_curve(yte, pr)
        ax[1].plot(rcl, pcs, label=f"{mn} ({average_precision_score(yte, pr):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="Kurva ROC (data uji)", xlabel="False Positive Rate",
              ylabel="True Positive Rate")
    ax[1].set(title="Kurva Precision-Recall (data uji)", xlabel="Recall",
              ylabel="Precision")
    ax[0].legend(); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "2_kurva_roc_pr.png"), dpi=140)
    plt.close()

    ax = test_df[["roc_auc", "pr_auc", "f1", "recall", "precision"]].plot(
        kind="bar", figsize=(10, 4.8), rot=15)
    ax.set_title("Perbandingan klasifikasi pada data uji"); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "2_perbandingan_klasifikasi.png"), dpi=140)
    plt.close()

    best = cv_df["roc_auc"].idxmax()
    print(f"\n[Pilihan] Model terbaik berdasarkan ROC-AUC CV: {best}")
    return best


# ----------------------------------------------------------------------------
# 3. Perbandingan metode validasi
# ----------------------------------------------------------------------------
def _auc_f1(model, Xa, ya, Xb, yb):
    model.fit(Xa, ya)
    pr = model.predict_proba(Xb)[:, 1]
    return roc_auc_score(yb, pr), f1_score(yb, (pr >= 0.5).astype(int))


def run_validation_methods(X, y, scaler, model_name, pos_w, truth,
                           k, repeats, n_boot):
    """Menjalankan metode validasi (Holdout, Random Subsampling, K-Fold, Bootstrap)."""
    mk = lambda: pipe(scaler, model_name, pos_w)
    rows = []

    def add(name, aucs, f1s, n_fits, t0):
        aucs = np.asarray(aucs, dtype=float)
        rows.append({"metode": name, "jumlah_model_dilatih": n_fits,
                     "roc_auc": aucs.mean(),
                     "roc_auc_std": aucs.std() if len(aucs) > 1 else np.nan,
                     "f1": float(np.mean(f1s)),
                     "selisih_vs_data_uji": abs(aucs.mean() - truth),
                     "waktu_s": time.time() - t0})
        print(f"  [Val] {name:34s} AUC={aucs.mean():.4f}")

    # 1) Holdout: satu kali pembagian 75/25
    t0 = time.time()
    a, b, ya, yb = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    auc, f1 = _auc_f1(mk(), a, ya, b, yb)
    add("Holdout (75/25)", [auc], [f1], 1, t0)

    # 2) Random subsampling: holdout diulang beberapa kali dengan pembagian acak
    t0 = time.time()
    ss = StratifiedShuffleSplit(n_splits=repeats, test_size=0.25, random_state=SEED)
    res = cross_validate(mk(), X, y, cv=ss, scoring={"roc_auc": "roc_auc", "f1": "f1"})
    add(f"Random Subsampling ({repeats}x 75/25)", res["test_roc_auc"], res["test_f1"], repeats, t0)

    # 3) K-Fold cross validation (stratified)
    t0 = time.time()
    res = cross_validate(mk(), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=SEED),
                         scoring={"roc_auc": "roc_auc", "f1": "f1"})
    add(f"K-Fold CV (k={k})", res["test_roc_auc"], res["test_f1"], k, t0)

    # 4) Bootstrap: latih pada sampel dengan pengembalian, uji pada data out-of-bag
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    n = len(y)
    oob_auc, oob_f1 = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(np.unique(y[oob])) < 2:
            continue
        auc, f1 = _auc_f1(mk(), X.iloc[idx], y[idx], X.iloc[oob], y[oob])
        oob_auc.append(auc); oob_f1.append(f1)
    add(f"Bootstrap OOB (B={n_boot})", oob_auc, oob_f1, n_boot, t0)

    # Bootstrap .632: gabungan skor out-of-bag dan skor pada data latih (apparent)
    full = mk().fit(X, y)
    apparent = roc_auc_score(y, full.predict_proba(X)[:, 1])
    b632 = 0.368 * apparent + 0.632 * np.mean(oob_auc)
    rows.append({"metode": f"Bootstrap .632 (B={n_boot})", "jumlah_model_dilatih": n_boot,
                 "roc_auc": b632, "roc_auc_std": np.nan, "f1": np.nan,
                 "selisih_vs_data_uji": abs(b632 - truth), "waktu_s": np.nan})
    print(f"  [Val] {'Bootstrap .632':34s} AUC={b632:.4f} (apparent={apparent:.4f})")
    return pd.DataFrame(rows).set_index("metode")


def compare_validation(Xtr, ytr, Xte, yte, scaler, model_name, pos_w, out,
                       k, repeats, n_boot):
    # pembanding: skor pada data uji yang tidak pernah disentuh saat validasi
    final = pipe(scaler, model_name, pos_w).fit(Xtr, ytr)
    truth = roc_auc_score(yte, final.predict_proba(Xte)[:, 1])
    print(f"\n[Info] AUC data uji (model dilatih di seluruh data latih) = {truth:.4f}")

    df = run_validation_methods(Xtr, ytr, scaler, model_name, pos_w, truth,
                                k, repeats, n_boot)
    save_table(df, out, "3_perbandingan_validasi")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(df["roc_auc"], range(len(df)), xerr=df["roc_auc_std"].fillna(0),
                fmt="o", capsize=4)
    ax.axvline(truth, color="r", ls="--", label=f"AUC data uji = {truth:.4f}")
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC")
    ax.set_title(f"Estimasi performa tiap metode validasi ({model_name})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "3_validasi.png"), dpi=140)
    plt.close()
    print("[Catatan] Leave-One-Out tidak diikutkan: butuh satu model per baris data "
          f"(~{len(ytr)} model), terlalu lambat.")
    return df


# ----------------------------------------------------------------------------
# Tuning threshold + evaluasi akhir
# ----------------------------------------------------------------------------
def final_evaluation(Xtr, ytr, Xte, yte, scaler, model_name, pos_w, out):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = pipe(scaler, model_name, pos_w)
    oof = cross_val_predict(p, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    ths = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(ytr, (oof >= t).astype(int)) for t in ths]
    th = float(ths[int(np.argmax(f1s))])          # dipilih TANPA melihat data uji
    p.fit(Xtr, ytr)
    proba = p.predict_proba(Xte)[:, 1]
    rows = {}
    for label, t in [("threshold 0.5", 0.5), (f"threshold tuned ({th:.2f})", th)]:
        pred = (proba >= t).astype(int)
        rows[label] = {"accuracy": accuracy_score(yte, pred),
                       "precision": precision_score(yte, pred),
                       "recall": recall_score(yte, pred),
                       "f1": f1_score(yte, pred),
                       "roc_auc": roc_auc_score(yte, proba),
                       "pr_auc": average_precision_score(yte, proba),
                       "ks": ks_stat(yte, proba)}
    save_table(pd.DataFrame(rows).T, out, f"4_evaluasi_akhir_{model_name.replace(' ', '_')}")

    cm = confusion_matrix(yte, (proba >= th).astype(int))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Disetujui", "Ditolak"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Aman", "Gagal bayar"])
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title(f"Confusion matrix ({model_name}, th={th:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "4_confusion_matrix.png"), dpi=140)
    plt.close()


# ----------------------------------------------------------------------------
# Simpan model siap-pakai (deployable) untuk data baru
# ----------------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def build_deployable(Xtr, ytr, scaler_name, model_name, pos_w):
    """Latih pipeline final + kalibrasi Platt (probabilitas -> PD terkalibrasi).

    Zona keputusan dihitung dari prediksi out-of-fold data latih (tanpa data uji):
      t_low  : threshold dengan F1 terbaik          -> di bawahnya = disetujui
      t_high : threshold pertama dengan precision >= 0.90 -> di atasnya = ditolak
    """
    base = pipe(scaler_name, model_name, pos_w)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof_raw = cross_val_predict(base, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    calibrator = LogisticRegression(max_iter=1000).fit(_logit(oof_raw), ytr)
    oof_cal = calibrator.predict_proba(_logit(oof_raw))[:, 1]

    ths = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(ytr, (oof_cal >= t).astype(int)) for t in ths]
    t_low = float(ths[int(np.argmax(f1s))])

    min_pos = max(30, int(0.005 * len(ytr)))
    t_high = min(0.99, t_low + 0.10)
    for t in ths:
        pred = (oof_cal >= t).astype(int)
        if pred.sum() >= min_pos and precision_score(ytr, pred, zero_division=0) >= 0.90:
            t_high = float(t)
            break
    if t_high <= t_low:
        t_high = min(0.99, t_low + 0.10)

    base.fit(Xtr, ytr)
    train_ranges = {c: {"min": float(Xtr[c].min()), "max": float(Xtr[c].max())}
                    for c in NUM_COLS if c in Xtr.columns}
    print(f"[Deploy] Kalibrasi Platt | t_low={t_low:.2f} t_high={t_high:.2f} "
          f"| OOF AUC={roc_auc_score(ytr, oof_raw):.4f}")
    return {"pipeline": base, "calibrator": calibrator,
            "t_low": t_low, "t_high": t_high,
            "scaler": scaler_name, "model": model_name,
            "features": list(Xtr.columns),
            "oof_auc": float(roc_auc_score(ytr, oof_raw)),
            "oof_pr_auc": float(average_precision_score(ytr, oof_raw)),
            "base_rate": float(np.mean(ytr)),
            "train_ranges": train_ranges,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_deployable(Xtr, ytr, scaler_name, model_name, pos_w, out_dir="model"):
    os.makedirs(out_dir, exist_ok=True)
    bundle = build_deployable(Xtr, ytr, scaler_name, model_name, pos_w)
    path = os.path.join(out_dir, "model.joblib")
    joblib.dump(bundle, path)
    print(f"[Deploy] Model + preprocessing disimpan ke {path}")
    return path


# ----------------------------------------------------------------------------
# Segmentasi (K-Means) memakai normalisasi terbaik
# ----------------------------------------------------------------------------
def compare_clustering(X, scaler_name, out):
    Z = GradeMedianImputer().fit(X).transform(X)
    num = SCALERS[scaler_name]().fit_transform(Z[NUM_COLS])
    Z = np.hstack([num, Z[BIN_COLS].values])
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(Z), size=min(8000, len(Z)), replace=False)
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        rows.append({"k": k, "SSE": km.inertia_,
                     "silhouette": silhouette_score(Z[idx], km.labels_[idx]),
                     "davies_bouldin": davies_bouldin_score(Z, km.labels_)})
    df = pd.DataFrame(rows).set_index("k")
    save_table(df, out, "5_segmentasi_kmeans")
    print(f"[Pilihan] k terbaik menurut silhouette = {df['silhouette'].idxmax()}, "
          f"menurut Davies-Bouldin = {df['davies_bouldin'].idxmin()}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, col, t in zip(ax, df.columns, ["SSE (elbow)", "Silhouette (maks)", "Davies-Bouldin (min)"]):
        a.plot(df.index, df[col], "o-"); a.set_title(t); a.set_xlabel("k")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "5_segmentasi_kmeans.png"), dpi=140)
    plt.close()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="credit_risk_dataset.csv")
    ap.add_argument("--out", default="hasil")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--boot", type=int, default=20)
    ap.add_argument("--model-out", default="model",
                    help="folder untuk menyimpan model.joblib siap pakai")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y = load_and_clean(args.data)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    pos_w = (ytr == 0).sum() / (ytr == 1).sum()
    print(f"[Split] train={len(Xtr)}  test={len(Xte)}  scale_pos_weight={pos_w:.2f}")

    print("\n##### 1. NORMALISASI #####")
    best_scaler = compare_normalization(Xtr, ytr, pos_w, args.out)

    print("\n##### 2. KLASIFIKASI #####")
    best_model = compare_classifiers(Xtr, ytr, Xte, yte, best_scaler, pos_w, args.out)

    print("\n##### 3. VALIDASI #####")
    compare_validation(Xtr, ytr, Xte, yte, best_scaler, best_model, pos_w, args.out,
                       args.k, args.repeats, args.boot)
    final_evaluation(Xtr, ytr, Xte, yte, best_scaler, best_model, pos_w, args.out)
    save_deployable(Xtr, ytr, best_scaler, best_model, pos_w, args.model_out)

    print("\n##### 4. SEGMENTASI #####")
    compare_clustering(X, best_scaler, args.out)

    print(f"\nSelesai. Semua tabel (CSV) & grafik (PNG) ada di folder '{args.out}/'.")


if __name__ == "__main__":
    main()