"""Aplikasi web sederhana untuk menilai risiko kredit nasabah baru.

Jalankan:
    streamlit run app.py

Model yang dipakai: pipeline lengkap (imputasi -> Z-Score -> one-hot -> XGBoost)
ditambah kalibrasi Platt, disimpan di model/model.joblib. Bila file model belum
ada, aplikasi akan melatihnya otomatis dari credit_risk_dataset.csv.
"""
import numpy as np
import pandas as pd
import streamlit as st
import requests
import predictor

API_URL = "http://localhost:8000"


@st.cache_resource(show_spinner=False)
def get_bundle():
    return predictor.load_bundle()


st.set_page_config(page_title="Evaluasi Risiko Kredit", layout="wide")

with st.spinner("Menyiapkan model..."):
    bundle = get_bundle()

st.title("Sistem Evaluasi Risiko Kredit")
st.caption("Prediksi risiko gagal bayar nasabah baru — model XGBoost + Z-Score "
           "dengan probabilitas terkalibrasi (Platt scaling).")

# ---------------------------------------------------------------------------
# Sidebar: info model + pengaturan zona keputusan
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Informasi Model")
    st.write(f"**Algoritma:** {bundle['model']}")
    st.write(f"**Normalisasi:** {bundle['scaler']}")
    st.write("**Kalibrasi:** Platt scaling (sigmoid)")
    st.write(f"**ROC-AUC (OOF latih):** {bundle['oof_auc']:.3f}")
    st.write(f"**PR-AUC (OOF latih):** {bundle['oof_pr_auc']:.3f}")
    st.write(f"**Proporsi gagal bayar data latih:** {bundle['base_rate']:.1%}")
    st.write(f"**Dilatih:** {bundle['trained_at']}")

    st.divider()
    st.header("Zona Keputusan")
    st.write("Geser batas sesuai kebijakan risiko. Default berasal dari data latih.")

    t_low_def = float(np.clip(round(bundle["t_low"], 2), 0.01, 0.95))
    t_high_def = float(np.clip(round(bundle["t_high"], 2), t_low_def + 0.03, 0.99))
    t_low = st.slider("Batas bawah — di bawah ini DISETUJUI", 0.01, 0.99,
                      t_low_def, 0.01)
    t_high = st.slider("Batas atas — di atas ini DITOLAK", 0.01, 0.99,
                       t_high_def, 0.01)
    if t_low >= t_high:
        st.error("Batas bawah harus lebih kecil dari batas atas.")
        st.stop()
    st.caption(f"Disetujui: PD < {t_low:.0%} | Review: {t_low:.0%}–{t_high:.0%} "
               f"| Ditolak: PD >= {t_high:.0%}")

    with st.expander("Catatan penting"):
        st.write(
            "Probabilitas sudah dikalibrasi, tetapi dataset ini memiliki tingkat "
            f"gagal bayar {bundle['base_rate']:.1%} — jauh di atas portofolio bank "
            "nyata (~1–5%). Karena itu angka PD di sini tidak bisa langsung "
            "dibandingkan dengan cutoff bank seperti 4–10%. Threshold default "
            "dihitung dari data latih (F1 terbaik untuk batas bawah, precision "
            ">= 90% untuk batas atas), bukan angka standar industri.")

# ---------------------------------------------------------------------------
# Form input nasabah
# ---------------------------------------------------------------------------
with st.form("form_nasabah"):
    st.subheader("Data Nasabah")
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Usia (tahun)", min_value=18, max_value=100,
                              value=30, step=1)
        income = st.number_input("Pendapatan tahunan", min_value=1000,
                                 value=55_000, step=1000, format="%d")
        emp = st.number_input("Lama kerja (tahun) — kosongkan jika tidak diketahui",
                              min_value=0.0, max_value=60.0, value=None,
                              step=0.5, format="%.1f")
        home = st.selectbox("Status kepemilikan rumah", predictor.HOME_OPTIONS,
                            index=predictor.HOME_OPTIONS.index("RENT"))
        hist = st.number_input("Panjang riwayat kredit (tahun)", min_value=0,
                               max_value=50, value=4, step=1)
        default_file = st.selectbox("Pernah gagal bayar tercatat (Y/N)",
                                    predictor.DEFAULT_OPTIONS)

    with c2:
        intent = st.selectbox("Tujuan pinjaman", predictor.INTENT_OPTIONS)
        grade = st.selectbox("Grade pinjaman", predictor.GRADE_OPTIONS, index=3)
        amnt = st.number_input("Jumlah pinjaman", min_value=500,
                               max_value=100_000_000, value=8_000, step=500,
                               format="%d")
        int_rate = st.number_input("Suku bunga (%) — kosongkan jika tidak diketahui",
                                   min_value=0.0, max_value=30.0, value=None,
                                   step=0.01, format="%.2f")
        pct_income = st.number_input("Rasio pinjaman terhadap pendapatan (0–1)",
                                     min_value=0.0, max_value=1.0, value=0.15,
                                     step=0.01, format="%.2f")

    submitted = st.form_submit_button("Prediksi Risiko", type="primary")

if submitted:
    st.session_state["nasabah"] = {
        "person_age": int(age),
        "person_income": int(income),
        "person_emp_length": None if emp is None else float(emp),
        "loan_amnt": int(amnt),
        "loan_int_rate": None if int_rate is None else float(int_rate),
        "loan_percent_income": float(pct_income),
        "cb_person_cred_hist_length": int(hist),
        "person_home_ownership": home,
        "loan_intent": intent,
        "loan_grade": grade,
        "cb_person_default_on_file": default_file,
    }

# ---------------------------------------------------------------------------
# Hasil prediksi (tetap tampil walau slider threshold digeser)
# ---------------------------------------------------------------------------
row = st.session_state.get("nasabah")
if row:
    st.divider()
    st.subheader("Hasil Prediksi")
    payload = {
        **row,
        "threshold": {"t_low": float(t_low), "t_high": float(t_high)}
    }
    try:
        # 1. Kirim data ke Backend API (Otomatis disimpan ke PostgreSQL oleh api.py)
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        
        if response.status_code == 200:
            hasil = response.json()
        else:
            st.error(f"Gagal memproses prediksi dari API: {response.text}")
            st.stop()
            
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Backend API (api.py) belum dijalankan! Jalankan `uvicorn api:app --port 8000` terlebih dahulu.")
        st.stop()

    pd_cal = hasil["probabilitas_gagal_bayar"]
    raw = hasil["skor_mentah"]
    status = hasil["zona"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimasi probabilitas gagal bayar (PD)", f"{pd_cal:.1%}")
    m2.metric("Zona saat ini", {
        "approve": "Disetujui",
        "review": "Review manual",
        "decline": "Ditolak"}[status])
    m3.metric("Batas aktif", f"{t_low:.0%} / {t_high:.0%}")
    st.progress(float(np.clip(pd_cal, 0.0, 1.0)))

    for pesan in hasil.get("peringatan", []):
        st.warning(pesan)

    if status == "approve":
        st.success(f"DISETUJUI — risiko rendah. PD {pd_cal:.1%} berada di bawah "
                   f"batas {t_low:.0%}.")
        st.write("Berdasarkan model, nasabah ini tergolong aman untuk diberikan pinjaman.")
    elif status == "review":
        st.warning(f"REVIEW MANUAL — risiko menengah. PD {pd_cal:.1%} berada di "
                   f"antara {t_low:.0%} dan {t_high:.0%}.")
        st.write("Model belum cukup yakin. Dalam praktik bank, kasus seperti ini "
                 "diteruskan ke analis untuk penilaian tambahan (dokumen, agunan, "
                 "penghasilan terverifikasi, dan lain-lain).")
    else:
        st.error(f"DITOLAK — risiko tinggi. PD {pd_cal:.1%} melampaui batas "
                 f"{t_high:.0%}.")
        st.write("Model merekomendasikan penolakan karena estimasi gagal bayar "
                 "berada di atas ambang risiko yang ditetapkan.")

    with st.expander("Detail teknis prediksi"):
        st.write(f"Skor mentah model (belum terkalibrasi): `{raw:.4f}`")
        st.write(f"PD terkalibrasi (Platt scaling): `{pd_cal:.4f}`")
        st.write(f"Threshold aktif: disetujui < {t_low:.2f}; review "
                 f"{t_low:.2f}–{t_high:.2f}; ditolak >= {t_high:.2f}")
        for catatan in hasil.get("catatan", []):
            st.write(catatan)

    with st.expander("Data yang diprediksi"):
        tampil = pd.DataFrame([row]).T.reset_index()
        tampil.columns = ["Atribut", "Nilai"]
        tampil["Nilai"] = tampil["Nilai"].astype(str)
        st.dataframe(tampil, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Menampilkan Riwayat Pengajuan dari Database PostgreSQL
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📋 Riwayat Pengajuan Kredit (Live dari PostgreSQL)")

try:
    hist_resp = requests.get(f"{API_URL}/history?limit=10", timeout=5)
    if hist_resp.status_code == 200:
        data_histori = hist_resp.json()
        if data_histori:
            df_hist = pd.DataFrame(data_histori)
            # Tampilkan kolom-kolom penting
            kolom_tampil = ["id", "created_at", "person_income", "loan_amnt", "probabilitas_gagal_bayar", "keputusan", "risiko"]
            kolom_tersedia = [c for c in kolom_tampil if c in df_hist.columns]
            st.dataframe(df_hist[kolom_tersedia], width="stretch", hide_index=True)
        else:
            st.info("Belum ada riwayat pengajuan di database.")
except Exception:
    st.caption("Tidak dapat memuat riwayat database.")