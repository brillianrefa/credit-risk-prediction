# Sistem Evaluasi Risiko & Segmentasi Kredit Nasabah

Proyek machine learning untuk memprediksi risiko **kredit macet** (gagal bayar)
nasabah, membandingkan metode normalisasi, algoritma klasifikasi, metode validasi,
serta melakukan segmentasi nasabah.

Hasil akhir proyek adalah **model siap pakai** (`model/model.joblib`) yang dapat
diintegrasikan ke platform melalui **REST API** (`api.py`) atau diuji lewat
**aplikasi web** (`app.py`).

- Dataset: `credit_risk_dataset.csv`
- Target: `loan_status` — `1` = gagal bayar, `0` = lancar
- Model terpilih: **XGBoost + Z-Score + kalibrasi Platt**
- Deliverable: artefak model, API, aplikasi demo, dokumentasi

---

## Daftar Isi

1. [Struktur Proyek](#struktur-proyek)
2. [Dataset](#dataset)
3. [Praproses Data](#praproses-data)
4. [Pertimbangan Pemilihan Metode](#pertimbangan-pemilihan-metode)
   - [Normalisasi](#1-normalisasi)
   - [Klasifikasi](#2-klasifikasi)
   - [Validasi](#3-validasi)
   - [Threshold](#4-threshold-keputusan)
   - [Segmentasi](#5-segmentasi-k-means)
   - [Kalibrasi](#6-kalibrasi-probabilitas)
5. [Hasil Akhir](#hasil-akhir)
6. [Deliverable Model untuk Backend](#deliverable-model-untuk-backend)
7. [Cara Menjalankan](#cara-menjalankan)
8. [Keterbatasan & Rekomendasi](#keterbatasan--rekomendasi)

---

## Struktur Proyek

```
project_ml_uts/
├── credit_risk_dataset.csv    # dataset mentah
├── main.py                    # eksperimen lengkap + simpan model
├── predictor.py               # inti prediksi (dipakai API & Streamlit)
├── api.py                     # REST API FastAPI
├── app.py                     # aplikasi demo Streamlit
├── requirements.txt
├── README.md                  # dokumen ini
├── API_DOCUMENTATION.md       # dokumentasi endpoint untuk tim backend
├── istilah.md                 # kamus istilah metrik & ML
├── model/
│   └── model.joblib           # pipeline + kalibrasi + threshold (artefak)
└── hasil/                     # tabel CSV & grafik PNG hasil eksperimen
```

---

## Dataset

`credit_risk_dataset.csv` berisi **32.581 baris × 12 kolom** data pengajuan pinjaman.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `person_age` | numerik | Usia nasabah |
| `person_income` | numerik | Pendapatan tahunan |
| `person_home_ownership` | kategori | Status kepemilikan rumah |
| `person_emp_length` | numerik | Lama kerja (tahun) |
| `loan_intent` | kategori | Tujuan pinjaman |
| `loan_grade` | kategori | Grade pinjaman A–G |
| `loan_amnt` | numerik | Jumlah pinjaman |
| `loan_int_rate` | numerik | Suku bunga (%) |
| `loan_status` | biner | **Target**: 1 = gagal bayar |
| `loan_percent_income` | numerik | Rasio pinjaman / pendapatan |
| `cb_person_default_on_file` | kategori | Riwayat gagal bayar (Y/N) |
| `cb_person_cred_hist_length` | numerik | Panjang riwayat kredit |

**Makna `loan_grade`:** peringkat risiko pinjaman dari pemberi pinjaman —
**A = risiko terendah (terbaik), G = risiko tertinggi (terburuk)**. Makin buruk
grade, bunga makin tinggi dan makin sering gagal bayar:

| Grade | Rata-rata bunga | Rasio gagal bayar | Jumlah |
|---|---|---|---|
| A | 7,33% | 10% | 10.777 |
| B | 11,00% | 16% | 10.451 |
| C | 13,46% | 21% | 6.458 |
| D | 15,36% | 59% | 3.626 |
| E | 17,01% | 64% | 964 |
| F | 18,61% | 71% | 241 |
| G | 20,25% | 98% | 64 |

**Karakteristik penting:**

- **Kelas tidak seimbang**: 78,2% lancar vs **21,8% gagal bayar**.
- **Missing value**: `loan_int_rate` (3.116 baris) dan `person_emp_length` (895 baris).
- **Data tidak wajar**: usia hingga 144 tahun, lama kerja hingga 123 tahun.
- **Duplikat**: 165 baris.
- **Outlier**: pendapatan hingga 6.000.000 (jauh di atas median 55.000).

Ketidakseimbangan kelas menjadi alasan utama **accuracy tidak bisa dijadikan
metrik utama** — model yang menebak "semua lancar" sudah mendapat akurasi 78,2%.

**Satuan & batas input:** dataset tidak menyebut mata uang secara eksplisit;
nilainya lazim dianggap dolar (pendapatan rata-rata ~66.000/tahun, pinjaman
500–35.000). Proyek ini **tidak melakukan konversi mata uang** — seluruh input
dianggap berada pada satuan yang sama dengan data latih. Batas input prototype
longgar (pendapatan tanpa batas atas, pinjaman maks 100.000.000, riwayat kredit
0–50); nilai yang melewati rentang data latih memicu **peringatan ekstrapolasi**.

---

## Praproses Data

### Pembersihan

| Aturan | Alasan |
|---|---|
| Hapus duplikat | Mencegah model menghafal baris yang sama |
| Usia ≤ 100 tahun | Usia > 100 tidak masuk akal untuk pengajuan kredit |
| Lama kerja ≤ 60 dan ≤ (usia − 14) | Lama kerja tidak mungkin melebihi usia produktif |
| **Tidak menghapus** baris dengan missing value | Informasi tetap berguna; missing diimputasi |

Hasil: **32.581 → 32.409 baris** (proporsi gagal bayar tetap 21,9%).

### Imputasi (di dalam pipeline)

| Kolom | Metode | Alasan |
|---|---|---|
| `loan_int_rate` | Median **per `loan_grade`** | Suku bunga sangat ditentukan grade pinjaman |
| `person_emp_length` | Median global | Distribusi miring; median lebih tahan outlier |

### Rekayasa fitur

- `loan_grade_ord` — grade A–G diubah menjadi 0–6 (urutan alami).
- `cb_default_bin` — `cb_person_default_on_file` menjadi 0/1.

### Encoding & pembagian data

- Kategori (`person_home_ownership`, `loan_intent`) → **one-hot encoding**.
- Split **80% latih / 20% uji**, **stratified** agar proporsi kelas sama.
- **Semua preprocessing di-`fit` hanya pada data latih** (di dalam `Pipeline`),
  lalu diterapkan ke data uji. Ini mencegah **data leakage**.

---

## Pertimbangan Pemilihan Metode

### 1. Normalisasi

**Mengapa dibandingkan?** Karena model berbasis jarak (k-NN) dan model linier
(Logistic Regression) sensitif terhadap skala. Contoh: `person_income` bernilai
puluhan ribu–juta, sedangkan `loan_percent_income` hanya 0–1. Tanpa normalisasi,
pendapatan akan mendominasi perhitungan jarak/koefisien.

Lima metode diuji (rumus sesuai materi):

| Metode | Rumus | Rentang hasil |
|---|---|---|
| Min-Max | `(x − min) / (max − min)` | [0, 1] |
| Z-Score | `(x − mean) / std` | mean 0, std 1 |
| Decimal Scaling | `x / 10^j`, `j` terkecil agar `|x'| < 1` | (−1, 1) |
| Sigmoidal | `(1 − e^-a) / (1 + e^-a)`, `a = (x − mean)/std` | (−1, 1) |
| Softmax | `1 / (1 + e^-a)`, `a = (x − mean)/(std / 2π)` | (0, 1) |

Diuji pada Logistic Regression, k-NN, dan Decision Tree (sebagai kontrol).
Rata-rata ROC-AUC LR + k-NN:

| Peringkat | Normalisasi | ROC-AUC | F1 |
|---|---|---|---|
| **1** | **Z-Score** | **0,8761** | 0,6576 |
| 2 | Decimal Scaling | 0,8736 | 0,6670 |
| 3 | Min-Max | 0,8715 | 0,6500 |
| 4 | Sigmoidal | 0,8703 | 0,6423 |
| 5 | Softmax | 0,8282 | 0,5427 |

**Keputusan: Z-Score.**

- Menjaga bentuk sebaran dan tidak "tertarik" outlier ekstrem seperti Min-Max.
- Softmax terburuk karena pembagi `std / 2π` membuat kurva terlalu tajam —
  nilai menumpuk di 0/1 sehingga informasi jarak hilang.
- **Catatan:** Decision Tree menghasilkan angka **identik** di semua normalisasi
  (ROC-AUC 0,90385). Ini membuktikan model berbasis pohon tidak peduli skala —
  normalisasi hanya penting untuk model linier/jarak.

### 2. Klasifikasi

Lima algoritma dibandingkan dengan alasan pemilihan yang berbeda:

| Model | Peran / alasan dipilih |
|---|---|
| **Logistic Regression** | Baseline linier, mudah diinterpretasi (koefisien) |
| **Decision Tree** | Aturan if-then yang mudah dijelaskan; kontrol non-linier |
| **k-NN** | Baseline non-parametrik berbasis kemiripan nasabah |
| **Random Forest** | Ensemble bagging, tahan outlier & fitur saling berkorelasi |
| **XGBoost** | Ensemble boosting, biasanya terbaik untuk data tabular |

Hyperparameter utama: LR `class_weight="balanced"`; DT `max_depth=6`,
`min_samples_leaf=30`; k-NN `n_neighbors=25`; RF 200 pohon,
`min_samples_leaf=3`; XGBoost 300 pohon, `max_depth=5`, `learning_rate=0.08`,
`scale_pos_weight` untuk kelas tidak seimbang.

**Hasil cross-validation 5-fold (data latih):**

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|
| Logistic Regression | 0,8636 | 0,7025 | 0,6189 | 0,7792 | 0,5133 | 0,7901 |
| Decision Tree | 0,9039 | 0,8345 | 0,7553 | 0,7439 | 0,7714 | 0,8942 |
| k-NN | 0,8885 | 0,7941 | 0,6962 | 0,5621 | 0,9147 | 0,8927 |
| Random Forest | 0,9314 | 0,8827 | 0,8174 | 0,7257 | 0,9356 | 0,9291 |
| **XGBoost** | **0,9480** | **0,9035** | 0,8085 | 0,7951 | 0,8224 | 0,9176 |

**Hasil pada data uji (holdout 20%, threshold 0,5):**

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | Accuracy | KS |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0,8648 | 0,6900 | 0,6231 | 0,7842 | 0,5170 | 0,7925 | 0,582 |
| Decision Tree | 0,9072 | 0,8406 | 0,7568 | 0,7602 | 0,7533 | 0,8931 | 0,702 |
| k-NN | 0,8957 | 0,8028 | 0,7065 | 0,5755 | 0,9148 | 0,8954 | 0,649 |
| Random Forest | 0,9380 | 0,8927 | 0,8288 | 0,7377 | 0,9458 | 0,9334 | 0,744 |
| **XGBoost** | **0,9544** | **0,9139** | 0,8236 | 0,8230 | 0,8242 | 0,9229 | **0,780** |

**Keputusan: XGBoost.**

- ROC-AUC dan PR-AUC tertinggi di CV **dan** data uji (0,948 → 0,954).
- KS 0,780 — pemisahan kelas sangat kuat (standar industri credit scoring).
- PR-AUC 0,914 jauh di atas baseline acak (0,218), artinya model benar-benar
  belajar pola kelas minoritas.
- Random Forest adalah alternatif terdekat: F1 sedikit lebih tinggi pada
  threshold 0,5, tetapi ranking (AUC/PR-AUC/KS) lebih rendah dan lebih lambat.
- Logistic Regression hanya 0,865 — akurasinya 79% hampir menyamai tebakan
  "semua lancar" (78%), sehingga tidak layak menjadi model utama.

### 3. Validasi

Empat metode validasi dibandingkan untuk melihat mana yang paling dekat dengan
performa sebenarnya (data uji). Model yang divalidasi: XGBoost + Z-Score.

| Metode | ROC-AUC | Std | Selisih vs data uji | Waktu |
|---|---|---|---|---|
| Holdout (75/25) | 0,9498 | — | 0,0046 | ~1 s |
| Random Subsampling (10×) | 0,9463 | 0,0030 | 0,0081 | ~12 s |
| **K-Fold CV (k=5)** | **0,9480** | **0,0028** | 0,0064 | ~11 s |
| Bootstrap OOB (B=20) | 0,9442 | 0,0023 | 0,0102 | ~41 s |
| Bootstrap .632 (B=20) | 0,9582 | — | 0,0038 | — |

AUC data uji sesungguhnya = **0,9544**.

**Keputusan: Stratified K-Fold CV (k=5).**

- Standar, stabil (std hanya 0,0028), dan memakai seluruh data latih.
- Selisih dengan data uji kecil (0,006), jadi estimasi bisa dipercaya.
- Holdout hanya sekali pembagian → rentan keberuntungan.
- Bootstrap .632 paling dekat secara angka, tetapi rumusnya lebih rumit dan
  jarang dipakai untuk pelaporan; cukup sebagai pembanding.
- **Leave-One-Out tidak dipakai**: butuh satu model per baris (~26 ribu model),
  terlalu lambat.

### 4. Threshold Keputusan

Model mengeluarkan probabilitas; threshold menentukan batas "gagal bayar".
Default 0,5 tidak selalu optimal. Threshold dipilih dari **out-of-fold data latih**
(tanpa melihat data uji) dengan memaksimalkan F1:

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0,50 | 0,9229 | 0,8242 | 0,8230 | 0,8236 |
| **0,72 (tuned)** | **0,9386** | **0,9679** | 0,7440 | **0,8413** |

**Keputusan:** memakai threshold hasil tuning. Konsekuensinya: bank hampir tidak
pernah menolak nasabah baik (precision 97%), tetapi sebagian kasus ragu
(probabilitas 50–71%) tetap diterima (recall turun ke 74%). Threshold adalah
**keputusan kebijakan**, bukan angka statistik murni.

### 5. Segmentasi K-Means

Segmentasi dilakukan untuk memahami kelompok nasabah (bukan untuk prediksi).
K-Means diuji `k = 2..8` dengan tiga metrik:

| k | SSE | Silhouette | Davies-Bouldin |
|---|---|---|---|
| 2 | 217.400 | 0,2003 | 1,9445 |
| 3 | 183.217 | 0,2116 | 1,6099 |
| 4 | 161.949 | 0,2156 | 1,4899 |
| 5 | 149.302 | 0,2209 | 1,5097 |
| **6** | **139.004** | **0,2246** | **1,3919** |
| 7 | 129.813 | 0,2024 | 1,4404 |
| 8 | 122.929 | 0,1774 | 1,4738 |

**Keputusan: k = 6** — silhouette tertinggi dan Davies-Bouldin terendah.

**Catatan jujur:** silhouette 0,225 tergolong **lemah–sedang** (ideal > 0,5).
Segmen ini cocok untuk eksplorasi, bukan klaim segmentasi yang tegas.

### 6. Kalibrasi Probabilitas

**Masalah:** XGBoost dilatih dengan `scale_pos_weight` untuk menangani kelas tidak
seimbang. Akibatnya skor mentah **bukan probabilitas sebenarnya** sehingga tidak
bisa disebut PD (Probability of Default).

**Solusi:** **Platt scaling** (regresi logistik pada logit skor out-of-fold).
Setelah kalibrasi:

- Rata-rata PD data uji = **22,2%** ≈ base rate data uji **21,9%** → terkalibrasi.
- ROC-AUC tetap **0,9544**, PR-AUC **0,9139** (ranking tidak berubah).
- Threshold optimal bergeser dari 0,72 (skor mentah) menjadi **0,54** (PD).
- Zona keputusan final: disetujui PD < 0,54; review manual 0,54–0,64; ditolak ≥ 0,64.
  Batas atas 0,64 dipilih saat precision ≥ 90%.

**Penting:** dataset ini memiliki base rate 21,9%, jauh di atas portofolio bank
nyata (~1–5%). Angka PD tidak bisa langsung dibandingkan dengan cutoff bank
seperti 4–10%. Untuk produksi, model harus dikalibrasi ulang dengan data nyata.

---

## Hasil Akhir

| Komponen | Pilihan | Alasan singkat |
|---|---|---|
| Normalisasi | **Z-Score** | AUC tertinggi, tahan outlier |
| Klasifikasi | **XGBoost** | AUC 0,954, PR-AUC 0,914, KS 0,780 |
| Validasi | **Stratified K-Fold (k=5)** | Stabil (±0,003), dekat dengan data uji |
| Threshold | 0,54 / 0,64 (setelah kalibrasi) | Presisi 97% pada batas tolak |
| Segmentasi | K-Means k=6 | Silhouette & Davies-Bouldin terbaik |
| Kalibrasi | Platt scaling | PD rata-rata ≈ base rate |

---

## Deliverable Model untuk Backend

Model diserahkan dalam bentuk:

| Artefak | Fungsi |
|---|---|
| `model/model.joblib` | Pipeline lengkap: imputasi → Z-Score → one-hot → XGBoost + kalibrator + threshold |
| `predictor.py` | Fungsi `predict_one(features)` dan `model_info()` — bisa diimpor langsung |
| `api.py` | REST API FastAPI (`/health`, `/model-info`, `/predict`) |
| `API_DOCUMENTATION.md` | Dokumentasi endpoint, field, contoh request/response, error |
| `app.py` | Aplikasi demo Streamlit |

**Prinsip penting:** tim backend **tidak perlu** melakukan scaling/encoding.
Cukup kirim data mentah — seluruh preprocessing memakai parameter yang dipelajari
dari data latih dan tersimpan di `model.joblib`. Setiap respons prediksi juga
memuat `peringatan` bila input di luar rentang data latih.

### Alur prediksi data baru

```
Data nasabah baru (JSON / form)
   ↓ build_row: turunkan loan_grade_ord & cb_default_bin
   ↓ imputasi missing (median data latih)
   ↓ Z-Score (mean/std data latih)
   ↓ one-hot encoding (kategori data latih)
   ↓ XGBoost.predict_proba → skor mentah
   ↓ kalibrasi Platt → PD
   ↓ bandingkan dengan t_low / t_high
   → DISETUJUI / REVIEW_MANUAL / DITOLAK
```

---

## Cara Menjalankan

### Instalasi

```bash
pip install -r requirements.txt
```

### 1. Menjalankan eksperimen lengkap (opsional, beberapa menit)

```bash
python main.py --data credit_risk_dataset.csv --out hasil
```

Menghasilkan seluruh tabel CSV & grafik PNG di `hasil/`, dan otomatis menyimpan
model siap pakai ke `model/model.joblib`.

### 2. Menjalankan REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

- Swagger UI: <http://localhost:8000/docs>
- Dokumentasi lengkap: [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md)

### 3. Menjalankan aplikasi demo Streamlit

```bash
streamlit run app.py
```

Bila `model/model.joblib` belum ada, aplikasi/API akan melatih model secara
otomatis dari dataset (beberapa detik).

---

## Keterbatasan & Rekomendasi

**Keterbatasan:**

1. Dataset publik (Kaggle-style) dengan base rate 21,9% — bukan portofolio bank
   nyata, sehingga tingkat PD tidak realistis untuk produksi.
2. Validasi hanya acak (bukan berbasis waktu); performa pada periode ekonomi
   berbeda belum diuji.
3. Belum ada audit keadilan (fairness) terhadap atribut sensitif.
4. Segmentasi K-Means masih lemah (silhouette 0,225).
5. Threshold bersifat kebijakan; belum memperhitungkan biaya FN/FP, LGD, EAD,
   dan risk appetite bank.

**Rekomendasi lanjutan:**

1. Kalibrasi ulang dengan data portofolio nyata sebelum produksi.
2. Tambahkan monitoring drift dan evaluasi berkala.
3. Pertimbangkan strategi tiga zona (approve/review/decline) dengan analis manusia
   — sudah disediakan di API/app.
4. Bandingkan dengan model scorecard (WOE + logistic regression) yang lazim di
   industri untuk interpretabilitas regulasi.
5. Gunakan validasi berbasis waktu (out-of-time) untuk mengukur kestabilan model.
