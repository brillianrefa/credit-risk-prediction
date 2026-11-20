# Dokumentasi API Prediksi Risiko Kredit

REST API untuk memprediksi risiko gagal bayar nasabah. Ditujukan bagi tim backend
yang akan mengintegrasikan model ke platform.

- **Framework:** FastAPI
- **Model:** XGBoost + normalisasi Z-Score + kalibrasi Platt (sigmoid)
- **Artefak model:** `model/model.joblib`
- **Format input:** JSON dengan nama kolom asli dataset (tanpa `loan_status`)

## Menjalankan

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

- Swagger UI: <http://localhost:8000/docs>
- OpenAPI JSON: <http://localhost:8000/openapi.json>
- Model dimuat sekali saat startup. Setelah model dilatih ulang, **restart service**.

## Endpoint

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | Status service dan model |
| GET | `/model-info` | Metadata model dan daftar field input |
| POST | `/predict` | Prediksi satu nasabah |

### GET /health

```json
{"status": "ok", "model_loaded": true}
```

Bila model gagal dimuat: `{"status": "degraded", "model_loaded": false, "detail": "..."}`.

### GET /model-info

```json
{
  "model": "XGBoost",
  "normalisasi": "Z-Score",
  "kalibrasi": "Platt scaling (sigmoid)",
  "roc_auc_oof": 0.948,
  "pr_auc_oof": 0.9034,
  "base_rate_latih": 0.219,
  "trained_at": "2026-11-14 11:23:54",
  "threshold_default": {"t_low": 0.54, "t_high": 0.64},
  "input_wajib": ["person_age", "person_income", "person_home_ownership",
                  "loan_intent", "loan_grade", "loan_amnt",
                  "loan_percent_income", "cb_person_default_on_file",
                  "cb_person_cred_hist_length"],
  "input_opsional": ["person_emp_length", "loan_int_rate"],
  "nilai_kategori": {
    "person_home_ownership": ["MORTGAGE", "OTHER", "OWN", "RENT"],
    "loan_intent": ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
                    "MEDICAL", "PERSONAL", "VENTURE"],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"]
  }
}
```

## POST /predict

### Field input

| Field | Tipe | Wajib | Rentang / nilai | Keterangan |
|---|---|---|---|---|
| `person_age` | integer | ya | 18–100 | Usia (tahun) |
| `person_income` | number | ya | > 0 (tanpa batas atas) | Pendapatan tahunan |
| `person_home_ownership` | string | ya | `RENT` `OWN` `MORTGAGE` `OTHER` | Status kepemilikan rumah |
| `person_emp_length` | number \| null | tidak | 0–60 | Lama kerja (tahun). `null` = tidak diketahui → diimputasi median data latih |
| `loan_intent` | string | ya | `DEBTCONSOLIDATION` `EDUCATION` `HOMEIMPROVEMENT` `MEDICAL` `PERSONAL` `VENTURE` | Tujuan pinjaman |
| `loan_grade` | string | ya | `A`–`G` | Grade pinjaman: A = risiko terendah, G = risiko tertinggi (G hampir pasti gagal bayar di data latih) |
| `loan_amnt` | number | ya | > 0, ≤ 100.000.000 | Jumlah pinjaman |
| `loan_int_rate` | number \| null | tidak | 0–30 | Suku bunga (%). `null` = tidak diketahui → diimputasi median per grade |
| `loan_percent_income` | number | ya | 0–1 | Rasio pinjaman terhadap pendapatan |
| `cb_person_default_on_file` | string | ya | `Y` / `N` | Pernah gagal bayar tercatat |
| `cb_person_cred_hist_length` | integer | ya | 0–50 | Panjang riwayat kredit (tahun) |
| `threshold` | object | tidak | lihat di bawah | Override batas keputusan |

### Field response

| Field | Tipe | Keterangan |
|---|---|---|
| `probabilitas_gagal_bayar` | number | PD terkalibrasi, 0–1 |
| `skor_mentah` | number | Skor model sebelum kalibrasi |
| `keputusan` | string | `DISETUJUI` / `REVIEW_MANUAL` / `DITOLAK` |
| `zona` | string | `approve` / `review` / `decline` |
| `risiko` | string | `RENDAH` / `SEDANG` / `TINGGI` |
| `threshold_dipakai` | object | `t_low`, `t_high`, sumber (`model` / `override`) |
| `catatan` | array | Mis. informasi field yang diimputasi |
| `peringatan` | array | Pesan bila ada input di luar rentang data latih (prediksi ekstrapolasi) |

Aturan keputusan: `PD < t_low` → DISETUJUI; `t_low ≤ PD < t_high` → REVIEW_MANUAL;
`PD ≥ t_high` → DITOLAK. Default dari model: `t_low = 0.54`, `t_high = 0.64`.

### Contoh 1 — disetujui

Request:

```json
{
  "person_age": 35,
  "person_income": 100000,
  "person_home_ownership": "OWN",
  "person_emp_length": 10.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "A",
  "loan_amnt": 5000,
  "loan_int_rate": 7.0,
  "loan_percent_income": 0.05,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 10
}
```

Response (`200`):

```json
{
  "probabilitas_gagal_bayar": 0.000035,
  "skor_mentah": 0.000393,
  "keputusan": "DISETUJUI",
  "zona": "approve",
  "risiko": "RENDAH",
  "threshold_dipakai": {"t_low": 0.54, "t_high": 0.64, "sumber": "model"},
  "catatan": [],
  "peringatan": []
}
```

### Contoh 2 — review manual

Request:

```json
{
  "person_age": 26,
  "person_income": 30000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "D",
  "loan_amnt": 12000,
  "loan_int_rate": 12.0,
  "loan_percent_income": 0.14,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

Response (`200`):

```json
{
  "probabilitas_gagal_bayar": 0.6115,
  "skor_mentah": 0.7634,
  "keputusan": "REVIEW_MANUAL",
  "zona": "review",
  "risiko": "SEDANG",
  "threshold_dipakai": {"t_low": 0.54, "t_high": 0.64, "sumber": "model"},
  "catatan": [],
  "peringatan": []
}
```

### Contoh 3 — ditolak, dengan field kosong

Request:

```json
{
  "person_age": 25,
  "person_income": 30000,
  "person_home_ownership": "RENT",
  "person_emp_length": null,
  "loan_intent": "PERSONAL",
  "loan_grade": "D",
  "loan_amnt": 15000,
  "loan_int_rate": null,
  "loan_percent_income": 0.5,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

Response (`200`):

```json
{
  "probabilitas_gagal_bayar": 0.998,
  "skor_mentah": 0.9976,
  "keputusan": "DITOLAK",
  "zona": "decline",
  "risiko": "TINGGI",
  "threshold_dipakai": {"t_low": 0.54, "t_high": 0.64, "sumber": "model"},
  "catatan": [
    "person_emp_length kosong -> diimputasi median data latih",
    "loan_int_rate kosong -> diimputasi median per loan_grade"
  ],
  "peringatan": []
}
```

### Contoh 4 — peringatan di luar rentang data latih

Input yang melewati rentang data latih tetap diterima, tetapi respons berisi
peringatan `peringatan` (contoh: pendapatan 60 juta, padahal data latih
maksimal 2.039.784):

```json
{
  "probabilitas_gagal_bayar": 0.0442,
  "skor_mentah": 0.1423,
  "keputusan": "DISETUJUI",
  "zona": "approve",
  "risiko": "RENDAH",
  "threshold_dipakai": {"t_low": 0.54, "t_high": 0.64, "sumber": "model"},
  "catatan": [],
  "peringatan": [
    "person_income = 60000000 di luar rentang data latih (4080 - 2039784) - prediksi bersifat ekstrapolasi"
  ]
}
```

### Override threshold

Threshold bisa diganti per request (mis. mengikuti kebijakan risiko yang berbeda).
Nilai harus memenuhi `0 < t_low < t_high < 1`.

```json
{
  "...": "field lainnya sama",
  "threshold": {"t_low": 0.70, "t_high": 0.85}
}
```

Respons akan memuat `"threshold_dipakai": {"t_low": 0.70, "t_high": 0.85, "sumber": "override"}`.

### Contoh cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 35,
    "person_income": 100000,
    "person_home_ownership": "OWN",
    "person_emp_length": 10.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "A",
    "loan_amnt": 5000,
    "loan_int_rate": 7.0,
    "loan_percent_income": 0.05,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 10
  }'
```

### Contoh Python (backend)

```python
import requests

payload = {
    "person_age": 35,
    "person_income": 100000,
    "person_home_ownership": "OWN",
    "person_emp_length": 10.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "A",
    "loan_amnt": 5000,
    "loan_int_rate": 7.0,
    "loan_percent_income": 0.05,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 10,
}

respons = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
respons.raise_for_status()
hasil = respons.json()
print(hasil["keputusan"], f'{hasil["probabilitas_gagal_bayar"]:.1%}')
```

## Kode error

| Status | Arti |
|---|---|
| `200` | Prediksi berhasil |
| `400` | Threshold tidak valid (mis. `t_low >= t_high`) |
| `422` | Validasi input gagal (field wajib hilang, tipe salah, di luar rentang) |
| `500` | Kesalahan internal saat prediksi |
| `503` | Model belum tersedia / gagal dimuat |

Contoh respons `422`:

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "person_age"],
      "msg": "Input should be less than or equal to 100",
      "input": 150
    }
  ]
}
```

## Catatan penting

1. **Preprocessing otomatis.** Imputasi, Z-Score, dan one-hot encoding memakai
   parameter yang dipelajari dari data latih dan tersimpan di `model.joblib`.
   Tim backend cukup mengirim data mentah — tidak perlu melakukan scaling/encoding.
2. **PD bukan skor mentah.** Model dilatih dengan `scale_pos_weight` untuk data
   tidak seimbang, sehingga skor mentah bukan probabilitas sebenarnya. Kalibrasi
   Platt mengubahnya menjadi PD yang rata-ratanya mendekati base rate data
   (21,9% saat training).
3. **Base rate dataset tinggi.** Dataset ini memiliki 21,9% gagal bayar, jauh di
   atas portofolio bank nyata (~1–5%). Angka PD tidak bisa langsung dibandingkan
   dengan cutoff bank seperti 4–10%. Untuk penggunaan produksi, kalibrasi ulang
   dengan data portofolio nyata.
4. **Threshold default** (`0.54` / `0.64`) dihitung dari data latih: batas bawah
   memaksimalkan F1, batas atas dipilih saat precision ≥ 90%.
5. **Retrain model:** jalankan `python main.py` (menghasilkan `model/model.joblib`
   baru), lalu restart service API.
6. **CORS** saat ini terbuka (`*`) untuk memudahkan pengembangan. Untuk produksi,
   batasi `allow_origins` sesuai domain frontend.
7. **Satuan & rentang input.** Dataset tidak menyebut mata uang secara eksplisit
   dan tidak ada konversi apa pun — input harus berada pada satuan yang sama
   dengan data latih. Rentang data latih tersimpan di `model.joblib`; respons
   menyertakan `peringatan` bila ada input di luar rentang tersebut (prediksi
   tetap dihitung, tetapi bersifat ekstrapolasi).
