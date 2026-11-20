# Daftar Istilah — Proyek Credit Risk

Dataset: `credit_risk_dataset.csv` | Target: `loan_status` (1 = gagal bayar)

| Istilah | Penjelasan | Kepanjangan | Arti |
|---|---|---|---|
| **TP** | Jumlah kasus gagal bayar yang benar diprediksi gagal bayar | True Positive | Tebakan positif yang benar |
| **TN** | Jumlah kasus lancar yang benar diprediksi lancar | True Negative | Tebakan negatif yang benar |
| **FP** | Kasus lancar tapi diprediksi gagal bayar (salah tuduh) | False Positive | Positif palsu / alarm palsu |
| **FN** | Kasus gagal bayar tapi diprediksi lancar (paling berbahaya) | False Negative | Negatif palsu / risiko lolos |
| **Accuracy** | Persentase semua tebakan yang benar dari total data | — | Ketepatan keseluruhan |
| **Precision** | Dari yang diprediksi gagal bayar, berapa persen yang benar | — | Ketepatan saat "menuduh" |
| **Recall** | Dari yang aktual gagal bayar, berapa persen yang tertangkap | Sensitivity / TPR | Daya tangkap kasus positif |
| **F1-Score** | Rata-rata harmonik precision dan recall | — | Keseimbangan precision–recall |
| **Confusion Matrix** | Tabel 2×2 berisi TP, TN, FP, FN | — | Ringkasan hasil tebakan |
| **TPR** | Proporsi positif yang tertangkap = recall | True Positive Rate | Rasio tangkapan |
| **FPR** | Proporsi negatif yang salah ditolak | False Positive Rate | Rasio alarm palsu |
| **ROC** | Kurva TPR vs FPR pada semua ambang | Receiver Operating Characteristic | Grafik trade-off model |
| **AUC** | Luas area di bawah kurva ROC, rentang 0–1 | Area Under the Curve | Ukuran kualitas pemisahan |
| **ROC-AUC** | Peluang model memberi skor lebih tinggi ke kasus positif acak dibanding negatif acak | — | Kemampuan ranking model |
| **PR-AUC** | Luas area di bawah kurva precision–recall | Precision-Recall Area Under the Curve | Kualitas model pada data imbalanced |
| **AP** | Bentuk lain dari PR-AUC | Average Precision | Rata-rata precision di tiap level recall |
| **KS** | Jarak maksimum antara TPR dan FPR | Kolmogorov-Smirnov | Metrik pemisahan khas credit scoring |
| **Threshold** | Ambang probabilitas untuk memutuskan kelas (default 0,5) | — | Batas keputusan |
| **Class Imbalance** | Kondisi kelas tidak seimbang (78% lancar vs 22% gagal bayar) | — | Ketidakseimbangan kelas |
| **CV** | Validasi dengan membagi data bergiliran | Cross-Validation | Uji silang |
| **K-Fold CV** | Data dibagi k bagian; tiap bagian jadi data uji bergiliran | k-Fold Cross-Validation | Validasi k lipatan |
| **Stratified** | Tiap lipatan menjaga proporsi kelas | — | Berlapis sesuai proporsi |
| **Holdout** | Sekali bagi data jadi latih dan uji (mis. 75/25) | — | Pembagian tetap |
| **Random Subsampling** | Holdout diulang beberapa kali dengan pembagian acak | — | Sampel acak berulang |
| **Bootstrap** | Sampling dengan pengembalian | — | Resampling acak |
| **OOB** | Data yang tidak terambil saat bootstrap, dipakai untuk uji | Out-of-Bag | Sampel di luar kantong |
| **Bootstrap .632** | Gabungan 0,368 × skor latih + 0,632 × skor OOB | — | Estimasi gabungan |
| **Std** | Sebaran nilai antar lipatan; kecil = stabil | Standard Deviation | Simpangan baku |
| **Overfitting** | Model hafal data latih, gagal di data baru | — | Terlalu menyesuaikan data latih |
| **Data Leakage** | Informasi data uji bocor ke proses latih | — | Kebocoran data |
| **Normalisasi** | Menyamakan skala fitur | — | Penyetaraan skala |
| **Min-Max** | `(x − min)/(max − min)` → rentang 0–1 | — | Skala minimum–maksimum |
| **Z-Score** | `(x − mean)/std` → mean 0, std 1 | — | Skor baku |
| **Decimal Scaling** | `x / 10^j` agar nilai absolut < 1 | — | Penskalaan desimal |
| **Sigmoidal** | Menekan nilai ke rentang (−1, 1) | — | Penskalaan sigmoid |
| **Softmax** | Menekan nilai ke rentang (0, 1) | — | Penskalaan softmax |
| **One-Hot Encoding** | Mengubah kategori jadi kolom 0/1 | — | Pengodean kategori |
| **Imputation** | Mengisi nilai yang hilang | — | Penambalan missing value |
| **Logistic Regression** | Model linier untuk klasifikasi biner | — | Regresi logistik |
| **Decision Tree** | Model pohon keputusan berbasis aturan | — | Pohon keputusan |
| **k-NN** | Klasifikasi berdasarkan k tetangga terdekat | k-Nearest Neighbors | k tetangga terdekat |
| **Random Forest** | Kumpulan banyak pohon keputusan | — | Hutan acak |
| **XGBoost** | Gradient boosting yang dioptimalkan dan cepat | Extreme Gradient Boosting | Boosting gradien ekstrem |
| **K-Means** | Clustering menjadi k kelompok berdasarkan jarak | — | Pengelompokan k rata-rata |
| **SSE** | Total jarak kuadrat data ke pusat cluster; dipakai metode elbow | Sum of Squared Errors | Jumlah kuadrat error |
| **Silhouette** | Skor kualitas cluster (−1 s.d. 1; makin tinggi makin baik) | Silhouette Score | Skor siluet |
| **Davies-Bouldin** | Rasio sebaran dalam cluster vs jarak antar cluster; makin kecil makin baik | Davies-Bouldin Index | Indeks Davies-Bouldin |
| **Hyperparameter** | Pengaturan model yang ditentukan sebelum latih (mis. `max_depth`) | — | Parameter konfigurasi |
| **Feature** | Kolom/variabel input model | — | Fitur/atribut |
| **Target/Label** | Kolom yang diprediksi (`loan_status`) | — | Variabel sasaran |
