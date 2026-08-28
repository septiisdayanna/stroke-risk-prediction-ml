# Stroke Risk Prediction with Machine Learning

## Overview
Project ini membangun model klasifikasi machine learning untuk memprediksi risiko stroke pada seseorang berdasarkan data demografis dan riwayat kesehatan. Selain melatih model, project ini secara khusus fokus menangani tantangan **ketidakseimbangan kelas** yang umum terjadi pada data medis (kasus stroke hanya ~5% dari total data), melalui eksperimen bertahap: model baseline, SMOTE, hyperparameter tuning, hingga model ensemble.

## Background / Problem
Stroke adalah salah satu penyebab utama kematian dan kecacatan di dunia, termasuk di Indonesia. Deteksi dini terhadap faktor risiko dapat menurunkan angka kematian secara signifikan, namun sistem deteksi manual berbasis pemeriksaan klinis memiliki keterbatasan: bergantung pada dokter spesialis yang jumlahnya terbatas, lambat, dan sulit menjangkau populasi luas — terutama di daerah terpencil. Project ini mencoba menjawab kebutuhan akan sistem deteksi risiko stroke yang lebih cepat dan scalable menggunakan machine learning.

## Objectives
- Membangun model klasifikasi yang dapat memprediksi risiko stroke (tinggi/rendah) dari data kesehatan pasien.
- Mengatasi ketidakseimbangan kelas pada data (stroke hanya 4.9% dari total data).
- Mencapai target performa: Recall ≥ 60%, F1-score ≥ 30%, ROC-AUC ≥ 70% pada kelas stroke.
- Membandingkan beberapa algoritma dan strategi penanganan imbalance untuk menemukan model paling sesuai untuk konteks medis.

## Features
- **Eksperimen multi-algoritma** — membandingkan 5 algoritma klasifikasi (Naive Bayes, SVM, Random Forest, KNN, Decision Tree) sebagai baseline.
- **Penanganan class imbalance** — penerapan SMOTE dengan rasio 0.3 (bukan default 1.0), dipilih berdasarkan eksperimen perbandingan beberapa rasio, bukan asumsi.
- **Hyperparameter tuning** — GridSearchCV dengan 5-fold cross-validation, serta optimasi threshold keputusan berdasarkan Precision-Recall Curve (bukan default 0.5).
- **Model ensemble alternatif** — Balanced Random Forest sebagai pembanding pendekatan tanpa data sintetis (SMOTE).
- **Evaluasi menyeluruh** — Accuracy, Precision, Recall, F1-score, ROC-AUC, dan confusion matrix untuk setiap tahap eksperimen.

## Tech Stack
- **Bahasa:** Python
- **Data processing:** pandas, NumPy
- **Machine Learning:** scikit-learn (Naive Bayes, SVM, Random Forest, KNN, Decision Tree, GridSearchCV, MinMaxScaler, LabelEncoder)
- **Imbalanced data handling:** imbalanced-learn (`SMOTE`, `BalancedRandomForestClassifier`)
- **Visualisasi:** *(laporan menyebutkan bar chart, histogram, heatmap, boxplot, confusion matrix, ROC curve — [PERLU DIISI] library spesifik: matplotlib/seaborn)*

## How It Works / Methodology

**Dataset:** [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) — 5.110 baris data pasien dengan 12 fitur (usia, jenis kelamin, hipertensi, penyakit jantung, status pernikahan, jenis pekerjaan, tempat tinggal, kadar glukosa, BMI, status merokok, dan target `stroke`).

**1. Data Preparation**
- Menghapus kolom `id` (tidak informatif) dan `bmi` (banyak missing value, banyak outlier, korelasi sangat rendah terhadap target).
- Menghapus 1 baris dengan kategori `gender = "Other"` karena terlalu sedikit untuk dianalisis secara statistik.
- Menangani outlier pada `avg_glucose_level` menggunakan metode IQR dengan threshold konservatif (±2.5 × IQR), karena distribusinya right-skewed.
- Label encoding untuk seluruh fitur kategorikal, Min-Max Scaling untuk fitur numerik.
- Split data 80:20 dengan stratifikasi agar proporsi kelas stroke tetap terjaga di data train dan test.

**2. Eksperimen Model (4 Strategi)**
1. **Baseline** — 5 algoritma dilatih tanpa penanganan imbalance, untuk melihat performa dasar.
2. **SMOTE** — data training di-oversample dengan rasio 0.3 (dipilih setelah membandingkan rasio 0.1/0.3/0.5/1.0), lalu kelima model dilatih ulang.
3. **Hyperparameter Tuning** — GridSearchCV pada data asli (tanpa SMOTE) untuk menghindari bias dari data sintetis, dengan optimasi threshold berbasis Precision-Recall Curve.
4. **Ensemble** — Balanced Random Forest sebagai pendekatan alternatif tanpa SMOTE, dengan threshold tuning berbasis F1-score tertinggi.

**3. Pemilihan Model Final**
Naive Bayes (tanpa SMOTE, setelah tuning) dipilih sebagai model terbaik karena memiliki ROC-AUC tertinggi (0.82) dan keseimbangan recall-precision yang paling masuk akal untuk konteks klinis, dibanding Balanced Random Forest dan SVM yang diuji sebagai pembanding.

## Dataset
- **Sumber:** [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Jumlah data:** 5.110 baris, 12 fitur (termasuk target)
- **Distribusi kelas:** Stroke 4.9% vs Non-Stroke 95.1% (sangat tidak seimbang)
- **Insight EDA kunci:** usia memiliki korelasi tertinggi terhadap stroke (r = 0.25, risiko naik 4.2x pada usia > 60 tahun); riwayat hipertensi dan penyakit jantung masing-masing menaikkan risiko 3.5x dan 3.8x.

## Results / Output

Model final (**Naive Bayes, tanpa SMOTE, setelah hyperparameter tuning**) mencapai performa berikut pada data uji:

| Metrik | Hasil | Target Awal |
|---|---|---|
| ROC-AUC | **0.82** | ≥ 0.70 ✔ tercapai |
| F1-Score | **0.34** | ≥ 0.30 ✔ tercapai |
| Recall | **0.55** | ≥ 0.60 (belum tercapai, mendekati) |
| Precision | 0.24 | – |

Dibandingkan dua model lain yang diuji (Balanced Random Forest dan SVM tuned), Naive Bayes memberikan kombinasi ROC-AUC dan interpretabilitas terbaik untuk konteks medis, meski precision-nya masih tergolong rendah (relevan diketahui: dari prediksi "berisiko stroke", hanya sekitar 24% yang benar-benar berisiko — sehingga model ini lebih cocok sebagai alat *skrining awal* yang perlu ditindaklanjuti pemeriksaan lanjutan, bukan alat diagnosis final).

> Catatan: klaim dampak bisnis seperti "mempercepat deteksi 2.3x" atau "mengurangi pasien terlewat 40%" pada laporan merupakan proyeksi kualitatif berdasarkan interpretasi hasil model, bukan hasil pengukuran langsung di lapangan.

## Installation
```bash
git clone https://github.com/septiisdayanna/Proyek_Predictive_Analytics.git
cd Proyek_Predictive_Analytics
pip install pandas numpy scikit-learn imbalanced-learn
```

## Usage
Jalankan notebook untuk melihat alur lengkap dari data understanding hingga evaluasi:
```bash
jupyter notebook Proyek_Analisis_Prediksi_Septi_Isdayanna.ipynb
```
Atau jalankan versi script Python-nya secara langsung.

## Project Structure
```
Proyek_Predictive_Analytics/
├── Laporan Proyek Prediksi Risiko Stroke dengan Machine Learning - Septi Isdayanna.md   # Laporan lengkap (domain proyek → evaluasi)
├── Proyek_Analisis_Prediksi_Septi_Isdayanna.ipynb                                       # Notebook end-to-end
└── Proyek_Analisis_Prediksi_Septi_Isdayanna.py                                          # Versi script Python
```
> Untuk detail lengkap — termasuk seluruh tabel eksperimen (baseline, SMOTE, tuning, ensemble) dan referensi medis yang digunakan — baca [laporan lengkap di sini](<https://github.com/septiisdayanna/stroke-risk-prediction-ml/blob/b9a03993946e04a7bfc89a609268d9bca8899c17/Laporan%20Proyek%20Prediksi%20Risiko%20Stroke%20dengan%20Machine%20Learning%20-%20Septi%20Isdayanna.md>).

## Limitations
- Precision model final masih rendah (0.24), artinya model menghasilkan cukup banyak false positive — perlu ditindaklanjuti pemeriksaan klinis lanjutan, bukan digunakan sebagai diagnosis mandiri.
- Recall (0.55) belum mencapai target awal (≥60%), meski F1 dan ROC-AUC sudah melebihi target.
- Fitur `bmi` dihapus karena banyak missing value, padahal BMI adalah indikator kesehatan yang relevan secara klinis — ini trade-off antara kebersihan data dan kelengkapan informasi.
- Model dan artefak hasil training (weights/pickle) belum disertakan di repo — hanya notebook yang menjalankan proses training.

## Future Improvements
- Melakukan imputasi yang lebih baik pada fitur `bmi` (misalnya median imputation per grup usia) alih-alih menghapusnya seluruhnya.
- Mengeksplorasi kombinasi SMOTE + cost-sensitive learning untuk memperbaiki precision tanpa mengorbankan recall.
- Menyimpan model terlatih (`.pkl`) agar dapat digunakan langsung tanpa training ulang.
- Membangun antarmuka sederhana (misalnya Streamlit) untuk mendemonstrasikan model sebagai alat skrining interaktif.

## Author
**Septi Isdayanna**
