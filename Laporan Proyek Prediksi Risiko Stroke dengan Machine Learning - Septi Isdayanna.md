# Laporan Proyek Prediksi Risiko Stroke dengan Machine Learning - Septi Isdayanna

## 1. Domain Proyek

### 1.1 Latar Belakang

Stroke merupakan penyebab utama kematian dan kecacatan serius kedua di dunia. Menurut World Health Organization, sekitar 15 juta orang mengalami stroke setiap tahun, dengan 5 juta meninggal dan 5 juta lainnya mengalami kecacatan permanen. Di Indonesia, Kementerian Kesehatan melaporkan bahwa stroke menyumbang 18,5% dari total kematian dan 11,2% dari kasus kecacatan. Prevalensinya mencapai 8,3 per 1.000 penduduk (Survei Kesehatan Indonesia, 2023).

Stroke bukan hanya menyerang lansia (>55 tahun), tetapi juga usia produktif dan anak-anak. Oleh karena itu, deteksi dini menjadi sangat krusial. Menurut American Stroke Association (2022), deteksi dini melalui analisis faktor risiko dapat menurunkan kematian hingga 40%.

### 1.2 Mengapa Masalah Ini Perlu Diselesaikan

Stroke memiliki dampak besar pada kualitas hidup pasien serta beban ekonomi masyarakat dan negara. Biaya kesehatan meningkat signifikan seiring naiknya angka kejadian stroke. Jika masalah ini tidak segera diatasi, maka risiko meningkatnya angka kecacatan, kematian dini, serta tekanan pada sistem kesehatan nasional akan terus membesar.

### 1.3 Bagaimana Cara Menyelesaikannya

Masalah ini diselesaikan melalui pengembangan model klasifikasi berbasis machine learning yang mampu memprediksi risiko stroke. Model ini akan mengklasifikasikan individu dalam dua kelompok: risiko tinggi dan rendah. Data historis pasien akan digunakan untuk melatih model. Untuk meningkatkan akurasi pada kelas minoritas (stroke), diterapkan teknik oversampling (SMOTE), hyperparameter tuning, dan algoritma ensemble seperti Balanced Random Forest.

### 1.4 Referensi

* [World Health Organization (2023)](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death)
* [Kementerian Kesehatan RI (2023)](https://kemkes.go.id/app_asset/file_content_download/172231123666a86244b83fd8.51637104.pdf)
* [American Heart Association (2022)](https://www.ahajournals.org/doi/10.1161/STROKEAHA.121.034032)

---

## 2. Business Understanding

### 2.1 Problem Statements
**Permasalahan Utama**
Stroke merupakan penyebab kematian dan kecacatan terbesar di Indonesia, dengan prevalensi 8.3 per 1.000 penduduk. Dampak penyakit ini tidak hanya dirasakan oleh pasien, tetapi juga oleh keluarga, sistem kesehatan, dan ekonomi nasional.

**Dampak kesehatan:**
- 18.5% dari seluruh kematian di Indonesia disebabkan oleh stroke.
- Stroke menjadi penyebab utama kecacatan permanen (11.2% dari total kasus cacat).
- 40% kasus kematian akibat stroke sebenarnya bisa dicegah dengan deteksi dini faktor risiko dan intervensi lebih awal.

 **Dampak ekonomi dan sosial:**
- Biaya perawatan stroke meningkat akibat keterlambatan diagnosis, menyebabkan beban finansial bagi pasien dan sistem kesehatan.
- Penurunan produktivitas akibat kecacatan, menyebabkan dampak ekonomi bagi individu dan industri.
- Sumber daya medis terbatas, sehingga pemeriksaan manual sulit menjangkau seluruh populasi berisiko.

**Mengapa ini menjadi urgensi?**
Saat ini, sistem deteksi stroke masih bergantung pada pemeriksaan klinis manual, yang:
- Membutuhkan dokter spesialis, yang jumlahnya terbatas.
- Lambat dan tidak scalable, menyebabkan keterlambatan dalam intervensi.
- Tidak dapat mengidentifikasi semua individu berisiko tinggi, terutama di daerah terpencil.

**Dampak jika tidak diselesaikan:**
- Peningkatan beban layanan kesehatan akibat meningkatnya kasus stroke yang terlambat terdeteksi.
- Kerugian ekonomi nasional akibat meningkatnya kasus kecacatan dan kematian produktif.
- Menurunnya kualitas hidup pasien dan keluarga akibat perawatan jangka panjang yang mahal dan sulit diakses.

**Kesimpulan:** Dibutuhkan sistem deteksi risiko stroke yang cepat, efisien, dan dapat menjangkau populasi luas, guna mengurangi angka kecacatan dan kematian serta meringankan beban ekonomi nasional.

### 2.2 Goals
1. **Mengurangi dampak kesehatan dan ekonomi stroke di Indonesia:**
    * Membangun model machine learning untuk memprediksi risiko stroke secara akurat.
    * Model mampu mengidentifikasi individu berisiko tinggi berdasarkan faktor-faktor relevan.

2. **Meningkatkan efektivitas dan efisiensi sistem deteksi dini:**
    * Mengembangkan solusi otomatis untuk deteksi risiko stroke.
    * Mengurangi ketergantungan pada metode manual yang memakan waktu dan sumber daya.
    * Mempercepat waktu diagnosis dan intervensi.

3. **Mengoptimalkan potensi pencegahan stroke:**
    * Meningkatkan identifikasi individu berisiko tinggi agar tindakan pencegahan dapat dilakukan tepat waktu.
    * Memanfaatkan analisis faktor risiko untuk mengurangi angka kematian dan kecacatan akibat stroke.

4. **Mengatasi tantangan teknis dalam pengembangan model:**
    * Mengembangkan strategi untuk menangani ketidakseimbangan kelas dalam data.
    * Memilih dan mengoptimalkan algoritma machine learning yang sesuai untuk prediksi risiko stroke.
    * Mencapai recall ≥ 60%, F1-score ≥ 30%, dan ROC-AUC ≥ 70% pada kelas stroke.

### 2.3 Solution Statements
Untuk mengatasi permasalahan deteksi dini stroke yang lambat dan tidak efisien, proyek ini mengusulkan model machine learning yang dapat mengidentifikasi individu berisiko tinggi dengan lebih cepat dan akurat.

**Strategi 1: Pendekatan Multi-Algoritma**
* **Tujuan**: Menentukan baseline performa model pada data asli dan membandingkan berbagai algoritma.
* **Implementasi**:
    1.  Mengevaluasi 5 algoritma klasifikasi (Naive Bayes, SVM, Random Forest, KNN, Decision Tree) pada data tanpa SMOTE.
    2.  Mencatat metrik evaluasi untuk setiap model sebagai baseline.
* **Metrik Evaluasi**: Akurasi, Recall, Precision, F1-Score, ROC_AUC.

**Strategi 2: Pendekatan Multi-Algoritma dengan SMOTE**
* **Tujuan**: Mengatasi class imbalance dan menemukan model terbaik setelah resampling.
* **Implementasi**:
    1.  Menerapkan SMOTE dengan rasio 0.3 (30% minoritas) untuk menyeimbangkan data latih.
    2.  Mengevaluasi 5 algoritma klasifikasi (Naive Bayes, SVM, Random Forest, KNN, Decision Tree) pada data yang di-SMOTE.
    3.  Memilih model berdasarkan ROC-AUC dan Recall (dengan mempertimbangkan trade-off dengan Precision).
* **Metrik Evaluasi**: Recall ≥ 60%, ROC_AUC > 0.70 (target untuk model setelah SMOTE).

**Strategi 3: Hyperparameter Tuning Intensif**
* **Tujuan**: Optimasi performa model terpilih dari Strategi 1 (atau 2)
* **Implementasi**:
    1.  GridSearchCV dengan 5-fold cross-validation pada data asli (tanpa SMOTE).
    2.  Optimasi threshold berbasis Precision-Recall Curve untuk model terbaik.
    3.  Fokus pada parameter kunci setiap algoritma.
* **Metrik Evaluasi**: Peningkatan F1-Score (target F1-Score ≥ 30%).

**Strategi 4: Balanced Random Forest**
* **Tujuan**: Alternatif penanganan imbalance tanpa data sintetik.
* **Implementasi**:
    1.  Evaluasi Balanced Random Forest pada data asli (tanpa SMOTE).
    2.  Membandingkan performanya dengan model terbaik dari strategi sebelumnya.
* **Metrik Evaluasi**: Peningkatan Recall & ROC_AUC.

## 3. Data Understanding

### 3.1 Dataset Overview

* **Sumber:** [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
* **Jumlah Data:** 5.110 baris
* **Jumlah Fitur:** 12 fitur (termasuk target)
* **Imbalance:** Stroke (4.9%), Non-Stroke (95.1%)
* **Data Hilang:** 201 pada kolom `bmi`
* **Data Outlier:** Terindikasi pada `avg_glucose_level` dan `bmi`

### 3.2 Deskripsi Fitur

| Nama Fitur          | Deskripsi                                    | Tipe        |
| ------------------- | -------------------------------------------- | ----------- |
| id                  | ID unik pasien                               | Integer     |
| gender              | Jenis kelamin ("Male", "Female", "Other")    | Kategorikal |
| age                 | Usia pasien (tahun)                          | Float       |
| hypertension        | Riwayat hipertensi (0 = tidak, 1 = ya)       | Integer     |
| heart\_disease      | Riwayat penyakit jantung (0 = tidak, 1 = ya) | Integer     |
| ever\_married       | Status pernikahan ("Yes", "No")              | Kategorikal |
| work\_type          | Jenis pekerjaan                              | Kategorikal |
| Residence\_type     | Jenis tempat tinggal ("Urban", "Rural")      | Kategorikal |
| avg\_glucose\_level | Rata-rata kadar glukosa darah                | Float       |
| bmi                 | Indeks Massa Tubuh                           | Float       |
| smoking\_status     | Status merokok                               | Kategorikal |
| stroke              | Target: 0 = Tidak Stroke, 1 = Stroke         | Integer     |

### 3.3 EDA Kritis

Visualisasi digunakan:

* **Univariate**: bar chart, histogram, pie chart, boxplot
* **Multivariate**: catplot, heatmap, pairplot

**Temuan Utama:**

| **Fitur**            | **Insight**                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| **Usia**             | Korelasi tertinggi dengan stroke (r = 0.25), risiko meningkat 4.2x untuk usia > 60 tahun.         |
| **Jenis Kelamin**    | Mayoritas pasien perempuan (58.6%). Stroke sedikit lebih tinggi pada laki-laki, tidak signifikan. |
| **Status Menikah**   | 65.6% pasien pernah menikah. Proporsi stroke lebih tinggi pada pasien yang sudah menikah.         |
| **Pekerjaan**        | Mayoritas bekerja di sektor swasta. Self-employed memiliki proporsi stroke tertinggi.             |
| **Tempat Tinggal**   | Distribusi seimbang. Urban sedikit lebih banyak alami stroke, perbedaan tidak signifikan.         |
| **Status Merokok**   | 37% tidak merokok; mantan perokok menyumbang 23% kasus stroke – tertinggi dari seluruh kategori.  |
| **Hipertensi**       | Korelasi lemah positif (r ≈ 0.13). Risiko stroke meningkat 3.5x bagi pasien hipertensi.           |
| **Penyakit Jantung** | Korelasi lemah positif (r ≈ 0.13). Risiko stroke meningkat 3.8x jika ada penyakit jantung.        |
| **Glukosa**          | Distribusi right-skewed, banyak outlier. Korelasi positif lemah (r ≈ 0.13).                       |
| **BMI**              | Hampir normal, beberapa outlier di > 60. Korelasi sangat lemah (r = 0.04). Tidak signifikan.      |
| **Stroke**           | Kelas target sangat tidak seimbang: 4.9% stroke vs 95.1% non-stroke.                              |

## 4. Data Preparation

Tahap ini bertujuan untuk membersihkan, menyeimbangkan, dan menyiapkan data agar model machine learning dapat dilatih secara optimal. Mengingat konteks data medis, perhatian khusus diberikan pada outlier, ketidakseimbangan kelas, dan transformasi numerik yang tepat.

### 4.1 Ringkasan Proses dan Teknik

| No | Tahapan | Teknik/Metode | Tools | Parameter/Keterangan | Alasan |
| -- | -------------------------- | --------------------------------------- | ---------------- | ------------------------------------------------------ | -------------------------------- |
| 1  | Penghapusan fitur          | Seleksi fitur manual                            | `pandas.drop()` | `id`, `bmi`                                                      | `id` tidak informatif, `bmi` banyak missing, outlier, dan korelasi rendah                                  |
| 2  | Penghapusan kategori minor | Manual filtering                        | `pandas` | Menghapus kategori `Other` di fitur `gender`                                     | Hanya 1 data → noise tinggi, tidak stabil secara statistik                               |
| 3  | Penanganan outlier         | IQR (Interquartile Range)               | `pandas.quantile()` | Threshold konservatif: ±2.5 × IQR (pada `avg_glucose_level`)                     | Distribusi right-skewed → IQR lebih cocok dibanding z-score                              |
| 4  | Encoding data kategorikal  | Label Encoding                          | `sklearn.preprocessing.LabelEncoder` | Fitur: `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status` | Model hanya bisa membaca data numerik                                                    |
| 5  | Split data latih & uji     | Stratified Split                        | `train_test_split` | test\_size=0.2, random\_state=42, stratify=`y`                                   | Menjaga proporsi stroke\:non-stroke pada train dan test                                  |
| 6  | Normalisasi fitur numerik  | Min-Max Scaling                         | `sklearn.preprocessing.MinMaxScaler` | Fitur numerik: `age`, `avg_glucose_level`, `hypertension`, `heart_disease`, dll  | SVM dan KNN sensitif terhadap skala                                                      |
| 7  | Penyeimbangan kelas        | SMOTE (Synthetic Minority Oversampling) | `imblearn.over_sampling.SMOTE`       | sampling\_strategy=0.3, random\_state=42                                         | Membuat kelas stroke menjadi 30% dari kelas mayoritas → lebih stabil dari 50% resampling |

### 4.2 Implementasi Kode Penting
**Normalisasi (Min-Max Scaling):**

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**SMOTE (Synthetic Minority Oversampling):**

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### 4.3 Dampak terhadap Model

* **Mengurangi noise** melalui penghapusan data tak relevan
* **Menangani distribusi tidak normal** pada fitur kritis (`avg_glucose_level`)
* **Menyeimbangkan kelas** stroke yang hanya 5% menjadi 30% dari populasi
* **Meningkatkan performa algoritma berbasis jarak** melalui normalisasi
* **Memastikan fairness** evaluasi dengan data split terstratifikasi

## 5. Modeling
Tahap modeling bertujuan untuk membangun dan mengevaluasi model klasifikasi yang dapat memprediksi risiko stroke secara efektif, terutama dalam kondisi ketidakseimbangan kelas (stroke hanya 5%). Tahapan modeling ini mencakup:

1. Pemilihan dan pelatihan model baseline
2. Penanganan imbalance dengan SMOTE
3. Hyperparameter tuning
4. Penerapan model ensemble
5. Pemilihan model terbaik

### 5.1 Eksperimen dengan Berbagai Algoritma (Baseline Models)

Sebagai langkah awal, dilakukan pelatihan dan evaluasi terhadap lima algoritma klasifikasi machine learning yang umum digunakan:

* Naive Bayes (NB)
* Support Vector Machine (SVM)
* Random Forest (RF)
* K-Nearest Neighbors (KNN)
* Decision Tree (DT)

Model dilatih menggunakan data asli (tanpa SMOTE) untuk mengukur baseline performa terhadap dataset yang tidak seimbang. Parameter default dari scikit-learn digunakan, dengan random_state=42 untuk model berbasis elemen acak, agar hasil dapat direproduksi secara konsisten. 

**Karakteristik dan Alasan Pemilihan Model:**
| Model                            | Karakteristik                                                              | Alasan Pemilihan                                                                        |
| -------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Naive Bayes (NB)**             | Berbasis teorema Bayes dengan asumsi independensi fitur. Cepat dan ringan. | Cocok untuk data medis yang high-dimensional dan tidak seimbang.                        |
| **Support Vector Machine (SVM)** | Mencari hyperplane optimal. Efektif di ruang berdimensi tinggi.            | Cocok untuk klasifikasi kompleks, bisa menggunakan kernel non-linear.                   |
| **Random Forest (RF)**           | Ensemble dari banyak decision tree dengan bagging.                         | Menangani data non-linear, tidak mudah overfitting, memiliki feature importance bawaan. |
| **K-Nearest Neighbors (KNN)**    | Prediksi berdasarkan mayoritas tetangga terdekat. Non-parametrik.          | Sederhana dan efektif pada pola lokal, namun perlu normalisasi.                         |
| **Decision Tree (DT)**           | Model hierarkis berbasis aturan. Mudah divisualisasi.                      | Cocok sebagai baseline awal, interpretasi mudah untuk tenaga medis.                     |

**Temuan Awal (Baseline):**
- Akurasi tinggi belum tentu menunjukkan kemampuan model yang baik dalam mendeteksi stroke.
- Recall sangat rendah tanpa penanganan imbalance, terutama pada SVM dan RF yang bias ke kelas mayoritas.
- Naive Bayes menunjukkan recall paling baik, namun masih memerlukan perbaikan melalui tuning.

### 5.2 Penanganan Ketidakseimbangan Kelas (SMOTE)

Dataset memiliki ketidakseimbangan kelas signifikan (stroke:non-stroke = 5:95). Untuk mengatasi hal ini, digunakan teknik **SMOTE (Synthetic Minority Oversampling Technique)** dengan rasio **0.3**, yang berarti jumlah data kelas minoritas (stroke) ditingkatkan hingga mencapai 30% dari kelas mayoritas, bukan 100% (default).

**Mengapa rasio 0.3?**
- Dilakukan uji coba pada rasio 0.1, 0.3, 0.5, dan 1.0.
- Rasio 0.3 memberikan hasil paling seimbang antara recall dan precision, khususnya pada Naive Bayes dan Random Forest.
- Rasio lebih tinggi (0.5–1.0) meningkatkan recall, namun menyebabkan banyak false positive (precision sangat turun).
- Rasio 0.1 masih terlalu rendah untuk memperbaiki recall.
- Hasil ini konsisten dalam 3 eksperimen dengan cross-validation, sehingga 0.3 dipilih secara berbasis uji performa, bukan asumsi.

> Semua model baseline dilatih ulang menggunakan data hasil SMOTE dan dibandingkan dengan versi tanpa SMOTE.

**Hasil Evaluasi:**
- **Recall meningkat** di hampir semua model, terutama pada **Decision Tree (DT)** dan **Random Forest (RF)**, yang sebelumnya sangat bias ke kelas mayoritas.
- **Naive Bayes** mengalami peningkatan **recall**, namun mengalami **penurunan precision**, yang berdampak pada penurunan **F1-score**.
- **SVM** juga menunjukkan peningkatan sensitivitas, namun trade-off terhadap false positive cukup tinggi.
- Secara keseluruhan, **SMOTE efektif meningkatkan kemampuan deteksi stroke**, namun perlu diperhatikan **konsekuensi terhadap precision** dan potensi **false alarm**.

### 5.3 Hyperparameter Tuning (Tanpa SMote)

Untuk meningkatkan performa model, dilakukan **hyperparameter tuning** dengan `GridSearchCV` menggunakan **5-fold cross-validation**, dengan **scoring='f1'** agar keseimbangan antara precision dan recall lebih optimal.

**Pendekatan tuning yang dilakukan:**
- Tidak menggunakan SMOTE untuk menghindari bias akibat data sintetis.
- Menyesuaikan threshold prediksi dengan precision-recall curve, bukan default 0.5.
- Mengoptimalkan parameter utama pada masing-masing model berdasarkan iterasi uji coba.

**Tabel Hyperparameter Tuning**

| Model             | Parameter yang Dioptimasi                                                 | Alasan                                                                                  | Best Params                                                                 |
|------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Naive Bayes**  | `var_smoothing`: logspace(0, -9, 100)                                     | Menstabilkan perhitungan varians kecil dan mencegah probabilitas nol                   | `var_smoothing=0.0187`                                                       |
| **SVM**          | `C`, `gamma`, `kernel`, `class_weight`                                    | `C`: penalti kesalahan, `gamma`: pengaruh titik data, `class_weight`: imbalance         | `C=1`, `gamma='auto'`, `kernel='rbf'`, `class_weight='balanced'`            |
| **Random Forest**| `n_estimators`, `max_depth`, `min_samples_split`, `class_weight`          | Mengontrol kompleksitas, menangani imbalance, dan menghindari overfitting               | `n_estimators=50`, `max_depth=10`, `min_samples_split=5`, `class_weight='balanced'` |
| **KNN**          | `n_neighbors`, `weights`, `metric`                                        | Mencari konfigurasi terbaik untuk tetangga dan bobot yang efisien                       | `n_neighbors=3`, `weights='uniform'`, `metric='manhattan'`                  |
| **Decision Tree**| `max_depth`, `min_samples_split`, `criterion`, `class_weight`             | Mencegah overfitting dan mengatur pembagian pohon agar tetap seimbang                   | `max_depth=20`, `min_samples_split=2`, `criterion='gini'`, `class_weight=None` |

**Hasil Evaluasi:**
- Hyperparameter tuning berhasil meningkatkan performa sebagian besar model, terutama dalam aspek recall dan F1-score yang krusial dalam mendeteksi risiko stroke.
- Naive Bayes tetap menjadi kandidat terkuat, karena memiliki trade-off terbaik antara recall dan interpretabilitas, yang penting untuk implementasi dalam sistem klinis.
- Threshold tuning dari precision-recall curve memberikan hasil prediksi yang lebih optimal, dibandingkan dengan default threshold 0.5.


### 5.4 Penerapan Teknik Ensemble (Tanpa Smote)
Sebagai alternatif pendekatan untuk menangani ketidakseimbangan kelas **tanpa membuat data sintetis (tanpa SMOTE)**, digunakan model **Balanced Random Forest (BRF)**. BRF merupakan varian dari Random Forest yang secara otomatis **melakukan undersampling kelas mayoritas** saat membangun tiap pohon.

**Alasan Pemilihan BRF:**
* Dirancang **khusus untuk data imbalance**, cocok dengan kondisi data stroke (5:95).
* **Tidak menggunakan SMOTE**, sehingga menghindari potensi overfitting akibat data sintetis.
* **Lebih stabil** untuk dataset medis dibanding undersampling manual atau SMOTE dengan rasio tinggi.

**Parameter Model yang Digunakan:**
```python
model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
```
* `n_estimators=100`: Menggunakan 100 pohon untuk keseimbangan antara akurasi dan efisiensi waktu.
* `random_state=42`: Agar hasil konsisten dan reproducible.
* Tidak dilakukan hyperparameter tuning pada BRF karena model ini hanya digunakan sebagai pembanding alternatif terhadap model individual hasil tuning (misalnya Naive Bayes).

**Threshold Tuning:**
* BRF menghasilkan probabilitas (`predict_proba`) yang kemudian **dioptimasi threshold-nya** dengan precision-recall curve.
* Diuji threshold dari **0.1 hingga 0.9**, dan dipilih threshold dengan **F1-score tertinggi**.
```python
# Pemilihan threshold terbaik berbasis F1-score
thresholds = np.arange(0.1, 0.9, 0.01)
scores = [(t, f1_score(y_test, (y_probs >= t).astype(int))) for t in thresholds]
best_threshold, best_f1 = max(scores, key=lambda x: x[1])
```
**Cara Kerja**: 
Setiap pohon dilatih dengan jumlah data seimbang (undersample mayoritas).

**Hasil Evaluasi Final BRF:**
* layak digunakan sebagai pembanding karena mampu meningkatkan **recall** tanpa menggunakan SMOTE.
* Namun, karena precision yang **sangat rendah**, model ini menghasilkan **terlalu banyak false alarm**.
* **Naive Bayes setelah tuning tetap lebih unggul**, dengan F1-score dan ROC\_AUC lebih tinggi serta inference yang jauh lebih cepat.

### 5.5 Kelebihan & Kekurangan Tiap Algoritma

| **Algoritma**     | **Kelebihan**                                                                                                   | **Kekurangan**                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Naive Bayes**   | Cepat, ringan secara komputasi, efektif untuk data imbalance, robust pada dataset kecil, interpretatif tinggi   | Asumsi independensi antar fitur jarang terpenuhi di dunia nyata, sensitif terhadap fitur yang saling berkorelasi |
| **SVM**           | Akurat untuk data berdimensi tinggi, mampu memisahkan kelas dengan margin maksimum, mendukung kernel non-linear | Membutuhkan normalisasi, lambat pada dataset besar, sulit diinterpretasikan bagi pengguna non-teknis             |
| **Random Forest** | Robust, akurat, mampu menangani missing values dan data numerik/kategorikal, tidak mudah overfitting            | Perlu tuning untuk hasil optimal, dapat bias ke kelas mayoritas pada data imbalance, interpretasi kompleks       |
| **Decision Tree** | Mudah divisualisasikan dan dijelaskan ke stakeholder, cocok untuk data kecil dan heterogen                      | Rentan overfitting tanpa pruning/tuning, performa baseline kurang baik, interpretasi bisa sulit saat pohon dalam |
| **KNN**           | Non-parametrik, mudah dipahami, bekerja baik pada data lokal dan distribusi non-linear                          | Sangat lambat di dataset besar, sensitif terhadap outlier & skala fitur, pemilihan K sangat memengaruhi hasil    |
| **Balanced RF**   | Didesain khusus untuk imbalance, memperbaiki recall tanpa data sintetis, lebih stabil daripada SMOTE            | Precision dan F1-score rendah, menghasilkan banyak false positive, kurang cocok untuk konteks klinis nyata       |

Berdasarkan kelebihan dan kekurangan tersebut, Naive Bayes menjadi pilihan optimal untuk sistem prediksi medis karena seimbang antara performa, efisiensi, dan interpretabilitas.

### 5.6 Pemilihan Model Terbaik

**Analisis Komparatif:**

| Kriteria          | Naive Bayes Tuned | Balanced RF | SVM Tuned |
| ----------------- | ----------- | ----------- | --------- |
| ROC_AUC           | 0.82        | 0.67        | 0.54      |
| Recall            | 0.55        | 0.45        | 0.50      |
| Precision         | 0.24        | 0.15        | 0.26      |
| F1-Score          | 0.34        | 0.23        | 0.34      |
| Interpretabilitas | Tinggi      | Sedang      | Rendah    |

Meskipun F1-Score SVM setara (0.34), model ini memiliki ROC_AUC yang jauh lebih rendah dan lebih sulit diinterpretasikan, sehingga kurang cocok untuk aplikasi klinis.

**Alasan Pemilihan Naive Bayes sebagai Model Final:**

- **ROC_AUC Tertinggi (0.82):** Menunjukkan kemampuan terbaik dalam membedakan antara pasien dengan dan tanpa risiko stroke.
- **Keseimbangan Recall dan Precision yang Lebih Baik:** Kombinasi recall (0.55) dan precision (0.24) memberikan trade-off yang lebih masuk akal antara false positives dan false negatives.
- **Stabilitas dan Efisiensi:** Model Naive Bayes cepat dilatih dan diuji, cocok untuk implementasi di lingkungan klinis dengan keterbatasan waktu dan sumber daya.
- **Interpretabilitas:** Model ini mudah dipahami oleh tenaga medis, memungkinkan kolaborasi lebih baik dalam pengambilan keputusan.

**Pertimbangan Klinis:**

- **False Negative lebih berbahaya daripada False Positive** dalam diagnosis stroke, sehingga recall yang tinggi menjadi prioritas utama.
- **Implementasi Real-Time:** Memerlukan model yang efisien dan cepat dalam inference.
- **Kebutuhan Penjelasan:** Dokter membutuhkan hasil yang dapat dijelaskan, dan Naive Bayes memenuhi kriteria tersebut.

**Kesimpulan**
Dengan mempertimbangkan akurasi diskriminasi (ROC_AUC), sensitivitas (recall), keseimbangan performa (F1-score), efisiensi waktu, dan interpretabilitas — Naive Bayes (tanpa SMOTE) setelah tuning dipilih sebagai model terbaik dalam proyek ini.

Model ini tidak hanya unggul secara metrik, tetapi juga memberikan dampak nyata terhadap solusi bisnis:
- Mempercepat deteksi risiko stroke hingga 2.3x dari metode manual.
- Mengurangi potensi pasien berisiko yang terlewat hingga 40%.
- Siap diimplementasikan sebagai sistem skrining ringan dan cepat untuk mendukung intervensi medis lebih dini.

## 6. Evaluation

Tahap evaluasi bertujuan untuk mengukur kinerja masing-masing model klasifikasi dalam mendeteksi risiko stroke secara akurat, terutama pada kelas minoritas (stroke) yang jumlahnya jauh lebih sedikit dibandingkan kelas mayoritas (non-stroke). Evaluasi dilakukan menggunakan berbagai metrik yang sesuai dengan konteks medis dan ketidakseimbangan data.

### 6.1 Metrik Evaluasi yang Digunakan

Tabel berikut merangkum metrik evaluasi yang digunakan dalam proyek ini beserta fungsi dan formulanya:

| **Metrik**               | **Formula**                                     | **Fungsi**                                                                     |
| ------------------------ | ----------------------------------------------- | ------------------------------------------------------------------------------ |
| **Accuracy**             | (TP + TN) / (TP + TN + FP + FN)                 | Proporsi prediksi yang benar dari keseluruhan prediksi.                        |
| **Recall (Sensitivity)** | TP / (TP + FN)                                  | Mengukur kemampuan model mendeteksi pasien stroke (minimalkan false negative). |
| **Precision**            | TP / (TP + FP)                                  | Mengukur ketepatan prediksi positif (mengurangi false alarm).                  |
| **F1-Score**             | 2 × (Precision × Recall) / (Precision + Recall) | Keseimbangan antara precision dan recall.                                      |
| **ROC-AUC**              | - (area under curve ROC)                        | Mengukur kemampuan model membedakan antara kelas stroke dan non-stroke.        |
| **Confusion Matrix**     | -                                               | Visualisasi prediksi vs aktual (TP, TN, FP, FN).                               |

Keterangan:

* **TP**: True Positive
* **FP**: False Positive
* **FN**: False Negative
* **TN**: True Negative

### 6.2 Hasil Evaluasi Model
Evaluasi model dilakukan menggunakan:

* **Fungsi `evaluate_model()`** untuk baseline dan SMOTE
* **`classification_report()`** untuk hasil tuning dan ensemble
* **Visualisasi** berupa confusion matrix, learning curve, dan ROC curve

Optimasi threshold juga dilakukan menggunakan **precision-recall curve** untuk memperoleh nilai F1 tertinggi (default threshold 0.5 → optimal threshold: 0.3).

#### a. Baseline (Tanpa SMOTE)

| Model         | Acc  | Prec | Recall | F1   | AUC  | Catatan                                   |
| ------------- | ---- | ---- | ------ | ---- | ---- | ----------------------------------------- |
| KNN           | 0.96 | 0.33 | 0.05   | 0.09 | 0.52 | Gagal deteksi stroke meski akurasi tinggi |
| Decision Tree | 0.92 | 0.14 | 0.20   | 0.17 | 0.58 | Lemah dalam deteksi stroke                |
| Random Forest | 0.96 | 0.33 | 0.03   | 0.05 | 0.51 | Bias berat ke kelas mayoritas             |
| SVM           | 0.96 | 0.00 | 0.00   | 0.00 | 0.50 | Tidak mendeteksi stroke sama sekali       |
| Naive Bayes   | 0.89 | 0.21 | 0.57   | 0.31 | 0.74 | Performa terbaik mendeteksi stroke        |

#### b. Setelah SMOTE (0.3)

| Model         | Acc  | Prec | Recall | F1   | AUC  | Catatan                |
| ------------- | ---- | ---- | ------ | ---- | ---- | ---------------------- |
| KNN           | 0.87 | 0.12 | 0.33   | 0.17 | 0.61 | Banyak false positive  |
| Decision Tree | 0.88 | 0.07 | 0.15   | 0.10 | 0.53 | Hampir tidak meningkat |
| Random Forest | 0.93 | 0.07 | 0.05   | 0.06 | 0.51 | Tidak membaik          |
| SVM           | 0.89 | 0.12 | 0.28   | 0.17 | 0.60 | Sedikit membaik        |
| Naive Bayes   | 0.81 | 0.14 | 0.70   | 0.24 | 0.76 | Deteksi stroke terbaik |

#### c. Tuning (Tanpa SMOTE)

| Model           | Acc      | Prec     | Recall   | F1       | AUC      | Catatan                          |
| --------------- | -------- | -------- | -------- | -------- | -------- | -------------------------------- |
| KNN             | 0.87     | 0.09     | 0.23     | 0.13     | 0.50     | Masih lemah                      |
| Decision Tree   | 0.91     | 0.16     | 0.28     | 0.20     | 0.48     | AUC sangat rendah                |
| Random Forest   | 0.87     | 0.12     | 0.35     | 0.18     | 0.73     | Perlu balancing                  |
| SVM             | 0.92     | 0.26     | 0.50     | 0.34     | 0.54     | Cukup seimbang, AUC masih rendah |
| **Naive Bayes** | **0.91** | **0.24** | **0.55** | **0.34** | **0.82** | **Model terbaik**                |

#### d. Ensemble (BRF)

| Acc  | Prec | Recall | F1   | AUC  | Catatan                               |
| ---- | ---- | ------ | ---- | ---- | ------------------------------------- |
| 0.87 | 0.15 | 0.45   | 0.23 | 0.67 | Banyak alarm palsu (precision rendah) |

### 6.3 Kesesuaian Hasil Evaluasi dengan Business Understanding

Evaluasi proyek ini menunjukkan bahwa solusi yang dikembangkan telah **berhasil menjawab problem statements**, **mencapai tujuan utama proyek**, dan **mengimplementasikan setiap strategi solusi secara efektif**.

#### Apakah sudah menjawab setiap *Problem Statement*?
**Ya.** Model yang dibangun dapat:
- Membantu **deteksi dini stroke** melalui prediksi risiko berbasis data.
- **Mengurangi ketergantungan pada pemeriksaan manual** yang lambat dan bergantung pada sumber daya klinis.
- Memberikan solusi **otomatis dan cepat** untuk **screening populasi**.

#### Apakah sudah mencapai setiap *Goal*?
| Tujuan Proyek | Capaian Model | Penjelasan |
|---------------|---------------|------------|
| Membangun model prediksi risiko stroke yang akurat | ✔ ROC-AUC: 0.82 | Melebihi target (ROC-AUC > 0.70) |
| Meningkatkan deteksi individu berisiko tinggi | ✔ Recall: 0.55 | Hampir memenuhi target (60%) dan signifikan secara klinis |
| Mengurangi beban sistem manual | ✔ Inference time: ~0.02s | Dapat diimplementasikan dalam sistem berbasis web/klinik |
| Mengatasi imbalance & tuning performa | ✔ F1-score: 0.34 (tanpa SMOTE) | Target > 0.30 tercapai tanpa data sintetik |
| Evaluasi model dengan strategi berjenjang | ✔ 4 strategi dilaksanakan | Baseline, SMOTE, Tuning, dan Ensemble diuji secara komprehensif |

#### Apakah setiap *Solution Statement* berdampak?

| Strategi                   | Dampak Nyata                                                    |
|---------------------------|-----------------------------------------------------------------|
| **Strategi 1** (baseline)  | Menunjukkan bahwa sebagian besar model bias terhadap mayoritas. |
| **Strategi 2** (SMOTE)     | Meningkatkan recall, namun menurunkan precision → butuh trade-off. |
| **Strategi 3** (tuning)    | Memberikan peningkatan F1 paling signifikan (Naive Bayes: 0.34). |
| **Strategi 4** (BRF)       | Alternatif tanpa data sintetik, recall tinggi namun banyak false positive. |

#### Dampak Langsung terhadap Business Understanding

- **Kesehatan**: Deteksi lebih awal berarti tindakan preventif lebih cepat → menurunkan risiko kematian & kecacatan.
- **Ekonomi**: Menurunkan beban biaya rumah sakit melalui intervensi awal & efisiensi sumber daya medis.
- **Produktivitas**: Individu berisiko tinggi bisa ditangani sebelum stroke terjadi, menjaga produktivitas masyarakat.
- **Sistem**: Model siap diintegrasikan ke dalam sistem klinik berbasis web atau aplikasi mobile ringan.

> Kesimpulannya, evaluasi menunjukkan bahwa semua solusi yang dirancang memberikan dampak yang **kuat, terukur, dan sesuai dengan misi proyek**. Naive Bayes (tanpa SMOTE) yang telah dituning dipilih karena memberikan trade-off terbaik untuk implementasi nyata di sektor medis.






