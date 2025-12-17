# 📘 Judul Proyek
Klasifikasi Inflamasi Kandung Kemih Berdasarkan Gejala Klinis Menggunakan Machine Learning dan Deep Learning

## 👤 Informasi
- **Nama:** Rian Cahyo Anggoro
- **NIM:** 234311052
- **Repo:** https://github.com/Riancahyo/DataScience-Acude-Inflammations.git  
- **Video:** 

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan deteksi dini risiko Inflamasi Kandung Kemih berdasarkan data gejala klinis sederhana.
- Melakukan Data Preparation meliputi cleaning, encoding data kategorikal, dan scaling menggunakan **StandardScaler**.
- Membangun 3 model: **Baseline (Logistic Regression)**, **Advanced (Random Forest)**, **Deep Learning (MLP)**.
- Melakukan evaluasi menggunakan metrik **Accuracy, F1-Score, dan Recall** untuk menangani dataset yang tidak seimbang (*imbalanced*).

---

# 2. 📄 Problem & Goals
**Problem Statements:**
- Diagnosis Inflamasi Kandung Kemih (Bladder Inflammation) memerlukan tes laboratorium yang membutuhkan waktu, padahal keputusan pengobatan harus cepat.
- Diperlukan metode cepat dan non-invasif untuk memprediksi risiko hanya berdasarkan gejala klinis dasar.
- Data yang digunakan relatif kecil, sehingga diperlukan model yang efisien dan stabil.

**Goals:**
- Membangun model klasifikasi biner dengan target akurasi > 80%.
- Menganalisis dan membandingkan performa model Linear, Ensemble, dan Neural Network.
- Menentukan model terbaik yang optimal dari segi performa dan efisiensi.

---
## 📁 Struktur Folder
```
project/
│
├── data/
│   └── diagnosis.data
│   └── diagnosis.names
|
├── images/
│   └── Cek Noise Outlier.png
│   └── Confusion Matrix Logistic Regression.png
│   └── Confusion matrix MLP.png
│   └── Confusion matrix Random Forest.png
|   └── Visualisasi Class Distribution Plot.png
|   └── Visualisasi Heatmap Korelasi.png
|   └── Visualisasi Histogram.png
|   └── Visualisasi Perbandingan Performa Model.png
|   └── Visualisasi Training dan Validation (Loss dan Accuracy).png
|
├── models/
│   ├── deep_learning_model.h5
│   ├── logistic_regression_model.pkl
│   └── random_forest_model.pkl
│
├── notebooks/
│   └── 234311052_Rian_Cahyo_UAS_Data_Science.ipynb
│
├── src/
│   └── Data_Cleaning.py
│   └── Data_Splitting.py
│   └── Data_Transformation.py
│   └── Deskripsi_Dataset.py
│   └── Import_dan_Load_Dataset.py
│   └── Kondisi_Data.py
│   └── Model_Deep_Learning_MLP.py
│   └── Model_Logistic_Regression.py
|   └── Model_Random_Forest.py
|   └── Visualisasi_EDA.py
|   └── Visualisasi_Perbandingan_3_Model.py
│
├── Laporan Proyek Machine Learning.pdf
├── Checklist Submit Proyek.md
├── LICENSE
├── README.md
└── requirements.txt
```

---
# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository
- **Jumlah Data:** 120 Baris, 6 Fitur Utama
- **Tipe:** Tabular

### Fitur Utama
| Fitur | Deskripsi |
|------|-----------|
|Temperature | Suhu tubuh pasien (Body Temperature), biasanya dalam rentang 35.5 °C hingga 41.5 °C.|
| Nausea | Kondisi mual (1 = Ya, 0 = Tidak). |
| Urine Pushing | Adanya dorongan kuat dan sering untuk buang air kecil (1 = Ya, 0 = Tidak). |
| Lumbago | Adanya nyeri punggung atau pinggang (1 = Ya, 0 = Tidak). |
| Micturition Pain | Rasa sakit saat buang air kecil (Dysuria) (1 = Ya, 0 = Tidak). |
| Burning of Urination | Diagnosis akhir: Inflammation (1 = Ya) atau No Inflammation (0 = Tidak). |

---

# 4. 🔧 Data Preparation
- **Cleaning:** Pengecekan missing values (Data bersih 100%).
- **Transformation:** Encoding target menjadi biner (Inflamasi vs Non-Inflamasi) dan Feature Scaling (StandardScaler).
- **Splitting:** Stratified Split (60% Train, 20% Val, 20% Test).
- **Handling Imbalance:** Data dianggap relatif seimbang, dan Stratified Splitting sudah diterapkan.

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** **Logistic Regression** (Linear model, simple & fast).
- **Model 2 – Advanced ML:** **Random Forest** (Ensemble berbasis Decision Tree).
- **Model 3 – Deep Learning:** **Multilayer Perceptron (MLP)** dengan arsitektur: Input(6) -> Dense(64, ReLU) -> Dropout -> Dense(32, ReLU) -> Dropout -> Output(1, Sigmoid).

---

# 6. 🧪 Evaluation
**Metrik:** **F1-Score (Macro)** & Accuracy..

### Hasil Singkat
| Model | Accuracy | F1-Score | Catatan |
|-------|--------|---------|---------|
| Baseline (LogReg) | **1.00** | **1.00** | Model Terbaik. Sempurna dan paling efisien. |
| Advanced (SVM) | 1.00 | 1.00 | Sempurna, namun lebih lambat dan kompleks. |
| Deep Learning (MLP) | 1.00 | 1.00 | Sempurna, namun paling lambat dan overfitting ringan. |

---

# 7. 🏁 Kesimpulan
- **Model terbaik:** Logistic Regression.
- **Alasan:** Model linier sudah mencapai performa maksimum (Accuracy 100%). Model yang lebih kompleks tidak memberikan nilai tambah performa dan hanya meningkatkan cost komputasi.
- **Insight penting:** Fitur gejala klinis memiliki hubungan yang sangat linier dan diskriminatif terhadap target Inflamasi Kandung Kemih.

---

# 8. 🔮 Future Work
- [x] Hyperparameter tuning lebih ekstensif
- [x] Ensemble methods (combining models)
- [ ] Menambah variasi data responden dari negara lain
- [ ] Deployment (Streamlit/FastAPI)

---

# 9. 🔁 Reproducibility
Untuk menjalankan proyek ini di lokal, gunakan environment berikut:

Clone Repository:
```bash
git clone https://github.com/Riancahyo/DataScience-Acude-Inflammations.git 

cd DataScience-Acude-Inflammations

Install Dependencies:

pip install -r requirements.txt
```
Jalankan Notebook: Buka file di notebooks/234311052_Rian_Cahyo_UAS_Data_Science.ipynb menggunakan Jupyter Notebook atau VS Code.

Gunakan environment:
**Python 3.10+**
Libraries utama:
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow` (Keras)
- `seaborn`
- `joblib`

Instalasi:
```bash
pip install -r requirements.txt
