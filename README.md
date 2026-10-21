# Bank Transaction Segmentation & Classification

Proyek akhir kelas **Belajar Machine Learning untuk Pemula** (Dicoding). Proyek ini terdiri dari dua tahap yang saling berhubungan: **clustering** untuk mengelompokkan nasabah berdasarkan pola transaksinya, lalu **klasifikasi** untuk memprediksi kelompok tersebut dari fitur transaksi.

## Alur Proyek

```
Dataset transaksi bank
        │
        ▼
[1] Clustering (K-Means)  ──►  data_clustering.csv / data_clustering_inverse.csv (kolom Target)
        │
        ▼
[2] Klasifikasi (Decision Tree, Random Forest, Random Forest + GridSearchCV)
```

Label cluster dari tahap 1 dipakai sebagai kolom `Target` pada tahap 2.

## Dataset

Dataset transaksi bank dengan sekitar **2.512 baris** yang berisi atribut transaksi, demografi nasabah, dan pola penggunaan (cocok untuk eksplorasi fraud detection dan anomali). Dataset dimuat langsung dari Google Sheets (CSV) di dalam notebook.

Fitur yang dipakai setelah pra-pemrosesan:

| Tipe | Fitur |
|---|---|
| Numerik | `TransactionAmount`, `CustomerAge`, `TransactionDuration`, `LoginAttempts`, `AccountBalance` |
| Kategorikal | `TransactionType`, `Location`, `Channel`, `CustomerOccupation` |

Kolom ID, alamat IP, dan tanggal dibuang karena tidak relevan untuk pemodelan.

## Tahap 1 — Clustering

Notebook: `[Clustering]_Submission_Akhir_BMLP_Alesandro_Yoel_Deca_Putranto.ipynb`

1. **Pembersihan data**: hapus nilai kosong dan 21 baris duplikat, buang kolom ID/IP/tanggal.
2. **Encoding**: `LabelEncoder` untuk fitur kategorikal.
3. **Outlier**: dihapus dengan metode IQR, menyisakan **1.945 baris**.
4. **Scaling**: `StandardScaler` untuk fitur numerik.
5. **Binning**: `CustomerAge` dikelompokkan menjadi Muda / Dewasa / Tua (`pd.qcut`).
6. **Pemilihan jumlah cluster**: `KElbowVisualizer` dengan metrik silhouette (k = 2–10), dipilih **k = 3**.
7. **Model**: `KMeans(n_clusters=3, random_state=42)` dengan **Silhouette Score ≈ 0,496**.
8. **Visualisasi**: reduksi dimensi PCA 2 komponen.
9. **Eksperimen tambahan**: K-Means yang dilatih pada data hasil PCA (`kmeans_pca`).
10. **Interpretasi**: agregasi mean/min/max per cluster, sebelum dan sesudah `inverse_transform`.
11. **Ekspor**: hasil disimpan sebagai `data_clustering.csv` dan `data_clustering_inverse.csv`.

## Tahap 2 — Klasifikasi

Notebook: `[Klasifikasi]_Submission_Akhir_BMLP_Alesandro_Yoel_Deca_Putranto.ipynb`

- Input: `data_clustering_inverse.csv`, fitur kategorikal di-encode dengan `pd.get_dummies`.
- Pembagian data: 80% latih / 20% uji (`stratify=y`, `random_state=42`).

| Model | Accuracy | Macro F1 |
|---|---|---|
| Decision Tree | 0,95 | 0,95 |
| Random Forest | 0,99 | 0,99 |
| Random Forest + GridSearchCV (cv=5) | 0,99 | 0,99 |

Hyperparameter yang di-tuning: `n_estimators`, `max_depth`, `min_samples_split`.

> Catatan: label `Target` berasal dari K-Means pada fitur yang sama, sehingga skor klasifikasi yang tinggi wajar terjadi. Skor ini menunjukkan seberapa mudah cluster dipisahkan oleh model, bukan akurasi terhadap label asli di dunia nyata.

## Struktur Repository

```
.
├── README.md
├── notebooks/
│   ├── [Clustering]_Submission_Akhir_BMLP_....ipynb
│   └── [Klasifikasi]_Submission_Akhir_BMLP_....ipynb
├── data/
│   ├── data_clustering.csv
│   └── data_clustering_inverse.csv
└── models/
    ├── model_clustering.h5
    ├── PCA_model_clustering.h5
    ├── decision_tree_model.h5
    ├── explore_randomforest_classification.h5
    └── tuning_classification.h5
```

## Cara Menjalankan

1. Clone repository dan install dependensi:

   ```bash
   git clone github.com/alesandroyoel/Bank-transaction-clustering-classification.git
   cd Bank-transaction-clustering-classification
   pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick joblib jupyter
   ```

2. Jalankan notebook **Clustering** terlebih dahulu sampai selesai agar `data_clustering_inverse.csv` terbentuk.
3. Jalankan notebook **Klasifikasi** (sesuaikan path file CSV jika tidak memakai Google Colab).

## Teknologi

Python, pandas, NumPy, scikit-learn, yellowbrick, seaborn, matplotlib, joblib.

## Penulis

**Alesandro Yoel Deca Putranto**
Proyek ini dibuat sebagai submission akhir kelas *Belajar Machine Learning untuk Pemula* di Dicoding.
