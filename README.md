# 🛡️ AEVigil: Optimasi Deteksi Anomali Jaringan IoT Berbasis Per-Feature Threshold (VRE)

**AEVigil** adalah sistem deteksi anomali trafik jaringan *Internet of Things* (IoT) berbasis **Unsupervised Deep Autoencoder** yang dioptimalkan menggunakan mekanisme ambang batas adaptif per fitur (**Vector Reconstruction Error / VRE**) pada dataset trafik waktu nyata **RT-IoT2022** (123.117 rekaman aliran jaringan, 83 atribut asli / 92 fitur numerik hasil *one-hot encoding*).

Proyek ini membandingkan kelemahan perata-rataan galat skalar konvensional (**Global Threshold MSE**) terhadap ketajaman deteksi tingkat atribut pada pendekatan **Per-Feature Threshold (VRE)** melalui protokol partisi data normal disjoin murni (**70% Latih : 15% Validasi : 15% Uji**) yang bebas dari kebocoran data (*data leakage*).

---

## ✨ Fitur Utama

* **Arsitektur Autoencoder Simetris Linier (`92-256-128-64-32-64-128-256-92`)**: Menggunakan lapisan penutup *decoder* linier murni (tanpa pemotongan *Sigmoid*) yang selaras dengan distribusi *Z-score* (`StandardScaler`), menurunkan *training loss* sebesar **98,56%** hingga mencapai *validation loss* terbaik **0,026462**.
* **Evaluasi Multi-Skema VRE**: Mendukung 3 skema kalibrasi ambang per fitur (`Percentile`, `Statistical`, dan `Mixed`) yang dipadukan dengan 4 strategi agregasi keputusan (`any`, `count`, `weighted`, dan `majority`).
* **Penjelasan Keputusan Visual (*Explainable VRE Triggers*)**: Dasbor interaktif Streamlit secara otomatis menyorot atribut jaringan spesifik (berwarna kuning) yang melampaui ambang batas pada setiap baris serangan.
* **Proyeksi Ruang Laten 32-Dimensi (`t-SNE`)**: Visualisasi interaktif separabilitas klaster trafik normal (`Thing_Speak`, `MQTT_Publish`, `Wipro_bulb`) terhadap 9 kategori serangan siber.

---

## 📊 Ringkasan Kinerja Eksperimen (Partisi Disjoin 70:15:15)

Pengujian dilakukan pada **112.487 sampel uji campuran** (1.877 sampel normal disjoin dan 110.610 sampel anomali):

| Skenario Deteksi | Accuracy | Normal Precision | Normal Recall (TNR) | Normal F1-Score | Anomaly Precision | Anomaly Recall (TPR) | Anomaly F1-Score | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Global Threshold P97)** | 0,9777 | 0,4264 | 0,9696 | 0,5924 | 0,9995 | 0,9779 | 0,9886 | 2,21% (2.448 lolos) |
| **AEVigil (VRE Mixed + Any)** | **0,9950** | **0,8058 (+89,0%)** | **0,9265** | **0,8620 (+45,5%)** | **0,9987** | **0,9962** | **0,9975** | **0,38% (-82,9% / 419 lolos)** |
| **AEVigil (VRE Percentile + Any)** | **0,9965** | **0,8911** | **0,9185** | **0,8911** | **0,9984** | **0,9965** | **0,9982** | **0,35%** |

### Tingkat Deteksi per Kategori Serangan (`VRE Mixed + Any`)

| Kategori Serangan (`Attack_type`) | Total Sampel Uji | Terdeteksi Benar (TP) | Lolos Deteksi (FN) | Tingkat Deteksi (%) |
| :--- | :---: | :---: | :---: | :---: |
| `DOS_SYN_Hping` | 94.659 | 94.659 | 0 | **100,00%** |
| `ARP_poisioning` | 7.750 | 7.336 | 414 | **94,66%** |
| `NMAP_UDP_SCAN` | 2.590 | 2.586 | 4 | **99,85%** |
| `NMAP_XMAS_TREE_SCAN` | 2.010 | 2.010 | 0 | **100,00%** |
| `NMAP_OS_DETECTION` | 2.000 | 2.000 | 0 | **100,00%** |
| `NMAP_TCP_scan` | 1.002 | 1.002 | 0 | **100,00%** |
| `DDOS_Slowloris` | 534 | 534 | 0 | **100,00%** |
| `Metasploit_Brute_Force_SSH` | 37 | 36 | 1 | **97,30%** |
| `NMAP_FIN_SCAN` | 28 | 28 | 0 | **100,00%** |
| **Total Keseluruhan Anomali** | **110.610** | **110.191** | **419** | **99,62%** |

---

## 📁 Struktur Direktori Repositori

```text
AEVigil/
├── data/
│   └── sample_test_iot.csv                 # Sampel uji terstratifikasi (2.000 baris) untuk demo instan
├── models/
│   └── aevigil_bundle.joblib               # Artefak terbundel (bobot model, scaler, kolom, & ambang VRE)
├── notebooks/
│   └── AEVigil_JISKA_Experiment.ipynb      # Notebook pelatihan, evaluasi VRE, & ekspor grafik 300 DPI
├── reports/
│   └── figures/                            # Grafik publikasi resolusi tinggi (300 DPI)
├── app.py                                  # Aplikasi web interaktif Streamlit
├── pyproject.toml                          # Konfigurasi proyek & dependensi uv (CUDA 12.8 lokal)
├── requirements-render.txt                 # Dependensi ringan khusus deployment CPU di Render
├── render.yaml                             # Spesifikasi Blueprint deployment otomatis Render
└── README.md
```

---

## 🚀 Panduan Instalasi & Penggunaan Lokal (dengan `uv`)

Pastikan Anda telah menginstal manajer paket **[uv](https://docs.astral.sh/uv/)**.

1. **Kloning Repositori**:
   ```bash
   git clone https://github.com/NicGendy/AEVigil
   cd AEVigil
   ```

2. **Sinkronisasi Lingkungan Virtual & Dependensi**:
   ```bash
   uv sync
   ```

3. **Menjalankan Aplikasi Web Interaktif Streamlit**:
   ```bash
   uv run streamlit run app.py
   ```
   Buka `http://localhost:8501` pada peramban Anda. Aplikasi akan langsung memuat `models/aevigil_bundle.joblib` dan sampel `data/sample_test_iot.csv` secara otomatis.

4. **Menjalankan Ulang Notebook Eksperimen (Opsional)**:
   ```bash
   uv run python -m ipykernel install --user --name aevigil --display-name "Python (AEVigil uv)"
   uv run jupyter lab
   ```
---

## 📜 Lisensi & Sitasi Dataset

* **Kode Sumber:** Seluruh kode sumber dalam repositori ini dilisensikan di bawah **MIT License**.
* **Dataset:** Dataset **RT-IoT2022** berasal dari *UC Irvine Machine Learning Repository* dan dilisensikan di bawah [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org).
  * **Sitasi:** S., B., & Nagapadma, R. (2023). *RT-IoT2022* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P338
  * **Catatan Modifikasi Data:** Berkas `data/sample_test_iot.csv` pada repositori ini merupakan cuplikan sampel terstratifikasi (2.000 baris) yang diekstrak dari dataset asli untuk keperluan demonstrasi aplikasi.
