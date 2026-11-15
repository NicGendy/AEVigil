# 🛡️ AEVigil: Optimasi Deteksi Anomali Jaringan IoT Berbasis Per-Feature Threshold (VRE)

**AEVigil** adalah sistem untuk mendeteksi anomali pada trafik jaringan *Internet of Things* (IoT). Sistem ini menggunakan **Unsupervised Deep Autoencoder** dan ambang batas adaptif untuk setiap fitur (**Vector Reconstruction Error / VRE**). Pengujian dilakukan pada dataset trafik waktu nyata **RT-IoT2022**, yang berisi 123.117 rekaman aliran jaringan, 83 atribut asli, dan 92 fitur numerik setelah *one-hot encoding*.

Proyek ini membandingkan dua metode deteksi. Metode **Global Threshold MSE** merata-ratakan galat menjadi satu nilai, sedangkan **Per-Feature Threshold (VRE)** mendeteksi anomali pada setiap atribut. Data dibagi menjadi data normal yang terpisah: **70% untuk latih, 15% untuk validasi, dan 15% untuk uji**. Pembagian ini mencegah kebocoran data (*data leakage*).

---

## ✨ Fitur Utama

- **Arsitektur Autoencoder Simetris Linier (`92-256-128-64-32-64-128-256-92`)**: Menggunakan lapisan akhir *decoder* yang linier, tanpa pemotongan *Sigmoid*. Desain ini sesuai dengan distribusi *Z-score* dari `StandardScaler`. Hasilnya, *training loss* turun **98,56%**, dengan *validation loss* terbaik **0,026462**.
- **Evaluasi Multi-Skema VRE**: Mendukung tiga skema kalibrasi ambang batas per fitur, yaitu `Percentile`, `Statistical`, dan `Mixed`. Skema ini dapat dipadukan dengan empat strategi agregasi keputusan: `any`, `count`, `weighted`, dan `majority`.
- **Penjelasan Keputusan Visual (*Explainable VRE Triggers*)**: Dasbor interaktif Streamlit menyorot atribut jaringan yang melewati ambang batas pada setiap baris serangan. Atribut tersebut ditandai dengan warna kuning.
- **Proyeksi Ruang Laten 32-Dimensi (`t-SNE`)**: Menampilkan visualisasi interaktif pemisahan klaster trafik normal (`Thing_Speak`, `MQTT_Publish`, `Wipro_bulb`) dari sembilan kategori serangan siber.
---

## 📊 Ringkasan Kinerja Eksperimen (Partisi Disjoin 70:15:15)

Pengujian menggunakan **112.487 sampel uji campuran**, yang terdiri atas 1.877 sampel normal disjoin dan 110.610 sampel anomali:

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
├── app.py                                  # Aplikasi web interaktif Streamlit (Pure NumPy Engine)
├── pyproject.toml                          # Konfigurasi proyek & dependensi uv (CUDA 12.8 lokal)
├── requirements-render.txt                 # Dependensi ringan khusus deployment CPU di Render
├── render.yaml                             # Spesifikasi Blueprint deployment otomatis Render
└── README.md
```

---

## 🚀 Panduan Instalasi & Penggunaan Lokal (dengan `uv`)

Pastikan manajer paket **[uv](https://docs.astral.sh/uv/)** sudah terpasang.

1. **Kloning repositori**:
   ```bash
   git clone https://github.com/NicGendy/AEVigil
   cd AEVigil
   ```

2. **Siapkan lingkungan virtual dan dependensi**:
   ```bash
   uv sync
   ```

3. **Jalankan aplikasi web Streamlit**:
   ```bash
   uv run streamlit run app.py
   ```
   Buka `http://localhost:8501` di peramban. Aplikasi akan memuat `models/aevigil_bundle.joblib` dan sampel `data/sample_test_iot.csv` secara otomatis.

4. **Jalankan notebook eksperimen lagi (opsional)**:
   ```bash
   uv run python -m ipykernel install --user --name aevigil --display-name "Python (AEVigil uv)"
   uv run jupyter lab
   ```
   
---

## 🧠 Justifikasi Arsitektur & Metodologi (FAQ Teknis)

### 1. Ketimpangan Kelas (*Class Imbalance* 1:59) pada Data Uji dan Validitas di Dunia Nyata

- **Mengapa data uji berisi 1.877 sampel normal dan 110.610 sampel anomali?** Dataset `RT-IoT2022` memiliki 12.507 sampel normal dan 110.610 sampel anomali. Dalam protokol *Unsupervised One-Class Learning*, *Autoencoder* dilatih hanya dengan trafik normal. Sebanyak 70% data normal (8.754 sampel) digunakan untuk pelatihan, dan 15% (1.876 sampel) digunakan untuk menentukan ambang batas. Sisa 15% (1.877 sampel) adalah trafik normal yang belum pernah dilihat model. Sampel ini diuji bersama seluruh 110.610 sampel anomali agar semua sembilan kategori serangan ikut diuji.

- **Apakah metrik ini tetap valid pada jaringan nyata yang sebagian besar trafiknya normal?** Untuk mengurangi dampak perbedaan rasio kelas (*base rate fallacy*), evaluasi utama `AEVigil` menggunakan metrik yang tidak banyak berubah saat rasio kelas berubah (*prevalence-invariant metrics*). Metrik tersebut adalah **Anomaly Recall / TPR (`99,62%`)**, **Normal Recall / TNR (`92,65%`)**, **False Negative Rate / FNR (`0,38%`)**, **Balanced Accuracy (`96,13%`)**, dan **ROC-AUC (`0,9960`)**. Nilainya dihitung berdasarkan masing-masing kelas aktual, sehingga perubahan rasio trafik di lapangan tidak langsung mengubah hasil metrik tersebut.

### 2. Alasan Memilih `VRE Mixed + Any` dibandingkan `VRE Percentile + Any`

- **`VRE Percentile + Any` (*Maximum Sensitivity*)**: Menetapkan ambang batas pada persentil ke-97 untuk setiap fitur, tanpa memperhitungkan simpangan baku. Pada dataset uji, metode ini menghasilkan **Accuracy (`0,9965`)**, **Normal F1 (`0,8911`)**, dan **Anomaly F1 (`0,9982`)** tertinggi.

- **`VRE Mixed + Any` (*Variance-Guarded / Primary Benchmark*)**: Menggunakan nilai yang lebih besar antara persentil ke-97 dan rata-rata ditambah 2,5 kali simpangan baku, yaitu $\theta_j^{\text{mix}} = \max(P_{97}, \mu_j + 2,5\sigma_j)$. Cara ini menjadi pelindung tambahan bagi fitur normal dengan variasi nilai yang lebar (*heavy-tailed variance*). Metode ini dipilih sebagai tolok ukur utama karena **Normal Recall (TNR)**-nya lebih tinggi: **`92,65%` dibandingkan `91,85%`**. Hasil ini menunjukkan bahwa metode tersebut lebih baik dalam mengurangi alarm palsu saat trafik normal berfluktuasi di lingkungan produksi.

### 3. Dampak Penurunan *Normal Recall* (TNR) dari `0,9696` menjadi `0,9265`

- TNR turun **4,31 poin persentase**. Akibatnya, jumlah *False Positive* meningkat dari 57 menjadi 138 aliran dari 1.877 sampel normal. Kenaikan ini merupakan *trade-off* probabilistik dari aturan disjungsi `any` pada 92 fitur.

- Namun, perubahan ini membantu sistem mendeteksi **2.029 serangan siber tambahan**. FNR turun **82,9%**, dari 2.448 menjadi 419 insiden. Dalam penggunaan IDS, serangan aktif yang tidak terdeteksi dapat menimbulkan dampak lebih besar daripada waktu yang diperlukan untuk memeriksa peringatan positif palsu.

### 4. Mengapa Lapisan Akhir *Decoder* Menggunakan Fungsi Linier, Bukan *Sigmoid*, *ReLU*, atau *Tanh*?

- Fitur masukan dinormalisasi dengan `StandardScaler` ($Z = \frac{X-\mu}{\sigma}$). Proses ini membuat nilai rata-rata menjadi `0`. Nilai hasil normalisasi dapat berupa angka negatif atau positif tanpa batas, dan sekitar 50% data bernilai negatif ($Z < 0$).

- Fungsi aktivasi seperti `Sigmoid` dengan rentang $[0, 1]$ atau `ReLU` dengan rentang $[0, \infty)$ tidak dapat menghasilkan nilai negatif. Akibatnya, nilai negatif pada hasil rekonstruksi menjadi `0`, dan *training loss* dapat macet di `0,75`. Sementara itu, `Tanh` memiliki rentang $[-1, 1]$, sehingga tidak dapat merekonstruksi nilai di luar rentang tersebut.

- Lapisan akhir **Linear** tanpa fungsi aktivasi ($\hat{x} = Wz + b$) dapat menghasilkan nilai negatif maupun positif. Dengan begitu, *decoder* dapat merekonstruksi rentang *Z-score* secara penuh. Penggunaan lapisan ini menurunkan *training loss* sebesar **98,56%**, hingga mencapai **0,008710**.

### 5. Tiga Skema Kalibrasi dan Empat Strategi Agregasi VRE

- **Galat vektor per fitur ($L_1$)**: Mengukur selisih antara nilai asli dan hasil rekonstruksi untuk setiap fitur:
  \[
  E_{ij} = |x_{ij} - \hat{x}_{ij}|,\quad j \in \{1,\dots,92\}.
  \]

- **Tiga skema kalibrasi ambang batas ($\theta_j$)**:
  1. `Percentile`: Menggunakan persentil ke-97 dari galat validasi untuk setiap fitur. Dengan demikian, 3% nilai galat tertinggi berada di atas ambang batas.
     \[
     \theta_j^{\text{perc}} = \text{Percentile}_{97}(E_{\cdot j}^{\text{val}})
     \]
  2. `Statistical`: Menggunakan rata-rata dan simpangan baku galat pada data validasi normal.
     \[
     \theta_j^{\text{stat}} = \mu_j + 2{,}5\sigma_j
     \]
  3. `Mixed`: Memilih nilai yang lebih besar antara ambang `Percentile` dan `Statistical`. Skema hibrida ini membantu mengurangi pengaruh pencilan ekstrem.
     \[
     \theta_j^{\text{mix}} = \max(\theta_j^{\text{perc}}, \theta_j^{\text{stat}})
     \]

- **Empat strategi agregasi keputusan**:
  1. `any`: Menandai sampel sebagai anomali jika setidaknya satu fitur melewati ambang batas. Strategi ini paling peka terhadap serangan yang hanya mengubah sedikit fitur. **Acc = 0,9950–0,9965**.
  2. `count`: Menandai sampel sebagai anomali jika setidaknya 10% fitur—atau 9 fitur—melewati ambang batas. **Acc = 0,9730–0,9929**.
  3. `weighted`: Menjumlahkan bobot fitur yang melewati ambang batas. Sampel ditandai sebagai anomali jika jumlah bobot mencapai `0,10`. Bobot berbanding terbalik dengan simpangan baku fitur ($w_j \propto 1/\sigma_j$).
     \[
     \sum w_j \mathbb{I}(E_{ij} > \theta_j) \ge 0{,}10
     \]
     **Acc = 0,9749–0,9923**.
  4. `majority`: Menandai sampel sebagai anomali jika lebih dari 50% fitur—setidaknya 47 fitur—melewati ambang batas. Akurasinya lebih rendah karena serangan jarang mengubah lebih dari separuh atribut jaringan sekaligus. **Acc = 0,8217–0,9194**.

### 6. Mengapa Deteksi `ARP_poisioning` Mencapai `94,66%` (414 dari 7.750 Serangan Tidak Terdeteksi)?

- Atribut dalam dataset `RT-IoT2022` diekstraksi menggunakan *Zeek Flowmeter*. Alat ini mencatat metrik **Layer 3/4 (IP/TCP/UDP)**. Sebaliknya, serangan *ARP Poisoning* bekerja pada **Layer 2 (Data Link)** dengan memalsukan alamat MAC (*MAC Spoofing*) untuk menempatkan penyerang di antara dua pihak (*Man-in-the-Middle* atau MitM).

- Jika penyerang meneruskan (*relay*) paket telemetri IoT yang sah, seperti `Thing_Speak` atau `MQTT_Publish`, tanpa mengubah ukuran *payload* atau bendera TCP, ciri aliran Layer 3/4 dapat terlihat seperti trafik normal. Karena itu, model dapat melewatkan sebagian serangan. Pola ini juga tampak pada visualisasi `t-SNE` di ruang laten.

### 7. Kesiapan *Real-Time*, Latensi Inferensi, dan Penggunaan di IoT

- `AEVigil` adalah *Network Intrusion Detection System* (NIDS) berbasis aliran (*flow-based*). Sistem ini dirancang untuk dijalankan pada **IoT Edge Gateway** atau **router lokal**, seperti Raspberry Pi, *fog node*, atau *gateway* yang mengumpulkan trafik dari sensor.

- Jaringan ini memiliki sembilan lapisan dan **134.140 parameter**. Dalam bentuk matriks *Pure NumPy* `float32`, ukurannya sekitar **536 KB**. Proses inferensi tidak memerlukan GPU. Waktu pemrosesan yang dilaporkan kurang dari **0,05 milidetik per aliran trafik**, atau lebih dari **20.000 aliran per detik** pada satu CPU.

---

## 📜 Lisensi & Sitasi Dataset

- **Kode sumber:** Seluruh kode dalam repositori ini menggunakan lisensi **MIT**.
- **Dataset:** Dataset **RT-IoT2022** berasal dari *UC Irvine Machine Learning Repository* dan menggunakan lisensi [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org).
  - **Sitasi:** S., B., & Nagapadma, R. (2023). *RT-IoT2022* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P338
  - **Modifikasi data:** Berkas `data/sample_test_iot.csv` berisi 2.000 baris sampel terstratifikasi dari dataset asli. Berkas ini disertakan untuk demonstrasi aplikasi.

---