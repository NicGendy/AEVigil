# 🛡️ AEVigil: Optimasi Deteksi Anomali Jaringan IoT Berbasis Per-Feature Threshold (VRE)

**AEVigil** adalah sistem deteksi anomali trafik jaringan *Internet of Things* (IoT) berbasis **Unsupervised Deep Autoencoder** yang dioptimalkan menggunakan mekanisme ambang batas adaptif per fitur (**Vector Reconstruction Error / VRE**) pada dataset trafik waktu nyata **RT-IoT2022** (123.117 rekaman aliran jaringan, 83 atribut asli / 92 fitur numerik hasil *one-hot encoding*).

Proyek ini membandingkan kelemahan perata-rataan galat skalar konvensional (**Global Threshold MSE**) terhadap ketajaman deteksi tingkat atribut pada pendekatan **Per-Feature Threshold (VRE)** melalui protokol partisi data normal disjoin murni (**70% Latih : 15% Validasi : 15% Uji**) yang bebas dari kebocoran data (*data leakage*).

---

## ✨ Fitur Utama

- **Arsitektur Autoencoder Simetris Linier (`92-256-128-64-32-64-128-256-92`)**: Menggunakan lapisan penutup *decoder* linier murni (tanpa pemotongan *Sigmoid*) yang selaras dengan distribusi *Z-score* (`StandardScaler`), menurunkan *training loss* sebesar **98,56%** hingga mencapai *validation loss* terbaik **0,026462**.
- **Evaluasi Multi-Skema VRE**: Mendukung 3 skema kalibrasi ambang per fitur (`Percentile`, `Statistical`, dan `Mixed`) yang dipadukan dengan 4 strategi agregasi keputusan (`any`, `count`, `weighted`, dan `majority`).
- **Penjelasan Keputusan Visual (*Explainable VRE Triggers*)**: Dasbor interaktif Streamlit secara otomatis menyorot atribut jaringan spesifik (berwarna kuning) yang melampaui ambang batas pada setiap baris serangan.
- **Proyeksi Ruang Laten 32-Dimensi (`t-SNE`)**: Visualisasi interaktif separabilitas klaster trafik normal (`Thing_Speak`, `MQTT_Publish`, `Wipro_bulb`) terhadap 9 kategori serangan siber.

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
├── app.py                                  # Aplikasi web interaktif Streamlit (Pure NumPy Engine)
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
   git clone [https://github.com/NicGendy/AEVigil](https://github.com/NicGendy/AEVigil)
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

## 🧠 Justifikasi Arsitektur & Metodologi (FAQ Teknis)

### 1. Ketimpangan Kelas (Class Imbalance 1:59) pada Data Uji & Validitas di Dunia Nyata
* **Mengapa rasio uji 1.877 Normal : 110.610 Anomali?** Dataset `RT-IoT2022` secara alami memiliki 12.507 sampel normal dan 110.610 sampel anomali. Sesuai protokol *Unsupervised One-Class Learning*, *Autoencoder* wajib dilatih (70% = 8.754 sampel) dan dikalibrasi ambang batasnya (15% = 1.876 sampel) secara eksklusif pada trafik normal. Tersisa 15% (1.877 sampel) trafik normal murni yang belum pernah dilihat model untuk diuji bersama 100% populasi serangan (110.610 sampel) agar kesembilan kategori serangan teruji secara utuh.
* **Apakah metrik tetap valid di jaringan nyata yang didominasi trafik normal?** Untuk menghindari bias proporsi kelas (*base rate fallacy*), evaluasi utama `AEVigil` mengandalkan metrik yang **kebal terhadap perubahan rasio kelas (*prevalence-invariant metrics*)**, yaitu **Anomaly Recall / TPR (`99,62%`)**, **Normal Recall / TNR (`92,65%`)**, **False Negative Rate / FNR (`0,38%`)**, **Balanced Accuracy (`96,13%`)**, dan **ROC-AUC (`0,9960`)**. Metrik-metrik tersebut dihitung secara horizontal per kelas aktual sehingga nilainya tetap konsisten meskipun rasio trafik di lapangan berubah.

### 2. Alasan Memilih `VRE Mixed + Any` vs. `VRE Percentile + Any`
* **`VRE Percentile + Any` (*Maximum Sensitivity*)**: Memotong ketat persentil ke-97 pada tiap fitur tanpa memperhitungkan simpangan baku, sehingga menghasilkan **Accuracy (`0,9965`)**, **Normal F1 (`0,8911`)**, dan **Anomaly F1 (`0,9982`)** tertinggi pada dataset uji.
* **`VRE Mixed + Any` (*Variance-Guarded / Primary Benchmark*)**: Mengambil nilai maksimum $\theta_j^{\text{mix}} = \max(P_{97}, \mu_j + 2,5\sigma_j)$ sebagai sabuk pengaman terhadap fitur normal yang memiliki distribusi ekor panjang (*heavy-tailed variance*). Skema ini dipilih sebagai sorotan utama karena memberikan **Normal Recall (TNR) yang lebih tinggi (`92,65%` berbanding `91,85%`)**, sehingga lebih tangguh menekan alarm palsu saat menghadapi fluktuasi trafik normal di lingkungan produksi.

### 3. Dampak Penurunan *Normal Recall* (TNR) dari `0,9696` ke `0,9265`
* Penurunan TNR sebesar `4,31%` (kenaikan *False Positive* dari 57 menjadi 138 aliran dari 1.877 sampel normal) merupakan pertukaran (*trade-off*) probabilistik dari aturan disjungsi `any` pada 92 fitur yang **sangat menguntungkan secara keamanan siber**: hanya dengan tambahan 81 peringatan positif palsu, sistem berhasil menggagalkan **2.029 serangan siber tambahan** (FNR terpangkas **82,9%** dari 2.448 menjadi 419 insiden). Dalam operasional IDS, biaya lolosnya serangan aktif jauh lebih destruktif dibandingkan verifikasi paket positif palsu.

### 4. Mekanisme Lapisan Penutup *Decoder* Linier (Mengapa Bukan *Sigmoid*, *ReLU*, atau *Tanh*?)
* Fitur masukan dinormalisasi menggunakan `StandardScaler` ($Z = \frac{X-\mu}{\sigma}$) yang memusatkan rata-rata di `0` dengan rentang nilai negatif hingga positif tak terbatas $(-\infty, +\infty)$, di mana ~50% data berada di area negatif ($Z < 0$).
* Penggunaan fungsi aktivasi terbatas pada lapisan penutup *decoder* seperti `Sigmoid` $[0, 1]$ atau `ReLU` $[0, \infty)$ akan memotong seluruh rekonstruksi nilai negatif menjadi `0` (menyebabkan *training loss* macet di `0,75`), sedangkan `Tanh` $[-1, 1]$ memotong nilai di luar $\pm 1\sigma$. Dengan lapisan penutup **Linear murni** ($\hat{x} = Wz + b$), *decoder* mampu merekonstruksi spektrum *Z-score* secara utuh dan menurunkan *training loss* sebesar **98,56%** hingga mencapai **0,008710**.

### 5. Perbedaan 3 Skema Kalibrasi dan 4 Strategi Agregasi VRE
* **Galat Vektor Per-Fitur ($L_1$)**: $E_{ij} = \vert{}x_{ij} - \hat{x}_{ij}\vert{}$ untuk tiap fitur $j \in \{1, \dots, 92\}$.
* **3 Skema Kalibrasi Ambang ($\theta_j$)**:
  1. `Percentile`: $\theta_j^{\text{perc}} = \text{Percentile}_{97}(E_{\cdot j}^{\text{val}})$ (non-parametrik, memotong 3% ekor atas tiap fitur).
  2. `Statistical`: $\theta_j^{\text{stat}} = \mu_j + 2,5\sigma_j$ (parametrik berbasis rata-rata dan simpangan baku validasi normal).
  3. `Mixed`: $\theta_j^{\text{mix}} = \max(\theta_j^{\text{perc}}, \theta_j^{\text{stat}})$ (hibrida konservatif penangkal pencilan ekstrem).
* **4 Strategi Agregasi Keputusan**:
  1. `any`: Anomali jika $\ge 1$ fitur melampaui $\theta_j$ (paling sensitif terhadap serangan berjejak sempit; `Acc = 0,9950–0,9965`).
  2. `count`: Anomali jika $\ge 10\%$ dimensi ($\ge 9$ fitur) melampaui $\theta_j$ (`Acc = 0,9730–0,9929`).
  3. `weighted`: Anomali jika akumulasi bobot fitur terpicu ($\sum w_j \mathbb{I}(E_{ij} > \theta_j)$ dengan $w_j \propto 1/\sigma_j$) $\ge 0,10$ (`Acc = 0,9749–0,9923`).
  4. `majority`: Anomali jika $> 50\%$ dimensi ($\ge 47$ fitur) melampaui $\theta_j$ (`Acc = 0,8217–0,9194`; rendah karena serangan jarang mengubah separuh atribut jaringan sekaligus).

### 6. Mengapa Deteksi `ARP_poisioning` Berada di Angka `94,66%` (414 Lolos dari 7.750)?
* Atribut pada dataset `RT-IoT2022` diekstraksi menggunakan *Zeek Flowmeter* yang merekam metrik **Layer 3/4 (IP/TCP/UDP)**. Serangan *ARP Poisoning* beroperasi pada **Layer 2 (Data Link / MAC Spoofing)** untuk memosisikan penyerang sebagai *Man-in-the-Middle* (MitM). Ketika penyerang meneruskan (*relay*) paket telemetri IoT sah (`Thing_Speak` atau `MQTT_Publish`) tanpa mengubah ukuran *payload* maupun bendera TCP, profil aliran Layer 3/4-nya berimpitan dengan trafik normal (terbukti pada visualisasi proyeksi ruang laten `t-SNE`).

### 7. Kesiapan *Real-Time*, Latensi Inferensi, dan Posisi *Deployment* di IoT
* Sebagai *Network Intrusion Detection System* (NIDS) berbasis aliran (*flow-based*), `AEVigil` dirancang untuk dijalankan pada **IoT Edge Gateway / Router Lokal** (seperti Raspberry Pi, *fog node*, atau *gateway* lokal) yang mengagregasi lalu lintas dari sensor.
* Jaringan 9 lapis ini hanya memiliki **134.140 parameter** (~536 KB dalam matriks *Pure NumPy* `float32`) tanpa ketergantungan *runtime* GPU pada fase inferensi, dengan latensi pemrosesan **< 0,05 milidetik per aliran trafik** (> 20.000 aliran/detik pada CPU tunggal).

---

## 📜 Lisensi & Sitasi Dataset

- **Kode Sumber:** Seluruh kode sumber dalam repositori ini dilisensikan di bawah **MIT License**.
- **Dataset:** Dataset **RT-IoT2022** berasal dari *UC Irvine Machine Learning Repository* dan dilisensikan di bawah [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org).
  * **Sitasi:** S., B., & Nagapadma, R. (2023). *RT-IoT2022* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P338
  * **Catatan Modifikasi Data:** Berkas `data/sample_test_iot.csv` pada repositori ini merupakan cuplikan sampel terstratifikasi (2.000 baris) yang diekstrak dari dataset asli untuk keperluan demonstrasi aplikasi.

---