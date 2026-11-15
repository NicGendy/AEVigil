from pathlib import Path
from typing import Any, Dict, List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import streamlit as st
import torch
import torch.nn as nn

# --- 1. KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="AEVigil | Deteksi Anomali Jaringan IoT (VRE)",
    page_icon="🛡️",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / "aevigil_bundle.joblib"
DEFAULT_SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_test_iot.csv"
NORMAL_CLASSES = ["Thing_Speak", "MQTT_Publish", "Wipro_bulb"]


# --- 2. DEFINISI ARSITEKTUR MODEL (IDENTIK DENGAN NOTEBOOK) ---
class AEVigilAutoencoder(nn.Module):
  """Arsitektur Autoencoder Simetris (92-256-128-64-32-64-128-256-92)."""

  def __init__(self, input_dim: int, use_sigmoid: bool = False) -> None:
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
    )
    decoder_layers: List[nn.Module] = [
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, input_dim),
    ]
    if use_sigmoid:
      decoder_layers.append(nn.Sigmoid())
    self.decoder = nn.Sequential(*decoder_layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.decoder(self.encoder(x))

  def get_latent(self, x: torch.Tensor) -> torch.Tensor:
    return self.encoder(x)


@st.cache_resource
def load_aevigil_bundle(bundle_path: str) -> Dict[str, Any]:
  """Memuat bundel artefak model, scaler, dan vektor ambang batas ke memori CPU."""
  payload = joblib.load(bundle_path)
  model = AEVigilAutoencoder(
      input_dim=payload["input_dim"],
      use_sigmoid=payload.get("use_sigmoid", False),
  )
  model.load_state_dict(payload["model_state_dict"])
  model.eval()
  payload["model"] = model
  return payload


@st.cache_data
def load_dataset_csv(file_source: Any) -> pd.DataFrame:
  """Memuat dataset uji dari berkas bawaan atau unggahan pengguna."""
  df = pd.read_csv(file_source)
  if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
  return df


@st.cache_data
def compute_tsne_projection(
    latent_matrix: np.ndarray,
    labels_bin: np.ndarray,
    attack_names: np.ndarray,
    sample_n: int,
    perplexity: int,
) -> pd.DataFrame:
  """Menghitung proyeksi 2D t-SNE pada ruang laten 32 dimensi."""
  idx = np.arange(len(labels_bin))
  if len(idx) > sample_n:
    _, idx = train_test_split(
        idx, test_size=sample_n, stratify=labels_bin, random_state=42
    )
  coords = TSNE(
      n_components=2, perplexity=perplexity, max_iter=1000, random_state=42
  ).fit_transform(latent_matrix[idx])
  return pd.DataFrame({
      "tsne_1": coords[:, 0],
      "tsne_2": coords[:, 1],
      "Kelas": np.where(labels_bin[idx] == 0, "Normal", "Anomali"),
      "Attack_type": attack_names[idx],
  })


def highlight_vre_rows(row: pd.Series) -> List[str]:
  """Memberi warna merah pada prediksi Anomali dan kuning pada fitur pemicu VRE."""
  styles = ["" for _ in row.index]
  if "Prediction" in row and row["Prediction"] == "Anomali":
    pred_idx = row.index.get_loc("Prediction")
    styles[pred_idx] = "background-color: #FF7F7F; color: black; font-weight: bold;"
    trig_set = row.get("_triggered_set", set())
    if isinstance(trig_set, set) and trig_set:
      for i, col_name in enumerate(row.index):
        if col_name in trig_set:
          styles[i] = "background-color: #FFF2CC; color: #7F6000; font-weight: bold;"
  return styles


# --- 3. HEADER UTAMA APLIKASI ---
st.title("🛡️ AEVigil: Sistem Deteksi Anomali Trafik Jaringan IoT")
st.markdown(
    "Platform evaluasi interaktif **Deep Autoencoder** pada dataset **RT-IoT2022** "
    "yang membandingkan pendekatan konvensional **Global Threshold (MSE Skalar)** "
    "terhadap metode optimasi **Per-Feature Threshold berbasis Vector Reconstruction Error (VRE)**."
)

if not DEFAULT_BUNDLE_PATH.exists():
  st.error(
      f"Berkas artefak `{DEFAULT_BUNDLE_PATH}` tidak ditemukan. "
      "Pastikan Anda telah menjalankan notebook eksperimen hingga Sel 4."
  )
  st.stop()

bundle = load_aevigil_bundle(str(DEFAULT_BUNDLE_PATH))
model: AEVigilAutoencoder = bundle["model"]
scaler = bundle["scaler"]
feature_cols: List[str] = bundle["feature_cols"]
categorical_cols: List[str] = bundle["categorical_cols"]
val_mse_errors: np.ndarray = bundle["val_mse_errors"]
vre_thresholds: Dict[str, np.ndarray] = bundle["vre_thresholds"]
feature_weights: np.ndarray = bundle["feature_weights"]

# --- 4. SIDEBAR KONFIGURASI ---
with st.sidebar:
  st.header("⚙️ Panel Konfigurasi")
  st.subheader("1. Sumber Data Uji")
  data_source_opt = st.radio(
      "Pilih Sumber Dataset:",
      ("Gunakan Sampel Bawaan RT-IoT2022 (2.000 Baris)", "Unggah Berkas CSV Sendiri"),
  )

  uploaded_csv = None
  if data_source_opt == "Unggah Berkas CSV Sendiri":
    uploaded_csv = st.file_uploader("Unggah CSV RT-IoT2022", type=["csv"])

  st.markdown("---")
  st.subheader("2. Metode Deteksi Anomali")
  detection_mode = st.radio(
      "Pendekatan Ambang Batas:",
      ("Optimasi: Per-Feature Threshold (VRE)", "Baseline: Global Threshold (MSE Skalar)"),
  )

  if detection_mode == "Baseline: Global Threshold (MSE Skalar)":
    global_method = st.selectbox(
        "Metode Kalibrasi Global:",
        ("Persentil", "Statistik (Mean + k*Std)", "IQR (Q3 + 1.5*IQR)"),
        index=0,
    )
    if global_method == "Persentil":
      p_val = st.slider("Persentil Validasi Normal:", 90.0, 99.9, 97.0, 0.5)
      active_global_thresh = float(np.percentile(val_mse_errors, p_val))
    elif global_method == "Statistik (Mean + k*Std)":
      k_val = st.slider("Pengali Simpangan Baku (k):", 1.0, 5.0, 2.0, 0.5)
      active_global_thresh = float(
          np.mean(val_mse_errors) + k_val * np.std(val_mse_errors)
      )
    else:
      q1, q3 = np.percentile(val_mse_errors, [25, 75])
      active_global_thresh = float(q3 + 1.5 * (q3 - q1))
    st.metric("Nilai Global Threshold (θ)", f"{active_global_thresh:.6f}")
  else:
    vre_scheme = st.selectbox(
        "Skema Ambang Per-Fitur:",
        ("Mixed", "Percentile", "Statistical"),
        index=0,
    )
    agg_strategy = st.selectbox(
        "Strategi Agregasi VRE:",
        ("any", "count", "weighted", "majority"),
        index=0,
    )
    count_pct = 10
    weight_min = 0.10
    if agg_strategy == "count":
      count_pct = st.slider("Kuota Minimum Fitur Pemicu (%):", 1, 50, 10, 1)
    elif agg_strategy == "weighted":
      weight_min = st.slider("Ambang Skor Terbobot:", 0.01, 0.50, 0.10, 0.01)

# --- 5. PEMUATAN DAN PRA-PEMROSESAN DATA ---
if data_source_opt.startswith("Gunakan Sampel"):
  if not DEFAULT_SAMPLE_CSV.exists():
    st.warning(f"Sampel bawaan `{DEFAULT_SAMPLE_CSV}` belum tersedia.")
    st.stop()
  df_raw = load_dataset_csv(str(DEFAULT_SAMPLE_CSV))
else:
  if uploaded_csv is None:
    st.info("Silakan unggah berkas `.csv` melalui panel di sebelah kiri.")
    st.stop()
  df_raw = load_dataset_csv(uploaded_csv)

df_proc = df_raw.copy()
present_cats = [c for c in categorical_cols if c in df_proc.columns]
if present_cats:
  df_proc = pd.get_dummies(df_proc, columns=present_cats, drop_first=True, dtype=float)

df_aligned = df_proc.reindex(columns=feature_cols, fill_value=0.0)
X_scaled = scaler.transform(df_aligned)
X_tensor = torch.FloatTensor(X_scaled)

# --- 6. EKSEKUSI INFERENSI MODEL ---
with torch.no_grad():
  recon_tensor = model(X_tensor)
  mse_scores = torch.mean((recon_tensor - X_tensor) ** 2, dim=1).numpy()
  l1_matrix = torch.abs(X_tensor - recon_tensor).numpy()

df_out = df_raw.copy()
feature_names_arr = np.array(feature_cols)

if detection_mode == "Baseline: Global Threshold (MSE Skalar)":
  preds_bin = (mse_scores > active_global_thresh).astype(int)
  df_out["Prediction"] = np.where(preds_bin == 1, "Anomali", "Normal")
  df_out["MSE_Global"] = np.round(mse_scores, 6)
  df_out["Alasan_Keputusan"] = np.where(
      preds_bin == 1,
      "ANOMALI: MSE (" + df_out["MSE_Global"].astype(str) + f") > θ ({active_global_thresh:.5f})",
      "NORMAL: MSE (" + df_out["MSE_Global"].astype(str) + f") <= θ ({active_global_thresh:.5f})",
  )
else:
  th_vec = vre_thresholds[vre_scheme]
  exceed_mat = l1_matrix > th_vec
  trig_counts = np.sum(exceed_mat, axis=1)

  trig_sets = [set(feature_names_arr[row_m]) for row_m in exceed_mat]
  df_out["_triggered_set"] = trig_sets
  df_out["Fitur_Terpicu"] = trig_counts

  if agg_strategy == "any":
    preds_bin = (trig_counts >= 1).astype(int)
    df_out["Alasan_Keputusan"] = np.where(
        preds_bin == 1,
        "ANOMALI: " + df_out["Fitur_Terpicu"].astype(str) + " fitur melampaui ambang (Aturan 'any' >= 1)",
        "NORMAL: 0 fitur melampaui ambang per-fitur",
    )
  elif agg_strategy == "majority":
    preds_bin = (np.mean(exceed_mat, axis=1) > 0.50).astype(int)
    df_out["Alasan_Keputusan"] = np.where(
        preds_bin == 1,
        "ANOMALI: >50% fitur melampaui ambang",
        "NORMAL: <=50% fitur melampaui ambang",
    )
  elif agg_strategy == "count":
    min_k = max(1, int(len(feature_cols) * (count_pct / 100.0)))
    preds_bin = (trig_counts >= min_k).astype(int)
    df_out["Alasan_Keputusan"] = np.where(
        preds_bin == 1,
        "ANOMALI: " + df_out["Fitur_Terpicu"].astype(str) + f" fitur terpicu (Min kuota: {min_k})",
        "NORMAL: Hanya " + df_out["Fitur_Terpicu"].astype(str) + f" fitur terpicu (< {min_k})",
    )
  else:
    w_scores = np.sum(exceed_mat * feature_weights, axis=1)
    preds_bin = (w_scores >= weight_min).astype(int)
    df_out["Skor_Bobot"] = np.round(w_scores, 4)
    df_out["Alasan_Keputusan"] = np.where(
        preds_bin == 1,
        "ANOMALI: Skor bobot (" + df_out["Skor_Bobot"].astype(str) + f") >= {weight_min:.2f}",
        "NORMAL: Skor bobot (" + df_out["Skor_Bobot"].astype(str) + f") < {weight_min:.2f}",
    )
  df_out["Prediction"] = np.where(preds_bin == 1, "Anomali", "Normal")

# --- 7. TAMPILAN METRIK RINGKASAN ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Aliran Trafik", f"{len(df_out):,}")
c2.metric("Terdeteksi Normal", f"{int(np.sum(preds_bin == 0)):,}")
c3.metric("Terdeteksi Anomali", f"{int(np.sum(preds_bin == 1)):,}")
c4.metric("Proporsi Anomali", f"{np.mean(preds_bin == 1) * 100:.2f}%")

has_ground_truth = "Attack_type" in df_out.columns
if has_ground_truth:
  y_true = df_out["Attack_type"].apply(lambda x: 0 if x in NORMAL_CLASSES else 1).to_numpy()
  cm = confusion_matrix(y_true, preds_bin)
  tn, fp, fn, tp = cm.ravel()

  st.markdown("---")
  tab1, tab2, tab3, tab4 = st.tabs([
      "📊 Evaluasi Kinerja & Confusion Matrix",
      "🔍 Inspeksi Tabel & Fitur Pemicu VRE",
      "🛡️ Analisis Serangan Lolos (False Negatives)",
      "🌌 Proyeksi Ruang Laten (t-SNE)",
  ])

  with tab1:
    col_cm, col_tbl = st.columns([1, 1.4])
    with col_cm:
      st.subheader("Matriks Kebingungan")
      fig_cm, ax_cm = plt.subplots(figsize=(4.8, 3.8))
      sns.heatmap(
          cm,
          annot=True,
          fmt=",d",
          cmap="Blues",
          cbar=False,
          xticklabels=["Prediksi Normal", "Prediksi Anomali"],
          yticklabels=["Aktual Normal", "Aktual Anomali"],
          ax=ax_cm,
          annot_kws={"size": 10, "weight": "bold"},
      )
      st.pyplot(fig_cm)
      plt.close(fig_cm)

    with col_tbl:
      st.subheader("Rincian Metrik Kinerja")
      fpr_v = fp / (fp + tn) if (fp + tn) > 0 else 0.0
      fnr_v = fn / (fn + tp) if (fn + tp) > 0 else 0.0
      df_metrics = pd.DataFrame({
          "Metrik Evaluasi": [
              "Accuracy",
              "Balanced Accuracy",
              "ROC-AUC (Skor Global)",
              "Precision (Kelas Normal)",
              "Recall / TNR (Kelas Normal)",
              "F1-Score (Kelas Normal)",
              "Precision (Kelas Anomali)",
              "Recall / TPR (Kelas Anomali)",
              "F1-Score (Kelas Anomali)",
              "False Negative Rate (FNR - Serangan Lolos)",
              "False Positive Rate (FPR - Alarm Palsu)",
          ],
          "Skor": [
              f"{accuracy_score(y_true, preds_bin):.4f}",
              f"{balanced_accuracy_score(y_true, preds_bin):.4f}",
              f"{roc_auc_score(y_true, mse_scores):.4f}",
              f"{precision_score(y_true, preds_bin, pos_label=0, zero_division=0):.4f}",
              f"{recall_score(y_true, preds_bin, pos_label=0, zero_division=0):.4f}",
              f"{f1_score(y_true, preds_bin, pos_label=0, zero_division=0):.4f}",
              f"{precision_score(y_true, preds_bin, pos_label=1, zero_division=0):.4f}",
              f"{recall_score(y_true, preds_bin, pos_label=1, zero_division=0):.4f}",
              f"{f1_score(y_true, preds_bin, pos_label=1, zero_division=0):.4f}",
              f"{fnr_v:.4f} ({fnr_v*100:.2f}%)",
              f"{fpr_v:.4f} ({fpr_v*100:.2f}%)",
          ],
      })
      st.table(df_metrics.set_index("Metrik Evaluasi"))

  with tab2:
    st.subheader("Tabel Keputusan Deteksi & Penyorotan Fitur Pemicu")
    if detection_mode.startswith("Optimasi"):
      st.caption(
          "Pada mode VRE, sel berwarna **kuning** menunjukkan atribut jaringan spesifik "
          "yang galat rekonstruksinya melampaui ambang batas fitur tersebut."
      )
      anom_idx = np.where(y_true == 1)[0]
      if len(anom_idx) > 0:
        top_trigs = pd.Series(
            np.sum(exceed_mat[anom_idx], axis=0), index=feature_cols
        ).sort_values(ascending=False).head(12)
        st.markdown("**12 Atribut Jaringan yang Paling Sering Memicu Deteksi Anomali:**")
        st.bar_chart(top_trigs)

    filter_anom = st.checkbox("Tampilkan hanya baris yang diprediksi Anomali", value=False)
    max_rows = st.slider("Jumlah baris ditampilkan:", 10, 200, 35)
    df_view = df_out[df_out["Prediction"] == "Anomali"] if filter_anom else df_out
    lead_cols = [c for c in ["Prediction", "Alasan_Keputusan", "Fitur_Terpicu", "MSE_Global", "Skor_Bobot", "Attack_type"] if c in df_view.columns]
    rest_cols = [c for c in df_view.columns if c not in lead_cols + ["_triggered_set"]]
    st.dataframe(
        df_view.head(max_rows).style.apply(highlight_vre_rows, axis=1),
        column_order=lead_cols + rest_cols,
        use_container_width=True,
    )

  with tab3:
    st.subheader("Bedah Kasus Serangan yang Lolos Deteksi (False Negatives)")
    fn_mask = (y_true == 1) & (preds_bin == 0)
    tot_anom_series = df_out.loc[y_true == 1, "Attack_type"].value_counts()
    fn_series = df_out.loc[fn_mask, "Attack_type"].value_counts()
    rows_err = []
    for att_label, tot_cnt in tot_anom_series.items():
      missed_cnt = int(fn_series.get(att_label, 0))
      det_cnt = int(tot_cnt - missed_cnt)
      rows_err.append({
          "Kategori Serangan (Attack_type)": att_label,
          "Total Populasi": int(tot_cnt),
          "Terdeteksi Benar (TP)": det_cnt,
          "Lolos Deteksi (FN)": missed_cnt,
          "Tingkat Deteksi (%)": f"{(det_cnt / tot_cnt) * 100:.2f}%",
      })
    st.table(pd.DataFrame(rows_err).set_index("Kategori Serangan (Attack_type)"))

  with tab4:
    st.subheader("Visualisasi Proyeksi Ruang Laten 32-Dimensi (t-SNE)")
    if st.button("Jalankan Proyeksi t-SNE pada Data Uji"):
      with st.spinner("Menghitung proyeksi t-SNE..."):
        with torch.no_grad():
          lat_np = model.get_latent(X_tensor).numpy()
        df_tsne = compute_tsne_projection(
            lat_np, y_true, df_out["Attack_type"].to_numpy(), min(1500, len(df_out)), 30
        )
        fig_t, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(11, 4.5))
        sns.scatterplot(
            data=df_tsne, x="tsne_1", y="tsne_2", hue="Kelas",
            palette={"Normal": "#1F77B4", "Anomali": "#D62728"}, alpha=0.75, ax=ax_t1
        )
        ax_t1.set_title("Separabilitas Normal vs Anomali")
        sns.scatterplot(
            data=df_tsne, x="tsne_1", y="tsne_2", hue="Attack_type",
            palette="tab20", alpha=0.8, ax=ax_t2
        )
        ax_t2.set_title("Distribusi per Kategori Trafik")
        ax_t2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_t)
        plt.close(fig_t)