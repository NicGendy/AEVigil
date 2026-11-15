import os

# --- 0. KUNCI THREAD & MEMORI SEBELUM IMPORT NUMPY/SKLEARN (PENTING UNTUK RENDER 512MB) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple
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

# --- 1. KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="AEVigil | Deteksi Anomali Jaringan IoT (VRE)",
    page_icon="🛡️",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / "aevigil_bundle.joblib"
DEFAULT_SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_test_iot.csv"
NORMAL_CLASSES = {"Thing_Speak", "MQTT_Publish", "Wipro_bulb"}


# --- 2. MESIN INFERENSI PURE NUMPY (IDENTIK DENGAN PYTORCH, HEMAT 250 MB RAM) ---
class NumpyAutoencoderEngine:
  """Mesin inferensi Autoencoder berbasis NumPy float32 (92-256-128-64-32-64-128-256-92)."""

  def __init__(self, weights: Dict[str, np.ndarray], use_sigmoid: bool = False) -> None:
    self.w = weights
    self.use_sigmoid = use_sigmoid
    assert "encoder.0.weight" in self.w, "Bobot encoder.0.weight tidak ditemukan."

  @staticmethod
  def _linear_relu(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = x @ w.T + b
    np.maximum(out, 0.0, out=out)  # In-place ReLU untuk menghemat alokasi RAM
    return out

  def encode(self, x: np.ndarray) -> np.ndarray:
    """Memetakan fitur masukan (N, 92) ke ruang laten (N, 32)."""
    h = self._linear_relu(x, self.w["encoder.0.weight"], self.w["encoder.0.bias"])
    h = self._linear_relu(h, self.w["encoder.2.weight"], self.w["encoder.2.bias"])
    h = self._linear_relu(h, self.w["encoder.4.weight"], self.w["encoder.4.bias"])
    z = self._linear_relu(h, self.w["encoder.6.weight"], self.w["encoder.6.bias"])
    return z

  def forward(self, x: np.ndarray) -> np.ndarray:
    """Merekonstruksi fitur masukan dari ruang laten (N, 92)."""
    z = self.encode(x)
    h = self._linear_relu(z, self.w["decoder.0.weight"], self.w["decoder.0.bias"])
    h = self._linear_relu(h, self.w["decoder.2.weight"], self.w["decoder.2.bias"])
    h = self._linear_relu(h, self.w["decoder.4.weight"], self.w["decoder.4.bias"])
    out = h @ self.w["decoder.6.weight"].T + self.w["decoder.6.bias"]
    if self.use_sigmoid:
      out = 1.0 / (1.0 + np.exp(-np.clip(out, -30.0, 30.0)))
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    return out


@st.cache_resource
def load_aevigil_bundle(bundle_path: str) -> Dict[str, Any]:
  """Memuat bundel artefak ke memori satu kali untuk seluruh sesi pengguna."""
  payload = joblib.load(bundle_path)
  if "numpy_state_dict" in payload:
    np_weights = payload["numpy_state_dict"]
  else:
    # Fallback otomatis jika masih berformat tensor
    np_weights = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in payload["model_state_dict"].items()
    }
  payload["engine"] = NumpyAutoencoderEngine(
      weights=np_weights,
      use_sigmoid=bool(payload.get("use_sigmoid", False)),
  )
  return payload


@st.cache_data(max_entries=2)
def load_dataset_csv(file_source: Any, max_limit: int = 5000) -> Tuple[pd.DataFrame, bool]:
  """Memuat CSV dengan batas baris aman agar tidak memicu Out-of-Memory di Render."""
  df = pd.read_csv(file_source)
  if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
  was_truncated = False
  if len(df) > max_limit:
    if "Attack_type" in df.columns:
      _, df = train_test_split(
          df, test_size=max_limit, stratify=df["Attack_type"], random_state=42
      )
    else:
      df = df.sample(n=max_limit, random_state=42)
    df = df.reset_index(drop=True)
    was_truncated = True
  return df, was_truncated


@st.cache_data(max_entries=2)
def compute_tsne_projection(
    latent_matrix: np.ndarray,
    labels_bin: np.ndarray,
    attack_names: np.ndarray,
    sample_n: int,
    perplexity: int,
) -> pd.DataFrame:
  """Menghitung proyeksi 2D t-SNE dengan sampel terkendali."""
  idx = np.arange(len(labels_bin))
  if len(idx) > sample_n:
    _, idx = train_test_split(
        idx, test_size=sample_n, stratify=labels_bin, random_state=42
    )
  coords = TSNE(
      n_components=2,
      perplexity=perplexity,
      max_iter=750,
      random_state=42,
      n_jobs=1,
  ).fit_transform(latent_matrix[idx])
  return pd.DataFrame({
      "tsne_1": coords[:, 0],
      "tsne_2": coords[:, 1],
      "Kelas": np.where(labels_bin[idx] == 0, "Normal", "Anomali"),
      "Attack_type": attack_names[idx],
  })


def highlight_vre_rows(row: pd.Series) -> List[str]:
  """Menyorot merah kolom Prediction Anomali dan kuning pada fitur pemicu VRE."""
  styles = ["" for _ in row.index]
  if "Prediction" in row and row["Prediction"] == "Anomali":
    pred_idx = row.index.get_loc("Prediction")
    styles[pred_idx] = (
        "background-color: #FF7F7F; color: black; font-weight: bold;"
    )
    trig_set = row.get("_triggered_set", set())
    if isinstance(trig_set, set) and trig_set:
      for i, col_name in enumerate(row.index):
        if col_name in trig_set:
          styles[i] = (
              "background-color: #FFF2CC; color: #7F6000; font-weight: bold;"
          )
  return styles


# --- 3. HEADER UTAMA APLIKASI ---
st.title("🛡️ AEVigil: Sistem Deteksi Anomali Trafik Jaringan IoT")
st.markdown(
    "Platform evaluasi interaktif **Unsupervised Deep Autoencoder** pada"
    " dataset **RT-IoT2022** yang membandingkan pendekatan konvensional"
    " **Global Threshold (MSE Skalar)** terhadap metode optimasi **Per-Feature"
    " Threshold berbasis Vector Reconstruction Error (VRE)**."
)

if not DEFAULT_BUNDLE_PATH.exists():
  st.error(f"Berkas artefak `{DEFAULT_BUNDLE_PATH}` tidak ditemukan.")
  st.stop()

bundle = load_aevigil_bundle(str(DEFAULT_BUNDLE_PATH))
engine: NumpyAutoencoderEngine = bundle["engine"]
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
      (
          "Gunakan Sampel Bawaan RT-IoT2022 (2.000 Baris)",
          "Unggah Berkas CSV Sendiri",
      ),
  )

  uploaded_csv = None
  if data_source_opt == "Unggah Berkas CSV Sendiri":
    uploaded_csv = st.file_uploader("Unggah CSV RT-IoT2022", type=["csv"])

  st.markdown("---")
  st.subheader("2. Metode Deteksi Anomali")
  detection_mode = st.radio(
      "Pendekatan Ambang Batas:",
      (
          "Optimasi: Per-Feature Threshold (VRE)",
          "Baseline: Global Threshold (MSE Skalar)",
      ),
  )

  active_global_thresh: float = float(np.percentile(val_mse_errors, 97.0))
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

# --- 5. PEMUATAN DAN PRA-PEMROSESAN DATA HEMAT MEMORI ---
if data_source_opt.startswith("Gunakan Sampel"):
  if not DEFAULT_SAMPLE_CSV.exists():
    st.warning(f"Sampel bawaan `{DEFAULT_SAMPLE_CSV}` belum tersedia.")
    st.stop()
  df_raw, truncated = load_dataset_csv(str(DEFAULT_SAMPLE_CSV))
else:
  if uploaded_csv is None:
    st.info("Silakan unggah berkas `.csv` melalui panel di sebelah kiri.")
    st.stop()
  df_raw, truncated = load_dataset_csv(uploaded_csv)
  if truncated:
    st.info(
        "Info: Dataset unggahan disampling secara terstratifikasi ke 5.000 baris "
        "untuk menjaga stabilitas memori server."
    )

present_cats = [c for c in categorical_cols if c in df_raw.columns]
df_proc = (
    pd.get_dummies(df_raw, columns=present_cats, drop_first=True, dtype=np.float32)
    if present_cats
    else df_raw
)
df_aligned = df_proc.reindex(columns=feature_cols, fill_value=0.0)
X_scaled = scaler.transform(df_aligned).astype(np.float32)
del df_proc, df_aligned  # Bebaskan memori DataFrame sementara

assert not np.isnan(X_scaled).any(), "Terdeteksi nilai NaN setelah penskalaan."

# --- 6. EKSEKUSI INFERENSI PURE NUMPY ---
recon_arr = engine.forward(X_scaled)
diff_arr = X_scaled - recon_arr
mse_scores = np.mean(diff_arr ** 2, axis=1)
l1_matrix = np.abs(diff_arr)
del recon_arr, diff_arr

df_out = df_raw.copy()
feature_names_arr = np.array(feature_cols)
exceed_mat = None

if detection_mode == "Baseline: Global Threshold (MSE Skalar)":
  preds_bin = (mse_scores > active_global_thresh).astype(np.int32)
  df_out["Prediction"] = np.where(preds_bin == 1, "Anomali", "Normal")
  df_out["MSE_Global"] = np.round(mse_scores, 6)
  df_out["Alasan_Keputusan"] = np.where(
      preds_bin == 1,
      "ANOMALI: MSE ("
      + df_out["MSE_Global"].astype(str)
      + f") > θ ({active_global_thresh:.5f})",
      "NORMAL: MSE ("
      + df_out["MSE_Global"].astype(str)
      + f") <= θ ({active_global_thresh:.5f})",
  )
else:
  th_vec = vre_thresholds[vre_scheme]
  exceed_mat = l1_matrix > th_vec
  trig_counts = np.sum(exceed_mat, axis=1)
  df_out["Fitur_Terpicu"] = trig_counts

  if agg_strategy == "any":
    preds_bin = (trig_counts >= 1).astype(np.int32)
    df_out["Alasan_Keputusan"] = np.where(
        preds_bin == 1,
        "ANOMALI: "
        + df_out["Fitur_Terpicu"].astype(str)
        + " fitur melampaui ambang (Aturan 'any' >= 1)",
        "NORMAL: 0 fitur melampaui ambang per-fitur",
    )
  elif agg_strategy == "majority":
    preds_bin = (np.mean(exceed_mat, axis=1) > 0.50).astype(np.int32)
    df_out["Alasan_Keputusan"] = np.where(
        preds_bin == 1,
        "ANOMALI: >50% fitur melampaui ambang",
        "NORMAL: <=50% fitur melampaui ambang",
    )
  elif agg_strategy == "count":
    min_k = max(1, int(len(feature_cols) * (count_pct / 100.0)))
    preds_bin = (trig_counts >= min_k).astype(np.int32)
    df_out["Alasan_Keputusan"] = np.where(
        preds_bin == 1,
        "ANOMALI: "
        + df_out["Fitur_Terpicu"].astype(str)
        + f" fitur terpicu (Min kuota: {min_k})",
        "NORMAL: Hanya "
        + df_out["Fitur_Terpicu"].astype(str)
        + f" fitur terpicu (< {min_k})",
    )
  else:
    w_scores = np.sum(exceed_mat * feature_weights, axis=1)
    preds_bin = (w_scores >= weight_min).astype(np.int32)
    df_out["Skor_Bobot"] = np.round(w_scores, 4)
    df_out["Alasan_Keputusan"] = np.where(
        preds_bin == 1,
        "ANOMALI: Skor bobot ("
        + df_out["Skor_Bobot"].astype(str)
        + f") >= {weight_min:.2f}",
        "NORMAL: Skor bobot ("
        + df_out["Skor_Bobot"].astype(str)
        + f") < {weight_min:.2f}",
    )
  df_out["Prediction"] = np.where(preds_bin == 1, "Anomali", "Normal")

del l1_matrix

# --- 7. TAMPILAN METRIK RINGKASAN ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Aliran Trafik", f"{len(df_out):,}")
c2.metric("Terdeteksi Normal", f"{int(np.sum(preds_bin == 0)):,}")
c3.metric("Terdeteksi Anomali", f"{int(np.sum(preds_bin == 1)):,}")
c4.metric("Proporsi Anomali", f"{np.mean(preds_bin == 1) * 100:.2f}%")

has_ground_truth = "Attack_type" in df_out.columns
if has_ground_truth:
  y_true = (
      df_out["Attack_type"]
      .apply(lambda x: 0 if x in NORMAL_CLASSES else 1)
      .to_numpy(dtype=np.int32)
  )
  cm = confusion_matrix(y_true, preds_bin)
  tn, fp, fn, tp = cm.ravel()

  st.markdown("---")
  tab1, tab2, tab3, tab4 = st.tabs([
      "📊 Evaluasi Kinerja & Kurva ROC",
      "🔍 Inspeksi Tabel & Fitur Pemicu VRE",
      "🛡️ Analisis Serangan Lolos (False Negatives)",
      "🌌 Proyeksi Ruang Laten (t-SNE)",
  ])

  with tab1:
    col_cm, col_tbl = st.columns([1, 1.35])
    with col_cm:
      st.subheader("Matriks Kebingungan")
      fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
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
              "ROC-AUC (Skor MSE Global)",
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

    st.markdown("---")
    col_roc, col_dist = st.columns(2)
    with col_roc:
      st.subheader("Kurva ROC-AUC")
      fpr_arr, tpr_arr, _ = roc_curve(y_true, mse_scores)
      auc_val = roc_auc_score(y_true, mse_scores)
      fig_roc, ax_roc = plt.subplots(figsize=(5.0, 3.4))
      ax_roc.plot(
          fpr_arr,
          tpr_arr,
          color="#D95F02",
          lw=2,
          label=f"ROC Curve (AUC = {auc_val:.4f})",
      )
      ax_roc.plot([0, 1], [0, 1], color="navy", lw=1.2, linestyle="--")
      ax_roc.set_xlabel("False Positive Rate")
      ax_roc.set_ylabel("True Positive Rate (Recall)")
      ax_roc.legend(loc="lower right")
      ax_roc.grid(True, linestyle="--", alpha=0.3)
      st.pyplot(fig_roc)
      plt.close(fig_roc)

    with col_dist:
      st.subheader("Distribusi Log10(MSE Error) Normal vs Anomali")
      fig_d, ax_d = plt.subplots(figsize=(5.0, 3.4))
      log_mse = np.log10(mse_scores + 1e-8)
      ax_d.hist(
          log_mse[y_true == 0],
          bins=35,
          color="#1F77B4",
          label="Normal",
          density=True,
          alpha=0.55,
      )
      ax_d.hist(
          log_mse[y_true == 1],
          bins=35,
          color="#D62728",
          label="Anomali",
          density=True,
          alpha=0.55,
      )
      ax_d.axvline(
          np.log10(active_global_thresh + 1e-8),
          color="green",
          linestyle="--",
          lw=1.8,
          label="Log10(Global θ)",
      )
      ax_d.set_xlabel("Log10(Reconstruction Error MSE)")
      ax_d.set_ylabel("Kerapatan (Density)")
      ax_d.legend()
      ax_d.grid(True, linestyle="--", alpha=0.3)
      st.pyplot(fig_d)
      plt.close(fig_d)

  with tab2:
    st.subheader("Tabel Keputusan Deteksi & Penyorotan Fitur Pemicu")
    if detection_mode.startswith("Optimasi") and exceed_mat is not None:
      st.caption(
          "Pada mode VRE, sel berwarna **kuning** menunjukkan atribut jaringan"
          " spesifik yang galat rekonstruksinya melampaui ambang batas fitur"
          " tersebut."
      )
      anom_idx = np.where(y_true == 1)[0]
      if len(anom_idx) > 0:
        top_trigs = (
            pd.Series(
                np.sum(exceed_mat[anom_idx], axis=0), index=feature_cols
            )
            .sort_values(ascending=False)
            .head(12)
        )
        st.markdown(
            "**12 Atribut Jaringan yang Paling Sering Memicu Deteksi Anomali:**"
        )
        st.bar_chart(top_trigs)

    filter_anom = st.checkbox(
        "Tampilkan hanya baris yang diprediksi Anomali", value=False
    )
    max_rows = st.slider("Jumlah baris ditampilkan:", 10, 100, 30)
    mask_view = (
        (df_out["Prediction"] == "Anomali").to_numpy()
        if filter_anom
        else np.ones(len(df_out), dtype=bool)
    )
    pos_indices = np.where(mask_view)[0][:max_rows]
    df_slice = df_out.iloc[pos_indices].copy()

    # Lazy Evaluation: Hanya bangun set nama fitur untuk baris yang tampil di layar!
    if exceed_mat is not None:
      df_slice["_triggered_set"] = [
          set(feature_names_arr[exceed_mat[idx]]) for idx in pos_indices
      ]

    lead_cols = [
        c
        for c in [
            "Prediction",
            "Alasan_Keputusan",
            "Fitur_Terpicu",
            "MSE_Global",
            "Skor_Bobot",
            "Attack_type",
        ]
        if c in df_slice.columns
    ]
    rest_cols = [
        c
        for c in df_slice.columns
        if c not in lead_cols + ["_triggered_set"]
    ]
    st.dataframe(
        df_slice.style.apply(highlight_vre_rows, axis=1),
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
    st.table(
        pd.DataFrame(rows_err).set_index("Kategori Serangan (Attack_type)")
    )

  with tab4:
    st.subheader("Visualisasi Proyeksi Ruang Laten 32-Dimensi (t-SNE)")
    if st.button("Jalankan Proyeksi t-SNE pada Data Uji"):
      with st.spinner("Menghitung proyeksi t-SNE..."):
        lat_np = engine.encode(X_scaled)
        df_tsne = compute_tsne_projection(
            lat_np,
            y_true,
            df_out["Attack_type"].to_numpy(),
            min(1200, len(df_out)),
            30,
        )
        fig_t, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
        sns.scatterplot(
            data=df_tsne,
            x="tsne_1",
            y="tsne_2",
            hue="Kelas",
            palette={"Normal": "#1F77B4", "Anomali": "#D62728"},
            alpha=0.75,
            s=20,
            ax=ax_t1,
        )
        ax_t1.set_title("Separabilitas Normal vs Anomali")
        sns.scatterplot(
            data=df_tsne,
            x="tsne_1",
            y="tsne_2",
            hue="Attack_type",
            palette="tab20",
            alpha=0.8,
            s=20,
            ax=ax_t2,
        )
        ax_t2.set_title("Distribusi per Kategori Trafik")
        ax_t2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_t)
        plt.close(fig_t)

gc.collect()