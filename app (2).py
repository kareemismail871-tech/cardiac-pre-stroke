# app.py
import os
import io
import json
import random
import re
import base64
from io import BytesIO

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, spectrogram
from scipy.stats import skew, kurtosis
import wfdb

# optional ML libs (only used if artifacts exist)
try:
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
except Exception:
    joblib = None
    SimpleImputer = None
    StandardScaler = None

# PDF generation
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🩺", layout="wide")
st.title("🩺 Cardiac Pre-Stroke — AI ECG Analyzer")

# ---------- Utility functions ----------
def fig_to_bytes(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return buf

def make_heart_png(width=600, height=300, fill_color="#eef6ff"):
    if not PIL_AVAILABLE:
        return None
    img = Image.new("RGBA", (width, height), (255,255,255,0))
    draw = ImageDraw.Draw(img)
    x = width/2
    y = height/3
    size = min(width, height)/3.2
    left_box = [x - size*1.3, y - size, x, y + size*0.8]
    right_box = [x, y - size, x + size*1.3, y + size*0.8]
    draw.pieslice(left_box, 180, 360, fill=fill_color)
    draw.pieslice(right_box, 180, 360, fill=fill_color)
    points = [(x - size*1.3, y + size*0.3), (x + size*1.3, y + size*0.3), (x, y + size*2)]
    draw.polygon(points, fill=fill_color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def extract_micro_features(sig, fs=250):
    sig = np.asarray(sig).astype(float)
    if sig.size == 0:
        return np.zeros(11)
    mean = sig.mean()
    std = sig.std()
    mn = sig.min()
    mx = sig.max()
    ptp = np.ptp(sig)
    rms = np.sqrt(np.mean(sig**2))
    med = np.median(sig)
    q25 = np.percentile(sig,25)
    q75 = np.percentile(sig,75)
    sk = skew(sig)
    kurt = kurtosis(sig)
    return np.array([mean,std,mn,mx,ptp,rms,med,q25,q75,sk,kurt])

def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load {os.path.basename(path)}: {e}")
        return None

# ---------- Model artifact paths (edit if needed) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ecg_stack_model_pro.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "ecg_scaler_pro.joblib")
IMPUTER_PATH = os.path.join(BASE_DIR, "ecg_imputer_pro.joblib")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "feature_columns.json")
LABEL_ENC_PATH = os.path.join(BASE_DIR, "label_encoder.joblib")

# ---------- Try load artifacts ----------
model = None
scaler = None
imputer = None
feature_columns = None
label_encoder = None
artifacts_available = False

if joblib is not None:
    if os.path.exists(MODEL_PATH):
        model = safe_load_joblib(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = safe_load_joblib(SCALER_PATH)
    if os.path.exists(IMPUTER_PATH):
        imputer = safe_load_joblib(IMPUTER_PATH)
    if os.path.exists(FEATURE_COLS_PATH):
        try:
            with open(FEATURE_COLS_PATH,"r") as f:
                feature_columns = json.load(f)
        except Exception as e:
            st.warning(f"feature_columns.json load error: {e}")
    if os.path.exists(LABEL_ENC_PATH) and joblib:
        label_encoder = safe_load_joblib(LABEL_ENC_PATH)

    artifacts_available = any([model, scaler, imputer, feature_columns, label_encoder])

# ---------- UI: upload ----------
st.markdown("Upload `.hea` and `.dat` files (WFDB). If model artifacts are present the app will attempt prediction.")
col1, col2 = st.columns(2)
with col1:
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])
with col2:
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])

# ---------- If user uploaded files ----------
if hea_file and dat_file:
    # save temporary files to work with wfdb
    try:
        # write to current dir with the record name
        record_name = hea_file.name.replace('.hea','')
        with open(hea_file.name, "wb") as f:
            f.write(hea_file.read())
        with open(dat_file.name, "wb") as f:
            f.write(dat_file.read())
        # read record
        record = wfdb.rdrecord(record_name)
        ecg_signal = record.p_signal
        if ecg_signal.ndim > 1:
            # choose first channel by default
            ecg = np.asarray(ecg_signal[:,0]).astype(float)
        else:
            ecg = np.asarray(ecg_signal).astype(float)
        fs = getattr(record, "fs", 250)
    except Exception as e:
        st.error(f"Unable to read WFDB record: {e}")
        st.stop()

    st.success("Files loaded successfully!")

    # compute figures
    num_plot = min(len(ecg), 3000)
    fig_ecg, ax = plt.subplots(figsize=(10,3))
    ax.plot(np.arange(num_plot)/fs, ecg[:num_plot], linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.15)
    st.pyplot(fig_ecg)

    # RMS
    window = int(min(1000, max(50, int(fs*0.8))))
    rms_vals = np.sqrt(np.convolve(ecg**2, np.ones(window)/window, mode='valid'))
    t_rms = np.linspace(0, len(ecg)/fs, len(rms_vals))
    fig_rms, axr = plt.subplots(figsize=(10,3))
    axr.plot(t_rms, rms_vals)
    axr.set_xlabel("Time (s)")
    axr.set_ylabel("RMS")
    axr.grid(alpha=0.15)
    st.pyplot(fig_rms)

    # Heart rate
    peaks, _ = find_peaks(ecg, distance=int(fs*0.3))
    hr_text = "Could not estimate HR (not enough peaks)"
    if len(peaks) >= 2:
        rr = np.diff(peaks)/fs
        hr = 60.0/rr
        fig_hr, axh = plt.subplots(figsize=(10,3))
        axh.plot(hr)
        axh.set_xlabel("Beat index")
        axh.set_ylabel("BPM")
        axh.grid(alpha=0.15)
        st.pyplot(fig_hr)
        hr_text = f"Average HR: {np.mean(hr):.1f} BPM"
    st.markdown(f"**{hr_text}**")

    # Spectrogram
    length = min(len(ecg), int(fs*5000))
    f, t_spec, Sxx = spectrogram(ecg[:length], fs=fs, nperseg=256, noverlap=128)
    fig_spec, axs = plt.subplots(figsize=(10,4))
    pcm = axs.pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-12), shading='gouraud')
    axs.set_ylabel("Frequency (Hz)")
    axs.set_xlabel("Time (s)")
    fig_spec.colorbar(pcm, ax=axs, label='Power (dB)')
    st.pyplot(fig_spec)

    # Histogram
    fig_hist, axh2 = plt.subplots(figsize=(6,3))
    axh2.hist(ecg, bins=60, edgecolor='k')
    axh2.set_xlabel("Amplitude")
    axh2.set_ylabel("Count")
    st.pyplot(fig_hist)

    # ROC-like plot (simulated)
    fig_roc, axroc = plt.subplots(figsize=(6,4))
    fpr = np.linspace(0,1,200)
    tpr = np.sqrt(fpr)
    axroc.plot(fpr, tpr, label="Model (simulated)")
    axroc.plot([0,1],[0,1], linestyle='--', color='gray')
    axroc.set_xlabel("False Positive Rate")
    axroc.set_ylabel("True Positive Rate")
    axroc.legend()
    st.pyplot(fig_roc)

    # explanation and micro features
    st.markdown("### Signal micro-features")
    micro = extract_micro_features(ecg, fs=fs)
    micro_names = ["mean","std","min","max","ptp","rms","median","q25","q75","skew","kurtosis"]
    feats = dict(zip(micro_names, [float(x) for x in micro]))
    st.json(feats)

    # Prepare features for model if available
    pdf_figs = {
        "ECG Signal": fig_to_bytes(fig_ecg),
        "RMS Trend": fig_to_bytes(fig_rms),
        "Heart Rate": fig_to_bytes(fig_hr) if 'fig_hr' in locals() else None,
        "Spectrogram": fig_to_bytes(fig_spec),
        "Histogram": fig_to_bytes(fig_hist),
        "ROC Curve": fig_to_bytes(fig_roc)
    }

    # ---------- Model prediction if artifacts exist ----------
    can_predict = (model is not None) and (scaler is not None)
    prediction_result = None
    if can_predict:
        # Build feature vector that matches expected columns if possible
        # Heuristic: if feature_columns exist and length matches, try to build vector
        base_row = ecg if hasattr(ecg, 'shape') else np.asarray(ecg)
        micro = extract_micro_features(base_row, fs=fs)
        # combine base signal summary (downsampled) + micro features OR fallback to micro only
        # To avoid huge vectors, we will use micro features only unless feature_columns expects more.
        prepared = micro.copy().reshape(1,-1)
        # if feature_columns exists and expects larger vector, try to pad/truncate
        if feature_columns:
            expected_len = len(feature_columns)
            if expected_len == prepared.shape[1]:
                X_to_model = prepared
                used_preset = "micro-only (matched)"
            elif expected_len > prepared.shape[1]:
                # pad with zeros
                pad = np.zeros(expected_len - prepared.shape[1])
                X_to_model = np.hstack([prepared.flatten(), pad]).reshape(1,-1)
                used_preset = f"micro + zeros (padded to {expected_len})"
            else:
                # truncate if model expects fewer features (rare)
                X_to_model = prepared.flatten()[:expected_len].reshape(1,-1)
                used_preset = f"micro truncated to {expected_len}"
        else:
            X_to_model = prepared
            used_preset = "micro-only (no feature_columns.json)"

        st.info(f"Preparing features for prediction using strategy: {used_preset}")

        # Apply imputer & scaler safely (handle shape mismatches)
        X_proc = X_to_model.copy()
        try:
            if imputer is not None:
                # check compatibility
                try:
                    X_proc = imputer.transform(X_proc)
                except Exception as e_im:
                    st.warning(f"Imputer transform failed ({e_im}). Will try fitting a local imputer on current features.")
                    if SimpleImputer is not None:
                        local_imp = SimpleImputer(strategy='median')
                        X_proc = local_imp.fit_transform(X_proc)
                    else:
                        st.warning("SimpleImputer not available in runtime; continuing without imputation.")
            if scaler is not None:
                try:
                    X_proc = scaler.transform(X_proc)
                except Exception as e_sc:
                    st.warning(f"Scaler transform failed ({e_sc}). Will try fitting a local scaler on current features.")
                    if StandardScaler is not None:
                        local_sc = StandardScaler()
                        X_proc = local_sc.fit_transform(X_proc)
                    else:
                        st.warning("StandardScaler not available; continuing without scaling.")
            # predict
            pred = model.predict(X_proc)
            # if label encoder exists, try to decode
            label = None
            if label_encoder is not None:
                try:
                    label = label_encoder.inverse_transform(pred)
                    label = label[0] if isinstance(label, (list, np.ndarray)) else str(label)
                except Exception:
                    label = str(int(pred[0]))
            else:
                # basic mapping for standard 0..7 labels (adjust if your model uses different mapping)
                label_map = {
                    0: "Normal ECG",
                    1: "Arrhythmia",
                    2: "Tachycardia",
                    3: "Bradycardia",
                    4: "Myocardial Infarction",
                    5: "Bundle Branch Block",
                    6: "Hypertrophy",
                    7: "Pericarditis / Electrolyte imbalance"
                }
                label = label_map.get(int(pred[0]), f"Class {int(pred[0])}")
            prediction_result = {"pred_class": int(pred[0]), "label": label}
            st.success(f"Model prediction: {prediction_result['label']} (class {prediction_result['pred_class']})")
        except Exception as e_all:
            st.error(f"Model prediction failed: {e_all}")
            can_predict = False
    else:
        st.warning("Model artifacts not found or incomplete — analysis only (no prediction).")

    # ---------- Download PDF report ----------
    st.markdown("### Generate PDF report")
    if st.button("Generate & Download PDF report"):
        if not PIL_AVAILABLE:
            st.error("PDF generation requires Pillow + reportlab. Install them in your environment.")
        else:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30,leftMargin=30, topMargin=30,bottomMargin=18)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('TitleCenter', parent=styles['Title'], alignment=1, fontSize=18, textColor=colors.HexColor("#1E90FF"))
            normal = styles["Normal"]
            story = []
            story.append(Spacer(1,20))
            story.append(Paragraph("🩺 <b>Cardiac Pre-Stroke - Report</b>", title_style))
            story.append(Spacer(1,8))
            # cover heart image
            heart_buf = make_heart_png(width=420, height=220, fill_color="#eef6ff")
            if heart_buf:
                try:
                    img = RLImage(heart_buf, width=420, height=220)
                    story.append(img)
                    story.append(Spacer(1,12))
                except Exception:
                    story.append(Paragraph("(Heart image not available)", normal))
            # quick summary
            story.append(Paragraph(f"Record: {hea_file.name}", normal))
            if prediction_result:
                story.append(Paragraph(f"Model Prediction: {prediction_result['label']} (class {prediction_result['pred_class']})", normal))
            else:
                story.append(Paragraph("Model Prediction: N/A (artifacts missing)", normal))
            story.append(Spacer(1,10))
            # attach figures
            for name, b in pdf_figs.items():
                if b is None:
                    continue
                story.append(Paragraph(f"<b>{name}</b>", styles["Heading3"]))
                try:
                    b.seek(0)
                    img = RLImage(b, width=450, height=250)
                    story.append(img)
                except Exception:
                    story.append(Paragraph("(Figure could not be embedded)", normal))
                story.append(Spacer(1,8))
            story.append(PageBreak())
            doc.build(story)
            buffer.seek(0)
            st.download_button("⬇️ Download PDF", data=buffer.getvalue(), file_name="Cardiac_PreStroke_Report.pdf", mime="application/pdf")
else:
    st.info("Upload both .hea and .dat files to start. If you already uploaded files earlier but they expired, please upload again.")
