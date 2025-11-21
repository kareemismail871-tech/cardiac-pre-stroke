# app.py
import os
import io
import json
import random
import re
from io import BytesIO

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, spectrogram
from scipy.stats import skew, kurtosis
import wfdb

# Optional ML libs (only used if artifacts exist)
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

# ---------- Utilities ----------
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

# ---------- Artifact paths (adjust if needed) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ecg_stack_model_pro.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "ecg_scaler_pro.joblib")
IMPUTER_PATH = os.path.join(BASE_DIR, "ecg_imputer_pro.joblib")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "feature_columns.json")
LABEL_ENC_PATH = os.path.join(BASE_DIR, "label_encoder.joblib")

# ---------- Try to load artifacts ----------
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

# ---------- Upload UI ----------
st.markdown("Upload a pair of WFDB files: `.hea` and `.dat`. If model artifacts are available the app will attempt a prediction.")
col1, col2 = st.columns(2)
with col1:
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])
with col2:
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])

# ---------- Main behavior when files are uploaded ----------
if hea_file and dat_file:
    # Save temporarily and load via wfdb
    try:
        record_name = hea_file.name.replace('.hea','')
        with open(hea_file.name, "wb") as f:
            f.write(hea_file.read())
        with open(dat_file.name, "wb") as f:
            f.write(dat_file.read())
        record = wfdb.rdrecord(record_name)
        ecg_signal = record.p_signal
        if ecg_signal.ndim > 1:
            ecg = np.asarray(ecg_signal[:,0]).astype(float)
        else:
            ecg = np.asarray(ecg_signal).astype(float)
        fs = getattr(record, "fs", 250)
    except Exception as e:
        st.error(f"Unable to read WFDB record: {e}")
        st.stop()

    st.success("Files loaded successfully!")

    # ---- ECG plot ----
    nplot = min(len(ecg), 3000)
    fig_ecg, ax = plt.subplots(figsize=(10,3))
    ax.plot(np.arange(nplot)/fs, ecg[:nplot], linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.15)
    st.pyplot(fig_ecg)

    # ---- RMS ----
    window = int(min(1000, max(50, int(fs*0.8))))
    rms_vals = np.sqrt(np.convolve(ecg**2, np.ones(window)/window, mode='valid'))
    t_rms = np.linspace(0, len(ecg)/fs, len(rms_vals))
    fig_rms, axr = plt.subplots(figsize=(10,3))
    axr.plot(t_rms, rms_vals)
    axr.set_xlabel("Time (s)")
    axr.set_ylabel("RMS")
    axr.grid(alpha=0.15)
    st.pyplot(fig_rms)

    # ---- Heart-rate ----
    peaks, _ = find_peaks(ecg, distance=int(fs*0.3))
    hr_text = "Could not estimate HR (not enough peaks)"
    fig_hr = None
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

    # ---- Spectrogram ----
    length = min(len(ecg), int(fs*5000))
    f, t_spec, Sxx = spectrogram(ecg[:length], fs=fs, nperseg=256, noverlap=128)
    fig_spec, axs = plt.subplots(figsize=(10,4))
    pcm = axs.pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-12), shading='gouraud')
    axs.set_ylabel("Frequency (Hz)")
    axs.set_xlabel("Time (s)")
    fig_spec.colorbar(pcm, ax=axs, label='Power (dB)')
    st.pyplot(fig_spec)

    # ---- Histogram ----
    fig_hist, axh2 = plt.subplots(figsize=(6,3))
    axh2.hist(ecg, bins=60, edgecolor='k')
    axh2.set_xlabel("Amplitude")
    axh2.set_ylabel("Count")
    st.pyplot(fig_hist)

    # ---- ROC-like (simulated) ----
    fig_roc, axroc = plt.subplots(figsize=(6,4))
    fpr = np.linspace(0,1,200)
    tpr = np.sqrt(fpr)
    axroc.plot(fpr, tpr, label="Model (simulated)")
    axroc.plot([0,1],[0,1], linestyle='--', color='gray')
    axroc.set_xlabel("False Positive Rate")
    axroc.set_ylabel("True Positive Rate")
    axroc.legend()
    st.pyplot(fig_roc)

    # ---- micro-features ----
    st.markdown("### Signal micro-features")
    micro = extract_micro_features(ecg, fs=fs)
    micro_names = ["mean","std","min","max","ptp","rms","median","q25","q75","skew","kurtosis"]
    feats = dict(zip(micro_names, [float(x) for x in micro]))
    st.json(feats)

    # Prepare figures for PDF
    pdf_figs = {
        "ECG Signal": fig_to_bytes(fig_ecg),
        "RMS Trend": fig_to_bytes(fig_rms),
        "Heart Rate": fig_to_bytes(fig_hr) if fig_hr is not None else None,
        "Spectrogram": fig_to_bytes(fig_spec),
        "Histogram": fig_to_bytes(fig_hist),
        "ROC Curve": fig_to_bytes(fig_roc)
    }

    # ---------- Prediction / Risk estimation ----------
    can_predict = (model is not None) and (scaler is not None)
    prediction_result = None
    risk_percent = None
    health_state_text = None

    if can_predict:
        st.info("Model artifacts detected — attempting prediction.")
        base_row = ecg
        micro = extract_micro_features(base_row, fs=fs)
        prepared = micro.copy().reshape(1,-1)

        # Build X_to_model according to feature_columns if provided
        if feature_columns:
            expected_len = len(feature_columns)
            if expected_len == prepared.shape[1]:
                X_to_model = prepared
                used_preset = "micro-only (matched)"
            elif expected_len > prepared.shape[1]:
                pad = np.zeros(expected_len - prepared.shape[1])
                X_to_model = np.hstack([prepared.flatten(), pad]).reshape(1,-1)
                used_preset = f"micro + zeros (padded to {expected_len})"
            else:
                X_to_model = prepared.flatten()[:expected_len].reshape(1,-1)
                used_preset = f"micro truncated to {expected_len}"
        else:
            X_to_model = prepared
            used_preset = "micro-only (no feature_columns.json)"

        st.info(f"Feature preparation strategy: {used_preset}")

        X_proc = X_to_model.copy()
        # Imputer
        try:
            if imputer is not None:
                X_proc = imputer.transform(X_proc)
            else:
                # try local imputer
                if SimpleImputer is not None:
                    local_imp = SimpleImputer(strategy='median')
                    X_proc = local_imp.fit_transform(X_proc)
        except Exception as e_imp:
            st.warning(f"Imputer transform failed ({e_imp}). Using local imputer fallback.")
            if SimpleImputer is not None:
                local_imp = SimpleImputer(strategy='median')
                X_proc = local_imp.fit_transform(X_proc)

        # Scaler
        try:
            if scaler is not None:
                X_proc = scaler.transform(X_proc)
        except Exception as e_sc:
            st.warning(f"Scaler transform failed ({e_sc}). Using local StandardScaler fallback.")
            if StandardScaler is not None:
                local_sc = StandardScaler()
                X_proc = local_sc.fit_transform(X_proc)

        # Predict
        try:
            pred = model.predict(X_proc)
            # Try predict_proba for risk%
            prob = None
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_proc)
                    # If multiclass: compute "abnormal" probability as 1 - prob(normal) if 0 is normal
                    if proba.shape[1] >= 2:
                        # heuristics: if label 0 = Normal
                        if 0 < proba.shape[1]:
                            prob_normal = proba[0,0]
                            prob_abnormal = 1.0 - prob_normal
                            prob = prob_abnormal * 100.0
                        else:
                            prob = np.max(proba)*100.0
                    else:
                        prob = (1.0 - proba[0,0])*100.0
            except Exception as e_pb:
                st.warning(f"predict_proba failed: {e_pb}")
                prob = None

            label = None
            if label_encoder is not None:
                try:
                    label = label_encoder.inverse_transform(pred)
                    label = label[0] if isinstance(label, (list, np.ndarray)) else str(label)
                except Exception:
                    label = str(int(pred[0]))
            else:
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

            # If no prob from model, derive simple heuristic risk %
            if prob is None:
                # Heuristic: use RMS & ptp to infer an anomaly score (not clinical)
                rms_val = float(np.sqrt(np.mean(ecg**2)))
                ptp = float(np.ptp(ecg))
                # normalize with arbitrary ranges observed in dataset (tunable)
                rms_score = min(1.0, (rms_val / (np.percentile(ecg, 90) - np.percentile(ecg,10)+1e-6)) )
                ptp_score = min(1.0, ptp / (np.std(ecg)*6 + 1e-6))
                est_score = np.clip(0.3*rms_score + 0.7*ptp_score, 0.0, 1.0)
                prob = float(est_score * 100.0)
                st.info("Using heuristic risk estimation (model probabilities not available).")

            risk_percent = float(np.clip(prob, 0.0, 100.0))

            # Health state text
            if "Normal" in label or prediction_result['pred_class'] == 0:
                health_state_text = ("Likely Normal", "من المحتمل أن الإشارة طبيعية")
            else:
                health_state_text = ("Possible Abnormality", "محتمل وجود حالة غير طبيعية — راجع طبيباً")

        except Exception as e_pred:
            st.error(f"Model prediction failed: {e_pred}")
            can_predict = False

    else:
        st.warning("Model artifacts not available — showing analysis only (no ML prediction).")
        # estimate an anomaly score from RMS/ptp for display
        rms_val = float(np.sqrt(np.mean(ecg**2)))
        ptp = float(np.ptp(ecg))
        # simple normalization (not clinical)
        est = np.clip((ptp / (np.std(ecg)*6 + 1e-6)), 0.0, 1.0)
        risk_percent = float(est*100.0)
        health_state_text = ("Analysis only", "تحليل فقط — الموديل غير متوفر")

    # ---------- Show diagnosis block with risk bar ----------
    st.markdown("## Quick Diagnosis")
    colL, colR = st.columns([2,1])
    with colL:
        if prediction_result:
            st.markdown(f"**Prediction:** {prediction_result['label']}  ")
            st.markdown(f"**Class:** {prediction_result['pred_class']}")
        else:
            st.markdown("**Prediction:** N/A")
        # bilingual health state
        if isinstance(health_state_text, tuple):
            st.markdown(f"**Status:** {health_state_text[0]} — **حالة:** {health_state_text[1]}")
        else:
            st.markdown(f"**Status:** {health_state_text}")

        # textual guidance
        if prediction_result and (prediction_result['pred_class'] != 0):
            st.markdown("**Recommendation:** Seek medical evaluation. This is an AI screening not a diagnosis. | التوصية: راجع الطبيب فورًا إن أمكن.")
        elif prediction_result and (prediction_result['pred_class'] == 0):
            st.markdown("**Recommendation:** Low immediate risk. Continue routine follow-up. | التوصية: متابعة دورية.")
        else:
            st.markdown("**Recommendation:** Based on signal analysis only. Consider clinical follow-up if concerned. | التوصية: إن كنت قلقًا راجع طبيبًا.")

    with colR:
        # Risk bar (horizontal)
        rp = risk_percent if risk_percent is not None else 0.0
        fig_bar, ax_bar = plt.subplots(figsize=(4,0.9))
        ax_bar.barh([0], [rp], height=0.6, color='#ff4c4c' if rp>50 else '#2ecc71')
        ax_bar.set_xlim(0,100)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([0,25,50,75,100])
        for spine in ax_bar.spines.values(): spine.set_visible(False)
        ax_bar.text(rp + (-8 if rp > 90 else 2), 0, f"{rp:.1f}%", va='center', fontweight='bold', color='white', bbox=dict(facecolor=('#ff4c4c' if rp>50 else '#2ecc71'), boxstyle='round,pad=0.2'))
        fig_bar.patch.set_alpha(0)
        st.pyplot(fig_bar)

    # ----------
    # PDF report generation UI
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
            story.append(Paragraph("🩺 Cardiac Pre-Stroke - Report", title_style))
            story.append(Spacer(1,8))
            heart_buf = make_heart_png(width=420, height=220, fill_color="#eef6ff")
            if heart_buf:
                try:
                    img = RLImage(heart_buf, width=420, height=220)
                    story.append(img)
                    story.append(Spacer(1,12))
                except Exception:
                    story.append(Paragraph("(Heart image not available)", normal))
            story.append(Paragraph(f"Record: {hea_file.name}", normal))
            if prediction_result:
                story.append(Paragraph(f"Model Prediction: {prediction_result['label']} (class {prediction_result['pred_class']})", normal))
            else:
                story.append(Paragraph("Model Prediction: N/A (artifacts missing)", normal))
            story.append(Paragraph(f"Risk estimate: {risk_percent:.1f}%", normal))
            story.append(Spacer(1,10))
            for name, b in pdf_figs.items():
                if b is None: continue
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
    st.info("Upload both .hea and .dat files to start. If previous uploads expired, please upload again.")
