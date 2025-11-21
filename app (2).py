import streamlit as st
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
import os

# ==============================
# Load model files if they exist
# ==============================
MODEL_FILES = {
    "model": "ecg_stack_model_pro.joblib",
    "scaler": "ecg_scaler_pro.joblib",
    "imputer": "ecg_imputer_pro.joblib",
    "features": "feature_columns.json",
    "labels": "label_encoder.joblib"
}

AVAILABLE_MODEL = all(os.path.exists(f) for f in MODEL_FILES.values())

if AVAILABLE_MODEL:
    model = joblib.load(MODEL_FILES["model"])
    scaler = joblib.load(MODEL_FILES["scaler"])
    imputer = joblib.load(MODEL_FILES["imputer"])
    with open(MODEL_FILES["features"], "r") as f:
        feature_columns = json.load(f)
    label_encoder = joblib.load(MODEL_FILES["labels"])

# ==================================
# Micro features function
# ==================================
def micro_features(sig):
    return [
        np.mean(sig), np.std(sig),
        np.min(sig), np.max(sig),
        np.ptp(sig), np.sqrt(np.mean(sig**2)),
        np.median(sig), np.percentile(sig,25),
        np.percentile(sig,75),
        float(np.mean((sig - np.mean(sig))**3)),
        float(np.mean((sig - np.mean(sig))**4)),
    ]

# ==================================
# ECG Analysis Functions
# ==================================

def plot_signal(sig):
    fig, ax = plt.subplots()
    ax.plot(sig)
    ax.set_title("ECG Signal")
    return fig

def plot_hist(sig):
    fig, ax = plt.subplots()
    ax.hist(sig, bins=50)
    ax.set_title("Signal Distribution")
    return fig

def plot_rms(sig):
    window = 200
    rms = np.sqrt(np.convolve(sig**2, np.ones(window)/window, mode='valid'))
    fig, ax = plt.subplots()
    ax.plot(rms)
    ax.set_title("RMS Trend")
    return fig

def spectrogram(sig):
    fig, ax = plt.subplots()
    ax.specgram(sig, Fs=500)
    ax.set_title("Spectrogram")
    return fig

# ==================================
# STREAMLIT UI
# ==================================
st.title("❤️ Cardiac Pre-Stroke AI Analyzer")

st.write("Upload your **.hea** and **.dat** files to analyze your ECG.")

hea_file = st.file_uploader("Upload .hea file", type=["hea"])
dat_file = st.file_uploader("Upload .dat file", type=["dat"])

if hea_file and dat_file:
    # Save uploaded files
    with open("temp.hea", "wb") as f:
        f.write(hea_file.read())
    with open("temp.dat", "wb") as f:
        f.write(dat_file.read())

    # Read ECG using wfdb
    record = wfdb.rdrecord("temp")
    sig = record.p_signal[:,0]

    st.subheader("📈 ECG Signal")
    st.pyplot(plot_signal(sig))

    st.subheader("📊 Histogram")
    st.pyplot(plot_hist(sig))

    st.subheader("📈 RMS Curve")
    st.pyplot(plot_rms(sig))

    st.subheader("🎵 Spectrogram")
    st.pyplot(spectrogram(sig))

    # ==================================
    # Prediction (if model exists)
    # ==================================
    if AVAILABLE_MODEL:
        st.subheader("🧠 AI Diagnosis")

        features = np.concatenate([sig[:len(feature_columns)-10], micro_features(sig)])
        features = features.reshape(1, -1)

        features = imputer.transform(features)
        features = scaler.transform(features)

        pred = model.predict(features)[0]
        label = label_encoder.inverse_transform([pred])[0]

        st.success(f"💓 Prediction: **{label}**")

        # Risk level
        if pred in [0]:
            st.info("Risk Level: LOW")
        elif pred in [1,2,3]:
            st.warning("Risk Level: MEDIUM")
        else:
            st.error("⚠️ Risk Level: HIGH — seek medical attention.")

    else:
        st.warning("⚠️ Model files not found. Showing analysis only.")

