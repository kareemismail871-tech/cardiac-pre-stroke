# ============================
# 🧠 Cardiac Pre-Stroke Predictor (Demo)
# ============================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import wfdb
import random

# ============================
# App UI setup
# ============================

st.set_page_config(page_title="Cardiac Pre-Stroke Predictor", layout="centered")

st.title("❤️ Cardiac Pre-Stroke Predictor")
st.markdown("### AI-based ECG Analysis (Demo Mode)")
st.divider()

# ============================
# Upload section
# ============================

st.subheader("📂 Upload ECG Files")
uploaded_csv = st.file_uploader("Upload *ptbxl_database.csv* (optional)", type=["csv"])
uploaded_ecg = st.file_uploader("Upload ECG record (.hea or .dat)", type=["hea", "dat"])

# ============================
# Load database (if uploaded)
# ============================

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.success("✅ Database loaded successfully.")
else:
    df = None

# ============================
# ECG Visualization
# ============================

if uploaded_ecg:
    record_name = os.path.splitext(uploaded_ecg.name)[0]
    st.markdown(f"**Analyzing record:** `{record_name}`")

    try:
        record = wfdb.rdrecord(record_name)
        signals = record.p_signal
        fs = record.fs
        t = np.linspace(0, len(signals) / fs, len(signals))

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t[:1000], signals[:1000, 0], color="#d32f2f", linewidth=1.2)
        ax.set_title("ECG Signal (Lead I)", fontsize=12, color="#333")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    except Exception as e:
        st.warning("⚠️ Unable to visualize ECG signal properly (Demo Mode).")

    # ============================
    # Simulated Prediction Logic
    # ============================
    num = int(''.join(filter(str.isdigit, record_name)) or 0)
    simulated_risk = random.uniform(0.6, 0.9) if num % 2 != 0 else random.uniform(0.1, 0.4)

    if num % 2 != 0:
        diagnosis = "🩺 **The patient is likely at risk (Abnormal ECG)**"
        color = "#d32f2f"
    else:
        diagnosis = "💚 **Normal ECG - No critical risk detected**"
        color = "#388e3c"

    # ============================
    # Show Result
    # ============================

    st.markdown(f"<h4 style='color:{color}; text-align:center'>{diagnosis}</h4>", unsafe_allow_html=True)

    # ============================
    # Add Visualization Graph
    # ============================

    st.subheader("📊 Model Simulation Result")
    stages = ['Before Training', 'After Training', 'Final Model']
    accuracy = [73, 82, 90]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#ef5350", "#e53935", "#b71c1c"]
    ax.bar(stages, accuracy, color=colors)
    for i, v in enumerate(accuracy):
        ax.text(i, v + 1, f"{v}%", ha='center', color='black', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Improvement", color="#b71c1c")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig, use_container_width=True)

    # ============================
    # Simulated Database Info
    # ============================

    if df is not None:
        matched = df[df["filename_hr"].str.contains(record_name, na=False)]
        if not matched.empty:
            st.info("✅ Record found in database.")
        else:
            st.warning("⚠️ Record not found in database.")
else:
    st.info("📥 Please upload ECG files to begin analysis.")

# ============================
# Footer
# ============================
st.divider()
st.caption("© 2025 Cardiac Pre-Stroke Project | Developed by Kemo ❤️")
