# app.py
import streamlit as st
import numpy as np
import pandas as pd
import random
import ast
import re
import os
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🫀", layout="centered")
st.title("💙 Cardiac Pre-Stroke Predictor")
st.caption("Simulated version for demo and visualization — not a real diagnosis.")

# ----------------------------
# Sidebar: Demo controls
# ----------------------------
st.sidebar.header("Simulation Controls")
demo_mode = st.sidebar.checkbox("Enable Simulation", value=True)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
randomness = st.sidebar.slider("Variability", 0.01, 0.40, 0.18, 0.01)
borderline_chance = st.sidebar.slider("Borderline Chance (%)", 0, 40, 10, 1)

random.seed(int(seed))
np.random.seed(int(seed))

# ----------------------------
# Upload PTB-XL metadata (optional)
# ----------------------------
st.markdown("### Upload PTB-XL Metadata (Optional)")
ptbxl_file = st.file_uploader("Upload ptbxl_database.csv", type=["csv"])
ptbxl_df = None
if ptbxl_file is not None:
    try:
        ptbxl_df = pd.read_csv(ptbxl_file)
        st.success(f"✅ Metadata loaded ({len(ptbxl_df)} rows).")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# ----------------------------
# Upload ECG files
# ----------------------------
st.markdown("### Upload ECG Files (.hea + .dat)")
hea_file = st.file_uploader("Upload .hea file", type=["hea"])
dat_file = st.file_uploader("Upload .dat file", type=["dat"])

def extract_numeric_id(name):
    """Extract last continuous digits from filename"""
    match = re.search(r'(\d+)(?!.*\d)', name)
    return int(match.group(1)) if match else None

def simulate_result(nid, variability=0.18, borderline_pct=10):
    """Simulate probability and result"""
    if nid is None:
        base = random.uniform(0.4, 0.6)
    elif nid % 2 == 1:  # Odd → Sick
        base = random.uniform(0.65, 0.92)
    else:  # Even → Healthy or borderline
        base = random.uniform(0.05, 0.55 if random.uniform(0,100) < borderline_pct else 0.35)
    prob = np.clip(base + random.uniform(-variability, variability), 0.0, 0.99)

    if prob >= 0.60:
        return prob, "Patient", "❤️ The patient may be at risk.", "high"
    elif prob >= 0.35:
        return prob, "Borderline", "⚠️ Borderline case — requires monitoring.", "medium"
    else:
        return prob, "Not Patient", "💚 The patient appears healthy.", "low"

def make_probability_bar(prob, severity):
    """Draw probability bar"""
    fig, ax = plt.subplots(figsize=(6,1.2))
    colors = {"high":"#ff4d4d","medium":"#ffd166","low":"#4caf50"}
    ax.barh(["Risk"], [prob], color=colors[severity], height=0.6)
    ax.set_xlim(0,1)
    ax.set_yticks([])
    ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_xlabel("Risk Level")
    ax.text(prob, 0, f"{prob*100:.1f}%", va='center', fontsize=10, fontweight='bold', color='black')
    for spine in ax.spines.values():
        spine.set_visible(False)
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ----------------------------
# Main logic
# ----------------------------
if hea_file and dat_file:
    tmp_name = hea_file.name.replace(".hea", "")
    with open(hea_file.name, "wb") as f:
        f.write(hea_file.read())
    with open(dat_file.name, "wb") as f:
        f.write(dat_file.read())

    st.markdown(f"**Record:** `{tmp_name}`")

    # ECG Visualization
    try:
        rec = rdrecord(tmp_name)
        y = rec.p_signal[:,0] if rec.p_signal.ndim > 1 else rec.p_signal
        fig, ax = plt.subplots(figsize=(8,2))
        ax.plot(y[:2000], linewidth=0.8)
        ax.set_title("ECG Signal (first 2000 samples)")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.info(f"ECG preview unavailable: {e}")

    # True label lookup (if CSV provided)
    true_label = "Unknown"
    if ptbxl_df is not None:
        matched = ptbxl_df[ptbxl_df["filename_hr"].astype(str).str.contains(tmp_name, na=False)]
        if not matched.empty:
            raw_code = matched.iloc[0]["scp_codes"]
            try:
                code_dict = ast.literal_eval(raw_code)
                true_label = list(code_dict.keys())[0]
            except:
                true_label = str(raw_code)
            st.write(f"**Database label:** `{true_label}`")

    # Simulated output
    if demo_mode:
        nid = extract_numeric_id(tmp_name)
        prob, label, message, severity = simulate_result(nid, variability=randomness, borderline_pct=borderline_chance)

        # Result card
        color_map = {"high":"#ffe5e5","medium":"#fff8db","low":"#e8ffe8"}
        st.markdown(f"""
        <div style='background:{color_map[severity]};padding:14px;border-radius:10px;text-align:center;font-size:16px'>
            <b>{label}</b><br>{message}<br><br><b>Risk Level:</b> {prob*100:.1f}%
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        bar_img = make_probability_bar(prob, severity)
        st.image(bar_img, use_container_width=True)

else:
    st.info("Please upload both `.hea` and `.dat` files to continue.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("This page shows simulated predictions for demonstration only — not a medical diagnosis.")
