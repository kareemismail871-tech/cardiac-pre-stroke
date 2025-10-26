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

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Cardiac Pre-Stroke - Demo Mode", page_icon="🫀", layout="centered")
st.title("💙 Cardiac Pre-Stroke Predictor — Demo/Simulation")
st.caption("Demo mode simulates realistic-looking predictions for presentation/testing. SIMULATED results are NOT clinical diagnoses.")

# ----------------------------
# Sidebar: Demo controls
# ----------------------------
st.sidebar.header("Demo / Simulation Controls")
demo_mode = st.sidebar.checkbox("Enable demo simulation (override model)", value=True)
st.sidebar.write("When enabled, odd-numbered records → simulated 'Patient'. Even → simulated 'Not Patient' (with variation).")
seed = st.sidebar.number_input("Random seed (set for reproducible demo)", value=42, step=1)
randomness = st.sidebar.slider("Variability range (how much probabilities can vary)", 0.05, 0.4, 0.18, 0.01)
borderline_chance = st.sidebar.slider("Chance that an even record becomes borderline (%)", 0, 40, 10, 1)
st.sidebar.markdown("---")
st.sidebar.markdown("**Important:** Simulation mode must be clearly labeled when used in demos. Do not use simulated output for clinical decisions.")

# set seed
random.seed(int(seed))
np.random.seed(int(seed))

# ----------------------------
# Optional PTB-XL CSV upload (to show true labels)
# ----------------------------
st.markdown("### Upload optional PTB-XL metadata (ptbxl_database.csv)")
ptbxl_file = st.file_uploader("Upload ptbxl_database.csv (optional)", type=["csv"])
ptbxl_df = None
if ptbxl_file is not None:
    try:
        ptbxl_df = pd.read_csv(ptbxl_file)
        st.success(f"Loaded PTB-XL metadata ({len(ptbxl_df)} rows).")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")

# ----------------------------
# Upload ECG .hea and .dat
# ----------------------------
st.markdown("### Upload ECG files (.hea + .dat)")
hea_file = st.file_uploader("Upload .hea file", type=["hea"])
dat_file = st.file_uploader("Upload .dat file", type=["dat"])

def extract_numeric_id(tmp_name):
    """Extract last continuous group of digits from name, return int or None."""
    m = re.search(r'(\d+)(?!.*\d)', tmp_name)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

def simulate_result_from_id(nid, variability=0.18, borderline_pct=10):
    """
    Produce simulated probability and label depending on numeric id parity.
    - Odd ids => higher risk distribution (0.55 .. 0.95) with noise
    - Even ids => lower risk distribution (0.02 .. 0.45) with chance to be borderline
    Returns: prob (0..1), label_str, textual_message, severity_tag
    """
    # base ranges
    if nid is None:
        # fallback moderate ambiguity
        base = random.uniform(0.3, 0.6)
        prob = np.clip(base + random.uniform(-variability, variability), 0.0, 0.99)
    else:
        if nid % 2 == 1:
            # odd -> sick, but varied
            base = random.uniform(0.60, 0.92)
            prob = np.clip(base + random.uniform(-variability, variability), 0.0, 0.99)
        else:
            # even -> mostly healthy, but sometimes borderline
            if random.uniform(0,100) < borderline_pct:
                # borderline case (e.g., mild)
                base = random.uniform(0.35, 0.55)
            else:
                base = random.uniform(0.02, 0.32)
            prob = np.clip(base + random.uniform(-variability, variability), 0.0, 0.99)

    # produce label and message
    if prob >= 0.60:
        label = "Patient (SIMULATED)"
        msg = random.choice([
            f"🚨 Simulated: High-risk pattern detected. Estimated risk ≈ {prob*100:.1f}%",
            f"❗ Simulated: Significant abnormality indicators. Risk score: {prob*100:.1f}%"
        ])
        severity = "high"
    elif prob >= 0.35:
        label = "Borderline (SIMULATED)"
        msg = random.choice([
            f"⚠️ Simulated: Borderline/observe. Estimated risk ≈ {prob*100:.1f}%",
            f"⚠️ Simulated: Mild abnormal signals — recommend follow-up. Score: {prob*100:.1f}%"
        ])
        severity = "medium"
    else:
        label = "Not Patient (SIMULATED)"
        msg = random.choice([
            f"💚 Simulated: No critical signs detected. Confidence ≈ {(1-prob)*100:.1f}%",
            f"✅ Simulated: Normal ECG-like pattern. Estimated risk ≈ {prob*100:.1f}%"
        ])
        severity = "low"

    return prob, label, msg, severity

# ----------------------------
# Main processing when files uploaded
# ----------------------------
if hea_file and dat_file:
    tmp = hea_file.name.replace(".hea", "")
    # write files locally (needed by wfdb.rdrecord)
    with open(hea_file.name, "wb") as f:
        f.write(hea_file.read())
    with open(dat_file.name, "wb") as f:
        f.write(dat_file.read())

    st.write(f"Record: `{tmp}`")

    # attempt to read record for visualization (safe try)
    try:
        rec = rdrecord(tmp)
        sig = rec.p_signal
        # show first channel if multi-channel
        channel = 0
        if sig.ndim > 1:
            y = sig[:, channel]
        else:
            y = sig
        st.line_chart(y[:2000], height=220)
        st.caption("ECG preview (first 2000 samples)")
    except Exception as e:
        st.info(f"Could not render full ECG preview (wfdb read error): {e}")
        y = None

    # show any matched true label from CSV if available
    true_label_text = "Unknown"
    if ptbxl_df is not None:
        matched = ptbxl_df[
            ptbxl_df["filename_hr"].astype(str).str.contains(tmp, na=False) |
            ptbxl_df["filename_lr"].astype(str).str.contains(tmp, na=False)
        ]
        if not matched.empty:
            raw_code = matched.iloc[0]["scp_codes"]
            try:
                code_dict = ast.literal_eval(raw_code) if isinstance(raw_code, str) else raw_code
                true_label_main = list(code_dict.keys())[0] if len(code_dict) > 0 else "Unknown"
                true_label_text = true_label_main
            except Exception:
                true_label_text = str(raw_code)
            st.info(f"True label from PTB-XL metadata: `{true_label_text}` (raw: {raw_code})")
        else:
            st.info("No matching record in PTB-XL metadata.")

    # Simulation override (demo_mode)
    if demo_mode:
        nid = extract_numeric_id(tmp)
        prob, label, message, severity = simulate_result_from_id(nid, variability=randomness, borderline_pct=borderline_chance)
        # display prominent simulated banner
        st.error("⚠️ DEMO MODE: This result is SIMULATED — NOT a real clinical diagnosis.")
        st.markdown(f"**Simulated decision note:** {message}")
        # Show table
        out_df = pd.DataFrame({
            "Record": [tmp],
            "PTB-XL True Label": [true_label_text],
            "Simulated Label": [label],
            "Simulated Probability": [f"{prob*100:.1f}%"],
            "Note": [ "SIMULATED RESULT - NOT CLINICAL" ]
        })
        st.dataframe(out_df)

        # color card by severity
        if severity == "high":
            st.markdown("<div style='background:#ffdddd;padding:12px;border-radius:8px;text-align:center;'>"
                        "<b>SIMULATED HIGH RISK</b> — For demo only. Recommend further clinical evaluation.</div>",
                        unsafe_allow_html=True)
        elif severity == "medium":
            st.markdown("<div style='background:#fff3cd;padding:12px;border-radius:8px;text-align:center;'>"
                        "<b>SIMULATED BORDERLINE</b> — For demo only. Consider monitoring.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#ddffdd;padding:12px;border-radius:8px;text-align:center;'>"
                        "<b>SIMULATED NORMAL</b> — For demo only.</div>",
                        unsafe_allow_html=True)

        # probability bar
        fig, ax = plt.subplots(figsize=(4,2))
        ax.barh([0], [prob], color="#ff6b6b" if severity!="low" else "#6cc070")
        ax.set_xlim(0,1)
        ax.set_yticks([])
        ax.set_xlabel("Simulated Risk Probability")
        ax.set_title("Simulated Risk")
        ax.text(prob + 0.02 if prob < 0.9 else prob - 0.08, 0, f"{prob*100:.1f}%", va='center')
        st.pyplot(fig)

    else:
        # normal (non-demo) flow: we only display model absence message because model path not included here
        st.info("Demo mode is OFF. Real model inference path not included in this demo snippet.")
        # If you have a real model loaded you can run inference here and show real prob/label.
else:
    st.info("Please upload both a .hea and a .dat file to run the demo simulation.")

# ----------------------------
# Footer / notes
# ----------------------------
st.markdown("---")
st.markdown("""
**Notes:**  
- This page runs a **SIMULATED** decision when Demo Mode is ON — results are synthetic and intended only for presentations/testing.  
- For real diagnosis, use the trained model and clinical evaluation; never rely on simulated outputs.  
""")
