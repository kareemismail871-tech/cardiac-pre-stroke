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
seed = st.sidebar.number_input("Random seed (reproducible)", value=42, step=1)
randomness = st.sidebar.slider("Variability range", 0.01, 0.40, 0.18, 0.01)
borderline_chance = st.sidebar.slider("Chance (percent) that even record is borderline", 0, 40, 10, 1)
st.sidebar.markdown("---")
st.sidebar.markdown("**Important:** Simulation mode must be clearly labeled when used in demos. Do not use simulated output for clinical decisions.")

# reproducible randomness
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
    Returns: prob (0..1), label_str, textual_message, severity_tag
    """
    if nid is None:
        base = random.uniform(0.3, 0.6)
        prob = np.clip(base + random.uniform(-variability, variability), 0.0, 0.99)
    else:
        if nid % 2 == 1:
            base = random.uniform(0.60, 0.92)
            prob = np.clip(base + random.uniform(-variability, variability), 0.0, 0.99)
        else:
            if random.uniform(0,100) < borderline_pct:
                base = random.uniform(0.35, 0.55)
            else:
                base = random.uniform(0.02, 0.32)
            prob = np.clip(base + random.uniform(-variability, variability), 0.0, 0.99)

    if prob >= 0.60:
        label = "Patient (SIMULATED)"
        msg = f"Simulated high risk — estimated: {prob*100:.1f}%"
        severity = "high"
    elif prob >= 0.35:
        label = "Borderline (SIMULATED)"
        msg = f"Simulated borderline — estimated: {prob*100:.1f}%"
        severity = "medium"
    else:
        label = "Not Patient (SIMULATED)"
        msg = f"Simulated normal — estimated risk: {prob*100:.1f}%"
        severity = "low"

    return prob, label, msg, severity

def make_probability_bar_png(prob, severity, width=600, height=80):
    """Create a horizontal probability bar as PNG bytes"""
    fig, ax = plt.subplots(figsize=(6,1.2))
    ax.barh([0], [prob], color="#ff6b6b" if severity!="low" else "#6cc070", height=0.6)
    ax.set_xlim(0,1)
    ax.set_yticks([])
    ax.set_xlabel("Risk probability")
    ax.set_title("Predicted Risk")
    txtx = prob + 0.02 if prob < 0.88 else prob - 0.12
    ax.text(txtx, 0, f"{prob*100:.1f}%", va='center', fontsize=10, fontweight='bold', color='black')
    for spine in ax.spines.values():
        spine.set_visible(False)
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ----------------------------
# Main processing when files uploaded
# ----------------------------
if hea_file and dat_file:
    tmp = hea_file.name.replace(".hea", "")
    # save uploaded files locally to read with wfdb
    with open(hea_file.name, "wb") as f:
        f.write(hea_file.read())
    with open(dat_file.name, "wb") as f:
        f.write(dat_file.read())

    st.write(f"Record: `{tmp}`")

    # attempt to read record for visualization
    try:
        rec = rdrecord(tmp)
        sig = rec.p_signal
        # take first channel for display
        if sig.ndim > 1:
            y = sig[:,0]
        else:
            y = sig
        st.markdown("**ECG Waveform (first 2000 samples)**")
        fig_ecg, ax_ecg = plt.subplots(figsize=(8,2))
        ax_ecg.plot(y[:2000], linewidth=0.8)
        ax_ecg.set_xlim(0, min(2000, len(y)))
        ax_ecg.set_ylabel("Amplitude")
        ax_ecg.set_xlabel("Samples")
        ax_ecg.set_title("ECG (channel 0)")
        st.pyplot(fig_ecg)
        plt.close(fig_ecg)

        # histogram
        fig_hist, ax_hist = plt.subplots(figsize=(6,2))
        ax_hist.hist(y, bins=60, alpha=0.9)
        ax_hist.set_title("Signal Amplitude Distribution")
        ax_hist.set_xlabel("Amplitude")
        ax_hist.set_ylabel("Count")
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        # mini-sparkline: rolling RMS
        window = min(500, len(y))
        if len(y) > 50:
            rms = np.sqrt(pd.Series(y).rolling(window=50).mean().fillna(method='bfill').values)
            fig_s, ax_s = plt.subplots(figsize=(6,1.2))
            ax_s.plot(rms[-200:], linewidth=0.9)
            ax_s.set_title("RMS (mini)")
            ax_s.set_yticks([])
            ax_s.set_xticks([])
            st.pyplot(fig_s)
            plt.close(fig_s)
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
        # prominent simulated banner
        st.error("⚠️ DEMO MODE: This result is SIMULATED — NOT a real clinical diagnosis.")
        st.markdown(f"**Simulated note:** {message}")

        # Build table
        out_df = pd.DataFrame({
            "Record": [tmp],
            "PTB-XL True Label": [true_label_text],
            "Simulated Label": [label],
            "Simulated Probability": [f"{prob*100:.1f}%"],
            "Note": ["SIMULATED - NOT CLINICAL"]
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

        # improved probability bar + download
        st.markdown("**Simulated Risk Gauge**")
        prob_img_bytes = make_probability_bar_png(prob, severity)
        st.image(prob_img_bytes, use_column_width=True)

        # allow download of bar PNG
        st.download_button("Download probability chart (PNG)", prob_img_bytes, file_name=f"{tmp}_sim_prob.png", mime="image/png")

        # allow download of results CSV
        csv_buf = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download simulated report (CSV)", csv_buf, file_name=f"{tmp}_sim_report.csv", mime="text/csv")

    else:
        st.info("Demo mode is OFF. Real model inference path not included in this demo snippet. Upload real model artifacts and enable inference if available.")

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
