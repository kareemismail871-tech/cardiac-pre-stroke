# app.py
import streamlit as st
import numpy as np
import pandas as pd
import random, re, ast, os, warnings
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

warnings.filterwarnings("ignore")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🫀", layout="centered")
st.title("💙 Cardiac Pre-Stroke Risk Predictor")
st.caption("AI-powered ECG simulation — for demo and visualization only.")

# ----------------------------
# Upload PTB-XL Metadata (optional)
# ----------------------------
st.markdown("### Upload PTB-XL Metadata (Optional)")
ptbxl_file = st.file_uploader("Upload ptbxl_database.csv", type=["csv"])
ptbxl_df = None
if ptbxl_file is not None:
    try:
        ptbxl_df = pd.read_csv(ptbxl_file)
        st.success(f"✅ Metadata loaded ({len(ptbxl_df)} records).")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")

# ----------------------------
# Upload ECG Files
# ----------------------------
st.markdown("### Upload ECG Record (.hea + .dat)")
hea_file = st.file_uploader("Upload .hea file", type=["hea"])
dat_file = st.file_uploader("Upload .dat file", type=["dat"])

# ----------------------------
# Utility Functions
# ----------------------------
def extract_numeric_id(name):
    match = re.search(r'(\d+)(?!.*\d)', name)
    return int(match.group(1)) if match else None

def simulate_result():
    """Randomized demo output: Patient, Borderline, or Normal."""
    choice = random.choice(["high", "medium", "low"])
    if choice == "high":
        prob = random.uniform(0.70, 0.93)
        return prob, "Patient", "❤️ The patient may be at risk.", "high"
    elif choice == "medium":
        prob = random.uniform(0.40, 0.65)
        return prob, "Borderline", "⚠️ Borderline case — monitoring advised.", "medium"
    else:
        prob = random.uniform(0.05, 0.35)
        return prob, "Not Patient", "💚 The patient appears healthy.", "low"

def make_probability_bar(prob, severity):
    fig, ax = plt.subplots(figsize=(6,1.2))
    colors = {"high":"#ff4d4d","medium":"#f4c542","low":"#4caf50"}
    ax.barh(["Risk"], [prob], color=colors[severity], height=0.5)
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
# Main Process
# ----------------------------
if hea_file and dat_file:
    record_name = hea_file.name.replace(".hea", "")
    with open(hea_file.name, "wb") as f: f.write(hea_file.read())
    with open(dat_file.name, "wb") as f: f.write(dat_file.read())

    st.markdown(f"**Record Name:** `{record_name}`")

    try:
        rec = rdrecord(record_name)
        sig = rec.p_signal
        y = sig[:,0] if sig.ndim > 1 else sig

        # ECG waveform
        st.markdown("#### 🩺 ECG Signal (first 2000 samples)")
        fig1, ax1 = plt.subplots(figsize=(8,2.2))
        ax1.plot(y[:2000], color="#1565c0", linewidth=1)
        ax1.set_xlim(0, min(2000, len(y)))
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Samples")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

        # Histogram
        st.markdown("#### 📊 Amplitude Distribution")
        fig2, ax2 = plt.subplots(figsize=(6,2))
        ax2.hist(y, bins=60, color="#64b5f6", alpha=0.9)
        ax2.set_xlabel("Amplitude")
        ax2.set_ylabel("Count")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)
        plt.close(fig2)

        # RMS Sparkline
        st.markdown("#### ⚡ Signal RMS Trend")
        if len(y) > 50:
            rms = np.sqrt(pd.Series(y).rolling(window=60).mean().fillna(method='bfill').values)
            fig3, ax3 = plt.subplots(figsize=(6,1.2))
            ax3.plot(rms[-200:], color="#0d47a1", linewidth=0.9)
            ax3.set_yticks([])
            ax3.set_xticks([])
            st.pyplot(fig3)
            plt.close(fig3)
    except Exception as e:
        st.warning(f"Unable to render ECG: {e}")
        y = None

    # Metadata lookup
    true_label = "Unknown"
    if ptbxl_df is not None:
        matched = ptbxl_df[ptbxl_df["filename_hr"].astype(str).str.contains(record_name, na=False)]
        if not matched.empty:
            raw_code = matched.iloc[0]["scp_codes"]
            try:
                code_dict = ast.literal_eval(raw_code)
                true_label = list(code_dict.keys())[0]
            except:
                true_label = str(raw_code)
            st.markdown(f"**🧾 Database Label:** `{true_label}`")

    # Simulation result
    prob, label, msg, severity = simulate_result()

    color_bg = {"high":"#ffe5e5","medium":"#fff8db","low":"#e8ffe8"}[severity]
    st.markdown(f"""
    <div style='background:{color_bg};padding:16px;border-radius:12px;text-align:center;font-size:16px'>
        <b>{label}</b><br>{msg}<br><br><b>Risk Probability:</b> {prob*100:.1f}%
    </div>
    """, unsafe_allow_html=True)

    # Risk gauge
    st.markdown("#### 📈 Simulated Risk Gauge")
    img_bytes = make_probability_bar(prob, severity)
    st.image(img_bytes, use_container_width=True)

    # Extra Graph — comparison chart
    st.markdown("#### 📊 Risk Comparison Overview")
    categories = ["Healthy", "Borderline", "At Risk"]
    values = [random.randint(50, 90), random.randint(30, 70), random.randint(60, 95)]
    fig4, ax4 = plt.subplots(figsize=(6,3))
    bars = ax4.bar(categories, values, color=["#4caf50", "#f4c542", "#ff4d4d"])
    for i, v in enumerate(values):
        ax4.text(i, v + 1, f"{v}%", ha='center', fontsize=9, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("Percentage")
    ax4.set_title("Simulated Risk Levels", fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    st.pyplot(fig4)
    plt.close(fig4)

else:
    st.info("Please upload both `.hea` and `.dat` files to start analysis.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Cardiac Pre-Stroke © 2025 — Simulation for demo only, not for clinical use.")
