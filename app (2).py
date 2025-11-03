# app.py
import streamlit as st
import numpy as np
import pandas as pd
import random, re, ast, warnings
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

warnings.filterwarnings("ignore")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Cardiac Pre-Stroke", page_icon="🫀", layout="centered")

# ----------------------------
# Custom CSS (Teal Gradient Theme)
# ----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #00b4d8 0%, #90e0ef 100%);
    color: #ffffff;
}
[data-testid="stSidebar"] {
    display: none;
}
h1, h2, h3, h4 {
    color: #ffffff;
}
.stButton>button {
    background-color: #023e8a;
    color: #ffffff;
    border-radius: 10px;
    border: 1px solid #ffffff;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #90e0ef;
    color: #023e8a;
    border: 1px solid #023e8a;
}
div.stAlert {
    background-color: rgba(0, 62, 138, 0.3) !important;
    border-left: 4px solid #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.title("💙 Cardiac Pre-Stroke Risk Predictor")
st.caption("AI-based simulated prediction — for demonstration purposes only.")

# ----------------------------
# Upload ECG Files
# ----------------------------
st.markdown("### 📤 Upload ECG Record (.hea + .dat)")
hea_file = st.file_uploader("Upload .hea file", type=["hea"])
dat_file = st.file_uploader("Upload .dat file", type=["dat"])

# ----------------------------
# Utility Functions
# ----------------------------
def extract_numeric_id(name):
    match = re.search(r'(\d+)(?!.*\d)', name)
    return int(match.group(1)) if match else None

def simulate_auto_result(nid):
    if nid is None:
        prob = random.uniform(0.4, 0.6)
        return prob, "Unknown", "⚠ Unable to determine automatically.", "medium"

    if nid % 2 == 1:
        prob = random.uniform(0.74, 0.90)  # sick (odd)
        return prob, "Patient", "⚠ The patient may be at cardiac pre-stroke risk.", "high"
    else:
        prob = random.uniform(0.05, 0.20)  # healthy (even)
        return prob, "Not Patient", "💚 Appears healthy — low risk detected.", "low"

def make_probability_bar(prob, severity):
    fig, ax = plt.subplots(figsize=(6,1.2))
    colors = {"high":"#ff4d4d","medium":"#f4c542","low":"#4caf50"}
    ax.barh(["Risk"], [prob], color=colors[severity], height=0.5)
    ax.set_xlim(0,1)
    ax.set_yticks([])
    ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_xlabel("Risk Level", color="#ffffff")
    ax.text(prob, 0, f"{prob*100:.1f}%", va='center', fontsize=11, fontweight='bold', color='white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    buf = BytesIO()
    plt.tight_layout()
    fig.patch.set_alpha(0)
    fig.savefig(buf, format="png", dpi=120, bbox_inches='tight', transparent=True)
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

    st.markdown(f"*Record Name:* {record_name}")

    try:
        rec = rdrecord(record_name)
        sig = rec.p_signal
        y = sig[:,0] if sig.ndim > 1 else sig

        # ECG waveform
        st.markdown("#### 🩺 ECG Signal (first 2000 samples)")
        fig1, ax1 = plt.subplots(figsize=(8,2.2))
        ax1.plot(y[:2000], color="#03045e", linewidth=0.9)
        ax1.set_xlim(0, min(2000, len(y)))
        ax1.set_ylabel("Amplitude", color="#ffffff")
        ax1.set_xlabel("Samples", color="#ffffff")
        ax1.grid(alpha=0.3)
        fig1.patch.set_alpha(0)
        st.pyplot(fig1)
        plt.close(fig1)

        # Histogram
        st.markdown("#### 📊 Amplitude Distribution")
        fig2, ax2 = plt.subplots(figsize=(6,2))
        ax2.hist(y, bins=60, color="#0077b6", alpha=0.9)
        ax2.set_xlabel("Amplitude", color="#ffffff")
        ax2.set_ylabel("Count", color="#ffffff")
        ax2.grid(alpha=0.2)
        fig2.patch.set_alpha(0)
        st.pyplot(fig2)
        plt.close(fig2)

        # RMS Sparkline
        st.markdown("#### ⚡ Signal RMS Trend")
        if len(y) > 50:
            rms = np.sqrt(pd.Series(y).rolling(window=60).mean().fillna(method='bfill').values)
            fig3, ax3 = plt.subplots(figsize=(6,1.2))
            ax3.plot(rms[-200:], color="#03045e", linewidth=0.9)
            ax3.set_yticks([])
            ax3.set_xticks([])
            fig3.patch.set_alpha(0)
            st.pyplot(fig3)
            plt.close(fig3)
    except Exception as e:
        st.warning(f"Unable to render ECG: {e}")
        y = None

    # Simulated Auto Result
    nid = extract_numeric_id(record_name)
    prob, label, msg, severity = simulate_auto_result(nid)

    color_bg = {"high":"#ff6b6b","medium":"#ffd166","low":"#06d6a0"}[severity]
    st.markdown(f"""
    <div style='background:{color_bg};padding:16px;border-radius:12px;text-align:center;font-size:17px;color:white'>
        <b>{label}</b><br>{msg}<br><br><b>Risk Probability:</b> {prob*100:.1f}%
    </div>
    """, unsafe_allow_html=True)

    # Risk gauge
    st.markdown("#### 📈 Risk Gauge")
    img_bytes = make_probability_bar(prob, severity)
    st.image(img_bytes, use_container_width=True)

else:
    st.info("Please upload both .hea and .dat files to start analysis.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("💙 Cardiac Pre-Stroke © 2025 — Smart Auto Mode — Not for clinical use.")
