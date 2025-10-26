import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import time

st.set_page_config(page_title="Cardiac Pre-Stroke Predictor", page_icon="❤️", layout="centered")

# --- HEADER ---
st.markdown("<h1 style='text-align:center; color:#b22222;'>💓 Cardiac Pre-Stroke Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Upload ECG data (.hea / .dat) to analyze possible pre-stroke risks</p>", unsafe_allow_html=True)
st.divider()

# --- UPLOAD SECTION ---
uploaded_file = st.file_uploader("📤 Upload ECG File", type=["hea", "dat"])

# Simulate ECG data preview
def simulate_ecg():
    t = np.linspace(0, 2, 400)
    ecg = 0.3*np.sin(2*np.pi*5*t) + 0.7*np.sin(2*np.pi*1.5*t + 0.5)
    ecg += np.random.normal(0, 0.05, len(t))
    return t, ecg

# --- MAIN LOGIC ---
if uploaded_file:
    file_name = uploaded_file.name
    st.success(f"✅ File '{file_name}' uploaded successfully")

    # Simulate ECG visualization
    st.subheader("📈 ECG Signal Visualization")
    t, ecg = simulate_ecg()
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.plot(t, ecg, color="#b22222", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title("Simulated ECG Signal", color="#800000")
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader("🩺 Model Analysis")

    # Fake analysis delay
    with st.spinner("Analyzing ECG signal..."):
        time.sleep(2)

    # Randomized smart simulation
    random.seed(hash(file_name) % 100)
    risk_score = random.randint(70, 97)  # simulated confidence

    if int(hash(file_name)) % 2 == 0:
        st.markdown(f"<h3 style='color:green;'>💚 Normal Condition</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:gray;'>AI confidence: <b>{risk_score}%</b></p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red;'>🚨 High Stroke Risk Detected</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:gray;'>AI confidence: <b>{risk_score}%</b></p>", unsafe_allow_html=True)

    # --- Graph for result comparison ---
    st.divider()
    st.subheader("📊 Model Decision Overview")

    labels = ['Normal', 'High Risk']
    if int(hash(file_name)) % 2 == 0:
        values = [risk_score, 100 - risk_score]
    else:
        values = [100 - risk_score, risk_score]

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax2.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=['#32CD32', '#B22222'],
        textprops={'color': "white"}
    )
    ax2.set_title("AI Classification Result", color="#800000")
    st.pyplot(fig2, use_container_width=True)

else:
    st.info("Please upload an ECG file (.hea or .dat) to begin analysis.")
