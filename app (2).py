import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, ast
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

# =============================
# PAGE CONFIGURATION
# =============================
st.set_page_config(page_title="Cardiac Pre-Stroke Predictor", page_icon="🫀", layout="centered")
st.title("💙 Cardiac Pre-Stroke Predictor")
st.caption("Upload ECG signals or feature files, process them, and predict stroke risk using AI.")

# =============================
# UPLOAD PTB-XL DATABASE
# =============================
st.markdown("### 🩺 Upload PTB-XL Metadata File (ptbxl_database.csv)")
ptbxl_file = st.file_uploader("Upload ptbxl_database.csv", type=["csv"])

if ptbxl_file is not None:
    ptbxl_df = pd.read_csv(ptbxl_file)
    st.success(f"✅ Metadata loaded successfully with {len(ptbxl_df)} records.")
    st.session_state["ptbxl_df"] = ptbxl_df
else:
    st.warning("⚠️ Please upload ptbxl_database.csv to enable label matching.")

# =============================
# MODEL FILES
# =============================
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"
FEATURES_PATH = "features_selected.npy"

st.markdown("### ⚙️ Upload Model Files:")
up_model = st.file_uploader("meta_logreg.joblib", type=["joblib", "pkl"])
up_scaler = st.file_uploader("scaler.joblib", type=["joblib", "pkl"])
up_imputer = st.file_uploader("imputer.joblib", type=["joblib", "pkl"])
up_feats = st.file_uploader("features_selected.npy (optional)", type=["npy"])

if st.button("💾 Save Uploaded Files"):
    if up_model: open(MODEL_PATH, "wb").write(up_model.read())
    if up_scaler: open(SCALER_PATH, "wb").write(up_scaler.read())
    if up_imputer: open(IMPUTER_PATH, "wb").write(up_imputer.read())
    if up_feats: open(FEATURES_PATH, "wb").write(up_feats.read())
    st.success("✅ All files saved successfully. Please rerun the app to reload them.")

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    selected_idx = None
    if os.path.exists(FEATURES_PATH):
        selected_idx = np.load(FEATURES_PATH)
        st.info(f"✅ Feature selection index loaded ({len(selected_idx)} features).")
    else:
        st.warning("⚠️ features_selected.npy not found — using all features.")
    return model, scaler, imputer, selected_idx

try:
    model, scaler, imputer, selected_idx = load_artifacts()
except Exception as e:
    st.stop()
    st.error(f"❌ Failed to load model or preprocessing files: {e}")

# =============================
# FEATURE EXTRACTION
# =============================
def extract_micro_features(sig):
    sig = np.asarray(sig, dtype=float)
    diffs = np.diff(sig)
    return np.array([
        np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
        np.ptp(sig), np.sqrt(np.mean(sig**2)), np.median(sig),
        np.percentile(sig, 25), np.percentile(sig, 75),
        skew(sig), kurtosis(sig),
        np.mean(np.abs(diffs)), np.std(diffs), np.max(diffs),
        np.mean(np.square(diffs)), np.percentile(diffs, 90), np.percentile(diffs, 10)
    ])

def align(X, expected, name):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if expected is None:
        return X
    if X.shape[1] < expected:
        add = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], add))])
        st.info(f"Added {add} placeholder columns for {name}.")
    elif X.shape[1] > expected:
        cut = X.shape[1] - expected
        X = X[:, :expected]
        st.info(f"Trimmed {cut} extra features for {name}.")
    return X

def apply_feature_selection(X, selected_idx):
    if selected_idx is not None:
        if X.shape[1] >= len(selected_idx):
            X = X[:, selected_idx]
            st.success(f"✅ Applied feature selection ({len(selected_idx)} features).")
        else:
            st.warning("⚠️ Feature selection skipped (not enough features).")
    return X

# =============================
# MAIN INTERFACE
# =============================
st.markdown("---")
mode = st.radio("Select Input Type:", ["Raw ECG (.hea + .dat)", "Feature File (CSV / NPY)"])
threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)

# =============================
# RAW ECG MODE
# =============================
if mode == "Raw ECG (.hea + .dat)":
    hea_file = st.file_uploader("Upload .hea file", type=["hea"])
    dat_file = st.file_uploader("Upload .dat file", type=["dat"])

    if hea_file and dat_file:
        tmp = hea_file.name.replace(".hea", "")
        open(hea_file.name, "wb").write(hea_file.read())
        open(dat_file.name, "wb").write(dat_file.read())

        try:
            rec = rdrecord(tmp)
            sig = rec.p_signal[:, 0]
            st.line_chart(sig[:2000], height=200)
            st.caption("Preview: First 2000 ECG samples")

            # ====== MATCH WITH PTBXL DATABASE ======
            true_label = "Unknown"
            if "ptbxl_df" in st.session_state:
                df = st.session_state["ptbxl_df"]
                matched = df[df["filename_hr"].str.contains(tmp, na=False)]
                if len(matched) > 0:
                    raw_code = matched["scp_codes"].values[0]
                    try:
                        code_dict = ast.literal_eval(raw_code) if isinstance(raw_code, str) else raw_code
                        main_label = list(code_dict.keys())[0] if len(code_dict) > 0 else "Unknown"
                        true_label = main_label
                        st.info(f"🩸 True label from database: {true_label}")
                    except Exception:
                        st.info(f"🩸 Raw code text: {raw_code}")
                else:
                    st.warning("⚠️ No matching record found in ptbxl_database.csv.")

            # ====== FEATURE EXTRACTION AND PREDICTION ======
            feats = extract_micro_features(sig).reshape(1, -1)
            feats = apply_feature_selection(feats, selected_idx)
            feats = align(feats, len(imputer.statistics_), "Imputer")
            X_imp = imputer.transform(feats)
            X_imp = align(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, model.n_features_in_, "Model")

            prob = model.predict_proba(X_scaled)[0, 1]
            pred_label = "Patient" if prob >= threshold else "Not Patient"

            # ====== DISPLAY RESULT ======
            st.markdown("### 🧠 Prediction Result:")
            result_df = pd.DataFrame({
                "Record": [tmp],
                "True Label": [true_label],
                "Predicted": [pred_label],
                "Probability": [f"{prob*100:.2f}%"]
            })
            st.dataframe(result_df)

            # ====== FEEDBACK CARD ======
            if pred_label == "Patient":
                st.markdown(
                    "<div style='background-color:#ffcccc; padding:15px; border-radius:10px; text-align:center; font-size:18px;'>🚨 <b>Warning:</b> The patient is likely suffering from a cardiac abnormality.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='background-color:#ccffcc; padding:15px; border-radius:10px; text-align:center; font-size:18px;'>💚 <b>Good News:</b> The patient shows no critical signs of cardiac abnormality.</div>",
                    unsafe_allow_html=True,
                )

            # ====== LABEL CONSISTENCY CHECK ======
            if "NORM" not in str(true_label).upper() and pred_label == "Not Patient":
                st.warning("⚠️ Model predicted (Not Patient) but the database label suggests an abnormal ECG.")
            elif "NORM" in str(true_label).upper() and pred_label == "Patient":
                st.warning("⚠️ Model predicted (Patient) but the database label suggests a normal ECG.")
            else:
                st.success("✅ Model prediction matches the database label!")

            # ====== PLOT PROBABILITY ======
            fig1, ax1 = plt.subplots()
            ax1.bar(["Not Patient", "Patient"], [1 - prob, prob],
                    color=["#6cc070", "#ff6b6b"])
            ax1.set_ylabel("Probability")
            ax1.set_title("Predicted Stroke Risk Probability")
            st.pyplot(fig1)

        except Exception as e:
            st.error(f"❌ Error processing ECG record: {e}")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("""
✅ **Notes:**
- Integrated with PTB-XL metadata for real ECG label validation.  
- Automatically aligns and scales input features.  
- Designed for **research and educational purposes only** — not for clinical diagnosis.
""")
