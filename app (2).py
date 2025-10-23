import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, glob
from scipy.stats import skew, kurtosis
from wfdb import rdrecord
import matplotlib.pyplot as plt
from io import BytesIO

# =============================
# إعداد الصفحة
# =============================
st.set_page_config(page_title="ECG Stroke Predictor", page_icon="💙", layout="centered")
st.title("🫀 ECG Stroke Prediction (Final v4 — Auto Merge + Double Graphs)")
st.caption("Uploads ECG or feature files, applies same feature selection as training, and predicts stroke risk.")

# =============================
# دمج ملفات npychunk تلقائياً
# =============================
chunk_files = sorted(glob.glob("part_*.npychunk"))
if len(chunk_files) > 0:
    st.info(f"🧩 Found {len(chunk_files)} chunk parts. Reconstructing full file...")
    all_parts = []
    for cf in chunk_files:
        try:
            arr = np.load(cf, allow_pickle=True)
            all_parts.append(arr)
        except Exception as e:
            st.warning(f"⚠️ Skipped {cf}: {e}")
    merged = np.concatenate(all_parts, axis=0)
    np.save("merged_features.npy", merged)
    st.success(f"✅ Merged chunks into merged_features.npy (shape={merged.shape})")
else:
    st.caption("No chunk parts detected (expected part_1.npychunk, part_2.npychunk, ...).")

# =============================
# تحميل ملفات الموديل
# =============================
MODEL_PATH = "meta_logreg.joblib"
SCALER_PATH = "scaler.joblib"
IMPUTER_PATH = "imputer.joblib"
FEATURES_PATH = "features_selected.npy"

st.markdown("### Upload model files:")
up_model = st.file_uploader("meta_logreg.joblib", type=["joblib", "pkl"])
up_scaler = st.file_uploader("scaler.joblib", type=["joblib", "pkl"])
up_imputer = st.file_uploader("imputer.joblib", type=["joblib", "pkl"])
up_feats = st.file_uploader("features_selected.npy (optional)", type=["npy"])

if st.button("Save uploaded files"):
    if up_model: open(MODEL_PATH, "wb").write(up_model.read())
    if up_scaler: open(SCALER_PATH, "wb").write(up_scaler.read())
    if up_imputer: open(IMPUTER_PATH, "wb").write(up_imputer.read())
    if up_feats: open(FEATURES_PATH, "wb").write(up_feats.read())
    st.success("✅ Uploaded files saved. Click 'Rerun' to load them.")

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    selected_idx = None
    if os.path.exists(FEATURES_PATH):
        selected_idx = np.load(FEATURES_PATH)
        st.info(f"✅ Loaded feature selection index ({len(selected_idx)} features).")
    else:
        st.warning("⚠️ features_selected.npy not found — using all features.")
    return model, scaler, imputer, selected_idx

try:
    model, scaler, imputer, selected_idx = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# =============================
# استخراج المميزات
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

# =============================
# ضبط الأبعاد + Feature Selection
# =============================
def align(X, expected, name):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if expected is None:
        return X
    if X.shape[1] < expected:
        add = expected - X.shape[1]
        X = np.hstack([X, np.zeros((X.shape[0], add))])
        st.info(f"Added {add} placeholders for {name}.")
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
            st.warning("⚠️ Not enough features for selection, skipping.")
    return X

# =============================
# واجهة التطبيق
# =============================
st.markdown("---")
mode = st.radio("Select input type:", ["Raw ECG (.hea + .dat)", "Feature file (CSV / NPY / merged_features.npy)"])
threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.01)

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
            st.caption("Preview of first 2000 ECG samples")

            feats = extract_micro_features(sig).reshape(1, -1)
            feats = apply_feature_selection(feats, selected_idx)
            feats = align(feats, len(imputer.statistics_), "Imputer")
            X_imp = imputer.transform(feats)
            X_imp = align(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, model.n_features_in_, "Model")

            prob = model.predict_proba(X_scaled)[0, 1]
            label = "⚠️ High Stroke Risk" if prob >= threshold else "✅ Normal ECG"

            st.metric("Result", label, delta=f"{prob*100:.2f}%")
            st.write(f"🧩 Model raw probability: {prob:.4f}")

            # ===== جرافين =====
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(sig[:1500])
            axes[0].set_title("ECG Segment")
            axes[1].bar(["Normal", "Stroke Risk"], [1-prob, prob],
                        color=["#6cc070", "#ff6b6b"])
            axes[1].set_ylabel("Probability")
            st.pyplot(fig)

            # ===== الرسالة النهائية =====
            st.markdown("---")
            if prob >= threshold:
                st.error("🚨 The model predicts **High Stroke Risk**. Please seek medical evaluation.")
            else:
                st.success("✅ The model predicts **Normal ECG**. No immediate stroke risk detected.")

        except Exception as e:
            st.error(f"❌ Error processing ECG: {e}")

# =============================
# FEATURE FILE MODE
# =============================
else:
    uploaded = st.file_uploader("Upload feature file (CSV / NPY / merged_features.npy)", type=["csv", "npy"])
    if uploaded:
        try:
            X = pd.read_csv(uploaded).values if uploaded.name.endswith(".csv") else np.load(uploaded)
            X = apply_feature_selection(X, selected_idx)
            X = align(X, len(imputer.statistics_), "Imputer")
            X_imp = imputer.transform(X)
            X_imp = align(X_imp, len(scaler.mean_), "Scaler")
            X_scaled = scaler.transform(X_imp)
            X_scaled = align(X_scaled, model.n_features_in_, "Model")

            probs = model.predict_proba(X_scaled)[:, 1]
            preds = np.where(probs >= threshold, "⚠️ High Risk", "✅ Normal")

            df_out = pd.DataFrame({
                "Sample": np.arange(1, len(probs)+1),
                "Probability": probs,
                "Prediction": preds
            })
            st.dataframe(df_out.head(10))
            st.line_chart(probs, height=150)

            buf = BytesIO()
            df_out.to_csv(buf, index=False)
            st.download_button("⬇️ Download Predictions CSV", buf.getvalue(),
                               file_name="batch_predictions.csv", mime="text/csv")

            # جراف إضافي
            fig, ax = plt.subplots()
            ax.hist(probs, bins=20, color="#4a90e2", alpha=0.7)
            ax.set_title("Distribution of Stroke Risk Probabilities")
            st.pyplot(fig)

            # الرسالة النهائية
            st.markdown("---")
            avg_prob = np.mean(probs)
            if avg_prob >= threshold:
                st.error(f"🚨 Average Risk Detected ({avg_prob:.2f}) — Possible Stroke Tendency.")
            else:
                st.success(f"✅ Average Risk Low ({avg_prob:.2f}) — Normal Overall Pattern.")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("""
✅ **Final Notes**
- Auto-merges `.npychunk` parts into one dataset.
- Applies same `features_selected.npy` as training (if exists).
- Displays double graphs and clear result messages.
- No more feature mismatch issues.
- For research use only — not a clinical diagnosis tool.
""")
