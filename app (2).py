import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# =========================
# ⬆ تحميل الملفات من GitHub
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("ecg_model.joblib")
    scaler = joblib.load("scaler.joblib")
    imputer = joblib.load("ecg_imputer_pro.joblib")
    encoder = joblib.load("label_encoder.joblib")
    with open("feature_columns.json", "r") as f:
        feature_cols = json.load(f)
    return model, scaler, imputer, encoder, feature_cols

model, scaler, imputer, encoder, feature_cols = load_model()

# =========================
# 🧠 دالة التنبؤ
# =========================
def predict_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    # التأكد إن الداتا فيها نفس الأعمدة
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if len(missing_cols) > 0:
        return f"⚠ Missing columns in uploaded file: {missing_cols}"

    df = df[feature_cols]

    # معالجة القيم الناقصة
    df = imputer.transform(df)

    # Scaling
    df = scaler.transform(df)

    # Prediction
    preds = model.predict(df)
    preds = encoder.inverse_transform(preds)

    return preds

# =========================
# 🎨 واجهة التطبيق
# =========================
st.title("🫀 Cardiac Pre-Stroke Detection AI")
st.subheader("Upload ECG File to Analyze")

uploaded_file = st.file_uploader("Upload CSV ECG Data", type=["csv"])

if uploaded_file is not None:
    st.success("File Uploaded Successfully ✔")

    if st.button("🔍 Analyze ECG"):
        with st.spinner("Analyzing..."):
            results = predict_from_csv(uploaded_file)

            if isinstance(results, str):
                st.error(results)
            else:
                st.subheader("📌 Prediction Results:")
                result_df = pd.DataFrame({"Prediction": results})
                st.dataframe(result_df)

                risk_count = result_df["Prediction"].value_counts()
                st.bar_chart(risk_count)

else:
    st.info("Upload ECG CSV file to begin analysis.")
