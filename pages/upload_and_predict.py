import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("📄 Upload Permission CSV & Predict Malware")

# Load trained model + PCA + scaler + imputer
model = joblib.load("model/xgboost_model.joblib")
pca = joblib.load("model/pca_model.joblib")
scaler = joblib.load("model/scaler.joblib")
imputer = joblib.load("model/imputer.joblib")

uploaded = st.file_uploader("Upload Android_Permission CSV", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("### Preview of uploaded data:")
    st.dataframe(df.head())

    # =============================
    # 1. METADATA SCALING (4 columns)
    # =============================
    scale_cols = [
        'Rating',
        'Number of ratings',
        'Dangerous permissions count',
        'Safe permissions count'
    ]

    # Ensure required columns exist
    for col in scale_cols:
        if col not in df.columns:
            st.error(f"❌ Missing required column: {col}")
            st.stop()

    # Apply imputer → scaler
    meta_values = imputer.transform(df[scale_cols])
    meta_scaled = scaler.transform(meta_values)

    # =============================
    # 2. PERMISSION PROCESSING → PCA
    # =============================
    # Permission columns = ALL numeric minus metadata & Class
    perm_cols = [
        col for col in df.columns
        if col not in scale_cols + ["Class", "App"] and df[col].dtype != "object"
    ]

    if len(perm_cols) == 0:
        st.error("❌ No permission columns found!")
        st.stop()

    perm_matrix = df[perm_cols].values

    # PCA
    pca_out = pca.transform(perm_matrix)

    # Same reconstruction+binarization used during training
    reconstructed = pca.inverse_transform(pca_out)
    reconstructed[reconstructed <= 0] = 0
    reconstructed[reconstructed != 0] = 1
    reconstructed = reconstructed.astype(int)

    # Use only first 10 PCA-binary features
    perm_final = reconstructed[:, :10]

    # =============================
    # 3. FINAL MODEL INPUT
    # =============================
    X_input = perm_final   # model was trained ONLY on these 10 features

    # =============================
    # 4. Prediction
    # =============================
    preds = model.predict(X_input)
    probs = model.predict_proba(X_input)[:, 1]

    result_df = pd.DataFrame({
        "Prediction": preds,
        "Malware_Probability": probs
    })

    st.write("### 🔍 Prediction Results")
    st.dataframe(result_df)

    malicious_count = np.sum(preds)
    st.metric("Detected Malicious Apps", malicious_count)

    st.success("✅ Prediction Completed!")
