import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.title("🧪 Manual Permission Tester")
st.write("Toggle permissions and test if the app becomes Malicious or Benign.")

# =====================================
# Load Models
# =====================================
model = joblib.load("model/xgboost_model.joblib")
pca = joblib.load("model/pca_model.joblib")

# =====================================
# Read CSV (used only for detecting available permission names in dataset)
# =====================================
df = pd.read_csv("Dataset/Android_Permission.csv")

# detect 0/1 columns in CSV (original method)
detected_permissions = [
    col for col in df.columns
    if df[col].dropna().isin([0, 1]).all() and col.lower() != "class"
]

# =====================================
# Determine the permission list to show / use (align to PCA if possible)
# =====================================
# pca.n_features_in_ is the number of features the PCA expects
expected_n = getattr(pca, "n_features_in_", None)

# Prefer PCA's saved feature names (guarantees correct order)
if hasattr(pca, "feature_names_in_"):
    permission_list = list(pca.feature_names_in_)
    st.info(f"Using PCA's saved feature names ({len(permission_list)} features).")
else:
    # fallback to detected permissions from CSV
    permission_list = detected_permissions.copy()
    st.warning("PCA model does not contain 'feature_names_in_'. Using detected columns from CSV. "
               "If lengths mismatch we'll pad/truncate to match PCA's expected input length.")

st.write(f"Permissions available for toggles: {len(permission_list)}")
if expected_n is not None:
    st.write(f"PCA expects features: {expected_n}")

# =====================================
# Permission toggles (use permission_list)
# =====================================
perm_values = []
cols = st.columns(4)
c = 0

st.subheader("Select Permissions")
for perm in permission_list:
    # default unchecked (False) — you can change default if desired
    perm_values.append(cols[c].checkbox(perm, False))
    c = (c + 1) % 4

# Convert to 1×N binary array (for the displayed permission_list)
perm_array = np.array(perm_values).reshape(1, -1)  # shape (1, len(permission_list))

# =====================================
# Align perm_array to PCA expected number of features
# =====================================
if expected_n is not None:
    current_n = perm_array.shape[1]
    if current_n < expected_n:
        # pad zeros on the right to match expected
        pad_width = expected_n - current_n
        st.warning(f"Detected {current_n} toggles but PCA expects {expected_n} features — padding with {pad_width} zeros.")
        perm_array = np.pad(perm_array, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
    elif current_n > expected_n:
        # truncate extra toggles (keeps the first expected_n)
        st.warning(f"Detected {current_n} toggles but PCA expects {expected_n} features — truncating to first {expected_n} features.")
        perm_array = perm_array[:, :expected_n]
else:
    # If PCA doesn't provide n_features_in_, we still proceed with whatever length the UI produced.
    st.info(f"PCA has no n_features_in_. Proceeding with {perm_array.shape[1]} features from toggles.")

# Extra diagnostic: if PCA had feature_names_in_ but some of those names are not present in the CSV,
# show which ones won't map to dataset values (useful if you want to know which toggles came from CSV)
if hasattr(pca, "feature_names_in_"):
    missing_from_csv = [name for name in pca.feature_names_in_ if name not in detected_permissions]
    if missing_from_csv:
        st.warning(f"The following PCA-feature names were NOT found in your CSV (they will default to the checkbox values you set): {missing_from_csv[:10]}{'...' if len(missing_from_csv)>10 else ''}")

# =====================================
# PCA → Inverse → Binarize
# =====================================
# Now perm_array should have the correct number of features expected by PCA (or at least a consistent size)
pca_out = pca.transform(perm_array)
reconstructed = pca.inverse_transform(pca_out)

# Binarize reconstructed back to 0/1
reconstructed[reconstructed <= 0] = 0
reconstructed[reconstructed != 0] = 1
reconstructed = reconstructed.astype(int)

# First 10 PCA-binary features (keeps your original behavior)
perm_final = reconstructed[:, :10]

# =====================================
# Prediction
# =====================================
pred = model.predict(perm_final)[0]
prob = model.predict_proba(perm_final)[0][1]

st.subheader("Prediction Result")

if pred == 1:
    st.error(f"🚨 MALICIOUS (Probability = {prob:.2f})")
else:
    st.success(f"✅ BENIGN (Probability = {1 - prob:.2f})")

# Final debug info
st.write("PCA expects features (n_features_in_):", getattr(pca, "n_features_in_", "Unknown"))
st.write("Number of permission toggles shown:", len(permission_list))
st.write("Final input shape to PCA:", perm_array.shape)
