import streamlit as st

st.set_page_config(page_title="Android Malware Detector", layout="wide")

st.title("🔐 Android Malware Detection App")
st.subheader("Using PCA + XGBoost + MLP on Android Permissions")

st.write("""
Welcome to the **Android Malware Detector**.  
This tool analyzes Android App permissions and predicts whether an app is **Benign** or **Malicious**.
Use the sidebar to navigate through:
- **Upload & Predict**
- **Visualization**
- **Test Permissions Manually**
- **About the project**
""")

st.info("👈 Use the sidebar to navigate.")
