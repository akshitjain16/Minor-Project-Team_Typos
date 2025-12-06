import streamlit as st

st.title("ℹ️ About This Project")

st.write("""
### 📌 Project Overview
This Streamlit app uses:
- **XGBoost**
- **PCA Feature Extraction**
- **MLP Classifier**
- **Android Permission Dataset (30,000 apps)**  

It predicts whether an Android app is **Benign** or **Malicious** based on its permission set.

### 🔧 ML Pipeline
1. Missing value imputation  
2. Permission feature extraction  
3. Standard Scaling  
4. PCA (100 components → binarized → top 10 features)  
5. Classification using XGBoost  

### 👨‍💻 Author
Built by: *Your Name*

### 📚 Dataset Source
Android Permissions + App Metadata Dataset.
""")
