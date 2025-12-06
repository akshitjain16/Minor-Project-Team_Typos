import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.title("📊 Dataset Visualization")

uploaded = st.file_uploader("Upload Android_Permission CSV", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Numeric heatmap
    df_num = df.select_dtypes(include=[np.number])

    st.write("### 🔥 Permission Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_num.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Permission count
    st.write("### Permission Distribution")
    fig2, ax2 = plt.subplots()
    df_num.sum().sort_values(ascending=False).head(20).plot(kind='bar', ax=ax2)
    st.pyplot(fig2)
