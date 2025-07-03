# Bank Customer Analytics Dashboard App (Simplified Version)
import streamlit as st
import pandas as pd

st.set_page_config(page_title="🏦 Bank Customer Analytics", layout="wide")

st.title("📊 Bank Customer Analytics Dashboard")

uploaded_file = st.file_uploader("Upload your Excel dataset (Cleaned data sheet)", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name='Cleaned data')
    st.write("Data Preview:")
    st.dataframe(df.head())
    st.success(f"Uploaded {df.shape[0]} records with {df.shape[1]} columns.")
else:
    st.warning("Please upload your dataset to begin.")
