# ------------------------------------------------------------
# Bank Customer Analytics Dashboard  (Unified v2)                                                   
# ------------------------------------------------------------
# ▸ Global light theme & black text                                                         
# ▸ Sidebar flow:   Upload ➜ Info ➜ Objectives ➜ How‑to ➜ Segment filters                   
# ▸ Tabs (top):     Data Viz · Classification · Clustering · Association · Regression · TS  
# ▸ Original analyses preserved, version‑safe RMSE, elbow chart, etc.                       
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1️⃣  Page config & global CSS (light + black text)
# --------------------------------------------------
st.set_page_config(page_title="🏦 Bank Customer Analytics", layout="wide")
LIGHT_CSS = """
<style>
html, body, [data-testid='stApp'], .main {
    background-color: #f9f9f9 !important;
    color: #000000 !important;
}
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# --------------------------------------------------
# 2️⃣  SIDEBAR – Upload, info, objectives, how‑to
# --------------------------------------------------
with st.sidebar:
    st.title("🏦 Bank Analytics Dashboard")
    uploaded_file = st.file_uploader("Upload Excel dataset (with 'Cleaned data' sheet)", type=["xlsx"])
    st.markdown("---")
    st.info("1. Upload ▶ 2. Filter ▶ 3. Explore ▶ 4. Download", icon="ℹ️")

    st.markdown("## 🎯 Objectives")
    st.markdown("""
- **Predict Customer Churn**  
- **Estimate Satisfaction Scores**  
- **Segment Customers**  
- **Identify Retention Patterns**  
- **Quantify FinAdvisor Impact**
""")

    st.markdown("## 💡 How to Use")
    st.markdown("""
1. Upload your **Cleaned data** sheet  
2. Adjust **Segment Filters** below  
3. Switch between analysis tabs  
4. Download / screenshot insights
""")
    st.markdown("---")

# --------------------------------------------------
# 3️⃣  Data loading guard
# --------------------------------------------------
if uploaded_file is None and 'df' not in st.session_state:
    st.warning('📁 Upload an Excel file to unlock dashboard features.')
    st.stop()

if uploaded_file is not None:
    try:
        st.session_state['df'] = pd.read_excel(uploaded_file, sheet_name='Cleaned data')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df: pd.DataFrame = st.session_state['df']

# --------------------------------------------------
# 4️⃣  Sidebar – dynamic segment filters
# --------------------------------------------------
with st.sidebar:
    st.subheader('Segment Filters')
    col1, col2 = st.columns(2)
    with col1:
        gender_filter = st.multiselect('Gender', df['Gender'].unique().tolist(), default=df['Gender'].unique().tolist())
        account_filter = st.multiselect('Account Type', df['Account_Type'].unique().tolist(), default=df['Account_Type'].unique().tolist())
        region_filter = st.multiselect('Region', df['Region'].unique().tolist(), default=df['Region'].unique().tolist())
    with col2:
        marital_filter = st.multiselect('Marital Status', df['Marital_Status'].unique().tolist(), default=df['Marital_Status'].unique().tolist())
        loan_type_filter = st.multiselect('Loan Type', df['Loan_Type'].unique().tolist(), default=df['Loan_Type'].unique().tolist())

    age_range = st.slider('Age Range', int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    income_range = st.slider('Annual Income', int(df['Annual_Income'].min()), int(df['Annual_Income'].max()), (int(df['Annual_Income'].min()), int(df['Annual_Income'].max())))

@st.cache_data(show_spinner=False)
def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    return data[
        data['Gender'].isin(gender_filter) &
        data['Account_Type'].isin(account_filter) &
        data['Region'].isin(region_filter) &
        data['Marital_Status'].isin(marital_filter) &
        data['Loan_Type'].isin(loan_type_filter) &
        data['Age'].between(*age_range) &
        data['Annual_Income'].between(*income_range)
    ].copy()

filtered_df = apply_filters(df)

# --------------------------------------------------
# 5️⃣  Top‑level analysis tabs
# --------------------------------------------------
TAB_NAMES = [
    '📊 Data Visualisation',
    '🤖 Classification',
    '🧩 Clustering',
    '🔗 Association Rules',
    '📈 Regression',
    '⏳ Time Series Trends'
]
T_VIZ, T_CLASSIFY, T_CLUSTER, T_ASSOC, T_REGR, T_TS = st.tabs(TAB_NAMES)

# ==================================================
# 📊 DATA VISUALISATION
# ==================================================
with T_VIZ:
    st.header('📊 Data Visualisation')

    if filtered_df.empty:
        st.warning('No records match current filters.')
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric('Churn Rate (%)', f"{filtered_df['Churn_Label'].mean()*100:.2f}" if 'Churn_Label' in filtered_df.columns else 'N/A')
    k2.metric('Avg. Satisfaction', f"{filtered_df['Customer_Satisfaction_Score'].mean():.2f}" if 'Customer_Satisfaction_Score' in filtered_df.columns else 'N/A')
    k3.metric('Avg. Account Balance', f"{filtered_df['Account_Balance'].mean():,.0f}" if 'Account_Balance' in filtered_df.columns else 'N/A')
    st.divider()

    # --- Churn by Account Type ---
    if {'Account_Type', 'Churn_Label'}.issubset(filtered_df.columns):
        churn_rate = filtered_df.groupby('Account_Type')['Churn_Label'].mean().reset_index()
        fig = px.bar(churn_rate, x='Account_Type', y='Churn_Label', text_auto='.2%', color='Churn_Label', color_continuous_scale='Reds')
        fig.update_layout(showlegend=False, yaxis_title='Churn Rate')
        st.plotly_chart(fig, use_container_width=True)

    # --- Avg Balance by Region ---
    if {'Region', 'Account_Balance'}.issubset(filtered_df.columns):
        bal = filtered_df.groupby('Region')['Account_Balance'].mean().reset_index()
        fig = px.bar(bal, x='Region', y='Account_Balance', text_auto='.2s', color='Account_Balance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    # --- Churn by Age Group ---
    if {'Age', 'Churn_Label'}.issubset(filtered_df.columns):
        bins = [17, 25, 35, 45, 55, 65, 80]
        labels = ['18‑24', '25‑34', '35‑44', '45‑54', '55‑64', '65+']
        tmp = filtered_df.copy()
        tmp['Age_Group'] = pd.cut(tmp['Age'], bins=bins, labels=labels, include_lowest=True)
        cg = tmp.groupby('Age_Group')['Churn_Label'].mean().reset_index()
        fig = px.line(cg, x='Age_Group', y='Churn_Label', markers=True)
        fig.update_traces(line_color='red')
        st.plotly_chart(fig, use_container_width=True)

    # --- Satisfaction by Account Type ---
    if {'Account_Type', 'Customer_Satisfaction_Score'}.issubset(filtered_df.columns):
        sat = filtered_df.groupby('Account_Type')['Customer_Satisfaction_Score'].mean().reset_index()
        fig = px.bar(sat, x='Account_Type', y='Customer_Satisfaction_Score', text_auto='.2f', color='Customer_Satisfaction_Score', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)

    # --- Loan Amount by Loan Type ---
    if {'Loan_Type', 'Loan_Amount'}.issubset(filtered_df.columns):
        loan = filtered_df.groupby('Loan_Type')['Loan_Amount'].sum().reset_index().sort_values('Loan_Amount', ascending=False)
        fig = px.bar(loan, x='Loan_Type', y='Loan_Amount', text_auto='.2s', color='Loan_Amount', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

    # --- Credit Score Distribution ---
    if {'Churn_Label', 'Credit_Score'}.issubset(filtered_df.columns):
        fig = px.box(filtered_df, x='Churn_Label', y='Credit_Score', color='Churn_Label', labels={'Churn_Label': 'Churned'}, points='all')
        fig.update_xaxes(tickvals=[0, 1], ticktext=['Not Churned', 'Churned'])
        st.plotly_chart(fig, use_container_width=True)

    # --- Branch count ---
    if 'Branch' in filtered_df.columns:
        top_b = filtered_df['Branch'].value_counts().head(10).reset_index()
        top_b.columns = ['Branch', 'Count']
        fig = px.bar(top_b, x='Branch', y='Count', color='Count', color_continuous_scale='Teal')
        st.plotly_chart(fig, use_container_width=True)

    # --- Transaction Type Pie ---
    if 'Transaction_Type' in filtered_df.columns:
        trx = filtered_df['Transaction_Type'].value_counts().reset_index()
        trx.columns = ['Type', 'Count']
        fig = px.pie(trx, names='Type', values='Count', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

    # --- Monthly Transaction Trend ---
    if {'Transaction_Date', 'Transaction_Amount'}.issubset(filtered_df.columns):
        tmp = filtered_df.copy()
        tmp['Transaction_Month'] = pd.to_datetime(tmp['Transaction_Date']).dt.to_period('M').astype(str)
        mt = tmp.groupby('Transaction_Month')['Transaction_Amount'].sum().reset_index()
        fig = px.line(mt, x='Transaction_Month', y='Transaction_Amount', markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # --- Correlation Heatmap ---
    num_cols = filtered_df.select_dtypes(include='number').drop(columns=['Churn_Label'], errors='ignore')
    if num_cols.shape[1] > 1:
        corr = num_cols.corr()
        fig_hm, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig_hm)

# ==================================================
# 🤖 CLASSIFICATION
# ==================================================
with T_CLASSIFY:
    st.header('🤖 Churn Prediction (Classification)')
    target = 'Churn_Label'
    drop_cols = ['Customer_ID', 'Transaction_Date', 'Account_Open_Date', 'Last_Transaction_Date', 'Churn_Timeframe', '
