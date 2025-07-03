# ------------------------------------------------------------
# Bank Customer Analytics Dashboard  
# (Light mode, black text, objectives/how‑to in sidebar, full feature set)  
# ------------------------------------------------------------
# ▸ Global light theme & black text  
# ▸ Upload + objectives + how‑to + segment filters in sidebar  
# ▸ Tabs: Data Viz · Classification · Clustering · Assoc Rules · Regression · Time Series  
# ▸ Version‑safe RMSE calculation (old/new scikit‑learn)  
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

# ---------------- Page config & CSS ----------------
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

# ---------------- Sidebar: upload + info + static text -------------
with st.sidebar:
    st.title("🏦 Bank Analytics Dashboard")
    uploaded_file = st.file_uploader("Upload Excel dataset (with 'Cleaned data' sheet)", type=["xlsx"])
    st.markdown("---")
    st.info("1. Upload data → 2. Set filters → 3. Explore tabs → 4. Download insights!", icon="ℹ️")

    st.markdown("## 🎯 Objectives")
    st.markdown("""
- **Predict Customer Churn** for retention  
- **Estimate Satisfaction Scores** to focus on at‑risk clients  
- **Segment Customers** for personalised offers  
- **Identify High‑Retention Patterns** to build loyalty  
- **Quantify FinAdvisor Impact** for a tech business case
""")

    st.markdown("## 💡 How to Use")
    st.markdown("""
1. Upload your **Cleaned data** sheet  
2. Adjust **Segment Filters** below  
3. Explore the top‑level analysis tabs  
4. Download or screenshot results for action
""")
    st.markdown("---")

# --------------- Halt if no data --------------------
if uploaded_file is None and 'df' not in st.session_state:
    st.warning('📁 Please upload your Excel file to unlock dashboard features.')
    st.stop()

# --------------- Load data --------------------------
if uploaded_file is not None:
    try:
        st.session_state['df'] = pd.read_excel(uploaded_file, sheet_name='Cleaned data')
    except Exception as e:
        st.error(f'Error loading data: {e}')
        st.stop()

df: pd.DataFrame = st.session_state['df']

# --------------- Sidebar: dynamic filters -----------
with st.sidebar:
    st.subheader('Segment Filters')
    col1, col2 = st.columns(2)
    with col1:
        gender_filter = st.multiselect('Gender', df['Gender'].unique().tolist(), default=df['Gender'].unique().tolist())
        account_filter = st.multiselect('Account Type', df['Account_Type'].unique().tolist(), default=df['Account_Type'].unique().tolist())
        region_filter = st.multiselect('Region', df['Region'].unique().tolist(), default=df['Region'].unique().tolist())
    with col2:
        marital_filter = st.multiselect('Marital Status', df['Marital_Status'].unique().tolist(), default=df['Marital_Status'].unique().tolist())
        loan_type_filter = st.multiselect('Loan Type', df['Loan_Type'].unique().tolist(), default=df['Loan_Type'].unique().tolist())

    age_range = st.slider('Age Range', int(df['Age'].min()), int(df['Age'].max()),
                          (int(df['Age'].min()), int(df['Age'].max())))
    income_range = st.slider('Annual Income', int(df['Annual_Income'].min()), int(df['Annual_Income'].max()),
                             (int(df['Annual_Income'].min()), int(df['Annual_Income'].max())))

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

# ---------------- Tabs (analytics only) --------------
TAB_NAMES = [
    '📊 Data Visualisation',
    '🤖 Classification',
    '🧩 Clustering',
    '🔗 Association Rules',
    '📈 Regression',
    '⏳ Time Series Trends'
]
T_VIZ, T_CLASSIFY, T_CLUSTER, T_ASSOC, T_REGR, T_TS = st.tabs(TAB_NAMES)

# ====================================================
# 📊 DATA VISUALISATION TAB
# ====================================================
with T_VIZ:
    st.header('📊 Data Visualisation')

    if filtered_df.empty:
        st.warning('No records match your filter selection. Try adjusting filters.')
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric('Churn Rate (%)', f"{filtered_df['Churn_Label'].mean()*100:.2f}" if 'Churn_Label' in filtered_df else 'N/A')
    k2.metric('Avg. Satisfaction', f"{filtered_df['Customer_Satisfaction_Score'].mean():.2f}" if 'Customer_Satisfaction_Score' in filtered_df else 'N/A')
    k3.metric('Avg. Account Balance', f"{filtered_df['Account_Balance'].mean():,.0f}" if 'Account_Balance' in filtered_df else 'N/A')
    st.markdown('---')

    # Example visualisations (add more as needed)
    if {'Account_Type', 'Churn_Label'}.issubset(filtered_df.columns):
        churn_rate = filtered_df.groupby('Account_Type')['Churn_Label'].mean().reset_index()
        fig = px.bar(churn_rate, x='Account_Type', y='Churn_Label', text_auto='.2%', color='Churn_Label',
                     color_continuous_scale='Reds')
        fig.update_layout(showlegend=False, yaxis_title='Churn Rate')
        st.plotly_chart(fig, use_container_width=True)

# ====================================================
# 🤖 CLASSIFICATION TAB
# ====================================================
with T_CLASSIFY:
    st.header('🤖 Churn Prediction (Classification)')
    target = 'Churn_Label'
    drop_cols = ['Customer_ID', 'Transaction_Date', 'Account_Open_Date', 'Last_Transaction_Date',
                 'Churn_Timeframe', 'Simulated_New_Churn_Label']

    if target not in filtered_df.columns:
        st.warning('Churn_Label column missing.')
    elif filtered_df[target].nunique() < 2:
        st.warning('Need both classes in filtered data. Adjust filters.')
    else:
        X = filtered_df.drop(columns=drop_cols + [target], errors='ignore')
        y = filtered_df[target]

        X_enc = X.copy()
        for c in X_enc.select_dtypes(include=['object', 'category']):
            X_enc[c] = LabelEncoder().fit_transform(X_enc[c].astype(str))
        X_enc = X_enc.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.25,
                                                            random_state=42, stratify=y)
        with st.spinner('Training Random Forest ...'):
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]

        st.metric('Accuracy', f"{accuracy_score(y_test, y_pred):.2%}")
        cm = confusion_matrix(y_test, y_pred)
        st.write('Confusion Matrix:', cm)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = auc(fpr, tpr)
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC={auc_val:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.legend()
        st.pyplot(fig_roc)

# ====================================================
# 🧩 CLUSTERING TAB
# ====================================================
with T_CLUSTER:
    st.header('🧩 Customer Clustering')
    numeric_exclude = ['Customer_ID', 'Churn_Label', 'Simulated_New_Churn
