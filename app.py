# ------------------------------------------------------------
# Bank Customer Analytics Dashboard  (Unified v3 – syntax‑fixed)                                                
# ------------------------------------------------------------
# • Global light theme & black text                                                            
# • Sidebar flow: Upload → Info → Objectives → How‑to → Segment filters                        
# • Tabs: Data Viz · Classification · Clustering · Association · Regression · Time‑Series      
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

# 1️⃣ Page config & CSS
st.set_page_config(page_title="🏦 Bank Customer Analytics", layout="wide")
st.markdown("""
<style>
html, body, [data-testid='stApp'], .main { background-color:#f9f9f9 !important; color:#000 !important; }
</style>
""", unsafe_allow_html=True)

# 2️⃣ Sidebar: upload + static guidance
with st.sidebar:
    st.title("🏦 Bank Analytics Dashboard")
    uploaded_file = st.file_uploader("Upload Excel dataset (sheet: Cleaned data)", type=["xlsx"])
    st.markdown("---")
    st.info("1. Upload ▶ 2. Filter ▶ 3. Explore ▶ 4. Download", icon="ℹ️")

    st.markdown("## 🎯 Objectives")
    st.markdown("""- Predict **Customer Churn**  
- Estimate **Satisfaction Scores**  
- **Segment Customers**  
- Identify **Retention Patterns**  
- Quantify **FinAdvisor Impact**""")

    st.markdown("## 💡 How to Use")
    st.markdown("""1. Upload your **Cleaned data** sheet  
2. Adjust **Segment Filters** below  
3. Explore analysis tabs  
4. Download or screenshot results""")
    st.markdown("---")

# 3️⃣ Load dataset or halt
if uploaded_file is None and 'df' not in st.session_state:
    st.warning("📁 Upload an Excel file to start.")
    st.stop()

if uploaded_file is not None:
    try:
        st.session_state['df'] = pd.read_excel(uploaded_file, sheet_name='Cleaned data')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df: pd.DataFrame = st.session_state['df']

# 4️⃣ Sidebar: dynamic filters
with st.sidebar:
    st.subheader("Segment Filters")
    col1, col2 = st.columns(2)
    with col1:
        gender_filter = st.multiselect('Gender', df['Gender'].unique().tolist(), default=list(df['Gender'].unique()))
        account_filter = st.multiselect('Account Type', df['Account_Type'].unique().tolist(), default=list(df['Account_Type'].unique()))
        region_filter = st.multiselect('Region', df['Region'].unique().tolist(), default=list(df['Region'].unique()))
    with col2:
        marital_filter = st.multiselect('Marital Status', df['Marital_Status'].unique().tolist(), default=list(df['Marital_Status'].unique()))
        loan_type_filter = st.multiselect('Loan Type', df['Loan_Type'].unique().tolist(), default=list(df['Loan_Type'].unique()))

    age_range = st.slider('Age Range', int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    income_range = st.slider('Annual Income', int(df['Annual_Income'].min()), int(df['Annual_Income'].max()), (int(df['Annual_Income'].min()), int(df['Annual_Income'].max())))

@st.cache_data(show_spinner=False)
def filter_df(data: pd.DataFrame) -> pd.DataFrame:
    return data[
        data['Gender'].isin(gender_filter) &
        data['Account_Type'].isin(account_filter) &
        data['Region'].isin(region_filter) &
        data['Marital_Status'].isin(marital_filter) &
        data['Loan_Type'].isin(loan_type_filter) &
        data['Age'].between(*age_range) &
        data['Annual_Income'].between(*income_range)
    ].copy()

filtered_df = filter_df(df)

# 5️⃣ Tabs setup
TAB_NAMES = ['📊 Data Visualisation', '🤖 Classification', '🧩 Clustering', '🔗 Association Rules', '📈 Regression', '⏳ Time Series Trends']
T_VIZ, T_CLASS, T_CLUST, T_ASSOC, T_REGR, T_TS = st.tabs(TAB_NAMES)

# -----------------------
# 📊 VISUALISATION TAB   
# -----------------------
with T_VIZ:
    st.header("📊 Data Visualisation")
    if filtered_df.empty:
        st.warning("No data for selected filters.")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric("Churn Rate (%)", f"{filtered_df['Churn_Label'].mean()*100:.2f}" if 'Churn_Label' in filtered_df else 'N/A')
    k2.metric("Avg Satisfaction", f"{filtered_df['Customer_Satisfaction_Score'].mean():.2f}" if 'Customer_Satisfaction_Score' in filtered_df else 'N/A')
    k3.metric("Avg Balance", f"{filtered_df['Account_Balance'].mean():,.0f}" if 'Account_Balance' in filtered_df else 'N/A')
    st.divider()

    # Example chart: Churn by Account Type
    if {'Account_Type','Churn_Label'}.issubset(filtered_df.columns):
        chart = filtered_df.groupby('Account_Type')['Churn_Label'].mean().reset_index()
        st.plotly_chart(px.bar(chart, x='Account_Type', y='Churn_Label', color='Churn_Label', text_auto='.2%', color_continuous_scale='Reds'), use_container_width=True)

# -----------------------
# 🤖 CLASSIFICATION TAB  
# -----------------------
with T_CLASS:
    st.header("🤖 Churn Prediction (Classification)")
    target = 'Churn_Label'
    drop_cols = ['Customer_ID','Transaction_Date','Account_Open_Date','Last_Transaction_Date','Churn_Timeframe','Simulated_New_Churn_Label']
    if target not in filtered_df.columns or filtered_df[target].nunique() < 2:
        st.warning("Need Churn_Label with both classes in data.")
    else:
        X = filtered_df.drop(columns=drop_cols + [target], errors='ignore')
        y = filtered_df[target]
        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=['object','category']):
            X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
        X_train,X_test,y_train,y_test = train_test_split(X_enc.fillna(0), y, test_size=0.25, random_state=42, stratify=y)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:,1]
        st.metric("Accuracy", f"{accuracy_score(y_test,y_pred):.2%}")
        st.write("Confusion Matrix", confusion_matrix(y_test,y_pred))
        fpr,tpr,_=roc_curve(y_test,y_prob)
        fig,ax=plt.subplots()
        ax.plot(fpr,tpr,label=f"AUC={auc(fpr,tpr):.2f}"); ax.plot([0,1],[0,1],'k--'); ax.legend();
        st.pyplot(fig)

# -----------------------
# 🧩 CLUSTERING TAB      
# -----------------------
with T_CLUST:
    st.header("🧩 Customer Clustering")
    num_cols = filtered_df.select_dtypes(include='number').columns.difference(['Customer_ID','Churn_Label','Simulated_New_Churn_Label'])
    if len(num_cols) < 2:
        st.warning("Need ≥2 numeric cols after filters.")
    else:
        X = StandardScaler().fit_transform(filtered_df[num_cols])
        k = st.slider("k (clusters)", 2, 10, 5)
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        clusters = km.labels_
        df_clust = filtered_df.assign(Cluster=clusters)
        st.dataframe(df_clust.groupby('Cluster')[num_cols].mean().round(2))
        st.download_button("Download clusters", df_clust.to_csv(index=False).encode(), "clusters.csv", "text/csv")

# -----------------------
# 🔗 ASSOCIATION RULES   
# -----------------------
with T_ASSOC:
    st.header("🔗 Association Rules")
    cat_cols = filtered_df.select_dtypes(include='object').columns.tolist()
    if len(cat_cols) < 2:
        st.warning("Need ≥2 categorical columns.")
    else:
        cols = st.multiselect("Categorical columns", cat_cols, default=cat_cols[:2])
        if len(cols) >= 2:
            encoded = pd.get_dummies(filtered_df[cols].astype(str))
            freq = apriori(encoded, min_support=0.05, use_colnames=True)
            rules = association_rules(freq, metric="confidence", min_threshold=0.3)
            if rules.empty:
                st.warning("No rules; lower thresholds.")
            else:
                rules = rules.sort_values('confidence', ascending=False)
                rules['antecedents'] = rules['antecedents'].apply(lambda x:', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x:', '.join(list(x)))
                st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

# -----------------------
# 📈 REGRESSION TAB      
# -----------------------
with T_REGR:
    st.header("📈 Regression")
    targets = [c for c in ['Account_Balance','Annual_Income','Customer_Satisfaction_Score'] if c in filtered_df.columns]
    if not targets:
        st.warning("No numeric targets.")
    else:
        y_col = st.selectbox("Target variable", targets)
        X = filtered_df.drop(columns=['Customer_ID',y_col,'Churn_Label','Simulated_New_Churn_Label'], errors='ignore')
        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=['object','category']):
            X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
        y = filtered_df[y_col]
        X_train,X_test,y_train,y_test = train_test_split(X_enc.fillna(0), y, test_size=0.25, random_state=42)
        reg = LinearRegression().fit(X_train,y_train)
        y_pred = reg.predict(X_test)
        try:
            rmse = mean_squared_error(y_test,y_pred,squared=False)
        except TypeError:
            rmse = np.sqrt(mean_squared
