# ------------------------------
# Bank Customer Analytics Dashboard (Combined & Enhanced)
# ------------------------------
# * Light mode (CSS override for consistency)
# * All segment filters consolidated in the sidebar (left)
# * Original tabs/features preserved (Objectives, How-to, Viz, ML etc.)
# * Shared filtered dataframe used by every tab for a reactive experience
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# ------------------------------------------------------------------
# Page-wide settings + light theme CSS tweak
# ------------------------------------------------------------------
st.set_page_config(
    page_title="🏦 Bank Customer Analytics",
    layout="wide"
)
LIGHT_CSS = """
<style>
/* force light background even if Streamlit is in dark mode */
html, body, [data-testid="stApp"], .main { background-color: #f9f9f9 !important; color: #1a1a1a; }
</style>
"""
st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Tabs (unchanged from original)
# ------------------------------------------------------------------
tab_names = [
    "🎯 Objectives",
    "💡 How to Use",
    "📊 Data Visualisation",
    "🤖 Classification",
    "🧩 Clustering",
    "🔗 Association Rules",
    "📈 Regression",
    "⏳ Time Series Trends"
]
T_OBJECTIVES, T_HOWTO, T_VIZ, T_CLASSIFY, T_CLUSTER, T_ASSOC, T_REGR, T_TS = st.tabs(tab_names)

# ------------------------------------------------------------------
# Sidebar – Upload + Global Filters (NEW)
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Bank Analytics Dashboard")
    uploaded_file = st.file_uploader("Upload Excel dataset (with 'Cleaned data' sheet)", type=["xlsx"])
    st.markdown("---")
    st.info("1. Upload data → 2. Set filters → 3. Explore tabs → 4. Download insights!", icon="ℹ️")

# ---------------------
# 1️⃣ Data Loading
# ---------------------
if uploaded_file is None and 'df' not in st.session_state:
    st.warning("📁 Please upload your Excel file to unlock dashboard features.")
    st.stop()

if uploaded_file is not None:
    try:
        st.session_state['df'] = pd.read_excel(uploaded_file, sheet_name='Cleaned data')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df: pd.DataFrame = st.session_state['df']

# ---------------------
# 2️⃣ Sidebar Filters (GLOBAL) – applied once, used everywhere
# ---------------------
with st.sidebar:
    st.subheader("Segment Filters")
    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        gender_filter = st.multiselect("Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))
        account_filter = st.multiselect("Account Type", options=df['Account_Type'].unique(), default=list(df['Account_Type'].unique()))
        region_filter = st.multiselect("Region", options=df['Region'].unique(), default=list(df['Region'].unique()))
    with col_cat2:
        marital_filter = st.multiselect("Marital Status", options=df['Marital_Status'].unique(), default=list(df['Marital_Status'].unique()))
        loan_type_filter = st.multiselect("Loan Type", options=df['Loan_Type'].unique(), default=list(df['Loan_Type'].unique()))

    age_range = st.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    income_range = st.slider("Annual Income", int(df['Annual_Income'].min()), int(df['Annual_Income'].max()), (int(df['Annual_Income'].min()), int(df['Annual_Income'].max())))

@st.cache_data(show_spinner=False)
def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe subset according to sidebar filters."""
    subset = data[
        data['Gender'].isin(gender_filter) &
        data['Account_Type'].isin(account_filter) &
        data['Region'].isin(region_filter) &
        data['Marital_Status'].isin(marital_filter) &
        data['Loan_Type'].isin(loan_type_filter) &
        data['Age'].between(*age_range) &
        data['Annual_Income'].between(*income_range)
    ].copy()
    return subset

filtered_df = apply_filters(df)

# ------------------------------------------------------------------
# 3️⃣ Objectives Tab (unchanged)
# ------------------------------------------------------------------
with T_OBJECTIVES:
    st.markdown("## 🎯 Dashboard Objectives")
    st.markdown("""
**This dashboard helps you:**
- Predict Customer Churn (retain clients)
- Estimate Satisfaction Scores (focus on at-risk clients)
- Segment Customers for Offers (target personas)
- Find High Retention Patterns (build loyalty)
- Quantify FinAdvisor's impact (business case for tech)

_Navigate tabs above to explore each goal!_
""")

# ------------------------------------------------------------------
# 4️⃣ How to Use Tab (unchanged)
# ------------------------------------------------------------------
with T_HOWTO:
    st.markdown("## 💡 How to Use This Dashboard")
    st.markdown("""
**Steps:**
1. Upload your Excel data (`Cleaned data` sheet).
2. Use the sidebar to set global filters (left side).
3. Explore analysis tabs for insights.
4. Download results for presentations or action.
""")

# ------------------------------------------------------------------
# 5️⃣ Data Visualisation Tab (uses filtered_df)
# ------------------------------------------------------------------
with T_VIZ:
    st.header("📊 Data Visualisation")

    if filtered_df.empty:
        st.warning("No records match your filter selection. Try adjusting filters.")
        st.stop()

    # --- KPI Section ---
    k1, k2, k3 = st.columns(3)
    k1.metric("Churn Rate (%)", f"{filtered_df['Churn_Label'].mean()*100:.2f}" if 'Churn_Label' in filtered_df.columns else "N/A")
    k2.metric("Avg. Satisfaction", f"{filtered_df['Customer_Satisfaction_Score'].mean():.2f}" if 'Customer_Satisfaction_Score' in filtered_df.columns else "N/A")
    k3.metric("Avg. Account Balance", f"{filtered_df['Account_Balance'].mean():,.0f}" if 'Account_Balance' in filtered_df.columns else "N/A")

    st.markdown("---")

    # 1. Churn Rate by Account Type
    if {'Account_Type', 'Churn_Label'}.issubset(filtered_df.columns):
        st.subheader("1️⃣ Churn Rate by Account Type")
        churn_rate = filtered_df.groupby('Account_Type')['Churn_Label'].mean().reset_index()
        fig_cr = px.bar(churn_rate, x='Account_Type', y='Churn_Label', text_auto='.2%', color='Churn_Label', color_continuous_scale='Reds')
        fig_cr.update_layout(showlegend=False, yaxis_title="Churn Rate")
        st.plotly_chart(fig_cr, use_container_width=True)

    # 2. Average Account Balance by Region
    if {'Region', 'Account_Balance'}.issubset(filtered_df.columns):
        st.subheader("2️⃣ Average Account Balance by Region")
        region_balance = filtered_df.groupby('Region')['Account_Balance'].mean().reset_index()
        fig_ab = px.bar(region_balance, x='Region', y='Account_Balance', text_auto='.2s', color='Account_Balance', color_continuous_scale='Blues')
        st.plotly_chart(fig_ab, use_container_width=True)

    # 3. Churn by Age Group
    if {'Age', 'Churn_Label'}.issubset(filtered_df.columns):
        st.subheader("3️⃣ Churn Rate by Age Group")
        bins = [17, 25, 35, 45, 55, 65, 80]
        labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        temp_df = filtered_df.copy()
        temp_df['Age_Group'] = pd.cut(temp_df['Age'], bins=bins, labels=labels, include_lowest=True)
        churn_by_age = temp_df.groupby('Age_Group')['Churn_Label'].mean().reset_index()
        fig_cage = px.line(churn_by_age, x='Age_Group', y='Churn_Label', markers=True)
        fig_cage.update_traces(line_color='red')
        st.plotly_chart(fig_cage, use_container_width=True)

    # 4. Satisfaction by Account Type
    if {'Account_Type', 'Customer_Satisfaction_Score'}.issubset(filtered_df.columns):
        st.subheader("4️⃣ Customer Satisfaction by Account Type")
        satisfaction = filtered_df.groupby('Account_Type')['Customer_Satisfaction_Score'].mean().reset_index()
        fig_sat = px.bar(satisfaction, x='Account_Type', y='Customer_Satisfaction_Score', text_auto='.2f', color='Customer_Satisfaction_Score', color_continuous_scale='Greens')
        st.plotly_chart(fig_sat, use_container_width=True)

    # 5. Loan Amount Distribution by Loan Type
    if {'Loan_Type', 'Loan_Amount'}.issubset(filtered_df.columns):
        st.subheader("5️⃣ Loan Amount Distribution by Loan Type")
        loan_dist = filtered_df.groupby('Loan_Type')['Loan_Amount'].sum().reset_index().sort_values("Loan_Amount", ascending=False)
        fig_loan = px.bar(loan_dist, x='Loan_Type', y='Loan_Amount', text_auto='.2s', color='Loan_Amount', color_continuous_scale='Viridis')
        st.plotly_chart(fig_loan, use_container_width=True)

    # 6. Credit Score Distribution
    if {'Churn_Label', 'Credit_Score'}.issubset(filtered_df.columns):
        st.subheader("6️⃣ Credit Score Distribution: Churned vs. Non-Churned")
        fig_box = px.box(filtered_df, x='Churn_Label', y='Credit_Score', color='Churn_Label', labels={'Churn_Label': 'Churned'}, points='all')
        fig_box.update_xaxes(tickvals=[0, 1], ticktext=['Not Churned', 'Churned'])
        st.plotly_chart(fig_box, use_container_width=True)

    # 7. Customer Count by Branch
    if 'Branch' in filtered_df.columns:
        st.subheader("7️⃣ Top 10 Branches by Customer Count")
        top_branches = filtered_df['Branch'].value_counts().head(10).reset_index()
        top_branches.columns = ['Branch', 'Count']
        fig_br = px.bar(top_branches, x='Branch', y='Count', color='Count', color_continuous_scale='teal')
        st.plotly_chart(fig_br, use_container_width=True)

    # 8. Transaction Type Distribution
    if 'Transaction_Type' in filtered_df.columns:
        st.subheader("8️⃣ Transaction Type Distribution")
        trx_dist = filtered_df['Transaction_Type'].value_counts().reset_index()
        trx_dist.columns = ['Transaction Type', 'Count']
        fig3 = px.pie(trx_dist, names='Transaction Type', values='Count', hole=0.3)
        st.plotly_chart(fig3, use_container_width=True)

    # 9. Monthly Transaction Amount Trend
    if {'Transaction_Date', 'Transaction_Amount'}.issubset(filtered_df.columns):
        st.subheader("9️⃣ Monthly Transaction Amount Trend")
        tmp = filtered_df.copy()
        tmp['Transaction_Month'] = pd.to_datetime(tmp['Transaction_Date']).dt.to_period('M').astype(str)
        monthly_trx = tmp.groupby('Transaction_Month')['Transaction_Amount'].sum().reset_index()
        fig_mt = px.line(monthly_trx, x='Transaction_Month', y='Transaction_Amount', markers=True)
        st.plotly_chart(fig_mt, use_container_width=True)

    # 10. Correlation Heatmap
    numeric_cols = filtered_df.select_dtypes(include='number').drop(columns=['Churn_Label'], errors='ignore')
    if len(numeric_cols.columns) > 1:
        st.subheader("🔟 Correlation Heatmap")
        corr = numeric_cols.corr()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
        st.pyplot(fig2)

# ------------------------------------------------------------------
# 6️⃣ Classification Tab – Churn Prediction (uses filtered_df)
# ------------------------------------------------------------------
with T_CLASSIFY:
    st.header("🤖 Churn Prediction (Classification)")
    target = 'Churn_Label'
    drop_cols = ['Customer_ID', 'Transaction_Date', 'Account_Open_Date', 'Last_Transaction_Date', 'Churn_Timeframe', 'Simulated_New_Churn_Label']

    if target not in filtered_df.columns:
        st.warning("Churn_Label column missing.")
    elif filtered_df[target].nunique() < 2:
        st.warning("Need both classes (churn & non-churn) in filtered data. Adjust filters.")
    else:
        features = [col for col in filtered_df.columns if col not in drop_cols + [target]]
        X = filtered_df[features]
        y = filtered_df[target]

        # Encode categoricals
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']):
            X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
        X_encoded = X_encoded.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42, stratify=y)
        with st.spinner("Training Random Forest classifier ..."):
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:", cm)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], "k--")
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate"); ax_roc.legend()
        st.pyplot(fig_roc)

# ------------------------------------------------------------------
# 7️⃣ Clustering Tab
# ------------------------------------------------------------------
with T_CLUSTER:
    st.header("🧩 Customer Clustering")
    numeric_exclude = ["Customer_ID", "Churn_Label", "Simulated_New_Churn_Label", "Transaction_Date", "Account_Open_Date", "Last_Transaction_Date", "Churn_Timeframe"]
    cluster_features = filtered_df.select_dtypes(include='number').drop(columns=numeric_exclude, errors='ignore').columns.tolist()

    if len(cluster_features) < 2:
        st.warning("Need at least two numeric features for clustering after filters.")
    else:
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(filtered_df[cluster_features])
        k = st.slider("Select k (clusters)", 2, 20, 5)
        with st.spinner("Running KMeans ..."):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_cluster)
        df_clusters = filtered_df.copy()
        df_clusters['Cluster'] = labels
        st.success(f"Clustered {len(df_clusters)} records into {k} segments.")

        # Elbow
        inertias = []
        elbow_range = range(2, 11)
        for ki in elbow_range:
            km = KMeans(n_clusters=ki, random_state=42)
            km.fit(X_cluster)
            inertias.append(km.inertia_)
        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(elbow_range, inertias, marker="o"); ax_elbow.set_xlabel("k"); ax_elbow.set_ylabel("Inertia"); ax_elbow.set_title("Elbow Curve")
        st.pyplot(fig_elbow)

        st.dataframe(df_clusters.groupby('Cluster')[cluster_features].mean().round(2))

        # Download
        csv = df_clusters.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data", csv, "clustered_customers.csv", "text/csv")

# ------------------------------------------------------------------
# 8️⃣ Association Rules Tab
# ------------------------------------------------------------------
with T_ASSOC:
    st.header("🔗 Association Rule Mining")
    cat_cols = filtered_df.select_dtypes(include='object').columns.tolist()

    if len(cat_cols) < 2:
        st.warning("Need at least 2 categorical cols for association mining.")
    else:
        apriori_cols = st.multiselect("Choose categorical columns (≥2):", options=cat_cols, default=cat_cols[:2])
        min_sup = st.slider("Min Support", 0.01, 0.2, 0.05, step=0.01)
        min_conf = st.slider("Min Confidence", 0.01, 1.0, 0.3, step=0.01)
        min_lift = st.slider("Min Lift", 1.0, 5.0, 1.2, step=0.1)
        if len(apriori_cols) >= 2:
            encoded_df = pd.get_dummies(filtered_df[apriori_cols].astype(str))
            with st.spinner("Running Apriori ..."):
                freq_items = apriori(encoded_df, min_support=min_sup, use_colnames=True)
                rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
                rules = rules[rules['lift'] >= min_lift]
            if rules.empty:
                st.warning("No rules found. Try lowering thresholds.")
            else:
                rules = rules.sort_values('confidence', ascending=False)
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].reset_index(drop=True))

# ------------------------------------------------------------------
# 9️⃣ Regression Tab
# ------------------------------------------------------------------
with T_REGR:
    st.header("📈 Regression (Predict Numeric Targets)")
    possible_targets = [c for c in ['Account_Balance', 'Annual_Income', 'Customer_Satisfaction_Score'] if c in filtered_df.columns]
    if not possible_targets:
        st.warning("No numeric targets present after filtering.")
    else:
        target_reg = st.selectbox("Select target variable", possible_targets)
        reg_drop = ['Customer_ID', 'Transaction_Date', 'Account_Open_Date', 'Last_Transaction_Date', 'Churn_Label', 'Simulated_New_Churn_Label', 'Churn_Timeframe'] + possible_targets
        reg_features = [c for c in filtered_df.columns if c not in reg_drop]
        if not reg_features:
            st.warning("No explanatory features available.")
        else:
            X = filtered_df[reg_features]
            y = filtered_df[target_reg]
            X_enc = X.copy()
            for c in X_enc.select_dtypes(include=['object', 'category']):
                X_enc[c] = LabelEncoder().fit_transform(X_enc[c].astype(str))
            X_enc = X_enc.fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.25, random_state=42)
            with st.spinner("Training Linear Regression ..."):
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)
            st.metric("R²", f"{r2_score(y_test, y_pred):.2f}")
            st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
            st.metric("RMSE", f"{mean_squared_error(y_test, y_pred, squared=False):.2f}")

# ------------------------------------------------------------------
# 🔟 Time Series Trends Tab
# ------------------------------------------------------------------
with T_TS:
    st.header("⏳ Time Series Trends")
    if {'Transaction_Date'}.issubset(filtered_df.columns):
        tmp = filtered_df.copy()
        tmp['Transaction_Month'] = pd.to_datetime(tmp['Transaction_Date']).dt.to_period('M').astype(str)
        metric_cols = [c for c in ['Transaction_Amount', 'Account_Balance', 'Customer_Satisfaction_Score'] if c in tmp.columns]
        if metric_cols:
            monthly_metrics = tmp.groupby('Transaction_Month')[metric_cols].mean().reset_index()
            fig_ts = px.line(monthly_metrics, x='Transaction_Month', y=metric_cols, markers=True)
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.warning("Numeric metrics not found for time series plot.")
    else:
        st.warning("Transaction_Date column not found in data.")

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.caption("*If a feature doesn't show up, your data might be missing required columns.*")
