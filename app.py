# ------------------------------------------------------------
# Bank Customer Analytics Dashboard  (Unified v4)
# ------------------------------------------------------------
# • Light theme, black text
# • Sidebar: Upload → Info → Objectives → How-to → Filters
# • Tabs: Data Viz · Classification · Clustering · Association
#         · Regression · Time-Series
# • Version-safe RMSE handling
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
# 1️⃣  Page config & CSS
# --------------------------------------------------
st.set_page_config(page_title="🏦 Bank Customer Analytics", layout="wide")
st.set_page_config(
    page_title="🏦 Bank Customer Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="auto",   # optional
)
# ---- GLOBAL CSS: force light background + true-black text everywhere ----
st.markdown(
    """
    <style>
        /* whole app */
        html, body, [data-testid="stApp"], .main {
            background-color: #f9f9f9 !important;   /* light */
            color: #000000 !important;               /* black */
        }

        /* sidebar container + every child element */
        [data-testid="stSidebar"], [data-testid="stSidebar"] * {
            background-color: #f9f9f9 !important;
            color: #000000 !important;
        }

        /* metric numbers & labels (they ignore inheriting text-color) */
        .stMetric > div > div {
            color: #000000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 2️⃣  SIDEBAR – upload + static guidance
# --------------------------------------------------
with st.sidebar:
    st.title("🏦 Bank Analytics Dashboard")
    uploaded_file = st.file_uploader(
        "Upload Excel dataset (sheet name: 'Cleaned data')", type=["xlsx"]
    )
    st.markdown("---")
    st.info("1. Upload ▶ 2. Filter ▶ 3. Explore ▶ 4. Download", icon="ℹ️")

    st.markdown("## 🎯 Objectives")
    st.markdown(
        """
- **Predict Customer Churn**  
- **Estimate Satisfaction Scores**  
- **Segment Customers**  
- **Identify Retention Patterns**  
- **Quantify FinAdvisor Impact**
        """
    )

    st.markdown("## 💡 How to Use")
    st.markdown(
        """
1. Upload the **Cleaned data** sheet.  
2. Adjust **Segment Filters** below.  
3. Switch analysis tabs (top).  
4. Download or capture insights.
        """
    )
    st.markdown("---")

# --------------------------------------------------
# 3️⃣  Data loading guard
# --------------------------------------------------
if uploaded_file is None and "df" not in st.session_state:
    st.warning("📁 Please upload an Excel file to start.")
    st.stop()

if uploaded_file is not None:
    try:
        st.session_state["df"] = pd.read_excel(
            uploaded_file, sheet_name="Cleaned data"
        )
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df: pd.DataFrame = st.session_state["df"]

# --------------------------------------------------
# 4️⃣  Sidebar – dynamic filters
# --------------------------------------------------
with st.sidebar:
    st.subheader("Segment Filters")
    col1, col2 = st.columns(2)
    with col1:
        gender_filter = st.multiselect(
            "Gender", df["Gender"].unique().tolist(), default=list(df["Gender"].unique())
        )
        account_filter = st.multiselect(
            "Account Type",
            df["Account_Type"].unique().tolist(),
            default=list(df["Account_Type"].unique()),
        )
        region_filter = st.multiselect(
            "Region", df["Region"].unique().tolist(), default=list(df["Region"].unique())
        )
    with col2:
        marital_filter = st.multiselect(
            "Marital Status",
            df["Marital_Status"].unique().tolist(),
            default=list(df["Marital_Status"].unique()),
        )
        loan_type_filter = st.multiselect(
            "Loan Type",
            df["Loan_Type"].unique().tolist(),
            default=list(df["Loan_Type"].unique()),
        )

    age_range = st.slider(
        "Age Range",
        int(df["Age"].min()),
        int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max())),
    )
    income_range = st.slider(
        "Annual Income",
        int(df["Annual_Income"].min()),
        int(df["Annual_Income"].max()),
        (int(df["Annual_Income"].min()), int(df["Annual_Income"].max())),
    )


@st.cache_data(show_spinner=False)
def filter_df(data: pd.DataFrame) -> pd.DataFrame:
    return data[
        data["Gender"].isin(gender_filter)
        & data["Account_Type"].isin(account_filter)
        & data["Region"].isin(region_filter)
        & data["Marital_Status"].isin(marital_filter)
        & data["Loan_Type"].isin(loan_type_filter)
        & data["Age"].between(*age_range)
        & data["Annual_Income"].between(*income_range)
    ].copy()


filtered_df = filter_df(df)

# --------------------------------------------------
# 5️⃣  Tabs layout
# --------------------------------------------------
TAB_NAMES = [
    "📊 Data Visualisation",
    "🤖 Classification",
    "🧩 Clustering",
    "🔗 Association Rules",
    "📈 Regression",
    "⏳ Time Series Trends",
]
T_VIZ, T_CLASS, T_CLUST, T_ASSOC, T_REGR, T_TS = st.tabs(TAB_NAMES)

# ==================================================
# 📊 DATA VISUALISATION TAB
# ==================================================
with T_VIZ:
    st.header("📊 Data Visualisation")

    if filtered_df.empty:
        st.warning("No data for current filters.")
        st.stop()

    k1, k2, k3 = st.columns(3)
    k1.metric(
        "Churn Rate (%)",
        f"{filtered_df['Churn_Label'].mean() * 100:.2f}"
        if "Churn_Label" in filtered_df.columns
        else "N/A",
    )
    k2.metric(
        "Avg Satisfaction",
        f"{filtered_df['Customer_Satisfaction_Score'].mean():.2f}"
        if "Customer_Satisfaction_Score" in filtered_df.columns
        else "N/A",
    )
    k3.metric(
        "Avg Balance",
        f"{filtered_df['Account_Balance'].mean():,.0f}"
        if "Account_Balance" in filtered_df.columns
        else "N/A",
    )
    st.divider()

    # Churn by Account Type
    if {"Account_Type", "Churn_Label"}.issubset(filtered_df.columns):
        churn_df = (
            filtered_df.groupby("Account_Type")["Churn_Label"].mean().reset_index()
        )
        fig = px.bar(
            churn_df,
            x="Account_Type",
            y="Churn_Label",
            color="Churn_Label",
            text_auto=".2%",
            color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, yaxis_title="Churn Rate")
        st.plotly_chart(fig, use_container_width=True)

    # (Add additional charts as desired …)

# ==================================================
# 🤖 CLASSIFICATION TAB
# ==================================================
with T_CLASS:
    st.header("🤖 Churn Prediction (Classification)")
    target = "Churn_Label"
    drop_cols = [
        "Customer_ID",
        "Transaction_Date",
        "Account_Open_Date",
        "Last_Transaction_Date",
        "Churn_Timeframe",
        "Simulated_New_Churn_Label",
    ]

    if target not in filtered_df.columns or filtered_df[target].nunique() < 2:
        st.warning("Churn_Label must exist with both classes.")
    else:
        X = filtered_df.drop(columns=drop_cols + [target], errors="ignore")
        y = filtered_df[target]

        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=["object", "category"]):
            X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X_enc.fillna(0), y, test_size=0.25, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
        st.write("Confusion Matrix", confusion_matrix(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.2f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.legend()
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        st.pyplot(fig)

# ==================================================
# 🧩 CLUSTERING TAB
# ==================================================
with T_CLUST:
    st.header("🧩 Customer Clustering")

    numeric_cols = (
        filtered_df.select_dtypes(include="number")
        .columns.difference(
            [
                "Customer_ID",
                "Churn_Label",
                "Simulated_New_Churn_Label",
                "Churn_Timeframe",
            ]
        )
    )

    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns after filtering.")
    else:
        X_scaled = StandardScaler().fit_transform(filtered_df[numeric_cols])
        k = st.slider("k (clusters)", 2, 10, 5)
        km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        clusters = km.labels_
        df_clust = filtered_df.assign(Cluster=clusters)

        st.success(f"Clustered {len(df_clust)} records into {k} segments.")
        st.dataframe(df_clust.groupby("Cluster")[numeric_cols].mean().round(2))

        # Download option
        st.download_button(
            "Download clusters",
            df_clust.to_csv(index=False).encode(),
            "clusters.csv",
            "text/csv",
        )

# ==================================================
# 🔗 ASSOCIATION RULES TAB
# ==================================================
with T_ASSOC:
    st.header("🔗 Association Rules")

    cat_cols = filtered_df.select_dtypes(include="object").columns.tolist()

    if len(cat_cols) < 2:
        st.warning("Need at least two categorical columns.")
    else:
        cols = st.multiselect("Categorical columns", cat_cols, default=cat_cols[:2])
        min_sup = st.slider("Min support", 0.01, 0.2, 0.05, 0.01)
        min_conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
        min_lift = st.slider("Min lift", 1.0, 5.0, 1.2, 0.1)

        if len(cols) >= 2:
            encoded = pd.get_dummies(filtered_df[cols].astype(str))
            freq = apriori(encoded, min_support=min_sup, use_colnames=True)
            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
            rules = rules[rules["lift"] >= min_lift]

            if rules.empty:
                st.warning("No rules found — try lower thresholds.")
            else:
                rules = rules.sort_values("confidence", ascending=False)
                rules["antecedents"] = rules["antecedents"].apply(
                    lambda x: ", ".join(list(x))
                )
                rules["consequents"] = rules["consequents"].apply(
                    lambda x: ", ".join(list(x))
                )
                st.dataframe(
                    rules[["antecedents", "consequents", "support", "confidence", "lift"]]
                )

# ==================================================
# 📈 REGRESSION TAB
# ==================================================
with T_REGR:
    st.header("📈 Regression")

    targets = [
        c
        for c in ["Account_Balance", "Annual_Income", "Customer_Satisfaction_Score"]
        if c in filtered_df.columns
    ]

    if not targets:
        st.warning("No numeric targets available.")
    else:
        y_col = st.selectbox("Target variable", targets)

        # ---- Build feature matrix ----
        base_drop = [
            "Customer_ID",
            y_col,
            "Churn_Label",
            "Simulated_New_Churn_Label",
            "Churn_Timeframe",
            "Transaction_Date",
            "Account_Open_Date",
            "Last_Transaction_Date",
        ]
        X = filtered_df.drop(columns=base_drop, errors="ignore")

        # Drop all datetime columns
        dt_cols = X.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns
        X = X.drop(columns=dt_cols)

        # Encode categoricals & booleans
        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=["object", "category"]):
            X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
        for col in X_enc.select_dtypes(include=["bool"]):
            X_enc[col] = X_enc[col].astype(int)

        y = filtered_df[y_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X_enc.fillna(0), y, test_size=0.25, random_state=42
        )

        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        # version-tolerant RMSE
        try:
            rmse_val = mean_squared_error(y_test, y_pred, squared=False)
        except TypeError:
            rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))

        st.metric("R²", f"{r2_score(y_test, y_pred):.2f}")
        st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        st.metric("RMSE", f"{rmse_val:.2f}")

# ==================================================
# ⏳ TIME SERIES TRENDS TAB
# ==================================================
with T_TS:
    st.header("⏳ Time Series Trends")

    if "Transaction_Date" not in filtered_df.columns:
        st.warning("Transaction_Date column not found.")
    else:
        tmp = filtered_df.copy()
        tmp["Transaction_Month"] = (
            pd.to_datetime(tmp["Transaction_Date"]).dt.to_period("M").astype(str)
        )
        metric_cols = [
            c
            for c in [
                "Transaction_Amount",
                "Account_Balance",
                "Customer_Satisfaction_Score",
            ]
            if c in tmp.columns
        ]

        if metric_cols:
            monthly = tmp.groupby("Transaction_Month")[metric_cols].mean().reset_index()
            fig = px.line(
                monthly, x="Transaction_Month", y=metric_cols, markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric metrics available for time-series plot.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption(
    "*If a feature doesn’t appear, your dataset may be missing the required columns.*"
)
