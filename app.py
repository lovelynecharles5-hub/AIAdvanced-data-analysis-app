import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
import io
from openai import OpenAI

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI Advanced Data Analysis App", layout="wide")
st.title("📊🤖 AI Advanced Data Analysis App")

# ==============================
# SIDEBAR MENU
# ==============================
menu = st.sidebar.radio(
    "📌 Main Menu",
    ["📂 Upload Data", "🛠 Handle Missing Values", "📊 Data Analysis", 
     "🧪 Hypothesis Tests", "📈 Regression Models", "🤝 Clustering", 
     "💡 AI Assistant", "📑 Export Report"]
)

# ==============================
# DATA UPLOAD
# ==============================
if menu == "📂 Upload Data":
    st.header("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())

# ==============================
# HANDLE MISSING VALUES
# ==============================
elif menu == "🛠 Handle Missing Values":
    st.header("🛠 Handle Missing Values")
    if "df" in st.session_state:
        df = st.session_state["df"]
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.warning("⚠️ Missing values detected:")
            st.write(missing[missing > 0])

            method = st.selectbox(
                "Choose a method to correct all missing values",
                ["Do nothing", "Drop rows", "Drop columns", 
                 "Fill with Mean", "Fill with Median", 
                 "Fill with Mode", "Fill with Zero"]
            )

            if st.button("Apply Correction"):
                if method == "Drop rows":
                    df = df.dropna()
                elif method == "Drop columns":
                    df = df.dropna(axis=1)
                elif method == "Fill with Mean":
                    df = df.fillna(df.mean(numeric_only=True))
                elif method == "Fill with Median":
                    df = df.fillna(df.median(numeric_only=True))
                elif method == "Fill with Mode":
                    for col in df.columns:
                        df[col] = df[col].fillna(df[col].mode()[0])
                elif method == "Fill with Zero":
                    df = df.fillna(0)

                st.session_state["df"] = df
                st.success(f"✅ Missing values handled using: {method}")
                st.dataframe(df.head())
        else:
            st.success("🎉 No missing values found!")
    else:
        st.warning("Please upload a dataset first.")

# ==============================
# DATA ANALYSIS
# ==============================
elif menu == "📊 Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    if "df" in st.session_state:
        df = st.session_state["df"]

        analysis_type = st.selectbox(
            "Choose Analysis",
            ["Summary Statistics", "Correlation Heatmap", 
             "Histogram", "Boxplot", "Scatter Plot", "Line Plot"]
        )

        if analysis_type == "Summary Statistics":
            st.write(df.describe(include="all"))

        elif analysis_type == "Correlation Heatmap":
            numeric_df = df.select_dtypes(include=["float64", "int64"])
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.info("No numeric columns available.")

        elif analysis_type == "Histogram":
            column = st.selectbox("Select column", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)

        elif analysis_type == "Boxplot":
            column = st.selectbox("Select column", df.columns)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[column], ax=ax)
            st.pyplot(fig)

        elif analysis_type == "Scatter Plot":
            col1 = st.selectbox("X-axis", df.columns, key="xcol")
            col2 = st.selectbox("Y-axis", df.columns, key="ycol")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
            st.pyplot(fig)

        elif analysis_type == "Line Plot":
            col1 = st.selectbox("X-axis", df.columns, key="lxcol")
            col2 = st.selectbox("Y-axis", df.columns, key="lycol")
            fig, ax = plt.subplots()
            sns.lineplot(x=df[col1], y=df[col2], ax=ax)
            st.pyplot(fig)

# ==============================
# HYPOTHESIS TESTS
# ==============================
elif menu == "🧪 Hypothesis Tests":
    st.header("🧪 Hypothesis Testing")
    if "df" in st.session_state:
        df = st.session_state["df"]

        test_type = st.selectbox("Choose Test", ["T-test", "ANOVA", "Chi-square"])

        if test_type == "T-test":
            num_cols = df.select_dtypes(include=["float64", "int64"]).columns
            col = st.selectbox("Select numeric column", num_cols)
            group_col = st.selectbox("Select group column", df.columns)
            groups = df[group_col].dropna().unique()
            if len(groups) == 2:
                g1 = df[df[group_col] == groups[0]][col].dropna()
                g2 = df[df[group_col] == groups[1]][col].dropna()
                t, p = stats.ttest_ind(g1, g2)
                st.write(f"T-test result: t={t:.3f}, p={p:.3f}")
            else:
                st.error("Group column must have exactly 2 unique values.")

        elif test_type == "ANOVA":
            num_col = st.selectbox("Select numeric column", df.select_dtypes(include=["float64", "int64"]).columns)
            group_col = st.selectbox("Select group column", df.columns)
            groups = [df[df[group_col] == g][num_col].dropna() for g in df[group_col].dropna().unique()]
            f, p = stats.f_oneway(*groups)
            st.write(f"ANOVA result: F={f:.3f}, p={p:.3f}")

        elif test_type == "Chi-square":
            col1 = st.selectbox("Select first categorical column", df.select_dtypes(include=["object"]).columns, key="chi1")
            col2 = st.selectbox("Select second categorical column", df.select_dtypes(include=["object"]).columns, key="chi2")
            table = pd.crosstab(df[col1], df[col2])
            chi2, p, dof, expected = stats.chi2_contingency(table)
            st.write(f"Chi-square result: chi2={chi2:.3f}, p={p:.3f}")

# ==============================
# REGRESSION
# ==============================
elif menu == "📈 Regression Models":
    st.header("📈 Regression Models")
    if "df" in st.session_state:
        df = st.session_state["df"]
        target = st.selectbox("Select dependent variable (Y)", df.columns)
        predictors = st.multiselect("Select independent variables (X)", [c for c in df.columns if c != target])

        if st.button("Run OLS Regression"):
            formula = f"{target} ~ {' + '.join(predictors)}"
            model = smf.ols(formula=formula, data=df).fit()
            st.write(model.summary())

# ==============================
# CLUSTERING
# ==============================
elif menu == "🤝 Clustering":
    st.header("🤝 K-Means Clustering")
    if "df" in st.session_state:
        df = st.session_state["df"]
        numeric_df = df.select_dtypes(include=["float64", "int64"]).dropna()

        k = st.slider("Number of clusters", 2, 10, 3)
        if st.button("Run Clustering"):
            km = KMeans(n_clusters=k, random_state=42).fit(numeric_df)
            df["Cluster"] = km.labels_
            st.session_state["df"] = df
            st.success("✅ Clustering done! Column 'Cluster' added.")
            st.dataframe(df.head())

# ==============================
# AI ASSISTANT
# ==============================
elif menu == "💡 AI Assistant":
    st.header("💡 AI Assistant")
    if "df" in st.session_state:
        df = st.session_state["df"]
        question = st.text_input("Ask AI about your dataset (English/Swahili)")
        if st.button("Get Answer"):
            try:
                client = OpenAI(api_key=st.secrets["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"])
                df_sample = df.head(50).to_csv(index=False)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst AI."},
                        {"role": "user", "content": f"Dataset sample:\n{df_sample}\n\nQuestion: {question}"}
                    ]
                )
                st.success(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")

# ==============================
# EXPORT REPORT
# ==============================
elif menu == "📑 Export Report":
    st.header("📑 Export Cleaned Data")
    if "df" in st.session_state:
        df = st.session_state["df"]
        export_type = st.selectbox("Choose format", ["CSV", "Excel"])
        if export_type == "CSV":
            st.download_button("Download CSV", data=df.to_csv(index=False), file_name="cleaned_data.csv")
        else:
            towrite = io.BytesIO()
            df.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button("Download Excel", data=towrite, file_name="cleaned_data.xlsx", mime="application/vnd.ms-excel")
