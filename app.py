import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="AI Statistical Assistant",
    page_icon="📊",
    layout="wide"
)

# ==============================
# Theme Switch (Dark/Light Mode)
# ==============================
theme = st.sidebar.radio("🎨 Choose Theme", ["🌞 Light Mode", "🌙 Dark Mode"])

if theme == "🌙 Dark Mode":
    st.markdown(
        """
        <style>
        .main {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stTabs [role="tab"] {
            background-color: #262730;
            color: #FAFAFA;
        }
        h1, h2, h3, h4 {
            color: #1DB954;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .main {
            background-color: #F8F9FA;
            color: #2C3E50;
        }
        .stTabs [role="tab"] {
            background-color: #ffffff;
            color: #2C3E50;
        }
        h1, h2, h3, h4 {
            color: #2C3E50;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ==============================
# Header
# ==============================
st.markdown("<h1>📊 AI Statistical Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>A modern web app for data analysis & visualization</p>", unsafe_allow_html=True)

# ==============================
# App Description
# ==============================
with st.expander("ℹ️ About this App", expanded=True):
    st.markdown("""
    Welcome to **AI Statistical Assistant** 🎉  

    This app helps you:
    - 📂 Upload your dataset (CSV)  
    - 📊 Explore data with **EDA** and **Visualizations**  
    - 🔬 Run **Regression Models** and **Statistical Tests**  
    - 💡 Get **Smart Insights & Recommendations**  
    - 📤 Export results as CSV or PDF  

    👉 *To get started: Go to the **Upload tab**, upload your dataset, then explore the **Analysis tab**.*
    """)

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📂 Upload", "📈 Analysis", "🔮 Smart Insights", "📤 Export"]
)

# ==============================
# Upload Tab
# ==============================
with tab1:
    st.header("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(st.session_state["df"].head())

# ==============================
# Analysis Tab
# ==============================
with tab2:
    st.header("📈 Data Analysis")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first in the Upload tab.")
    else:
        df = st.session_state["df"]

        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Basic EDA", "Visualization", "Regression Models", "Statistical Tests"]
        )

        # BASIC EDA
        if analysis_type == "Basic EDA":
            st.subheader("📊 Exploratory Data Analysis")
            if st.checkbox("Show Summary Statistics"):
                st.write(df.describe())
            if st.checkbox("Show Correlation Matrix"):
                numeric_df = df.select_dtypes(include=["float64", "int64"])
                if not numeric_df.empty:
                    fig, ax = plt.subplots()
                    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("No numeric columns found.")

        # VISUALIZATION
        elif analysis_type == "Visualization":
            st.subheader("📈 Visualizations")
            viz_option = st.selectbox(
                "Choose Visualization",
                [
                    "Histogram", "Scatterplot", "Boxplot", "Line Chart",
                    "Bar Chart", "Pie Chart", "Stacked Bar Chart",
                    "Violin Plot", "Pairplot", "Heatmap"
                ]
            )

            if viz_option == "Histogram":
                col = st.selectbox("Select Column", df.columns)
                fig, ax = plt.subplots()
                df[col].hist(bins=20, ax=ax)
                ax.set_title(f"Histogram of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

            elif viz_option == "Scatterplot":
                x_col = st.selectbox("X-axis", df.columns)
                y_col = st.selectbox("Y-axis", df.columns)
                fig, ax = plt.subplots()
                ax.scatter(df[x_col], df[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)

            elif viz_option == "Boxplot":
                col = st.selectbox("Select Column", df.columns, key="box_col")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)

            elif viz_option == "Line Chart":
                col = st.selectbox("Select Column", df.columns, key="line_col")
                st.line_chart(df[col])

            elif viz_option == "Bar Chart":
                col = st.selectbox("Select Categorical Column", df.columns, key="bar_col")
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind="bar", ax=ax)
                ax.set_title(f"Bar Chart of {col}")
                st.pyplot(fig)

            elif viz_option == "Pie Chart":
                col = st.selectbox("Select Categorical Column", df.columns, key="pie_col")
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                ax.set_title(f"Pie Chart of {col}")
                st.pyplot(fig)

            elif viz_option == "Stacked Bar Chart":
                col1 = st.selectbox("First Categorical Column", df.columns, key="stack1")
                col2 = st.selectbox("Second Categorical Column", df.columns, key="stack2")
                crosstab = pd.crosstab(df[col1], df[col2])
                crosstab.plot(kind="bar", stacked=True)
                st.pyplot(plt.gcf())

            elif viz_option == "Violin Plot":
                num_col = st.selectbox("Numeric Column", df.columns, key="violin_num")
                cat_col = st.selectbox("Categorical Column", df.columns, key="violin_cat")
                fig, ax = plt.subplots()
                sns.violinplot(x=df[cat_col], y=df[num_col], ax=ax)
                st.pyplot(fig)

            elif viz_option == "Pairplot":
                numeric_df = df.select_dtypes(include=["float64", "int64"])
                if not numeric_df.empty:
                    fig = sns.pairplot(numeric_df)
                    st.pyplot(fig)
                else:
                    st.warning("No numeric columns available for pairplot.")

            elif viz_option == "Heatmap":
                numeric_df = df.select_dtypes(include=["float64", "int64"])
                if not numeric_df.empty:
                    fig, ax = plt.subplots()
                    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("No numeric columns available for heatmap.")

        # REGRESSION MODELS
        elif analysis_type == "Regression Models":
            st.subheader("📉 Regression Models")
            model_type = st.selectbox("Choose Model", ["Linear Regression", "Logistic Regression"])
            if model_type == "Linear Regression":
                target = st.selectbox("Select Target (Y)", df.columns)
                predictors = st.multiselect("Select Predictors (X)", [c for c in df.columns if c != target])
                if st.button("Run Linear Regression"):
                    if target and predictors:
                        formula = f"{target} ~ {' + '.join(predictors)}"
                        model = smf.ols(formula, data=df).fit()
                        st.write(model.summary())
            elif model_type == "Logistic Regression":
                st.info("Logistic regression setup will go here.")

        # STATISTICAL TESTS
        elif analysis_type == "Statistical Tests":
            st.subheader("🔬 Statistical Tests")

            test_type = st.selectbox(
                "Select Test",
                ["T-test", "ANOVA", "Chi-Square", "Correlation"]
            )

            if test_type == "T-test":
                num_col = st.selectbox("Numeric Column", df.select_dtypes(include=["float64", "int64"]).columns)
                group_col = st.selectbox("Grouping Column", df.columns)
                if st.button("Run T-test"):
                    groups = df[group_col].dropna().unique()
                    if len(groups) == 2:
                        g1 = df[df[group_col] == groups[0]][num_col]
                        g2 = df[df[group_col] == groups[1]][num_col]
                        t, p = stats.ttest_ind(g1, g2)
                        st.write(f"T-test results: t = {t:.3f}, p = {p:.3f}")
                    else:
                        st.error("Grouping column must have exactly 2 unique values.")

            elif test_type == "ANOVA":
                anova_col = st.selectbox("Numeric Column", df.select_dtypes(include=["float64", "int64"]).columns, key="anova")
                group_col = st.selectbox("Grouping Column", df.columns, key="anova_group")
                if st.button("Run ANOVA"):
                    groups = [group[anova_col].dropna() for name, group in df.groupby(group_col)]
                    f, p = stats.f_oneway(*groups)
                    st.write(f"ANOVA results: F = {f:.3f}, p = {p:.3f}")

            elif test_type == "Chi-Square":
                col1 = st.selectbox("Column 1 (categorical)", df.columns, key="chi1")
                col2 = st.selectbox("Column 2 (categorical)", df.columns, key="chi2")
                if st.button("Run Chi-Square Test"):
                    table = pd.crosstab(df[col1], df[col2])
                    chi2, p, dof, expected = stats.chi2_contingency(table)
                    st.write(f"Chi-square results: χ² = {chi2:.3f}, p = {p:.3f}")

            elif test_type == "Correlation":
                col_x = st.selectbox("Column X (numeric)", df.select_dtypes(include=["float64", "int64"]).columns, key="corrx")
                col_y = st.selectbox("Column Y (numeric)", df.select_dtypes(include=["float64", "int64"]).columns, key="corry")
                if st.button("Run Correlation"):
                    pearson_corr, p = stats.pearsonr(df[col_x], df[col_y])
                    st.write(f"Pearson correlation: r = {pearson_corr:.3f}, p = {p:.3f}")

# ==============================
# Smart Insights Tab
# ==============================
with tab3:
    st.header("🔮 Smart Insights")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state["df"]
        st.subheader("Quick Overview")
        st.write(f"Dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
        st.write("### Columns and Data Types")
        st.write(df.dtypes)

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            st.write(missing[missing > 0])
        else:
            st.success("No missing values 🎉")

        st.subheader("✨ Recommended Analyses")
        recommendations = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                recommendations.append(f"🔹 `{col}` is numeric → Histogram, Boxplot, Correlation.")
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
                recommendations.append(f"🔹 `{col}` is categorical → Bar Chart or Chi-Square Test.")
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(numeric_cols) >= 2:
            recommendations.append("🔹 Multiple numeric columns → Scatterplots, Pairplot, Regression.")
        if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
            recommendations.append("🔹 Numeric + categorical → T-test, ANOVA, Grouped Boxplots.")
        for rec in recommendations:
            st.write(rec)

# ==============================
# Export Tab
# ==============================
with tab4:
    st.header("📤 Export Data & Reports")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state["df"]
        st.download_button(
            label="💾 Download CSV",
            data=df.to_csv(index=False),
            file_name="exported_data.csv",
            mime="text/csv"
        )
        if st.button("Export Summary Report (PDF)"):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()
            story = [
                Paragraph("📊 Data Summary Report", styles["Title"]),
                Paragraph(df.describe().to_html(), styles["Normal"])
            ]
            doc.build(story)
            st.download_button(
                label="💾 Download PDF Report",
                data=buffer.getvalue(),
                file_name="report.pdf",
                mime="application/pdf"
            )
