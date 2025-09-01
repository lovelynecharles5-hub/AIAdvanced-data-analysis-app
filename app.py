import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from analysis_python import (
    load_dataframe, basic_eda, t_test_numeric_by_group, ols_regression,
    logistic_regression, plot_histogram, plot_scatter, correlation_matrix,
    anova_test, chi_square_test, plot_boxplot, plot_line,
    export_csv, export_excel, export_plot,
    classification_model, regression_model, clustering_model
)

st.set_page_config(page_title="🤖 AI Statistical Assistant", layout="wide")
st.title("🤖 AI Statistical Assistant")

# ==============================
# File Upload
# ==============================
uploaded_file = st.sidebar.file_uploader("📂 Upload Data (CSV / Excel)", type=["csv", "xls", "xlsx"])
df = None
if uploaded_file is not None:
    df = load_dataframe(uploaded_file.read(), uploaded_file.name)
    st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")

# ==============================
# Menu Options
# ==============================
menu = st.sidebar.radio("📌 Select Analysis Mode", [
    "📊 Basic EDA",
    "🧹 Data Cleaning & Editing",
    "📈 Visualization",
    "🧪 Statistical Tests",
    "📉 Regression Models",
    "🧬 Machine Learning",
    "📤 Export Data",
    
])

# ==============================
# Basic EDA
# ==============================
if menu == "📊 Basic EDA" and df is not None:
    st.subheader("Exploratory Data Analysis")
    res = basic_eda(df)
    st.write("**Data Preview**", res['sample'])
    st.write("**Summary Stats**", res['summary'])
    st.write("**Missing Values**", res['missing'])
    st.write("**Info**", res['info'])

# ==============================
# Data Cleaning & Editing
# ==============================
elif menu == "🧹 Data Cleaning & Editing" and df is not None:
    st.subheader("Data Cleaning Options")
    clean_option = st.radio("Handle Missing Values", [
        "Do Nothing", "Drop Rows", "Fill with 0", "Fill with Mean", "Fill with Mode"
    ])

    if clean_option == "Drop Rows":
        df = df.dropna()
    elif clean_option == "Fill with 0":
        df = df.fillna(0)
    elif clean_option == "Fill with Mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif clean_option == "Fill with Mode":
        for col in df.columns:
            try:
                df[col] = df[col].fillna(df[col].mode()[0])
            except:
                pass

    st.write("✏️ Edit Data (click cells to change values)")
    df = st.data_editor(df, num_rows="dynamic")

# ==============================
# Visualization
# ==============================
elif menu == "📈 Visualization" and df is not None:
    viz_option = st.selectbox("Choose Visualization", [
        "Histogram", "Scatter Plot", "Correlation Matrix", "Box Plot", "Line Plot"
    ])

    if viz_option == "Histogram":
        col = st.selectbox("Column", df.select_dtypes(include='number').columns)
        fig = plot_histogram(df, col)
        st.pyplot(fig)

    elif viz_option == "Scatter Plot":
        x = st.selectbox("X-axis", df.columns)
        y = st.selectbox("Y-axis", df.columns)
        fig = plot_scatter(df, x, y)
        st.pyplot(fig)

    elif viz_option == "Correlation Matrix":
        fig, corr = correlation_matrix(df)
        st.pyplot(fig)

    elif viz_option == "Box Plot":
        col = st.selectbox("Numeric Column", df.select_dtypes(include='number').columns)
        group = st.selectbox("Group Column", df.columns)
        fig = plot_boxplot(df, col, group)
        st.pyplot(fig)

    elif viz_option == "Line Plot":
        x = st.selectbox("X-axis", df.columns)
        y = st.selectbox("Y-axis", df.select_dtypes(include='number').columns)
        fig = plot_line(df, x, y)
        st.pyplot(fig)

# ==============================
# Statistical Tests
# ==============================
elif menu == "🧪 Statistical Tests" and df is not None:
    test_type = st.selectbox("Choose Test", ["T-test", "ANOVA", "Chi-Square"])
    if test_type == "T-test":
        numeric_col = st.selectbox("Numeric Column", df.select_dtypes(include='number').columns)
        group_col = st.selectbox("Grouping Column", df.columns)
        if st.button("Run T-test"):
            st.json(t_test_numeric_by_group(df, numeric_col, group_col))
    elif test_type == "ANOVA":
        numeric_col = st.selectbox("Numeric Column", df.select_dtypes(include='number').columns)
        group_col = st.selectbox("Grouping Column", df.columns)
        if st.button("Run ANOVA"):
            st.json(anova_test(df, numeric_col, group_col))
    elif test_type == "Chi-Square":
        col1 = st.selectbox("Column 1", df.columns)
        col2 = st.selectbox("Column 2", df.columns)
        if st.button("Run Chi-Square"):
            st.json(chi_square_test(df, col1, col2))

# ==============================
# Regression Models
# ==============================
elif menu == "📉 Regression Models" and df is not None:
    reg_type = st.selectbox("Regression Type", ["OLS", "Logistic"])
    target = st.selectbox("Target (y)", df.columns)
    features = st.multiselect("Features (X)", [c for c in df.columns if c != target])
    if st.button("Run Regression"):
        if reg_type == "OLS":
            res = ols_regression(df, target, features)
            st.text(res['summary_text'])
        else:
            res = logistic_regression(df, target, features)
            st.text(res['summary_text'])

# ==============================
# Machine Learning
# ==============================
elif menu == "🧬 Machine Learning" and df is not None:
    ml_task = st.selectbox("Choose ML Task", ["Classification", "Regression", "Clustering"])

    if ml_task == "Classification":
        model_type = st.radio("Choose Model", ["logistic", "random_forest"])
        target = st.selectbox("Target (y)", df.columns)
        features = st.multiselect("Features (X)", [c for c in df.columns if c != target])
        if st.button("Run Classification"):
            st.json(classification_model(df, target, features, model_type))

    elif ml_task == "Regression":
        model_type = st.radio("Choose Model", ["linear", "random_forest"])
        target = st.selectbox("Target (y)", df.columns)
        features = st.multiselect("Features (X)", [c for c in df.columns if c != target])
        if st.button("Run Regression"):
            st.json(regression_model(df, target, features, model_type))

    elif ml_task == "Clustering":
        features = st.multiselect("Features for Clustering", df.columns)
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        if st.button("Run Clustering"):
            res = clustering_model(df, features, n_clusters)
            st.write("Cluster Centers:", res["centers"])
            st.pyplot(res["figure"])

# ==============================
# Export Data
# ==============================
elif menu == "📤 Export Data" and df is not None:
    st.subheader("Download Data / Results")
    st.download_button("⬇️ Download CSV", export_csv(df), "data.csv", "text/csv")
    st.download_button("⬇️ Download Excel", export_excel(df), "data.xlsx")


    report = sv.analyze(df)
    report.show_html("report.html")
    with open("report.html", "r", encoding="utf-8") as f:
        html = f.read()
        components.html(html, height=800, scrolling=True)
