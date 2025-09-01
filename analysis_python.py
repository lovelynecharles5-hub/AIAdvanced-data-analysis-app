from typing import Dict, Any, List, Optional, Tuple
import io
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats

# ==============================
# Load Data
# ==============================
def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(io.BytesIO(file_bytes))
    else:
        try:
            return pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            return pd.read_csv(io.BytesIO(file_bytes), sep=';')

# ==============================
# Basic EDA
# ==============================
def basic_eda(df: pd.DataFrame) -> Dict[str, Any]:
    desc = df.describe(include='all') \
             .transpose().reset_index().rename(columns={'index': 'column'})
    
    missing = df.isna().mean().reset_index()
    missing.columns = ['column', 'missing_rate']
    
    sample = df.head(10)
    
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
    }
    
    return {
        'summary': desc,
        'missing': missing,
        'sample': sample,
        'info': info
    }

# ==============================
# Statistical Tests
# ==============================
def t_test_numeric_by_group(df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
    sub = df[[numeric_col, group_col]].dropna()
    levels = sub[group_col].astype(str).unique()
    if len(levels) != 2:
        raise ValueError(f"T-test needs exactly 2 groups; found {len(levels)} unique values in {group_col}.")
    g1, g2 = levels[:2]
    x = sub[sub[group_col].astype(str) == g1][numeric_col].astype(float)
    y = sub[sub[group_col].astype(str) == g2][numeric_col].astype(float)
    tstat, pval = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')
    return {
        'groups': (str(g1), str(g2)),
        'means': (float(np.nanmean(x)), float(np.nanmean(y))),
        'tstat': float(tstat),
        'pval': float(pval),
        'n': (int(x.size), int(y.size)),
        'interpretation': f"P-value = {pval:.4g}. {'Difference is statistically significant.' if pval < 0.05 else 'No statistically significant difference at 5% level.'}"
    }

def ols_regression(df: pd.DataFrame, y: str, X: List[str]) -> Dict[str, Any]:
    data = df[[y] + X].dropna().copy()
    formula = f"{y} ~ " + " + ".join(X)
    model = smf.ols(formula, data=data).fit()
    return {
        'formula': formula,
        'summary_text': model.summary().as_text(),
        'coeff_table': model.summary2().tables[1].reset_index().rename(columns={'index':'term'}),
        'r2': float(model.rsquared),
        'adj_r2': float(model.rsquared_adj)
    }

def logistic_regression(df: pd.DataFrame, y: str, X: List[str]) -> Dict[str, Any]:
    data = df[[y] + X].dropna().copy()
    if data[y].nunique() != 2:
        raise ValueError("Logistic regression requires a binary target variable (two unique values).")
    if set(data[y].unique()) != {0, 1}:
        mapping = {val: i for i, val in enumerate(sorted(data[y].unique()))}
        data[y] = data[y].map(mapping)
    formula = f"{y} ~ " + " + ".join(X)
    model = smf.logit(formula, data=data).fit(disp=False)
    return {
        'formula': formula,
        'summary_text': model.summary().as_text(),
        'coeff_table': model.summary2().tables[1].reset_index().rename(columns={'index':'term'}),
        'pseudo_r2': float(model.prsquared)
    }

def anova_test(df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
    sub = df[[numeric_col, group_col]].dropna()
    groups = [sub[sub[group_col] == level][numeric_col].values for level in sub[group_col].unique()]
    f_stat, p_val = stats.f_oneway(*groups)
    return {
        "f_stat": float(f_stat),
        "p_value": float(p_val),
        "interpretation": f"P-value = {p_val:.4g}. {'There is a significant difference between groups.' if p_val < 0.05 else 'No significant difference between groups at 5% level.'}"
    }

def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "expected": expected,
        "contingency_table": contingency,
        "interpretation": f"P-value = {p:.4g}. {'Variables are dependent.' if p < 0.05 else 'No significant association between variables.'}"
    }

# ==============================
# Visualization
# ==============================
def plot_histogram(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    df[col].dropna().astype(float).plot(kind='hist', ax=ax)
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    return fig

def plot_scatter(df: pd.DataFrame, x: str, y: str):
    fig, ax = plt.subplots()
    ax.scatter(df[x], df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    return fig

def correlation_matrix(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Matrix", pad=20)
    return fig, corr

# ==============================
# Export Functions
# ==============================
def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def export_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def export_plot(fig) -> bytes:
    output = io.BytesIO()
    fig.savefig(output, format="png")
    return output.getvalue()

# ==============================
# Machine Learning
# ==============================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import seaborn as sns

# Classification
def classification_model(df: pd.DataFrame, y: str, X: List[str], model_type: str = "logistic") -> Dict[str, Any]:
    X_data = df[X].select_dtypes(include=[np.number]).dropna()
    y_data = df[y].iloc[X_data.index]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": model_type,
        "accuracy": accuracy_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True)
    }

# Regression
def regression_model(df: pd.DataFrame, y: str, X: List[str], model_type: str = "linear") -> Dict[str, Any]:
    X_data = df[X].select_dtypes(include=[np.number]).dropna()
    y_data = df[y].iloc[X_data.index]

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    if model_type == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": model_type,
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "mse": float(mean_squared_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds)))  # fixed here
    }

# Clustering
def clustering_model(df: pd.DataFrame, features: List[str], n_clusters: int = 3) -> Dict[str, Any]:
    X_data = df[features].select_dtypes(include=[np.number]).dropna()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_data)

    df_result = df.copy()
    df_result["Cluster"] = clusters

    fig, ax = plt.subplots()
    if len(features) >= 2:
        sns.scatterplot(x=X_data[features[0]], y=X_data[features[1]], hue=clusters, palette="tab10", ax=ax)
        ax.set_title("K-Means Clustering")
    else:
        ax.hist(clusters)
        ax.set_title("Cluster Distribution")

    return {
        "centers": kmeans.cluster_centers_.tolist(),
        "labels": clusters.tolist(),
        "figure": fig
    }

def plot_boxplot(df: pd.DataFrame, col: str, group_col: Optional[str] = None):
    """
    Create a boxplot for a numeric column, optionally grouped by another column.

    Parameters:
    df : pd.DataFrame
        The dataset
    col : str
        The numeric column to plot
    group_col : Optional[str]
        Column to group by (e.g. category)

    Returns:
    fig : matplotlib.figure.Figure
        The boxplot figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    if group_col and group_col in df.columns:
        df.boxplot(column=col, by=group_col, ax=ax)
        ax.set_title(f"Boxplot of {col} by {group_col}")
        plt.suptitle("")  # hufuta title ya default ya pandas boxplot
    else:
        df.boxplot(column=col, ax=ax)
        ax.set_title(f"Boxplot of {col}")
    ax.set_ylabel(col)
    return fig

def plot_line(df: pd.DataFrame, x: str, y: str):
    """
    Create a line plot between two columns.

    Parameters:
    df : pd.DataFrame
        Dataset
    x : str
        Column to use for X-axis
    y : str
        Column to use for Y-axis

    Returns:
    fig : matplotlib.figure.Figure
        Line plot figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df[x], df[y], marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Line Plot of {y} over {x}")
    return fig
