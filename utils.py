from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA_PATH = Path(__file__).parent / "Telco-Customer-Churn.csv"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

CHART_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = ["#315C8C", "#E07A5F", "#81B29A", "#F2CC8F", "#6D597A"]


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the uploaded source data using the notebook's documented preparation."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    required = {"Churn", "tenure", "MonthlyCharges", "TotalCharges"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # In the notebook, blank TotalCharges values for customers with zero tenure
    # are interpreted as zero total charges.
    zero_tenure = df["tenure"].eq(0)
    df.loc[zero_tenure & df["TotalCharges"].isna(), "TotalCharges"] = 0

    # Defensive fallback for any remaining missing numeric values.
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median) else median)

    for col in ["Churn"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map(
            lambda x: "Yes" if str(x).strip() in ["1", "1.0", "Yes", "yes"] else
            ("No" if str(x).strip() in ["0", "0.0", "No", "no"] else str(x))
        )

    # Remove fully empty rows and rows without a valid churn label.
    df = df.dropna(how="all")
    df = df[df["Churn"].isin(["Yes", "No"])].copy()

    if "tenure" in df.columns:
        bins = [-1, 12, 24, 48, 60, np.inf]
        labels = ["0–12 months", "13–24 months", "25–48 months", "49–60 months", "61+ months"]
        df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df


def render_sidebar():
    st.sidebar.header("Dataset")

    try:
        raw = pd.read_csv(DATA_PATH)
        df = prepare_data(raw)

        st.sidebar.success("Dataset loaded successfully")
        st.sidebar.caption(f"File: {DATA_PATH.name}")

        return df

    except Exception as exc:
        st.sidebar.error(f"Could not load dataset: {exc}")
        return None


def require_data():
    df = render_sidebar()
    if df is None:
        st.title("Dataset required")
        st.info("Upload the Telco Customer Churn CSV using the sidebar to view this page.")
        st.stop()
    return df


def churn_rate(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0
    return 100 * data["Churn"].eq("Yes").mean()


def kpi_row(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{churn_rate(df):.1f}%")
    c3.metric("Average tenure", f"{df['tenure'].mean():.1f} months")
    c4.metric("Average monthly charge", f"${df['MonthlyCharges'].mean():,.2f}")


def category_churn_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    result = (
        df.groupby(column, dropna=False)
        .agg(
            Customers=("Churn", "size"),
            Churned=("Churn", lambda s: s.eq("Yes").sum()),
            Churn_Rate=("Churn", lambda s: 100 * s.eq("Yes").mean()),
        )
        .reset_index()
        .sort_values("Churn_Rate", ascending=False)
    )
    result["Churn_Rate"] = result["Churn_Rate"].round(1)
    result["Customer_Share"] = (100 * result["Customers"] / len(df)).round(1)
    return result


def style_fig(fig, height=420):
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        font=dict(family="Arial", size=12),
        legend_title_text="",
    )
    return fig
