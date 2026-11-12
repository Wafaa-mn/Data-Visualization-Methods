import streamlit as st
from utils import render_sidebar
st.set_page_config(
    page_title="Telecom Customer Churn | Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Telecom Customer Churn Analysis")
st.markdown(
    """
    An interactive, **descriptive analytics** dashboard built around the Telco Customer Churn
    dataset. Explore the customer base, compare churn patterns, and investigate customer
    profiles using descriptive statistics and transparent segment comparisons.
    """
)

df = render_sidebar()
if df is not None:
    st.success(f"Dataset loaded: **{len(df):,} customers** and **{len(df.columns):,} columns**.")
    st.subheader("What's inside")
    left, right = st.columns(2)
    with left:
        st.markdown("### 1. Exploratory Data Analysis")
        st.write(
            "Dataset overview, distributions, numerical relationships, categorical "
            "composition, and churn comparisons."
        )
    with right:
        st.markdown("### 2. Deeper Analysis")
        st.write(
            "Churn-rate drivers, customer profile comparisons, tenure and contract "
            "patterns, and business implications."
        )
    st.info("Choose a page from the navigation sidebar to begin.")
else:
    st.info(
        "Upload the Telco Customer Churn CSV in the sidebar to start. "
        "The expected file is commonly named `WA_Fn-UseC_-Telco-Customer-Churn.csv`."
    )

st.caption(
    "Methodology: this dashboard uses descriptive statistics and aggregations only. "
    "It does not train or run machine-learning models."
)
