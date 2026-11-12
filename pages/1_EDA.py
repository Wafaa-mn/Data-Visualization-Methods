import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    require_data, NUMERIC_COLS, CATEGORICAL_COLS, COLOR_SEQUENCE,
    CHART_TEMPLATE, style_fig, kpi_row, category_churn_table
)

st.set_page_config(page_title="EDA | Telecom Churn", page_icon="🔎", layout="wide")
df = require_data()

st.title("Exploratory Data Analysis")
st.caption("Understand the structure, distributions, and relationships in the customer dataset.")

kpi_row(df)
st.divider()

# Dataset overview
st.subheader("1. Dataset overview")
a, b = st.columns([1.15, 1])
with a:
    st.write("**Preview of the cleaned dataset**")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
with b:
    overview = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.write("**Data quality and column summary**")
    st.dataframe(overview, use_container_width=True, hide_index=True, height=350)

st.divider()

# Numerical features
st.subheader("2. Numerical feature distributions")
num_cols = [c for c in NUMERIC_COLS if c in df.columns]
selected_num = st.selectbox("Choose a numerical variable", num_cols)
fig = px.histogram(
    df, x=selected_num, nbins=35, marginal="box",
    color_discrete_sequence=[COLOR_SEQUENCE[0]],
    title=f"Distribution of {selected_num}",
)
st.plotly_chart(style_fig(fig, 430), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    fig = px.box(
        df, x="Churn", y=selected_num, color="Churn",
        color_discrete_map={"No": COLOR_SEQUENCE[0], "Yes": COLOR_SEQUENCE[1]},
        points="outliers", title=f"{selected_num} by churn status",
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)
with c2:
    if len(num_cols) >= 2:
        xvar = st.selectbox("X-axis", num_cols, index=0, key="eda_x")
        yvar = st.selectbox("Y-axis", num_cols, index=min(2, len(num_cols)-1), key="eda_y")
        fig = px.scatter(
            df, x=xvar, y=yvar, color="Churn", opacity=0.55,
            color_discrete_map={"No": COLOR_SEQUENCE[0], "Yes": COLOR_SEQUENCE[1]},
            title=f"{yvar} vs. {xvar}",
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

corr = df[num_cols].corr(numeric_only=True)
fig = px.imshow(
    corr, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
    title="Correlation between numerical variables",
)
st.plotly_chart(style_fig(fig, 350), use_container_width=True)

st.divider()

# Categorical distributions
st.subheader("3. Categorical feature composition")
cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
cat_choice = st.selectbox("Choose a categorical variable", cat_cols)
composition = (
    df[cat_choice].fillna("Missing").value_counts()
    .rename_axis(cat_choice).reset_index(name="Customers")
)
fig = px.bar(
    composition, x=cat_choice, y="Customers",
    color="Customers", color_continuous_scale="Blues",
    title=f"Customer distribution by {cat_choice}",
    text="Customers",
)
fig.update_layout(coloraxis_showscale=False)
st.plotly_chart(style_fig(fig), use_container_width=True)

st.divider()

# Churn bivariate analysis
st.subheader("4. Categorical variables and churn")
cat_choice_churn = st.selectbox(
    "Compare churn across", cat_cols, index=cat_cols.index("Contract") if "Contract" in cat_cols else 0
)
summary = category_churn_table(df, cat_choice_churn)
fig = px.bar(
    summary.sort_values("Churn_Rate"),
    x="Churn_Rate", y=cat_choice_churn, orientation="h",
    text="Churn_Rate", color="Churn_Rate", color_continuous_scale="OrRd",
    title=f"Churn rate by {cat_choice_churn}",
    labels={"Churn_Rate": "Churn rate (%)"},
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_layout(coloraxis_showscale=False, xaxis_title="Churn rate (%)", yaxis_title="")
st.plotly_chart(style_fig(fig, 450), use_container_width=True)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()

# Relationships between categorical variables
st.subheader("5. Relationships between categorical variables")
pair_options = [
    ("Contract", "PaymentMethod"),
    ("InternetService", "OnlineSecurity"),
    ("Contract", "InternetService"),
    ("PaymentMethod", "PaperlessBilling"),
]
available_pairs = [(x, y) for x, y in pair_options if x in df.columns and y in df.columns]
if available_pairs:
    pair = st.selectbox(
        "Select a pair of variables",
        available_pairs,
        format_func=lambda x: f"{x[0]} vs. {x[1]}",
    )
    cross = pd.crosstab(df[pair[0]], df[pair[1]])
    fig = px.imshow(
        cross, text_auto=True, aspect="auto", color_continuous_scale="YlGnBu",
        labels=dict(x=pair[1], y=pair[0], color="Customers"),
        title=f"Customer counts: {pair[0]} vs. {pair[1]}",
    )
    st.plotly_chart(style_fig(fig, 430), use_container_width=True)

st.caption(
    "EDA note: percentages and counts describe the observed dataset. "
    "Associations do not establish causation."
)
