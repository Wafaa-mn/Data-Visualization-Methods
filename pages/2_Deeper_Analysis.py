import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import (
    require_data, CATEGORICAL_COLS, COLOR_SEQUENCE, style_fig,
    kpi_row, churn_rate, category_churn_table
)

st.set_page_config(page_title="Deeper Analysis | Telecom Churn", page_icon="📈", layout="wide")
df = require_data()

st.title("Deeper Churn Analysis")
st.caption(
    "A descriptive investigation of churn patterns and customer profiles. "
    "All comparisons are calculated directly from the observed data; no predictive model is used."
)
kpi_row(df)
overall_rate = churn_rate(df)
st.divider()

# Churn comparison across categories
st.subheader("1. Which customer characteristics are associated with higher churn?")
available_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
selected_factors = st.multiselect(
    "Choose factors to compare",
    available_cats,
    default=[c for c in ["Contract", "PaymentMethod", "InternetService", "PaperlessBilling"] if c in available_cats],
)
if selected_factors:
    factor_tabs = st.tabs(selected_factors)
    for tab, factor in zip(factor_tabs, selected_factors):
        with tab:
            summary = category_churn_table(df, factor)
            fig = px.bar(
                summary.sort_values("Churn_Rate"),
                x="Churn_Rate", y=factor, orientation="h",
                color="Churn_Rate", color_continuous_scale="OrRd",
                text="Churn_Rate", title=f"Churn rate by {factor}",
                hover_data=["Customers", "Customer_Share", "Churned"],
            )
            fig.add_vline(
                x=overall_rate, line_dash="dash", line_color="#315C8C",
                annotation_text=f"Overall: {overall_rate:.1f}%",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Churn rate (%)", yaxis_title="")
            st.plotly_chart(style_fig(fig, 390), use_container_width=True)
            st.dataframe(summary, use_container_width=True, hide_index=True)
else:
    st.info("Select at least one categorical factor to compare.")

st.divider()

# Tenure and churn
st.subheader("2. Tenure and customer retention")
if "TenureGroup" in df.columns:
    tenure_summary = (
        df.groupby("TenureGroup", observed=False)
        .agg(
            Customers=("Churn", "size"),
            Churn_Rate=("Churn", lambda s: 100 * s.eq("Yes").mean()),
        ).reset_index()
    )
    tenure_summary["Churn_Rate"] = tenure_summary["Churn_Rate"].round(1)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            tenure_summary, x="TenureGroup", y="Churn_Rate",
            text="Churn_Rate", color="Churn_Rate", color_continuous_scale="OrRd",
            title="Churn rate across tenure groups",
        )
        fig.add_hline(y=overall_rate, line_dash="dash", line_color="#315C8C")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Churn rate (%)")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        fig = px.bar(
            tenure_summary, x="TenureGroup", y="Customers",
            text="Customers", color="Customers", color_continuous_scale="Blues",
            title="Number of customers across tenure groups",
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Customers")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    st.dataframe(tenure_summary, use_container_width=True, hide_index=True)

#new
st.divider()
st.subheader("Churn Rate Across Customer Tenure")

tenure_churn = (
    df.groupby("tenure")
    .agg(churn_rate=("Churn", lambda x: x.eq("Yes").mean() * 100))
    .reset_index()
    .sort_values("tenure")
)

fig = px.area(
    tenure_churn,
    x="tenure",
    y="churn_rate",
    labels={
        "tenure": "Tenure (months)",
        "churn_rate": "Churn rate (%)"
    },
    title="Observed Churn Rate by Customer Tenure"
)

fig.update_traces(
    line=dict(width=2),
    fill="tozeroy"
)

st.plotly_chart(style_fig(fig, 450), use_container_width=True)

st.caption(
    "Shows the observed churn rate among customers at each tenure value. "
    "Rates at individual tenure values may be based on relatively few customers."
)

# Contract and payment method profile
st.divider()

# 3. Customer relationship matrix
st.divider()
st.subheader("3. Customer Relationship Matrix")

st.write(
    "Explore how combinations of customer characteristics relate to "
    "customer churn. Each cell represents either the number of customers "
    "or the observed churn rate for that combination."
)

col1, col2, col3 = st.columns(3)

with col1:
    row_var = st.selectbox(
        "Rows",
        available_cats,
        index=available_cats.index("Contract")
        if "Contract" in available_cats else 0,
        key="matrix_rows"
    )

with col2:
    col_options = [c for c in available_cats if c != row_var]
    col_var = st.selectbox(
        "Columns",
        col_options,
        index=col_options.index("PaymentMethod")
        if "PaymentMethod" in col_options else 0,
        key="matrix_cols"
    )

with col3:
    matrix_metric = st.selectbox(
        "Display metric",
        ["Churn rate (%)", "Customer count"],
        key="matrix_metric"
    )

if matrix_metric == "Churn rate (%)":
    matrix = pd.pivot_table(
        df,
        index=row_var,
        columns=col_var,
        values="Churn",
        aggfunc=lambda x: 100 * x.eq("Yes").mean()
    )

    color_label = "Churn rate (%)"
    text_format = ".1f"

else:
    matrix = pd.crosstab(
        df[row_var],
        df[col_var]
    )

    color_label = "Customers"
    text_format = True

fig = px.imshow(
    matrix,
    text_auto=text_format,
    aspect="auto",
    color_continuous_scale="OrRd"
    if matrix_metric == "Churn rate (%)"
    else "Blues",
    labels={
        "x": col_var,
        "y": row_var,
        "color": color_label
    },
    title=f"{matrix_metric}: {row_var} vs. {col_var}"
)

st.plotly_chart(
    style_fig(fig, 500),
    use_container_width=True
)

st.caption(
    "Each cell represents a combination of two customer characteristics. "
    "Churn rates are descriptive and do not establish causation."
)


# Customer profile explorer (transparent filtering, not a model)
st.divider()
st.subheader("4. Customer profile explorer")
st.write(
    "Combine customer characteristics to inspect a profile and compare its observed churn rate "
    "with the overall dataset. This is a filter-based descriptive analysis, not a risk prediction."
)

filter_cols = st.columns(3)
profile_filters = {}
filter_candidates = ["Contract", "PaymentMethod", "InternetService", "PaperlessBilling", "OnlineSecurity"]
for i, col in enumerate(filter_candidates):
    if col in df.columns:
        options = ["All"] + sorted(df[col].dropna().astype(str).unique().tolist())
        profile_filters[col] = filter_cols[i % 3].selectbox(
            col, options, key=f"profile_{col}"
        )

filtered = df.copy()
for col, value in profile_filters.items():
    if value != "All":
        filtered = filtered[filtered[col].astype(str) == value]

min_tenure, max_tenure = int(df["tenure"].min()), int(df["tenure"].max())
tenure_range = st.slider(
    "Tenure range (months)", min_value=min_tenure, max_value=max_tenure,
    value=(min_tenure, max_tenure)
)
filtered = filtered[filtered["tenure"].between(*tenure_range)]

p1, p2, p3 = st.columns(3)
p1.metric("Customers in selected profile", f"{len(filtered):,}")
p2.metric("Profile churn rate", f"{churn_rate(filtered):.1f}%" if len(filtered) else "—")
difference = churn_rate(filtered) - overall_rate if len(filtered) else 0
p3.metric("Difference vs. overall", f"{difference:+.1f} pp" if len(filtered) else "—")

if len(filtered):
    status_counts = filtered["Churn"].value_counts().rename_axis("Churn").reset_index(name="Customers")
    fig = px.pie(
        status_counts, names="Churn", values="Customers", hole=0.55,
        color="Churn", color_discrete_map={"No": COLOR_SEQUENCE[0], "Yes": COLOR_SEQUENCE[1]},
        title="Churn composition of selected profile",
    )
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)
else:
    st.warning("No customers match the selected combination. Broaden the filters.")

st.divider()

# Rule-based profile comparisons inspired by notebook's selected factors, not clustering.
st.subheader("5. Comparing selected customer profiles")
st.write(
    "The notebook's segmentation work highlighted combinations involving contract type, "
    "payment method, internet service, tenure, and paperless billing. The profiles below "
    "compare these combinations directly, without clustering customers or assigning model-generated labels."
)

profile_definitions = []
if all(c in df.columns for c in ["Contract", "PaymentMethod", "InternetService"]):
    profile_definitions = [
        ("Month-to-month + electronic check",
         (df["Contract"] == "Month-to-month") & (df["PaymentMethod"] == "Electronic check")),
        ("Fiber optic + month-to-month",
         (df["InternetService"] == "Fiber optic") & (df["Contract"] == "Month-to-month")),
        ("Two-year contract",
         df["Contract"] == "Two year"),
    ]
    if "PaperlessBilling" in df.columns:
        profile_definitions.append((
            "Month-to-month + paperless billing",
            (df["Contract"] == "Month-to-month") & (df["PaperlessBilling"] == "Yes")
        ))

rows = []
for label, mask in profile_definitions:
    group = df[mask]
    rows.append({
        "Profile": label,
        "Customers": len(group),
        "Churned": int(group["Churn"].eq("Yes").sum()),
        "Churn rate (%)": round(churn_rate(group), 1),
        "Share of dataset (%)": round(100 * len(group) / len(df), 1),
    })
if rows:
    profile_df = pd.DataFrame(rows).sort_values("Churn rate (%)", ascending=False)
    fig = px.bar(
        profile_df.sort_values("Churn rate (%)"),
        x="Churn rate (%)", y="Profile", orientation="h",
        color="Churn rate (%)", color_continuous_scale="OrRd",
        text="Churn rate (%)", hover_data=["Customers", "Churned", "Share of dataset (%)"],
        title="Observed churn rate across selected profiles",
    )
    fig.add_vline(x=overall_rate, line_dash="dash", line_color="#315C8C",
                  annotation_text=f"Overall: {overall_rate:.1f}%")
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(style_fig(fig, 400), use_container_width=True)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

st.divider()

# Business interpretation, grounded in the currently loaded data
st.subheader("6. Business interpretation")
st.markdown(
    """
    Use the comparisons above to frame retention recommendations. A higher observed churn
    rate identifies a group worth investigating; it does **not** prove that the characteristic
    causes churn.
    """
)
if "Contract" in df.columns:
    contract_summary = category_churn_table(df, "Contract")
    highest_contract = contract_summary.iloc[0]
    st.markdown(
        f"- **Contract strategy:** The highest observed churn rate is for "
        f"**{highest_contract['Contract']}** customers "
        f"({highest_contract['Churn_Rate']:.1f}%). Consider testing retention offers or "
        f"contract-transition incentives for this group."
    )
if "PaymentMethod" in df.columns:
    payment_summary = category_churn_table(df, "PaymentMethod")
    highest_payment = payment_summary.iloc[0]
    st.markdown(
        f"- **Payment experience:** **{highest_payment['PaymentMethod']}** has the highest "
        f"observed churn rate ({highest_payment['Churn_Rate']:.1f}%). Investigate payment "
        f"friction, failed payments, and customer experience before assuming the method itself "
        f"is responsible."
    )
if "tenure" in df.columns:
    early = df[df["tenure"] <= 12]
    st.markdown(
        f"- **Early customer journey:** Customers with 12 months or less tenure have an "
        f"observed churn rate of {churn_rate(early):.1f}%. This supports examining onboarding, "
        f"early engagement, and first-year service experience."
    )

st.caption(
    "Interpretation boundary: all findings are descriptive associations in this dataset. "
    "No causal inference, customer-level prediction, or machine-learning scoring is performed."
)
