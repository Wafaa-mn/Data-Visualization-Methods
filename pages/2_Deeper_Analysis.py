import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    require_data, NUMERIC_COLS, CATEGORICAL_COLS, COLOR_SEQUENCE, style_fig,
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


# Numerical features by churn status
st.divider()
st.subheader("2.Numerical Features by Churn Status")

num_cols = [c for c in NUMERIC_COLS if c in df.columns]

selected_box_num = st.selectbox(
    "Choose a numerical feature to compare",
    num_cols,
    key="deep_box_choice"
)

fig = px.box(
    df,
    x="Churn",
    y=selected_box_num,
    color="Churn",
    color_discrete_map={
        "No": COLOR_SEQUENCE[0],
        "Yes": COLOR_SEQUENCE[1]
    },
    points="outliers",
    title=f"{selected_box_num} Distribution by Churn Status",
    labels={
        "Churn": "Churn Status",
        selected_box_num: selected_box_num
    },
)

fig.update_layout(
    xaxis_title="Churn Status",
    yaxis_title=selected_box_num,
    boxmode="group",
)

st.plotly_chart(
    style_fig(fig, 450),
    use_container_width=True
)

st.caption(
    "Compare the median, spread, and outliers of each numerical feature "
    "between customers who churned and those who stayed."
)


# Tenure and churn
st.subheader("3. Tenure and customer retention")
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

st.divider()
st.subheader("4. Service Bundling & Retention Mosaic Chart")
st.write(
    "Investigate which combinations of internet service and add-on features retain customers best. "
    "Bar width represents customer volume, while bar height shows the exact retention vs. churn split."
)

addon_cols = [
    c for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"] if c in df.columns
]

# Filter to internet subscribers so add-on bundling comparisons are meaningful
internet_df = df[df["InternetService"] != "No"].copy() if "InternetService" in df.columns else df.copy()

if addon_cols and "InternetService" in internet_df.columns:
    internet_df["AddonCount"] = internet_df[addon_cols].eq("Yes").sum(axis=1)
    internet_df["BundleTier"] = pd.cut(
        internet_df["AddonCount"],
        bins=[-1, 0, 2, 6],
        labels=["0 Add-ons", "1–2 Add-ons", "3–6 Add-ons"],
    )

    b_col1, b_col2 = st.columns([1, 1.2])
    with b_col1:
        bundle_mode = st.radio(
            "Bundling perspective",
            ["Internet + Specific Add-on", "Internet + Total Add-on Tier"],
            horizontal=True,
            key="bundle_mode",
        )
    with b_col2:
        if bundle_mode == "Internet + Specific Add-on":
            selected_addon = st.selectbox(
                "Choose add-on service",
                addon_cols,
                index=addon_cols.index("TechSupport") if "TechSupport" in addon_cols else 0,
                key="mosaic_addon",
            )
            combo = (
                internet_df["InternetService"].astype(str)
                + " | "
                + selected_addon
                + ": "
                + internet_df[selected_addon].astype(str)
            )
            chart_title = f"Service Bundling Mosaic: Internet + {selected_addon}"
        else:
            combo = (
                internet_df["InternetService"].astype(str)
                + " | "
                + internet_df["BundleTier"].astype(str)
            )
            chart_title = "Service Bundling Mosaic: Internet + Add-on Bundle Size"

    ct = pd.crosstab(combo, internet_df["Churn"])[["No", "Yes"]]
    # Sort by retention rate ("No" %) descending so the best-retaining bundles appear first
    retention_order = (ct["No"] / ct.sum(axis=1)).sort_values(ascending=False).index
    ct = ct.loc[retention_order]

    widths = ct.sum(axis=1).values
    props = ct.div(ct.sum(axis=1), axis=0).mul(100).round(1)
    x_pos = np.cumsum(widths) - widths / 2

    fig_bundle = go.Figure()
    for status, label_name, color in [
        ("No", "Retained (No Churn)", COLOR_SEQUENCE[0]),
        ("Yes", "Churned (Yes)", COLOR_SEQUENCE[1]),
    ]:
        fig_bundle.add_trace(
            go.Bar(
                name=label_name,
                x=x_pos,
                y=props[status],
                width=widths,
                marker_color=color,
                marker_line=dict(color="white", width=1.5),
                text=[f"{p:.0f}%" if w > widths.sum() * 0.05 else "" for p, w in zip(props[status], widths)],
                textposition="inside",
                customdata=np.stack([ct.index, ct[status], widths], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{label_name}: %{{y:.1f}}% (%{{customdata[1]:,}} customers)<br>"
                    "Bundle total: %{customdata[2]:,}<extra></extra>"
                ),
            )
        )

    fig_bundle.update_layout(
        barmode="stack",
        title=f"{chart_title} (Width = Customers, Height = Retention/Churn %)",
        yaxis=dict(title="Share of bundle (%)", range=[0, 100]),
        xaxis=dict(
            title="Service bundles ordered from highest retention to lowest retention",
            tickvals=x_pos,
            ticktext=ct.index,
            tickangle=-20,
        ),
    )
    st.plotly_chart(style_fig(fig_bundle, 480), use_container_width=True)
    st.caption(
        "Excludes customers without internet service. Bundles are ordered left-to-right from highest retention rate to highest churn rate."
    )

# Customer profile explorer (transparent filtering, not a model)
st.divider()
st.subheader("5. Customer profile explorer")
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
st.subheader("6. Comparing selected customer profiles")
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
    # Add average monthly charge per profile
    avg_charges = [round(df[mask]["MonthlyCharges"].mean(), 2) for _, mask in profile_definitions]
    profile_df["Avg Monthly Charge ($)"] = avg_charges

    fig_bubble = px.scatter(
        profile_df,
        x="Avg Monthly Charge ($)",
        y="Churn rate (%)",
        size="Customers",
        color="Profile",
        text="Profile",
        size_max=55,
        color_discrete_sequence=COLOR_SEQUENCE,
        hover_data=["Customers", "Churned", "Share of dataset (%)"],
        title="Segment Risk vs. Revenue (Bubble Size = Customer Count)",
    )
    fig_bubble.add_hline(y=overall_rate, line_dash="dash", line_color="#315C8C",
                         annotation_text=f"Overall: {overall_rate:.1f}%")
    fig_bubble.update_traces(textposition="top center")
    st.plotly_chart(style_fig(fig_bubble, 440), use_container_width=True)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)
st.divider()

# Business interpretation, grounded in the currently loaded data
st.subheader("7. Business interpretation")
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
