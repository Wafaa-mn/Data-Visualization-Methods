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

selected_num = st.selectbox(
    "Choose a numerical variable",
    num_cols,
    key="eda_num_choice"
)

# Numerical distribution
fig = px.histogram(
    df,
    x=selected_num,
    nbins=35,
    marginal="box",
    color_discrete_sequence=[COLOR_SEQUENCE[0]],
    title=f"Distribution of {selected_num}",
)

st.plotly_chart(
    style_fig(fig, 430),
    use_container_width=True
)

# Total Charges vs. Tenure
plot_sample = df.sample(
    min(len(df), 1200),
    random_state=42
)

fig = px.scatter(
    plot_sample,
    x="tenure",
    y="TotalCharges",
    color="Churn",
    symbol="Contract" if "Contract" in df.columns else None,
    opacity=0.65,
    color_discrete_map={
        "No": COLOR_SEQUENCE[0],
        "Yes": COLOR_SEQUENCE[1]
    },
    labels={
        "tenure": "Tenure (months)",
        "TotalCharges": "Total Charges (USD)",
    },
    title="Total Charges vs. Tenure (Color = Churn, Shape = Contract)",
)

st.plotly_chart(
    style_fig(fig),
    use_container_width=True
)



# Bubble correlation matrix
corr = df[num_cols].corr(numeric_only=True)

bubble_data = (
    corr.stack()
    .reset_index()
)
bubble_data.columns = ["Variable 1", "Variable 2", "Correlation"]
bubble_data["AbsCorrelation"] = bubble_data["Correlation"].abs()

fig = px.scatter(
    bubble_data,
    x="Variable 1",
    y="Variable 2",
    size="AbsCorrelation",
    color="Correlation",
    color_continuous_scale="RdBu",
    range_color=[-1, 1],
    size_max=45,
    text=bubble_data["Correlation"].map(lambda x: f"{x:.2f}"),
    title="Correlation Between Numerical Variables",
    labels={"Correlation": "Correlation"}
)

fig.update_traces(
    textposition="middle center",
    marker=dict(line=dict(width=0.5, color="white")),
    hovertemplate=(
        "%{x} vs. %{y}<br>"
        "Correlation: %{marker.color:.2f}"
        "<extra></extra>"
    )
)

fig.update_layout(
    xaxis_title="",
    yaxis_title="",
    yaxis=dict(autorange="reversed"),
    coloraxis_colorbar=dict(title="Correlation"),
    height=500
)

st.plotly_chart(style_fig(fig, 500), use_container_width=True)


st.divider()


# Categorical distributions - Sunburst
st.subheader("3. Categorical Feature Composition")

cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

sunburst_data = []

for feature in cat_cols:
    counts = df[feature].fillna("Missing").value_counts()

    for category, count in counts.items():
        sunburst_data.append({
            "Feature": feature,
            "Category": str(category),
            "Customers": int(count)
        })

sunburst_df = pd.DataFrame(sunburst_data)

fig = px.sunburst(
    sunburst_df,
    path=["Feature", "Category"],
    values="Customers",
    color="Customers",
    color_continuous_scale=[
    "#102A43",
    "#1D4E89",
    "#287D8E",
    "#43AA8B",
    "#A8DADC"
],
    title="Customer Distribution Across Categorical Features"
)

fig.update_traces(
    textinfo="label",
    insidetextorientation="radial",
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Customers: %{value:,}<br>"
        "Share of parent: %{percentParent:.1%}"
        "<extra></extra>"
    )
)

fig.update_layout(
    coloraxis_showscale=False
)

st.plotly_chart(
    style_fig(fig, 650),
    use_container_width=True
)

st.caption(
    "Click a feature to explore its categories. "
    "Category percentages represent the share of customers "
    "within the selected feature."
)

st.divider()




# Relationships between categorical variables
st.subheader("5. Relationships between categorical variables")

pair_options = [
    ("Contract", "PaymentMethod"),
    ("InternetService", "OnlineSecurity"),
    ("Contract", "InternetService"),
    ("PaymentMethod", "PaperlessBilling"),
]

available_pairs = [
    (x, y) for x, y in pair_options
    if x in df.columns and y in df.columns
]

if available_pairs:
    pair = st.selectbox(
        "Select a pair of variables",
        available_pairs,
        format_func=lambda x: f"{x[0]} vs. {x[1]}",
    )

    # Prepare customer counts
    cross = pd.crosstab(
        df[pair[0]].fillna("Missing"),
        df[pair[1]].fillna("Missing")
    )

    left_labels = [f"{pair[0]}: {x}" for x in cross.index]
    right_labels = [f"{pair[1]}: {x}" for x in cross.columns]
    labels = left_labels + right_labels

    sources = []
    targets = []
    values = []

    for i, row in enumerate(cross.index):
        for j, col in enumerate(cross.columns):
            count = int(cross.loc[row, col])

            if count > 0:
                sources.append(i)
                targets.append(len(left_labels) + j)
                values.append(count)

    # Color palette for the flows
    palette = [
        "#287D8E",
        "#43AA8B",
        "#1D4E89",
        "#E9A03B",
        "#8E6CBA",
        "#D16D8D",
    ]

    # Assign a color to each starting category
    left_colors = [
        palette[i % len(palette)]
        for i in range(len(left_labels))
    ]

    # Give each flow the color of its source category
    link_colors = [
        left_colors[source].replace("#", "")
        for source in sources
    ]

    # Convert colors to transparent RGBA
    link_colors = [
        f"rgba({int(c[0:2], 16)}, {int(c[2:4], 16)}, "
        f"{int(c[4:6], 16)}, 0.55)"
        for c in link_colors
    ]

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=25,
            thickness=22,
            label=labels,
            color=left_colors + ["#A8B5C2"] * len(right_labels),
            line=dict(color="white", width=0.5),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate=(
                "%{source.label} → %{target.label}<br>"
                "Customers: %{value:,}"
                "<extra></extra>"
            ),
        )
    )])

    fig.update_layout(
        title=f"Customer Relationships: {pair[0]} → {pair[1]}",
        height=500,
        font=dict(size=12),
    )

    st.plotly_chart(
        style_fig(fig, 500),
        use_container_width=True
    )

else:
    st.info("No valid categorical variable pairs are available.")

st.caption(
    "Flow width represents the number of customers sharing each "
    "combination of categories. Colors distinguish starting categories. "
    "These relationships are descriptive and do not establish causation."
)

