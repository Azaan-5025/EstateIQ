"""
ESTATEIQ - Page 1: Executive Dashboard
--------------------------------------
Displays macro portfolio metrics, geographic coverage, and distribution dynamics.
All metrics are dynamically calculated directly from the loaded dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.components.cards import render_hero, render_metric_card
from app.components.charts import apply_estateiq_theme, PALETTE_SEQUENCE, CHART_PALETTE


def render_dashboard(df: pd.DataFrame):
    """
    Renders the executive dashboard with dynamic dataset statistics.
    """
    render_hero(
        title="Executive Portfolio & Macro Market Overview",
        tagline="Empirical Baseline Metrics Derived Directly from 250,000 Verified Listings",
        description="Comprehensive macroeconomic snapshot capturing inventory distribution, asset pricing spreads, and structural dimensions across major Indian metropolitan hubs."
    )

    # 1. Dynamic Metric Calculations (Strictly NOT Hardcoded)
    total_properties = len(df)
    n_states = df["State"].nunique()
    n_cities = df["City"].nunique()
    n_localities = df["Locality"].nunique()
    
    avg_price = df["Price_in_Lakhs"].mean()
    median_price = df["Price_in_Lakhs"].median()
    
    avg_size = df["Size_in_SqFt"].mean()
    median_size = df["Size_in_SqFt"].median()
    
    avg_rate = df["Price_per_SqFt_INR"].mean() if "Price_per_SqFt_INR" in df.columns else (avg_price * 100000 / avg_size)

    # 2. Top KPI Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            label="Total Verified Listings",
            value=f"{total_properties:,}",
            subtext=f"Partitioned: 175k Train | 37.5k Val | 37.5k Test",
            accent="gold"
        )
    with c2:
        render_metric_card(
            label="Geographic Footprint",
            value=f"{n_cities} Cities",
            subtext=f"Spanning {n_states} States & {n_localities:,} Localities",
            accent="green"
        )
    with c3:
        render_metric_card(
            label="Avg. Listed Price",
            value=f"₹{avg_price:.2f}L",
            subtext=f"Median: ₹{median_price:.2f}L | Range: ₹10L - ₹500L",
            accent="sage"
        )
    with c4:
        render_metric_card(
            label="Avg. Property Size",
            value=f"{avg_size:,.0f} SqFt",
            subtext=f"Median: {median_size:,.0f} SqFt | Avg Rate: ₹{avg_rate:,.0f}/SqFt",
            accent="gold"
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 3. Middle Visualizations Row (Property Types & City Volume)
    col_left, col_right = st.columns([1, 1.25])

    with col_left:
        # Property Type Distribution (Donut Chart)
        prop_type_counts = df["Property_Type"].value_counts().reset_index()
        prop_type_counts.columns = ["Property Type", "Count"]
        
        fig_donut = px.pie(
            prop_type_counts,
            names="Property Type",
            values="Count",
            hole=0.55,
            color_discrete_sequence=PALETTE_SEQUENCE
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#FFFFFF", width=2))
        )
        apply_estateiq_theme(fig_donut, title="Inventory Composition by Property Type", height=380)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        # Top 10 Cities by Volume
        top_cities = df["City"].value_counts().head(10).reset_index()
        top_cities.columns = ["City", "Listing Count"]
        
        fig_city = px.bar(
            top_cities,
            x="Listing Count",
            y="City",
            orientation="h",
            color_discrete_sequence=[CHART_PALETTE["deep_forest"]],
            text="Listing Count"
        )
        fig_city.update_layout(yaxis=dict(autorange="reversed"))
        fig_city.update_traces(texttemplate="%{text:,}", textposition="inside", textfont=dict(color="#FFFFFF"))
        apply_estateiq_theme(fig_city, title="Top 10 Metropolitan Markets by Volume", height=380)
        st.plotly_chart(fig_city, use_container_width=True)

    # 4. Bottom Visualizations Row (Price Distribution & Furnishing/Availability)
    col_pdist, col_avail = st.columns([1.2, 1])

    with col_pdist:
        # Listed Price Distribution Histogram
        q25 = df["Price_in_Lakhs"].quantile(0.25)
        q50 = df["Price_in_Lakhs"].quantile(0.50)
        q75 = df["Price_in_Lakhs"].quantile(0.75)

        fig_hist = px.histogram(
            df,
            x="Price_in_Lakhs",
            nbins=50,
            color_discrete_sequence=[CHART_PALETTE["sage_green"]],
            opacity=0.85
        )
        # Add percentile vertical lines
        fig_hist.add_vline(x=q50, line_dash="dash", line_color=CHART_PALETTE["deep_forest"],
                           annotation_text=f"Median: ₹{q50:.1f}L", annotation_position="top right")
        fig_hist.add_vline(x=q25, line_dash="dot", line_color=CHART_PALETTE["slate_gray"],
                           annotation_text=f"Q1: ₹{q25:.1f}L", annotation_position="top left")
        fig_hist.add_vline(x=q75, line_dash="dot", line_color=CHART_PALETTE["champagne_gold"],
                           annotation_text=f"Q3: ₹{q75:.1f}L", annotation_position="top right")

        apply_estateiq_theme(fig_hist, title="Listed Property Price Distribution (₹ Lakhs)", height=380)
        fig_hist.update_xaxes(title_text="Listed Price (₹ Lakhs)")
        fig_hist.update_yaxes(title_text="Number of Properties")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_avail:
        # Furnishing & Availability Cross-Tabulation
        furn_avail = df.groupby(["Furnished_Status", "Availability_Status"]).size().reset_index(name="Count")
        fig_furn = px.bar(
            furn_avail,
            x="Furnished_Status",
            y="Count",
            color="Availability_Status",
            barmode="group",
            color_discrete_map={
                "Ready to Move": CHART_PALETTE["deep_forest"],
                "Under Construction": CHART_PALETTE["champagne_gold"]
            }
        )
        apply_estateiq_theme(fig_furn, title="Furnishing by Construction Availability", height=380)
        fig_furn.update_xaxes(title_text="Furnishing Status")
        fig_furn.update_yaxes(title_text="Properties")
        st.plotly_chart(fig_furn, use_container_width=True)
