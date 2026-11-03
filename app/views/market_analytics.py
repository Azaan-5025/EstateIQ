"""
ESTATEIQ - Page 5: Market Analytics
-----------------------------------
Multi-dimensional market exploratory analytics with dynamic filtering
across states, cities, property types, and bedroom configurations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.components.cards import render_hero, render_metric_card
from app.components.charts import apply_estateiq_theme, PALETTE_SEQUENCE, CHART_PALETTE


def render_market_analytics(df: pd.DataFrame):
    """
    Renders the market analytics deep-dive with interactive cascading filters.
    """
    render_hero(
        title="Institutional Market Intelligence & Spatial Analytics",
        tagline="Empirical Multi-Dimensional Cross-Sectional Exploration",
        description="Filter and analyze pricing spreads, density metrics, recreational infrastructure, and asset depreciation tiers across major regional markets."
    )

    # 1. Interactive Filter Controls
    st.markdown("### 🔍 Market Segmentation Filters")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        states = ["All States"] + sorted(df["State"].dropna().unique().tolist())
        selected_state = st.selectbox("Filter State", options=states, index=0)

    # Cascading City filter
    with col_f2:
        if selected_state != "All States":
            available_cities = sorted(df[df["State"] == selected_state]["City"].dropna().unique().tolist())
        else:
            available_cities = sorted(df["City"].dropna().unique().tolist())
        cities = ["All Cities"] + available_cities
        selected_city = st.selectbox("Filter City", options=cities, index=0)

    with col_f3:
        prop_types = ["All Types"] + sorted(df["Property_Type"].dropna().unique().tolist())
        selected_type = st.selectbox("Property Type", options=prop_types, index=0)

    with col_f4:
        bhk_options = ["All BHKs"] + sorted(df["BHK"].dropna().unique().tolist())
        selected_bhk = st.selectbox("Bedrooms (BHK)", options=bhk_options, index=0)

    # Apply filters dynamically
    filtered_df = df.copy()
    if selected_state != "All States":
        filtered_df = filtered_df[filtered_df["State"] == selected_state]
    if selected_city != "All Cities":
        filtered_df = filtered_df[filtered_df["City"] == selected_city]
    if selected_type != "All Types":
        filtered_df = filtered_df[filtered_df["Property_Type"] == selected_type]
    if selected_bhk != "All BHKs":
        filtered_df = filtered_df[filtered_df["BHK"] == selected_bhk]

    # Filter summary badge
    st.caption(f"Showing **{len(filtered_df):,}** properties matching selected filter criteria (out of {len(df):,} total).")

    if len(filtered_df) == 0:
        st.warning("No properties match the selected filter combination. Please broaden your selection.")
        return

    # Filtered KPI Summary Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Filtered Properties", f"{len(filtered_df):,}", "Subset Size", accent="gold")
    with k2:
        render_metric_card("Avg. Listed Price", f"₹{filtered_df['Price_in_Lakhs'].mean():.2f}L", f"Median: ₹{filtered_df['Price_in_Lakhs'].median():.2f}L", accent="green")
    with k3:
        render_metric_card("Avg. Property Size", f"{filtered_df['Size_in_SqFt'].mean():,.0f} SqFt", f"Median: {filtered_df['Size_in_SqFt'].median():,.0f} SqFt", accent="sage")
    with k4:
        avg_sqft_rate = (filtered_df['Price_in_Lakhs'].mean() * 100000) / filtered_df['Size_in_SqFt'].mean()
        render_metric_card("Avg. Rate / SqFt", f"₹{avg_sqft_rate:,.0f}", "Unit Rate Proxy", accent="gold")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Row 1: State & City Listed Price Distributions
    tab1, tab2 = st.tabs(["🏙️ Pricing by Geography", "📐 Structural & Physical Dimensions"])

    with tab1:
        c_left, c_right = st.columns(2)
        with c_left:
            # City-wise median price comparison (top 15)
            city_pricing = (
                filtered_df.groupby("City")["Price_in_Lakhs"]
                .agg(["median", "count"])
                .reset_index()
                .sort_values(by="median", ascending=False)
                .head(15)
            )
            fig_city_p = px.bar(
                city_pricing,
                x="City",
                y="median",
                color_discrete_sequence=[CHART_PALETTE["deep_forest"]],
                text="median"
            )
            fig_city_p.update_traces(texttemplate="₹%{text:.1f}L", textposition="outside")
            apply_estateiq_theme(fig_city_p, title="Median Listed Price by City (Top 15)", height=420)
            fig_city_p.update_yaxes(title_text="Median Listed Price (₹ Lakhs)")
            st.plotly_chart(fig_city_p, use_container_width=True)

        with c_right:
            # State-wise property prices
            state_pricing = (
                filtered_df.groupby("State")["Price_in_Lakhs"]
                .median()
                .reset_index()
                .sort_values(by="Price_in_Lakhs", ascending=False)
                .head(12)
            )
            fig_state_p = px.bar(
                state_pricing,
                x="Price_in_Lakhs",
                y="State",
                orientation="h",
                color_discrete_sequence=[CHART_PALETTE["champagne_gold"]],
                text="Price_in_Lakhs"
            )
            fig_state_p.update_layout(yaxis=dict(autorange="reversed"))
            fig_state_p.update_traces(texttemplate="₹%{text:.1f}L", textposition="inside", textfont=dict(color="#172923"))
            apply_estateiq_theme(fig_state_p, title="Median Listed Price by State", height=420)
            fig_state_p.update_xaxes(title_text="Median Price (₹ Lakhs)")
            st.plotly_chart(fig_state_p, use_container_width=True)

    with tab2:
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            # Property Type Comparison
            type_comp = filtered_df.groupby("Property_Type")[["Price_in_Lakhs", "Size_in_SqFt"]].mean().reset_index()
            fig_pt = px.bar(
                type_comp,
                x="Property_Type",
                y="Size_in_SqFt",
                color="Property_Type",
                color_discrete_sequence=PALETTE_SEQUENCE,
                text="Size_in_SqFt"
            )
            fig_pt.update_traces(texttemplate="%{text:,.0f} SqFt", textposition="outside")
            apply_estateiq_theme(fig_pt, title="Average Property Size by Property Type", height=420)
            fig_pt.update_yaxes(title_text="Average Size (SqFt)")
            st.plotly_chart(fig_pt, use_container_width=True)

        with c_p2:
            # BHK Distribution
            bhk_counts = filtered_df["BHK"].value_counts().sort_index().reset_index()
            bhk_counts.columns = ["BHK", "Properties"]
            fig_bhk = px.bar(
                bhk_counts,
                x="BHK",
                y="Properties",
                color_discrete_sequence=[CHART_PALETTE["sage_green"]],
                text="Properties"
            )
            fig_bhk.update_traces(texttemplate="%{text:,}", textposition="outside")
            apply_estateiq_theme(fig_bhk, title="Listing Distribution by Bedroom Count (BHK)", height=420)
            fig_bhk.update_yaxes(title_text="Number of Listings")
            st.plotly_chart(fig_bhk, use_container_width=True)

    # Row 2: Infrastructure, Amenities & Asset Vintage
    st.markdown("### 🏊 Infrastructure, Amenities & Building Vintage")
    a_col1, a_col2 = st.columns(2)

    with a_col1:
        # Amenities Prevalence Rates
        amenity_cols = ["Has_Gym", "Has_Pool", "Has_Garden", "Has_Playground", "Has_Clubhouse"]
        amenity_names = ["Fitness Gym", "Swimming Pool", "Garden", "Playground", "Clubhouse"]
        amenity_rates = []
        for col, name in zip(amenity_cols, amenity_names):
            if col in filtered_df.columns:
                rate = filtered_df[col].mean() * 100.0
                amenity_rates.append({"Amenity": name, "Prevalence Rate (%)": rate})

        am_df = pd.DataFrame(amenity_rates).sort_values(by="Prevalence Rate (%)", ascending=False)
        fig_am = px.bar(
            am_df,
            x="Prevalence Rate (%)",
            y="Amenity",
            orientation="h",
            color_discrete_sequence=[CHART_PALETTE["emerald"]],
            text="Prevalence Rate (%)"
        )
        fig_am.update_layout(yaxis=dict(autorange="reversed"))
        fig_am.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        apply_estateiq_theme(fig_am, title="Recreational Amenity Penetration Rate", height=360)
        fig_am.update_xaxes(range=[0, 105], title_text="Penetration Rate (%)")
        st.plotly_chart(fig_am, use_container_width=True)

    with a_col2:
        # Building Vintage Distribution
        if "Age_of_Property" in filtered_df.columns:
            age_bins = [-1, 5, 15, 25, 100]
            age_labels = ["New (<5y)", "Modern (5-15y)", "Mature (15-25y)", "Aging (25y+)"]
            filtered_df["Vintage_Tier"] = pd.cut(filtered_df["Age_of_Property"], bins=age_bins, labels=age_labels)
            vintage_counts = filtered_df["Vintage_Tier"].value_counts().reset_index()
            vintage_counts.columns = ["Vintage Tier", "Count"]

            fig_vint = px.pie(
                vintage_counts,
                names="Vintage Tier",
                values="Count",
                hole=0.45,
                color_discrete_sequence=[
                    CHART_PALETTE["emerald"],
                    CHART_PALETTE["champagne_gold"],
                    CHART_PALETTE["sage_green"],
                    CHART_PALETTE["slate_gray"]
                ]
            )
            fig_vint.update_traces(textinfo="percent+label", textposition="inside")
            apply_estateiq_theme(fig_vint, title="Asset Vintage & Building Age Tiers", height=360)
            st.plotly_chart(fig_vint, use_container_width=True)
