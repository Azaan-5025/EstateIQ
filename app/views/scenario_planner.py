"""
ESTATEIQ - Page 4: Five-Year Scenario Planner
---------------------------------------------
Financial compound growth projection engine across Conservative, Moderate,
Optimistic, and Custom multi-year scenarios.
Explicitly discloses assumption-based financial modeling (NOT an ML forecast).
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.components.cards import render_hero, render_disclosure_banner, render_metric_card
from app.components.charts import create_scenario_trajectory_chart


def render_scenario_planner(default_price: float = 250.0):
    """
    Renders the interactive multi-year capital appreciation scenario planner.
    """
    render_hero(
        title="Five-Year Financial Scenario Planner",
        tagline="Compound Capital Appreciation Modeling Across Defined Investment Horizons",
        description="Simulates forward investment horizons using institutional compound annual growth rate (CAGR) assumptions and user-configurable holding periods."
    )

    # 1. Mandatory Mathematical Modeling Disclosure
    render_disclosure_banner(
        "<strong>ASSUMPTION-BASED FINANCIAL PROJECTION NOTICE:</strong><br>"
        "These projections are calculated using standard compound annual growth mathematics: "
        "<code>Future Value = Current Value × (1 + r)^t</code>. "
        "<strong>They are purely financial simulation scenarios based on user-defined growth parameters, NOT machine learning forecasts, statistical regressions, or guaranteed investment returns.</strong>",
        banner_type="warning"
    )

    # 2. Input Parameters
    st.markdown("### ⚙️ Configure Investment & Horizon Parameters")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        current_price = st.number_input(
            "Current Property Value (₹ Lakhs)",
            min_value=5.0,
            max_value=2000.0,
            value=float(default_price),
            step=5.0,
            format="%.2f"
        )
    with col_in2:
        holding_years = st.slider(
            "Holding Period Horizon (Years)",
            min_value=1,
            max_value=15,
            value=5,
            step=1
        )
    with col_in3:
        custom_rate = st.number_input(
            "Custom Annual Appreciation Rate (%)",
            min_value=-5.0,
            max_value=25.0,
            value=7.0,
            step=0.5,
            format="%.1f"
        )

    # 3. Compound Growth Scenarios Calculation
    rates = {
        "Conservative (4.5%)": 0.045,
        "Moderate (6.5%)": 0.065,
        "Optimistic (8.5%)": 0.085,
        "Custom Scenario": custom_rate / 100.0
    }

    scenarios_trajectories = {}
    summary_data = []

    for name, rate in rates.items():
        # Trajectory from year 0 to year t
        trajectory = [current_price * ((1.0 + rate) ** y) for y in range(holding_years + 1)]
        scenarios_trajectories[name] = trajectory

        final_val = trajectory[-1]
        total_gain = final_val - current_price
        pct_gain = (total_gain / current_price) * 100.0

        summary_data.append({
            "Scenario": name,
            "Annual Rate": f"{rate*100:.1f}%",
            "Current Value": f"₹{current_price:.2f}L",
            "Projected Future Value": f"₹{final_val:.2f}L",
            "Total Appreciation": f"₹{total_gain:.2f}L",
            "Total Growth": f"{pct_gain:+.1f}%",
            "_raw_final": final_val,
            "_raw_gain": total_gain,
            "_raw_pct": pct_gain
        })

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"### 📊 Projected Valuations at Year {holding_years}")

    # 4. Display KPI Cards for 3 Core Scenarios
    c1, c2, c3 = st.columns(3)
    with c1:
        s = summary_data[0]
        render_metric_card(
            label="Conservative Scenario (4.5% p.a.)",
            value=s["Projected Future Value"],
            subtext=f"Total Gain: {s['Total Appreciation']} ({s['Total Growth']})",
            accent="sage"
        )
    with c2:
        s = summary_data[1]
        render_metric_card(
            label="Moderate Scenario (6.5% p.a.)",
            value=s["Projected Future Value"],
            subtext=f"Total Gain: {s['Total Appreciation']} ({s['Total Growth']})",
            accent="gold"
        )
    with c3:
        s = summary_data[2]
        render_metric_card(
            label="Optimistic Scenario (8.5% p.a.)",
            value=s["Projected Future Value"],
            subtext=f"Total Gain: {s['Total Appreciation']} ({s['Total Growth']})",
            accent="green"
        )

    # Custom Scenario if different
    if custom_rate not in [4.5, 6.5, 8.5]:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        s_custom = summary_data[3]
        st.info(
            f"**Your Custom Scenario ({custom_rate:.1f}% p.a.):** Projected Value after {holding_years} years is "
            f"**{s_custom['Projected Future Value']}**, representing a cumulative gain of **{s_custom['Total Appreciation']}** ({s_custom['Total Growth']})."
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 5. Interactive Trajectory Chart
    fig_trajectory = create_scenario_trajectory_chart(scenarios_trajectories, holding_years)
    st.plotly_chart(fig_trajectory, use_container_width=True)

    # 6. Detailed Annual Progression Schedule
    with st.expander(f"📅 View Year-by-Year Valuation Schedule (Years 0 to {holding_years})"):
        schedule_rows = []
        for y in range(holding_years + 1):
            row = {"Year": f"Year {y}" if y > 0 else "Base (Year 0)"}
            for name, traj in scenarios_trajectories.items():
                row[name] = f"₹{traj[y]:.2f} Lakhs"
            schedule_rows.append(row)
        
        st.dataframe(pd.DataFrame(schedule_rows), use_container_width=True)
