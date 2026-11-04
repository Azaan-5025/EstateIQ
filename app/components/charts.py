"""
ESTATEIQ - Plotly Chart Styling Component
-----------------------------------------
Applies ESTATEIQ luxury palette and consistent formatting to all interactive charts.
"""

import plotly.express as px
import plotly.graph_objects as go

# Centralized palette matching the reference Luxury Forest & Gold aesthetic
CHART_PALETTE = {
    "deep_forest": "#102D23",
    "dark_forest": "#17372B",
    "sage_green": "#DCE7D5",
    "light_sage": "#C9D9C1",
    "champagne_gold": "#D9B76E",
    "warm_sand": "#E9E3D7",
    "ivory": "#F8F7F2",
    "primary_text": "#18352A",
    "secondary_text": "#52665B",
    "white": "#FFFFFF",
    "accent_red": "#E63946",
    # Aliases for existing chart references
    "slate_gray": "#52665B",
    "crimson_accent": "#E63946",
    "emerald": "#17372B"
}

PALETTE_SEQUENCE = [
    CHART_PALETTE["deep_forest"],
    CHART_PALETTE["champagne_gold"],
    CHART_PALETTE["dark_forest"],
    CHART_PALETTE["light_sage"],
    CHART_PALETTE["secondary_text"],
    "#3182CE",
    "#805AD5"
]

def apply_estateiq_theme(
    fig: go.Figure,
    title: str = "",
    height: int = 450,
    xaxis_title: str = None,
    yaxis_title: str = None
) -> go.Figure:
    """
    Applies custom luxury ESTATEIQ typography, margins, and borders to any Plotly figure.
    """
    # Extract existing axis title texts if already present on figure
    xtitle = xaxis_title
    if xtitle is None:
        try:
            xtitle = fig.layout.xaxis.title.text
        except Exception:
            xtitle = ""
    if xtitle is None:
        xtitle = ""

    ytitle = yaxis_title
    if ytitle is None:
        try:
            ytitle = fig.layout.yaxis.title.text
        except Exception:
            ytitle = ""
    if ytitle is None:
        ytitle = ""

    fig.update_layout(
        title={
            "text": title,
            "font": {"family": "Playfair Display, Georgia, serif", "size": 17, "color": CHART_PALETTE["deep_forest"]},
            "x": 0.02,
            "y": 0.95
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Plus Jakarta Sans, sans-serif", "size": 12, "color": CHART_PALETTE["primary_text"]},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        height=height,
        hoverlabel={
            "bgcolor": CHART_PALETTE["deep_forest"],
            "font": {
                "size": 12,
                "color": "#F8F7F2",
                "family": "Plus Jakarta Sans, sans-serif"
            }
        },
        xaxis={
            "gridcolor": CHART_PALETTE["warm_sand"],
            "zerolinecolor": CHART_PALETTE["warm_sand"],
            "linecolor": CHART_PALETTE["deep_forest"],
            "tickfont": {"color": CHART_PALETTE["secondary_text"], "family": "Plus Jakarta Sans, sans-serif"},
            "title": {
                "text": xtitle,
                "font": {"color": CHART_PALETTE["primary_text"], "family": "Plus Jakarta Sans, sans-serif"}
            }
        },
        yaxis={
            "gridcolor": CHART_PALETTE["warm_sand"],
            "zerolinecolor": CHART_PALETTE["warm_sand"],
            "linecolor": CHART_PALETTE["deep_forest"],
            "tickfont": {"color": CHART_PALETTE["secondary_text"], "family": "Plus Jakarta Sans, sans-serif"},
            "title": {
                "text": ytitle,
                "font": {"color": CHART_PALETTE["primary_text"], "family": "Plus Jakarta Sans, sans-serif"}
            }
        },
        legend={
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "bordercolor": CHART_PALETTE["warm_sand"],
            "borderwidth": 1,
            "font": {"color": CHART_PALETTE["primary_text"], "family": "Plus Jakarta Sans, sans-serif"}
        }
    )
    return fig


def create_gauge_chart(score: float) -> go.Figure:
    """
    Renders an elegant circular gauge chart displaying model estimated match score.
    """
    pct = score * 100.0

    bar_color = CHART_PALETTE["dark_forest"] if pct >= 65.0 else CHART_PALETTE["crimson_accent"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"family": "Playfair Display, Georgia, serif", "size": 36, "color": CHART_PALETTE["deep_forest"]}},
        title={"text": "Estimated IVS Match Probability", "font": {"size": 13, "color": CHART_PALETTE["secondary_text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": CHART_PALETTE["deep_forest"]},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "#FFFFFF",
            "borderwidth": 1.5,
            "bordercolor": CHART_PALETTE["warm_sand"],
            "steps": [
                {"range": [0, 50], "color": "rgba(230, 57, 70, 0.08)"},
                {"range": [50, 65], "color": "rgba(217, 183, 110, 0.15)"},
                {"range": [65, 100], "color": "rgba(16, 45, 35, 0.12)"}
            ],
            "threshold": {
                "line": {"color": CHART_PALETTE["deep_forest"], "width": 3},
                "thickness": 0.8,
                "value": 65.0
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        height=260,
        margin=dict(l=30, r=30, t=40, b=20)
    )
    return fig


def create_scenario_trajectory_chart(scenarios: dict, years: int) -> go.Figure:
    """
    Plots multi-scenario capital growth curves over holding period t.
    """
    fig = go.Figure()

    colors = {
        "Conservative (4.5%)": CHART_PALETTE["secondary_text"],
        "Moderate (6.5%)": CHART_PALETTE["champagne_gold"],
        "Optimistic (8.5%)": CHART_PALETTE["dark_forest"],
        "Custom Scenario": CHART_PALETTE["deep_forest"]
    }

    year_labels = [f"Year {y}" for y in range(years + 1)]

    for name, trajectory in scenarios.items():
        color = colors.get(name, CHART_PALETTE["deep_forest"])
        is_custom = "Custom" in name
        dash_style = "dash" if is_custom else "solid"
        width = 3.5 if is_custom else 2.5

        fig.add_trace(go.Scatter(
            x=year_labels,
            y=trajectory,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=width, dash=dash_style),
            marker=dict(size=7, color=color),
            hovertemplate="<b>%{x}</b><br>Projected Value: ₹%{y:.2f} Lakhs<extra></extra>"
        ))

    apply_estateiq_theme(fig, title="Capital Appreciation Trajectories Across Scenarios", height=450)
    fig.update_yaxes(title_text="Projected Value (₹ Lakhs)")
    fig.update_xaxes(title_text="Investment Timeline")
    return fig
