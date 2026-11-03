"""
ESTATEIQ - UI Cards & Banners Component
---------------------------------------
Reusable KPI cards, decision verdict banners, and mandatory disclosure callouts.
"""

import streamlit as st

def render_hero(title: str, tagline: str, description: str):
    """
    Renders the luxury hero header container for each page.
    """
    html = f"""
    <div class="estateiq-hero">
        <h1>{title}</h1>
        <p class="tagline">{tagline}</p>
        <p class="desc">{description}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = "", accent: str = "gold"):
    """
    Renders an ESTATEIQ luxury KPI metric card.
    accent options: 'gold', 'green', 'sage'
    """
    accent_class = f"metric-accent-{accent}"
    html = f"""
    <div class="metric-card {accent_class}">
        <div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        <div class="metric-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_decision_badge(is_good: bool, score: float):
    """
    Renders the model classification decision banner.
    Note: Labelled strictly as the model's estimated probability of matching
    the rule-based Investment Viability Score class (NOT financial confidence or profit probability).
    """
    score_pct = score * 100.0

    if is_good:
        html = f"""
        <div class="decision-card-good">
            <div class="decision-title-good">✓ Good Investment Candidate</div>
            <p style="font-size: 1.05rem; color: #18352A; margin-bottom: 0.8rem; font-weight: 500;">
                The property's intrinsic physical, spatial, and civic infrastructure attributes match the underwriting criteria of the 
                Investment Viability Score (IVS) framework.
            </p>
            <div style="background: rgba(255, 255, 255, 0.9); padding: 0.75rem 1rem; border-radius: 6px; font-size: 0.9rem; color: #102D23; border: 1px solid #C9D9C1;">
                <strong>Model Estimated Probability of Matching IVS Viable Class:</strong> {score_pct:.1f}%<br>
                <span style="font-size: 0.82rem; color: #52665B;">
                    (This reflects model alignment with multi-criteria underwriting rules; it is not a probability of financial market profit).
                </span>
            </div>
        </div>
        """
    else:
        html = f"""
        <div class="decision-card-suboptimal">
            <div class="decision-title-suboptimal">⚠ Not a Good Investment Candidate</div>
            <p style="font-size: 1.05rem; color: #18352A; margin-bottom: 0.8rem; font-weight: 500;">
                The property exhibits one or more structural, layout, vintage, or infrastructure deficits relative to institutional underwriting benchmarks.
            </p>
            <div style="background: rgba(255, 255, 255, 0.9); padding: 0.75rem 1rem; border-radius: 6px; font-size: 0.9rem; color: #E63946; border: 1px solid #E9E3D7;">
                <strong>Model Estimated Probability of Matching IVS Viable Class:</strong> {score_pct:.1f}%<br>
                <span style="font-size: 0.82rem; color: #52665B;">
                    (This reflects model alignment with multi-criteria underwriting rules; it is not a probability of financial market profit).
                </span>
            </div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


def render_disclosure_banner(text: str, banner_type: str = "info"):
    """
    Renders standardized domain disclosure and limitation banners.
    """
    css_class = "warning-banner" if banner_type == "warning" else "disclosure-banner"
    prefix = "⚠️ <strong>MANDATORY DISCLOSURE:</strong> " if banner_type == "warning" else "ℹ️ <strong>DOMAIN NOTICE:</strong> "
    html = f"""
    <div class="{css_class}">
        {prefix}{text}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
