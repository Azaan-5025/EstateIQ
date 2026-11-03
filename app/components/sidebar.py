"""
ESTATEIQ - Sidebar Component
----------------------------
Renders the navigation menu, brand header, and model governance badge.
"""

import streamlit as st

def render_sidebar() -> str:
    """
    Renders the custom branded sidebar and returns the selected page name.
    """
    with st.sidebar:
        # Injected Scoped CSS for Horizontal Navigation & Sidebar Width Integrity
        st.markdown(
            """
            <style>
            /* Desktop Sidebar Width: Locked between 300px - 330px */
            section[data-testid="stSidebar"],
            [data-testid="stSidebar"] {
                width: 320px !important;
                min-width: 300px !important;
                max-width: 330px !important;
                flex-basis: 320px !important;
                flex-shrink: 0 !important;
            }
            section[data-testid="stSidebar"] > div:first-child,
            div[data-testid="stSidebarContent"],
            div[data-testid="stSidebarUserContent"] {
                width: 320px !important;
                min-width: 300px !important;
                max-width: 330px !important;
                box-sizing: border-box !important;
            }
            section[data-testid="stSidebar"] .block-container {
                width: 100% !important;
                max-width: 100% !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                box-sizing: border-box !important;
            }

            /* Full Width & Flex Layout for Radio Group */
            section[data-testid="stSidebar"] div[data-testid="stRadio"],
            section[data-testid="stSidebar"] div[data-testid="stRadio"] > div,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"],
            section[data-testid="stSidebar"] div[data-testid="stRadio"] div[data-testid="stRadioGroup"] {
                width: 100% !important;
                min-width: 100% !important;
                max-width: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                gap: 8px !important;
                box-sizing: border-box !important;
            }

            /* Each Radio Option: Full Width, Horizontal Single Line */
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"],
            section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {
                width: 100% !important;
                min-width: 100% !important;
                max-width: 100% !important;
                display: flex !important;
                flex-direction: row !important;
                align-items: center !important;
                justify-content: flex-start !important;
                flex-wrap: nowrap !important;
                box-sizing: border-box !important;
                padding: 0.65rem 0.85rem !important;
                margin: 0 !important;
                border-radius: 8px !important;
                cursor: pointer !important;
                min-height: 44px !important;
            }

            /* Strict Prevention of ALL Text Wrapping Across ALL Radio Elements */
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"],
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label *,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label div,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label span,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"],
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {
                white-space: nowrap !important;
                word-break: normal !important;
                overflow-wrap: normal !important;
                writing-mode: horizontal-tb !important;
                flex-wrap: nowrap !important;
            }

            /* Radio Circles Aligned to Left */
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label input + div,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label div[data-baseweb="radio"] > div:first-child {
                flex-shrink: 0 !important;
                min-width: 16px !important;
                width: 16px !important;
                height: 16px !important;
                margin: 0 !important;
                margin-right: 0.75rem !important;
            }

            /* Icons and Text Horizontally Aligned */
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:last-child,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] {
                display: inline-flex !important;
                flex-direction: row !important;
                align-items: center !important;
                flex-grow: 1 !important;
                min-width: 0 !important;
                width: auto !important;
                max-width: none !important;
                padding: 0 !important;
                margin: 0 !important;
            }

            section[data-testid="stSidebar"] div[data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label span {
                display: inline-block !important;
                font-family: 'Plus Jakarta Sans', sans-serif !important;
                font-size: 0.92rem !important;
                line-height: 1.35 !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            /* Active Selected Option: Lighter Forest Green with Champagne Gold accent */
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
                background: #24483A !important;
                background-color: #24483A !important;
                border: 1px solid rgba(217, 183, 110, 0.5) !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) *,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) span {
                color: #F8F7F2 !important;
                font-weight: 700 !important;
                background: transparent !important;
                background-color: transparent !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Brand Header
        st.markdown(
            """
            <div class="sidebar-brand">
                <h2>ESTATEIQ</h2>
                <p>Real Estate Investment Advisor</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption("Smarter Property Decisions Through Data-Driven Insights.")

        # Navigation Options
        pages = {
            "dashboard": "🏛️ Executive Dashboard",
            "investment_advisor": "🎯 Investment Advisor",
            "price_estimator": "🏷️ Listed Price Estimator",
            "scenario_planner": "📈 5-Year Scenario Planner",
            "market_analytics": "📊 Market Analytics"
        }

        selected_label = st.radio(
            "Navigation",
            options=list(pages.values()),
            index=0,
            label_visibility="collapsed"
        )

        # Invert dictionary to return page key
        selected_key = [k for k, v in pages.items() if v == selected_label][0]

        # Model Governance Status Card
        st.markdown(
            """
            <div class="sidebar-status-card">
                <strong>System Governance & Health</strong><br>
                • <strong>Classification:</strong> XGBoost v1.0<br>
                &nbsp;&nbsp;Holdout Test Acc: <strong>73.29%</strong> | AUC: <strong>0.8228</strong><br>
                • <strong>Price Regressor:</strong> Random Forest v1.0<br>
                &nbsp;&nbsp;Holdout Test MAE: <strong>₹122.81L</strong> (R² ≈ 0.00)<br>
                • <strong>Dataset:</strong> 250,000 Verified Listings<br>
                • <strong>Partitioning:</strong> 70 / 15 / 15 (Zero Leakage)
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="margin-top: 1.5rem; font-size: 0.72rem; color: #C9D9C1; text-align: center;">
                ESTATEIQ Intelligence v1.0<br>Institutional Advisory Platform
            </div>
            """,
            unsafe_allow_html=True
        )

    return selected_key
