"""
ESTATEIQ - Real Estate Investment Advisor Web Application
---------------------------------------------------------
Tagline: Smarter Property Decisions Through Data-Driven Insights.
Main entry point coordinating modular pages, cached resource loading, and routing.
"""

import sys
import os
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configure sys.path so app packages are fully discoverable
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Streamlit Page Setup
st.set_page_config(
    page_title="ESTATEIQ — Real Estate Investment Advisor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import UI Components
from app.components.sidebar import render_sidebar
from app.views.dashboard import render_dashboard
from app.views.investment_advisor import render_investment_advisor
from app.views.price_estimator import render_price_estimator
from app.views.scenario_planner import render_scenario_planner
from app.views.market_analytics import render_market_analytics


# =============================================================================
# RESOURCE CACHING & INITIALIZATION
# =============================================================================

def load_custom_css():
    """
    Injects custom luxury styling from assets/style.css.
    """
    css_path = app_dir / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_cleaned_dataset() -> pd.DataFrame:
    """
    Caches and loads the 250,000 cleaned real estate records.
    """
    data_path = project_root / "data" / "processed" / "india_housing_cleaned.csv"
    if not data_path.exists():
        st.error(f"Processed dataset missing at {data_path.as_posix()}")
        st.stop()
    df = pd.read_csv(data_path)
    return df


@st.cache_resource(show_spinner=False)
def load_trained_models():
    """
    Caches and loads the pre-fitted classification and regression pipelines.
    """
    models_dir = project_root / "models"
    clf_path = models_dir / "best_classification_model.joblib"
    reg_path = models_dir / "best_regression_model.joblib"

    if not clf_path.exists() or not reg_path.exists():
        st.error("Model artifacts missing in models/ directory.")
        st.stop()

    clf_pipe = joblib.load(clf_path)
    reg_pipe = joblib.load(reg_path)
    return clf_pipe, reg_pipe


@st.cache_data(show_spinner=False)
def load_model_metadata():
    """
    Loads model training metadata records.
    """
    models_dir = project_root / "models"
    clf_meta_path = models_dir / "classification_model_metadata.json"
    reg_meta_path = models_dir / "regression_model_metadata.json"

    with open(clf_meta_path, "r", encoding="utf-8") as f:
        clf_meta = json.load(f)
    with open(reg_meta_path, "r", encoding="utf-8") as f:
        reg_meta = json.load(f)

    return clf_meta, reg_meta


# =============================================================================
# MAIN APPLICATION CONTROLLER
# =============================================================================

def main():
    # 1. Load styles
    load_custom_css()

    # 2. Load cached resources
    with st.spinner("Initializing ESTATEIQ Institutional Intelligence Engine..."):
        df = load_cleaned_dataset()
        clf_pipe, reg_pipe = load_trained_models()
        clf_meta, reg_meta = load_model_metadata()

    # 3. Render branded sidebar and get selected page
    selected_page = render_sidebar()

    # 4. Route to selected view
    if selected_page == "dashboard":
        render_dashboard(df)
    elif selected_page == "investment_advisor":
        render_investment_advisor(df, clf_pipe, clf_meta)
    elif selected_page == "price_estimator":
        render_price_estimator(df, reg_pipe, reg_meta)
    elif selected_page == "scenario_planner":
        render_scenario_planner(default_price=250.0)
    elif selected_page == "market_analytics":
        render_market_analytics(df)
    else:
        render_dashboard(df)


if __name__ == "__main__":
    main()
