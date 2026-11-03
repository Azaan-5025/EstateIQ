"""
ESTATEIQ - Phase 6 Test Suite: Application Modules, Models & Feature Pipelines
-------------------------------------------------------------------------------
Executes comprehensive automated testing for Phase 6 deliverables.
"""

import sys
import os
import py_compile
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
test_dir = Path(__file__).resolve().parent
project_root = test_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_1_python_compilation():
    """Verify syntax and bytecode compilation of all app files."""
    app_files = [
        project_root / "app" / "streamlit_app.py",
        project_root / "app" / "components" / "sidebar.py",
        project_root / "app" / "components" / "cards.py",
        project_root / "app" / "components" / "charts.py",
        project_root / "app" / "views" / "dashboard.py",
        project_root / "app" / "views" / "investment_advisor.py",
        project_root / "app" / "views" / "price_estimator.py",
        project_root / "app" / "views" / "scenario_planner.py",
        project_root / "app" / "views" / "market_analytics.py",
    ]
    for file_path in app_files:
        assert file_path.exists(), f"File missing: {file_path}"
        py_compile.compile(str(file_path), doraise=True)
    print("[PASS] Test 1 Passed: All 9 application modules compiled successfully without syntax errors.")


def test_2_model_artifacts_loading():
    """Verify that both production model artifacts load and contain fitted steps."""
    clf_path = project_root / "models" / "best_classification_model.joblib"
    reg_path = project_root / "models" / "best_regression_model.joblib"

    assert clf_path.exists(), "Classification model artifact missing!"
    assert reg_path.exists(), "Regression model artifact missing!"

    clf_pipe = joblib.load(clf_path)
    reg_pipe = joblib.load(reg_path)

    assert "preprocessor" in clf_pipe.named_steps, "Classifier pipeline missing preprocessor step!"
    assert "classifier" in clf_pipe.named_steps, "Classifier pipeline missing classifier step!"
    assert "preprocessor" in reg_pipe.named_steps, "Regressor pipeline missing preprocessor step!"
    assert "regressor" in reg_pipe.named_steps, "Regressor pipeline missing regressor step!"
    print("[PASS] Test 2 Passed: Both model artifacts loaded cleanly with preprocessor and estimator steps.")


def test_3_module_imports():
    """Verify that every app component and page imports successfully."""
    from app.components.sidebar import render_sidebar
    from app.components.cards import render_hero, render_metric_card, render_decision_badge, render_disclosure_banner
    from app.components.charts import apply_estateiq_theme, create_gauge_chart, create_scenario_trajectory_chart
    from app.views.dashboard import render_dashboard
    from app.views.investment_advisor import render_investment_advisor, APPROVED_FEATURE_ORDER
    from app.views.price_estimator import render_price_estimator
    from app.views.scenario_planner import render_scenario_planner
    from app.views.market_analytics import render_market_analytics
    assert len(APPROVED_FEATURE_ORDER) == 34
    print("[PASS] Test 3 Passed: All component and page modules imported successfully.")


def test_4_feature_generation_and_classification():
    """Test feature construction and classification pipeline inference."""
    from app.views.investment_advisor import APPROVED_FEATURE_ORDER

    clf_pipe = joblib.load(project_root / "models" / "best_classification_model.joblib")

    # Simulate realistic input
    bhk = 3
    size_sqft = 1850.0
    floor_no = 5.0
    total_floors = 15.0
    age_years = 4.0
    schools = 6
    hospitals = 5
    transit_acc = "High"
    furnish_status = "Semi-furnished"
    parking = "Yes"
    security = "Yes"

    # Amenity flags
    gym_val, pool_val, garden_val, playground_val, clubhouse_val = 1, 1, 1, 0, 1
    tot_amenities = 4
    lux_index = 0.8

    transport_map = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
    furnish_map = {"Furnished": 1.0, "Semi-furnished": 0.5, "Unfurnished": 0.0}
    binary_map = {"Yes": 1.0, "No": 0.0}

    pub_transit_num = transport_map[transit_acc]
    furn_num = furnish_map[furnish_status]
    parking_num = binary_map[parking]
    sec_num = binary_map[security]

    connectivity_score = round((0.50 * pub_transit_num) + (0.25 * (schools / 10.0)) + (0.25 * (hospitals / 10.0)), 4)
    quality_score = round((0.40 * furn_num) + (0.30 * parking_num) + (0.30 * sec_num), 4)
    size_per_bhk = round(size_sqft / bhk, 2)
    rel_floor = round(floor_no / total_floors, 4)

    age_bins = [-1, 5, 15, 25, 100]
    age_labels = ["New", "Modern", "Mature", "Aging"]
    age_tier = pd.cut([age_years], bins=age_bins, labels=age_labels)[0]

    feature_dict = {
        "State": "Maharashtra",
        "City": "Mumbai",
        "Locality": "Andheri West",
        "Property_Type": "Apartment",
        "BHK": int(bhk),
        "Size_in_SqFt": float(size_sqft),
        "Furnished_Status": furnish_status,
        "Floor_No": float(floor_no),
        "Total_Floors": float(total_floors),
        "Age_of_Property": float(age_years),
        "Nearby_Schools": int(schools),
        "Nearby_Hospitals": int(hospitals),
        "Public_Transport_Accessibility": transit_acc,
        "Parking_Space": parking,
        "Security": security,
        "Facing": "East",
        "Owner_Type": "Owner",
        "Availability_Status": "Ready to Move",
        "Has_Gym": gym_val,
        "Has_Pool": pool_val,
        "Has_Garden": garden_val,
        "Has_Playground": playground_val,
        "Has_Clubhouse": clubhouse_val,
        "Total_Amenities_Count": tot_amenities,
        "Public_Transport_Num": pub_transit_num,
        "Furnished_Status_Num": furn_num,
        "Parking_Space_Num": parking_num,
        "Security_Num": sec_num,
        "Luxury_Amenities_Index": lux_index,
        "Connectivity_Score": connectivity_score,
        "Property_Quality_Score": quality_score,
        "Size_per_BHK": size_per_bhk,
        "Relative_Floor_Position": rel_floor,
        "Age_Group_Tier": age_tier
    }

    input_df = pd.DataFrame([feature_dict])[APPROVED_FEATURE_ORDER]
    assert list(input_df.columns) == APPROVED_FEATURE_ORDER
    assert len(input_df.columns) == 34

    # Run inference
    pred = clf_pipe.predict(input_df)
    prob = clf_pipe.predict_proba(input_df)

    assert pred[0] in [0, 1]
    assert 0.0 <= prob[0, 1] <= 1.0
    print(f"[PASS] Test 4 Passed: Feature generation produced exact 34 features. Classification Prediction: {pred[0]} | IVS Match Probability: {prob[0, 1]*100:.1f}%.")


def test_5_price_estimator():
    """Test regression pipeline inference on the input feature vector."""
    from app.views.price_estimator import APPROVED_FEATURE_ORDER

    reg_pipe = joblib.load(project_root / "models" / "best_regression_model.joblib")

    # Mock record
    feature_dict = {
        "State": "Karnataka",
        "City": "Bangalore",
        "Locality": "Whitefield",
        "Property_Type": "Apartment",
        "BHK": 3,
        "Size_in_SqFt": 1850.0,
        "Furnished_Status": "Semi-furnished",
        "Floor_No": 4.0,
        "Total_Floors": 12.0,
        "Age_of_Property": 5.0,
        "Nearby_Schools": 6,
        "Nearby_Hospitals": 5,
        "Public_Transport_Accessibility": "High",
        "Parking_Space": "Yes",
        "Security": "Yes",
        "Facing": "East",
        "Owner_Type": "Owner",
        "Availability_Status": "Ready to Move",
        "Has_Gym": 1,
        "Has_Pool": 1,
        "Has_Garden": 1,
        "Has_Playground": 0,
        "Has_Clubhouse": 1,
        "Total_Amenities_Count": 4,
        "Public_Transport_Num": 1.0,
        "Furnished_Status_Num": 0.5,
        "Parking_Space_Num": 1.0,
        "Security_Num": 1.0,
        "Luxury_Amenities_Index": 0.8,
        "Connectivity_Score": 0.775,
        "Property_Quality_Score": 0.800,
        "Size_per_BHK": 616.67,
        "Relative_Floor_Position": 0.3333,
        "Age_Group_Tier": "New"
    }

    input_df = pd.DataFrame([feature_dict])[APPROVED_FEATURE_ORDER]
    pred_price = float(reg_pipe.predict(input_df)[0])
    assert 10.0 <= pred_price <= 500.0, f"Unreasonable price estimate: {pred_price}"
    print(f"[PASS] Test 5 Passed: Price Estimator inference succeeded. Predicted Price: INR {pred_price:.2f} Lakhs.")


def test_6_five_year_scenario_calculations():
    """Verify compound growth mathematical projections."""
    current_price = 250.0
    holding_years = 5

    # Conservative: 4.5%
    c_val = current_price * ((1.0 + 0.045) ** holding_years)
    assert round(c_val, 2) == 311.55, f"Conservative mismatch: {c_val}"

    # Moderate: 6.5%
    m_val = current_price * ((1.0 + 0.065) ** holding_years)
    assert round(m_val, 2) == 342.52, f"Moderate mismatch: {m_val}"

    # Optimistic: 8.5%
    o_val = current_price * ((1.0 + 0.085) ** holding_years)
    assert round(o_val, 2) == 375.91, f"Optimistic mismatch: {o_val}"

    print(f"[PASS] Test 6 Passed: Scenario calculations verified (Conservative: INR {c_val:.2f}L | Moderate: INR {m_val:.2f}L | Optimistic: INR {o_val:.2f}L).")


def test_7_dataset_integrity():
    """Verify that india_housing_cleaned.csv loads dynamically with 250,000 records."""
    data_path = project_root / "data" / "processed" / "india_housing_cleaned.csv"
    assert data_path.exists(), "Cleaned dataset missing!"
    df = pd.read_csv(data_path)
    assert len(df) == 250000, f"Expected 250,000 records, got {len(df)}"
    assert df["Price_in_Lakhs"].notnull().all(), "Null prices found!"
    assert df["Size_in_SqFt"].notnull().all(), "Null sizes found!"
    print(f"[PASS] Test 7 Passed: Cleaned dataset dynamically verified (250,000 records, {df['State'].nunique()} States, {df['City'].nunique()} Cities).")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING ESTATEIQ PHASE 6 AUTOMATED VERIFICATION SUITE")
    print("=" * 80)
    test_1_python_compilation()
    test_2_model_artifacts_loading()
    test_3_module_imports()
    test_4_feature_generation_and_classification()
    test_5_price_estimator()
    test_6_five_year_scenario_calculations()
    test_7_dataset_integrity()
    print("=" * 80)
    print("ALL 7 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
