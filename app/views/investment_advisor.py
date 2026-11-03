"""
ESTATEIQ - Page 2: Investment Advisor
-------------------------------------
Interactive property evaluation interface utilizing the trained XGBoost Classifier pipeline.
Generates the exact 34 intrinsic features without price leakage.
Evaluates alignment with the rule-based Investment Viability Score (IVS) framework.
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.components.cards import render_hero, render_decision_badge, render_disclosure_banner, render_metric_card
from app.components.charts import create_gauge_chart


# Exact 34 Feature Order from Training Schema
APPROVED_FEATURE_ORDER = [
    "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
    "Furnished_Status", "Floor_No", "Total_Floors", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility",
    "Parking_Space", "Security", "Facing", "Owner_Type", "Availability_Status",
    "Has_Gym", "Has_Pool", "Has_Garden", "Has_Playground", "Has_Clubhouse",
    "Total_Amenities_Count", "Public_Transport_Num", "Furnished_Status_Num",
    "Parking_Space_Num", "Security_Num", "Luxury_Amenities_Index",
    "Connectivity_Score", "Property_Quality_Score", "Size_per_BHK",
    "Relative_Floor_Position", "Age_Group_Tier"
]


def render_investment_advisor(df: pd.DataFrame, clf_pipeline, clf_meta: dict):
    """
    Renders the interactive Investment Advisor form and prediction view.
    """
    render_hero(
        title="Institutional Investment Viability Advisor",
        tagline="XGBoost Classifier Underwriting Engine (Holdout Test Verified: 73.29% Accuracy | 0.8228 ROC-AUC)",
        description="Evaluates whether a residential asset's intrinsic physical, spatial, structural, and infrastructure attributes meet institutional Investment Viability Score (IVS) standards."
    )

    # 1. Mandatory Pre-Form Disclosure
    render_disclosure_banner(
        "This evaluation calculates the model's estimated probability of matching the rule-based Investment Viability Score (IVS) criteria. "
        "It evaluates intrinsic physical build, layout density, transit proximity, and asset vintage. "
        "It does NOT represent a guarantee of financial return or real-world capital appreciation.",
        banner_type="info"
    )

    # 2. Interactive Property Input Form
    st.markdown("### 📋 Enter Property Specifications")

    with st.form("investment_advisor_form"):
        # Section A: Geographic Location
        st.markdown("##### 📍 Location & Asset Classification")
        col_g1, col_g2, col_g3 = st.columns(3)

        states_list = sorted(df["State"].dropna().unique().tolist())
        with col_g1:
            selected_state = st.selectbox("State", options=states_list, index=0)

        cities_in_state = sorted(df[df["State"] == selected_state]["City"].dropna().unique().tolist())
        if not cities_in_state:
            cities_in_state = sorted(df["City"].dropna().unique().tolist())
        with col_g2:
            selected_city = st.selectbox("City", options=cities_in_state, index=0)

        localities_in_city = sorted(df[df["City"] == selected_city]["Locality"].dropna().unique().tolist())
        if not localities_in_city:
            localities_in_city = sorted(df["Locality"].dropna().unique().tolist())
        with col_g3:
            selected_locality = st.selectbox("Locality", options=localities_in_city, index=0)

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            prop_types = sorted(df["Property_Type"].dropna().unique().tolist())
            selected_prop_type = st.selectbox("Property Type", options=prop_types, index=0)
        with col_t2:
            owner_types = sorted(df["Owner_Type"].dropna().unique().tolist())
            selected_owner = st.selectbox("Owner / Transaction Type", options=owner_types, index=0)
        with col_t3:
            avail_types = sorted(df["Availability_Status"].dropna().unique().tolist())
            selected_avail = st.selectbox("Availability Status", options=avail_types, index=0)

        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #E9E3D7;'>", unsafe_allow_html=True)

        # Section B: Physical & Structural Dimensions
        st.markdown("##### 📐 Physical Dimensions & Vertical Elevation")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=8, value=3, step=1)
        with col_p2:
            size_sqft = st.number_input("Total Size (SqFt)", min_value=300, max_value=15000, value=1850, step=50)
        with col_p3:
            age_years = st.number_input("Property Age (Years)", min_value=0, max_value=60, value=4, step=1)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            floor_no = st.number_input("Floor Number", min_value=0, max_value=100, value=5, step=1,
                                       help="Ground floor is 0. If independent house/villa, enter 0 or 1.")
        with col_f2:
            total_floors = st.number_input("Total Building Floors", min_value=1, max_value=100, value=15, step=1)
        with col_f3:
            facings = sorted(df["Facing"].dropna().unique().tolist())
            selected_facing = st.selectbox("Orientation (Facing)", options=facings, index=0)

        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #E9E3D7;'>", unsafe_allow_html=True)

        # Section C: Civic Infrastructure & Safety
        st.markdown("##### 🚆 Infrastructure, Quality & Recreational Amenities")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            transit_opts = ["High", "Medium", "Low"]
            transit_acc = st.selectbox("Public Transport Accessibility", options=transit_opts, index=0)
        with col_c2:
            furnish_opts = ["Furnished", "Semi-furnished", "Unfurnished"]
            furnish_status = st.selectbox("Furnished Status", options=furnish_opts, index=1)
        with col_c3:
            parking = st.selectbox("Reserved Parking Space", options=["Yes", "No"], index=0)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            security = st.selectbox("24/7 Security Service", options=["Yes", "No"], index=0)
        with col_s2:
            schools = st.slider("Nearby Schools (Count within 3km)", min_value=0, max_value=10, value=6)
        with col_s3:
            hospitals = st.slider("Nearby Hospitals (Count within 3km)", min_value=0, max_value=10, value=5)

        # Recreational Amenities Checkboxes
        st.markdown("<strong>Recreational & Clubhouse Amenities</strong>", unsafe_allow_html=True)
        col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns(5)
        with col_a1:
            has_gym = st.checkbox("Gym", value=True, key="adv_gym")
        with col_a2:
            has_pool = st.checkbox("Swimming Pool", value=True, key="adv_pool")
        with col_a3:
            has_garden = st.checkbox("Lawn / Garden", value=True, key="adv_garden")
        with col_a4:
            has_playground = st.checkbox("Children's Playground", value=False, key="adv_play")
        with col_a5:
            has_clubhouse = st.checkbox("Residents Clubhouse", value=True, key="adv_club")

        st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)

        # Submit CTA
        submitted = st.form_submit_button("🏛️ Analyze Investment Viability", use_container_width=True)

    # 3. Post-Submission Inference & Presentation
    if submitted:
        # A. Input Validation
        if floor_no > total_floors:
            st.warning("⚠️ Floor Number is greater than Total Floors. As per institutional cleaning rules, Relative Floor Position is treated as unobserved/missing and handled via median imputation.")
            rel_floor = np.nan
        else:
            rel_floor = round(floor_no / total_floors, 4)

        # B. Feature Engineering (Strictly replicating build_features.py)
        # Amenity atomic flags & count
        gym_val = 1 if has_gym else 0
        pool_val = 1 if has_pool else 0
        garden_val = 1 if has_garden else 0
        playground_val = 1 if has_playground else 0
        clubhouse_val = 1 if has_clubhouse else 0
        tot_amenities = gym_val + pool_val + garden_val + playground_val + clubhouse_val
        lux_index = round(tot_amenities / 5.0, 4)

        # Numerical mappings
        transport_map = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
        furnish_map = {"Furnished": 1.0, "Semi-furnished": 0.5, "Unfurnished": 0.0}
        binary_map = {"Yes": 1.0, "No": 0.0}

        pub_transit_num = transport_map[transit_acc]
        furn_num = furnish_map[furnish_status]
        parking_num = binary_map[parking]
        sec_num = binary_map[security]

        # Domain composites
        connectivity_score = round(
            (0.50 * pub_transit_num) + (0.25 * (schools / 10.0)) + (0.25 * (hospitals / 10.0)), 4
        )
        quality_score = round(
            (0.40 * furn_num) + (0.30 * parking_num) + (0.30 * sec_num), 4
        )
        size_per_bhk = round(size_sqft / bhk, 2)

        # Age vintage tier
        age_bins = [-1, 5, 15, 25, 100]
        age_labels = ["New", "Modern", "Mature", "Aging"]
        age_tier = pd.cut([age_years], bins=age_bins, labels=age_labels)[0]

        # C. Assemble DataFrame in Exact Approved Order
        feature_dict = {
            "State": selected_state,
            "City": selected_city,
            "Locality": selected_locality,
            "Property_Type": selected_prop_type,
            "BHK": int(bhk),
            "Size_in_SqFt": float(size_sqft),
            "Furnished_Status": furnish_status,
            "Floor_No": float(floor_no) if floor_no <= total_floors else np.nan,
            "Total_Floors": float(total_floors),
            "Age_of_Property": float(age_years),
            "Nearby_Schools": int(schools),
            "Nearby_Hospitals": int(hospitals),
            "Public_Transport_Accessibility": transit_acc,
            "Parking_Space": parking,
            "Security": security,
            "Facing": selected_facing,
            "Owner_Type": selected_owner,
            "Availability_Status": selected_avail,
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

        # D. Run Model Inference
        y_pred = clf_pipeline.predict(input_df)[0]
        y_prob = clf_pipeline.predict_proba(input_df)[0, 1]
        is_good = bool(y_pred == 1)

        # E. Display Decision & Metrics
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 🏆 Investment Underwriting Verdict")

        col_verdict, col_gauge = st.columns([1.3, 1])

        with col_verdict:
            render_decision_badge(is_good=is_good, score=y_prob)

        with col_gauge:
            fig_gauge = create_gauge_chart(y_prob)
            st.plotly_chart(fig_gauge, use_container_width=True)

        # F. Property Underwriting Profile Breakdown
        st.markdown("#### 🔍 Property Characteristic Profile")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card(
                label="Connectivity Score",
                value=f"{connectivity_score * 100:.1f}%",
                subtext=f"Transit: {transit_acc} | {schools} Schools, {hospitals} Hosps",
                accent="green" if connectivity_score >= 0.6 else "gold"
            )
        with m2:
            render_metric_card(
                label="Quality Score",
                value=f"{quality_score * 100:.1f}%",
                subtext=f"{furnish_status} | Security: {security}",
                accent="green" if quality_score >= 0.7 else "gold"
            )
        with m3:
            render_metric_card(
                label="Layout Density",
                value=f"{size_per_bhk:,.0f} SqFt/BHK",
                subtext=f"{bhk} BHK across {size_sqft:,} SqFt",
                accent="sage"
            )
        with m4:
            render_metric_card(
                label="Vintage & Elevation",
                value=f"{age_tier} ({age_years}y)",
                subtext=f"Floor {floor_no} of {total_floors}",
                accent="gold"
            )

        # Technical Underwriting Details Expander
        with st.expander("ℹ️ Technical Model Details & Feature Inputs"):
            st.write("Processed Feature Vector fed into XGBoost Classifier:")
            st.dataframe(input_df)
            st.caption(
                f"Evaluation Model: XGBoost Classifier v1.0 | "
                f"Validation Accuracy: {clf_meta['val_metrics']['accuracy']*100:.2f}% | "
                f"Holdout Test Accuracy: 73.29% | Holdout ROC-AUC: 0.8228"
            )
