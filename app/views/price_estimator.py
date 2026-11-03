"""
ESTATEIQ - Page 3: Current Listed-Price Estimator
------------------------------------------------
Accepts property specifications and generates a central-tendency listed-price estimate
using the Random Forest Regressor pipeline.
Prominently discloses model limitations (R² ≈ 0.00, Test MAE ₹122.81 Lakhs).
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.components.cards import render_hero, render_disclosure_banner, render_metric_card


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


def render_price_estimator(df: pd.DataFrame, reg_pipeline, reg_meta: dict):
    """
    Renders the Current Listed-Price Estimator with prominent scientific candor disclosures.
    """
    render_hero(
        title="Current Listed-Price Estimation Engine",
        tagline="Statistical Asking-Price Benchmark Modeling (Random Forest Regressor)",
        description="Estimates the expected listed asking price in Indian Lakhs based strictly on intrinsic property and civic attributes."
    )

    # 1. Mandatory Performance & Valuation Disclosure Banner
    render_disclosure_banner(
        "<strong>LOW PREDICTIVE SIGNAL DISCLOSURE (R² ≈ -0.0002 | Test MAE: ₹122.81 Lakhs | Validation MAE: ₹122.88 Lakhs):</strong><br>"
        "In this housing dataset, recorded listing prices are uniformly distributed across ₹10L to ₹500L and exhibit near-zero correlation with physical property features. "
        "Because price-per-sqft rates were strictly purged to eliminate circular data leakage, the model essentially predicts near the population empirical mean (~₹254 Lakhs). "
        "<strong>This estimate does NOT represent an independently verified fair market value, professional appraisal, bank underwriting quote, or guaranteed transaction price.</strong>",
        banner_type="warning"
    )

    # 2. Input Form
    st.markdown("### 📋 Enter Property Attributes for Price Estimation")

    with st.form("price_estimator_form"):
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

        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #E9E3D7;'>", unsafe_allow_html=True)

        # Section B: Physical & Structural Dimensions
        st.markdown("##### 📐 Physical Dimensions & Vertical Elevation")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            prop_types = sorted(df["Property_Type"].dropna().unique().tolist())
            selected_prop_type = st.selectbox("Property Type", options=prop_types, index=0)
        with col_t2:
            bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=8, value=3, step=1)
        with col_t3:
            size_sqft = st.number_input("Size in SqFt", min_value=300, max_value=15000, value=1850, step=50)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            floor_no = st.number_input("Floor Number", min_value=0, max_value=100, value=4, step=1)
        with col_f2:
            total_floors = st.number_input("Total Building Floors", min_value=1, max_value=100, value=12, step=1)
        with col_f3:
            age_years = st.number_input("Property Age (Years)", min_value=0, max_value=60, value=5, step=1)

        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #E9E3D7;'>", unsafe_allow_html=True)

        # Section C: Civic Infrastructure & Safety
        st.markdown("##### 🚆 Infrastructure, Quality & Recreational Amenities")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            furnish_status = st.selectbox("Furnished Status", options=["Furnished", "Semi-furnished", "Unfurnished"], index=1)
        with col_c2:
            transit_acc = st.selectbox("Public Transport Access", options=["High", "Medium", "Low"], index=0)
        with col_c3:
            avail_status = st.selectbox("Availability", options=["Ready to Move", "Under Construction"], index=0)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            parking = st.selectbox("Parking Space", options=["Yes", "No"], index=0)
        with col_s2:
            security = st.selectbox("Security Service", options=["Yes", "No"], index=0)
        with col_s3:
            facing = st.selectbox("Facing Orientation", options=["East", "North", "West", "South"], index=0)

        # Quick Amenities
        st.markdown("<strong>Recreational Amenities</strong>", unsafe_allow_html=True)
        a1, a2, a3, a4, a5 = st.columns(5)
        with a1:
            has_gym = st.checkbox("Gym", value=True, key="reg_gym")
        with a2:
            has_pool = st.checkbox("Swimming Pool", value=True, key="reg_pool")
        with a3:
            has_garden = st.checkbox("Lawn / Garden", value=True, key="reg_garden")
        with a4:
            has_playground = st.checkbox("Children's Playground", value=False, key="reg_play")
        with a5:
            has_clubhouse = st.checkbox("Residents Clubhouse", value=True, key="reg_club")

        st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)

        submit_price = st.form_submit_button("🏷️ Estimate Current Listed Price", use_container_width=True)

    # 3. Prediction Output
    if submit_price:
        # Preprocessing & Feature Engineering
        gym_val = 1 if has_gym else 0
        pool_val = 1 if has_pool else 0
        garden_val = 1 if has_garden else 0
        playground_val = 1 if has_playground else 0
        clubhouse_val = 1 if has_clubhouse else 0
        tot_amenities = gym_val + pool_val + garden_val + playground_val + clubhouse_val
        lux_index = round(tot_amenities / 5.0, 4)

        transport_map = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
        furnish_map = {"Furnished": 1.0, "Semi-furnished": 0.5, "Unfurnished": 0.0}
        binary_map = {"Yes": 1.0, "No": 0.0}

        pub_transit_num = transport_map[transit_acc]
        furn_num = furnish_map[furnish_status]
        parking_num = binary_map[parking]
        sec_num = binary_map[security]

        connectivity_score = round((0.50 * pub_transit_num) + (0.25 * 0.6) + (0.25 * 0.5), 4)
        quality_score = round((0.40 * furn_num) + (0.30 * parking_num) + (0.30 * sec_num), 4)
        size_per_bhk = round(size_sqft / bhk, 2)
        rel_floor = round(floor_no / total_floors, 4) if floor_no <= total_floors else np.nan

        age_bins = [-1, 5, 15, 25, 100]
        age_labels = ["New", "Modern", "Mature", "Aging"]
        age_tier = pd.cut([age_years], bins=age_bins, labels=age_labels)[0]

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
            "Nearby_Schools": 6,
            "Nearby_Hospitals": 5,
            "Public_Transport_Accessibility": transit_acc,
            "Parking_Space": parking,
            "Security": security,
            "Facing": facing,
            "Owner_Type": "Owner",
            "Availability_Status": avail_status,
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

        # Run inference
        predicted_price = float(reg_pipeline.predict(input_df)[0])
        test_mae = 122.81
        lower_bound = max(10.0, predicted_price - test_mae)
        upper_bound = min(500.0, predicted_price + test_mae)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 📊 Model Output: Estimated Listed Asking Price")

        p1, p2, p3 = st.columns(3)
        with p1:
            render_metric_card(
                label="Estimated Listed Price",
                value=f"₹{predicted_price:.2f} Lakhs",
                subtext="Central Tendency Prediction",
                accent="gold"
            )
        with p2:
            render_metric_card(
                label="Expected Error Margin (MAE)",
                value=f"± ₹{test_mae:.2f}L",
                subtext=f"Expected Range: ₹{lower_bound:.1f}L – ₹{upper_bound:.1f}L",
                accent="sage"
            )
        with p3:
            render_metric_card(
                label="Dataset Population Mean",
                value=f"₹{df['Price_in_Lakhs'].mean():.2f} Lakhs",
                subtext="Std Dev: ₹141.20L across 250,000 listings",
                accent="green"
            )

        # Contextual explanation
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E9E3D7; border-radius: 8px; padding: 1.25rem; margin-top: 1rem; font-size: 0.9rem; line-height: 1.6; color: #18352A;">
                <strong style="color: #102D23;">Analytical Commentary:</strong><br>
                The model outputs a listed-price estimate of <strong>₹{predicted_price:.2f} Lakhs</strong>. 
                Because the dataset exhibits minimal statistical correlation between physical property attributes and unanchored listing asking prices, 
                this figure closely reflects the training population mean (₹254.72 Lakhs).<br><br>
                <strong>Underwriting Guidance:</strong> In actual real estate acquisitions, price estimation must be anchored to verified recent transaction deeds 
                and micro-market price-per-square-foot benchmarks rather than uncalibrated asking price listings.
            </div>
            """,
            unsafe_allow_html=True
        )
