# ESTATEIQ - Phase 3 (Stage 2): Target Construction & Leakage Defense Audit Report
### Rule-Based Algorithmic Classification Proxy & Current Listed-Price Regression Setup

---

## Executive Summary

This report documents the final leakage audit and completion of **Phase 3 - Stage 2: Target Construction** for the **ESTATEIQ** platform.

Following the final pre-modeling leakage audit:
1. **Classification Target (`Is_Good_Investment`)**: Generated via the objective 4-pillar **Investment Viability Score (IVS)** formula, adhering strictly to the approved threshold of >= 65.0.
2. **Strict Classification Leakage Defense**: The classification feature catalog strictly excludes `Price_in_Lakhs`, `Price_per_SqFt_INR`, `Price_to_Locality_Ratio`, `Locality_Benchmark_Price_per_SqFt`, `Locality_Price_ZScore`, and `Locality_Benchmark_Tier`, alongside all IVS intermediate scores and targets. This ensures the classifier learns investment viability purely from intrinsic property, spatial, and civic attributes, preventing non-linear models from reconstructing the valuation pillar.
3. **Strict Regression Leakage Defense**: `Price_in_Lakhs` is retained as the target for **Current Listed-Price Estimation**. All price-derived features and target components are purged from the regression feature matrix.
4. **Feature Matrix Symmetry**: Both classification (X_class) and regression (X_reg) models operate on the identical set of **34 pure intrinsic features**, with **17 columns** strictly excluded.
5. **Auditing Data Preserved**: All 51 columns (including unit prices, locality benchmarks, score components, and targets) remain intact in the partitioned CSV files for full audit traceability.
6. **Empirical Class Balance**: Yields a well-conditioned **34.81% positive rate in Train**, **34.82% in Val**, and **35.00% in Test**, confirming that no degenerate class imbalance exists.
7. **Baseline Benchmarks**: Naive dummy regressor baselines (Mean and Median) are established across all three partitions.

---

## Part 1: Classification Target Construction - Investment Viability Score (IVS)

> [!IMPORTANT]
> **Domain Classification Target Disclosure**:
> `Is_Good_Investment` is an **algorithmic proxy label** formulated from private-equity real estate underwriting principles. Evaluating a machine learning classifier on this label evaluates the model's capacity to **generalize multi-criteria investment heuristics**, rather than guaranteeing future financial returns.

### 1. Mathematical Formulation & Pillar Weight Allocation

The continuous **Investment Viability Score (IVS)** spans from **0.0 to 100.0** points across 4 objective pillars:

```text
IVS = Score_Valuation + Score_Connectivity + Score_Quality + Score_Vintage_Delivery

Is_Good_Investment = 1 if IVS >= 65.0 else 0
```

| Pillar | Max Points | Metric / Input Formula | Scoring Rules & Point Tiers | Real Estate Investment Rationale |
| :--- | :---: | :--- | :--- | :--- |
| **Pillar 1: Relative Valuation** | **40 Pts** | `Price_to_Locality_Ratio` *(OOF/Train-anchored)* | • **40 pts**: Ratio <= 0.85 (>15% discount to locality par)<br>• **25 pts**: 0.85 < Ratio <= 1.00 (0-15% discount)<br>• **10 pts**: 1.00 < Ratio <= 1.15 (0-15% premium)<br>• **0 pts**: Ratio > 1.15 (>15% premium over neighborhood) | Buying below neighborhood median price per sqft provides an immediate margin of safety against market downturns. |
| **Pillar 2: Infrastructure & Transit** | **30 Pts** | `Connectivity_Score` | • Points = 30.0 * Connectivity_Score<br>*(Observed range in data: [1.50, 30.00])* | Proximity to transit hubs, schools, and hospitals guarantees sustained rental demand and low vacancy risk. |
| **Pillar 3: Asset Quality & Amenities** | **20 Pts** | `Property_Quality_Score` & `Luxury_Amenities_Index` | • Points = (10.0 * Property_Quality_Score) + (10.0 * Luxury_Amenities_Index)<br>*(Observed range in data: [2.00, 20.00])* | Rewards verified parking, 24/7 security, complete furnishing, and lifestyle amenities. `Facing` is strictly excluded. |
| **Pillar 4: Asset Vintage & Delivery** | **10 Pts** | `Age_of_Property` & `Availability_Status` | • **Age Component (5 pts)**:<br>  - 5 pts: Age <= 5y<br>  - 3 pts: 6y <= Age <= 15y<br>  - 1 pt: 16y <= Age <= 25y<br>  - 0 pts: Age > 25y<br>• **Delivery Component (5 pts)**:<br>  - 5 pts: Ready_to_Move<br>  - 2 pts: Under_Construction | Balances structural maintenance CapEx against completion/developer delay risk. `Owner_Type` is strictly excluded. |
| **TOTAL** | **100 Pts** | - | **Continuous Bounded Score (0.0 to 100.0)** | **Decision Threshold: IVS >= 65.0 -> 1, else 0** |

---

## Part 2: Empirical Target Distributions & Statistical Verification

### 1. Classification Target Distribution (`Is_Good_Investment`)

The empirical distribution demonstrates ideal stability across the stratified partitions:

| Partition | Total Records | Negative Class (`0`) | Negative % | Positive Class (`1`) | Positive % | Odds Ratio (Pos : Neg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training (70%)** | **175,000** | 114,084 | 65.19% | 60,916 | **34.81%** | 1 : 1.87 |
| **Validation (15%)** | **37,500** | 24,444 | 65.18% | 13,056 | **34.82%** | 1 : 1.87 |
| **Test (15%)** | **37,500** | 24,376 | 65.00% | 13,124 | **35.00%** | 1 : 1.86 |
| **Combined Total** | **250,000** | **162,904** | **65.16%** | **87,096** | **34.84%** | **1 : 1.87** |

> [!NOTE]
> **Class Balance Assessment**:
> The positive prevalence of **34.84%** aligns precisely with the design target (30%-40%). The dataset exhibits moderate, natural imbalance (roughly 1:2 ratio) which is optimal for binary classification using standard cross-entropy loss, ROC-AUC, PR-AUC, and F1-score without requiring aggressive synthetic oversampling (SMOTE).

### 2. Continuous Investment Viability Score (IVS) Statistical Summary

| Partition | Min | Q25 (25th %) | Median | Mean | Q75 (75th %) | Max | Standard Deviation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Set** | 5.50 | 36.25 | 50.00 | 51.60 | 66.75 | 98.00 | 18.90 |
| **Validation Set** | 8.00 | 36.50 | 50.25 | 51.65 | 66.75 | 97.00 | 18.84 |
| **Test Set** | 6.25 | 36.50 | 50.00 | 51.57 | 66.75 | 97.25 | 18.91 |

- **Theoretical Range**: [5.50, 100.00]
- **Empirical Observed Range**: [5.50, 98.00]
- **Median vs. Mean**: Identical at approximately 50.0 and 51.6, exhibiting a symmetric, bell-shaped distribution centered near 50 points.
- **Threshold Position**: The 65.0 threshold sits at approximately the **65th percentile**, designating the top ~35% of properties as viable investment opportunities.

---

## Part 3: Leakage Prevention & Segregated Feature Catalogs

To prevent any form of target leakage, lookahead bias, or circular reasoning, features are partitioned into two strictly segregated, documented catalogs.

### 1. Classification Feature Catalog (X_class)

The classification feature matrix consists of **34 pure intrinsic model inputs**:

| Feature Group | Column Names | Count | Type | Modeling Role |
| :--- | :--- | :---: | :---: | :--- |
| **Spatial Nominals** | `State`, `City`, `Locality` | 3 | Categorical | Geographic location and regional encoding |
| **Asset Nominals** | `Property_Type`, `Furnished_Status`, `Facing`, `Owner_Type`, `Availability_Status` | 5 | Categorical | Asset attributes (One-Hot Encoded) |
| **Structural Dimensions** | `BHK`, `Size_in_SqFt`, `Floor_No`, `Total_Floors`, `Size_per_BHK`, `Relative_Floor_Position` | 6 | Numerical | Physical size, layout density, vertical floor elevation |
| **Vintage & Depreciation** | `Age_of_Property`, `Age_Group_Tier` | 2 | Num / Cat | Continuous building age and categorical vintage tier |
| **Civic Amenities** | `Nearby_Schools`, `Nearby_Hospitals` | 2 | Integer | Infrastructure proximity counts |
| **Transit & Security** | `Public_Transport_Accessibility`, `Parking_Space`, `Security` | 3 | Categorical | Original infrastructure categories |
| **Atomic Amenities** | `Has_Gym`, `Has_Pool`, `Has_Garden`, `Has_Playground`, `Has_Clubhouse`, `Total_Amenities_Count` | 6 | Binary / Int | Physical recreational amenities |
| **Numerical Mappings** | `Public_Transport_Num`, `Furnished_Status_Num`, `Parking_Space_Num`, `Security_Num`, `Luxury_Amenities_Index` | 5 | Float [0, 1] | Normalized numeric inputs |
| **Domain Composites** | `Connectivity_Score`, `Property_Quality_Score` | 2 | Float [0, 1] | Intrinsic property quality & transit composites |
| **TOTAL** | - | **34** | - | **Input Matrix X_class** |

#### Why Price-Derived Features Are Excluded from Classification:
Following the pre-modeling audit, all price-related columns (`Price_in_Lakhs`, `Price_per_SqFt_INR`, `Price_to_Locality_Ratio`, `Locality_Benchmark_Price_per_SqFt`, `Locality_Price_ZScore`, `Locality_Benchmark_Tier`) are strictly excluded from X_class for three critical reasons:
1. **Prevention of Valuation Pillar Inversion (Trivial Proxy Leakage)**:
   `Score_Valuation` constitutes 40% (40 points) of the total IVS formula and is calculated directly from `Price_to_Locality_Ratio = Price_per_SqFt_INR / Locality_Benchmark_Price_per_SqFt`. If a non-linear decision tree or gradient boosting algorithm (Random Forest, XGBoost, LightGBM) has access to `Price_per_SqFt_INR` and `Locality_Benchmark_Price_per_SqFt`, it can effortlessly create numerical splits approximating this quotient, trivially reverse-engineering the 40-point valuation score. Furthermore, `Locality_Price_ZScore` is a direct standardized proxy of this ratio. Purging all price columns eliminates this mathematical shortcut.
2. **Forcing Pure Intrinsic Real Estate Underwriting**:
   An investment recommendation model should learn whether an asset's intrinsic physical, spatial, structural, and infrastructure attributes (spaciousness per BHK, vertical elevation, building vintage, transit accessibility, security, parking, amenities) merit investment viability. Excluding listed price forces the model to discover deep property and civic value drivers rather than relying on asking prices.
3. **Insulation from Asking Price Noise**:
   As proven during Phase 2 EDA, the raw listed prices in this dataset exhibit near-zero correlation with physical property features (r ~ 0.00). Removing listed price from classification protects the classifier from overfitting to uncalibrated asking price noise.

#### Columns Strictly Excluded from Classification (17 Columns):
The following **17 columns** are strictly dropped from X_class:
1. `Is_Good_Investment`: **Classification Target Variable**.
2. `Price_in_Lakhs`: Direct listed price (prevents price leakage).
3. `Price_per_SqFt_INR`: High-precision unit price (prevents valuation reconstruction).
4. `Price_per_SqFt`: Original uncalibrated unit price (contains 2,534 zeros).
5. `Price_to_Locality_Ratio`: Direct discrete ratio defining `Score_Valuation` cutoffs.
6. `Locality_Benchmark_Price_per_SqFt`: Neighborhood median price benchmark rate.
7. `Locality_Price_ZScore`: Standardized price spread relative to locality.
8. `Locality_Benchmark_Tier`: Audit governance category string.
9. `Investment_Viability_Score`: Continuous composite score (direct sum formula).
10. `Score_Valuation`: Intermediate IVS scoring component (Pillar 1).
11. `Score_Connectivity`: Intermediate IVS scoring component (Pillar 2).
12. `Score_Quality`: Intermediate IVS scoring component (Pillar 3).
13. `Score_Vintage_Delivery`: Intermediate IVS scoring component (Pillar 4).
14. `ID`: Database primary key; prevents arbitrary index memorization.
15. `Is_Floor_Inconsistent`: Cleaning audit flag; not a predictive property attribute.
16. `Year_Built`: Deterministically collinear with `Age_of_Property` (rho = -1.0).
17. `Amenities`: Raw unparsed string; superseded by atomic flags.

---

### 2. Regression Feature Catalog (X_reg)

The regression task is **Current Listed-Price Estimation** (predicting `Price_in_Lakhs`).

> [!CAUTION]
> **Zero Price-Leakage Directive**:
> If any price-derived feature (such as `Price_per_SqFt_INR`) entered the regression matrix, the model would trivially recover `Price_in_Lakhs = (Price_per_SqFt_INR * Size_in_SqFt) / 100,000`, causing 100% artificial target leakage (R² = 1.0). Therefore, **all price-derived columns are strictly purged**.

The regression feature matrix consists of **34 pure physical, structural, and civic inputs** (identical to X_class):

| Feature Group | Column Names | Count | Type | Modeling Role |
| :--- | :--- | :---: | :---: | :--- |
| **Spatial Nominals** | `State`, `City`, `Locality` | 3 | Categorical | Geographic location and regional encoding |
| **Asset Nominals** | `Property_Type`, `Furnished_Status`, `Facing`, `Owner_Type`, `Availability_Status` | 5 | Categorical | Asset attributes (One-Hot Encoded) |
| **Structural Dimensions** | `BHK`, `Size_in_SqFt`, `Floor_No`, `Total_Floors`, `Size_per_BHK`, `Relative_Floor_Position` | 6 | Numerical | Physical size, layout density, vertical floor elevation |
| **Vintage & Depreciation** | `Age_of_Property`, `Age_Group_Tier` | 2 | Num / Cat | Continuous building age and categorical vintage tier |
| **Civic Amenities** | `Nearby_Schools`, `Nearby_Hospitals` | 2 | Integer | Infrastructure proximity counts |
| **Transit & Security** | `Public_Transport_Accessibility`, `Parking_Space`, `Security` | 3 | Categorical | Original infrastructure categories |
| **Atomic Amenities** | `Has_Gym`, `Has_Pool`, `Has_Garden`, `Has_Playground`, `Has_Clubhouse`, `Total_Amenities_Count` | 6 | Binary / Int | Physical recreational amenities |
| **Numerical Mappings** | `Public_Transport_Num`, `Furnished_Status_Num`, `Parking_Space_Num`, `Security_Num`, `Luxury_Amenities_Index` | 5 | Float [0, 1] | Normalized numeric inputs |
| **Domain Composites** | `Connectivity_Score`, `Property_Quality_Score` | 2 | Float [0, 1] | Intrinsic property quality & transit composites |
| **TOTAL** | - | **34** | - | **Input Matrix X_reg** |

#### Columns Strictly Excluded from Regression (17 Columns):
The following **17 columns** are strictly dropped from X_reg:
1. `Price_in_Lakhs`: **Regression Target Variable**.
2. `Price_per_SqFt_INR`: Direct price per sqft rate (mathematical leakage).
3. `Price_per_SqFt`: Original unit price (mathematical leakage).
4. `Price_to_Locality_Ratio`: Price-derived ratio (leakage).
5. `Locality_Price_ZScore`: Price-derived standardized spread (leakage).
6. `Locality_Benchmark_Price_per_SqFt`: Neighborhood price benchmark rate.
7. `Locality_Benchmark_Tier`: Benchmark audit tier.
8. `Is_Good_Investment`: Classification Target.
9. `Investment_Viability_Score`: Target-derived composite score.
10. `Score_Valuation`: Target-derived score component.
11. `Score_Connectivity`: Target-derived score component.
12. `Score_Quality`: Target-derived score component.
13. `Score_Vintage_Delivery`: Target-derived score component.
14. `ID`: Database primary key sequence.
15. `Is_Floor_Inconsistent`: Cleaning audit flag.
16. `Year_Built`: Collinear with `Age_of_Property`.
17. `Amenities`: Raw unparsed string.

---

## Part 4: Mandatory Regression Baselines (Current Listed-Price Estimation)

To fulfill the explicit regression framing approved in the proposal, naive dummy baselines were evaluated to establish the performance benchmark for future supervised models.

### 1. Training Set Price Baselines
- **Training Set Mean Price**: **254.7153 Lakhs** (INR 2.55 Crore)
- **Training Set Median Price**: **254.1400 Lakhs** (INR 2.54 Crore)

### 2. Baseline Model Performance across Partitions

| Partition | Baseline Strategy | MAE (Lakhs) | RMSE (Lakhs) | R² Score |
| :--- | :--- | :---: | :---: | :---: |
| **Training (175k)** | Predict Training Mean (254.72L) | 122.27 Lakhs | 141.20 Lakhs | 0.000000 |
| **Training (175k)** | Predict Training Median (254.14L) | 122.27 Lakhs | 141.20 Lakhs | -0.000017 |
| **Validation (37.5k)** | Predict Training Mean (254.72L) | 122.87 Lakhs | 141.68 Lakhs | -0.000013 |
| **Validation (37.5k)** | Predict Training Median (254.14L) | 122.87 Lakhs | 141.68 Lakhs | -0.000000 |
| **Test (37.5k)** | Predict Training Mean (254.72L) | 122.81 Lakhs | 141.71 Lakhs | -0.000006 |
| **Test (37.5k)** | Predict Training Median (254.14L) | 122.80 Lakhs | 141.71 Lakhs | -0.000002 |

> [!NOTE]
> **Baseline Benchmark Takeaways**:
> - A naive predictor predicting the training mean achieves an **MAE of ~122.8 Lakhs** and an **RMSE of ~141.7 Lakhs** on validation and test sets.
> - As established in the EDA report, the raw features exhibit very low linear correlations with listed price (r ~ 0.00). Any supervised regression algorithm (Ridge, Random Forest, XGBoost, LightGBM) must demonstrate statistically significant error reductions below the **MAE = 122.8 Lakhs** baseline floor to prove that it is learning actionable signal.

---

## Part 5: Quality Assurance & Validation Results

All mandatory validation checks were executed and confirmed:

| Check # | Validation Criterion | Test Condition / Rule | Result | Status |
| :---: | :--- | :--- | :---: | :---: |
| **1** | **IVS Bounded Interval** | 0.0 <= Investment_Viability_Score <= 100.0 | Observed [5.50, 98.00] | **PASSED** |
| **2** | **Target Threshold Adherence** | Is_Good_Investment == 1 <=> IVS >= 65.0 | 100.0% exact match | **PASSED** |
| **3** | **Missing Target Values** | Null_Count(Target_Columns) == 0 | 0 nulls across Train, Val, Test | **PASSED** |
| **4** | **Class Balance Sanity** | Positive prevalence between 30.0% and 40.0% | 34.81% to 35.00% | **PASSED** |
| **5** | **Partition ID Invariance** | Record IDs unchanged and mutually disjoint | 0 overlap; N = 250,000 | **PASSED** |
| **6** | **Classification Leakage Purity** | Zero price-derived or target-derived columns in X_class | 0 leak columns detected (34 features) | **PASSED** |
| **7** | **Regression Leakage Purity** | Zero price-derived or target-derived columns in X_reg | 0 leak columns detected (34 features) | **PASSED** |
| **8** | **Feature Matrix Symmetry** | Exact feature match between X_class and X_reg | 100% identical 34 intrinsic features | **PASSED** |
| **9** | **Neutral Factor Exclusion** | `Facing` and `Owner_Type` excluded from scoring formula | 100% verified excluded | **PASSED** |

---

## Part 6: Output Artifacts Summary

The updated datasets are persisted to disk with **51 total columns**:

| Dataset File | Observations | Features | File Path | File Size |
| :--- | :---: | :---: | :--- | :---: |
| **Training Partition** | **175,000** | **51** | [`data/processed/train_features.csv`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/data/processed/train_features.csv) | ~51.9 MB |
| **Validation Partition** | **37,500** | **51** | [`data/processed/val_features.csv`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/data/processed/val_features.csv) | ~11.1 MB |
| **Test Partition** | **37,500** | **51** | [`data/processed/test_features.csv`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/data/processed/test_features.csv) | ~11.1 MB |

---

## Phase 3 Completion Status

**Phase 3 (Feature Engineering, Target Construction & Final Leakage Defense Audit) is 100% complete.**
- No machine learning models have been trained.
- No hyperparameters have been tuned.
- No MLflow experiment runs have been initialized.
- Raw and cleaned source datasets remain completely unmodified.

*Awaiting your review and approval before proceeding to Phase 4 (Classification & Regression Modeling with MLflow Tracking).*
