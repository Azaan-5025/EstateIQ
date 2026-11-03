# ESTATEIQ — Phase 3: Feature Engineering & Data Partitioning Audit Report
### Leakage-Safe Feature Pipelines, Stratified Splitting & Locality Benchmarking

---

## 🏛️ Executive Summary

This report documents the execution of **Phase 3 (Stage 1 & Stage 2) — Feature Engineering, Data Partitioning, and Target Construction** for the **ESTATEIQ** platform. 

All operations were executed in accordance with the final approved proposal:
- The cleaned dataset was partitioned into **70% Training (175,000)**, **15% Validation (37,500)**, and **15% Test (37,500)** partitions, stratified across 126 `(City, Property_Type)` strata cells.
- Exact numerical mappings and composite domain indicators (`Connectivity_Score`, `Property_Quality_Score`, `Size_per_BHK`, `Relative_Floor_Position`, `Age_Group_Tier`) were implemented with strict bounded intervals.
- Locality pricing benchmarks were computed using a **5-fold Out-Of-Fold (OOF)** estimation architecture on the training fold, and master static lookups with hierarchical fallbacks were applied to validation and test partitions.
- `Facing` was preserved purely as an optional categorical feature and strictly excluded from quality and viability scoring.
- Classification target `Is_Good_Investment` was constructed via the 4-pillar Investment Viability Score (IVS) with strict leakage isolation.
- Regression target `Price_in_Lakhs` was designated for Current Listed-Price Estimation with complete price-leakage purging.

---

## 📊 Partition Dimensions & Stratification Verification

| Partition | Observation Count | Percentage | File Path | File Size |
| :--- | :---: | :---: | :--- | :---: |
| **Training Set** | **175,000** | **70.0%** | [`data/processed/train_features.csv`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/data/processed/train_features.csv) | ~49.5 MB |
| **Validation Set** | **37,500** | **15.0%** | [`data/processed/val_features.csv`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/data/processed/val_features.csv) | ~10.6 MB |
| **Test Set** | **37,500** | **15.0%** | [`data/processed/test_features.csv`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/data/processed/test_features.csv) | ~10.6 MB |
| **Total** | **250,000** | **100.0%** | — | **~70.7 MB** |

### Stratification Sanity Check across 126 Strata Cells `(City, Property_Type)`:
- Total unique strata combinations: **126 cells** (42 Cities x 3 Property Types).
- Distribution: Each stratum contains between **1,338 and 1,438 records** in the training partition, and **286 to 308 records** in both validation and test partitions.
- Mutual Exclusivity: Train, validation, and test index intersections are strictly **0** (mutually disjoint).

---

## 🛠️ Features Engineered & Exact Numerical Mappings

The feature pipeline expanded the schema from **31 columns** (cleaned dataset) to **51 columns** (+20 newly engineered features, score components, and targets):

### 1. Exact Categorical-to-Numerical Mappings

| Feature Name | Source Column | Mapping Values & Scale | Theoretical Bounds |
| :--- | :--- | :--- | :---: |
| `Public_Transport_Num` | `Public_Transport_Accessibility` | High -> 1.0, Medium -> 0.5, Low -> 0.0 | [0.0, 1.0] |
| `Furnished_Status_Num` | `Furnished_Status` | Furnished -> 1.0, Semi-furnished -> 0.5, Unfurnished -> 0.0 | [0.0, 1.0] |
| `Parking_Space_Num` | `Parking_Space` | Yes -> 1.0, No -> 0.0 | [0.0, 1.0] |
| `Security_Num` | `Security` | Yes -> 1.0, No -> 0.0 | [0.0, 1.0] |
| `Luxury_Amenities_Index` | `Total_Amenities_Count` | Total_Amenities_Count / 5.0 | [0.20, 1.00] |

### 2. Composite Domain Features

| Feature Name | Mathematical Formula | Range in Data | Business Rationale |
| :--- | :--- | :---: | :--- |
| `Connectivity_Score` | (0.50 * Transport) + (0.25 * Schools/10) + (0.25 * Hospitals/10) | [0.05, 1.00] | Urban accessibility & tenant occupancy driver |
| `Property_Quality_Score` | (0.40 * Furnish) + (0.30 * Parking) + (0.30 * Security) | [0.00, 1.00] | Physical asset readiness index (`Facing` strictly excluded) |
| `Size_per_BHK` | Size_in_SqFt / BHK | [100.0, 5000.0] | Spaciousness quotient; layout density |
| `Relative_Floor_Position` | Floor_No / Total_Floors (where `Floor_No` is valid) | [0.00, 1.00] | Vertical building position (preserves `NaN` for 116,304 invalid records) |
| `Age_Group_Tier` | Binned: New (<=5y), Modern (6-15y), Mature (16-25y), Aging (>25y) | 4 Levels | Non-linear depreciation & maintenance bracket |

### 3. Leakage-Safe Locality Relative Pricing Features

| Feature Name | Derivation Method | Formula | Role |
| :--- | :--- | :--- | :--- |
| `Locality_Benchmark_Price_per_SqFt` | 5-Fold OOF in Train; Master Static Lookup in Val & Test | Median(Price_per_SqFt_INR) with hierarchical fallback | Neighborhood benchmark price anchor |
| `Price_to_Locality_Ratio` | Ratio against benchmark | Price_per_SqFt_INR / Locality_Benchmark_Price_per_SqFt | Relative valuation discount (<1.0) or premium (>1.0) |
| `Locality_Price_ZScore` | Standardized spread | (Price_per_SqFt_INR - Locality_Mean) / Locality_Std | Statistical spread from neighborhood mean |
| `Locality_Benchmark_Tier` | Audit category | Locality, City_Fallback, State_Fallback, Global_Fallback | Audit traceability of benchmark level |

### 4. Investment Viability Score & Target Components

| Feature Name | Pillar | Points / Formula | Role |
| :--- | :--- | :---: | :--- |
| `Score_Valuation` | Pillar 1: Valuation | 40, 25, 10, 0 pts based on Price_to_Locality_Ratio | Margin of safety score |
| `Score_Connectivity` | Pillar 2: Connectivity | 30.0 * Connectivity_Score | Urban accessibility points |
| `Score_Quality` | Pillar 3: Quality | (10.0 * Quality) + (10.0 * Luxury_Amenities_Index) | Physical asset quality points |
| `Score_Vintage_Delivery` | Pillar 4: Vintage & Delivery | Age Points (0-5) + Delivery Points (2 or 5) | Structural vintage & completion risk |
| `Investment_Viability_Score` | Total Score | Sum of Pillars 1 to 4 [Theoretical: 5.5 to 100.0] | Continuous viability rating |
| `Is_Good_Investment` | Binary Target | 1 if IVS >= 65.0 else 0 | Classification Target |

---

## 🛡️ Locality Out-Of-Fold (OOF) & Fallback Audit

| Partition | Locality Tier (Primary) | City Fallback | State Fallback | Global Fallback | Total Records |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training (OOF)** | **175,000 (100.00%)** | 0 | 0 | 0 | **175,000** |
| **Validation (Master)** | **37,500 (100.00%)** | 0 | 0 | 0 | **37,500** |
| **Test (Master)** | **37,500 (100.00%)** | 0 | 0 | 0 | **37,500** |

*Finding*: Across all 500 localities, sample counts in training were sufficient (N >= 10) such that **100.0% of records in Train, Val, and Test utilized primary Locality-level benchmarks**, with zero reliance on fallback tiers.

---

## 🔍 Missing Value & Integrity Audit

- **Newly Engineered Features & Targets**: **0 missing values** across all partitions for all 18 newly added numerical, categorical, and target columns.
- **Expected Missing Values (Strictly Preserved)**:
  - `Floor_No`: Train: 81,399 nulls (46.51%), Val: 17,493 nulls (46.65%), Test: 17,412 nulls (46.43%).
  - `Relative_Floor_Position`: Identical null counts matching `Floor_No`.
- **Immutable Variables Verification**:
  - `ID`: Strictly sequential, 0 duplicates, 0 nulls across all partitions.
  - `Price_in_Lakhs`: 100.0% byte-for-byte identical to raw source.
  - `Size_in_SqFt`: 100.0% byte-for-byte identical to raw source.

---

## 🚦 Phase 3 Status

Phase 3 is fully executed and validated:
- Partitioned datasets with target columns are exported to `data/processed/`.
- Target construction and leakage prevention details are fully documented in [`reports/target_construction_report.md`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/target_construction_report.md).
