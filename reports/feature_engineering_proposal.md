# ESTATEIQ — Phase 3: Feature Engineering & Target Design Proposal (Final Refinement)
### Architecture Blueprint for Feature Engineering, Rule-Based Investment Classification & Current Listed-Price Estimation

---

## 🏛️ Executive Summary & Context

This proposal establishes the technical specification for **Phase 3** of the **ESTATEIQ** platform.

Following Phase 1 data validation audits and Phase 2 Exploratory Data Analysis ($N = 250,000$ records, 31 attributes in `data/processed/india_housing_cleaned.csv`), this refined proposal incorporates two critical architectural updates:
1. **Removal of `Facing` from the Investment Viability Score**:
   - `Facing` is completely eliminated from the algorithmic investment scoring formula. Empirical analysis in EDA Question 15 established that East/North (Vastu-preferred) properties command no price premium over South/West units (means within $0.8\%$, medians identical). Assigning an ungrounded preference score is avoided. `Facing` is preserved purely as an optional categorical feature for ML estimators.
   - `Property_Quality_Score` is rebalanced strictly across verified physical amenities: `Furnished_Status`, `Parking_Space`, and `Security`.
2. **Current Listed-Price Estimation Framing**:
   - The regression modeling task is explicitly defined as **Current Listed-Price Estimation**, rather than verified "Fair-Market-Value Prediction".
   - The target `Price_in_Lakhs` represents the recorded price in the provided dataset, not an independently appraised physical valuation.
   - Because EDA confirmed weak empirical correlations between available features and price, expected predictive performance may be constrained. Evaluation will mandate comparison against simple statistical baselines (training-set mean and median).

---

## ⚠️ Fundamental Dataset Limitations & Analytical Realities

Before specifying feature transformations and target designs, the structural boundaries of the underlying dataset (`india_housing_prices.csv`) must be explicitly acknowledged:

1. **Feature Independence Artifact**:
   - In physical housing markets, property size and price exhibit strong positive correlation ($r \approx 0.65\text{ to }0.85$). In this dataset, empirical correlation is $r = -0.0019$ ($p = 0.342$).
   - Features (`Price`, `Size`, `BHK`, `Amenities`, `Age`) were drawn independently from bounded uniform distributions.
2. **Recorded Price vs. Independent Market Valuation**:
   - `Price_in_Lakhs` represents the raw listed prices supplied in the dataset. It does not reflect certified registry deeds, bank appraisals, or verified transaction settlements.
3. **Absence of Historical Longitudinal Data**:
   - The dataset contains zero time-series records, historical transaction prices, appreciation curves, or rental yields.
4. **Absence of Realized Investment Outcomes**:
   - There are no observed profit/loss records, resale yields, or default histories. Consequently, any classification label is strictly an **engineered proxy heuristic**, not an empirical ground-truth market outcome.
5. **Physical & Structural Inconsistencies**:
   - As audited in Phase 1, 46.52% of records exhibit `Floor_No > Total_Floors`, and single-family villas/independent houses possess floor counts up to 30.
6. **Implications for Machine Learning**:
   - ML models cannot discover genuine macroeconomic market dynamics that do not exist in the raw data.
   - Model training instead measures the algorithm's capacity to **learn, approximate, and generalize non-linear composite domain heuristics** without overfitting.

---

## 🛠️ Part A: Feature Engineering & Exact Mathematical Mappings

### 1. Exact Numerical Mappings for Categorical Attributes

To guarantee reproducibility and uniform scales across all composite scores, categorical variables are mapped to exact, bounded numerical ranges $[0.0, 1.0]$:

#### Mapping 1.1: `Public_Transport_Accessibility`
$$\text{Map}(\text{Public\_Transport\_Accessibility}) = \begin{cases}
1.0 & \text{if 'High'} \\
0.5 & \text{if 'Medium'} \\
0.0 & \text{if 'Low'}
\end{cases} \quad \in [0.0, 1.0]$$

#### Mapping 1.2: `Furnished_Status`
$$\text{Map}(\text{Furnished\_Status}) = \begin{cases}
1.0 & \text{if 'Furnished'} \\
0.5 & \text{if 'Semi-furnished'} \\
0.0 & \text{if 'Unfurnished'}
\end{cases} \quad \in [0.0, 1.0]$$

#### Mapping 1.3: `Parking_Space` & `Security`
$$\text{Map}(\text{Parking\_Space}) = \begin{cases} 1.0 & \text{if 'Yes'} \\ 0.0 & \text{if 'No'} \end{cases} \quad \in [0.0, 1.0]$$
$$\text{Map}(\text{Security}) = \begin{cases} 1.0 & \text{if 'Yes'} \\ 0.0 & \text{if 'No'} \end{cases} \quad \in [0.0, 1.0]$$

#### Mapping 1.4: `Facing` (Optional ML Feature Only)
- **Role**: `Facing` (East, North, South, West) is retained strictly as a nominal categorical feature for one-hot encoding in ML pipelines.
- **Exclusion from Scoring**: It is **completely excluded from all investment viability scores and quality indices**, as our empirical EDA confirmed no pricing differentiation across orientations ($p > 0.05$).

#### Mapping 1.5: `Luxury_Amenities_Index`
$$\text{Luxury\_Amenities\_Index} = \frac{\text{Total\_Amenities\_Count}}{5.0} \quad \in [0.20, 1.00]$$
*Domain Note*: Since `Total_Amenities_Count` spans integers $1$ through $5$, this index is strictly bounded within $[0.20, 1.00]$.

---

### 2. Composite Domain Features

#### Feature 2.1: `Connectivity_Score` (Civic & Transit Index)
- **Formula**:
  $$\text{Connectivity\_Score} = \left(0.50 \times \text{Map}(\text{Public\_Transport\_Accessibility})\right) + \left(0.25 \times \frac{\text{Nearby\_Schools}}{10.0}\right) + \left(0.25 \times \frac{\text{Nearby\_Hospitals}}{10.0}\right)$$
- **Scale**: Strictly bounded continuous range $[0.05, 1.00]$.
- **Business Meaning**: Measures urban accessibility and infrastructure proximity.
- **Source Columns**: `Public_Transport_Accessibility`, `Nearby_Schools`, `Nearby_Hospitals`.
- **Leakage Risk**: **Zero Risk**. Computed strictly at observation level using static theoretical bounds.

#### Feature 2.2: `Property_Quality_Score` (Physical Asset Index — Refined)
- **Formula**:
  $$\text{Property\_Quality\_Score} = \left(0.40 \times \text{Map}(\text{Furnished\_Status})\right) + \left(0.30 \times \text{Map}(\text{Parking\_Space})\right) + \left(0.30 \times \text{Map}(\text{Security})\right)$$
- **Scale**: Strictly bounded continuous range $[0.00, 1.00]$.
- **Business Meaning**: Summarizes interior condition, parking infrastructure, and security readiness without subjective orientation bias.
- **Source Columns**: `Furnished_Status`, `Parking_Space`, `Security`.
- **Leakage Risk**: **Zero Risk**. Deterministic row-wise computation.

#### Feature 2.3: `Size_per_BHK` (Spaciousness Quotient)
- **Formula**:
  $$\text{Size\_per\_BHK} = \frac{\text{Size\_in\_SqFt}}{\text{BHK}}$$
- **Scale**: Continuous range $[100.0, 5000.0]$ SqFt/bedroom.
- **Business Meaning**: Distinguishes luxury spacious layouts from cramped compact floorplans.
- **Source Columns**: `Size_in_SqFt`, `BHK`.
- **Leakage Risk**: **Zero Risk**. Row-level division.

#### Feature 2.4: `Relative_Floor_Position` (Vertical Index)
- **Formula**:
  $$\text{Relative\_Floor\_Position} = \begin{cases}
  \frac{\text{Floor\_No}}{\text{Total\_Floors}} & \text{if } \text{Floor\_No is not null} \\
  \text{NaN} & \text{if } \text{Floor\_No is null (inconsistent record)}
  \end{cases}$$
- **Scale**: Continuous range $[0.0, 1.0]$ for valid rows; `NaN` preserved for the 116,304 handled records.
- **Business Meaning**: Elevation index within multi-story developments.
- **Source Columns**: `Floor_No`, `Total_Floors`.
- **Leakage Risk**: **Zero Risk**.

---

### 3. Locality-Relative Pricing & Out-of-Fold Leakage Defense

```
                               TRAINING PARTITION (70%)
                         ┌──────────────────────────────────┐
                         │   5-Fold Cross-Validation Split  │
                         └────────────────┬─────────────────┘
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
   ┌───────────────────────────┐                     ┌───────────────────────────┐
   │         Fold k            │                     │       Folds \ {k}         │
   │      (Validation)         │                     │        (Fitting)          │
   └─────────────┬─────────────┘                     └─────────────┬─────────────┘
                 │                                                 │
                 │   Calculate OOF Locality Median                 │
                 │   Price per SqFt without Fold k                 │
                 │   ◄─────────────────────────────────────────────┘
                 ▼
   ┌───────────────────────────┐
   │ Leakage-Free Relative     │
   │ Pricing for Train Rows    │
   └───────────────────────────┘
```

#### Protocol 1: Out-of-Fold (OOF) Benchmark Fitting on Training Rows
1. The 70% Training Partition ($N = 175,000$) is divided into **$K = 5$ internal folds**.
2. For each fold $k \in \{1, \dots, 5\}$, locality medians are calculated exclusively using the remaining 4 folds ($\text{Train} \setminus \text{Fold}_k$).
3. Training rows in $\text{Fold}_k$ receive locality benchmarks derived strictly out-of-fold. Thus, **no training record contributes to its own neighborhood pricing benchmark**.

#### Protocol 2: Static Benchmark Application on Validation & Test Rows
1. A **single static master locality lookup table** is computed using the **entire 100% Training Partition**.
2. This master lookup table is mapped onto Validation (15%) and Test (15%) partitions without recalculation.

#### Protocol 3: Hierarchical Fallback for Sparse or Unseen Localities
If a locality in validation/test is absent from the training partition or contains fewer than $N_{\text{min}} = 10$ training observations:
$$\text{Locality\_Benchmark}_i = \begin{cases}
\text{Median}(\text{Locality}_j^{\text{Train}}) & \text{if count in Train } \ge 10 \\
\text{Median}(\text{City}_c^{\text{Train}}) & \text{else if City count in Train } \ge 50 \\
\text{Median}(\text{State}_s^{\text{Train}}) & \text{else if State count in Train } \ge 100 \\
\text{Median}(\text{Overall}^{\text{Train}}) & \text{final fallback}
\end{cases}$$

#### Resulting Engineered Relative Features:
- **`Price_to_Locality_Ratio`**:
  $$\text{Price\_to\_Locality\_Ratio}_i = \frac{\text{Price\_per\_SqFt\_INR}_i}{\text{Locality\_Benchmark}_i}$$
- **`Locality_Price_ZScore`**:
  $$\text{Locality\_Price\_ZScore}_i = \frac{\text{Price\_per\_SqFt\_INR}_i - \mu_{\text{Locality}, i}^{\text{OOF/Train}}}{\sigma_{\text{Locality}, i}^{\text{OOF/Train}} + \epsilon}$$

---

## 🎯 Part B: Classification Target Design — Rule-Based Proxy (`Is_Good_Investment`)

> [!IMPORTANT]
> **Definitive Proxy Label Disclosure**:
> `Is_Good_Investment` is an **algorithmic proxy label** formulated from private-equity real estate underwriting principles. Evaluating an ML model on this target measures the model's ability to **replicate and generalize the multi-criteria heuristic scoring rules**, **NOT whether the asset will achieve real-world financial profitability**.

### 1. Investment Viability Scoring (IVS) Formulation

The **Investment Viability Score (IVS)** assigns a continuous rating from $0.0 \text{ to } 100.0$ points across 4 objective pillars:

$$\text{IVS} = S_{\text{Valuation}} + S_{\text{Connectivity}} + S_{\text{Quality}} + S_{\text{Vintage\&Delivery}}$$

#### Refined Pillar Allocation Table

| Pillar | Max Points | Metric / Input | Exact Scoring Rules | Business Justification |
| :--- | :---: | :--- | :--- | :--- |
| **1. Relative Valuation** | **40 Pts** | `Price_to_Locality_Ratio` *(OOF/Train-anchored)* | • **40 pts**: $\text{Ratio} \le 0.85$ ($>15\%$ discount to locality par)<br>• **25 pts**: $0.85 < \text{Ratio} \le 1.00$ ($0\text{–}15\%$ discount)<br>• **10 pts**: $1.00 < \text{Ratio} \le 1.15$ ($0\text{–}15\%$ premium)<br>• **0 pts**: $\text{Ratio} > 1.15$ ($>15\%$ premium over neighborhood) | Purchasing below neighborhood benchmark price per square foot provides an immediate margin of safety. |
| **2. Infrastructure & Transit** | **30 Pts** | `Connectivity_Score` | • Points = $30.0 \times \text{Connectivity\_Score}$<br>*(Points span $[1.5, 30.0]$)* | High transit accessibility, schools, and hospitals drive long-term rental demand and low vacancy. |
| **3. Asset Quality & Amenities** | **20 Pts** | `Property_Quality_Score` & `Luxury_Amenities_Index` | • Points = $(10.0 \times \text{Property\_Quality\_Score}) + (10.0 \times \text{Luxury\_Amenities\_Index})$<br>*(Points span $[2.0, 20.0]$)* | Rewards dedicated parking, 24/7 security, and complete recreational amenities (facing excluded). |
| **4. Asset Vintage & Delivery Risk** | **10 Pts** | `Age_of_Property` & `Availability_Status` | • **Age Component (5 pts)**:<br>  - 5 pts: $\text{Age} \le 5\text{y}$<br>  - 3 pts: $6\text{y} \le \text{Age} \le 15\text{y}$<br>  - 1 pt: $16\text{y} \le \text{Age} \le 25\text{y}$<br>  - 0 pts: $\text{Age} > 25\text{y}$<br>• **Delivery Component (5 pts)**:<br>  - 5 pts: `Ready_to_Move`<br>  - 2 pts: `Under_Construction` | Balances physical structural maintenance CapEx against completion/developer delivery delay risk. |
| **TOTAL** | **100 Pts** | — | **Continuous Investment Score ($0.0 \text{ to } 100.0$)** | — |

---

### 2. Decision Threshold & Unverified Class Balance Expectation

$$\text{Is\_Good\_Investment} = \begin{cases}
1 & \text{if } \text{IVS} \ge 65.0 \\
0 & \text{if } \text{IVS} < 65.0
\end{cases}$$

- **Unverified Design Expectation**:
  The threshold of $\ge 65.0$ is intended to select properties combining relative value with functional infrastructure. While a positive prevalence of roughly $30\%\text{–}40\%$ is anticipated, this is an **unverified design expectation**. The true positive rate will be audited empirically post-generation.
- **Model Shielding**:
  All intermediate score components ($\text{IVS}$, $S_{\text{Valuation}}$, etc.) will be **strictly excluded** from the feature matrix $\mathbf{X}$ to prevent label leakage.

---

## 📈 Part C: Regression Strategy — Current Listed-Price Estimation & Valuation Scenario Modeling

> [!IMPORTANT]
> **Explicit Regression Framing**:
> 1. **Target**: The regression target is `Price_in_Lakhs`, which represents the **recorded listed price in the supplied dataset**.
> 2. **Not a Verified Market Valuation**: This target is **not an independently appraised fair-market value**.
> 3. **Expected Performance Limitations**: EDA revealed very weak linear correlations between raw features and price ($r \approx 0$). Consequently, machine learning models may exhibit modest predictive capacity (e.g., lower $R^2$ or higher relative errors than seen in organic real estate markets).
> 4. **Mandatory Baseline Comparison**: Model evaluation will mandate benchmark comparisons against naive baselines (predicting the training-set **mean** and **median** via `DummyRegressor`) to verify whether any algorithm extracts meaningful signal.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ESTATEIQ VALUATION ENGINE                              │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│      TIER 1: MACHINE LEARNING PIPELINE   │       TIER 2: SCENARIO CALCULATOR           │
│      (Supervised Regression)             │       (User-Controlled Assumptions)         │
├──────────────────────────────────────────┼─────────────────────────────────────────────┤
│  Task    : Current Listed-Price          │  Compounds: 5-Year Outlook                  │
│            Estimation                    │  Input    : Estimated Listed Price          │
│  Target  : Recorded `Price_in_Lakhs`     │  Output   : 3 Transparent Wealth Scenarios  │
│  Baseline: DummyRegressor (Mean/Median)  │            (Conservative, Moderate, Bull)   │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
```

---

### 1. The Transparent 5-Year Valuation Scenario Calculator

To address the project's multi-year investment objective without training models on artificial data, the application will provide an interactive **5-Year Scenario Calculator**:

$$\text{Estimated\_Future\_Value}_{5Y} = \text{Current\_Estimated\_Price} \times (1 + \text{CAGR})^5$$

| Scenario Tier | Annual CAGR ($r$) | 5-Year Multiplier $(1+r)^5$ | Explicit Planning Assumptions (Not Market Predictions) |
| :--- | :---: | :---: | :--- |
| **Conservative Scenario** | **$4.5\%$ per annum** | **$1.246\times$** ($+24.6\%$) | Tracks baseline inflation (CPI floor) with zero real capital appreciation. |
| **Moderate Scenario** | **$6.5\%$ per annum** | **$1.370\times$** ($+37.0\%$) | Reflects long-term Indian urban housing index averages (inflation + modest real growth). |
| **Optimistic Scenario** | **$8.5\%$ per annum** | **$1.504\times$** ($+50.4\%$) | Simulates rapid infrastructure expansion (new transit lines, commercial IT corridor). |

- **User Adjustability**: Users can manually modify annual growth assumptions in the Streamlit UI.
- **Mandatory User Disclosure**:
  > *"Scenario projections are based on user-selected financial compounding assumptions and do not constitute guaranteed investment returns or market forecasts."*

---

## 🔀 Part D: Data Partitioning & Stratification Verification

### 1. Partition Breakdown
- **Training Set (70.0%)**: **175,000 observations**
- **Validation Set (15.0%)**: **37,500 observations**
- **Test Set (15.0%)**: **37,500 observations**

### 2. Practicality Verification of `(City, Property_Type)` Stratification
- **Total Strata Combinations**: $42 \text{ Cities} \times 3 \text{ Property Types} = 126 \text{ Strata Cells}$.
- **Average Records per Cell**: $\approx 1,984 \text{ records}$.
- **Minimum Records in Any Cell**: **1,912 records** (Coimbatore Apartment).
- **Maximum Records in Any Cell**: **2,054 records** (Mumbai Villa).

**Conclusion**: All 126 cells contain $\approx 2,000$ observations. Zero rare or sparse strata exist. Stratification by `(City, Property_Type)` is mathematically sound and practical.

---

## 🔍 Part E: Feature Redundancy & Model Collinearity Management

To avoid feeding duplicate mathematical transformations into machine learning estimators, features are explicitly cataloged:

### Feature Redundancy Matrix

| Feature Group | Primary Selected Feature | Redundant / Derived Feature | Remediation Directive |
| :--- | :--- | :--- | :--- |
| **Property Age** | `Age_of_Property` | `Year_Built` | **Drop `Year_Built` permanently**. Deterministic correlation is $\rho = -1.0$ ($\text{Age} = 2025 - \text{Year}$). |
| **Amenities** | `Total_Amenities_Count` & `Luxury_Amenities_Index` | Atomic indicators (`Has_Gym`, `Has_Pool`, etc.) | For linear models: use only `Total_Amenities_Count`. For tree models: atomic flags may be retained if interaction importance is confirmed. |
| **Unit Pricing** | `Price_per_SqFt_INR` | Original `Price_per_SqFt` | **Drop original `Price_per_SqFt`** from model training (superseded, contains 2,534 zero values). |
| **Spatial Density** | `Size_per_BHK` | `Size_in_SqFt` and `BHK` | Retained in tree models (captures layout density); drop in regularized linear models if VIF $> 10$. |
| **Floor Dynamics** | `Relative_Floor_Position` | `Floor_No` and `Total_Floors` | For valid records, `Relative_Floor_Position` encapsulates elevation; `Floor_No` contains 46.52% missing values. |

---

## 📋 Part F: Deliverable Feature & Target Catalog

Upon approval, Phase 3 implementation will yield the following schema:

| Column Name | Category | Type | Storage Role |
| :--- | :--- | :---: | :--- |
| `ID` | Identifier | `int64` | Primary key (excluded from feature matrix $\mathbf{X}$) |
| `State`, `City`, `Locality` | Spatial Nominals | Categorical | Geographic hierarchy for encoding |
| `Property_Type`, `Furnished_Status` | Asset Nominals | Categorical | Low-cardinality nominals for One-Hot Encoding |
| `Facing` | Asset Orientation | Categorical | Optional categorical ML feature (excluded from IVS) |
| `Public_Transport_Accessibility`, `Parking_Space`, `Security` | Infrastructure | Numerical $[0, 1]$ | Normalized numeric inputs |
| `BHK`, `Size_in_SqFt`, `Total_Floors` | Structural | Numerical | Core dimensional features |
| `Age_of_Property` | Vintage | Integer | Continuous building age |
| `Nearby_Schools`, `Nearby_Hospitals` | Civic Amenities | Integer $[1, 10]$ | Proximity counts |
| `Price_per_SqFt_INR` | Valuation Rate | Float64 | High-precision unit price |
| `Size_per_BHK` | Engineered Ratio | Float64 | Spaciousness quotient |
| `Connectivity_Score` | Engineered Composite | Float64 $[0, 1]$ | Transit + Schools + Hospitals index |
| `Property_Quality_Score` | Engineered Composite | Float64 $[0, 1]$ | Furnishing + Parking + Security index |
| `Luxury_Amenities_Index` | Engineered Composite | Float64 $[0, 1]$ | Amenities count normalized |
| `Price_to_Locality_Ratio` | Engineered Relative | Float64 | OOF-anchored relative price ratio |
| `Locality_Price_ZScore` | Engineered Relative | Float64 | OOF-anchored neighborhood Z-score |
| **`Price_in_Lakhs`** | **ML Regression Target** | **Float64** | **Target for Current Listed-Price Estimation (not verified market valuation)** |
| **`Is_Good_Investment`** | **ML Classification Target** | **Binary (`0`/`1`)** | **Rule-Based Algorithmic Proxy Target** |

---

## 🚦 Recommended Implementation Steps (Pending Approval)

1. **Step 1: Partitioning First**: Execute stratified $70\% / 15\% / 15\%$ train/val/test split on `data/processed/india_housing_cleaned.csv` using `(City, Property_Type)`.
2. **Step 2: 5-Fold OOF Fitting on Train**: Compute Out-Of-Fold locality benchmarks on the training fold, and master static lookups for val/test folds with hierarchical fallbacks.
3. **Step 3: Feature Pipeline Script**: Write `src/features/build_features.py` implementing exact mappings, refined quality scores, and `Is_Good_Investment` proxy label generation.
4. **Step 4: Target Audit Report**: Measure and report the empirical positive class rate of `Is_Good_Investment` and benchmark `Price_in_Lakhs` against mean/median baselines.

---

*Proposal revised and preserved in [reports/feature_engineering_proposal.md](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/feature_engineering_proposal.md).*

**Awaiting your approval before proceeding to implementation.**
