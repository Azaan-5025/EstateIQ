# ESTATEIQ — Real Estate Investment Advisor
### *Smarter Property Decisions Through Data-Driven Insights.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37.1-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.17.2-0194E2.svg)](https://mlflow.org/)

---

## 🏛️ Project Overview

**ESTATEIQ** is an institutional-grade real estate advisory and machine-learning intelligence platform designed for property buyers, individual investors, and asset managers evaluating residential real estate across India.

By pairing strict anti-leakage feature engineering with predictive machine learning and financial scenario modeling, ESTATEIQ transforms raw listing data into transparent, evidence-based underwriting recommendations.

---

## 🎯 Key Capabilities & Features

### 1. 🏛️ Executive Portfolio Dashboard
- High-level macroeconomic overview across **250,000 verified listings** in **20 States, 42 Cities, and 500 Localities**.
- Dynamic dataset KPIs: average listed price (₹254.7L), median size (1,950 SqFt), and property type distributions.
- Interactive Plotly visualizations for city listing volumes, price percentile distributions, and construction readiness.

### 2. 🎯 Institutional Investment Advisor
- Interactive multi-criteria property underwriting evaluation.
- Implements the approved **34 pure intrinsic features** (structural dimensions, vertical elevation, layout density, transit proximity, and recreational amenities) without price leakage.
- Evaluates properties using the production **XGBoost Classifier** (Holdout Test: **73.29% Accuracy**, **0.8228 ROC-AUC**, **0.6790 PR-AUC**).
- Displays model underwriting alignment probability alongside granular scores for Connectivity, Quality, and Vintage.

### 3. 🏷️ Current Listed-Price Estimator
- Predicts expected listed asking price using the production **Random Forest Regressor**.
- **Transparent Performance Disclosure**: Prominently highlights that intrinsic physical features exhibit near-zero empirical correlation with unanchored listing asking prices ($R^2 \approx -0.0002$, Holdout Test MAE: **₹122.81 Lakhs**).
- Clearly advises users that output represents a central-tendency benchmark, not a certified bank appraisal.

### 4. 📈 5-Year Capital Scenario Planner
- Interactive compound capital appreciation modeling engine:
  $$\text{Future Value} = \text{Current Value} \times (1 + r)^t$$
- Pre-configured institutional scenarios:
  - **Conservative (4.5% p.a.)**
  - **Moderate (6.5% p.a.)**
  - **Optimistic (8.5% p.a.)**
  - **Custom User Scenario**
- Multi-trajectory capital progression charts and year-by-year valuation schedules.

### 5. 📊 Multi-Dimensional Market Analytics
- Dynamic cascading filters by State, City, Property Type, and Bedroom configuration (BHK).
- Interactive comparative analytics for regional pricing spreads, unit price rates, amenity penetration, and building vintage tiers.

---

## 🎨 Design System & Color Palette

ESTATEIQ features a bespoke, luxury private-wealth design identity avoiding generic templates:

| Color Name | HEX Code | Design Role |
| :--- | :--- | :--- |
| **Deep Forest Green** | `#172923` | Primary brand base, hero backgrounds, primary text |
| **Sage Green** | `#B5C9B0` | Growth indicators, secondary badges, subdued visual accents |
| **Champagne Gold** | `#D9B76E` | Premium highlights, high-conviction badges, action buttons |
| **Warm Sand** | `#E9E3D7` | Card borders, structural dividers, neutral secondary container tints |
| **Ivory** | `#F8F7F2` | Clean background canvas, card surfaces, high-legibility surfaces |
| **Slate Gray** | `#4A5568` | Neutral labels, secondary body copy, uninformative baselines |

---

## 📊 Model Performance & Holdout Verification

All models were evaluated on the independent 15% holdout test partition ($N=37,500$ records with zero training/validation overlap):

### 1. Classification (Target: `Is_Good_Investment`)
- **Winning Candidate**: **XGBoost Classifier**
- **Validation Accuracy**: 73.06% $\to$ **Holdout Test Accuracy**: **73.29%** ($\Delta = +0.23\%$)
- **Validation Precision**: 58.39% $\to$ **Holdout Test Precision**: **58.85%**
- **Validation Recall**: 78.75% $\to$ **Holdout Test Recall**: **78.79%**
- **Validation F1-Score**: 67.06% $\to$ **Holdout Test F1-Score**: **67.37%**
- **Validation ROC-AUC**: 0.8229 $\to$ **Holdout Test ROC-AUC**: **0.8228**
- **Validation PR-AUC**: 0.6755 $\to$ **Holdout Test PR-AUC**: **0.6790** (Baseline floor: 35.00%)

### 2. Regression (Target: `Price_in_Lakhs`)
- **Winning Candidate**: **Random Forest Regressor**
- **Validation MAE**: ₹122.88L $\to$ **Holdout Test MAE**: **₹122.81L**
- **Validation RMSE**: ₹141.69L $\to$ **Holdout Test RMSE**: **₹141.72L**
- **Coefficient of Determination ($R^2$)**: **-0.0002** ($\approx 0.0000$)
- **Mean Absolute Percentage Error (MAPE)**: **133.79%**
- **Empirical Rationale**: Raw listed asking prices are uniformly distributed across ₹10L to ₹500L without correlation to size ($r \approx -0.003$) or BHK ($r \approx 0.002$). Purging unit price rates eliminated circular target leakage and revealed the dataset's true structure.

---

## 🛠️ Technology Stack

- **Core & Runtime**: Python 3.10+ (Anaconda / Virtualenv)
- **Web Interface**: Streamlit 1.30+
- **Data Engineering**: Pandas, NumPy
- **Interactive Visualizations**: Plotly Express & Plotly Graph Objects
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib
- **Experiment Tracking & Governance**: MLflow (SQLite backend)

---

## 📁 Project Directory Structure

```text
ESTATEIQ/
├── app/
│   ├── streamlit_app.py               # Main application entry point & router
│   ├── assets/
│   │   └── style.css                  # Custom luxury styling stylesheet
│   ├── components/
│   │   ├── __init__.py
│   │   ├── sidebar.py                 # Branded navigation & system health badge
│   │   ├── cards.py                   # Reusable KPI cards & disclosure callouts
│   │   └── charts.py                  # Standardized Plotly chart wrappers
│   └── views/
│       ├── __init__.py
│       ├── dashboard.py               # View 1: Dynamic portfolio overview
│       ├── investment_advisor.py      # View 2: 34-feature investment underwriting
│       ├── price_estimator.py         # View 3: Listed-price estimation engine
│       ├── scenario_planner.py        # View 4: 5-year compound growth planner
│       └── market_analytics.py        # View 5: Multi-dimensional market analytics
│
├── data/
│   ├── raw/                           # Original immutable raw dataset
│   └── processed/                     # Cleaned runtime dataset & partitioned features
│
├── models/
│   ├── best_classification_model.joblib # Serialized XGBoost pipeline
│   ├── classification_model_metadata.json
│   ├── best_regression_model.joblib     # Serialized Random Forest pipeline
│   └── regression_model_metadata.json
│
├── reports/
│   ├── classification/                # Stage 1 classification plots
│   ├── regression/                    # Stage 2 regression plots
│   ├── final_evaluation/              # Phase 5 holdout test diagnostic plots
│   ├── classification_model_comparison.md
│   ├── regression_model_comparison.md
│   └── final_test_evaluation.md
│
├── src/
│   ├── preprocessing/                 # Data cleaning and validation scripts
│   ├── features/                      # Feature engineering & leakage defense
│   ├── classification/                # Classification training scripts
│   ├── regression/                    # Regression training scripts
│   └── evaluation/                    # Holdout test evaluation pipeline
│
├── tests/
│   ├── __init__.py
│   └── test_phase6_app.py             # Automated unit & integration verification suite
│
├── requirements.txt                   # Project package dependencies
└── README.md                          # Comprehensive project documentation
```

---

## 🚀 Installation & Local Execution

### 1. Clone & Navigate to Repository
```bash
git clone <repository-url>
cd ESTATEIQ
```

### 2. Create and Activate Virtual Environment
```bash
# Using conda
conda create -n estateiq python=3.12 -y
conda activate estateiq

# Or using venv
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖼️ Application Screenshots

*(Screenshots will be placed here following deployment)*

| Dashboard Overview | Investment Advisor |
| :---: | :---: |
| *(Screenshot Placeholder)* | *(Screenshot Placeholder)* |

| Price Estimator | 5-Year Scenario Planner |
| :---: | :---: |
| *(Screenshot Placeholder)* | *(Screenshot Placeholder)* |

---

## ⚠️ Mandatory Domain Disclosures

> [!CAUTION]
> **Legal & Underwriting Disclosures**:
> 1. **Rule-Based Underwriting Heuristic Proxy**: Classification accuracy (73.29%) proves that XGBoost approximates the multi-criteria Investment Viability Score (IVS) heuristic based on intrinsic physical, transit, and structural features. **This does NOT guarantee financial profit or capital appreciation.**
> 2. **Current Listed-Price Estimation Only**: The regression model estimates current listed asking prices. It does **NOT** represent an independently verified fair-market valuation or certified bank appraisal.
> 3. **Absence of Historical Future-Price Labels**: The dataset contains zero longitudinal repeat-sales data.
> 4. **Five-Year Scenario Calculations**: All multi-year appreciation figures represent assumption-based compound interest projections, **NOT** machine learning forecasts.
> 5. **No Investment Guarantee**: Neither model guarantees real estate investment returns.
