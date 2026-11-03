"""
ESTATEIQ - Phase 1.2: Dataset Understanding & Initial Inspection
----------------------------------------------------------------
This script performs a comprehensive, read-only inspection of the original
raw dataset `data/raw/india_housing_prices.csv`.

Strict Constraints:
- No dataset modifications.
- No record removal or cleaning.
- No imputation or synthetic generation.
- No feature encoding or target generation.
- Outputs detailed findings to terminal and saves a report to
  `reports/dataset_initial_inspection.txt`.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

def run_inspection():
    # Resolve project root path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "raw" / "india_housing_prices.csv"
    report_path = project_root / "reports" / "dataset_initial_inspection.txt"

    print("=" * 80)
    print("ESTATEIQ: PHASE 1.2 — DATASET UNDERSTANDING & INITIAL INSPECTION")
    print("=" * 80)
    print(f"Target dataset: {data_path}")

    # Ensure report directory exists
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 0. Load Dataset with robust error handling
    # ---------------------------------------------------------
    if not data_path.exists():
        print(f"ERROR: Raw dataset not found at expected location: {data_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(data_path)
        print("Successfully loaded dataset.\n")
    except pd.errors.EmptyDataError:
        print("ERROR: Raw dataset is empty.", file=sys.stderr)
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"ERROR: Failed to parse CSV file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED ERROR while loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Output buffer to collect report content
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 80)
    log("ESTATEIQ — RAW DATASET INITIAL INSPECTION REPORT")
    log("=" * 80)
    log(f"File Path: {data_path.as_posix()}")
    log(f"File Size on Disk: {os.path.getsize(data_path) / (1024 * 1024):.2f} MB")
    log()

    # ---------------------------------------------------------
    # 1. Dataset Shape & Memory Usage
    # ---------------------------------------------------------
    n_rows, n_cols = df.shape
    mem_usage_bytes = df.memory_usage(deep=True).sum()
    mem_usage_mb = mem_usage_bytes / (1024 * 1024)

    log("--------------------------------------------------------------------------------")
    log("1. DATASET DIMENSIONS & MEMORY USAGE")
    log("--------------------------------------------------------------------------------")
    log(f"Total Records (Rows)   : {n_rows:,}")
    log(f"Total Attributes (Cols): {n_cols:,}")
    log(f"Total Memory Usage     : {mem_usage_mb:.2f} MB ({mem_usage_bytes:,} bytes)")
    log()

    # ---------------------------------------------------------
    # 2. Complete Column List & Data Types
    # ---------------------------------------------------------
    log("--------------------------------------------------------------------------------")
    log("2. COLUMN ROSTER & DATA TYPES")
    log("--------------------------------------------------------------------------------")
    col_info = []
    for idx, (col, dtype) in enumerate(df.dtypes.items(), start=1):
        non_null = df[col].notnull().sum()
        col_info.append({
            "Index": idx,
            "Column Name": col,
            "Data Type": str(dtype),
            "Non-Null Count": non_null,
            "Null Count": n_rows - non_null
        })
    col_df = pd.DataFrame(col_info)
    log(col_df.to_string(index=False))
    log()

    # ---------------------------------------------------------
    # 3. Missing Value Analysis
    # ---------------------------------------------------------
    log("--------------------------------------------------------------------------------")
    log("3. MISSING VALUE ANALYSIS")
    log("--------------------------------------------------------------------------------")
    null_counts = df.isnull().sum()
    null_pct = (null_counts / n_rows) * 100
    missing_df = pd.DataFrame({
        "Column Name": df.columns,
        "Missing Count": null_counts,
        "Missing Percentage (%)": null_pct.round(4)
    }).sort_values(by="Missing Count", ascending=False)

    total_missing_cells = null_counts.sum()
    total_cells = n_rows * n_cols
    log(f"Total Missing Cells Across Dataset: {total_missing_cells:,} ({(total_missing_cells / total_cells) * 100:.4f}% of all cells)")
    log()
    log(missing_df.to_string(index=False))
    log()

    # ---------------------------------------------------------
    # 4. Duplicate Records Analysis
    # ---------------------------------------------------------
    log("--------------------------------------------------------------------------------")
    log("4. DUPLICATE RECORDS ANALYSIS")
    log("--------------------------------------------------------------------------------")
    exact_duplicates = df.duplicated().sum()
    log(f"Exact Duplicate Rows (across all columns)          : {exact_duplicates:,}")

    # Check duplicates ignoring 'ID' if present
    cols_without_id = [c for c in df.columns if c.lower() != "id"]
    if len(cols_without_id) < n_cols:
        dups_excl_id = df.duplicated(subset=cols_without_id).sum()
        log(f"Duplicate Feature Profiles (ignoring 'ID' column)  : {dups_excl_id:,}")
    log()

    # ---------------------------------------------------------
    # 5. Categorical Columns Analysis
    # ---------------------------------------------------------
    log("--------------------------------------------------------------------------------")
    log("5. CATEGORICAL COLUMNS ANALYSIS")
    log("--------------------------------------------------------------------------------")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    log(f"Total Categorical Columns Identified: {len(cat_cols)}")
    log()
    
    cat_summary = []
    for c in cat_cols:
        n_unique = df[c].nunique(dropna=False)
        sample_vals = df[c].dropna().unique()[:5].tolist()
        sample_str = ", ".join([str(v) for v in sample_vals])
        if len(sample_vals) < n_unique:
            sample_str += ", ..."
        cat_summary.append({
            "Column Name": c,
            "Unique Count": n_unique,
            "Sample Distinct Values": sample_str
        })
    cat_df = pd.DataFrame(cat_summary)
    log(cat_df.to_string(index=False))
    log()

    # Detailed frequency distributions for low-cardinality categorical columns
    log("Detailed Distribution for Low-Cardinality Categoricals (<= 15 unique values):")
    for c in cat_cols:
        if df[c].nunique() <= 15:
            log(f"\n--- Distribution for [{c}] ---")
            val_counts = df[c].value_counts(dropna=False)
            val_pct = df[c].value_counts(dropna=False, normalize=True) * 100
            dist_table = pd.DataFrame({
                "Count": val_counts,
                "Percentage (%)": val_pct.round(2)
            })
            log(dist_table.to_string())
    log()

    # ---------------------------------------------------------
    # 6. Numerical Columns Statistical Summary
    # ---------------------------------------------------------
    log("--------------------------------------------------------------------------------")
    log("6. NUMERICAL COLUMNS STATISTICAL SUMMARY")
    log("--------------------------------------------------------------------------------")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    log(f"Total Numerical Columns Identified: {len(num_cols)}")
    log()

    num_summary = df[num_cols].describe().T
    num_summary["IQR"] = num_summary["75%"] - num_summary["25%"]
    log(num_summary[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "IQR"]].to_string())
    log()

    # ---------------------------------------------------------
    # 7. Extremes & Min/Max of Key Numerical Columns
    # ---------------------------------------------------------
    log("--------------------------------------------------------------------------------")
    log("7. MINIMUM AND MAXIMUM VALUES OF IMPORTANT NUMERICAL COLUMNS")
    log("--------------------------------------------------------------------------------")
    key_metrics = []
    for c in num_cols:
        key_metrics.append({
            "Numerical Column": c,
            "Minimum Value": df[c].min(),
            "Median (50%)": df[c].median(),
            "Maximum Value": df[c].max(),
            "Mean": round(df[c].mean(), 4),
            "Std Dev": round(df[c].std(), 4)
        })
    log(pd.DataFrame(key_metrics).to_string(index=False))
    log()

    # ---------------------------------------------------------
    # 8. Sample Records (First 10 & Last 5)
    # ---------------------------------------------------------
    log("--------------------------------------------------------------------------------")
    log("8. SAMPLE RECORDS")
    log("--------------------------------------------------------------------------------")
    log(">>> FIRST 10 RECORDS:")
    log(df.head(10).to_string())
    log()
    log(">>> LAST 5 RECORDS:")
    log(df.tail(5).to_string())
    log()

    # ---------------------------------------------------------
    # 9. Investigation of Project Objectives (Task 3)
    # ---------------------------------------------------------
    log("=" * 80)
    log("9. INVESTIGATION OF PROJECT OBJECTIVES & MODELING IMPLICATIONS")
    log("=" * 80)
    log("Objectives Review:")
    log("  A. Classification: Predict whether a property is a 'Good Investment'.")
    log("  B. Regression: Predict estimated property price after 5 years.\n")

    # Check 1: Future price column
    future_price_candidates = [
        c for c in df.columns 
        if any(term in c.lower() for term in ["future", "5_year", "5yr", "five_year", "projected", "forecast", "target"])
    ]
    log(f"1. Five-Year Future-Price Column Check:")
    if future_price_candidates:
        log(f"   Candidates detected: {future_price_candidates}")
    else:
        log("   [FINDING] NO explicit 5-year future price column exists in the raw dataset.")
        log("   Current price columns present: ['Price_in_Lakhs', 'Price_per_SqFt'].")
        log("   Implication: The 5-year future valuation target must be formulated through sound financial")
        log("   and domain-grounded appreciation modeling (e.g., CAGR based on locality/city growth rates,")
        log("   property age, public transport accessibility, and infrastructure metrics).")
    log()

    # Check 2: Historical price / appreciation information
    appreciation_candidates = [
        c for c in df.columns 
        if any(term in c.lower() for term in ["appreciat", "histor", "past", "cagr", "growth", "roi"])
    ]
    log(f"2. Historical Price / Appreciation Information Check:")
    if appreciation_candidates:
        log(f"   Candidates detected: {appreciation_candidates}")
    else:
        log("   [FINDING] NO explicit historical price time-series or appreciation rate column is present.")
        log("   However, historical timeline proxies exist: 'Year_Built' (spanning past construction years)")
        log("   and 'Age_of_Property'.")
    log()

    # Check 3: Features available for Investment Classification
    log(f"3. Feature Suitability for Investment Classification:")
    log("   The dataset contains strong multidimensional signals across 4 key real estate pillars:")
    log("   - Financial / Valuation: 'Price_in_Lakhs', 'Price_per_SqFt', 'Size_in_SqFt'")
    log("   - Location & Infrastructure: 'State', 'City', 'Locality', 'Public_Transport_Accessibility',")
    log("                                'Nearby_Schools', 'Nearby_Hospitals'")
    log("   - Property Attributes: 'Property_Type', 'BHK', 'Floor_No', 'Total_Floors', 'Age_of_Property',")
    log("                          'Year_Built', 'Facing', 'Furnished_Status'")
    log("   - Market & Quality Signals: 'Amenities', 'Parking_Space', 'Security', 'Owner_Type', 'Availability_Status'")
    log("   Conclusion: Highly viable feature space for constructing an investment viability score/classifier.")
    log()

    # Check 4: Data Leakage & Target Formulation Concerns
    log(f"4. Data Leakage & Target Construction Concerns:")
    log("   - Mathematical Collinearity: Note that 'Price_per_SqFt' is directly correlated with")
    log("     'Price_in_Lakhs' and 'Size_in_SqFt'. We must examine their exact relation:")
    
    # Calculate sample check on Price_per_SqFt
    sample_df = df[["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]].head(5)
    log("     Sample values comparison:")
    for _, row in sample_df.iterrows():
        calc_ratio = (row["Price_in_Lakhs"] * 100000) / row["Size_in_SqFt"] if row["Size_in_SqFt"] > 0 else np.nan
        log(f"     Price: {row['Price_in_Lakhs']} Lakhs, Size: {row['Size_in_SqFt']} SqFt -> Calculated Rs/SqFt: {calc_ratio:.1f} | Raw Price_per_SqFt col: {row['Price_per_SqFt']}")

    log("   - Anomaly Alert: Check floor numbers vs total floors:")
    floor_anomalies = df[df["Floor_No"] > df["Total_Floors"]]
    log(f"     Records where Floor_No > Total_Floors: {len(floor_anomalies):,} ({(len(floor_anomalies)/n_rows)*100:.2f}%)")
    log("     This indicates potential data generation artifacts or inverted columns that data cleaning")
    log("     must address in subsequent phases.")
    log("   - Leakage Prevention: Any target variable created for 5-year valuation or investment classification")
    log("     must strictly partition train/validation splits before computing regional statistics")
    log("     or synthetic baselines to avoid out-of-fold data leakage.")
    log("=" * 80)
    log("INITIAL INSPECTION COMPLETE.")
    log("=" * 80)

    # Save to text report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport successfully saved to: {report_path.as_posix()}")

if __name__ == "__main__":
    run_inspection()
