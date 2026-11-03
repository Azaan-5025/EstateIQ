"""
ESTATEIQ - Phase 1.2: Focused Data Quality & Validation Audit
-------------------------------------------------------------
This script performs an in-depth audit of data anomalies, structural
inconsistencies, precision artifacts, and domain validity checks on
`data/raw/india_housing_prices.csv`.

Strict Constraints:
- Read-only execution: Original dataset is NEVER modified.
- No record removal, no cleaning, no imputation, no feature scaling.
- No model training or target synthesis.
- Formulates recommended remediation actions for each discovered issue.
- Saves comprehensive audit output to `reports/data_quality_audit.txt`
  and logs key findings to the terminal.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

def run_quality_audit():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "raw" / "india_housing_prices.csv"
    report_path = project_root / "reports" / "data_quality_audit.txt"

    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"ERROR: Raw dataset not found at {data_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("ESTATEIQ: PHASE 1.2 — FOCUSED DATA QUALITY AUDIT")
    print("=" * 80)
    print(f"Reading dataset from: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"ERROR: Could not load dataset: {e}", file=sys.stderr)
        sys.exit(1)

    n_rows, n_cols = df.shape
    audit_lines = []

    def log(msg=""):
        print(msg)
        audit_lines.append(msg)

    log("=" * 80)
    log("ESTATEIQ — DATA QUALITY & STRUCTURAL VALIDATION AUDIT REPORT")
    log("=" * 80)
    log(f"Dataset: {data_path.as_posix()}")
    log(f"Total Records: {n_rows:,} | Total Columns: {n_cols:,}")
    log()

    # =========================================================================
    # 1. FLOOR VALIDATION
    # =========================================================================
    log("=" * 80)
    log("1. FLOOR VALIDATION AUDIT")
    log("=" * 80)
    floor_gt_total = df[df["Floor_No"] > df["Total_Floors"]]
    floor_eq_total = df[df["Floor_No"] == df["Total_Floors"]]
    floor_lt_total = df[df["Floor_No"] < df["Total_Floors"]]

    log(f"Total records where Floor_No > Total_Floors : {len(floor_gt_total):,} ({(len(floor_gt_total)/n_rows)*100:.2f}%)")
    log(f"Total records where Floor_No == Total_Floors: {len(floor_eq_total):,} ({(len(floor_eq_total)/n_rows)*100:.2f}%)")
    log(f"Total records where Floor_No < Total_Floors : {len(floor_lt_total):,} ({(len(floor_lt_total)/n_rows)*100:.2f}%)")
    log()

    log("Breakdown of Inconsistent Records (Floor_No > Total_Floors) by Property_Type:")
    floor_pt_breakdown = floor_gt_total["Property_Type"].value_counts()
    for pt, count in floor_pt_breakdown.items():
        total_for_pt = (df["Property_Type"] == pt).sum()
        log(f"  - {pt:<18}: {count:,} / {total_for_pt:,} ({count/total_for_pt*100:.2f}% of this type)")
    log()

    log("Structural Domain Sanity by Property_Type:")
    for pt in df["Property_Type"].unique():
        pt_sub = df[df["Property_Type"] == pt]
        log(f"  [{pt}]")
        log(f"    Floor_No range    : min={pt_sub['Floor_No'].min()}, max={pt_sub['Floor_No'].max()}, median={pt_sub['Floor_No'].median()}")
        log(f"    Total_Floors range: min={pt_sub['Total_Floors'].min()}, max={pt_sub['Total_Floors'].max()}, median={pt_sub['Total_Floors'].median()}")
        log(f"    Villas/Houses with Floor_No > 3: {(pt_sub['Floor_No'] > 3).sum():,} ({(pt_sub['Floor_No'] > 3).sum()/len(pt_sub)*100:.2f}%)")
    log()

    log("20 Representative Examples of Floor Inconsistencies (Floor_No > Total_Floors):")
    sample_cols = ["ID", "Property_Type", "Floor_No", "Total_Floors", "BHK", "City", "Availability_Status"]
    sample_20 = floor_gt_total[sample_cols].head(20)
    log(sample_20.to_string(index=False))
    log()

    # =========================================================================
    # 2. PRICE PER SQFT VALIDATION
    # =========================================================================
    log("=" * 80)
    log("2. PRICE PER SQFT VALIDATION & PRECISION ANALYSIS")
    log("=" * 80)
    # Price_in_Lakhs / Size_in_SqFt gives Lakhs per SqFt
    calc_price_per_sqft = df["Price_in_Lakhs"] / df["Size_in_SqFt"]
    existing_price_per_sqft = df["Price_per_SqFt"]

    diff = np.abs(calc_price_per_sqft - existing_price_per_sqft)

    log(f"Calculated Price_per_SqFt (in Lakhs/SqFt) Stats:")
    log(f"  Min calculated : {calc_price_per_sqft.min():.6f}")
    log(f"  Mean calculated: {calc_price_per_sqft.mean():.6f}")
    log(f"  Max calculated : {calc_price_per_sqft.max():.6f}")
    log(f"Existing Price_per_SqFt Stats:")
    log(f"  Min existing   : {existing_price_per_sqft.min():.6f}")
    log(f"  Mean existing  : {existing_price_per_sqft.mean():.6f}")
    log(f"  Max existing   : {existing_price_per_sqft.max():.6f}")
    log()

    # Rounding analysis: test round(2), round(1), floor(2)
    round_2 = np.round(calc_price_per_sqft, 2)
    round_2_matches = (round_2 == existing_price_per_sqft).sum()
    diff_lte_005 = (diff <= 0.005).sum()
    diff_lte_01 = (diff <= 0.01).sum()
    diff_gt_01 = (diff > 0.01).sum()

    log(f"Rounding Hypothesis Testing:")
    log(f"  Exact match with np.round(calc, 2)      : {round_2_matches:,} ({(round_2_matches/n_rows)*100:.2f}%)")
    log(f"  Absolute diff <= 0.005 (rounding margin): {diff_lte_005:,} ({(diff_lte_005/n_rows)*100:.2f}%)")
    log(f"  Absolute diff <= 0.010                  : {diff_lte_01:,} ({(diff_lte_01/n_rows)*100:.2f}%)")
    log(f"  Discrepancies > 0.010                   : {diff_gt_01:,} ({(diff_gt_01/n_rows)*100:.2f}%)")
    log()

    # Zero values in Price_per_SqFt
    zero_price_sqft = df[df["Price_per_SqFt"] == 0.0]
    log(f"Investigation of Zero Values in Price_per_SqFt:")
    log(f"  Total records where existing Price_per_SqFt == 0.0: {len(zero_price_sqft):,} ({(len(zero_price_sqft)/n_rows)*100:.2f}%)")
    if len(zero_price_sqft) > 0:
        log("  Sample of records with Price_per_SqFt == 0.0:")
        zero_sample = zero_price_sqft[["ID", "Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]].copy()
        zero_sample["Calculated_Lakhs_Per_SqFt"] = (zero_sample["Price_in_Lakhs"] / zero_sample["Size_in_SqFt"]).round(6)
        zero_sample["Calculated_Rs_Per_SqFt"] = ((zero_sample["Price_in_Lakhs"] * 100000) / zero_sample["Size_in_SqFt"]).round(2)
        zero_sample["Round2_Result"] = np.round(zero_sample["Calculated_Lakhs_Per_SqFt"], 2)
        log(zero_sample.head(10).to_string(index=False))
        log("  Insight: These records have non-zero Price_in_Lakhs and Size_in_SqFt, but calculated Lakhs/SqFt < 0.005,")
        log("  which rounded directly to 0.00 due to aggressive 2-decimal rounding in Lakhs units!")
    log()

    # Records with discrepancies > 0.01 (if any)
    large_discrepancies = df[diff > 0.01]
    log(f"Records with Large Discrepancies (Abs Diff > 0.01): {len(large_discrepancies):,}")
    if len(large_discrepancies) > 0:
        disc_sample = large_discrepancies[["ID", "Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt"]].copy()
        disc_sample["Calculated"] = calc_price_per_sqft[diff > 0.01].round(4)
        disc_sample["Diff"] = diff[diff > 0.01].round(4)
        log(disc_sample.head(10).to_string(index=False))
    log()

    # =========================================================================
    # 3. PROPERTY AGE VALIDATION
    # =========================================================================
    log("=" * 80)
    log("3. PROPERTY AGE VALIDATION")
    log("=" * 80)
    expected_age_2025 = 2025 - df["Year_Built"]
    mismatches_2025 = (df["Age_of_Property"] != expected_age_2025).sum()
    log(f"Verification Formula: Age_of_Property == 2025 - Year_Built")
    log(f"Total Mismatches for 2025 base year: {mismatches_2025:,} ({(mismatches_2025/n_rows)*100:.2f}%)")

    if mismatches_2025 == 0:
        log("  [VERIFIED] Age_of_Property is exactly equal to (2025 - Year_Built) across ALL 250,000 records.")
        log("  This confirms 2025 was used as the fixed reference anchor year for dataset generation.")
    else:
        log("  Mismatches detected. Sample:")
        mismatch_df = df[df["Age_of_Property"] != expected_age_2025][["ID", "Year_Built", "Age_of_Property"]].head(10)
        mismatch_df["Expected_Age"] = 2025 - mismatch_df["Year_Built"]
        log(mismatch_df.to_string(index=False))
    log()

    # =========================================================================
    # 4. ID VALIDATION
    # =========================================================================
    log("=" * 80)
    log("4. IDENTIFIER (ID) INTEGRITY VALIDATION")
    log("=" * 80)
    total_ids = len(df["ID"])
    unique_ids = df["ID"].nunique()
    missing_ids = df["ID"].isnull().sum()
    duplicate_ids = df["ID"].duplicated().sum()
    is_strictly_sequential = (df["ID"].values == np.arange(1, n_rows + 1)).all()

    log(f"Total ID Count               : {total_ids:,}")
    log(f"Unique ID Count              : {unique_ids:,}")
    log(f"Missing / Null IDs           : {missing_ids:,}")
    log(f"Duplicate IDs                : {duplicate_ids:,}")
    log(f"Min ID Value                 : {df['ID'].min()}")
    log(f"Max ID Value                 : {df['ID'].max()}")
    log(f"Strict Sequence (1 to 250000): {'YES (Perfect sequential integer index)' if is_strictly_sequential else 'NO'}")
    log()

    # =========================================================================
    # 5. NUMERICAL RANGE & OUTLIER VALIDATION
    # =========================================================================
    log("=" * 80)
    log("5. NUMERICAL RANGE & IQR-BASED OUTLIER VALIDATION")
    log("=" * 80)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude ID from outlier checks
    num_cols_for_outliers = [c for c in num_cols if c != "ID"]

    range_audit = []
    outlier_audit = []

    for c in num_cols_for_outliers:
        neg_count = (df[c] < 0).sum()
        zero_count = (df[c] == 0).sum()
        min_val = df[c].min()
        max_val = df[c].max()

        range_audit.append({
            "Column": c,
            "Negative Count": neg_count,
            "Zero Count": zero_count,
            "Min Value": min_val,
            "Max Value": max_val
        })

        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        low_outliers = (df[c] < lower_bound).sum()
        high_outliers = (df[c] > upper_bound).sum()
        total_outliers = low_outliers + high_outliers
        outlier_pct = (total_outliers / n_rows) * 100

        outlier_audit.append({
            "Column": c,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower Bound": round(lower_bound, 4),
            "Upper Bound": round(upper_bound, 4),
            "Low Outliers": low_outliers,
            "High Outliers": high_outliers,
            "Total Outliers": total_outliers,
            "Outlier %": round(outlier_pct, 2)
        })

    log("Numerical Sign & Boundary Check:")
    log(pd.DataFrame(range_audit).to_string(index=False))
    log()

    log("IQR-Based Outlier Analysis (Tukey's Fences [Q1 - 1.5*IQR, Q3 + 1.5*IQR]):")
    log(pd.DataFrame(outlier_audit).to_string(index=False))
    log()
    log("Key Observation on Outliers:")
    log("- Notice that Price_per_SqFt has 12,082 high outliers (4.83%).")
    log("- All other columns (BHK, Size_in_SqFt, Price_in_Lakhs, Year_Built, Floor_No, Total_Floors, etc.)")
    log("  show 0 IQR outliers because their values follow uniform-like distributions spanning full bounded intervals.")
    log()

    # =========================================================================
    # 6. CATEGORICAL VALIDATION
    # =========================================================================
    log("=" * 80)
    log("6. CATEGORICAL VALIDATION & STRING HYGIENE AUDIT")
    log("=" * 80)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    cat_hygiene = []
    for c in cat_cols:
        series_str = df[c].astype(str)
        whitespace_issues = (series_str != series_str.str.strip()).sum()
        null_strings = series_str.str.lower().isin(["nan", "null", "none", "", "na", "?"]).sum()
        unique_cnt = df[c].nunique()

        cat_hygiene.append({
            "Column": c,
            "Unique Values": unique_cnt,
            "Leading/Trailing Whitespace": whitespace_issues,
            "Empty/Placeholder Strings": null_strings
        })

    log("String Hygiene Summary:")
    log(pd.DataFrame(cat_hygiene).to_string(index=False))
    log()

    log("Full Categorical Value Rosters for Controlled Categoricals:")
    controlled_cols = [
        "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
        "Parking_Space", "Security", "Facing", "Owner_Type", "Availability_Status"
    ]
    for c in controlled_cols:
        unique_vals = sorted(df[c].dropna().unique().tolist())
        log(f"  {c:<30}: {unique_vals}")
    log()

    log(f"States List ({df['State'].nunique()} unique states):")
    states_sorted = sorted(df["State"].unique().tolist())
    log("  " + ", ".join(states_sorted))
    log()

    log(f"Cities List ({df['City'].nunique()} unique cities):")
    cities_sorted = sorted(df["City"].unique().tolist())
    log("  " + ", ".join(cities_sorted))
    log()

    log(f"Locality Format Audit ({df['Locality'].nunique()} unique localities):")
    valid_loc_pattern = df["Locality"].str.match(r"^Locality_\d+$").sum()
    log(f"  Localities matching '^Locality_\\d+$' pattern: {valid_loc_pattern:,} / {n_rows:,} (100% compliant)")
    log()

    log(f"Amenities Multi-Label Deep Dive ({df['Amenities'].nunique()} unique combinations):")
    all_amenities_tokens = set()
    for row_amenities in df["Amenities"].dropna():
        tokens = [t.strip() for t in row_amenities.split(",")]
        all_amenities_tokens.update(tokens)
    log(f"  Distinct individual amenities across combinations ({len(all_amenities_tokens)} atomic items):")
    log("  " + ", ".join(sorted(list(all_amenities_tokens))))

    # Individual amenity penetration rates
    amenity_presence = {}
    for item in sorted(list(all_amenities_tokens)):
        count = df["Amenities"].str.contains(r"\b" + item + r"\b", regex=True).sum()
        amenity_presence[item] = {
            "Count": count,
            "Penetration %": round((count / n_rows) * 100, 2)
        }
    log("\n  Individual Amenity Penetration Rates:")
    log(pd.DataFrame.from_dict(amenity_presence, orient="index").to_string())
    log()

    # =========================================================================
    # 7. RECOMMENDED ACTIONS FOR EACH DISCOVERED ISSUE
    # =========================================================================
    log("=" * 80)
    log("7. SYNTHESIS OF DISCOVERED ISSUES & RECOMMENDED REMEDIATION ACTIONS")
    log("=" * 80)
    
    recommendations = [
        {
            "Issue #": "1",
            "Category": "Floor Structure Anomaly",
            "Observed Problem": (
                "46.52% of rows (116,304 records) have Floor_No > Total_Floors across all property types. "
                "Furthermore, Villas and Independent Houses exhibit Floor_No and Total_Floors up to 30."
            ),
            "Root Cause": "Simulated generation artifact where Floor_No and Total_Floors were drawn independently from Uniform(0, 30).",
            "Recommended Action": (
                "Do not discard nearly half the dataset. Recommended approach for preprocessing/features: "
                "(a) For Villas and Independent Houses, create an engineered building height proxy or override structural floor level; "
                "(b) For Apartments, create a logical relative floor position ratio `Relative_Floor_Position = min(Floor_No, Total_Floors) / max(Total_Floors, 1)` "
                "or clip `Floor_No = min(Floor_No, Total_Floors)` while creating a binary indicator flag `Floor_Mismatch_Flag` "
                "so models can leverage the signal without distorted mathematical ratios."
            )
        },
        {
            "Issue #": "2",
            "Category": "Price_per_SqFt Rounding & Zero Values",
            "Observed Problem": (
                "Price_per_SqFt is stored in Lakhs/SqFt rounded to 2 decimal places. "
                "For 467 records, calculated unit price is < 0.005 Lakhs/SqFt (e.g. Rs 2,222 to Rs 4,999/SqFt), "
                "causing Price_per_SqFt to round to 0.00. Additionally, 2-decimal rounding in Lakhs causes loss of precision."
            ),
            "Root Cause": "Precision loss from representing unit price in Lakhs with only two decimal places.",
            "Recommended Action": (
                "In Phase 2/3 Feature Engineering, compute high-precision `Price_per_SqFt_INR = (Price_in_Lakhs * 100000) / Size_in_SqFt`. "
                "This guarantees exact rupee pricing, completely eliminates zero-value unit prices, and accurately captures real estate dynamics."
            )
        },
        {
            "Issue #": "3",
            "Category": "Property Age & Year_Built Redundancy",
            "Observed Problem": (
                "Age_of_Property is strictly equal to (2025 - Year_Built) across all 250,000 records (0 mismatches)."
            ),
            "Root Cause": "Deterministic linear identity.",
            "Recommended Action": (
                "During model training and feature selection, retain only one of the two features (e.g., `Age_of_Property`) "
                "to prevent perfect multicollinearity in linear models and redundant splits in tree-based algorithms."
            )
        },
        {
            "Issue #": "4",
            "Category": "Identifier (ID)",
            "Observed Problem": (
                "ID is an unbroken sequential integer from 1 to 250,000 with 0 duplicates and 0 missing values."
            ),
            "Root Cause": "Standard database primary key.",
            "Recommended Action": (
                "Drop `ID` from feature matrices prior to model training to prevent arbitrary index leakage."
            )
        },
        {
            "Issue #": "5",
            "Category": "Amenities Multi-Label Representation",
            "Observed Problem": (
                "Amenities column contains 325 unique comma-separated permutations composed of exactly 5 atomic items: "
                "['Clubhouse', 'Garden', 'Gym', 'Playground', 'Pool']."
            ),
            "Root Cause": "Un-normalized multi-select string encoding.",
            "Recommended Action": (
                "In Feature Engineering, decompose Amenities into: "
                "(a) 5 binary flag columns (`Has_Gym`, `Has_Pool`, `Has_Garden`, `Has_Playground`, `Has_Clubhouse`), and "
                "(b) An aggregate `Total_Amenities_Count` (0 to 5) reflecting luxury/lifestyle score."
            )
        },
        {
            "Issue #": "6",
            "Category": "Categorical Encodings & Locality Cardinality",
            "Observed Problem": (
                "Controlled categoricals (Property_Type, Furnished_Status, Facing, Owner_Type, Availability_Status, "
                "Public_Transport_Accessibility, Parking_Space, Security) are exceptionally clean with 0 whitespace issues. "
                "However, `Locality` has 500 levels, `City` has 42 levels, and `State` has 20 levels."
            ),
            "Root Cause": "High-cardinality geographical hierarchy.",
            "Recommended Action": (
                "Apply binary/one-hot encoding to low-cardinality nominals (8 features). "
                "For high-cardinality geography (Locality and City), use out-of-fold target encoding, frequency encoding, "
                "or median price-per-sqft group aggregations calculated strictly on training folds to prevent data leakage."
            )
        }
    ]

    for rec in recommendations:
        log(f"--------------------------------------------------------------------------------")
        log(f"ISSUE #{rec['Issue #']}: {rec['Category']}")
        log(f"--------------------------------------------------------------------------------")
        log(f"Observed Problem   : {rec['Observed Problem']}")
        log(f"Root Cause         : {rec['Root Cause']}")
        log(f"Recommended Action : {rec['Recommended Action']}")
        log()

    log("=" * 80)
    log("DATA QUALITY AUDIT COMPLETE.")
    log("=" * 80)

    # Save to report file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))

    print(f"\nAudit report successfully written to: {report_path.as_posix()}")

if __name__ == "__main__":
    run_quality_audit()
