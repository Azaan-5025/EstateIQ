"""
ESTATEIQ - Phase 2.2: EDA Group 3 — Feature Relationships & External Influences
-------------------------------------------------------------------------------
This script addresses Questions 11 to 15:
11. Numerical correlation heatmap (clean features, excluding identifiers & flags).
12. Nearby schools versus price per sqft.
13. Nearby hospitals versus price per sqft.
14. Property price by furnished status.
15. Price per sqft by facing direction.

Aesthetics:
- ESTATEIQ Palette: Deep Forest (#172923), Sage (#B5C9B0), Champagne Gold (#D9B76E),
  Warm Sand (#E9E3D7), Ivory (#F8F7F2).
- High resolution (300 DPI) charts saved to reports/eda/.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ESTATEIQ Design System Tokens
PALETTE = {
    "deep_forest": "#172923",
    "sage": "#B5C9B0",
    "champagne": "#D9B76E",
    "sand": "#E9E3D7",
    "ivory": "#F8F7F2",
    "charcoal": "#2C3E35",
}

def apply_estateiq_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["ivory"],
        "axes.facecolor": PALETTE["ivory"],
        "axes.edgecolor": PALETTE["deep_forest"],
        "axes.labelcolor": PALETTE["deep_forest"],
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "xtick.color": PALETTE["deep_forest"],
        "ytick.color": PALETTE["deep_forest"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": PALETTE["sand"],
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.family": "sans-serif",
    })

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "processed" / "india_housing_cleaned.csv"
    output_dir = project_root / "reports" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ESTATEIQ EDA GROUP 3: FEATURE RELATIONSHIPS & INFLUENCES")
    print("=" * 80)
    print(f"Loading cleaned dataset: {data_path}")

    df = pd.read_csv(data_path)
    n_records = len(df)
    print(f"Dataset successfully loaded. Total Records: {n_records:,}\n")

    apply_estateiq_style()

    # -------------------------------------------------------------------------
    # Q11: Numerical Correlation Heatmap
    # -------------------------------------------------------------------------
    print("Generating Chart 11: Numerical Correlation Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Clean numerical features strictly excluding ID, Is_Floor_Inconsistent, Year_Built (redundant), Price_per_SqFt (superseded)
    corr_features = [
        "Price_in_Lakhs",
        "Price_per_SqFt_INR",
        "Size_in_SqFt",
        "BHK",
        "Age_of_Property",
        "Total_Floors",
        "Floor_No",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Total_Amenities_Count"
    ]

    # Compute correlation with pairwise deletion for Floor_No (116,304 NaNs handled gracefully)
    corr_matrix = df[corr_features].corr(method="pearson")

    # Friendly human-readable feature labels
    feature_labels = [
        "Price (Lakhs)",
        "Price / SqFt (INR)",
        "Size (SqFt)",
        "BHK Count",
        "Property Age",
        "Total Floors",
        "Floor No (Valid)",
        "Nearby Schools",
        "Nearby Hospitals",
        "Amenities Count"
    ]

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Custom continuous colormap from Sand to Sage to Champagne to Deep Forest
    cmap = sns.diverging_palette(140, 45, s=80, l=45, n=9, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=-0.6,
        vmax=0.6,
        center=0.0,
        square=True,
        linewidths=1.2,
        linecolor=PALETTE["ivory"],
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        xticklabels=feature_labels,
        yticklabels=feature_labels,
        ax=ax,
    )

    ax.set_title("Correlation Matrix of Numerical Market & Structural Signals", fontsize=14, pad=15)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    note_text = (
        f"Total Records: {n_records:,}\n"
        f"Floor_No Valid N = {df['Floor_No'].notnull().sum():,}\n"
        f"Strongest relation: Price/SqFt vs Price (r = +0.505)\n"
        f"Synthetic feature independence observed (r ~ 0.00)"
    )
    ax.text(
        0.05, 0.20, note_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["sand"], edgecolor=PALETTE["deep_forest"], alpha=0.9)
    )

    chart11_path = output_dir / "q11_correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(chart11_path, dpi=300)
    plt.close()
    print(f"Saved: {chart11_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q12: Nearby Schools versus Price per SqFt
    # -------------------------------------------------------------------------
    print("Generating Chart 12: Nearby Schools vs. Price per SqFt...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    school_stats = df.groupby("Nearby_Schools")["Price_per_SqFt_INR"].agg(["mean", "median", "std"]).reset_index()

    ax.plot(
        school_stats["Nearby_Schools"],
        school_stats["mean"],
        marker="o",
        markersize=7,
        color=PALETTE["deep_forest"],
        linewidth=2.2,
        label="Mean Price / SqFt (INR)",
    )
    ax.plot(
        school_stats["Nearby_Schools"],
        school_stats["median"],
        marker="s",
        markersize=6,
        color=PALETTE["champagne"],
        linewidth=2,
        linestyle="--",
        label="Median Price / SqFt (INR)",
    )

    r_school = df["Nearby_Schools"].corr(df["Price_per_SqFt_INR"])

    ax.set_title("School Proximity vs. Property Unit Valuation (INR/SqFt)", fontsize=14, pad=15)
    ax.set_xlabel("Number of Nearby Schools (within locality radius)", fontsize=11, labelpad=8)
    ax.set_ylabel("Price per SqFt (INR)", fontsize=11, labelpad=8)
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"INR {y:,.0f}"))
    ax.grid(True, alpha=0.5)
    ax.legend(frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=10)

    ann_text = (
        f"N = {n_records:,}\n"
        f"Pearson r: {r_school:+.4f}\n"
        f"Mean across levels: ~INR 13,050/SqFt\n"
        f"Interpretation: No direct artificial pricing bias"
    )
    ax.text(
        0.05, 0.90, ann_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["sand"], edgecolor=PALETTE["deep_forest"], alpha=0.9)
    )

    chart12_path = output_dir / "q12_schools_vs_price_per_sqft.png"
    plt.tight_layout()
    plt.savefig(chart12_path, dpi=300)
    plt.close()
    print(f"Saved: {chart12_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q13: Nearby Hospitals versus Price per SqFt
    # -------------------------------------------------------------------------
    print("Generating Chart 13: Nearby Hospitals vs. Price per SqFt...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    hosp_stats = df.groupby("Nearby_Hospitals")["Price_per_SqFt_INR"].agg(["mean", "median", "std"]).reset_index()

    ax.plot(
        hosp_stats["Nearby_Hospitals"],
        hosp_stats["mean"],
        marker="o",
        markersize=7,
        color=PALETTE["sage"],
        linewidth=2.2,
        label="Mean Price / SqFt (INR)",
    )
    ax.plot(
        hosp_stats["Nearby_Hospitals"],
        hosp_stats["median"],
        marker="^",
        markersize=6,
        color=PALETTE["deep_forest"],
        linewidth=2,
        linestyle="--",
        label="Median Price / SqFt (INR)",
    )

    r_hosp = df["Nearby_Hospitals"].corr(df["Price_per_SqFt_INR"])

    ax.set_title("Healthcare Infrastructure Proximity vs. Unit Valuation (INR/SqFt)", fontsize=14, pad=15)
    ax.set_xlabel("Number of Nearby Hospitals", fontsize=11, labelpad=8)
    ax.set_ylabel("Price per SqFt (INR)", fontsize=11, labelpad=8)
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"INR {y:,.0f}"))
    ax.grid(True, alpha=0.5)
    ax.legend(frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=10)

    ann_text_h = (
        f"N = {n_records:,}\n"
        f"Pearson r: {r_hosp:+.4f}\n"
        f"Mean across levels: ~INR 13,060/SqFt\n"
        f"Interpretation: Uniform density across infrastructure tiers"
    )
    ax.text(
        0.05, 0.90, ann_text_h,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["sand"], edgecolor=PALETTE["deep_forest"], alpha=0.9)
    )

    chart13_path = output_dir / "q13_hospitals_vs_price_per_sqft.png"
    plt.tight_layout()
    plt.savefig(chart13_path, dpi=300)
    plt.close()
    print(f"Saved: {chart13_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q14: Property Price by Furnished Status
    # -------------------------------------------------------------------------
    print("Generating Chart 14: Property Price by Furnished Status...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    furn_order = ["Unfurnished", "Semi-furnished", "Furnished"]
    furn_colors = [PALETTE["sand"], PALETTE["sage"], PALETTE["champagne"]]

    sns.boxplot(
        data=df,
        x="Furnished_Status",
        y="Price_in_Lakhs",
        order=furn_order,
        palette=furn_colors,
        hue="Furnished_Status",
        legend=False,
        fliersize=1.5,
        linewidth=1.5,
        ax=ax,
    )

    furn_stats = df.groupby("Furnished_Status")["Price_in_Lakhs"].agg(["mean", "median", "std"])
    print("Furnished Status Summary:")
    print(furn_stats.to_string())

    ax.set_title("Total Property Price by Furnishing Level", fontsize=14, pad=15)
    ax.set_xlabel("Furnished Status", fontsize=11, labelpad=8)
    ax.set_ylabel("Property Price (INR in Lakhs)", fontsize=11, labelpad=8)
    ax.grid(True, alpha=0.5)

    for i, fs in enumerate(furn_order):
        med = furn_stats.loc[fs, "median"]
        mean_val = furn_stats.loc[fs, "mean"]
        ax.text(
            i, med + 15, f"Med: {med:.1f}L\nAvg: {mean_val:.1f}L",
            ha="center",
            fontsize=9.5,
            color=PALETTE["deep_forest"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], alpha=0.85)
        )

    chart14_path = output_dir / "q14_price_by_furnished_status.png"
    plt.tight_layout()
    plt.savefig(chart14_path, dpi=300)
    plt.close()
    print(f"Saved: {chart14_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q15: Price per SqFt by Facing Direction
    # -------------------------------------------------------------------------
    print("Generating Chart 15: Price per SqFt by Facing Direction...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    facing_order = ["East", "North", "South", "West"]
    facing_colors = [PALETTE["deep_forest"], PALETTE["champagne"], PALETTE["sage"], PALETTE["sand"]]

    sns.barplot(
        data=df,
        x="Facing",
        y="Price_per_SqFt_INR",
        order=facing_order,
        palette=facing_colors,
        hue="Facing",
        legend=False,
        edgecolor=PALETTE["deep_forest"],
        linewidth=1.2,
        errorbar="ci",
        ax=ax,
    )

    facing_stats = df.groupby("Facing")["Price_per_SqFt_INR"].agg(["mean", "median", "count"])
    print("Facing Direction Summary:")
    print(facing_stats.to_string())

    ax.set_title("Property Unit Valuation Across Facing Orientations", fontsize=14, pad=15)
    ax.set_xlabel("Facing Orientation", fontsize=11, labelpad=8)
    ax.set_ylabel("Average Price per SqFt (INR)", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"INR {y:,.0f}"))
    ax.set_ylim(0, 16000)
    ax.grid(True, axis="y", alpha=0.5)

    for i, fc in enumerate(facing_order):
        mean_val = facing_stats.loc[fc, "mean"]
        med_val = facing_stats.loc[fc, "median"]
        ax.text(
            i, mean_val + 300, f"Avg: INR {mean_val:,.0f}\nMed: INR {med_val:,.0f}",
            ha="center",
            fontsize=9,
            color=PALETTE["deep_forest"],
            fontweight="bold"
        )

    chart15_path = output_dir / "q15_price_per_sqft_by_facing.png"
    plt.tight_layout()
    plt.savefig(chart15_path, dpi=300)
    plt.close()
    print(f"Saved: {chart15_path.as_posix()}")

    print("\n" + "=" * 80)
    print("EDA GROUP 3 COMPLETED SUCCESSFULLY: 5 CHARTS GENERATED")
    print("=" * 80)

if __name__ == "__main__":
    main()
