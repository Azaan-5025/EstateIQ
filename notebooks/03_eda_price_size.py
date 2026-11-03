"""
ESTATEIQ - Phase 2.2: EDA Group 1 — Price & Property Size
---------------------------------------------------------
This script addresses Questions 1 to 5:
1. Distribution of property prices (Price_in_Lakhs).
2. Distribution of property sizes (Size_in_SqFt).
3. Price per sqft (Price_per_SqFt_INR) by property type.
4. Relationship between property size and price.
5. Outlier analysis for price per sqft and size.

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
    "accent_gold": "#C59B4E",
}

def apply_estateiq_style():
    """Configures global Matplotlib and Seaborn styles to match ESTATEIQ identity."""
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
    print("ESTATEIQ EDA GROUP 1: PRICE & SIZE DYNAMICS")
    print("=" * 80)
    print(f"Loading cleaned dataset: {data_path}")

    df = pd.read_csv(data_path)
    n_records = len(df)
    print(f"Dataset successfully loaded. Total Records: {n_records:,}\n")

    apply_estateiq_style()

    # -------------------------------------------------------------------------
    # Q1: Distribution of Property Prices
    # -------------------------------------------------------------------------
    print("Generating Chart 1: Distribution of Property Prices...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    price_mean = df["Price_in_Lakhs"].mean()
    price_median = df["Price_in_Lakhs"].median()
    price_std = df["Price_in_Lakhs"].std()
    price_min = df["Price_in_Lakhs"].min()
    price_max = df["Price_in_Lakhs"].max()

    sns.histplot(
        df["Price_in_Lakhs"],
        bins=50,
        kde=True,
        color=PALETTE["deep_forest"],
        edgecolor=PALETTE["sand"],
        alpha=0.85,
        ax=ax,
        stat="density",
    )
    
    # Overlay vertical reference lines
    ax.axvline(price_mean, color=PALETTE["champagne"], linestyle="--", linewidth=2, label=f"Mean: INR {price_mean:,.2f}L")
    ax.axvline(price_median, color=PALETTE["sage"], linestyle=":", linewidth=2.5, label=f"Median: INR {price_median:,.2f}L")

    ax.set_title("Distribution of Property Prices across Indian Markets", fontsize=14, pad=15)
    ax.set_xlabel("Property Price (INR in Lakhs)", fontsize=11, labelpad=8)
    ax.set_ylabel("Density", fontsize=11, labelpad=8)
    ax.grid(True, alpha=0.5)
    ax.legend(frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=10)

    # Annotation box for sample and bounds
    stats_text = (
        f"N = {n_records:,}\n"
        f"Range: [{price_min:.1f}L, {price_max:.1f}L]\n"
        f"Std Dev: {price_std:.2f}L\n"
        f"Distribution: Uniformly Bounded"
    )
    ax.text(
        0.97, 0.70, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["sand"], edgecolor=PALETTE["deep_forest"], alpha=0.9)
    )

    chart1_path = output_dir / "q01_price_distribution.png"
    plt.tight_layout()
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved: {chart1_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q2: Distribution of Property Sizes
    # -------------------------------------------------------------------------
    print("Generating Chart 2: Distribution of Property Sizes...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    size_mean = df["Size_in_SqFt"].mean()
    size_median = df["Size_in_SqFt"].median()
    size_std = df["Size_in_SqFt"].std()
    size_min = df["Size_in_SqFt"].min()
    size_max = df["Size_in_SqFt"].max()

    sns.histplot(
        df["Size_in_SqFt"],
        bins=50,
        kde=True,
        color=PALETTE["sage"],
        edgecolor=PALETTE["deep_forest"],
        alpha=0.85,
        ax=ax,
        stat="density",
    )

    ax.axvline(size_mean, color=PALETTE["deep_forest"], linestyle="--", linewidth=2, label=f"Mean: {size_mean:,.0f} SqFt")
    ax.axvline(size_median, color=PALETTE["champagne"], linestyle=":", linewidth=2.5, label=f"Median: {size_median:,.0f} SqFt")

    ax.set_title("Distribution of Built-up Property Sizes (SqFt)", fontsize=14, pad=15)
    ax.set_xlabel("Property Size (SqFt)", fontsize=11, labelpad=8)
    ax.set_ylabel("Density", fontsize=11, labelpad=8)
    ax.grid(True, alpha=0.5)
    ax.legend(frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=10)

    size_stats_text = (
        f"N = {n_records:,}\n"
        f"Range: [{size_min:,}, {size_max:,}] SqFt\n"
        f"Std Dev: {size_std:,.1f} SqFt\n"
        f"Distribution: Continuous Uniform"
    )
    ax.text(
        0.97, 0.70, size_stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["sand"], edgecolor=PALETTE["deep_forest"], alpha=0.9)
    )

    chart2_path = output_dir / "q02_size_distribution.png"
    plt.tight_layout()
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved: {chart2_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q3: Price per SqFt by Property Type
    # -------------------------------------------------------------------------
    print("Generating Chart 3: Price per SqFt by Property Type...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    property_types = ["Apartment", "Independent House", "Villa"]
    box_palette = [PALETTE["deep_forest"], PALETTE["sage"], PALETTE["champagne"]]

    sns.boxplot(
        data=df,
        x="Property_Type",
        y="Price_per_SqFt_INR",
        order=property_types,
        palette=box_palette,
        fliersize=1.5,
        linewidth=1.5,
        ax=ax,
    )

    pt_stats = df.groupby("Property_Type")["Price_per_SqFt_INR"].agg(["mean", "median", "std"])
    print("Property Type Price per SqFt Summary:")
    print(pt_stats.to_string())

    ax.set_title("Price per Square Foot (INR) across Property Types", fontsize=14, pad=15)
    ax.set_xlabel("Property Type", fontsize=11, labelpad=8)
    ax.set_ylabel("Price per SqFt (INR)", fontsize=11, labelpad=8)
    ax.grid(True, alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"INR {x:,.0f}"))

    # Add median labels on chart
    for i, pt in enumerate(property_types):
        med = pt_stats.loc[pt, "median"]
        mean_val = pt_stats.loc[pt, "mean"]
        ax.text(
            i, med + 1500, f"Med: INR {med:,.0f}\nAvg: INR {mean_val:,.0f}",
            horizontalalignment="center",
            fontsize=9,
            color=PALETTE["deep_forest"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], alpha=0.85)
        )

    chart3_path = output_dir / "q03_price_per_sqft_by_property_type.png"
    plt.tight_layout()
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"Saved: {chart3_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q4: Relationship Between Property Size and Price
    # -------------------------------------------------------------------------
    print("Generating Chart 4: Relationship between Property Size and Price...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # With 250,000 rows, a 2D hexbin plot reveals density without overwhelming overplotting
    hb = ax.hexbin(
        df["Size_in_SqFt"],
        df["Price_in_Lakhs"],
        gridsize=45,
        cmap="YlGnBu",
        mincnt=1,
        bins="log",
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Log(Record Density)", fontsize=10, color=PALETTE["deep_forest"])
    cb.ax.tick_params(colors=PALETTE["deep_forest"], labelsize=9)

    # Compute correlation
    corr_pearson = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"], method="pearson")
    corr_spearman = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"], method="spearman")

    # Add trendline
    m, b = np.polyfit(df["Size_in_SqFt"], df["Price_in_Lakhs"], 1)
    x_vals = np.linspace(df["Size_in_SqFt"].min(), df["Size_in_SqFt"].max(), 100)
    ax.plot(x_vals, m * x_vals + b, color=PALETTE["champagne"], linewidth=2.5, linestyle="--", label=f"Fit Line (slope: {m:.4f})")

    ax.set_title("Property Size vs. Price Dynamics (Hexbin Density)", fontsize=14, pad=15)
    ax.set_xlabel("Size (SqFt)", fontsize=11, labelpad=8)
    ax.set_ylabel("Price (INR in Lakhs)", fontsize=11, labelpad=8)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=10)

    corr_text = (
        f"N = {n_records:,}\n"
        f"Pearson r  : {corr_pearson:+.4f}\n"
        f"Spearman \u03c1: {corr_spearman:+.4f}\n"
        f"Status: Independent Distributions"
    )
    ax.text(
        0.05, 0.93, corr_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["ivory"], edgecolor=PALETTE["deep_forest"], alpha=0.9)
    )

    chart4_path = output_dir / "q04_size_vs_price.png"
    plt.tight_layout()
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"Saved: {chart4_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q5: Outlier Analysis for Price per SqFt and Size
    # -------------------------------------------------------------------------
    print("Generating Chart 5: Outlier Analysis for Price per SqFt and Size...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # 1. Price_per_SqFt_INR Outliers
    q1_p = df["Price_per_SqFt_INR"].quantile(0.25)
    q3_p = df["Price_per_SqFt_INR"].quantile(0.75)
    iqr_p = q3_p - q1_p
    ub_p = q3_p + 1.5 * iqr_p
    lb_p = q1_p - 1.5 * iqr_p
    outliers_p = (df["Price_per_SqFt_INR"] > ub_p).sum()
    outliers_p_pct = (outliers_p / n_records) * 100

    sns.boxplot(
        y=df["Price_per_SqFt_INR"],
        ax=ax1,
        color=PALETTE["deep_forest"],
        fliersize=2,
        linewidth=1.5,
    )
    ax1.axhline(ub_p, color=PALETTE["champagne"], linestyle="--", linewidth=2, label=f"Tukey Upper Bound: INR {ub_p:,.0f}")
    ax1.set_title("Price per SqFt (INR) Outlier Analysis", fontsize=12, pad=10)
    ax1.set_ylabel("Price per SqFt (INR)", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"INR {x:,.0f}"))
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=9)

    ax1_text = (
        f"Q1: INR {q1_p:,.0f}\n"
        f"Median: INR {df['Price_per_SqFt_INR'].median():,.0f}\n"
        f"Q3: INR {q3_p:,.0f}\n"
        f"IQR: INR {iqr_p:,.0f}\n"
        f"Outliers (> Upper Fence): {outliers_p:,} ({outliers_p_pct:.2f}%)"
    )
    ax1.text(
        0.05, 0.70, ax1_text,
        transform=ax1.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["sand"], edgecolor=PALETTE["deep_forest"], alpha=0.9)
    )

    # 2. Size_in_SqFt Outliers
    q1_s = df["Size_in_SqFt"].quantile(0.25)
    q3_s = df["Size_in_SqFt"].quantile(0.75)
    iqr_s = q3_s - q1_s
    ub_s = q3_s + 1.5 * iqr_s
    lb_s = q1_s - 1.5 * iqr_s
    outliers_s = ((df["Size_in_SqFt"] < lb_s) | (df["Size_in_SqFt"] > ub_s)).sum()
    outliers_s_pct = (outliers_s / n_records) * 100

    sns.boxplot(
        y=df["Size_in_SqFt"],
        ax=ax2,
        color=PALETTE["sage"],
        fliersize=2,
        linewidth=1.5,
    )
    ax2.set_title("Property Size (SqFt) Outlier Analysis", fontsize=12, pad=10)
    ax2.set_ylabel("Built-up Size (SqFt)", fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax2.grid(True, alpha=0.5)

    ax2_text = (
        f"Q1: {q1_s:,.0f} SqFt\n"
        f"Median: {df['Size_in_SqFt'].median():,.0f} SqFt\n"
        f"Q3: {q3_s:,.0f} SqFt\n"
        f"IQR: {iqr_s:,.0f} SqFt\n"
        f"Outliers (Outside Fences): {outliers_s} ({outliers_s_pct:.2f}%)\n"
        f"Reason: Uniform Distribution over [500, 5000]"
    )
    ax2.text(
        0.05, 0.92, ax2_text,
        transform=ax2.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["sand"], edgecolor=PALETTE["deep_forest"], alpha=0.9)
    )

    chart5_path = output_dir / "q05_outliers_price_sqft_size.png"
    plt.tight_layout()
    plt.savefig(chart5_path, dpi=300)
    plt.close()
    print(f"Saved: {chart5_path.as_posix()}")

    print("\n" + "=" * 80)
    print("EDA GROUP 1 COMPLETED SUCCESSFULLY: 5 CHARTS GENERATED")
    print("=" * 80)

if __name__ == "__main__":
    main()
