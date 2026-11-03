"""
ESTATEIQ - Phase 2.2: EDA Group 2 — Geographic & Locational Dynamics
--------------------------------------------------------------------
This script addresses Questions 6 to 10:
6. Average price per sqft by state (Price_per_SqFt_INR).
7. Average property price by city (Price_in_Lakhs).
8. Median property age by locality (Age_of_Property across 500 localities).
9. BHK distribution across top prominent cities.
10. Price comparison of the five most expensive localities vs national benchmark.

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
    "muted_gold": "#C59B4E",
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
    print("ESTATEIQ EDA GROUP 2: GEOGRAPHIC & LOCATIONAL DYNAMICS")
    print("=" * 80)
    print(f"Loading cleaned dataset: {data_path}")

    df = pd.read_csv(data_path)
    n_records = len(df)
    print(f"Dataset successfully loaded. Total Records: {n_records:,}\n")

    apply_estateiq_style()

    # -------------------------------------------------------------------------
    # Q6: Average Price per SqFt by State
    # -------------------------------------------------------------------------
    print("Generating Chart 6: Average Price per SqFt by State...")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    state_stats = (
        df.groupby("State")["Price_per_SqFt_INR"]
        .agg(["mean", "median", "count"])
        .sort_values(by="mean", ascending=True)
    )

    y_pos = np.arange(len(state_stats))
    bars = ax.barh(
        y_pos,
        state_stats["mean"],
        color=PALETTE["deep_forest"],
        edgecolor=PALETTE["sand"],
        height=0.65,
        alpha=0.9,
    )

    # Highlight top 3 states with Champagne Gold
    for i in range(len(state_stats) - 3, len(state_stats)):
        bars[i].set_color(PALETTE["champagne"])
        bars[i].set_edgecolor(PALETTE["deep_forest"])

    # National average benchmark line
    nat_avg = df["Price_per_SqFt_INR"].mean()
    ax.axvline(nat_avg, color=PALETTE["sage"], linestyle="--", linewidth=2, label=f"National Avg: INR {nat_avg:,.0f}/SqFt")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(state_stats.index, fontsize=9.5)
    ax.set_xlabel("Average Price per SqFt (INR)", fontsize=11, labelpad=8)
    ax.set_title("Average Price per SqFt across 20 Indian States", fontsize=14, pad=15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"INR {x:,.0f}"))
    ax.grid(True, axis="x", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=10)

    # Annotate values at end of bars
    for idx, (bar, mean_val) in enumerate(zip(bars, state_stats["mean"])):
        ax.text(
            mean_val + 150,
            bar.get_y() + bar.get_height() / 2,
            f"INR {mean_val:,.0f}",
            va="center",
            ha="left",
            fontsize=8,
            color=PALETTE["deep_forest"],
            fontweight="bold" if idx >= len(state_stats) - 3 else "normal",
        )

    chart6_path = output_dir / "q06_avg_price_per_sqft_by_state.png"
    plt.tight_layout()
    plt.savefig(chart6_path, dpi=300)
    plt.close()
    print(f"Saved: {chart6_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q7: Average Property Price by City
    # -------------------------------------------------------------------------
    print("Generating Chart 7: Average Property Price by City...")
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    city_stats = (
        df.groupby("City")["Price_in_Lakhs"]
        .agg(["mean", "median", "count"])
        .sort_values(by="mean", ascending=True)
    )

    y_pos = np.arange(len(city_stats))
    bars = ax.barh(
        y_pos,
        city_stats["mean"],
        color=PALETTE["sage"],
        edgecolor=PALETTE["deep_forest"],
        height=0.65,
        alpha=0.9,
    )

    # Highlight top 5 cities with Champagne Gold
    for i in range(len(city_stats) - 5, len(city_stats)):
        bars[i].set_color(PALETTE["champagne"])
        bars[i].set_edgecolor(PALETTE["deep_forest"])

    nat_city_avg = df["Price_in_Lakhs"].mean()
    ax.axvline(nat_city_avg, color=PALETTE["deep_forest"], linestyle="--", linewidth=2, label=f"Overall Avg: INR {nat_city_avg:,.1f}L")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(city_stats.index, fontsize=8.5)
    ax.set_xlabel("Average Property Price (INR in Lakhs)", fontsize=11, labelpad=8)
    ax.set_title("Average Property Price across 42 Indian Cities", fontsize=14, pad=15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}L"))
    ax.grid(True, axis="x", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=10)

    chart7_path = output_dir / "q07_avg_price_by_city.png"
    plt.tight_layout()
    plt.savefig(chart7_path, dpi=300)
    plt.close()
    print(f"Saved: {chart7_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q8: Median Property Age by Locality
    # -------------------------------------------------------------------------
    print("Generating Chart 8: Median Property Age by Locality...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    locality_age = df.groupby("Locality")["Age_of_Property"].median()

    sns.histplot(
        locality_age,
        bins=25,
        kde=True,
        color=PALETTE["deep_forest"],
        edgecolor=PALETTE["sand"],
        alpha=0.85,
        ax=ax,
        stat="density",
    )

    med_age_overall = locality_age.median()
    mean_age_overall = locality_age.mean()
    ax.axvline(med_age_overall, color=PALETTE["champagne"], linestyle="--", linewidth=2, label=f"Median of Medians: {med_age_overall:.1f} yrs")
    ax.axvline(mean_age_overall, color=PALETTE["sage"], linestyle=":", linewidth=2, label=f"Mean of Medians: {mean_age_overall:.1f} yrs")

    ax.set_title("Distribution of Median Property Age across 500 Localities", fontsize=14, pad=15)
    ax.set_xlabel("Locality Median Property Age (Years)", fontsize=11, labelpad=8)
    ax.set_ylabel("Density", fontsize=11, labelpad=8)
    ax.grid(True, alpha=0.5)
    ax.legend(frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=10)

    loc_stats_text = (
        f"Total Localities: {len(locality_age):,}\n"
        f"Min Locality Median: {locality_age.min():.1f} yrs\n"
        f"Max Locality Median: {locality_age.max():.1f} yrs\n"
        f"Std Dev across Localities: {locality_age.std():.2f} yrs\n"
        f"Insight: Balanced vintage across zones"
    )
    ax.text(
        0.97, 0.70, loc_stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["sand"], edgecolor=PALETTE["deep_forest"], alpha=0.9)
    )

    chart8_path = output_dir / "q08_median_age_by_locality.png"
    plt.tight_layout()
    plt.savefig(chart8_path, dpi=300)
    plt.close()
    print(f"Saved: {chart8_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q9: BHK Distribution Across Prominent Cities
    # -------------------------------------------------------------------------
    print("Generating Chart 9: BHK Distribution Across Prominent Cities...")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # Top 10 cities by observation volume
    top10_cities = df["City"].value_counts().head(10).index.tolist()
    df_top10 = df[df["City"].isin(top10_cities)]

    # Compute percentage distribution of BHK in each city
    bhk_city = (
        pd.crosstab(df_top10["City"], df_top10["BHK"], normalize="index") * 100
    ).loc[top10_cities]

    # Custom harmonious color gradient for 1 BHK to 5 BHK
    bhk_colors = [
        PALETTE["sand"],
        PALETTE["sage"],
        PALETTE["champagne"],
        PALETTE["charcoal"],
        PALETTE["deep_forest"],
    ]

    bhk_city.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=bhk_colors,
        edgecolor=PALETTE["deep_forest"],
        linewidth=0.8,
        width=0.7,
    )

    ax.set_title("BHK Composition Across Top 10 Prominent Real Estate Hubs", fontsize=14, pad=15)
    ax.set_xlabel("City", fontsize=11, labelpad=8)
    ax.set_ylabel("Proportion (%)", fontsize=11, labelpad=8)
    ax.set_xticklabels(top10_cities, rotation=35, ha="right", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y:.0f}%"))
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.5)
    ax.legend(
        title="BHK Configuration",
        labels=[f"{i} BHK" for i in range(1, 6)],
        frameon=True,
        facecolor=PALETTE["ivory"],
        edgecolor=PALETTE["sand"],
        fontsize=9.5,
        title_fontsize=10,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    chart9_path = output_dir / "q09_bhk_distribution_across_cities.png"
    plt.tight_layout()
    plt.savefig(chart9_path, dpi=300)
    plt.close()
    print(f"Saved: {chart9_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q10: Price Comparison of the 5 Most Expensive Localities
    # -------------------------------------------------------------------------
    print("Generating Chart 10: Price Comparison of 5 Most Expensive Localities...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Group by Locality and compute average Price_per_SqFt_INR and Price_in_Lakhs
    loc_summary = (
        df.groupby("Locality")
        .agg(
            mean_price_sqft=("Price_per_SqFt_INR", "mean"),
            median_price_sqft=("Price_per_SqFt_INR", "median"),
            mean_price_lakhs=("Price_in_Lakhs", "mean"),
            count=("ID", "count")
        )
        .sort_values(by="mean_price_sqft", ascending=False)
    )

    top5_localities = loc_summary.head(5)
    print("Top 5 Most Expensive Localities (by mean Price_per_SqFt_INR):")
    print(top5_localities.to_string())

    # Build comparative DataFrame: Top 5 + Overall Benchmark
    comp_df = top5_localities.copy()
    benchmark_row = pd.DataFrame([{
        "mean_price_sqft": df["Price_per_SqFt_INR"].mean(),
        "median_price_sqft": df["Price_per_SqFt_INR"].median(),
        "mean_price_lakhs": df["Price_in_Lakhs"].mean(),
        "count": len(df)
    }], index=["National Benchmark"])
    comp_df = pd.concat([comp_df, benchmark_row])

    x_indices = np.arange(len(comp_df))
    bar_width = 0.38

    bars1 = ax.bar(
        x_indices - bar_width / 2,
        comp_df["mean_price_sqft"],
        width=bar_width,
        color=PALETTE["deep_forest"],
        edgecolor=PALETTE["sand"],
        label="Mean Price / SqFt (INR)",
        alpha=0.9
    )
    bars2 = ax.bar(
        x_indices + bar_width / 2,
        comp_df["median_price_sqft"],
        width=bar_width,
        color=PALETTE["champagne"],
        edgecolor=PALETTE["deep_forest"],
        label="Median Price / SqFt (INR)",
        alpha=0.9
    )

    # Highlight benchmark bar differently
    bars1[-1].set_color(PALETTE["sage"])
    bars2[-1].set_color(PALETTE["sand"])

    ax.set_title("Unit Valuation: Top 5 Expensive Localities vs. National Benchmark", fontsize=14, pad=15)
    ax.set_xlabel("Locality", fontsize=11, labelpad=8)
    ax.set_ylabel("Price per SqFt (INR)", fontsize=11, labelpad=8)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(comp_df.index, rotation=25, ha="right", fontsize=9.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"INR {y:,.0f}"))
    ax.grid(True, axis="y", alpha=0.5)
    ax.legend(frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=10)

    # Annotate values on top of bars
    for b1, b2, (_, row) in zip(bars1, bars2, comp_df.iterrows()):
        ax.text(
            b1.get_x() + b1.get_width() / 2,
            b1.get_height() + 200,
            f"INR {row['mean_price_sqft']:,.0f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=PALETTE["deep_forest"],
            fontweight="bold"
        )

    chart10_path = output_dir / "q10_top5_expensive_localities.png"
    plt.tight_layout()
    plt.savefig(chart10_path, dpi=300)
    plt.close()
    print(f"Saved: {chart10_path.as_posix()}")

    print("\n" + "=" * 80)
    print("EDA GROUP 2 COMPLETED SUCCESSFULLY: 5 CHARTS GENERATED")
    print("=" * 80)

if __name__ == "__main__":
    main()
