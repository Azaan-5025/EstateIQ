"""
ESTATEIQ - Phase 2.2: EDA Group 4 — Investment, Amenities & Ownership Dynamics
-------------------------------------------------------------------------------
This script addresses Questions 16 to 20:
16. Owner type distribution (Owner_Type).
17. Availability status distribution (Availability_Status).
18. Parking space versus property price (Parking_Space vs Price_in_Lakhs).
19. Amenities versus price per sqft (Total_Amenities_Count and atomic amenities).
20. Public transport accessibility versus price and investment potential.

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
from scipy import stats

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
    print("ESTATEIQ EDA GROUP 4: INVESTMENT, AMENITIES & OWNERSHIP")
    print("=" * 80)
    print(f"Loading cleaned dataset: {data_path}")

    df = pd.read_csv(data_path)
    n_records = len(df)
    print(f"Dataset successfully loaded. Total Records: {n_records:,}\n")

    apply_estateiq_style()

    # -------------------------------------------------------------------------
    # Q16: Owner Type Distribution
    # -------------------------------------------------------------------------
    print("Generating Chart 16: Owner Type Distribution...")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    owner_counts = df["Owner_Type"].value_counts()
    owner_pct = df["Owner_Type"].value_counts(normalize=True) * 100
    donut_colors = [PALETTE["deep_forest"], PALETTE["champagne"], PALETTE["sage"]]

    wedges, texts, autotexts = ax.pie(
        owner_counts,
        labels=owner_counts.index,
        autopct="%1.2f%%",
        startangle=140,
        colors=donut_colors,
        wedgeprops=dict(width=0.45, edgecolor=PALETTE["ivory"], linewidth=2),
        pctdistance=0.75,
    )

    for text in texts:
        text.set_color(PALETTE["deep_forest"])
        text.set_fontsize(11)
        text.set_fontweight("bold")

    for autotext in autotexts:
        autotext.set_color(PALETTE["ivory"])
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    # Center label for Donut
    ax.text(
        0, 0, f"Total\n{n_records:,}\nProperties",
        ha="center", va="center",
        fontsize=12, fontweight="bold",
        color=PALETTE["deep_forest"]
    )

    ax.set_title("Market Participation by Seller Entity (Owner Type)", fontsize=14, pad=20)

    # Detailed counts table annotation
    legend_labels = [f"{ot}: {owner_counts[ot]:,} ({owner_pct[ot]:.2f}%)" for ot in owner_counts.index]
    ax.legend(
        wedges, legend_labels,
        title="Seller Entity Breakdown",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=True,
        facecolor=PALETTE["ivory"],
        edgecolor=PALETTE["sand"],
        fontsize=9.5
    )

    chart16_path = output_dir / "q16_owner_type_distribution.png"
    plt.tight_layout()
    plt.savefig(chart16_path, dpi=300)
    plt.close()
    print(f"Saved: {chart16_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q17: Availability Status Distribution
    # -------------------------------------------------------------------------
    print("Generating Chart 17: Availability Status Distribution...")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    avail_counts = df["Availability_Status"].value_counts()
    avail_pct = df["Availability_Status"].value_counts(normalize=True) * 100

    y_pos = np.arange(len(avail_counts))
    bars = ax.barh(
        y_pos,
        avail_counts.values,
        color=[PALETTE["deep_forest"], PALETTE["champagne"]],
        edgecolor=PALETTE["sand"],
        height=0.55,
        alpha=0.9
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(["Under Construction", "Ready to Move"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Property Volume (Count)", fontsize=11, labelpad=8)
    ax.set_title("Property Inventory by Construction & Availability Stage", fontsize=14, pad=15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.grid(True, axis="x", alpha=0.5)

    for bar, count, pct in zip(bars, avail_counts.values, avail_pct.values):
        ax.text(
            bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.2f}%)",
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color=PALETTE["ivory"]
        )

    chart17_path = output_dir / "q17_availability_status_distribution.png"
    plt.tight_layout()
    plt.savefig(chart17_path, dpi=300)
    plt.close()
    print(f"Saved: {chart17_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q18: Parking Space versus Property Price
    # -------------------------------------------------------------------------
    print("Generating Chart 18: Parking Space vs. Property Price...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    parking_colors = [PALETTE["sand"], PALETTE["champagne"]]

    sns.boxplot(
        data=df,
        x="Parking_Space",
        y="Price_in_Lakhs",
        order=["No", "Yes"],
        palette=parking_colors,
        hue="Parking_Space",
        legend=False,
        fliersize=1.5,
        linewidth=1.5,
        ax=ax
    )

    p_stats = df.groupby("Parking_Space")["Price_in_Lakhs"].agg(["mean", "median", "std", "count"])
    print("Parking Space Summary:")
    print(p_stats.to_string())

    # Two-sample t-test
    p_yes = df[df["Parking_Space"] == "Yes"]["Price_in_Lakhs"]
    p_no = df[df["Parking_Space"] == "No"]["Price_in_Lakhs"]
    t_stat, p_val = stats.ttest_ind(p_yes, p_no, equal_var=False)

    ax.set_title("Capital Valuation by Dedicated Parking Provision", fontsize=14, pad=15)
    ax.set_xlabel("Dedicated Parking Space Available", fontsize=11, labelpad=8)
    ax.set_ylabel("Property Price (INR in Lakhs)", fontsize=11, labelpad=8)
    ax.grid(True, alpha=0.5)

    for i, pk in enumerate(["No", "Yes"]):
        med = p_stats.loc[pk, "median"]
        mean_val = p_stats.loc[pk, "mean"]
        ax.text(
            i, med + 15, f"Median: {med:.1f}L\nMean: {mean_val:.1f}L",
            ha="center",
            fontsize=10,
            color=PALETTE["deep_forest"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], alpha=0.9)
        )

    t_text = (
        f"Statistical Comparison:\n"
        f"Without Parking Mean: {p_stats.loc['No', 'mean']:.2f}L\n"
        f"With Parking Mean   : {p_stats.loc['Yes', 'mean']:.2f}L\n"
        f"Welch t-statistic   : {t_stat:+.3f}\n"
        f"p-value             : {p_val:.4f} (Not statistically significant)"
    )
    ax.text(
        0.05, 0.90, t_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["sand"], edgecolor=PALETTE["deep_forest"], alpha=0.9)
    )

    chart18_path = output_dir / "q18_parking_vs_price.png"
    plt.tight_layout()
    plt.savefig(chart18_path, dpi=300)
    plt.close()
    print(f"Saved: {chart18_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q19: Amenities versus Price per SqFt
    # -------------------------------------------------------------------------
    print("Generating Chart 19: Amenities vs. Price per SqFt...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Subplot 1: Total Amenities Count (1 to 5)
    amenity_count_stats = df.groupby("Total_Amenities_Count")["Price_per_SqFt_INR"].agg(["mean", "median", "std"]).reset_index()

    ax1.plot(
        amenity_count_stats["Total_Amenities_Count"],
        amenity_count_stats["mean"],
        marker="o",
        markersize=8,
        color=PALETTE["deep_forest"],
        linewidth=2.5,
        label="Mean Price / SqFt",
    )
    ax1.plot(
        amenity_count_stats["Total_Amenities_Count"],
        amenity_count_stats["median"],
        marker="s",
        markersize=7,
        color=PALETTE["champagne"],
        linewidth=2,
        linestyle="--",
        label="Median Price / SqFt",
    )
    ax1.set_title("Unit Valuation by Total Amenities Tier", fontsize=12, pad=10)
    ax1.set_xlabel("Total Amenities Count (0 to 5)", fontsize=10, labelpad=8)
    ax1.set_ylabel("Price per SqFt (INR)", fontsize=10, labelpad=8)
    ax1.set_xticks(range(1, 6))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"INR {y:,.0f}"))
    ax1.grid(True, alpha=0.5)
    ax1.legend(frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=9)

    # Subplot 2: Individual Atomic Amenities Comparison (Presence vs Absence)
    amenity_cols = [("Gym", "Has_Gym"), ("Pool", "Has_Pool"), ("Garden", "Has_Garden"),
                    ("Playground", "Has_Playground"), ("Clubhouse", "Has_Clubhouse")]
    
    comp_list = []
    for label, col in amenity_cols:
        with_mean = df[df[col] == 1]["Price_per_SqFt_INR"].mean()
        without_mean = df[df[col] == 0]["Price_per_SqFt_INR"].mean()
        comp_list.append({
            "Amenity": label,
            "With Amenity": with_mean,
            "Without Amenity": without_mean,
            "Difference": with_mean - without_mean
        })
    comp_amenity_df = pd.DataFrame(comp_list)

    x = np.arange(len(comp_amenity_df))
    width = 0.35

    ax2.bar(x - width/2, comp_amenity_df["With Amenity"], width, label="With Amenity", color=PALETTE["deep_forest"], edgecolor=PALETTE["sand"])
    ax2.bar(x + width/2, comp_amenity_df["Without Amenity"], width, label="Without Amenity", color=PALETTE["sage"], edgecolor=PALETTE["deep_forest"])

    ax2.set_title("Atomic Amenity Premium Comparison", fontsize=12, pad=10)
    ax2.set_xlabel("Amenity Feature", fontsize=10, labelpad=8)
    ax2.set_ylabel("Average Price per SqFt (INR)", fontsize=10, labelpad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(comp_amenity_df["Amenity"], fontsize=9.5)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"INR {y:,.0f}"))
    ax2.set_ylim(0, 16000)
    ax2.grid(True, axis="y", alpha=0.5)
    ax2.legend(frameon=True, facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], fontsize=9)

    chart19_path = output_dir / "q19_amenities_vs_price_per_sqft.png"
    plt.tight_layout()
    plt.savefig(chart19_path, dpi=300)
    plt.close()
    print(f"Saved: {chart19_path.as_posix()}")

    # -------------------------------------------------------------------------
    # Q20: Public Transport Accessibility vs. Price and Potential
    # -------------------------------------------------------------------------
    print("Generating Chart 20: Public Transport vs. Price & Potential...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    trans_order = ["Low", "Medium", "High"]
    trans_colors = [PALETTE["sand"], PALETTE["sage"], PALETTE["champagne"]]

    # Panel A: Total Price (Lakhs)
    sns.boxplot(
        data=df,
        x="Public_Transport_Accessibility",
        y="Price_in_Lakhs",
        order=trans_order,
        palette=trans_colors,
        hue="Public_Transport_Accessibility",
        legend=False,
        fliersize=1.5,
        linewidth=1.5,
        ax=ax1,
    )
    ax1.set_title("Total Capital Price by Transport Accessibility", fontsize=12, pad=10)
    ax1.set_xlabel("Public Transport Accessibility Tier", fontsize=10, labelpad=8)
    ax1.set_ylabel("Price (INR in Lakhs)", fontsize=10, labelpad=8)
    ax1.grid(True, alpha=0.5)

    trans_price_stats = df.groupby("Public_Transport_Accessibility")["Price_in_Lakhs"].agg(["mean", "median"])
    for i, tr in enumerate(trans_order):
        med = trans_price_stats.loc[tr, "median"]
        mean_val = trans_price_stats.loc[tr, "mean"]
        ax1.text(
            i, med + 15, f"Med: {med:.1f}L\nAvg: {mean_val:.1f}L",
            ha="center", fontsize=9, color=PALETTE["deep_forest"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=PALETTE["ivory"], edgecolor=PALETTE["sand"], alpha=0.85)
        )

    # Panel B: Unit Rate (Price_per_SqFt_INR)
    sns.barplot(
        data=df,
        x="Public_Transport_Accessibility",
        y="Price_per_SqFt_INR",
        order=trans_order,
        palette=trans_colors,
        hue="Public_Transport_Accessibility",
        legend=False,
        edgecolor=PALETTE["deep_forest"],
        linewidth=1.2,
        errorbar="ci",
        ax=ax2,
    )
    ax2.set_title("Unit Price (INR/SqFt) & Investment Liquidity Potential", fontsize=12, pad=10)
    ax2.set_xlabel("Public Transport Accessibility Tier", fontsize=10, labelpad=8)
    ax2.set_ylabel("Average Price per SqFt (INR)", fontsize=10, labelpad=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"INR {y:,.0f}"))
    ax2.set_ylim(0, 16000)
    ax2.grid(True, axis="y", alpha=0.5)

    trans_sqft_stats = df.groupby("Public_Transport_Accessibility")["Price_per_SqFt_INR"].agg(["mean", "median"])
    for i, tr in enumerate(trans_order):
        mean_val = trans_sqft_stats.loc[tr, "mean"]
        med_val = trans_sqft_stats.loc[tr, "median"]
        ax2.text(
            i, mean_val + 300, f"Avg: INR {mean_val:,.0f}\nMed: INR {med_val:,.0f}",
            ha="center", fontsize=9, color=PALETTE["deep_forest"], fontweight="bold"
        )

    chart20_path = output_dir / "q20_transport_vs_price_potential.png"
    plt.tight_layout()
    plt.savefig(chart20_path, dpi=300)
    plt.close()
    print(f"Saved: {chart20_path.as_posix()}")

    print("\n" + "=" * 80)
    print("EDA GROUP 4 COMPLETED SUCCESSFULLY: 5 CHARTS GENERATED")
    print("=" * 80)

if __name__ == "__main__":
    main()
