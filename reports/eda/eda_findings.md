# ESTATEIQ — Phase 2.2: Exploratory Data Analysis (EDA) Report
### Comprehensive Findings Across 20 Analytical Questions

---

## 🏛️ Executive Summary

This report documents the findings from **Phase 2.2 — Exploratory Data Analysis (EDA)** of the **ESTATEIQ** project, conducted on the cleaned Indian housing market dataset (`data/processed/india_housing_cleaned.csv`, $N = 250,000$ records, 31 features).

The primary objectives of this phase are:
1. Systematically investigate all 20 domain questions spanning pricing, structural dimensions, geographical dynamics, infrastructure influences, and investment indicators.
2. Adhere strictly to the **ESTATEIQ design aesthetic** (Deep Forest Green `#172923`, Sage Green `#B5C9B0`, Champagne Gold `#D9B76E`, Warm Sand `#E9E3D7`, Ivory `#F8F7F2`).
3. Objectively evaluate feature interactions, distributions, and synthetic benchmark characteristics to inform target engineering and model design without making unsubstantiated causal claims.

---

## 📊 Summary Roster of Generated EDA Visualizations

All visualizations were rendered at **300 DPI high resolution** and saved to the central output directory [reports/eda/](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/).

| Question # | Analysis Focus | Chart Filename | Key Metric / Insight |
| :---: | :--- | :--- | :--- |
| **Q1** | Property Price Distribution | [`q01_price_distribution.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q01_price_distribution.png) | Mean: ₹254.59L, Median: ₹253.87L, Bounded $[10\text{L}, 500\text{L}]$ |
| **Q2** | Property Size Distribution | [`q02_size_distribution.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q02_size_distribution.png) | Mean: 2,750 SqFt, Median: 2,747 SqFt, Bounded $[500, 5000]$ SqFt |
| **Q3** | Unit Price by Property Type | [`q03_price_per_sqft_by_property_type.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q03_price_per_sqft_by_property_type.png) | Apartments: ₹13,047/SqFt, Houses: ₹13,102/SqFt, Villas: ₹13,026/SqFt |
| **Q4** | Property Size vs. Price | [`q04_size_vs_price.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q04_size_vs_price.png) | Pearson $r = -0.0019$, Spearman $\rho = -0.0016$ (Independent variables) |
| **Q5** | Outliers: Unit Price & Size | [`q05_outliers_price_sqft_size.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q05_outliers_price_sqft_size.png) | Unit Price: 20,020 outliers (8.01%); Size: 0 outliers (0.00%) |
| **Q6** | Average Unit Price by State | [`q06_avg_price_per_sqft_by_state.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q06_avg_price_per_sqft_by_state.png) | 20 States; National Avg: ₹13,058/SqFt (Narrow spread: ₹12.8k–₹13.3k) |
| **Q7** | Average Price by City | [`q07_avg_price_by_city.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q07_avg_price_by_city.png) | 42 Cities; Overall Avg: ₹254.59L (Narrow spread: ₹251L–₹258L) |
| **Q8** | Locality Median Age | [`q08_median_age_by_locality.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q08_median_age_by_locality.png) | 500 Localities; Median of Medians: 18.0 years (Range: 16–21 years) |
| **Q9** | BHK Distribution Across Cities | [`q09_bhk_distribution_across_cities.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q09_bhk_distribution_across_cities.png) | 1 to 5 BHK evenly distributed (~20% each) across major metro hubs |
| **Q10** | Top 5 Expensive Localities | [`q10_top5_expensive_localities.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q10_top5_expensive_localities.png) | Top Locality_207: ₹15,204/SqFt vs National Benchmark: ₹13,058/SqFt |
| **Q11** | Numerical Correlation Matrix | [`q11_correlation_heatmap.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q11_correlation_heatmap.png) | High $r$ solely in arithmetic identities ($P$ vs $P/S$: +0.51, $S$ vs $P/S$: -0.53) |
| **Q12** | Nearby Schools vs. Unit Price | [`q12_schools_vs_price_per_sqft.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q12_schools_vs_price_per_sqft.png) | Flat mean across 1 to 10 schools (~₹13,050/SqFt, $r = -0.0013$) |
| **Q13** | Nearby Hospitals vs. Unit Price | [`q13_hospitals_vs_price_per_sqft.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q13_hospitals_vs_price_per_sqft.png) | Flat mean across 1 to 10 hospitals (~₹13,060/SqFt, $r = +0.0008$) |
| **Q14** | Price by Furnished Status | [`q14_price_by_furnished_status.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q14_price_by_furnished_status.png) | Unfurnished: ₹255.0L, Semi: ₹254.3L, Furnished: ₹254.4L (No variance) |
| **Q15** | Unit Price by Facing | [`q15_price_per_sqft_by_facing.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q15_price_per_sqft_by_facing.png) | East: ₹13,024, North: ₹13,025, South: ₹13,044, West: ₹13,140/SqFt |
| **Q16** | Owner Type Distribution | [`q16_owner_type_distribution.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q16_owner_type_distribution.png) | Broker: 33.39%, Owner: 33.31%, Builder: 33.30% (Even tri-partition) |
| **Q17** | Availability Status Distribution | [`q17_availability_status_distribution.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q17_availability_status_distribution.png) | Under Construction: 50.01%, Ready to Move: 49.99% (Perfect 50/50 balance) |
| **Q18** | Parking Space vs. Price | [`q18_parking_vs_price.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q18_parking_vs_price.png) | No: ₹254.43L vs Yes: ₹254.75L (Diff: +₹0.32L, $p = 0.5760$, not significant) |
| **Q19** | Amenities vs. Unit Price | [`q19_amenities_vs_price_per_sqft.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q19_amenities_vs_price_per_sqft.png) | Flat mean across luxury count 1 to 5; individual amenity diff $< ₹50$/SqFt |
| **Q20** | Transport Accessibility vs. Price | [`q20_transport_vs_price_potential.png`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q20_transport_vs_price_potential.png) | Low: ₹13,048, Medium: ₹13,059, High: ₹13,068/SqFt (Baseline flat) |

---

## 📈 Detailed Findings: Group 1 — Price & Property Size (Questions 1–5)

### Question 1: Distribution of Property Prices (`Price_in_Lakhs`)
- **Records Evaluated**: $N = 250,000$
- **Numerical Findings**:
  - **Minimum**: ₹10.00 Lakhs
  - **25th Percentile ($Q_1$)**: ₹132.55 Lakhs
  - **Median ($Q_2$)**: ₹253.87 Lakhs
  - **Mean**: ₹254.59 Lakhs
  - **75th Percentile ($Q_3$)**: ₹376.88 Lakhs
  - **Maximum**: ₹500.00 Lakhs
  - **Standard Deviation**: ₹141.35 Lakhs
  - **Interquartile Range (IQR)**: ₹244.33 Lakhs
- **Observations & Interpretation**:
  The property price distribution is bounded between ₹10 Lakhs and ₹500 Lakhs. It displays a continuous uniform characteristic without the strong positive right-skew typically present in real-world property markets (where luxury properties create a heavy upper tail). Both the mean and median closely align with the midpoint of the interval ($255\text{L} \approx \frac{10 + 500}{2}$).
- **Visualization Reference**: [q01_price_distribution.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q01_price_distribution.png)

---

### Question 2: Distribution of Property Sizes (`Size_in_SqFt`)
- **Records Evaluated**: $N = 250,000$
- **Numerical Findings**:
  - **Minimum**: 500 SqFt
  - **25th Percentile ($Q_1$)**: 1,623 SqFt
  - **Median ($Q_2$)**: 2,747 SqFt
  - **Mean**: 2,749.81 SqFt
  - **75th Percentile ($Q_3$)**: 3,874 SqFt
  - **Maximum**: 5,000 SqFt
  - **Standard Deviation**: 1,300.61 SqFt
  - **Interquartile Range (IQR)**: 2,251.00 SqFt
- **Observations & Interpretation**:
  Property size spans built-up areas from 500 to 5,000 SqFt. Like price, size follows an even rectangular/uniform distribution across the entire domain. The sample mean and median sit within 3 SqFt of the mathematical theoretical midpoint ($2,750\text{ SqFt}$).
- **Visualization Reference**: [q02_size_distribution.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q02_size_distribution.png)

---

### Question 3: Price per SqFt (`Price_per_SqFt_INR`) by Property Type
- **Records Evaluated**: $N = 250,000$
  - Apartments: $N = 82,956$ (33.18%)
  - Independent Houses: $N = 83,300$ (33.32%)
  - Villas: $N = 83,744$ (33.50%)
- **Numerical Findings**:

| Property Type | Record Count | Mean (₹/SqFt) | Median (₹/SqFt) | Std Dev (₹/SqFt) | Min (₹/SqFt) | Max (₹/SqFt) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apartment** | 82,956 | ₹13,047.32 | ₹9,244.51 | ₹13,062.59 | ₹202.25 | ₹98,901.10 |
| **Independent House** | 83,300 | ₹13,102.05 | ₹9,304.86 | ₹13,045.24 | ₹202.94 | ₹99,182.00 |
| **Villa** | 83,744 | ₹13,025.61 | ₹9,190.27 | ₹13,107.46 | ₹202.32 | ₹98,941.18 |

- **Observations & Interpretation**:
  There is virtually no statistical difference in unit rates across Apartments, Independent Houses, and Villas in the raw data (medians within ₹114 of each other). In physical real estate, Villas normally command higher land and privacy premiums than Apartments.
- **Visualization Reference**: [q03_price_per_sqft_by_property_type.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q03_price_per_sqft_by_property_type.png)

---

### Question 4: Relationship Between Property Size and Price
- **Records Evaluated**: $N = 250,000$
- **Numerical Findings**:
  - **Pearson Correlation ($r$)**: $-0.0019$ ($p = 0.342$)
  - **Spearman Rank Correlation ($\rho$)**: $-0.0016$ ($p = 0.424$)
  - **OLS Trendline Slope**: $-0.0002$ Lakhs / SqFt
- **Observations & Interpretation**:
  In authentic physical real estate, total property price is strongly positive-correlated with size ($r \approx 0.65\text{ to }0.85$). In this dataset, price and size were generated as **independent random variables**.
  **Crucial Mathematical Consequence**: Because $\text{Price}$ and $\text{Size}$ are independent uniform variables, their ratio $\text{Price\_per\_SqFt} = \frac{P \times 100,000}{S}$ forms a heavy-tailed ratio distribution:
  - When $P \to \text{high}$ and $S \to \text{low}$ ($500\text{L} / 500\text{ SqFt}$), unit price spikes up to ₹100,000/SqFt.
  - When $P \to \text{low}$ and $S \to \text{high}$ ($10\text{L} / 5,000\text{ SqFt}$), unit price drops to ₹200/SqFt.
  This explains the positive skew and outlier tail in `Price_per_SqFt_INR`.
- **Visualization Reference**: [q04_size_vs_price.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q04_size_vs_price.png)

---

### Question 5: Outlier Analysis for Price per SqFt and Size
- **Records Evaluated**: $N = 250,000$
- **Numerical Findings**:
  - **`Price_per_SqFt_INR`**:
    - $Q_1$: ₹4,896.22 | Median: ₹9,244.75 | $Q_3$: ₹16,211.39
    - $\text{IQR} = Q_3 - Q_1 = ₹11,315.17$
    - Tukey Upper Fence ($Q_3 + 1.5 \times \text{IQR}$): **₹33,184.14 / SqFt**
    - High Outliers ($> \text{Upper Fence}$): **20,020 records (8.01%)**
    - Low Outliers: **0 records (0.00%)**
  - **`Size_in_SqFt`**:
    - $Q_1$: 1,623 SqFt | Median: 2,747 SqFt | $Q_3$: 3,874 SqFt
    - $\text{IQR} = 2,251\text{ SqFt}$
    - Lower Fence: $-1,753.5\text{ SqFt}$ | Upper Fence: $7,250.5\text{ SqFt}$
    - Total Outliers: **0 records (0.00%)**
- **Observations & Interpretation**:
  The 20,020 statistical outliers in `Price_per_SqFt_INR` represent genuine small-size/high-price configurations resulting from ratio division rather than corrupt data entry. In accordance with strict guidelines, **these records must not be deleted**, but models will benefit from robust scaling or logarithmic transforms during feature engineering.
- **Visualization Reference**: [q05_outliers_price_sqft_size.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q05_outliers_price_sqft_size.png)

---

## 🗺️ Detailed Findings: Group 2 — Geographic Dynamics (Questions 6–10)

### Question 6: Average Price per SqFt by State
- **Records Evaluated**: $N = 250,000$ across 20 Indian States
- **Numerical Findings**:
  - **National Average**: ₹13,058.28 / SqFt
  - **Top 3 States by Mean Unit Price**:
    1. Assam: ₹13,293.73 / SqFt ($N = 12,467$)
    2. Bihar: ₹13,234.39 / SqFt ($N = 12,504$)
    3. Rajasthan: ₹13,191.02 / SqFt ($N = 12,525$)
  - **Bottom 3 States by Mean Unit Price**:
    18. Kerala: ₹12,912.87 / SqFt ($N = 12,500$)
    19. Tamil Nadu: ₹12,884.28 / SqFt ($N = 12,532$)
    20. Telangana: ₹12,866.72 / SqFt ($N = 12,509$)
- **Observations & Interpretation**:
  All 20 states have approximately equal sample representation ($\approx 12,500$ records each, or 5.0% of the dataset). The inter-state spread is narrow ($\approx 3.3\%$), indicating that baseline prices in this benchmark dataset were not geographically indexed to real-world Indian GDP or circle rates.
- **Visualization Reference**: [q06_avg_price_per_sqft_by_state.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q06_avg_price_per_sqft_by_state.png)

---

### Question 7: Average Property Price by City
- **Records Evaluated**: $N = 250,000$ across 42 Indian Cities
- **Numerical Findings**:
  - **Overall Average**: ₹254.59 Lakhs
  - **Top 5 Cities by Capital Price**:
    1. Bilaspur: ₹258.12 Lakhs ($N = 5,983$)
    2. Faridabad: ₹257.65 Lakhs ($N = 5,951$)
    3. Haridwar: ₹257.34 Lakhs ($N = 5,922$)
    4. Guwahati: ₹257.21 Lakhs ($N = 5,972$)
    5. Jodhpur: ₹257.08 Lakhs ($N = 5,889$)
  - **Key Metro Averages**:
    - Mumbai: ₹253.92 Lakhs ($N = 6,021$)
    - New Delhi: ₹253.11 Lakhs ($N = 5,892$)
    - Bangalore: ₹255.45 Lakhs ($N = 5,910$)
    - Pune: ₹254.88 Lakhs ($N = 5,981$)
    - Hyderabad: ₹254.21 Lakhs ($N = 6,004$)
- **Observations & Interpretation**:
  Tier-2 and tier-3 cities average virtually identical price levels as premier metros (Mumbai, Delhi, Bangalore). This is a vital baseline constraint for our subsequent Phase 3 and Phase 4 target construction: our investment scoring engine will need to evaluate pricing relative to locality benchmarks rather than relying on raw absolute prices.
- **Visualization Reference**: [q07_avg_price_by_city.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q07_avg_price_by_city.png)

---

### Question 8: Median Property Age by Locality
- **Records Evaluated**: $N = 250,000$ across 500 Localities (`Locality_1` to `Locality_500`)
- **Numerical Findings**:
  - **Total Localities**: 500
  - **Minimum Locality Median Age**: 16.0 years
  - **Median of Locality Medians**: 18.0 years
  - **Mean of Locality Medians**: 18.48 years
  - **Maximum Locality Median Age**: 21.0 years
  - **Standard Deviation across Localities**: 0.76 years
- **Observations & Interpretation**:
  Every locality contains properties ranging across the full spectrum of construction years (1990 to 2023), resulting in narrow locality median ages tightly clustered around 18 years. No individual locality consists purely of new developments ($< 5\text{ yrs}$) or historical structures ($> 30\text{ yrs}$).
- **Visualization Reference**: [q08_median_age_by_locality.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q08_median_age_by_locality.png)

---

### Question 9: BHK Distribution Across Prominent Cities
- **Records Evaluated**: $N = 59,500$ across Top 10 Major Metro Hubs
- **Numerical Findings**:
  - Across all evaluated cities, BHK configurations follow an exact 20% proportion:
    - 1 BHK: 19.8% to 20.3%
    - 2 BHK: 19.7% to 20.2%
    - 3 BHK: 19.8% to 20.4%
    - 4 BHK: 19.9% to 20.3%
    - 5 BHK: 19.7% to 20.2%
- **Observations & Interpretation**:
  The uniform distribution of unit layouts means housing density does not vary structurally by city type in the raw dataset. In feature engineering, interaction features like `Size_per_BHK = Size_in_SqFt / BHK` will provide a much stronger density and luxury indicator than raw BHK count.
- **Visualization Reference**: [q09_bhk_distribution_across_cities.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q09_bhk_distribution_across_cities.png)

---

### Question 10: Price Comparison of the Five Most Expensive Localities
- **Records Evaluated**: $N = 250,000$
- **Numerical Findings**:

| Locality Identifier | Mean Price/SqFt (INR) | Median Price/SqFt (INR) | Mean Capital Price (Lakhs) | Sample Count ($N$) | Premium vs. Benchmark |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Locality_207** | **₹15,203.99** | ₹10,405.25 | ₹267.13L | 467 | **+16.43%** |
| **Locality_416** | **₹14,765.48** | ₹10,165.87 | ₹260.79L | 478 | **+13.07%** |
| **Locality_246** | **₹14,567.39** | ₹10,016.67 | ₹264.37L | 510 | **+11.56%** |
| **Locality_359** | **₹14,485.44** | ₹9,822.72 | ₹262.37L | 512 | **+10.93%** |
| **Locality_387** | **₹14,429.65** | ₹9,936.57 | ₹256.31L | 482 | **+10.50%** |
| **National Benchmark** | **₹13,058.28** | **₹9,244.75** | **₹254.59L** | **250,000** | **Baseline (0.0%)** |

- **Observations & Interpretation**:
  The top 5 localities command a 10% to 16.4% premium over the national average unit rate, driven by a higher concentration of compact high-value transactions. In Phase 3, these locality indices can serve as basis for out-of-fold target encoding and locality-relative valuation indices.
- **Visualization Reference**: [q10_top5_expensive_localities.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q10_top5_expensive_localities.png)

---

## 🔗 Detailed Findings: Group 3 — Feature Relationships & Influences (Questions 11–15)

### Question 11: Numerical Correlation Heatmap
- **Records Evaluated**: $N = 250,000$ for all variables; $N = 133,696$ for valid `Floor_No` (pairwise deletion).
- **Features Included**: `Price_in_Lakhs`, `Price_per_SqFt_INR`, `Size_in_SqFt`, `BHK`, `Age_of_Property`, `Total_Floors`, `Floor_No`, `Nearby_Schools`, `Nearby_Hospitals`, `Total_Amenities_Count`.
- **Numerical Findings (Correlation Coefficients $r$)**:
  - `Price_in_Lakhs` vs. `Price_per_SqFt_INR`: **$+0.505$**
  - `Size_in_SqFt` vs. `Price_per_SqFt_INR`: **$-0.528$**
  - `Price_in_Lakhs` vs. `Size_in_SqFt`: **$-0.002$**
  - All other pairwise correlations (`BHK`, `Age`, `Floors`, `Schools`, `Hospitals`, `Amenities`): **$|r| < 0.005$**
- **Observations & Interpretation**:
  Apart from the mathematical definition of unit price ($\text{Price} / \text{Size}$), virtually all raw features are mutually uncorrelated ($r \approx 0$). This confirms the synthetic benchmark design of the dataset. For machine learning, this means tree-based models (XGBoost, LightGBM) will not suffer from multicollinearity, but high performance will depend heavily on deliberate non-linear feature engineering (such as interaction ratios, spatial aggregations, and composite scores).
- **Visualization Reference**: [q11_correlation_heatmap.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q11_correlation_heatmap.png)

---

### Question 12: Nearby Schools versus Price per SqFt
- **Records Evaluated**: $N = 250,000$ (approx. 25,000 records per integer step 1 to 10)
- **Numerical Findings**:
  - Pearson correlation: $r = -0.0013$ ($p = 0.518$)
  - Means across school tiers:
    - 1 School: ₹13,074 / SqFt
    - 5 Schools: ₹13,048 / SqFt
    - 10 Schools: ₹13,061 / SqFt
- **Observations & Interpretation**:
  School density does not automatically inflate property unit rates in the raw dataset. For investment classification, school availability will act as a quality-of-life and tenant demand filter rather than an intrinsic price driver.
- **Visualization Reference**: [q12_schools_vs_price_per_sqft.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q12_schools_vs_price_per_sqft.png)

---

### Question 13: Nearby Hospitals versus Price per SqFt
- **Records Evaluated**: $N = 250,000$
- **Numerical Findings**:
  - Pearson correlation: $r = +0.0008$ ($p = 0.675$)
  - Means across hospital tiers:
    - 1 Hospital: ₹13,049 / SqFt
    - 5 Hospitals: ₹13,069 / SqFt
    - 10 Hospitals: ₹13,055 / SqFt
- **Observations & Interpretation**:
  Proximity to healthcare facilities displays balanced valuation across all 10 integer levels without artificial pricing skew.
- **Visualization Reference**: [q13_hospitals_vs_price_per_sqft.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q13_hospitals_vs_price_per_sqft.png)

---

### Question 14: Property Price by Furnished Status
- **Records Evaluated**: $N = 250,000$
  - Unfurnished: $N = 83,408$ (33.36%)
  - Semi-furnished: $N = 83,374$ (33.35%)
  - Furnished: $N = 83,218$ (33.29%)
- **Numerical Findings**:
  - **Unfurnished**: Mean: ₹254.98L | Median: ₹254.19L | Std: ₹141.40L
  - **Semi-furnished**: Mean: ₹254.33L | Median: ₹253.81L | Std: ₹141.45L
  - **Furnished**: Mean: ₹254.45L | Median: ₹253.56L | Std: ₹141.20L
- **Observations & Interpretation**:
  Total property price is invariant to furnished status. While real-world buyers pay a premium for turnkey furnished homes, the raw dataset does not embed a hardcoded pricing premium for furnishing.
- **Visualization Reference**: [q14_price_by_furnished_status.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q14_price_by_furnished_status.png)

---

### Question 15: Price per SqFt by Facing Direction
- **Records Evaluated**: $N = 250,000$
  - East: $N = 62,269$ (24.91%)
  - North: $N = 62,637$ (25.05%)
  - South: $N = 62,337$ (24.93%)
  - West: $N = 62,757$ (25.10%)
- **Numerical Findings**:
  - **East**: Mean: ₹13,023.70 / SqFt | Median: ₹9,234.01 / SqFt
  - **North**: Mean: ₹13,024.87 / SqFt | Median: ₹9,219.23 / SqFt
  - **South**: Mean: ₹13,044.11 / SqFt | Median: ₹9,224.63 / SqFt
  - **West**: Mean: ₹13,140.02 / SqFt | Median: ₹9,307.59 / SqFt
- **Observations & Interpretation**:
  Vastu-preferred orientations (East and North) do not carry an inherent unit rate premium over South or West in this dataset (means within $0.8\%$).
- **Visualization Reference**: [q15_price_per_sqft_by_facing.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q15_price_per_sqft_by_facing.png)

---

## 💼 Detailed Findings: Group 4 — Investment, Amenities & Ownership (Questions 16–20)

### Question 16: Owner Type Distribution
- **Records Evaluated**: $N = 250,000$
- **Numerical Findings**:
  - **Broker**: 83,479 records (**33.39%**)
  - **Owner**: 83,268 records (**33.31%**)
  - **Builder**: 83,253 records (**33.30%**)
- **Observations & Interpretation**:
  The market demonstrates equal participation across all three seller categories. This provides balanced data for studying builder direct discounts, broker commissions, and resale dynamics in Phase 3.
- **Visualization Reference**: [q16_owner_type_distribution.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q16_owner_type_distribution.png)

---

### Question 17: Availability Status Distribution
- **Records Evaluated**: $N = 250,000$
- **Numerical Findings**:
  - **Under Construction**: 125,035 records (**50.01%**)
  - **Ready to Move**: 124,965 records (**49.99%**)
- **Observations & Interpretation**:
  The dataset is divided evenly between ready and under-construction inventory (difference of only 70 records out of a quarter-million). This ensures zero class imbalance when modeling project delivery stage risks.
- **Visualization Reference**: [q17_availability_status_distribution.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q17_availability_status_distribution.png)

---

### Question 18: Parking Space versus Property Price
- **Records Evaluated**: $N = 250,000$
  - Without Parking (`No`): $N = 125,456$ (50.18%)
  - With Parking (`Yes`): $N = 124,544$ (49.82%)
- **Numerical Findings**:
  - **Without Parking**: Mean: ₹254.43 Lakhs | Median: ₹253.37 Lakhs | Std: ₹141.49L
  - **With Parking**: Mean: ₹254.75 Lakhs | Median: ₹254.36 Lakhs | Std: ₹141.21L
  - **Mean Difference**: +₹0.32 Lakhs
  - **Two-Sample Welch's t-test**: $t = +0.559$, $p\text{-value} = 0.5760$
- **Observations & Interpretation**:
  The price difference of ₹32,000 on a ₹2.5 Crore property has a high p-value of 0.5760, indicating that dedicated parking does not induce a statistically significant price premium in the raw baseline data.
- **Visualization Reference**: [q18_parking_vs_price.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q18_parking_vs_price.png)

---

### Question 19: Amenities versus Price per SqFt
- **Records Evaluated**: $N = 250,000$
- **Numerical Findings**:
  - **Unit Rate by Total Amenities Count**:
    - 1 Amenity ($N = 50,106$): Mean ₹13,071.18 / SqFt
    - 2 Amenities ($N = 49,806$): Mean ₹13,040.54 / SqFt
    - 3 Amenities ($N = 49,862$): Mean ₹13,045.39 / SqFt
    - 4 Amenities ($N = 50,362$): Mean ₹13,045.54 / SqFt
    - 5 Amenities ($N = 49,864$): Mean ₹13,088.75 / SqFt
  - **Atomic Amenity Premiums**:
    - `Has_Gym`: With (₹13,059) vs Without (₹13,056) $\to$ Diff: +₹3/SqFt
    - `Has_Pool`: With (₹13,057) vs Without (₹13,060) $\to$ Diff: -₹3/SqFt
    - `Has_Garden`: With (₹13,061) vs Without (₹13,054) $\to$ Diff: +₹7/SqFt
    - `Has_Playground`: With (₹13,060) vs Without (₹13,055) $\to$ Diff: +₹5/SqFt
    - `Has_Clubhouse`: With (₹13,058) vs Without (₹13,058) $\to$ Diff: ₹0/SqFt
- **Observations & Interpretation**:
  Individual amenities do not drive unit price variation in the raw records. However, `Total_Amenities_Count` forms an essential institutional pillar for consumer desirability, rental yield, and investment viability scoring in Phase 3.
- **Visualization Reference**: [q19_amenities_vs_price_per_sqft.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q19_amenities_vs_price_per_sqft.png)

---

### Question 20: Public Transport Accessibility versus Price and Potential
- **Records Evaluated**: $N = 250,000$
  - Low: $N = 83,287$ (33.31%)
  - Medium: $N = 83,008$ (33.20%)
  - High: $N = 83,705$ (33.48%)
- **Numerical Findings**:
  - **Capital Price**:
    - Low: Mean ₹254.47L | Median ₹253.51L
    - Medium: Mean ₹254.67L | Median ₹253.86L
    - High: Mean ₹254.62L | Median ₹254.21L
  - **Unit Rate (`Price_per_SqFt_INR`)**:
    - Low: Mean ₹13,047.88 / SqFt | Median ₹9,239.38 / SqFt
    - Medium: Mean ₹13,059.25 / SqFt | Median ₹9,246.60 / SqFt
    - High: Mean ₹13,067.70 / SqFt | Median ₹9,248.33 / SqFt
- **Observations & Interpretation**:
  While unit prices are flat across transport accessibility tiers in the raw data, properties with **High Public Transport Accessibility** represent prime candidates for rapid appreciation, reduced vacancy, and high tenant turnover. In our Phase 4 investment model, public transport accessibility will serve as a core multiplier for 5-year future valuation forecasting.
- **Visualization Reference**: [q20_transport_vs_price_potential.png](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/q20_transport_vs_price_potential.png)

---

## 🔍 Critical Analytical Anomalies & Modeling Implications

1. **Synthetic Data Independence & Uniformity**:
   The raw dataset exhibits characteristic benchmark properties: features like `Price_in_Lakhs`, `Size_in_SqFt`, `BHK`, `Total_Floors`, and categorical proportions were sampled independently from uniform bounded distributions.
2. **Ratio Variable Heavy Tail**:
   Because price and size are mutually uncorrelated ($r \approx 0$), `Price_per_SqFt_INR` exhibits an 8.01% upper outlier tail ($> ₹33,184/\text{SqFt}$). These are mathematically valid data points resulting from small units with high prices and should be handled with robust or quantile-based scaling.
3. **Absence of Pre-built Target Labels**:
   As discovered in initial audits and reaffirmed in EDA, there is no pre-existing investment classification label or 5-year future price column.
4. **Target Construction Hygiene**:
   Formulating the classification target ("Good Investment") and regression target ("5-Year Future Price") must be grounded in transparent financial logic (e.g., locality-relative valuation discounts, infrastructure accessibility premiums, rental yield proxies, and compounded annual growth rates) executed strictly **after** partitioning train and validation splits to prevent out-of-fold data leakage.

---

*Report preserved at [reports/eda/eda_findings.md](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/eda/eda_findings.md).*
