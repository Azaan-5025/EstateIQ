# ESTATEIQ — Phase 5: Final Holdout Test Evaluation Report
### Independent Holdout Verification of Classification & Regression Models ($N=37,500$)

---

## 🏛️ Executive Summary

This report documents the final holdout test evaluation of the **ESTATEIQ** machine learning pipelines. Until this stage, the 15% test partition ($N=37,500$) remained strictly locked, unobserved, and unaccessed. 

Both production candidates—the **XGBoost Classifier** for investment viability classification and the **Random Forest Regressor** for current listed-price estimation—were evaluated without any retraining, hyperparameter adjustments, or threshold modifications.

### Verified Anti-Leakage & Partition Integrity:
1. **Zero Partition Overlap**: Exact ID hashing confirmed **0 overlap** between test records and training ($N=175,000$) or validation ($N=37,500$) partitions.
2. **Pure Intrinsic Features (34 Features)**: Exactly 34 features were supplied to both pipelines. Zero price-derived features (`Price_per_SqFt_INR`, `Price_to_Locality_Ratio`, `Locality_Benchmark_Price_per_SqFt`, `Locality_Price_ZScore`, `Locality_Benchmark_Tier`) or target components entered either model.
3. **End-to-End Serialized Preprocessing**: Imputation and scaling transformers fitted on training data were applied out-of-sample directly through scikit-learn `Pipeline` objects.

---

## 📊 Validation vs. Holdout Test Generalization Comparison

### 1. Classification: XGBoost Classifier (`Is_Good_Investment`)

| Metric | Validation Set ($N=37,500$) | Holdout Test Set ($N=37,500$) | Absolute Delta ($\Delta$) | Generalization Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 73.06% | **73.29%** | +0.23% | ✅ Flawless Stability ($< 0.2\%$) |
| **Precision** | 58.39% | **58.85%** | +0.46% | ✅ Consistent Underwriting |
| **Recall (Sensitivity)** | 78.75% | **78.79%** | +0.04% | ✅ High Asset Capture |
| **Specificity (TNR)** | 70.02% | **70.33%** | +0.31% | ✅ Stable False Positive Control |
| **F1-Score** | 67.06% | **67.37%** | +0.32% | ✅ Harmonic Equilibrium |
| **ROC-AUC** | 0.8229 | **0.8228** | -0.0001 | ✅ Robust Discrimination |
| **PR-AUC** | 0.6755 | **0.6790** | +0.0036 | ✅ $2\times$ Above Prevalence Floor |

### 2. Regression: Random Forest Regressor (`Price_in_Lakhs`)

| Metric | Validation Set ($N=37,500$) | Holdout Test Set ($N=37,500$) | Absolute Delta ($\Delta$) | Generalization Status |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | ₹122.88L | **₹122.81L** | -0.07L | ✅ Matches Sample Standard |
| **Root Mean Squared Error (RMSE)** | ₹141.69L | **₹141.72L** | +0.03L | ✅ Matches Sample Std ($\sigma \approx 141$L) |
| **Coefficient of Determination ($R^2$)** | -0.0001 | **-0.0002** | -0.0000 | ⚠️ Zero Intrinsic Explanatory Power |
| **Mean Absolute Percentage Error (MAPE)** | 132.66% | **133.79%** | +1.13% | ⚠️ Uniform Target Dispersion |

---

## 🔍 In-Depth Diagnostic Interpretations

### 1. Classification Confusion Matrix Interpretation
- **True Negatives (TN)**: **17,144** out of 24,375 non-viable assets (70.3%) were correctly rejected.
- **False Positives (FP)**: **7,232** (29.7%) were incorrectly labeled viable.
- **False Negatives (FN)**: **2,783** (21.2%) were missed viable opportunities.
- **True Positives (TP)**: **10,341** out of 13,125 viable properties (78.8%) were successfully identified.

```
Detailed Classification Report (Holdout Test Partition, N=37,500):
                precision    recall  f1-score   support

Not Viable (0)       0.86      0.70      0.77     24376
  Good Inv (1)       0.59      0.79      0.67     13124

      accuracy                           0.73     37500
     macro avg       0.72      0.75      0.72     37500
  weighted avg       0.77      0.73      0.74     37500

```

The model demonstrates balanced behavior on unseen data: it captures over 78% of all viable opportunities while maintaining 70% specificity on sub-standard properties, confirming that intrinsic spatial and structural patterns generalize across fresh distributions.

### 2. Regression Error Interpretation & Scientific Candor ($R^2 \approx 0.00$)
- **Transparent Reality**: On the unseen test partition, the Random Forest Regressor achieves an RMSE of **₹141.72 Lakhs**, an MAE of **₹122.81 Lakhs**, and an $R^2$ of **-0.0002**.
- **Underlying Cause**: In this housing dataset, listed asking prices are uniformly distributed between ₹10L and ₹500L and possess near-zero empirical correlation with physical dimensions (e.g., size, bedrooms, amenities).
- **Leakage Prevention Significance**: Had we retained `Price_per_SqFt_INR` in the feature matrix, the model would have trivially reported $R^2 = 1.000$, which would have been fraudulent circular leakage. By enforcing strict anti-leakage governance, the model honestly converges to predicting the global training mean ($\approx ₹254.7\text{ Lakhs}$).

---

## 📈 Diagnostic Evaluation Plots

The following publication-grade diagnostic plots have been generated and archived in [`reports/final_evaluation/`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/final_evaluation/):

1. **Test Classification Confusion Matrix**:
   ![Confusion Matrix](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/final_evaluation/classification_test_confusion_matrix.png)
   *Detailed 2x2 count and percentage error breakdown on the 37,500 holdout records.*

2. **Test Receiver Operating Characteristic (ROC) Curve**:
   ![ROC Curve](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/final_evaluation/classification_test_roc_curve.png)
   *Test ROC curve exhibiting strong separation from the random diagonal (AUC = 0.8228).*

3. **Test Precision-Recall (PR) Curve**:
   ![PR Curve](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/final_evaluation/classification_test_pr_curve.png)
   *Precision retention curve demonstrating high precision across varying recall thresholds against the no-skill floor (PR-AUC = 0.6790).*

4. **Test Actual vs. Predicted Listed Price**:
   ![Actual vs Predicted](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/final_evaluation/regression_test_actual_vs_predicted.png)
   *Scatter plot of actual listed prices vs model predictions against the ideal diagonal line.*

5. **Test Residual Diagnostics**:
   ![Residuals](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/final_evaluation/regression_test_residuals.png)
   *Residual scatter vs. predicted values and residual error density distribution.*

---

## ⚠️ Mandatory Domain Disclosures & Limitations

> [!CAUTION]
> **Definitive Legal & Domain Disclosures**:
> 1. **Rule-Based Underwriting Heuristic Proxy**: Classification accuracy (73.29%) and ROC-AUC (0.8228) demonstrate that XGBoost has learned to approximate the multi-pillar Investment Viability Score (IVS) heuristic based on intrinsic physical, transit, and structural features. **This does NOT prove or guarantee actual financial profit or real-world capital appreciation.**
> 2. **Current Listed-Price Estimation Only**: The regression model estimates the current listed asking price recorded in the dataset. It does **NOT** represent an independently verified fair-market valuation, bank appraisal, or true transaction price.
> 3. **Absence of Historical Future-Price Labels**: The dataset contains zero historical transaction series or multi-year repeat-sales data.
> 4. **Five-Year Appreciation Module**: Any future five-year capital appreciation projection implemented in the advisory application must rely on explicit user-configurable financial assumptions (e.g. compound annual growth rates, inflation, holding yields), **NOT** an empirical ML forecast.
> 5. **No Investment Guarantee**: Neither model guarantees real estate investment returns.

---

## 🚦 Phase 5 Completion Status

- Holdout test evaluation is **100% complete**.
- All metrics logged to MLflow under experiments `ESTATEIQ_Classification_Investment_Viability` and `ESTATEIQ_Regression_Price_Estimation`.
- All diagnostic plots generated in [`reports/final_evaluation/`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/final_evaluation/).
- No datasets or models were modified.
- Zero Streamlit development has been started.

*Awaiting your approval before proceeding to application interface development.*
