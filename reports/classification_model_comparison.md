# ESTATEIQ — Phase 4 (Stage 1): Classification Model Evaluation & Comparison Report
### Systematic Benchmarking of Investment Viability Classifiers on Pure Intrinsic Features

---

## 🏛️ Executive Summary

This report documents the rigorous training, hyperparameter validation, and comparative evaluation of supervised machine learning classifiers for the **ESTATEIQ** real estate investment advisory platform.

### Core Architecture & Anti-Leakage Safeguards:
1. **Target Variable**: `Is_Good_Investment` (Binary: 1 = Viable, 0 = Not Viable), derived from the 4-pillar Investment Viability Score (IVS) at the approved threshold of $\ge 65.0$.
2. **Strict Leakage Purge**: The modeling matrix $\mathbf{X}_{\text{class}}$ contains **exactly 34 pure intrinsic physical, structural, spatial, and civic features**. All listed prices, unit price rates, price ratios, locality benchmarks, and target components were strictly purged.
3. **Partition Discipline**: All data transformations (imputation, standard scaling, and one-hot encoding) and model estimators were fitted strictly on the **Training partition (175,000 records, 70%)**. Model selection and comparative evaluation were conducted exclusively on the **Validation partition (37,500 records, 15%)**. The **Test partition (37,500 records, 15%)** remains strictly untouched.
4. **Experiment Tracking**: All 4 model runs, parameters, metrics, confusion matrices, and model artifacts were logged to **MLflow** under experiment `ESTATEIQ_Classification_Investment_Viability`.

---

## 📊 Comprehensive Model Comparison Table (Validation Set, N=37,500)

| Candidate Model | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Val ROC-AUC | Val PR-AUC | Train Accuracy | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Most Frequent)** | 65.18% | 0.00% | 0.00% | 0.00% | 0.5000 | 0.3482 | 65.19% | 0.58s |
| **Logistic Regression (Balanced)** | 71.43% | 57.08% | 72.35% | 63.82% | 0.7883 | 0.6446 | 71.78% | 1.59s |
| **Random Forest Classifier** | 71.66% | 57.01% | 75.67% | 65.02% | 0.8010 | 0.6425 | 73.98% | 68.59s |
| **XGBoost Classifier** | 73.06% | 58.39% | 78.75% | 67.06% | 0.8229 | 0.6755 | 74.63% | 17.36s |

> [!IMPORTANT]
> **Dominant Performance Verdict**:
> **XGBoost Classifier** emerges as the unequivocal champion across all primary discrimination and ranking criteria:
> - **Validation Accuracy**: **73.06%** (surpassing the Dummy Baseline by +7.88 percentage points).
> - **Validation F1-Score**: **67.06%** (superior harmonic balance between precision and coverage).
> - **Validation ROC-AUC**: **0.8229** (exceptional area under the receiver operating characteristic curve).
> - **Validation PR-AUC**: **0.6755** (substantially above the 34.82% no-skill baseline).

---

## 🔍 In-Depth Model Breakdown & Analysis

### 1. Dummy Classifier Baseline (Most Frequent)
- **Strategy**: Always predicts Class 0 (Not Viable), corresponding to the majority class (65.18%).
- **Validation Accuracy**: 65.18%
- **Validation F1-Score**: 0.00% (Precision = 0.00%, Recall = 0.00%)
- **Validation ROC-AUC**: 0.5000 (Random guess floor)
- **Role**: Establishes the non-trivial performance boundary. Any predictive model must demonstrate significant positive precision, recall, and ROC-AUC above 0.50 to prove it has learned meaningful domain signal.

### 2. Logistic Regression (L2 Regularized, Balanced Weights)
- **Parameters**: `C=1.0`, `solver='lbfgs'`, `class_weight='balanced'`, `max_iter=1000`.
- **Validation Accuracy**: 71.43%
- **Validation Precision**: 57.08%
- **Validation Recall**: 72.35%
- **Validation F1-Score**: 63.82%
- **Validation ROC-AUC**: 0.7883
- **Analysis**: As a linear estimator, Logistic Regression captures linear dependencies well and benefits from balanced sample weighting. However, its linear decision boundary cannot capture non-linear feature interactions (such as the compounding interplay between connectivity score, property quality, and asset age).

### 3. Random Forest Classifier
- **Parameters**: `n_estimators=150`, `max_depth=16`, `min_samples_split=15`, `min_samples_leaf=8`, `class_weight='balanced'`.
- **Validation Accuracy**: 71.66%
- **Validation Precision**: 57.01%
- **Validation Recall**: 75.67%
- **Validation F1-Score**: 65.02%
- **Validation ROC-AUC**: 0.8010
- **Analysis**: Random Forest's bagged ensembles reduce variance and capture high-order non-linear combinations across tree depths. The balanced sub-sampling produces strong sensitivity (recall) on viable investment assets.

### 4. XGBoost Classifier (Gradient Boosted Decision Trees)
- **Parameters**: `n_estimators=250`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=1.873`, `tree_method='hist'`.
- **Validation Accuracy**: 73.06%
- **Validation Precision**: 58.39%
- **Validation Recall**: 78.75%
- **Validation F1-Score**: 67.06%
- **Validation ROC-AUC**: 0.8229
- **Validation PR-AUC**: 0.6755
- **Analysis**: XGBoost's sequential second-order gradient boosting with histogram-based split optimization delivers the highest precision and discrimination power. It effectively identifies properties that score well on connectivity and asset quality while maintaining resilience against noisy spatial nominals.

---

## 📈 Evaluation Visualizations & Diagnostics

The following diagnostic plots have been generated and archived in [`reports/classification/`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/):

1. **Confusion Matrices Grid**:
   ![Confusion Matrices](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/confusion_matrices_comparison.png)
   *Detailed 2x2 comparison displaying true positives, false positives, true negatives, and false negatives across all 4 models.*

2. **Receiver Operating Characteristic (ROC) Curves**:
   ![ROC Curves](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/roc_curves_comparison.png)
   *Multi-model ROC curve demonstrating separation from the random diagonal.*

3. **Precision-Recall (PR) Curves**:
   ![Precision-Recall Curves](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/pr_curves_comparison.png)
   *Demonstrates precision retention across varying recall thresholds against the 34.82% no-skill prevalence floor.*

4. **Top 20 Intrinsic Feature Importances**:
   ![Feature Importances](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/feature_importances_top20.png)
   *Identifies the strongest intrinsic physical, structural, and civic drivers of investment viability.*

5. **Metrics Summary Bar Chart**:
   ![Metrics Comparison](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/reports/classification/metrics_comparison_barchart.png)
   *Comparative visualization across Accuracy, Precision, Recall, F1, and ROC-AUC.*

---

## 🏆 Selection of the Best Classification Model

### Winning Model: **XGBoost Classifier**
- **Persistence Path**: [`models/best_classification_model.joblib`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/models/best_classification_model.joblib)
- **Metadata**: [`models/classification_model_metadata.json`](file:///c:/Users/Syed%20Azaan%20Hussain/Desktop/ESTATEIQ/models/classification_model_metadata.json)

### Why This Model Was Selected:
1. **Optimal ROC-AUC & PR-AUC**: It delivers the highest area under both the ROC and PR curves, guaranteeing that threshold tuning can be flexibly adapted to conservative or aggressive investment profiles.
2. **High Precision Without Sacrificing Recall**: In real estate advisory, **precision is paramount**—recommending a bad property as a "Good Investment" (False Positive) damages investor trust and capital. The winning model minimizes False Positives while capturing a large fraction of truly viable properties.
3. **Generalization Stability**: Train accuracy vs. Validation accuracy demonstrates healthy regularization without catastrophic overfitting.

---

## ⚠️ Mandatory Domain Disclosures & Limitations

> [!CAUTION]
> **Definitive Proxy Label Disclosure**:
> 1. **Classification Accuracy ≠ Market Profitability**: High classification accuracy on `Is_Good_Investment` proves that the model has successfully learned and generalized the multi-criteria underwriting heuristic rules (connectivity, structural quality, asset vintage, and delivery status). It does **NOT** guarantee that an asset will appreciate in value or deliver financial profit in the real world.
> 2. **Absence of Temporal Data**: The dataset contains cross-sectional property listings rather than longitudinal transaction histories. The model classifies based on intrinsic feature quality at the recorded snapshot in time.

---

## 🚦 Phase 4 (Stage 1) Status

- Classification model training and evaluation are **100% complete**.
- Best model is saved to `models/best_classification_model.joblib`.
- All runs are tracked in MLflow.
- The test partition remains strictly untouched.
- Zero regression models have been trained.

*Awaiting your review and approval before proceeding to Phase 4 (Stage 2: Regression Model Training & Evaluation for Current Listed-Price Estimation).*
