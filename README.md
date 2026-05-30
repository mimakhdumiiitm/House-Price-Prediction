# House-Price-Prediction

This repository contains an end-to-end Machine Learning pipeline to solve a comprehensive regression problem. This project was developed as **Assignment 1** for the **Machine Learning Practice (MLP)** course, which is part of the **IIT Madras BS Degree in Data Science and Applications**.

The dataset originally originates from a Kaggle competition specifically tailored for this academic evaluation. Although the competition has now been made private by the authorities, this codebase reflects the complete workflow, optimization phases, and final high-performing ensemble submission file.

> ⚠️ **Notebook Note:** The notebook may not render on GitHub due to a persistent `nbformat`/`nbconvert` issue. Please use the Google Colab link below.
> [Open Notebook in Colab](https://colab.research.google.com/drive/1ItFHH4XjOrkh3DhKz9do3s70BnKDQPj0?usp=sharing)

## 📋 Project Objective
The goal is to predict property values based on multi-dimensional real estate characteristics (such as location features, spatial configuration, total square footage, bathroom/balcony counts, and area configurations).

---

## 🛠️ Pipeline Architecture & Workflow

### 1. Exploratory Data Analysis (EDA)
* **Data Typification:** Evaluated structural layouts across both numerical (`float64`, `int64`) and categorical (`object`) attributes.
* **Descriptive Statistics:** Audited training and test features for statistical variances, checking central tendency boundaries and dispersion scales (`mean`, `std`, `min`, `max`, percentiles).
* **Missing Value & Duplicate Audit:** Mapped exact distributions of missing entries and confirmed the absolute uniqueness of observations.

### 2. Custom Feature Engineering
Dynamic transformations were injected to capture non-linear relationships:
* **Text Mining Extractor:** Isolated structural capacity (`bedrooms`) by dynamically parsing textual size descriptions.
* **Derived Spatial Ratios:**
  * `total_rooms` = Sum of bedrooms, bathrooms, and balconies.
  * `sqft_per_room` & `sqft_per_bedroom` = Spatial densities relative to layout dimensions.
  * `bath_ratio` & `balcony_ratio` = Functional balance indices across available spaces.
* **Geographical Weighting:** Transformed high-cardinality values via calculated frequencies (`location_freq`) to handle variance in rare localities.
* **Binary Space Classification:** Isolated built-up configurations (`is_built_up`, `is_super_built`, `is_carpet`) and immediate occupancy statuses (`is_ready_to_move`).

### 3. Preprocessing & Scaler Operations
* **Imputation Strategy:** Utilized `SimpleImputer` to impute missing continuous variables via robust medians and categorical vectors via the most frequent observations.
* **Percentile Capping:** Outliers were tightly controlled across continuous attributes using 1st and 99th percentile IQR threshold boundaries.
* **Encoding & Regularization:** Categorical identifiers were transformed via structural `LabelEncoder`, and numerical scales normalized symmetrically utilizing `StandardScaler`.

### 4. Robust Regression Suite (7+ Architecture Setup)
A diverse ensemble matrix consisting of more than 7 algorithms was constructed to extract independent prediction expectations:
1. **XGBoost (Regressor 1):** Optimized tree architecture with balanced column sampling constraints.
2. **XGBoost (Regressor 2):** Diversified tree depth alternative to reduce collinear model correlation.
3. **LightGBM:** Fast, leaf-wise histogram gradient booster optimized for distributed boundaries.
4. **CatBoost:** Symmetric tree growth specifically tailored to smooth structural variances.
5. **Ridge Regression:** L2 regularized linear baseline to bound high-dimensional variance.
6. **Lasso Regression:** L1 feature-selecting estimator to penalize non-contributing dimensions.
7. **Gradient Boosting Regressor (GBR):** Deep sequential learning estimator focused on error residuals.
8. **Random Forest Regressor:** Multi-tree bagging model used to ensure generalized expectations.

### 5. Hyperparameter Optimization
Selected models were dynamically optimized through strategic 3-fold cross-validated grid tuning arrays (`GridSearchCV`) focusing on minimizing structural mean squared errors:
* Tuned alpha regularization vectors inside the `Ridge` space.
* Scaled depth metrics and estimators for the `RandomForest` matrix.
* Tailored structural leaves, learning rates, and estimator bounds within `LightGBM`.

### 6. Meta-Ensemble Stacking Plan
* Implemented a structured **5-Fold Stacking Regressor** to eliminate out-of-fold leakage.
* Baseline model prediction logs were collected as meta-features.
* A meta-level `Ridge(alpha=1.0)` estimator was fit on the predictions to compute a perfectly regularized consensus projection vectors.

---

## 📈 Cross-Validation Performance Summary

The individual 3-Fold Cross-Validation metrics evaluated on the target $log(1 + \text{price})$ space demonstrate high accuracy bounds:

| Model Architecture | 3-Fold CV Root Mean Squared Error (RMSE) |
|:---|:---:|
| **LightGBM (lgb)** | **0.3193** |
| **LightGBM (Tuned)** | **0.3205** |
| **XGBoost (xgb1)** | **0.3211** |
| **XGBoost (xgb2)** | **0.3238** |
| **Gradient Boosting (gbr)** | **0.3273** |
| **CatBoost (cat)** | **0.3313** |
| **Random Forest (rf)** | **0.3472** |
| **Ridge Regression** | **0.4315** |
| **Lasso Regression** | **0.4316** |

### Meta-Model Prediction Statistics (Exponentially Inverted Prices):
* **Mean Expected Valuation:** 107.69
* **Standard Deviation Index:** 91.14
* **Absolute Valuation Limits:** 17.14 to 1207.47

---

## 💾 Artifact Generation
The final stacked predictions were successfully mapped back to original scale limits using logarithmic inversion (`np.expm1`), generating a standalone, target evaluation output file named `submission.csv`.

