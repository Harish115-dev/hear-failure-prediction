<div align="center">

# ❤️ Heart Failure Clinical Prediction

**Predicting patient mortality from heart failure using machine learning on real clinical records**

*Best model: Random Forest · Accuracy 83.3% · ROC-AUC 0.899*
---

## 📌 Project Overview

Heart failure is a serious medical condition where the heart cannot pump blood effectively — and one of the leading causes of death worldwide. Early prediction of high-risk patients enables healthcare professionals to prioritize interventions and improve outcomes.

This project applies **six machine learning classifiers** to clinical health records, using SMOTE to handle class imbalance and GridSearchCV for hyperparameter tuning. Models are evaluated not just on accuracy, but on **recall** and **ROC-AUC** — the metrics that matter most in a clinical setting where missing a true death is far more costly than a false alarm.

---

## 🎯 Objectives

- Perform in-depth Exploratory Data Analysis (EDA) on heart failure clinical data
- Identify relationships between clinical features and patient mortality
- Handle class imbalance using SMOTE oversampling
- Build, tune, and compare six ML classification models
- Select the best model based on clinically relevant metrics (Recall, F1, ROC-AUC)

---

## 📊 Dataset Information

**Source:** [Heart Failure Clinical Records Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

| Property | Value |
|---|---|
| Total Records | 299 patients |
| Input Features | 12 clinical variables |
| Target Variable | `DEATH_EVENT` (0 = Survived, 1 = Died) |
| Class Split | ~68% Survived · ~32% Died |
| Missing Values | None |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `age` | float | Age of patient (years) |
| `anaemia` | binary | Decrease of red blood cells or haemoglobin |
| `creatinine_phosphokinase` | int | Level of CPK enzyme in blood (mcg/L) |
| `diabetes` | binary | Whether the patient has diabetes |
| `ejection_fraction` | int | % of blood leaving the heart at each contraction |
| `high_blood_pressure` | binary | Whether the patient has hypertension |
| `platelets` | float | Platelet count in blood (kiloplatelets/mL) |
| `serum_creatinine` | float | Level of serum creatinine in blood (mg/dL) |
| `serum_sodium` | int | Level of serum sodium in blood (mEq/L) |
| `sex` | binary | 0 = Female, 1 = Male |
| `smoking` | binary | Whether the patient smokes |
| `time` | int | Follow-up period (days) |
| `DEATH_EVENT` | binary | **Target** — 0 = Survived, 1 = Died |

---

## 📂 Project Structure

```text
Heart-Failure-Prediction/
│
├── dataset/
│   └── heart_failure_clinical_records_dataset.csv
│
├── notebooks/
│   └── Heart_Failure_Prediction.ipynb
│
├── images/
│   ├── correlation_heatmap.png
│   ├── age_distribution.png
│   ├── class_distribution.png
│   └── confusion_matrix_rf.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ▶️ Installation

```bash
# Clone the repository
git clone https://github.com/Harish115-dev/hear-failure-prediction.git
cd Heart-Failure-Prediction

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
xgboost
jupyter
```

---

## ▶️ Run the Project

```bash
jupyter notebook notebooks/Heart_Failure_Prediction.ipynb
```

Open the notebook and run all cells in order.

---

## 🛠 Technologies Used

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` | EDA visualizations, confusion matrices |
| `scikit-learn` | Preprocessing, modeling, GridSearchCV, evaluation |
| `imbalanced-learn` | SMOTE oversampling for class imbalance |
| `xgboost` | Gradient boosting classifier |
| `Jupyter Notebook` | Interactive development environment |

---

## 📈 Exploratory Data Analysis

The following analyses were performed:

- Dataset info, shape, and data types
- Statistical summary (`describe()`)
- Missing value analysis — **zero missing values found**
- Outlier detection using IQR method per numerical feature
- Class distribution of `DEATH_EVENT`
- Age distribution by death event
- Correlation matrix and heatmap
- Feature vs target analysis

### Outliers Detected (IQR Method)

| Feature | Outliers |
|---|---|
| `creatinine_phosphokinase` | 29 |
| `serum_creatinine` | 29 |
| `platelets` | 21 |
| `serum_sodium` | 4 |
| `ejection_fraction` | 2 |
| `age` / `time` | 0 |

> Outliers were **retained** as they likely reflect real clinical extremes rather than data errors.

### Top Correlations with `DEATH_EVENT`

| Feature | Correlation | Interpretation |
|---|---|---|
| `time` | −0.527 | Longer follow-up → lower mortality |
| `ejection_fraction` | −0.269 | Higher EF → lower mortality |
| `serum_sodium` | −0.195 | Higher sodium → lower mortality |
| `serum_creatinine` | +0.294 | Higher creatinine → higher mortality |
| `age` | +0.254 | Older age → higher mortality |

---

## ⚙️ Data Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Feature / target split
X = data.drop("DEATH_EVENT", axis=1)
Y = data["DEATH_EVENT"]

# 70/30 stratified split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.30, stratify=Y, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# SMOTE — applied to training set only to prevent data leakage
sm = SMOTE(random_state=42)
X_train, Y_train = sm.fit_resample(X_train, Y_train)
```

> **Why SMOTE?** The dataset has a ~68/32 class split. Without balancing, models tend to ignore the minority class (deaths). SMOTE synthetically oversamples the minority class in the training set, improving recall on high-risk patients.

---

## 🤖 Machine Learning Models

Six classifiers were trained and compared:

| # | Model | Notes |
|---|---|---|
| 1 | Logistic Regression | Baseline linear model, `class_weight="balanced"` |
| 2 | Support Vector Machine (SVM) | Linear SVC, `class_weight="balanced"` |
| 3 | K-Nearest Neighbors (KNN) | Distance-based, sensitive to scale |
| 4 | Random Forest | Ensemble of decision trees, `class_weight="balanced"` |
| 5 | XGBoost | Gradient boosting with `scale_pos_weight` |
| 6 | Tuned Random Forest | GridSearchCV optimized for **Recall** |

---

## 📋 Model Results

**Evaluation setup:** 70/30 stratified split · SMOTE on training set only · StandardScaler applied

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| KNN | 67.78% | 50.00% | 51.72% | 50.85% |
| SVM (Linear) | 76.67% | 63.33% | 65.52% | 64.41% |
| Logistic Regression | 76.67% | 63.33% | 65.52% | 64.41% |
| XGBoost | 82.22% | **84.21%** | 55.17% | 66.67% |
| Random Forest | **83.33%** | 75.00% | **72.41%** | **73.68%** |
| Tuned Random Forest | **83.33%** | 75.00% | **72.41%** | **73.68%** |

### 🏆 Best Model — Random Forest

```
Accuracy  :  83.33%
Precision :  75.00%
Recall    :  72.41%
F1 Score  :  73.68%
ROC-AUC   :  0.899
```

**GridSearchCV best parameters** (optimized for Recall):
```python
{
  "n_estimators":     200,
  "max_depth":        7,
  "min_samples_split": 5
}
```

**10-fold cross-validated recall:** 71.33% ± 33.89%

> **Why not XGBoost?** XGBoost achieved the highest precision (84.2%) but its recall fell to 55.2% — meaning it missed nearly half of actual deaths. In a clinical context, **false negatives are more dangerous than false positives**, making Random Forest the safer and better-balanced choice.

---

## 📈 Feature Importance

*Derived from the trained Random Forest model*

```
time                      ████████████████████  33.4%   ← strongest predictor
serum_creatinine          ██████████            17.0%   ← key clinical biomarker  
ejection_fraction         █████████             16.2%   ← key clinical biomarker
creatinine_phosphokinase  ████                   7.6%
serum_sodium              ███                    6.5%
platelets                 ███                    6.1%
age                       ██                     5.4%
anaemia                   █                      1.9%
high_blood_pressure       █                      1.8%
smoking                   █                      1.6%
sex                       █                      1.4%
diabetes                  █                      1.2%
```

**Key observations:**
- `time` (follow-up period) dominates at 33% — patients with longer follow-up were more likely to survive, reflecting both survivorship and continuity of care
- `serum_creatinine` and `ejection_fraction` are the most **clinically actionable** predictors, consistent with peer-reviewed literature (Chicco & Jurman, 2020)
- Binary lifestyle features (`diabetes`, `smoking`, `sex`) contribute minimally compared to organ function markers

---

## 📉 Limitations

- Dataset contains only **299 records** — small sample size limits generalizability
- High cross-validated recall variance (±33.89%) indicates model instability across folds
- `time` (follow-up period) may introduce **survivorship bias** in real-world deployment
- Limited clinical features compared to real-world EHR systems
- No external validation set

---

## 🚀 Future Improvements

- [ ] Model deployment using **Flask** or **Streamlit**
- [ ] Feature selection to remove low-importance variables (`diabetes`, `sex`, `smoking`)
- [ ] Test on larger, multi-centre clinical datasets
- [ ] Experiment with **LightGBM** and **CatBoost**
- [ ] SHAP values for explainable AI (XAI)
- [ ] Remove `time` feature and re-evaluate to reduce survivorship bias
- [ ] Build a patient risk scoring dashboard

---

## 📚 Key Learning Outcomes

- Exploratory Data Analysis on clinical data
- Handling class imbalance with SMOTE
- Feature scaling with StandardScaler
- Training and comparing 6 classification algorithms
- Hyperparameter tuning with GridSearchCV
- Evaluating models with Accuracy, Precision, Recall, F1, and ROC-AUC
- Understanding the cost of false negatives in healthcare ML
- Feature importance interpretation

---

## 👨‍💻 Author

**Harish Rathwa**

Machine Learning Project — Heart Failure Clinical Prediction

---


---

---
⭐ If you found this project useful, consider giving it a star on GitHub!</sub>
