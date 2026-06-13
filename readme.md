# ❤️ Heart Failure Prediction Using Machine Learning

## 📌 Project Overview

Heart failure is a serious medical condition that occurs when the heart cannot pump blood effectively. Early prediction of high-risk patients can help healthcare professionals provide timely treatment and improve patient outcomes.

This project applies Machine Learning algorithms to predict whether a patient is likely to survive or die based on clinical health records.

---

## 🎯 Objectives

* Perform Exploratory Data Analysis (EDA) on heart failure clinical data.
* Identify relationships between clinical features and patient outcomes.
* Build Machine Learning classification models.
* Evaluate model performance using multiple metrics.
* Compare different algorithms and select the best-performing model.

---

## 📊 Dataset Information

The project uses the **Heart Failure Clinical Records Dataset**.

### Dataset Statistics

* Total Records: **299**
* Features: **12 Input Features**
* Target Variable: **DEATH_EVENT**

### Features

| Feature                  | Description                                               |
| ------------------------ | --------------------------------------------------------- |
| age                      | Age of patient                                            |
| anaemia                  | Decrease of red blood cells or hemoglobin                 |
| creatinine_phosphokinase | Level of CPK enzyme in blood                              |
| diabetes                 | Whether the patient has diabetes                          |
| ejection_fraction        | Percentage of blood leaving the heart at each contraction |
| high_blood_pressure      | Whether the patient has hypertension                      |
| platelets                | Platelet count in blood                                   |
| serum_creatinine         | Level of serum creatinine                                 |
| serum_sodium             | Level of serum sodium                                     |
| sex                      | Gender of patient                                         |
| smoking                  | Whether the patient smokes                                |
| time                     | Follow-up period                                          |
| DEATH_EVENT              | Target Variable (0 = Survived, 1 = Died)                  |

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 📈 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Dataset Information
* Statistical Summary
* Missing Value Analysis
* Correlation Matrix
* Heatmap Visualization
* Feature Distribution Analysis
* Feature vs Target Analysis

### Correlation Analysis

A correlation matrix was generated to identify relationships among variables and understand feature importance.

---

## ⚙️ Data Preprocessing

### Steps Performed

1. Loaded dataset using Pandas
2. Checked for missing values
3. Separated features and target variable
4. Split dataset into training and testing sets
5. Applied StandardScaler for feature scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 🤖 Machine Learning Models

The following classification algorithms were implemented:

### 1. Logistic Regression

Used as a baseline classification model for predicting patient outcomes.

### 2. Support Vector Machine (SVM)

Used for finding the optimal decision boundary between classes.

### 3. K-Nearest Neighbors (KNN)

Classifies patients based on the nearest neighboring samples.

---

## 📋 Evaluation Metrics

The models were evaluated using:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Confusion Matrix

### Metric Definitions

**Accuracy**

* Percentage of correctly classified samples.

**Precision**

* Measures how many predicted positive cases are actually positive.

**Recall**

* Measures how many actual positive cases are correctly identified.

**F1 Score**

* Harmonic mean of Precision and Recall.

---

# 📊 Model Results

## Logistic Regression

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 78.89% |
| Precision | 67.86% |
| Recall    | 65.52% |
| F1 Score  | 66.67% |

---

## Support Vector Machine (SVM)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 78.89% |
| Precision | 67.86% |
| Recall    | 65.52% |
| F1 Score  | 66.67% |

---

## K-Nearest Neighbors (KNN)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 70.00% |
| Precision | 57.14% |
| Recall    | 27.59% |
| F1 Score  | 37.21% |

---

## 📌 Model Comparison

| Model                  | Accuracy | Precision | Recall | F1 Score |
| ---------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression    |   78.89% |    67.86% | 65.52% |   66.67% |
| Support Vector Machine |   78.89% |    67.86% | 65.52% |   66.67% |
| K-Nearest Neighbors    |   70.00% |    57.14% | 27.59% |   37.21% |

---

## 🏆 Best Performing Model

Based on the evaluation metrics:

* Logistic Regression performed best.
* Support Vector Machine achieved identical performance.
* KNN showed lower performance, especially in Recall and F1 Score.

Logistic Regression was selected as the final model because of its simplicity, interpretability, and competitive performance.

---

## 📉 Limitations

* Dataset contains only 299 records.
* Performance may vary with different train-test splits.
* Recall can be improved to identify more high-risk patients.
* Limited clinical features compared to real-world healthcare systems.

---

## 🚀 Future Improvements

* Hyperparameter Tuning using GridSearchCV
* Cross Validation
* Random Forest Classifier
* XGBoost Classifier
* Feature Selection Techniques
* Model Deployment using Flask or Streamlit
* Larger Clinical Datasets

---

## 📂 Project Structure

```text
Heart-Failure-Prediction/
│
├── dataset/
│   └── heart_failure_clinical_records.csv
│
├── notebooks/
│   └── Heart_Failure_Prediction.ipynb
│
├── images/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone  https://github.com/Harish115-dev/hear-failure-prediction.git
```

Move into the project directory:

```bash
cd Heart-Failure-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and execute all cells.

---

## 📚 Key Learning Outcomes

* Data Preprocessing
* Feature Scaling
* Exploratory Data Analysis
* Classification Algorithms
* Model Evaluation
* Confusion Matrix Interpretation
* Performance Comparison

---

## 👨‍💻 Author

**Harish Rathwa**

Machine Learning Project – Heart Failure Prediction

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
