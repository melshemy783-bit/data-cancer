# 🩺 Breast Cancer Classification

## 📌 Project Overview

This project focuses on predicting whether a breast tumor is **Benign** or **Malignant** using Machine Learning classification algorithms.

The dataset was preprocessed, explored, and used to train multiple classification models. Their performance was then compared using Accuracy.

---

## 📂 Dataset

**Dataset:** Breast Cancer Wisconsin Dataset

### Target Variable

- **M (Malignant)** → Cancerous Tumor
- **B (Benign)** → Non-Cancerous Tumor

The dataset contains **30 numerical features** extracted from digitized images of breast mass cell nuclei.

Examples of features:

- radius_mean
- texture_mean
- perimeter_mean
- area_mean
- smoothness_mean
- compactness_mean
- concavity_mean
- symmetry_mean
- fractal_dimension_mean

---

## 🛠️ Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset
- Removed unnecessary columns (`id`, `Unnamed: 32`)
- Encoded the target column
- Checked for missing values
- Checked for duplicate records
- Split the data into training and testing sets
- Applied StandardScaler (for models that require feature scaling)

---

## 🤖 Machine Learning Models

The following classification algorithms were implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- XGBoost

---

## 📊 Model Evaluation

Each model was evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

The models were also compared using a summary table of their accuracy scores.

---

## 📈 Visualization

The project includes:

- Accuracy Comparison Chart
- Confusion Matrix Heatmap
- Feature Importance (XGBoost)

---

## 🔍 Inference

A new patient's medical measurements can be entered to predict whether the tumor is:

- ✅ Benign
- ❌ Malignant

The model can also provide prediction probabilities.

---

## 📚 Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

---

## 🚀 Project Workflow

1. Load Dataset
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Train-Test Split
6. Feature Scaling
7. Train Machine Learning Models
8. Evaluate Models
9. Compare Performance
10. Perform Inference on New Data

---

## 📌 Results

The project compares six different Machine Learning algorithms to determine which model achieves the highest classification accuracy for breast cancer prediction.

---


Machine Learning Project
