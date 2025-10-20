# 🍷 Wine Quality Prediction using Machine Learning

This project aims to predict the quality of wine based on its physicochemical properties using a variety of machine learning algorithms. The models are built and evaluated using ensemble techniques, including KNN, SVM, Logistic Regression, Random Forest Classification, Gradient Boosting, XGBoost, and Stacking.

The trained model is used to predict the quality of wine on new data and output the results to a CSV file.

## 📌 Overview

Wine quality prediction typically involves assessing wines based on features such as acidity, alcohol content, and sulfur levels. This project implements various machine learning techniques to classify wines into three quality categories: Low, Medium, and High.

### Key Steps:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature scaling and selection
- Model training using ensemble techniques
- Evaluation with classification metrics
- Predicting wine quality on new data

---

## 📊 Dataset

The dataset used for training is provided in the `data` folder. It contains various physicochemical properties of red and white wines and their corresponding quality scores.

### Features:
1. **Type**: Red and White wine
2. **Fixed acidity**: Concentration of non-volatile acids (like tartaric acid) that do not evaporate easily and contribute to wine's taste.
3. **Volatile acidity**: Acetic acid content that can evaporate and may cause an unpleasant vinegar taste if too high.
4. **Citric acid**: A small amount of citric acid can add a refreshing flavor to wines.
5. **Residual sugar**: Sugar left after fermentation; higher values indicate sweeter wine.
6. **Chlorides**: Salt content in wine, primarily sodium chloride, which affects taste.
7. **Free sulfur dioxide**: Portion of SO2 (not chemically bound) available to prevent spoilage and oxidation.
8. **Total sulfur dioxide**: Sum of free and bound sulfur dioxide.
9. **Density**: The wine's mass-to-volume ratio, influenced by alcohol and sugar content.
10. **pH**: A measure of the wine's acidity or alkalinity; lower pH = more acidic.
11. **Sulphates**: Potassium sulphate content, which acts as a preservative and contributes to flavor.
12. **Alcohol**: Percentage of ethanol in the wine, influencing body, aroma, and warmth.
13. **Quality**: Sensory score (0-10) given by tasters to indicate overall wine quality.
14. **Quality binned** (created during preprocessing): 
   - 0–3: Low
   - 4–7: Medium
   - 8–10: High

> **Note**: The **Quality binned** column is not in the raw dataset but will be created during preprocessing.

---

## ⚙️ Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature scaling and selection
- Model training using ensemble techniques:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest Classification
  - Gradient Boosting
  - XGBoost
  - Stacking (Only With Ensemble Techniques)
- Evaluation using:
  - Classification Report (Precision, Recall, F1-Score, Support, Accuracy, Macro Avg, Weighted Avg)
  - Confusion Matrix
  - K-Fold Cross Validation

---
