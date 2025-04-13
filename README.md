# Titanic-Survival-Prediction
## Overview
This project tackles the Kaggle “Titanic: Machine Learning from Disaster” challenge. Using passenger data (age, sex, class, family relationships, etc.), the goal is to build a model that predicts which passengers survived.

---

## Background
The RMS Titanic sank on April 15, 1912, after striking an iceberg, resulting in over 1,500 fatalities. Kaggle’s Titanic competition provides a historical dataset of 891 training and 418 test passengers with features such as:
- **PassengerId, Name, Sex, Age**  
- **Pclass (ticket class), SibSp & Parch (family aboard)**  
- **Ticket, Fare, Cabin, Embarked (port of boarding)**  

Understanding which factors (e.g., gender, class, family size) influenced survival can reveal both social patterns of the era and best practices in modern feature engineering and classification.
---

## Model Implementation

This solution for the Titanic classification task leverages a modular machine learning pipeline, built using Scikit-learn, to automate preprocessing, training, and evaluation of several classifiers. Below is a breakdown of the full implementation process:

### 1. Data Splitting
- The train dataset is split using **StratifiedShuffleSplit** to ensure proportional class distribution.

### 2. Data Preprocessing Pipeline

Custom transformer classes are defined to handle preprocessing, and all steps are combined using `Pipeline` for consistency and reusability.

- **`AgeImputer`**: Fills missing `Age` values using the mean strategy.
- **`FeatureEncoder`**: Applies one-hot encoding to `Sex` and `Embarked` features using `OneHotEncoder`.
- **`FeatureDropper`**: Removes unnecessary or high-cardinality columns like `Name`, `Ticket`, `Cabin`, and encoded originals.


###  3. Feature Scaling
Numerical features are standardized using `StandardScaler` after preprocessing.

### 4. Model Training & Selection

Five popular machine learning classifiers were tested:
- **Random Forest**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting**

Each model underwent **Grid Search with 5-fold Cross-Validation** using `GridSearchCV`. Hyperparameters were tuned to maximize accuracy, and the best-performing model on the validation set was automatically selected.

### 5. Prediction on Test Set
- The test dataset undergoes the same pipeline transformation and scaling.
- Missing values (if any) are handled using forward fill.
- Predictions are generated using the best model from the training phase.

### 6. Submission
The output predictions are saved to `data/prediction.csv` in the required format:

| PassengerId | Survived |
|-------------|----------|
| 892         | 0        |
| 893         | 1        |
| ...         | ...      |

---
## Now Ready-to-submit output for Kaggle



