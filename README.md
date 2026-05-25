# ShopSmart E-commerce Conversion Prediction

## 📌 Project Overview
ShopSmart, an e-commerce company, seeks to better understand visitor browsing behavior to optimize marketing strategies and capture lost revenue. This project builds a predictive machine learning model to determine whether a website visitor is likely to complete a purchase based on their session data. 

**Goal:** Achieve an F1 Score greater than 0.55 on the imbalanced target variable (`Revenue`) using a Decision Tree-based classifier.
**Result:** Achieved an optimal F1 Score of **0.634** through hyperparameter tuning and class balancing.

## 📊 Dataset Description
The dataset contains **12,330 individual user sessions** collected over a one-year period. 
* **Features:** 17 categorical and numerical features capturing page view durations, bounce rates, exit rates, visitor types, and timing (e.g., Month, Weekend, Special Day).
* **Target Variable:** `Revenue` (Boolean) - Indicates whether the visitor made a purchase.

## ⚙️ Methodology & Machine Learning Pipeline
1. **Exploratory Data Analysis (EDA):** Analyzed feature distributions and the significant class imbalance in the target variable.
2. **Preprocessing:** * Encoded boolean features (`Weekend`, `Revenue`) to binary.
   * Applied One-Hot Encoding to categorical features (`Month`, `VisitorType`, `OperatingSystems`, etc.).
3. **Modeling:** Implemented a Scikit-Learn `Pipeline` using a `DecisionTreeClassifier`. Addressed the class imbalance by applying `class_weight='balanced'`.
4. **Hyperparameter Tuning:** Prevented overfitting and applied tree pruning via `GridSearchCV`. Tuned parameters using 5-fold cross-validation.

## 🚀 Results
The optimal model parameters found via GridSearch were:
* `max_depth`: 4
* `min_samples_leaf`: 50

**Performance Metrics:**
* **Cross-Validated F1 Score:** 0.634
* The model successfully surpassed the project benchmark of 0.55, providing a robust, generalized solution that accurately predicts the minority class (purchasers) without overfitting.

## 🛠️ How to Run the Project
1. Clone the repository:
   ```bash
   git clone [https://github.com/LearnerNikhilk/ShopSmart-E-commerce-Project.git](https://github.com/LearnerNikhilk/ShopSmart-E-commerce-Project.git)
