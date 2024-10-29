# Credit Card Fraud Detection

## Project Description
This project aims to build a machine learning model that can accurately detect fraudulent transactions in credit card data. The dataset consists of anonymized transaction records, and the primary goal is to identify fraudulent transactions while minimizing false positives.

## Objectives
- Preprocess the credit card transaction dataset by handling missing values and class imbalances.
- Visualize key data distributions to understand patterns in transactions.
- Train and evaluate multiple machine learning models for fraud detection.
- Select the best-performing model based on relevant evaluation metrics.

## Dataset
- **Source**: Kaggle (Nelgiriye Withana - Credit Card Fraud Detection Dataset 2023)
- **Link**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
- **Features**:
  - `Class`: Target variable (0 = Non-fraudulent, 1 = Fraudulent)
  - `Amount`: Transaction amount
  - `V1` to `V28`: Anonymized features derived from PCA to protect sensitive information
- **Size**: The dataset contains thousands of records, with a significant class imbalance between fraudulent and non-fraudulent transactions.

## Methodology
1. **Data Acquisition**:
   - Download the dataset using the Kaggle API and load it into a Pandas DataFrame.

2. **Exploratory Data Analysis (EDA)**:
   - Check for missing values and duplicates.
   - Visualize class distribution and transaction amounts to identify patterns and imbalances.
   - Analyze PCA components to understand feature distributions.

3. **Data Preprocessing**:
   - Handle class imbalance using SMOTE to oversample the minority class (fraudulent transactions).
   - Scale features using StandardScaler to ensure uniform contribution in model training.

4. **Model Training and Evaluation**:
   - Split the dataset into training and testing sets.
   - Train multiple models: Logistic Regression, Random Forest, and Gradient Boosting.
   - Evaluate each model using confusion matrices and classification reports, focusing on precision, recall, and F1-score.

5. **Model Comparison**:
   - Compare model performance based on evaluation metrics and select the best-performing model for deployment.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**:
  - Pandas for data manipulation
  - Seaborn and Matplotlib for data visualization
  - Scikit-learn for machine learning
  - Imbalanced-learn for handling class imbalance with SMOTE

## Installation
To run this project, you need to install the required libraries. You can do this using pip:

```bash
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn
