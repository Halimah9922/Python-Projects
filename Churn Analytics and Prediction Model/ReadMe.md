# Customer Churn Analysis Project

## Overview
This project aims to analyze customer churn for a telecommunications company. Customer churn refers to the phenomenon where customers stop doing business with a company, which can significantly impact revenue and growth. The dataset used in this analysis contains customer information and their churn status, providing insights into patterns and factors that influence customer retention.

## Dataset
The dataset utilized for this analysis is the **"Telco Customer Churn"** dataset, which was downloaded from Kaggle. It includes various customer attributes such as demographics, account information, and service usage.

**Dataset Link**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Steps
1. **Data Loading**: The dataset is loaded from the specified path after downloading.
2. **Data Exploration**: Initial inspection of the dataset is performed to understand its structure and identify missing values.
3. **Data Preprocessing**:
   - Convert the 'TotalCharges' column to numeric and handle any missing values.
   - Encode categorical variables, including converting the 'Churn' column to binary.
   - Create dummy variables for other categorical features and drop irrelevant columns.
4. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of churned vs. non-churned customers.
   - Generate a correlation heatmap to identify relationships between features and churn.
5. **Modeling**:
   - Split the dataset into training and testing sets.
   - Train a Random Forest classifier to predict customer churn and analyze feature importance.
   - Use Logistic Regression as a baseline model for comparison.
6. **Clustering**:
   - Perform K-Means clustering to segment customers based on selected features.
   - Analyze the mean churn rates for different clusters to identify trends.
7. **Survival Analysis**:
   - Conduct survival analysis using the Kaplan-Meier estimator to understand customer retention over time.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- lifelines
- kagglehub

You can install the required packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lifelines kagglehub
