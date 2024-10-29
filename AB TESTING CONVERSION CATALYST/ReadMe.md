# A/B Testing Dataset Analysis

## Overview

This repository contains an analysis of an A/B testing dataset to evaluate the effectiveness of a marketing campaign. The analysis focuses on comparing the performance metrics of a test group and a control group, assessing key indicators such as conversion rates, average order value, and overall impact on purchases.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Statistical Analysis](#statistical-analysis)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Kaggle Hub

## Dataset

The dataset used in this analysis can be found in the `data` directory, which includes the following files:

- `test_group.csv`: Data for the test group.
- `control_group.csv`: Data for the control group.

## Exploratory Data Analysis

The exploratory data analysis section includes:

- Data loading and cleaning.
- Descriptive statistics for both test and control groups.
- Visualization of key performance metrics such as spend, impressions, clicks, and purchases.

## Statistical Analysis

A two-sample t-test was conducted to evaluate:

- Conversion rates between the test and control groups.
- The number of purchases.
- The average order value.

The results include t-statistics and p-values to determine the statistical significance of the findings.

## Key Findings

- The test group showed a higher average conversion rate compared to the control group.
- There was no significant difference in the number of purchases between the two groups.
- The average order value did not significantly differ between groups.

## Installation

To run this analysis, clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/ab-testing-dataset-analysis.git
cd ab-testing-dataset-analysis
pip install -r requirements.txt
