# Sentiment Analysis for Financial News

## Objective
The aim of this project is to analyze sentiment in financial news headlines, applying both traditional and machine learning-based Natural Language Processing (NLP) techniques to gain insights into market sentiment. The project categorizes news sentiment as positive, negative, or neutral and visualizes sentiment trends across financial news, which can be useful for investors, analysts, or financial institutions to inform their strategies.

## Data Source
The dataset is sourced from Kaggle's [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news), which includes numerous financial news headlines labeled with sentiment. This dataset is downloaded and processed within the environment using `kagglehub`.

## Key Technologies Used
- **Python Libraries**: pandas, re, textblob, matplotlib, seaborn, wordcloud, vaderSentiment, sklearn, and networkx.
- **NLP Techniques**: Sentiment analysis, topic modeling with LDA, and text vectorization.

## Insights and Applications
- **Sentiment Insights**: The analysis provides a breakdown of sentiment across financial headlines, useful for tracking market mood and potential impacts on stock performance.
- **Topic Trends**: Topic modeling reveals recurring themes, helping stakeholders identify the most talked-about sectors or events.
- **Investment Decision Support**: By understanding sentiment and theme trends, investors can gauge overall market sentiment and make more informed financial decisions.

## Data Cleaning and Preprocessing
The data is cleansed by:
- Removing unnecessary characters, URLs, and mentions.
- Converting all text to lowercase to maintain consistency.

### Sentiment Scoring
Sentiment scoring is carried out in two ways:
- **TextBlob** for polarity scoring, categorizing headlines as positive, negative, or neutral based on polarity score.
- **VADER Sentiment Analysis** for additional sentiment insights, specifically tailored for short texts like news headlines.

### Topic Modeling with LDA
Topic modeling using Latent Dirichlet Allocation (LDA) is performed to uncover the main themes present in the dataset.

## Visualizations
- **Sentiment Distribution**: Visualized through histograms to show overall sentiment trends in the dataset.
- **Word Clouds**: Positive and negative word clouds are generated to reveal common words within each sentiment category.
- **Word Associations**: Network graphs visualize word associations in positive and negative sentiments to help understand keyword relationships within each sentiment category.

## Installation
To run this project, you will need Python 3.x and the following libraries installed:

```bash
pip install pandas textblob matplotlib seaborn wordcloud vaderSentiment scikit-learn networkx kagglehub
