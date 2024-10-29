# Sentiment Analysis of Financial News Headlines*

Objective: Analyze sentiment in financial news headlines using traditional and machine learning-based NLP techniques. The project aims to categorize news sentiment as positive, negative, or neutral and visualize sentiment trends, providing insights for investors and financial institutions.

Key Technologies Used
Python Libraries:

pandas
re
textblob
matplotlib
seaborn
wordcloud
vaderSentiment
sklearn
networkx
NLP Techniques:

### Sentiment analysis
Topic modeling (Latent Dirichlet Allocation - LDA)
Text vectorization
Project Sections
Data Source:

Dataset from Kaggle: "Sentiment Analysis for Financial News"
Processed within the environment using kagglehub.
Data Cleaning and Preprocessing:

Removal of URLs, mentions, and unnecessary characters.
Converted text to lowercase.
Sentiment Scoring:

TextBlob: Used for polarity scoring to categorize headlines.
VADER: Provided additional sentiment insights tailored for short texts.
Sentiment Categorization:

Defined categories based on polarity scores.

### Visualization:

Sentiment Distribution: Histogram and bar plots for sentiment scores and categories.
Word Clouds: Visual representation of common words in positive and negative sentiment.
Topic Modeling with LDA:

Identified prevalent themes in headlines and displayed top keywords for each topic.
Word Association Graphs:

Analyzed relationships between the most common words in sentiment categories.
Insights and Applications:

Trends in financial news sentiment.
Key themes identified through topic modeling.
Actionable insights for investment decision-making.

### Conclusion
The project successfully illustrates how sentiment analysis can be conducted using various NLP techniques and visualizations, providing valuable insights into financial news sentiment and its implications for market analysis.
