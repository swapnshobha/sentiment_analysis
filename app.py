import nltk
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st
import pandas as pd
import re
import contractions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk import FreqDist



# File upload
st.subheader("Upload CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display basic information about the DataFrame
    st.subheader("Basic Information about the Data")
    st.write("Number of rows and columns:", data.shape)
    st.write("Columns:", data.columns)

# Set configurations
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Title and header
st.title("Sentiment Analysis Dashboard")
st.header("Exploring and Visualizing Sentiment Analysis")

# Display basic information about the DataFrame
st.subheader("Basic Information about the Data")
st.write("Number of rows and columns:", data.shape)
st.write("Columns:", data.columns)

# Display summary statistics for numerical columns
st.subheader("Summary Statistics for Numerical Columns")
st.write(data.describe())

# Check for missing values
st.subheader("Missing Values")
st.write(data.isnull().sum())

# Display rating distribution
st.subheader("Rating Distribution")
rating_counts = data['reviews.rating'].value_counts().sort_index()
st.bar_chart(rating_counts)

# Display sentiment distribution
st.subheader("Sentiment Distribution")
sentiment_distribution = data['sentiment_category'].value_counts()
st.pie_chart(sentiment_distribution)

# Display top 30 common words
st.subheader("Top 30 Most Common Words")
all_words = ' '.join(data['cleaned_reviews']).split()
word_freq = FreqDist(all_words)
st.bar_chart(word_freq.most_common(30))

# Display average sentiment by user ratings
st.subheader("Average Sentiment by User Ratings")
avg_sentiment_by_rating = data.groupby('reviews.rating')['compound'].mean()
st.line_chart(avg_sentiment_by_rating)

# Display sentiment by category
st.subheader("Sentiment by Category")
category_sentiment = data.groupby(['Category', 'sentiment_category']).size().unstack()
st.bar_chart(category_sentiment)

# Display the original DataFrame
st.subheader("Original DataFrame")
st.write(data)

# End with a footer
st.markdown("---")
st.write("Created with ❤️ by Your Name")

# Run the app
if __name__ == "__main__":
    st.run()
