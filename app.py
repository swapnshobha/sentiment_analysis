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

# Set configurations
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Title and header
st.title("Sentiment Analysis Dashboard")
st.header("Exploring and Visualizing Sentiment Analysis")

# File upload
st.subheader("Upload CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

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
    st.write(rating_counts)

    st.subheader("Sentiment Distribution")
    sentiment_distribution = data['sentiment_category'].value_counts()
    st.write(sentiment_distribution)


    import plotly.express as px


    # Create a pie chart using Plotly
    fig = px.pie(names=sentiment_distribution.index,values=sentiment_distribution.values,title='Sentiment Distribution')

    # Display the pie chart using Streamlit
    st.plotly_chart(fig)


    # Display top 30 common words
    st.subheader("Top 30 Most Common Words")
    all_words = ' '.join(data['cleaned_reviews']).split()
    word_freq = Counter(all_words)

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # Display the word cloud using Matplotlib and Streamlit
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Display top 30 common words
    st.subheader("Top 30 Most Common Words")
    all_words = ' '.join(data['cleaned_reviews']).split()
    word_freq = FreqDist(all_words)
    st.bar_chart(word_freq.most_common(30))

    # Display the original DataFrame
    st.subheader("Original DataFrame")
    st.write(data)

else:
    st.warning("Please upload a CSV file.")

# End with a footer
st.markdown("---")
st.write("Created with ❤️ by Swarup Shinde")
