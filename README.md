# 🌐 Social Media Sentiment Analyzer
## 🧠 A Streamlit-based Web App for Sentiment Analysis using Machine Learning

**Overview**

The Social Media Sentiment Analyzer is an interactive Streamlit web application that performs sentiment analysis on social media text data.
It helps visualize, explore, and analyze the sentiments expressed in text — such as positive, neutral, or negative emotions — through a combination of data visualization, text analytics, and machine learning.

This project provides:

Insightful EDA (Exploratory Data Analysis) on text sentiments.

A search-based vocabulary analyzer with word clouds and charts.

A machine learning-based sentiment predictor using TF-IDF and SVM.

A clean, responsive Streamlit dashboard with custom CSS design.

**Features**

Displays dataset preview.

Shows sentiment distribution and frequency.

Provides warnings for data quality (missing or duplicate entries).

**Insights**

Interactive vocabulary-based exploration.

Sentiment distribution via Pie Chart.

Top 10 most frequent words visualization.

Word Cloud generation.

Dynamic text filtering and exploration through sidebar controls.

**Sentiment Analyzer**

Users can input any English text.

Uses TF-IDF Vectorization + Support Vector Machine (SVM).

Includes GridSearchCV for hyperparameter tuning.

Displays the predicted sentiment for user input.

Animated progress feedback with Streamlit st.status.

**Tech Stack**
| Category             | Technologies Used                        |
| -------------------- | ---------------------------------------- |
| **Frontend**         | Streamlit, HTML, CSS                     |
| **Data Handling**    | Pandas, NumPy                            |
| **Visualization**    | Seaborn, Matplotlib, WordCloud           |
| **Machine Learning** | scikit-learn (SVM, GridSearchCV, TF-IDF) |
| **Utilities**        | Logging, Regex, Collections              |
| **Images/Display**   | Pillow (PIL)                             |

**Requirements**
streamlit
pandas
numpy
matplotlib
seaborn
Pillow
wordcloud
scikit-learn

**Acknowledgments**

. Dataset inspired by open-source sentiment analysis datasets.

. Streamlit and scikit-learn community documentation.

