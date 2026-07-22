<div align="center">

# 🌐 Social Media Sentiment Analyzer

### 🧠 A Streamlit-based Web App for Sentiment Analysis using Machine Learning & NLP

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)

**🔗 [Live Demo — Check out the release »](#)**

</div>

---

## 📖 Overview

The **Social Media Sentiment Analyzer** is an interactive Streamlit web application that performs sentiment analysis on social media text data. It helps visualize, explore, and analyze the sentiments expressed in text — **positive**, **neutral**, or **negative** — through a combination of data visualization, text analytics, rule-based NLP, and machine learning.

This project provides:

- 📊 Insightful **EDA (Exploratory Data Analysis)** on text sentiments
- 🔍 A **search-based vocabulary analyzer** with word clouds and charts
- 🤖 A **machine learning-based sentiment predictor** using TF-IDF and Logistic Regression
- ⚡ A **VADER-powered rule-based sentiment engine** for fast, lexicon-driven analysis
- 🎨 A clean, responsive Streamlit dashboard with **custom CSS design**

---

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 📂 Data Overview
- Dataset preview
- Sentiment distribution & frequency
- Data quality warnings (missing/duplicate entries)

### 📈 Insights & EDA
- Interactive vocabulary-based exploration
- Sentiment distribution via Pie Chart
- Top 10 most frequent words visualization
- Word Cloud generation
- Dynamic text filtering via sidebar controls

</td>
<td width="50%" valign="top">

### 🤖 ML Sentiment Analyzer
- Custom English text input
- TF-IDF Vectorization + Logistic Regression
- GridSearchCV for hyperparameter tuning
- Animated progress feedback via `st.status`

### ⚡ VADER Sentiment Engine
- Lexicon & rule-based sentiment scoring
- Instant polarity scores (positive/negative/neutral/compound)
- Great for short, informal social media text
- Runs alongside ML model for **comparative sentiment analysis**

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

| Category               | Technologies Used                                  |
| ----------------------- | --------------------------------------------------- |
| **Frontend**            | Streamlit, HTML, CSS                                 |
| **Data Handling**       | Pandas, NumPy                                        |
| **Visualization**       | Seaborn, Matplotlib, WordCloud                       |
| **Machine Learning**    | scikit-learn (Logistic Regression, GridSearchCV, TF-IDF)             |
| **NLP / Sentiment**     | **VADER (vaderSentiment)**, scikit-learn             |
| **Utilities**           | Logging, Regex, Collections                          |
| **Containerization**    | Docker                                               |
| **CI/CD**               | GitHub Actions (automated build & smoke testing)     |

---

## 🧩 Why VADER + ML?

> **VADER** (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically tuned for social media text — handling slang, emojis, punctuation emphasis (e.g. `"good!!!"`), and capitalization cues out of the box.

Combining VADER with a trained **TF-IDF + LogR** model gives this app two complementary perspectives:

| Approach | Strength |
|---|---|
| ⚡ **VADER** | Instant, no training needed, great for short/informal text |
| 🤖 **ML (TF-IDF + Logistic Regression)** | Learns patterns from your dataset, adapts to domain-specific language |

---

## 📦 Requirements

```
streamlit
pandas
numpy
matplotlib
seaborn
Pillow
wordcloud
scikit-learn
vaderSentiment
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Bilall2003/social-media-sentimental-Analysis.git
cd social-media-sentimental-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run script/main.py
```

### 🐳 Run with Docker

```bash
docker build -t sentiment-app .
docker run -p 8501:8501 sentiment-app
```

---

## ⚙️ CI/CD Pipeline

This project uses **GitHub Actions** for continuous integration:

- ✅ Automated Docker image build on every push to `main`
- ✅ Smoke testing to verify the app starts successfully before it's considered a passing build

---

## 🙏 Acknowledgments

- Dataset inspired by open-source sentiment analysis datasets
- [Streamlit](https://streamlit.io/) & [scikit-learn](https://scikit-learn.org/) community documentation
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment) by C.J. Hutto

---

<div align="center">

Made by **[Bilal Ahmed](https://github.com/Bilall2003)**

</div>
