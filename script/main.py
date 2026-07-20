import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os
import logging
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- Streamlit page config ---

st.set_page_config(
    page_title="Social Media Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# --- Enhanced CSS ---
class CSS:
    def css(self):
        st.markdown("""
            <style>
            /* Import Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800;900&display=swap');
            
            * {
                font-family: 'Inter', sans-serif;
            }
            
            /* Main App Background */
            .stApp {
                background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
                background-attachment: fixed;
            }
            
            
            /* Gradient Title */
            .main-title {
                font-size: 56px;
                font-weight: 900;
                text-align: center;
                
                background: linear-gradient(
                    90deg,
                    #667eea,
                    #764ba2,
                    #f093fb
                );

                -webkit-background-clip: text;
                background-clip: text;

                -webkit-text-fill-color: transparent;

                display: inline-block;
                width: 100%;

                margin-top: 20px;
                margin-bottom: 30px;
            }
            
            @keyframes slideDown {
                from {
                    opacity: 0;
                    transform: translateY(-30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            /* Section Headers */
            .section-header {
                font-size: 32px;
                font-weight: 800;
                background: linear-gradient(90deg, #ff6ec4, #ffd700);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 30px 0 20px 0;
                filter: drop-shadow(0 0 10px rgba(255, 110, 196, 0.4));
            }
            
            /* Info Card */
            .info-card {
                background: linear-gradient(135deg, rgba(106, 17, 203, 0.2) 0%, rgba(37, 117, 252, 0.2) 100%);
                border-radius: 20px;
                padding: 30px;
                margin: 20px 0;
                border: 2px solid rgba(255, 110, 196, 0.3);
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(106, 17, 203, 0.3);
                animation: fadeIn 0.6s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .info-card p {
                color: #e0e0e0;
                font-size: 16px;
                line-height: 1.8;
            }
            
            /* Stats Cards */
            .stat-card {
                background: linear-gradient(135deg, rgba(255, 110, 196, 0.15) 0%, rgba(255, 215, 0, 0.15) 100%);
                border-radius: 18px;
                padding: 25px;
                text-align: center;
                border: 2px solid rgba(255, 110, 196, 0.4);
                transition: all 0.4s ease;
                margin: 10px 0;
            }
            
            .stat-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 15px 40px rgba(255, 110, 196, 0.4);
                border-color: rgba(255, 215, 0, 0.7);
            }
            
            .stat-number {
                font-size: 48px;
                font-weight: 900;
                background: linear-gradient(90deg, #00f2fe, #4facfe);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 10px 0;
            }
            
            .stat-label {
                font-size: 16px;
                color: #ffd700;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            /* Enhanced Buttons */
            .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: 2px solid rgba(255, 215, 0, 0.4) !important;
                border-radius: 15px !important;
                padding: 14px 40px !important;
                font-weight: 700 !important;
                font-size: 16px !important;
                transition: all 0.4s ease !important;
                box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
                width: 100% !important;
            }
            
            .stButton > button:hover {
                transform: translateY(-4px) scale(1.02) !important;
                box-shadow: 0 8px 30px rgba(255, 215, 0, 0.6) !important;
                border-color: rgba(255, 215, 0, 0.8) !important;
            }
            
            /* Text Area */
            .stTextArea > div > div > textarea {
                background: rgba(43, 0, 79, 0.4) !important;
                border: 2px solid rgba(255, 110, 196, 0.4) !important;
                border-radius: 15px !important;
                color: white !important;
                font-size: 16px !important;
                min-height: 150px !important;
                transition: all 0.3s ease !important;
            }
            
            .stTextArea > div > div > textarea:focus {
                border-color: rgba(255, 215, 0, 0.8) !important;
                box-shadow: 0 0 25px rgba(255, 215, 0, 0.4) !important;
            }
            
            /* Radio Buttons Navigation */
            div[role="radiogroup"] {
                display: flex;
                justify-content: center;
                gap: 20px;
                margin: 30px 0;
            }
            
            div[role="radiogroup"] label {
                background: linear-gradient(135deg, rgba(106, 17, 203, 0.3) 0%, rgba(37, 117, 252, 0.3) 100%);
                padding: 14px 30px;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 700;
                font-size: 16px;
                color: #e0e0e0;
                border: 2px solid rgba(255, 110, 196, 0.3);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
                user-select: none;
            }
            
            div[role="radiogroup"] label:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 25px rgba(255, 110, 196, 0.4);
                border-color: rgba(255, 215, 0, 0.6);
            }
            
            div[role="radiogroup"] label:has(input[type="radio"]:checked) {
                background: linear-gradient(135deg, #ff6ec4 0%, #ffd700 100%);
                color: #1a1a2e;
                border-color: rgba(255, 215, 0, 0.8);
                box-shadow: 0 8px 30px rgba(255, 215, 0, 0.5);
            }
            
            /* Result Cards */
            .result-card {
                background: linear-gradient(135deg, rgba(0, 242, 254, 0.15) 0%, rgba(79, 172, 254, 0.15) 100%);
                border-radius: 18px;
                padding: 25px;
                text-align: center;
                border: 2px solid rgba(0, 242, 254, 0.4);
                margin: 15px 0;
            }
            
            .result-value {
                font-size: 38px;
                font-weight: 900;
                background: linear-gradient(90deg, #ff6ec4, #ffd700);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 10px 0;
                animation: pulse 2s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
            
            .result-label {
                font-size: 14px;
                color: #00f2fe;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1.5px;
            }
            
            /* Dataframe Styling */
            .stDataFrame {
                border-radius: 15px;
                overflow: hidden;
                border: 2px solid rgba(255, 110, 196, 0.3);
            }
            
            /* Divider */
            hr {
                border: none;
                height: 3px;
                background: linear-gradient(90deg, #ff6ec4, #ffd700, #00f2fe);
                margin: 40px 0;
                border-radius: 5px;
            }
            
            /* Warning & Info Boxes */
            .stWarning, .stInfo {
                background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 152, 0, 0.15) 100%) !important;
                border-left: 4px solid #ffd700 !important;
                border-radius: 10px !important;
                color: #ffd700 !important;
            }
            
            /* Success Box */
            .stSuccess {
                background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(56, 142, 60, 0.15) 100%) !important;
                border-left: 4px solid #4caf50 !important;
                border-radius: 10px !important;
            }
            
            /* Sidebar Form Styling */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
                border-right: 3px solid rgba(255, 110, 196, 0.3);
            }
            
            .stSelectbox > div > div {
                background: rgba(106, 17, 203, 0.3) !important;
                border: 2px solid rgba(255, 110, 196, 0.4) !important;
                border-radius: 12px !important;
                color: white !important;
            }
            
            /* Slider */
            .stSlider > div > div > div {
                background: linear-gradient(90deg, #667eea, #764ba2) !important;
            }
            
            /* Tables */
            .stTable {
                background: rgba(106, 17, 203, 0.1);
                border-radius: 15px;
                overflow: hidden;
            }
            
            /* Caption */
            .stCaption {
                color: #a0a0a0 !important;
                font-size: 15px !important;
            }
            
            /* Emoji Styling */
            .emoji-large {
                font-size: 48px;
                text-align: center;
                margin: 10px 0;
                animation: bounce 2s ease-in-out infinite;
            }
            
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
            }
            
            </style>
        """, unsafe_allow_html=True)

# --- Info and EDA class ---
class info_insights(CSS):

    def load_data(self):
        file_path = "Data/train.csv"
        if os.path.exists(file_path):
            try:
                self.df = pd.read_csv(file_path, encoding="latin-1")
                self.df.dropna(inplace=True)
                self.df.drop_duplicates(keep="first", inplace=True)
                self.df.reset_index(drop=True, inplace=True)
                return self.df
            except Exception as e:
                logging.error(e)
                st.error(f"❌ Error loading data: {e}")
                return pd.DataFrame()
        else:
            logging.error("Invalid filepath.")
            st.error("❌ Dataset not found. Please check the file path.")
            return pd.DataFrame()

    def info(self):
        self.css()
        
        # Hero Section
        
        st.title("🎭 Social Media Sentiment Analyzer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='stat-card'>
                <div class='emoji-large'>📊</div>
                <div class='stat-number'>{:,}</div>
                <div class='stat-label'>Total Records</div>
            </div>
            """.format(len(self.df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='stat-card'>
                <div class='emoji-large'>🎯</div>
                <div class='stat-number'>{}</div>
                <div class='stat-label'>Sentiment Classes</div>
            </div>
            """.format(self.df["sentiment"].nunique()), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='stat-card'>
                <div class='emoji-large'>🤖</div>
                <div class='stat-number'>ML</div>
                <div class='stat-label'>Powered Analysis</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Info Card
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #ffd700; margin-bottom: 15px;'>ℹ️ About This Application</h3>
            <p>
                This advanced sentiment analysis tool uses <strong>Machine Learning</strong> to classify social media text into three emotional categories:
                <strong>Positive 😀</strong>, <strong>Neutral 😐</strong>, and <strong>Negative 😞</strong>.
            </p>
            <p>
                The model is trained on real-world social media data and achieves <strong>60-70% accuracy</strong>.
                Currently supports <strong>English language</strong> text only.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset Preview
        st.markdown("<h2 class='section-header'> Training Dataset Preview</h2>", unsafe_allow_html=True)
        
        columns = self.df[["text", "sentiment"]].head(100)
        st.dataframe(columns, use_container_width=True, height=400)
        
        # Sentiment Distribution
        st.markdown("<h2 class='section-header'>Sentiment Distribution</h2>", unsafe_allow_html=True)
        
        gr = self.df["sentiment"].value_counts().reset_index()
        gr.columns = ["sentiment", "Count"]

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#0F2027')
        ax.set_facecolor('#0F2027')
        
        colors = ['#667eea', '#764ba2', '#f093fb']
        bars = sns.barplot(x="sentiment", y="Count", data=gr, ax=ax, palette=colors)
        
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)
        
        ax.set_xlabel("Sentiment", color='white', fontsize=14, fontweight='bold')
        ax.set_ylabel("Count", color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white', labelsize=12)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)

    def eda(self):
        self.css()
        
        st.title("🔍 Deep Insights & Analysis")
        data = self.df[["text", "sentiment"]]
        X = data["text"]
        y = data["sentiment"]

        tf_idf = TfidfVectorizer(stop_words="english")
        x_vec = tf_idf.fit_transform(X)
        voc = tf_idf.vocabulary_

        # Sidebar Search Parameters
        with st.sidebar:
            st.markdown("<h2 style='color: #ffd700; text-align: center;'>🔎 Search Parameters</h2>", unsafe_allow_html=True)
            
            with st.form(key="search_form"):
                voc_sel = st.selectbox("🔤 Choose Vocabulary Word", (list(voc.keys())), key="voc_select")
                text_sel = st.slider("📝 Number of Texts to Display", min_value=5, max_value=500, value=50, key="num_tweets")
                submitted = st.form_submit_button("🚀 Analyze", use_container_width=True)
                st.info("💡 Large selections may take time to process...")

        if submitted:
            with st.spinner("🔄 Analyzing data... Please wait..."):
                filtered = data[data["text"].str.contains(voc_sel, case=False, na=False)][["text", "sentiment"]]
                sentiment_counts = filtered["sentiment"].value_counts()
                
                if not filtered.empty:
                    # Top Section - Pie Chart and Top Words
                    col1, col2 = st.columns([1, 1.5], gap="large")
                    
                    with col1:
                        st.markdown("<h2 class='section-header'>Sentiment Distribution</h2>", unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(8, 8))
                        fig.patch.set_facecolor('#0F2027')
                        
                        colors = ['#667eea', '#764ba2', '#f093fb']
                        explode = (0.05, 0.05, 0.05)
                        
                        wedges, texts, autotexts = ax.pie(
                            sentiment_counts, 
                            labels=sentiment_counts.index,
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=colors,
                            explode=explode,
                            shadow=True,
                            textprops={'color': 'white', 'weight': 'bold', 'fontsize': 12}
                        )
                        
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontsize(14)
                            autotext.set_weight('bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("<h2 class='section-header'>Top 10 Most Frequent Words</h2>", unsafe_allow_html=True)
                        
                        text = " ".join(filtered["text"].astype(str).tolist())
                        text = re.sub(r"[^\w\s]", "", text.lower())
                        words = text.split()
                        word_counts = Counter(words)
                        top_10 = word_counts.most_common(10)
                        df_10 = pd.DataFrame(top_10, columns=["Word", "Count"])
                        
                        fig1, ax1 = plt.subplots(figsize=(10, 6))
                        fig1.patch.set_facecolor('#0F2027')
                        ax1.set_facecolor('#0F2027')
                        
                        bars = sns.barplot(x="Word", y="Count", data=df_10, ax=ax1, palette="viridis")
                        
                        for bar in bars.patches:
                            height = bar.get_height()
                            ax1.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{int(height)}',
                                    ha='center', va='bottom', color='white', fontweight='bold')
                        
                        ax1.set_xlabel("Words", color='white', fontsize=12, fontweight='bold')
                        ax1.set_ylabel("Frequency", color='white', fontsize=12, fontweight='bold')
                        ax1.tick_params(colors='white', labelsize=10, rotation=45)
                        ax1.spines['bottom'].set_color('white')
                        ax1.spines['left'].set_color('white')
                        ax1.spines['top'].set_visible(False)
                        ax1.spines['right'].set_visible(False)
                        
                        plt.tight_layout()
                        st.pyplot(fig1)
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # Bottom Section - Data Table and Word Cloud
                    col3, col4 = st.columns([1.2, 1], gap="large")
                    
                    with col3:
                        st.markdown("<h2 class='section-header'>Filtered Text Data</h2>", unsafe_allow_html=True)
                        display_data = filtered.head(text_sel)[["sentiment", "text"]]
                        st.dataframe(display_data, use_container_width=True, height=400)
                        
                        if len(filtered) < text_sel:
                            st.warning(f"⚠️ Only {len(filtered)} texts found containing '{voc_sel}'.")
                    
                    with col4:
                        st.markdown("<h2 class='section-header'>Word Cloud</h2>", unsafe_allow_html=True)
                        
                        wordcloud = WordCloud(
                            background_color="#0F2027",
                            max_words=100,
                            colormap="plasma",
                            random_state=42,
                            collocations=False,
                            min_word_length=3,
                            width=800,
                            height=400,
                            max_font_size=150
                        ).generate(str(words))
                        
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        fig2.patch.set_facecolor('#0F2027')
                        ax2.imshow(wordcloud, interpolation='bilinear')
                        ax2.axis("off")
                        plt.tight_layout()
                        st.pyplot(fig2)
                else:
                    st.error(f"❌ No texts found containing the word '{voc_sel}'. Try a different word.")

# --- ML Class ---
class ML(info_insights):
    def ml(self):
        self.css()
        
        st.title("🧠 AI-Powered Sentiment Analysis")
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #ffd700; margin-bottom: 15px;'>💡 How It Works</h3>
            <p>
                Enter your text below and our machine learning model will analyze the sentiment in real-time.
                The model uses <strong>Logistic Regression with TF-IDF vectorization</strong> and achieves <strong>60-70% accuracy</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Spelling mistakes may affect the accuracy of the results.")
        
        user_text = st.text_area(
            label="Enter your text",
            placeholder="Type or paste your social media text here... (e.g., 'I love this product! It's amazing!')",
            label_visibility="collapsed",
            height=150
        )

        but_sel = st.button("🚀 Analyze Sentiment", use_container_width=True)
        
        if but_sel:
            if len(user_text.strip()) > 0:
                try:
                    with st.spinner("🔄 Analyzing sentiment... This may take a moment..."):
                        data = self.df[["text", "sentiment"]]
                        X = data["text"]
                        y = data["sentiment"]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
                        
                        operation = Pipeline([
                            ("tfidf", TfidfVectorizer(stop_words=None,ngram_range=(1,2))),
                            ("model", LogisticRegression(max_iter=1000))
                        ])
                        
                        param_grid = {
                            "tfidf__ngram_range": [(1,1), (1,2)],
                            "tfidf__min_df": [1, 2, 3],
                            "tfidf__max_df": [0.8, 0.9, 1.0],
                            "model__C": [0.1, 0.5, 1, 2, 5]
                        }
                        
                        gridmodel = GridSearchCV(estimator=operation, param_grid=para, cv=5, n_jobs=-1)
                        gridmodel.fit(X_train, y_train)

                        pred = gridmodel.predict([user_text])[0]
                        probs = gridmodel.predict_proba([user_text])[0]
                        classes = gridmodel.classes_

                    st.success("✅ Analysis Complete!")
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # Results Section
                    st.markdown("<h2 class='section-header'>📊 Analysis Results</h2>", unsafe_allow_html=True)
                    
                    idx = list(classes).index(pred)
                    confidence = probs[idx]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        emoji_map = {"positive": "😀", "neutral": "😐", "negative": "😞"}
                        emoji = emoji_map.get(pred.lower(), "🤔")
                        
                        st.markdown(f"""
                        <div class='result-card'>
                            <div class='result-label'>Predicted Sentiment</div>
                            <div class='result-value'>{pred.upper()}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='result-card'>
                            <div class='emoji-large'></div>
                            <div class='result-label'>Confidence Score</div>
                            <div class='result-value'>{confidence*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # Detailed Breakdown
                    st.subheader("📋 Detailed Confidence Breakdown")
                    
                    emoji_map = {"positive": "😀", "neutral": "😐", "negative": "😞"}
                    det_Score = pd.DataFrame({
                        "Emoji": [emoji_map.get(c.lower(), "🤔") for c in classes],
                        "Sentiment": [c.upper() for c in classes],
                        "Confidence": [f"{p*100:.2f}%" for p in probs],
                        "Probability": probs
                    })
                    
                    # Style the dataframe
                    st.dataframe(
                        det_Score[["Emoji", "Sentiment", "Confidence"]],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Something went wrong: {e}")
                    logging.error(f"ML Error: {e}")
            else:
                st.warning("⚠️ Please enter some text to analyze!")

# --- Main App ---
class App(ML):
    def run_info(self):
        self.load_data()
        if not self.df.empty:
            self.info()

    def run_eda(self):
        self.load_data()
        if not self.df.empty:
            self.eda()

    def run_ml(self):
        self.load_data()
        if not self.df.empty:
            self.ml()

    def app(self):
        options = {
            " Overview": self.run_info,
            " Insights": self.run_eda,
            " Analyzer": self.run_ml
        }

        col1, col2, col3 = st.columns([2, 2, 2])

        with col2:
            key_sel = st.radio(
                "Navigation",
                list(options.keys()),
                horizontal=True,
                label_visibility="collapsed"
            )
        st.markdown("<hr>", unsafe_allow_html=True)
                            
        val_Sel = options[key_sel]
        val_Sel()

# --- Run app ---
if __name__ == "__main__":
    obj = App()
    obj.app()
    