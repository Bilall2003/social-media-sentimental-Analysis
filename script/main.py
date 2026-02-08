import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os
import time
import logging
import re
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class CSS:
    
    def css(self):
        
        st.markdown("""
            <style>
            
            .gradient-text {
                font-size: 60px;
                font-weight: 900;
                background: linear-gradient(purple,pink,grey);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                padding: 10px 0;
                margin-right: 20px;
                margin-bottom: 30px;
            }
            
             .line, #line {
                width: 98%;
                height: 4px;
                margin: 20px 0;
                background: linear-gradient(pink,white);
            }
            
            .word {
                text-align: center;
                font-size: 20px;
                font-weight: 400;
                animation: blink 2s infinite; /* 1s duration, repeat forever */
            }

            @keyframes blink {
                0%, 50%, 100% { opacity: 1; }   
                25%, 75% { opacity: 0; }        
            }

     
            </style>
        """, unsafe_allow_html=True)
           
         
class info_insights(CSS):
    
        def load_data(self):
            file_path = "Data/train.csv"
            
            if os.path.exists(file_path):
                try:
                    self.df = pd.read_csv(file_path,encoding="latin-1")
                    self.df.dropna(inplace=True)#26% null values
                    self.df.drop_duplicates(keep="first",inplace=True)  # 26% duplicaate found
                    self.df.reset_index(drop=True,inplace=True)#reset index after cleaning
                    return self.df
                except Exception as e:
                    logging.error(e)
                    st.error(f"Error loading data: {e}")
                    return pd.DataFrame()
            else:
                logging.error("Invalid filepath.")
                st.error("Invalid filepath.")
                return pd.DataFrame()
        
        def info(self):
            
            self.css()
            st.markdown("<h1 class='gradient-text'>Social Media Sentiment Analyzer</h1>", unsafe_allow_html=True)
            st.warning("Read the instructions carefully....")
            st.caption("""
                This app performs sentiment analysis on the user input,Since app shows different sentiments [**Positive,Neutral,Negative**]
                sentiments.It is more suitable for App to predict on larger and clearer Text.This app only Supports **English Words**.
                Following is the Dataset used for Training 👇🏻""")
            
            
            columns=self.df[["text","sentiment"]]
            
            
            st.dataframe(columns)
            st.markdown("<h2 class='gradient-text'>Sentiments</h2>",unsafe_allow_html=True)
            
            gr=self.df["sentiment"].value_counts().reset_index()
            gr.columns = ["sentiment", "Count"]
            
            st.markdown("<h4 class='gradient-text'>Most Frequent sentiment</h4>",unsafe_allow_html=True)
            fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,6),dpi=100)
            sns.barplot(x="sentiment", y="Count", data=gr, ax=ax, palette="viridis")
            plt.xticks(rotation=90)
            st.pyplot(fig)
            
            
        def eda(self):
            
                st.set_page_config(layout="wide")
                data=self.df[["text","sentiment"]]
                
                X=data["text"]
                y=data["sentiment"]
                
                
                tf_idf=TfidfVectorizer(stop_words="english")
                x_vec=tf_idf.fit_transform(X)
                
                voc=tf_idf.vocabulary_
                
                form_adjust="""
                    <style>
                    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
                        margin-top: 150px;
                        text-align: center;
                        padding: 0.5px;
                    }
                    </style>
                    """
                st.write(form_adjust,unsafe_allow_html=True)

                with st.sidebar.form(key="search_form"):
                    
                    st.subheader("Search Parameters")
                    
                    voc_sel=st.selectbox("Choose Vocabulary",list(voc.keys()),key="voc_select")
                    text_sel=st.slider("Number of texts", min_value=5, max_value=735, key="num_tweets")
                    
                    submitted=st.form_submit_button("Search")
                    
                    st.markdown("Note: It may take a while to load results,especially with large number of texts")
                    
                    
                if submitted:
                    
                    col1,col2=st.columns([8,13],gap="large")
                    
                    with col1:
                        filtered = data[data["text"].str.contains(voc_sel, case=False, na=False)][["text", "sentiment"]]
                        sentiment_counts = filtered["sentiment"].value_counts()
                        
                        if not filtered.empty:
                            st.subheader(f"Sentiment distribution")

                            # --- Create Pie Chart
                            fig, ax = plt.subplots(figsize=(20, 6),dpi=50)
                            ax.pie(
                                sentiment_counts,
                                labels=sentiment_counts.index,
                                autopct='%1.1f%%',
                                startangle=90,
                                wedgeprops={'edgecolor': 'black'},
                                shadow=True
                            )
                            plt.tight_layout()
                            st.pyplot(fig)

                        else:
                            st.warning(f"No sentences found containing '{voc_sel}'.")

                    with col2:

                        # Combine all text into one long string
                        text = " ".join(filtered["text"].astype(str).tolist())

                        # Clean text: remove punctuation and lowercase
                        text = re.sub(r"[^\w\s]", "", text.lower())

                        words = text.split()
                        word_counts = Counter(words)
                        top_10 = word_counts.most_common(10)
                        
                        df_10=pd.DataFrame(top_10,columns=["Word","Count"])
                        
                        st.subheader(f"Top 10 Occuring Words")
                        fig1,ax1=plt.subplots(figsize=(20,10))
                        sns.barplot(x="Word",y="Count",data=df_10,color="green",ax=ax1)
                        plt.xlabel("vdfvnl")
                        plt.tight_layout()
                        st.pyplot(fig1)
                        

                    col3,col4=st.columns([1,2],gap="large")
                    
                    with col3:
                        st.subheader(f"Number of Texts")
                        if len(filtered)>=text_sel:
                        
                            st.dataframe(pd.DataFrame(filtered.head(text_sel)),column_order=["sentiment","text"])
                            
                        else :
                            st.dataframe(pd.DataFrame(filtered.head(text_sel)),column_order=["sentiment","text"])
                            st.warning(f"This vocabulary has not much text you selected : {text_sel}")
                            
                    with col4:
                        wordcloud = WordCloud(
                        background_color="white",
                        max_words=90,
                        colormap="Greens",
                        random_state=42,
                        collocations=False,
                        min_word_length=2,
                        max_font_size=250,
                    ).generate(str(words))

                    # Display it in Streamlit
                        st.subheader("Word Cloud of Text Data")
                        fig, ax = plt.subplots(figsize=(2,4))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        plt.tight_layout()
                        st.pyplot(fig)
                                                

class ML(info_insights):
        
        
        def ml(self):
            st.set_page_config(layout="centered")
            self.css()

            st.markdown("<h2 class='gradient-text'>🧠Sentiment Analysis with Ml Models</h2>",unsafe_allow_html=True)
            st.caption("Analysis is done on ML model which probably gives 60-70% accuracy or sometimes less so every text cannot give **True** answer")
            
            user_text=st.text_area(label="Enter your text",label_visibility="collapsed",placeholder="Enter your text")
            
            st.markdown("""
                <style>
                div.stButton > button:first-child {
                    width: 100%;
                    background: linear-gradient(grey, pink, purple);
                    font-weight: 700;
                    cursor: pointer;
                    border: none;
                    color: black;
                    padding: 1px 300px;
                    border-radius: 5px;
                    transition: all 0.3s ease;
                    font-size: 20px;
                }

                div.stButton > button:first-child:hover {
                    transform: scale(1.05);
                    opacity: 0.9;
                }
                </style>
            """, unsafe_allow_html=True)

            but_sel = st.button("Analyze Sentiment")

            if but_sel and len(user_text)>0:
                
                try:
                
                    st.info("Hold On...This may take a few Moments")
                    with st.status("Analyzing.....", expanded=True) as status:
                        st.write("Checking text...")
                        time.sleep(5)

                        st.write("Fetching information...")
                        time.sleep(5)

                        st.write("Running sentiment model...")

                        data=self.df[["text","sentiment"]]
                        
                        X=data["text"]
                        y=data["sentiment"]
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
                        
                        tf_idf=TfidfVectorizer(stop_words="english")
                        x_train_vec=tf_idf.fit_transform(X_train)
                        x_test_vec=tf_idf.transform(X_test)
                        
                        operation = Pipeline([
                            ("tfidf", TfidfVectorizer(stop_words="english")),
                            ("model", LogisticRegression(max_iter=1000))
                        ])
                        
                        para={
                            "model__penalty":["l2"],
                            'model__C': [0.1, 0.5, 1, 2, 5],
                            "model__solver":['lbfgs', 'liblinear']
                            
                        }
                        
                        gridmodel=GridSearchCV(estimator=operation,param_grid=para,cv=5,n_jobs=-1,verbose=2)
                        gridmodel.fit(X_train,y_train)
                        
                        pred = gridmodel.predict([user_text])[0]
                        probs = gridmodel.predict_proba([user_text])[0]
                        classes = gridmodel.classes_
                        
                        status.update(label="✅ Analysis complete!",state="complete")
                    
                    with st.form("Report"):
                        st.title("Analysis Report")
                        col1,col2=st.columns(2,gap="large")
                        col3,col4=st.columns(2,gap="large")
                        
                        idx = list(classes).index(pred)
                        confidence = probs[idx]

                        with col1:
                            st.markdown("**Predicted Sentiment**")
                        with col2:
                            st.markdown("**Confidence Score**")
                        with col3:
                            st.markdown(f"<p2 class='word'>{pred}</p2>",unsafe_allow_html=True)
                        with col4:
                            st.markdown(f"<p2 class='word'>{confidence*100:.2f}%</p2>",unsafe_allow_html=True)
                        
                        st.subheader("Detailed Scores👇🏻")
                        
                        det_Score=pd.DataFrame({" ":['😞',"😐","😀"],"Sentiment":classes,"Confidence":[f"{p*100:.2f}%" for p in probs]})
                        st.table(det_Score)
                        st.form_submit_button("Clear and Analyze another text")
                
                except Exception as e:
                    st.error(f"Something Went Wrong")
                
            elif len(user_text)==0:
                    st.warning("Please Enter Text first to proceed......")
                
                                
class App(ML):
    
    def run_info(self):
        
        self.load_data()
        self.info()
    
    def run_eda(self):
        
        self.load_data()
        self.eda()
    
    def run_ml(self):
        
        self.load_data()
        self.ml()
    
    def app(self):
        
        options={"OverView":self.run_info,
                "Insights": self.run_eda,
                "Sentiment Analyzer":self.run_ml
                }
        
        st.markdown("""
            <style>
            
            div[role="radiogroup"] {
                display: flex;
                justify-content: center;
                gap: 80px;
            }
            
            div[role="radiogroup"] label {
                background: linear-gradient(90deg, #ff758c, #ff7eb3);
                padding: 10px 20px;
                border-radius: 25px;
                cursor: pointer;
                font-weight: 900;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
                transition: all 0.25s ease;
                user-select: none;
            }

            div[role="radiogroup"] label:hover {
                transform: scale(1.05);
                opacity: 0.9;
            }

            div[role="radiogroup"] label:has(input[type="radio"]:checked) {
                background: linear-gradient(90deg, #8EC5FC, #E0C3FC);
                color: black;
            }
            </style>
            """, unsafe_allow_html=True)

        key_sel=st.radio("choose",list(options.keys()),horizontal=True,label_visibility="collapsed")
        val_Sel=options[key_sel]
        

        st.markdown("<hr id='line'>", unsafe_allow_html=True)


        val_Sel()

obj = App()
obj.app()
            