import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os
import logging
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV


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
            
            .word{
                text-align: center;
                font-size: 20px;
                font-weight: 400;
                # margin-lef
                
            }
     
            </style>
        """, unsafe_allow_html=True)
           
         
class info_insights(CSS):
    st.set_page_config(layout="centered")
    
    def load_data(self):
        file_path = "Data/sentimentdataset.csv"
        
        if os.path.exists(file_path):
            try:
                self.df = pd.read_csv(file_path)
                logging.info(self.df.isnull().sum() / len(self.df) * 100)
                logging.info(self.df.duplicated().sum() / len(self.df) * 100)
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
        st.markdown("""
            <p1 class='word'>This app performs sentiment analysis on the user input,Since app shows different sentiments too apart of [**Positive,Neutral,Negative**]
            sentiments.It is more suitable for App to predict on larger and clearer Text.This app only Supports **English Words**.
            Following is the Dataset used for Training 👇🏻</p1>""",unsafe_allow_html=True)
        
        
        columns=self.df[["Text","Sentiment"]]
        
        
        st.dataframe(columns)
        st.markdown("<h2 class='gradient-text'>Sentiments</h2>",unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(self.df["Sentiment"].unique(),columns=["Sentiments"]))
        
        gr=self.df["Sentiment"].value_counts().reset_index()
        gr.columns = ["Sentiment", "Count"]
        
        st.markdown("<h4 class='gradient-text'>Most Frequent Sentiments</h4>",unsafe_allow_html=True)
        gr_sel=gr[gr["Count"]>5]
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,6),dpi=100)
        sns.barplot(x="Sentiment", y="Count", data=gr_sel, ax=ax, palette="viridis")
        plt.xticks(rotation=90)
        st.pyplot(fig)
        
        
    def eda(self):
        pass

class ML(info_insights):
    
    
    def ml(self):
        
        st.set_page_config(layout="wide")
        data=self.df[["Text","Sentiment"]]
        
        X=data["Text"]
        y=data["Sentiment"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        
        tf_idf=TfidfVectorizer(stop_words="english")
        x_train_vec=tf_idf.fit_transform(X_train)
        x_test_vec=tf_idf.transform(X_test)
        
        voc=tf_idf.vocabulary_
        
        
        with st.sidebar.form(key="search_form"):
            
            st.subheader("Search Parameters")
            voc_sel=st.selectbox("choose",voc)
            text_sel=st.slider("Number of texts", min_value=100, max_value=735, key="num_tweets")
            
            if st.form_submit_button("Search"):
                pass
                
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
                "Insights📊": self.run_eda,
                 "Sentiment Analyzer🔎":self.run_ml}
        
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
            