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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report,accuracy_score
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
            
            .word{
                text-align: center;
                font-size: 20px;
                font-weight: 400;
            }
     
            </style>
        """, unsafe_allow_html=True)
           
         
class info_insights(CSS):
    
    def load_data(self):
        file_path = "Data/test.csv"
        
        if os.path.exists(file_path):
            try:
                self.df = pd.read_csv(file_path,encoding="latin-1")
                print(self.df.info())
                logging.info(self.df.isnull().sum() / len(self.df) * 100)
                logging.info(self.df.duplicated().sum() / len(self.df) * 100)
                self.df.dropna(inplace=True)
                self.df.drop_duplicates(keep="first",inplace=True)
                self.df.reset_index(drop=True,inplace=True)
                print(self.df.info())
                print(self.df.info())
                logging.info(self.df.isnull().sum() / len(self.df) * 100)
                logging.info(self.df.duplicated().sum() / len(self.df) * 100)
                return self.df
            except Exception as e:
                logging.error(e)
                return pd.DataFrame()
        else:
            logging.error("Invalid filepath.")
            return pd.DataFrame()
obj=info_insights()
obj.load_data()