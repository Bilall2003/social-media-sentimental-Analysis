import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os
import logging


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
                background: linear-gradient(purple,pink,white);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                padding: 10px 0;
                margin-right:20;
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
           
         
class EDA(CSS):
    
    def info(self):
        
        st.set_page_config(layout="centered")
        self.css()
        st.markdown("<h1 class='gradient-text'>Social Media Sentiment Analyzer</h1>", unsafe_allow_html=True)
        st.warning("Read the instructions carefully....")
        st.markdown("""
            <p1 class='word'>This app performs sentiment analysis on the user input,Since app shows different sentiments too apart of [**Positive,Neutral,Negative**]
            sentiments.It is more suitable for App to predict on larger and clearer Text.This app only Supports **English Words**.
            Following is the Dataset used for Training 👇🏻</p1>""",unsafe_allow_html=True)
        
        file_path="Data/sentimentdataset.csv"
        
        if os.path.exists(file_path):
            
            try:
                
                df=pd.read_csv(file_path)
                
                logging.info(df.isnull().sum()/len(df)*100) # no null values found
            
                logging.info(df.duplicated().sum()/len(df)*100) # no duplicate values found
            
            except Exception as e:
                
                logging.error(e)
                
        else:
            logging.error("invalid filepath...........")
        
        columns=df[["Text","Sentiment"]]
        
        
        st.dataframe(columns.style.background_gradient(cmap="Blues"))
       
        
        
    def eda(self):
        pass
        
            
class App(EDA):
    
    def run_info(self):
        
        self.info()
    
    def run_eda(self):
        
        self.eda()
    
    def app(self):
        
        options={"OverView":self.run_info,
                "Insights📊": self.run_eda,
                 "Sentiment Analyzer🔎":self.run_eda}
        
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
        
        # st.markdown("""
        #         <style>
        #         .line {
        #             width: 100%;
        #             height: 4px;
        #             margin: 20px 0;
        #             background: linear-gradient(90deg, #00C6FF, #FFD700);
        #         }
        #         </style>
        #     """, unsafe_allow_html=True)
        st.markdown("<hr id='line'>", unsafe_allow_html=True)


        val_Sel()

obj = App()
obj.app()
            