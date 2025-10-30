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


class EDA:
    
    def eda(self):
        
        file_path="Data/sentimentdataset.csv"
        
        if os.path.exists(file_path):
            
            try:
                
                df=pd.read_csv(file_path)
                print(df.head())
            
            except Exception as e:
                
                logging.error(e)
                
        else:
            logging.error("invalid filepath...........")

if __name__=="__main__":
    obj = EDA()
    obj.eda()
            