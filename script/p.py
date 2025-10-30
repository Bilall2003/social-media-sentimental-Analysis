import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import os


class EDA:
    
    def eda(self):
        
        file_path = "../Data/sentimentaldataset.csv"

        
        if os.path.exists(file_path):
            
            file=pd.read_csv(file_path)
            print(file)
        
        else:
            
            print("invalid path...")

obj=EDA()
obj.eda()
        
        
