import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class UserOverviewAnalysis:
    def __init__(self,df):
        self.df=df
    def top_10_handsets(self,df):
        # Count occurrences of each handset (Handset Type)
        handset_counts = df['Handset Type'].value_counts()
        
       #Get top 10 handsets counts
        top_10_handsets=handset_counts.head(10)
        print("Top 10 Handsets Used by Customers:")
        return top_10_handsets.to_frame().reset_index()
    
    def plot_top_10_handsets(self, df):
        handset_counts = df['Handset Type'].value_counts()
        top_10_handsets = handset_counts.head(10)

        # Plot the top 10 handsets
        top_10_handsets.plot(kind='bar', legend=False, color='skyblue')
        plt.title('Top 10 Handsets Used by Customers')
        plt.xlabel('Handset Type')
        plt.ylabel('Number of Users')
        plt.xticks(rotation=45, ha='right')
        plt.show()