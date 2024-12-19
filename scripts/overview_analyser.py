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
        
    def top_3_handset_manufacturers(self, df):
        """
        Plots the top 3 handset manufacturers based on the number of users.

        Args:
        df (pd.DataFrame): The input DataFrame containing handset information with a column 'Handset Manufacturer'.

        Returns:
        pd.DataFrame: A DataFrame with the top 3 handset manufacturers and their user counts.
        """
        # Count occurrences of each handset manufacturer
        handset_manufacturer_counts = df['Handset Manufacturer'].value_counts()

        # Get top 3 handset manufacturers
        top_3_handset_manufacturers = handset_manufacturer_counts.head(3)
        top_3_handset_manufacturers.to_frame().reset_index()
        # Print the top 3 handset manufacturers
        return top_3_handset_manufacturers
    def plot_top_3_handset_manufacturers(self,df):
        handset_manufacturer_counts=df['Handset Manufacturer'].value_counts()
        # Get top 3 handset manufacturers
        top_3_handset_manufacturers = handset_manufacturer_counts.head(3)
        
        # Plot the top 3 handset manufacturers
        top_3_handset_manufacturers.plot(kind='bar', legend=False, color='skyblue')
        plt.title('Top 3 Handset Manufacturers')
        plt.xlabel('Handset Manufacturers')
        plt.ylabel('Number of Users')
        plt.xticks(rotation=45, ha='right')
        plt.show()