import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

class SatisfactionAnalyzer:
    """
    A class to analyze user satisfaction based on engagement metrics and experience analysis.

    Attributes:
        df (DataFrame): The input DataFrame containing user data.
    """

    def __init__(self, df):
        """
        Initialize the SatisfactionAnalyzer with the input DataFrame.

        Args:
            df (DataFrame): The input DataFrame containing user data.
        """
        self.df = df

    def user_engagement(self,df):
        """
        Analyze user engagement by calculating session frequency, total duration,
        and total traffic. Normalize the metrics and apply K-Means clustering to 
        segment users into engagement clusters.

        Returns:
            DataFrame: A DataFrame containing user engagement metrics and their 
            assigned engagement clusters.
        """
        df['Total Duration']=df['Total UL (Bytes)']+df['Total DL (Bytes)']
        # Assume df is the DataFrame containing the dataset
        engagement_df = df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',  # This will give us the number of sessions per user
            'Total Duration': 'sum',  # Total duration of all sessions
            'Total UL (Bytes)': 'sum',  # Total upload bytes
            'Total DL (Bytes)': 'sum',  # Total download bytes
        }).reset_index()

        # Calculate the total traffic per user
        engagement_df['Total Traffic (Bytes)'] = engagement_df['Total UL (Bytes)'] + engagement_df['Total DL (Bytes)']

        # Rename columns for better understanding
        engagement_df.rename(columns={'Bearer Id': 'Session Frequency'}, inplace=True)
        
        # Selecting only the relevant columns for normalization
        metrics = ['Session Frequency', 'Total Duration', 'Total Traffic (Bytes)']
        scaler = MinMaxScaler()
        engagement_df[metrics] = scaler.fit_transform(engagement_df[metrics])

        # Applying K-Means clustering with k=3
        kmeans = KMeans(n_clusters=3, random_state=42)
        engagement_df['Engagement Cluster'] = kmeans.fit_predict(engagement_df[metrics])
        return engagement_df

    def get_least_engaged_cluster(self, df, cluster_column, metrics, verbose=False):
        """
        Determine the cluster with the least engagement based on specified metrics.

        Args:
            df (DataFrame): The input DataFrame containing the clustered data.
            cluster_column (str): The column name representing the cluster labels.
            metrics (list): List of metrics to consider for determining least engagement.
            verbose (bool): Whether to print intermediate results for debugging. Default is False.

        Returns:
            int: The cluster number with the least engagement.
        """
        # Calculate the centroids of each engagement cluster
        engagement_centroids = df.groupby(cluster_column)[metrics].mean()

        # Display the centroids for each cluster
        print(engagement_centroids)

        # Sum the normalized metrics for each cluster to get a measure of total engagement
        engagement_centroids['Total Engagement Score'] = engagement_centroids.sum(axis=1)

        # Identify the cluster with the lowest total engagement score
        least_engaged_cluster = engagement_centroids['Total Engagement Score'].idxmin()

        print(f"The least engaged cluster is: {least_engaged_cluster}")

        return least_engaged_cluster
    def get_worst_experience_cluster(self, df, cluster_column, metrics):
        """
        Determine the cluster with the worst experience based on specified metrics.
        
        Parameters:
        df (DataFrame): The input DataFrame containing the clustered data.
        cluster_column (str): The column name representing the cluster labels.
        metrics (list): List of metrics to consider for determining worst experience.
        
        Returns:
        int: The cluster number with the worst experience.
        """
        # Compute the mean values for each cluster
        cluster_means = df.groupby(cluster_column)[metrics].mean()

        # Determine the worst experience cluster
        # Define worst experience based on highest average values for the metrics
        # Example: Highest average RTT, TCP retransmission, etc.
        cluster_means['Average'] = cluster_means.mean(axis=1).idxmax()
        # Determine the worst experience cluster based on the highest average
        worst_experience_cluster = cluster_means['Average'].idxmax()

        print("Worst Experience Cluster:", worst_experience_cluster)
        return worst_experience_cluster

    # Function to calculate Euclidean distance
    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    # Task 4.1: Assign engagement and experience scores
    def calculate_scores(self, engagement_df, experience_df):
        
        # Remove 'MSISDN/Number' and 'Customer Number' it is not used in calculations
        engagement_columns = [col for col in engagement_df.columns if col != 'MSISDN/Number']
        experience_columns = [col for col in experience_df.columns if col != 'Customer Number']
        
        # Calculate the center of each engagement cluster
        engagement_centers = engagement_df.groupby('Engagement Cluster')[engagement_columns].mean()
        
        # Calculate engagement scores
        engagement_scores = []
        for _, row in engagement_df.iterrows():
            cluster_center = engagement_centers.loc[row['Engagement Cluster']]
            score = self.euclidean_distance(row[engagement_columns], cluster_center)
            engagement_scores.append(score)
        
        engagement_df['Engagement Score'] = engagement_scores
        
        # Calculate the center of each experience cluster
        numeric_cols = experience_df.select_dtypes(include=[np.number]).columns
        experience_centers = experience_df[numeric_cols].groupby('Experience Cluster').mean().drop(columns=['MSISDN/Number'], errors='ignore')
        
        # Calculate experience scores
        experience_scores = []
        for _, row in experience_df.iterrows():
            cluster_center = experience_centers.loc[row['Experience Cluster']]
            score = self.euclidean_distance(row[experience_columns], cluster_center)
            experience_scores.append(score)
        
        experience_df['Experience Score'] = experience_scores
        
        return engagement_df, experience_df