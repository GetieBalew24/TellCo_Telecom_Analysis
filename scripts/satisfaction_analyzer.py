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

    def user_engagement(self):
        """
        Analyze user engagement by calculating session frequency, total duration,
        and total traffic. Normalize the metrics and apply K-Means clustering to 
        segment users into engagement clusters.

        Returns:
            DataFrame: A DataFrame containing user engagement metrics and their 
            assigned engagement clusters.
        """
        # Calculate total duration for each session
        self.df['Total Duration'] = self.df['Total UL (Bytes)'] + self.df['Total DL (Bytes)']

        # Aggregate metrics at the user level
        engagement_df = self.df.groupby('MSISDN/Number').agg({
            'Bearer Id': 'count',  # Number of sessions per user
            'Total Duration': 'sum',  # Total duration of all sessions
            'Total UL (Bytes)': 'sum',  # Total uploaded bytes
            'Total DL (Bytes)': 'sum'  # Total downloaded bytes
        }).reset_index()

        # Calculate total traffic per user
        engagement_df['Total Traffic (Bytes)'] = engagement_df['Total UL (Bytes)'] + engagement_df['Total DL (Bytes)']

        # Rename columns for better clarity
        engagement_df.rename(columns={'Bearer Id': 'Session Frequency'}, inplace=True)

        # Select relevant columns for normalization
        metrics = ['Session Frequency', 'Total Duration', 'Total Traffic (Bytes)']

        # Normalize the metrics
        scaler = MinMaxScaler()
        engagement_df[metrics] = scaler.fit_transform(engagement_df[metrics])

        # Apply K-Means clustering with k=3
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
        # Step 1: Calculate the centroids (mean metric values) for each cluster
        engagement_centroids = df.groupby(cluster_column)[metrics].mean()

        if verbose:
            print("\nEngagement Centroids for Each Cluster:")
            print(engagement_centroids)

        # Step 2: Calculate the Total Engagement Score for each cluster
        engagement_centroids['Total Engagement Score'] = engagement_centroids.sum(axis=1)

        if verbose:
            print("\nTotal Engagement Scores:")
            print(engagement_centroids['Total Engagement Score'])

        # Step 3: Identify the cluster with the lowest total engagement score
        least_engaged_cluster = engagement_centroids['Total Engagement Score'].idxmin()

        if verbose:
            print(f"\nThe least engaged cluster is: {least_engaged_cluster}")

        return least_engaged_cluster

