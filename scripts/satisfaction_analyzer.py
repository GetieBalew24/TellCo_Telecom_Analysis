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

    def calculate_engagement_scores(self, engagement_df, least_engaged_cluster, metrics):
        """
        Calculate the engagement score for each user as the Euclidean distance 
        from the user’s data point to the centroid of the least engaged cluster.

        Args:
            engagement_df (DataFrame): The DataFrame containing user engagement data.
            least_engaged_cluster (int): The cluster with the least engagement.
            metrics (list): List of metrics used for clustering.

        Returns:
            DataFrame: Engagement scores for each user based on their distance to 
            the least engaged cluster's centroid.
        """
        # Step 1: Get the centroid of the least engaged cluster
        least_engaged_centroid = engagement_df[engagement_df['Engagement Cluster'] == least_engaged_cluster][metrics].mean()

        # Step 2: Calculate the Euclidean distance from each user to the least engaged cluster centroid
        engagement_df['Engagement Score'] = pairwise_distances(engagement_df[metrics], [least_engaged_centroid], metric='euclidean')

        return engagement_df
    def user_experience(self):
        """
        Analyze user experience by clustering users based on their experience metrics
        (e.g., retransmissions, RTT, throughput) and calculate the experience score
        using Euclidean distance from the worst experience cluster.

        Returns:
            DataFrame: A DataFrame containing user experience metrics and their 
            assigned experience clusters and experience scores.
        """
        # Define the relevant metrics for user experience
        metrics = [
            'Avg TCP DL Retransmission', 'Avg TCP UL Retransmission', 'Avg RTT DL', 
            'Avg RTT UL', 'Avg Throughput DL', 'Avg Throughput UL'
        ]

        # Normalize the metrics
        scaler = MinMaxScaler()
        df_normalized = self.df[metrics]
        df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized), columns=metrics)

        # Apply K-Means clustering with k=2 (good vs bad experience)
        kmeans = KMeans(n_clusters=2, random_state=42)
        self.df['Experience Cluster'] = kmeans.fit_predict(df_normalized)

        # Identify the worst experience cluster (the cluster with the highest average metric values)
        cluster_centroids = kmeans.cluster_centers_
        worst_cluster_index = np.argmax(cluster_centroids.sum(axis=1))  # Cluster with the highest sum of centroid values
        worst_cluster_centroid = cluster_centroids[worst_cluster_index]

        # Calculate the Euclidean distance for each user from the worst experience cluster
        self.df['Experience Score'] = pairwise_distances(df_normalized, [worst_cluster_centroid], metric='euclidean')

        # Return the relevant columns
        return self.df[['Customer Number', 'Experience Cluster', 'Experience Score']]
