# User Engagement Analysis
# User Engagement Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class EngagementAnalyzer:
    def __init__(self, df):
        self.df = df

    def user_engagement(self,df):
       # Calculate session frequency for each user
        session_frequency = df.groupby('MSISDN/Number').size().reset_index(name='Session Frequency')

        # Calculate duration of the session (already provided as 'Dur. (ms)')
        # If we need total session duration per user, you can sum it up
        session_duration = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='Total Session Duration (ms)')

        # Calculate session total traffic
        session_traffic = df.groupby('MSISDN/Number').agg({
            'Total UL (Bytes)': 'sum',
            'Total DL (Bytes)': 'sum'
        }).reset_index()
        session_traffic.columns = ['MSISDN/Number', 'Total UL (Bytes)', 'Total DL (Bytes)']

        # Merge all metrics into a single DataFrame
        user_engagement = session_frequency.merge(session_duration, on='MSISDN/Number')
        user_engagement = user_engagement.merge(session_traffic, on='MSISDN/Number')

        # Display the final DataFrame with user engagement metrics
        return user_engagement
    
    def high_engagement_users(self,df):
        user_engagement=self.user_engagement(df)
        # Define high engagement threshold (e.g., top 10% of each metric)
        freq_threshold = user_engagement['Session Frequency'].quantile(0.9)
        duration_threshold = user_engagement['Total Session Duration (ms)'].quantile(0.9)
        traffic_threshold = user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1).quantile(0.9)

        # Filter high engagement users
        high_engagement_users = user_engagement[
            (user_engagement['Session Frequency'] >= freq_threshold) &
            (user_engagement['Total Session Duration (ms)'] >= duration_threshold) &
            ((user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)) >= traffic_threshold)
        ]
        return high_engagement_users
    
    def plot_user_engagement(self,df):
        user_engagement=self.user_engagement(df)
        # Define high engagement threshold (e.g., top 10% of each metric)
        freq_threshold = user_engagement['Session Frequency'].quantile(0.9)
        duration_threshold = user_engagement['Total Session Duration (ms)'].quantile(0.9)
        traffic_threshold = user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1).quantile(0.9)

        # Filter high engagement users
        high_engagement_users = user_engagement[
            (user_engagement['Session Frequency'] >= freq_threshold) &
            (user_engagement['Total Session Duration (ms)'] >= duration_threshold) &
            ((user_engagement[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)) >= traffic_threshold)
        ]

        # Plot High Engagement Users

        # Set up the figure and axes
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # Plot High Engagement Users by Session Frequency
        sns.histplot(high_engagement_users['Session Frequency'], bins=50, kde=True, ax=axs[0], color='blue')
        axs[0].set_title('High Engagement Users - Session Frequency')
        axs[0].set_xlabel('Session Frequency')
        axs[0].set_ylabel('Number of High Engagement Users')

        # Plot High Engagement Users by Session Duration
        sns.histplot(high_engagement_users['Total Session Duration (ms)'], bins=50, kde=True, ax=axs[1], color='green')
        axs[1].set_title('High Engagement Users - Total Session Duration')
        axs[1].set_xlabel('Total Session Duration (ms)')
        axs[1].set_ylabel('Number of High Engagement Users')

        # Plot High Engagement Users by Total Traffic
        high_engagement_users['Total Traffic (Bytes)'] = high_engagement_users[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)
        sns.histplot(high_engagement_users['Total Traffic (Bytes)'], bins=50, kde=True, ax=axs[2], color='red')
        axs[2].set_title('High Engagement Users - Total Traffic')
        axs[2].set_xlabel('Total Traffic (Bytes)')
        axs[2].set_ylabel('Number of High Engagement Users')

        # Adjust layout.
        plt.tight_layout()
        plt.show()
    def top_10_users_per_metric(self, df):
        # Calculate the top 10 users based on the specified metric
        high_engagement_users=self.high_engagement_users(df)
        # Calculate Total Traffic using sum of UL and DL Bytes
        high_engagement_users['Total Traffic (Bytes)'] = high_engagement_users[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)
        top_10_users_freq = high_engagement_users.nlargest(10, 'Session Frequency')
        top_10_users_duration = high_engagement_users.nlargest(10, 'Total Session Duration (ms)')
        top_10_users_traffic = high_engagement_users.nlargest(10, 'Total Traffic (Bytes)')
        print("Top 10 Users by Session Frequency:\n", top_10_users_freq, "\n")
        print("Top 10 Users by Total Session Duration:\n", top_10_users_duration, "\n")
        print("Top 10 Users by Total Traffic:\n", top_10_users_traffic, "\n")
    
    
    def top_10_users(self,df):
        # Calculate the top 10 users based on the specified metric
        high_engagement_users=self.high_engagement_users(df)
        # Calculate Total Traffic using sum of UL and DL Bytes
        high_engagement_users['Total Traffic (Bytes)'] = high_engagement_users[['Total UL (Bytes)', 'Total DL (Bytes)']].sum(axis=1)
        top_10_users_freq = high_engagement_users.nlargest(10, 'Session Frequency')
        top_10_users_duration = high_engagement_users.nlargest(10, 'Total Session Duration (ms)')
        top_10_users_traffic = high_engagement_users.nlargest(10, 'Total Traffic (Bytes)')
        return top_10_users_freq, top_10_users_duration, top_10_users_traffic
    def aggregate_traffic_per_user(self, df, applications):
        """
        Aggregates the traffic per user for the specified applications.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the traffic data.
        applications (list of str): List of application columns to aggregate.

        Returns:
        pd.DataFrame: A DataFrame with aggregated traffic per user.
        """
        return df.groupby('MSISDN/Number')[applications].sum().reset_index()

    def calculate_total_traffic(self, app_engagement, applications):
        """
        Calculates the total traffic for each application (DL + UL) and adds it to the DataFrame.

        Parameters:
        app_engagement (pd.DataFrame): DataFrame with aggregated traffic per user.
        applications (list of str): List of application columns to calculate total traffic for.

        Returns:
        pd.DataFrame: Updated DataFrame with total traffic columns added.
        """
        for app in applications:
            total_col_name = app.replace(' (Bytes)', ' Total (Bytes)')
            app_engagement[total_col_name] = app_engagement[app]
        
        # Calculate total traffic for Social Media as an example
        app_engagement['Social Media Total (Bytes)'] = (
            app_engagement['Social Media DL Total (Bytes)'] + 
            app_engagement['Social Media UL Total (Bytes)']
        )
        
        return app_engagement
    def get_top_users(self, app_engagement, application, n=10):
        """
        Retrieves the top N users based on total traffic for a given application.

        Parameters:
        app_engagement (pd.DataFrame): DataFrame with total traffic per user.
        application (str): Application name for which to retrieve top users.
        n (int): Number of top users to retrieve (default is 10).

        Returns:
        pd.DataFrame: A DataFrame containing the top N users for the specified application.
        """
        column_name = f'{application} Total (Bytes)'
        return app_engagement.nlargest(n, column_name)
