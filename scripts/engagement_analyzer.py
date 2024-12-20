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