import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class ExperienceAnalyzer:
    def __init__(self, df):
        self.df = df

    def fill_missing_values(self):
        """
        This function handles missing values in a DataFrame by:
        - Replacing missing values in numerical columns with their mean.
        - Replacing missing values in categorical columns with their mode.
        - Using an empty string ("") for categorical columns where mode is undefined.

        Args:
            df (pd.DataFrame): The input DataFrame with potential missing values.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled.
        """
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            self.df[col].fillna(self.df[col].mean(), inplace=True)

        # Replace missing values with mode for categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if not self.df[col].mode().empty:  # Check if mode() returns non-empty
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            else:
                # Handle columns where mode is empty (e.g., all NaN values)
                self.df[col].fillna("", inplace=True)

        return self.df
    def aggregate_customer_data(self, df):
        """
        Aggregates the required information per customer (MSISDN/Number).
        
        Args:
        df (DataFrame): The input DataFrame containing the data.
        
        Returns:
        DataFrame: Aggregated DataFrame with mean values for numerical columns and the first entry for categorical columns.
        """
        aggregated_df = df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'TCP UL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg Bearer TP UL (kbps)': 'mean',
            'Handset Type': 'first'  # Taking the first handset type per customer
        }).reset_index()

        return aggregated_df
    def get_top_bottom_most_freq_values(self, df, column_name, top_n=10):
        """
        Retrieve the top N, bottom N, and most frequent N values for a specified column.

        This function analyzes a given column in the DataFrame to extract:
        - Top N values (highest values).
        - Bottom N values (lowest values).
        - Most frequent N values and their frequencies.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            column_name (str): The name of the column to analyze.
            top_n (int): The number of top, bottom, and most frequent values to retrieve. Default is 10.

        Returns:
            pd.DataFrame: A consolidated DataFrame containing:
                - Top N values with their indices.
                - Bottom N values with their indices.
                - Most frequent N values and their frequencies.
        """
        # Extract the top N values (highest values in the column)
        top_values = df[column_name].nlargest(top_n).reset_index(name=f'Top {column_name}')
        
        # Extract the bottom N values (lowest values in the column)
        bottom_values = df[column_name].nsmallest(top_n).reset_index(name=f'Bottom {column_name}')
        
        # Extract the most frequent N values in the column along with their counts
        most_freq_values = (
            df[column_name]
            .value_counts()
            .nlargest(top_n)
            .reset_index(name=f'Most Frequent {column_name}')
        )
        
        # Rename columns for clarity
        top_values.columns = ['Index', f'Top {column_name}']
        bottom_values.columns = ['Index', f'Bottom {column_name}']
        most_freq_values.columns = [f'Most Frequent {column_name}', 'Frequency']

        # Merge all results into a single DataFrame
        result_df = pd.concat([top_values, bottom_values, most_freq_values], axis=1)
        
        return result_df
