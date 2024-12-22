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
    def plot_top_10_throughput_distribution(self, df, column_name='Avg Bearer TP DL (kbps)', top_n=10):
        """
        Plot the distribution of the top N average throughput per handset type.
        
        Args:
        df (DataFrame): The input DataFrame containing the data.
        column_name (str): The column name representing average throughput. Default is 'Avg Bearer TP DL (kbps)'.
        top_n (int): The number of top handset types to include in the plot. Default is 10.
        
        Returns:
        None: Displays a boxplot showing the distribution of throughput values for the top N handset types.
        """
        # Sort the DataFrame to get the top N handset types based on average throughput
        top_throughput_df = df.sort_values(by=column_name, ascending=False).head(top_n)
        
        # Plot the distribution of average throughput using a boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Handset Type', y=column_name, data=top_throughput_df)
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.title(f'Distribution of {column_name} per Handset Type')  # Add a title to the plot
        plt.xlabel('Handset Type')  # Label for x-axis
        plt.ylabel(column_name)  # Label for y-axis
        plt.show()  # Display the plot

    def plot_top_10_tcp_retransmission(self, df, column_name='TCP DL Retrans. Vol (Bytes)', top_n=10):
        """
        Plot the average TCP retransmission volume for the top N handset types.
        
        Args:
        df (DataFrame): The input DataFrame containing the data.
        column_name (str): The column name representing TCP retransmission volume. Default is 'TCP DL Retrans. Vol (Bytes)'.
        top_n (int): The number of top handset types to include in the plot. Default is 10.
        
        Returns:
        None: Displays a bar plot showing the average TCP retransmission volume for the top N handset types.
        """
        # Group by 'Handset Type' and compute the average TCP retransmission volume
        avg_tcp_retrans_per_handset = df.groupby('Handset Type')[column_name].mean().reset_index()
        
        # Sort the grouped data to get the top N handset types
        top_tcp_retrans_df = avg_tcp_retrans_per_handset.sort_values(by=column_name, ascending=False).head(top_n)
        
        # Plot the average TCP retransmission volume using a barplot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Handset Type', y=column_name, data=top_tcp_retrans_df)
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.title(f'Average {column_name} per Handset Type')  # Add a title to the plot
        plt.xlabel('Handset Type')  # Label for x-axis
        plt.ylabel(f'Average {column_name}')  # Label for y-axis
        plt.show()  # Display the plot
