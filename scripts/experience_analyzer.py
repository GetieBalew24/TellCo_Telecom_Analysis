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