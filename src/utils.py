# This file contains useful utility functions.

import pandas
import numpy
from scipy import stats


def remove_illegal_symbols(s):
    """
    This function checks if a string variable contains any "illegal" characters that prevent the string to appear in a
    file name in the Windows OS.

    :param s (str): a string variable containing potentially illegal characters in Windows file name.
    :return: a string variable with any illegal characters removed.
    """

    illegal_symbols = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '^']
    cleaned_string = s

    for char in illegal_symbols:
        cleaned_string = cleaned_string.replace(char, '')

    return cleaned_string


def calculate_percentiles(data):
    # Calculate the percentile values for each data point
    percentile_values = [stats.percentileofscore(data, x, kind='rank') for x in data]
    return percentile_values


def calculate_monthly_average(data_df):
    try:
        # Resample data to monthly frequency and calculate the mean
        monthly_average_prices = data_df.resample('M').mean()
        return monthly_average_prices
    except Exception as e:
        print("An error occurred:", e)
        return None


def calculate_yoy_change(data_df):
    """
    Calculate the year-over-year (YoY) change for the economic data.

    Parameters:
        data_df (pd.DataFrame): DataFrame containing the economic data with date as index.

    Returns:
        pd.DataFrame: DataFrame containing the YoY changes with date as index.
    """
    # Ensure the DataFrame is sorted by date
    data_df = data_df.sort_index()

    # Calculate YoY change using pandas' shift() and pct_change() methods
    data_df['YoY Change'] = data_df.iloc[:, 0].pct_change(periods=12) * 100

    # Drop the first 12 rows since they don't have enough data for the YoY calculation
    data_df = data_df.iloc[12:]

    # Create a new DataFrame containing only the YoY Change column
    yoy_df = data_df[['YoY Change']].copy()

    return yoy_df


def calculate_running_percentile(data_df, window_size):
    """
    Calculate the running percentile rank of the current data in the past x months.

    Parameters:
        data_df (pd.DataFrame): DataFrame containing the economic data with date as index.
        window_size (int): Number of months for the rolling window.

    Returns:
        pd.DataFrame: DataFrame containing the running percentile rank values with date as index.
    """
    # Ensure the DataFrame is sorted by date
    data_df = data_df.sort_index()

    # Calculate the running percentile rank using rolling window and apply method
    def percentile_rank(x):
        return pd.Series(x).rank(pct=True).iloc[-1]

    running_percentile = data_df.rolling(window=f"{window_size}M").apply(percentile_rank)

    return running_percentile


