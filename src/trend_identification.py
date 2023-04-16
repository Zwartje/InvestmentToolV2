# This module contains the functions for the trend identification, including a.o. the following:
#   - finding all the reflection points and then the validated ones (vMARP & vMIRP).
#   - divide the price curve into upward and downward trends

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator

import utils as ut

# Function: find all the validated reflection points.
def find_RP(price, window_in_days):
    """
    This function tries to identify all the minimum reflection points (MIRP) in a given price curve.
    Given a horizon tau, at any time t, one can look for the local min as the minimum within the time interval
    [t-tau, t+tau]:

    .. math::

        MIRP_{t, \tau} = min_s\{S_s | s\in[t-\tau, t+\tau]\}

    If s happens to be equal to t, then :math:`S_t` is called a minimum reflection point (MIRP).

    :param price: the price of interest to find the MIRPs. It should be a dataframe and contain only one column: the
    price (numeric). Index should be datetime.
    :param window_in_days: the 2-sided horizon length
    :return RP_vector: the location (i.e. date) of the validated reflection points, together with the full price vector.
    :return RP_summary: the list of (only) the validated reflection points.
    """

    RP_vector = pd.DataFrame(data=price.iloc[:, 0], index=price.index)
    RP_vector["running_min"] = RP_vector.iloc[:, 0].rolling(window=window_in_days * 2, min_periods=window_in_days,
                                                            center=True).min()
    RP_vector["running_max"] = RP_vector.iloc[:, 0].rolling(window=window_in_days * 2, min_periods=window_in_days,
                                                            center=True).max()
    RP_vector['is_MIRP'] = (RP_vector['running_min'] == RP_vector.iloc[:, 0])
    RP_vector['is_MARP'] = (RP_vector['running_max'] == RP_vector.iloc[:, 0])
    RP_vector['is_RP'] = RP_vector.is_MIRP | RP_vector.is_MARP

    # RP_summary contains ONLY the reflection points, while RP_vector contains the entire price curve.
    RP_summary = RP_vector[RP_vector['is_RP']]

    # find validated reflection points. See the methodology documentation for more information.

    RP_summary['is_MIRP_previous'] = RP_summary['is_MIRP'].shift(1)
    RP_summary['is_MARP_previous'] = RP_summary['is_MARP'].shift(1)
    RP_summary['is_MIRP_next'] = RP_summary['is_MIRP'].shift(-1)
    RP_summary['is_MARP_next'] = RP_summary['is_MARP'].shift(-1)
    RP_summary['is_duplicate_minimum'] = (RP_summary['is_MIRP'] & RP_summary['is_MIRP_previous']) | (
                RP_summary['is_MIRP'] & RP_summary['is_MIRP_next'])
    RP_summary['is_duplicate_maximum'] = (RP_summary['is_MARP'] & RP_summary['is_MARP_previous']) | (
                RP_summary['is_MARP'] & RP_summary['is_MARP_next'])
    RP_summary['is_duplicate'] = (RP_summary['is_duplicate_minimum']) | (RP_summary['is_duplicate_maximum'])

    RP_summary['group'] = ((RP_summary['is_duplicate_minimum'] != RP_summary['is_duplicate_minimum'].shift()) | (
                RP_summary['is_duplicate_maximum'] != RP_summary['is_duplicate_maximum'].shift())).cumsum()
    RP_summary['is_vMIRP'] = RP_summary['is_MIRP']
    RP_summary['is_vMIRP'].astype(bool)
    RP_summary['is_vMARP'] = RP_summary['is_MARP']
    RP_summary['is_vMARP'].astype(bool)
    groups = RP_summary.groupby('group')

    for group_id, group_df in groups:
        if group_df.iloc[0, group_df.columns.get_loc('is_duplicate')]:
            if group_df.iloc[0, group_df.columns.get_loc('is_MIRP')]:
                index_of_min_value = group_df.iloc[:, 0].idxmin()
                A = (pd.DataFrame(group_df.index == index_of_min_value)).to_numpy()
                RP_summary.loc[group_df.index, 'is_vMIRP'] = A

            if group_df.iloc[0, group_df.columns.get_loc('is_MARP')]:
                index_of_max_value = group_df.iloc[:, 0].idxmax()
                B = (pd.DataFrame(group_df.index == index_of_max_value)).to_numpy()
                RP_summary.loc[group_df.index, 'is_vMARP'] = B

    RP_summary['is_vRP'] = RP_summary.is_vMIRP | RP_summary.is_vMARP
    false_alarm = RP_summary[RP_summary['is_vRP'] == False]
    RP_vector.loc[false_alarm.index, ['is_MARP', 'is_MIRP']] = False
    RP_summary = RP_summary[RP_summary['is_vRP'] == True]

    return RP_vector, RP_summary


# Function: calculate the duration and return for each identified trend
def calculate_trend_return(RP_summary):
    """
    This function calculates the return and duration per upward / downward trend.

    :param RP_summary: the list of (only) the validated reflection points.
    :return RP_summary_extended: the RP_summary extended with returns and duration.
    """

    RP_summary['return'] = RP_summary.iloc[:, 0].pct_change()
    RP_summary.loc[RP_summary['is_MIRP'], 'return_type'] = 'loss'
    RP_summary.loc[RP_summary['is_MARP'], 'return_type'] = 'gain'
    RP_summary['duration'] = pd.to_datetime(RP_summary.index).to_series().diff().dt.days
    return RP_summary


def drop_irregular_RP(RP_summary):
    RP_summary['is_irregular'] = ((RP_summary['is_vMIRP']) & (RP_summary['return'] > 0)) | (
                (RP_summary['is_vMARP']) & (RP_summary['return'] < 0))
    RP_summary['is_irregular'] = RP_summary['is_irregular'] | RP_summary['is_irregular'].shift(-1)
    RP_summary = RP_summary.loc[~RP_summary['is_irregular']]
    return RP_summary


def assign_trend(RP_vector, RP_summary):
    # Now identify the trend by assigning value 1 for upward trend and 0 for downward.
    RP_vector['is_upward_trend'] = False
    for start_date, end_date in zip(RP_summary.index, RP_summary.index[1:]):
        RP_vector.loc[start_date:end_date, 'is_upward_trend'] = RP_summary.loc[start_date, 'is_vMIRP']
    return RP_vector, RP_summary


# Function: plot the price curve with indication of trends
def trend_plot_curve(RP_vector, RP_summary, window_in_days):
    """
    This function plots the entire price curve with indication of each identified upward / downward trend.
    :param RP_vector: the location (i.e. date) of the validated reflection points, together with the full price vector.
    :param RP_summary: the list of (only) the validated reflection points.
    :return pop-up figure
    """
    MIRP_dates = RP_summary[RP_summary['is_MIRP']].index.tolist()
    MARP_dates = RP_summary[RP_summary['is_MARP']].index.tolist()
    # plt.plot(local_minimum, color='g')
    # plt.plot(local_maximum, color='r')
    plt.figure(figsize=(10, 6))
    plt.plot(RP_vector.iloc[:, 0])
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    plt.xticks(rotation=45)
    plt.legend(["price curve", "MIRP, window: " + str(window_in_days), "MARP, window: " +
                str(window_in_days)], loc="upper left")
    for date in MIRP_dates:
        plt.axvline(x=date, color='g', linestyle='--')
    for date in MARP_dates:
        plt.axvline(x=date, color='r', linestyle='--')
    plt.xlabel('date')
    plt.ylabel('price')

    # Use AutoDateLocator to automatically choose the date intervals
    locator = AutoDateLocator(maxticks=20)  # Adjust the maxticks value as needed
    plt.gca().xaxis.set_major_locator(locator)
    plt.tight_layout()
    # plt.show()


# Function: give a scatter plot of the durations and returns for the trends
def trend_plot_scatter(RP_summary, window_in_days):
    """
    This function gives a scatter plot of the durations and returns for all the identified upward / downward trends.
    :param RP_vector: the location (i.e. date) of the validated reflection points, together with the full price vector.
    :param RP_summary: the list of (only) the validated reflection points.
    :return pop-up figure
    """
    trend_summary = RP_summary[['duration', 'return', 'return_type']].dropna()
    for return_type in trend_summary['return_type'].unique():
        x_values = trend_summary.loc[trend_summary['return_type'] == return_type, 'duration']
        y_values = trend_summary.loc[trend_summary['return_type'] == return_type, 'return']
        plt.scatter(x_values, y_values, label=return_type, alpha=0.3, edgecolors='none')
    plt.xlabel('duration')
    plt.ylabel('returns in the trend')
    plt.legend()
    plt.grid(True)
    plt.title(f"window: {window_in_days}")
    # plt.show()


# Function: give a scatter plot of the durations and returns for the trends
def trend_plot_hist(RP_summary, number_bins=20):
    # Create a figure and two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the histograms for column1 and column2 in the subplots
    axs[0].hist(RP_summary.loc[RP_summary['return_type'] == 'gain', 'return'], bins=number_bins, edgecolor='black')
    axs[1].hist(RP_summary.loc[RP_summary['return_type'] == 'loss', 'return'], bins=number_bins, edgecolor='black')

    # Add labels and titles to the subplots
    axs[0].set_xlabel('Gain')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Histogram of gains')

    axs[1].set_xlabel('Loss')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of losses')

    # Display the subplots
    plt.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the histograms for column1 and column2 in the subplots
    axs2[0].hist(RP_summary.loc[RP_summary['return_type'] == 'gain', 'duration'], bins=number_bins, edgecolor='black')
    axs2[1].hist(RP_summary.loc[RP_summary['return_type'] == 'loss', 'duration'], bins=number_bins, edgecolor='black')

    # Add labels and titles to the subplots
    axs2[0].set_xlabel('Duration in days')
    axs2[0].set_ylabel('Frequency')
    axs2[0].set_title('Histogram of gains')

    axs2[1].set_xlabel('Duration in days')
    axs2[1].set_ylabel('Frequency')
    axs2[1].set_title('Histogram of losses')

    # Display the subplots
    plt.tight_layout()
    plt.show()


def trend_identification_main(price_raw, is_month_average=False, window_in_days=63):

    if is_month_average:
        price = ut.calculate_monthly_average(pd.DataFrame(price_raw))
        window = window_in_days // 21
    else:
        price = pd.DataFrame(price_raw)
        window = window_in_days

    RP_vector, RP_summary = find_RP(price, window)
    RP_summary = calculate_trend_return(RP_summary)
    RP_summary = drop_irregular_RP(RP_summary)
    RP_summary = calculate_trend_return(RP_summary)
    RP_summary = drop_irregular_RP(RP_summary)
    RP_vector, RP_summary = assign_trend(RP_vector, RP_summary)
    return RP_vector, RP_summary

