import data_sourcer as ds
import trend_identification as trend
import utils as ut
from configparser import ConfigParser, ExtendedInterpolation
import pandas as pd
import os
import matplotlib.pyplot as plt


# read configuration
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('config.ini')

# The following lines are used for testing the trend identification and plotting.
start_date = config['trend_identification']['start_date']
end_date = config['trend_identification']['end_date']
# code = config['trend_identification']['code']
code_list_str = config['trend_identification']['code_list']
code_list = [item.strip() for item in code_list_str.split(',')]
window_in_days = int(config['local_extreme']['window_in_days'])
window_in_days_list_str = config['local_extreme']['window_in_days_list']
window_in_days_list = [int(item.strip()) for item in window_in_days_list_str.split(',')]

current_folder = os.path.dirname(os.path.abspath(__file__))
figure_folder = os.path.join(current_folder, '..', 'output', 'figures')
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

for code in code_list:
    price_raw = ds.download_stock_price_daily_close(code, start_date, end_date)
    RP_vector, RP_summary = trend.trend_identification_main(price_raw, False, window_in_days)
    trend.trend_plot_curve(RP_vector, RP_summary, window_in_days)
    fig_file_name = ut.remove_illegal_symbols(code)
    plt.savefig(os.path.join(figure_folder, f"curve_{fig_file_name}_{window_in_days}D.png"))
    plt.close()

    # Create a 2x2 subplot grid for scatter plots
    fig, axs = plt.subplots(2, 2, figsize=[10, 8])
    # Generate figures and plot them in the subplots
    for i, ax in enumerate(axs.flatten(), 1):
        RP_vector, RP_summary = trend.trend_identification_main(price_raw, False, window_in_days_list[i-1])
        trend.trend_plot_scatter(RP_summary, window_in_days_list[i-1])
        plt.sca(ax)
    # Adjust the layout to prevent overlapping titles
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(figure_folder, f"scatter_{fig_file_name}.png"))
    plt.close()