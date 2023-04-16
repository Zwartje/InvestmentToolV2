import configparser
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the config.ini file
config.read('config.ini')

# Initialize an empty dictionary to store the keys and their values
section_data = []

# Iterate over all options in the specified section
for key, value in config.items('fred_data_label'):
    # Split the comma-separated values into a list
    values = [v.strip() for v in value.split(',')]
    section_data.extend(values)


start_date = config['trend_identification']['start_date']
end_date = pd.to_datetime('today')

# Download data for each series ID and store it in a dictionary
data_dict = {}
for series_id in section_data:
    data = web.DataReader(series_id, 'fred', start_date, end_date)
    data_dict[series_id] = data[series_id]

frequencies = {}
for series_id in section_data:
    df = pd.DataFrame(data_dict[series_id])
    frequencies[series_id] = pd.infer_freq(df.dropna().index)

# Merge data into a single DataFrame using the 'date' index
# df = pd.DataFrame(data_dict)
# df.index.name = 'date'
# df.reset_index(inplace=True)