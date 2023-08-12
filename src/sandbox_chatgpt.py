import pandas as pd

# Create example dataframes with date index and varying lengths
df1 = pd.DataFrame({'RiskDriver1': [1.0, 2.0]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
df2 = pd.DataFrame({'RiskDriver2': [4.0, 5.0, 6.0]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
df3 = pd.DataFrame({'RiskDriver3': [7.0, 8.0]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
target_df = pd.DataFrame({'Target': [0, 1, 1]}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

# Combine risk drivers and target into a single dataframe
dataframes = [df1, df2, df3, target_df]
combined_df = pd.concat(dataframes, axis=1, join='outer')
