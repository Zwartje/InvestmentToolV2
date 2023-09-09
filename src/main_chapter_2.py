import configparser
import pandas as pd
import pandas_datareader.data as web
import data_sourcer as ds
import trend_identification as trend
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Specifically to suppress the warning "A value is trying to be set on a copy of a slice from a DataFrame."
pd.options.mode.chained_assignment = None  # 'warn', 'raise', None

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

# Download data for each series ID and store it in a list of EconomicVariables objects
economic_variables_list = []
for series_id in tqdm(section_data, desc="Processing variables in the series ID list"):
    data_raw = web.DataReader(series_id, 'fred', start_date, end_date)
    data, frequency = ds.identify_data_frequency(data_raw)
    data_resampled = ds.resample_to_last_day_in_month(data)
    assert frequency == 30, f"Please check the variable {series_id}; it may not have the monthly frequency!"
    variable = ds.EconomicVariables(data_resampled, series_id, frequency, series_id[:2])
    variable.generate_yoy_change()
    variable.generate_running_percentile(window_size=36)
    economic_variables_list.append(variable)

# Prepare the data for logistic regression
risk_drivers_list = []
for variable in tqdm(economic_variables_list, desc="Collecting variables as risk drivers"):
    risk_drivers_list.append(variable.data)
    # risk_drivers_list.append(variable.data_yoy)
    risk_drivers_list.append(variable.data_r_p)

risk_drivers_df = pd.concat(risk_drivers_list, axis=1, join='inner')

# Next, process the dependent variable
code = config['trend_identification']['code']
window_in_days = int(config['local_extreme']['window_in_days'])
price_raw = ds.download_stock_price_daily_close(code, start_date, end_date)
price_raw_monthly = ds.calculate_monthly_average(price_raw) # Here we work with monthly data!
price_raw_monthly_resampled = ds.resample_to_last_day_in_month(price_raw_monthly)
RP_vector, RP_summary = trend.trend_identification_main(price_raw_monthly_resampled, True, window_in_days)

# Now, extend the df and prepare for logistic regression
risk_drivers_list.append(pd.DataFrame(RP_vector.is_upward_trend).astype(int))
regression_dataset_df = pd.concat(risk_drivers_list, axis=1, join='inner')
predictors = regression_dataset_df.drop('is_upward_trend', axis=1)
target = regression_dataset_df['is_upward_trend']

# Regression
model = LogisticRegression(max_iter=100000)
model.fit(predictors, target)
predictions = model.predict(predictors)
predictions_series = pd.Series(predictions, index=target.index)

# Evaluation
report = metrics.classification_report(target, predictions)
print("Classification Report:\n", report)

logits = model.decision_function(predictors)
y_pred_proba = model.predict_proba(predictors)[::,1]
fpr, tpr, _ = metrics.roc_curve(target,  y_pred_proba)
auc = metrics.roc_auc_score(target, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

plt.figure()
plt.scatter(logits, y_pred_proba, alpha=0.3)
plt.scatter(logits, target, alpha=0.3)



