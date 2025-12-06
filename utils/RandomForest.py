import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


stock_data = pd.read_csv('data/stock_data.csv')
unemployment = pd.read_csv('data/SeriesReport.csv')
#print('stock_data (head):')
#print(stock_data.head())   

# Converting the Observation Date Variable to a Datetime Variable
stock_data['dt'] = pd.to_datetime(stock_data['dt'])

# Unpivoting the Unemployment Data
unemployment_unpivot = unemployment.melt(id_vars='Year', var_name='Month', value_name='Unemployment Percent')

# Extracting the Year and Month from the Observation Date
stock_data['Year'] = stock_data['dt'].dt.year
stock_data['Month'] = stock_data['dt'].dt.month

# Replacing the Month Words with Month Numbers
month_replacement = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# Apply the mapping
unemployment_unpivot['Month'] = unemployment_unpivot['Month'].map(month_replacement)

# Merging the two DataFrames Together
stock_data_final = pd.merge(stock_data, unemployment_unpivot, on = ['Year', 'Month'], how = 'left')

#RANDOM FOREST REGRESSOR MODEL
# Defining Feature Columns and Target Variable
feature_cols = [
    'sp500',
    'sp500_volume',
    'djia',
    'hsi',
    'ads',
    'us3m',
    'joblessness',
    'epu',
    'GPRD',
    'Unemployment Percent'
]

x=stock_data_final[feature_cols]
y=stock_data_final['vix']

# Splitting the Data into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Initializing and Training the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
# Evaluating the Model
rf_mse = mean_squared_error(y_test, y_pred)
rf_r2 = r2_score(y_test, y_pred)
print(f'Random Forest MSE: {rf_mse}')
print(f'Random Forest R^2: {rf_r2}')

