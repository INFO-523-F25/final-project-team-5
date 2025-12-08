import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import numpy as np

print("ADF on original VIX:")
print(adf_test(vix))
#Check Stationarity using ADF - check if stationary or not, stationary if p-value < 0.05
# First difference for d=1 candidate to make stationary if needed
vix_diff = vix.diff()
print("\nADF on first-differenced VIX:")
print(adf_test(vix_diff))

#Train Test Split
split_idx = int(len(vix) * 0.8)
vix_train = vix.iloc[:split_idx]
vix_test = vix.iloc[split_idx:]

#ARIMA Model
model = ARIMA(vix_train, order=(1, 0, 1))
results = model.fit()
print(results.summary().as_text()) 
#Forecast
steps = len(vix_test)
forecast_res = results.get_forecast(steps=steps)
vix_pred = forecast_res.predicted_mean
vix_pred.index = vix_test.index


plt.figure(figsize=(12, 5))

# Plot actual test set values
plt.plot(vix_test.index, vix_test, label='Actual VIX', linewidth=2)

# Plot predicted values
plt.plot(vix_pred.index, vix_pred, label='Predicted VIX (ARIMA)', linestyle='--')

# Add title and labels
plt.title("VIX Forecast vs Actual (Test Data Portion)")
plt.xlabel("Date")
plt.ylabel("VIX Level")

# Add legend
plt.legend()

# Format layout
plt.tight_layout()
plt.show()

# Evaluate ARIMA(1,0,1) forecast on the test set
manual_order = (1, 0, 1)  # just for printing

manual_mae = mean_absolute_error(vix_test, vix_pred)
manual_rmse = np.sqrt(mean_squared_error(vix_test, vix_pred))

print(f"\nARIMA MAE:  {manual_mae:.3f}")
print(f"ARIMA RMSE: {manual_rmse:.3f}")