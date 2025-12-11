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

#=======ARIMA model (auto_arima)=========

#Decompose the time series
#Period for daily financial data, ~252 trading days ~ 1 year
decomposition = seasonal_decompose(vix, model='additive', period=252)

decomposition.plot()
plt.suptitle("VIX Decomposition", y=1.02)
plt.show()

#ADF Stationary Test
result = adfuller(vix.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Interpretation
if result[1] > 0.05:
    print("Series is NOT stationary (will likely need differencing).")
else:
    print("Series IS stationary (d will likely be 0).")

#ARIMA model
auto_model = auto_arima(
    vix,
    start_p=1, start_q=1,
    test='adf',          # use adftest to find optimal 'd'
    max_p=3, max_q=3,    # maximum p and q
    m=1,                 # frequency of series
    d=None,              # let model determine 'd'
    seasonal=False,      # No Seasonality
    start_P=0,
    D=0,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print("\nBest ARIMA model found by auto_arima:")
print(auto_model.summary())

# Fit ARIMA model 
model = auto_model

vix_fitted = model.predict_in_sample()

vix_fitted = pd.Series(vix_fitted, index=vix.index)

plot_df = pd.DataFrame({
    "vix": vix,
    "vix_fitted": vix_fitted
})

plot_df.plot(
    figsize=(12, 5),
    title='VIX: Original vs Fitted (auto_arima)'
)
plt.ylabel("VIX Level")
plt.show()

# Forecast the next 30 days
n_periods = 30
forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

# Build a future date index
future_index = pd.date_range(
    start=vix.index[-1] + pd.Timedelta(days=1),
    periods=n_periods,
    freq='B'  
)

# Create a Series with forecasted values aligned to future dates
forecast_series = pd.Series(forecast, index=future_index)

# Plot the forecast
plt.figure(figsize=(8, 6))
plt.plot(vix.index, vix, label='Historical VIX')
plt.plot(forecast_series.index, forecast_series, label='Forecast', linestyle='--')
plt.fill_between(
    future_index,
    conf_int[:, 0],
    conf_int[:, 1],
    color='red',
    alpha=0.3,
    label='95% Confidence Interval'
)
plt.title('VIX Forecast (auto_arima)')
plt.legend()
plt.show()

# Detrending using moving average
vix_ma = vix.rolling(window=252).mean()  # ~1 year of trading days
vix_detrended = vix - vix_ma

# Plot detrended data
plt.figure(figsize=(10, 5))
plt.plot(vix_detrended, label='Detrended VIX')
plt.title('Detrended VIX Time Series')
plt.ylabel("Detrended VIX Level")
plt.legend()
plt.show()