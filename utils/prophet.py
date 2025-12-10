'''
Contains the utility function necessary for creating the Prophet model.
For reference, I used the following websites to help me build this model:
https://facebook.github.io/prophet/docs/quick_start.html#python-api
https://facebook.github.io/prophet/docs/diagnostics.html
'''

from prophet import Prophet
from prophet.diagnostics import performance_metrics, register_performance_metric, rolling_mean_by_h
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import logging

def prophet_model(df, periods, cv_initial, cv_period, cv_horizon):
    '''
    Initializes a prophet model, fits the model to the DataFrame,
    predicts future results, performs cross-validation, returns performance 
    metrics on the model, and plots the historic and future results.

    Parameters
    ------------
    df (pd.DataFrame):
        The DataFrame that the model will be fit on.
    periods (int):
        The number of days that the model will forecast out on.
    cv_initial (str):
        The size of the initial traning period in days.
    cv_period (str):
        The amount of spacing between cutoff periods in days.
    cv_horizon (str):
        The number of days for the forecast horizon.
    Results
    -----------
    Mean Squared Error (str):
        A string value containing the Mean Squared Error for the entire model.
    R-Squared (str):
        A string value containing the R-Squared value for the entire model.
    df_performance_metrics (pd.DataFrame):
        A DataFrame containing performance metrics, including the Mean Squared Error,
        Root Mean Squared Error, Mean Average Error, Mean Absolute Error, Mean Absolute
        Percentage Error, Median Absolute Percentage Error, and Coverage.
    matplotlib.figure.Figure:
        The trend results from the prophet model.
    '''
    # Removing cmdstanpy messages from the output
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

    # Renaming the columns per the Prophet Model's Requirements
    df = df.rename(columns={'dt': 'ds', 'vix': 'y'})

    # Initializing a Prophet Model
    model = Prophet()

    # Fitting the Model to the DataFrame
    model.fit(df)

    # Initializing a Variable for a Dataframe that predicts future results
    future = model.make_future_dataframe(periods = periods)

    # Making Predictions based on the Model
    forecast = model.predict(future)

    # Plotting the Model's Results
    model.plot(forecast)

    # Initializing a Variable to perform Cross Validation on the Model
    df_cv = cross_validation(model, initial =  cv_initial, period= cv_period, horizon = cv_horizon)

    # Initializing a Variable to Calculate Performance Metrics for the Prophet Model
    df_performance_metrics = performance_metrics(df_cv)

    # Merge 'y' values from the original 'df' using 'ds' as the key
    forecast = pd.merge(forecast, df[['ds', 'y']], on='ds', how='left')

    # Filtering the DataFrame on Historical Data for MSE and R-Squared Purposes
    historical_forecast = forecast[forecast['y'].notna()]

    # Calculating Mean Squared Error on the Entire Model
    mse = mean_squared_error(historical_forecast['y'], historical_forecast['yhat'])

    # Returning the Mean Squared Error on the Entire Model
    print(f'Mean Squared Error: {mse:.4f}')

    # Calculating R-Squared on the Entire Model
    r2 = r2_score(historical_forecast['y'], historical_forecast['yhat'])

    # Returning the R-Sqared Score for the Entire Model
    print(f'R-Squared: {r2:.4f}')

    # Returning the Performance Metrics an Mean Squared Error
    print('Performance Metrics:')
    return df_performance_metrics
