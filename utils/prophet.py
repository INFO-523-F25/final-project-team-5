'''
Contains the utility function necessary for creating the Prophet model.
For reference, I used the following website to help me build this model:
https://facebook.github.io/prophet/docs/quick_start.html#python-api
'''

from prophet import Prophet

def prophet_model(df, periods):
    # Initializing a Prophet Model
    model = Prophet()

    # Fitting the Model to the DataFrame
    model.fit(df)

    # Initializing a Variable for a Dataframe that predicts future results
    future = model.make_future_dataframe(periods = periods)

    # Predicting Future Results
    forecast = model.predict(future)

    # Plotting the Results
    return model.plot(forecast)