'''
Contains the utility function necessary for creating the Prophet model.
For reference, I used the following website to help me build this model:
https://facebook.github.io/prophet/docs/quick_start.html#python-api
'''

from prophet import Prophet

def prophet_model(df, periods):
    '''
    Initializes a prophet model, fits the model to the DataFrame,
    predicts future results, and plots the historic and future results.

    Parameters
    ------------
    df (pd.DataFrame):
        The DataFrame that the model will be fit on.
    date (pd.Series):
        The name of the column that the model will use for its date column.
    y (pd.Series):
        The name of the column that the mdoel will use for its response variable.
    periods (int):
        The number of periods that the model will forecast out on.

    Results
    -----------
    matplotlib.figure.Figure:
        The trend results from the prophet model in a line graph.
    '''
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