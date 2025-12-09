import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

"""
Contains utility functions for creating, training, and testing
a LSTM model.

I used the following webiste for assistance on building the LSTM
model: https://medium.com/@techwithjulles/recurrent-neural-networks-rnns-and-long-short-term-memory-lstm-creating-an-lstm-model-in-13c88b7736e2
"""

def lstm_function(df, y):
    '''
    Initializes an LSTM model, trains the model on a training dataset, tests
    the model on a testing dataset, and returns the Mean Squared Error and R-Squared
    Values.

    Parameters
    ------------
    df (pd.DataFrame):
        The DataFrame that the LSTM model will be applied to.
    y (pd.Series):
        The response variable of the model.

    Results
    ------------
    results (str):
        The Mean-Squared and R-Squared results of the LSTM model.
    '''
    # Initializing a Variable for the Predictor Columns
    predictor_cols = df.columns.drop(y)

    # Initializing MinMaxScalers for Predictor and Response Variables
    predictor_scaler = MinMaxScaler()
    response_scaler = MinMaxScaler()

    X_all = predictor_scaler.fit_transform(df[predictor_cols])
    y_all = response_scaler.fit_transform(df[[y]]) 

    # Creating a one-step lag: current features -> next-step target
    X = X_all[:-1]
    y = y_all[1:]

    # Splitting the Data into Training and Testing Datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshaping for LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Creating the LSTM Model
    model = Sequential([LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                        Dense(1)])
    
    # Compiling the Model
    model.compile(optimizer='adam', loss='mse')

    # Fitting the Model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                        validation_split=0.2, verbose=0)

    # Evaluating the Model
    loss = model.evaluate(X_test, y_test)

    # Predicting the Model
    y_pred = model.predict(X_test)
    y_pred_inv = response_scaler.inverse_transform(y_pred)
    y_test_inv = response_scaler.inverse_transform(y_test)

    # Calculating the Mean Squared Error and R-Squared Value
    testScore = mean_squared_error(y_test_inv.ravel(), y_pred_inv.ravel())

    # Calculating the R-Squared Value
    r2_val = r2_score(y_test_inv.ravel(), y_pred_inv.ravel())

    # Returning the Results
    result = f'Mean Squared Error: {float(testScore)}\nR-Squared: {float(r2_val)}'
    
    return result
