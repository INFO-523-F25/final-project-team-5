import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def lasso_regression(df, y, alpha):
    # Initializing a variable for the predictor variables
    X = df.drop(y, axis = 1)

    y = df[y]

    # Splitting the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Initializing the Lasso Regression Model
    lasso_reg = Lasso(alpha = 1.0)

    # Fitting the Model
    lasso_reg.fit(X_train, y_train)

    # Predicting the Model on the Testing Set
    y_pred_lasso = lasso_reg.predict(X_test)

    # Calculating the Mean Squared Error
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)

    # Calcuating the R-Squared Value
    r2_lasso = r2_score(y_test, y_pred_lasso)

    # Initializing a Variable for the MSE and R-Squared Results
    results = f'Mean Squared Error: {mse_lasso} <br> R-Squared: {r2_lasso}'
    
    # Returning the Results
    return results
