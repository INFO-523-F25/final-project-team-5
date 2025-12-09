"""
Contains utility functions for creating, training, and testing
a Lasso Regression Model.
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def lasso_regression(df, y, alpha, random_state):
    '''
    Initializes a Lasso Regression model, trains the model on 
    a training dataset, tests the model on a testing dataset,
    and calculates and returns the Mean Squared Error and R-Squared Values.

    Parameters
    -------------
    df (pd.DataFrame):
        The DataFrame that the mdoel will be trained on. This DataFrame should
        only contain numeric values.
    y (pd.Series):
        The response variable from the DataFrame.
    alpha (float):
        The alpha variable for the Lasso Regression Model.
    random_state (int):
        The seed set for reproducibility within our model.
    
    Returns
    -------------
    results (str):
        The Mean-Squared and R-Squared results of the Lasso Regression Model.
    '''
    # Initializing a variable for the predictor variables
    X = df.drop(y, axis = 1)

    # Initializing a variable for the response variable
    y = df[y]

    # Splitting the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)

    # Initializing the Lasso Regression Model
    lasso_reg = Lasso(alpha = alpha)

    # Fitting the Model
    lasso_reg.fit(X_train, y_train)

    # Predicting the Model on the Testing Set
    y_pred_lasso = lasso_reg.predict(X_test)

    # Calculating the Mean Squared Error
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)

    # Calcuating the R-Squared Value
    r2_lasso = r2_score(y_test, y_pred_lasso)

    # Initializing a Variable for the MSE and R-Squared Results
    results = f'Mean Squared Error: {mse_lasso}\nR-Squared: {r2_lasso}'
    
    # Returning the Results
    return results

def lasso_hyperparameter_tuning(df, y, cv, max_iter, random_state):
    '''
    This function performs hyperparameter tuning on the lasso model,
    fits the model with the new parameters, and returns the Mean Squared 
    Error and R-Squared values.

    Parameters
    -------------
    df (pd.DataFrame):
        The DataFrame that the mdoel will be trained on. This DataFrame should
        only contain numeric values.
    y (pd.Series):
        The response variable from the DataFrame.
    cv (int):
        The number of cross-validation folds for the Lasso Regression model.
    max_iter (int):
        The max number of iterations.
    random_state (int):
        The seed set for reproducibility within our model.
    
    Returns
    -------------
    results (str):
        The Mean-Squared and R-Squared results of the Lasso Regression Model.
    '''
    # Initializing a variable for the predictor variables
    X = df.drop(y, axis = 1)

    # Initializing a variable for the response variable
    y = df[y]

    # Splitting the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)

    # Defining a range of alpha values for Lasso
    alphas = np.logspace(-4, 2, 100)

    # Initializing LassoCV
    lasso_cv = LassoCV(alphas = alphas, cv = cv, max_iter = max_iter, random_state = 42)

    # Fitting the Model
    lasso_cv.fit(X_train, y_train)

    # Re-Initializing the Model and Fitting it with the Optimal Alpha
    best_lasso = Lasso(alpha = lasso_cv.alpha_)
    best_lasso.fit(X_train, y_train)

    # Making New Predictions
    y_pred_best_lasso = best_lasso.predict(X_test)

    # Calculating the Mean Squared Error
    mse_best = mean_squared_error(y_test, y_pred_best_lasso)

    # Calculating the R-Squared Value
    r2_best = r2_score(y_test, y_pred_best_lasso)

    # Initializing a Variable for the MSE and R-Squared Results
    results = f'Mean Squared Error: {mse_best}\nR-Squared: {r2_best}'
    
    # Returning the Results
    return results