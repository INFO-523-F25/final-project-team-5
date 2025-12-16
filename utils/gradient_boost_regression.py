from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

"""
Contains utility functions for creating, training, and testing
a gradient boosting regression model.
"""

def gradient_boost(df, y, n_estimators, random_state):
    '''
    Initializes a Gradient Boosting Regression model, trains the model on 
    a training dataset, tests the model on a testing dataset,
    and calculates and returns the Mean Squared Error and R-Squared Values.

    Parameters
    -------------
    df (pd.DataFrame):
        The DataFrame that the mdoel will be trained on. This DataFrame should
        only contain numeric values.
    y (pd.Series):
        The response variable from the DataFrame.
    n_estimators (float):
        The number of boosting stages to be performed.
    random_state (int):
        The seed set for reproducibility within our model.
    
    Returns
    -------------
    results (str):
        The Mean-Squared and R-Squared results of the Gradient Boosting Regression Model.
    '''
    # Initializing a variable for the predictor variables
    X = df.drop(y, axis = 1)

    # Initializing a variable for the response variable
    y = df[y]

    # Splitting the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Initializing a variable for the gradient boost model
    gradient_boost = GradientBoostingRegressor(n_estimators = n_estimators, random_state = random_state)

    # Fitting the Gradient Boost Regression
    gradient_boost.fit(X_train, y_train)
    
    # Predicting the Model on the Testing Set
    y_pred_gradient_boost = gradient_boost.predict(X_test)

    # Calculating the Mean Squared Error
    mse_lasso = mean_squared_error(y_test, y_pred_gradient_boost)

    # Calcuating the R-Squared Value
    r2_lasso = r2_score(y_test, y_pred_gradient_boost)

    # Initializing a Variable for the MSE and R-Squared Results
    results = f'Mean Squared Error: {mse_lasso}\nR-Squared: {r2_lasso}'
    
    # Returning the Results
    return results

def gradient_boost_hyperparameter_tuning(df, y, param_grid, cv, random_state):
    '''
    This function performs hyperparameter tuning on the gradient boosting regression
    model via Grid Search Cross-Valiation, fits the model with the new parameters, 
    and returns the Mean Squared Error and R-Squared values.

    Parameters
    -------------
    df (pd.DataFrame):
        The DataFrame that the mdoel will be trained on. This DataFrame should
        only contain numeric values.
    y (pd.Series):
        The response variable from the DataFrame.
    param_grid (dict):
        The dictionary that holds the values for Grid Search Cross-Validation.
    cv (int):
        The number of cross-validation folds for the Gradient Boosting Regression model.
    random_state (int):
        The seed set for reproducibility within our model.
    
    Returns
    -------------
    results (str):
        The Mean-Squared and R-Squared results of the Gradient Boosting Regression Model.
    '''
    # Initializing a variable for the predictor variables
    X = df.drop(y, axis = 1)

    # Initializing a variable for the response variable
    y = df[y]

    # Splitting the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Initializing a variable for the gradient boost model
    gradient_boost = GradientBoostingRegressor(random_state = random_state)

    # Initializing a Variable for Grid Search
    grid_search = GridSearchCV(gradient_boost,
                               param_grid = param_grid,
                               cv = cv,
                               scoring = 'neg_mean_squared_error',
                               n_jobs = 1)
    
    # Fitting the Model
    grid_search.fit(X_train, y_train)
    
    # Re-Initializing and Fitting the Model with the new Parameters
    gradient_boost_best = gradient_boost.set_params(**grid_search.best_params_)
    gradient_boost_best.fit(X_train, y_train)

    # Returning the Best Parameters from Grid Search
    print(f'Best Parameters: {grid_search.best_params_}')

    # Making New Predictions on the Test Dataset
    y_pred_best = gradient_boost_best.predict(X_test)

    # Calculating the Mean Squared Error
    mse_lasso = mean_squared_error(y_test, y_pred_best)

    # Calcuating the R-Squared Value
    r2_lasso = r2_score(y_test, y_pred_best)

    # Initializing a Variable for the MSE and R-Squared Results
    results = f'Mean Squared Error: {mse_lasso}\nR-Squared: {r2_lasso}'
    
    # Returning the Results
    return results
