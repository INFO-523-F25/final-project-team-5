from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

"""
Contains utility functions for creating, training, and testing
a gradient boost regression model.
"""

def gradient_boost(df, y, n_estimators, random_state):
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