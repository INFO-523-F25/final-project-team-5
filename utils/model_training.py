from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
def batch_ridge(models: dict[Ridge, tuple[np.array,  np.array, str]], y_train, y_test):
    '''
    Runs batchs of Ridge Regression models

    Parameters
    ----------
    models : dict
        Dictionary that contains the model as a key and a tuple for the value. The tuple 
        has 3 items that represent the following (X_train, X_test descriptor)
    y_train : array-like
      Train target data
    y_test : array-like
      Test target data
    '''
    print('Begining Batch Model Traing')
    print('------------------------------------------------')
    for model, params in models.items():
        X_train = params[0]
        X_test = params[1]
        desc = params[2]
        print(f'Starting training for {desc}')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'Model results for {desc}')
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE:", mse)
        print("R^2:", model.score(X_test, y_test))
        print('---------------------------------------------')


def random_forest(x_train, x_test, y_train, y_test, param_dist=None, cv=5, n_iter=25, random_state=42):
    """
    Train a Random Forest regressor with optional hyperparameter tuning using RandomizedSearchCV.

    Parameters
    ----------
    x_train : np.ndarray or pd.DataFrame
        Training features.
    x_test : np.ndarray or pd.DataFrame
        Test features.
    y_train : np.ndarray or pd.Series
        Training target.
    y_test : np.ndarray or pd.Series
        Test target.
    param_dist : dict, optional
        Hyperparameter search space for RandomizedSearchCV. If None, a default
        distribution is used.
    cv : int, optional
        Number of cross-validation folds for hyperparameter search.
    n_iter : int, optional
        Number of parameter settings sampled in RandomizedSearchCV.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    best_model : RandomForestRegressor
        Best estimator found by RandomizedSearchCV (or the default model if no search).
    y_pred : np.ndarray
        Predictions on the test set.
    """
    # Base model
    base_rf = RandomForestRegressor(random_state=random_state)

    # Default hyperparameter search space if none provided
    if param_dist is None:
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.8],
            "bootstrap": [True, False],
        }

    # Hyperparameter tuning
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    print("Best parameters found:")
    print(search.best_params_)

    # Evaluate on test set
    y_pred = best_model.predict(x_test)
    rf_mse = mean_squared_error(y_test, y_pred)
    rf_r2 = r2_score(y_test, y_pred)

    print(f"Random Forest MSE: {rf_mse:.4f}")
    print(f"Random Forest R^2: {rf_r2:.4f}")
