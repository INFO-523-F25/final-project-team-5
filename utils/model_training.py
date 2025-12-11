from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
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
