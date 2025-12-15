import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def adf_test(series):
    """
    Run the Augmented Dickey-Fuller test on a time series and print results.

    Parameters
    ----------
    series : pd.Series
        Time series to test for stationarity. NaN values should be handled
        prior to passing into this function.

    Prints
    ------
    ADF Statistic : float
        The test statistic returned by adfuller.
    p-value : float
        The probability of observing the test statistic under the null hypothesis.
    Critical Values : dict
        Threshold values for rejecting the null hypothesis at various confidence levels.
    """
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")


def arima(vix_train, vix_test):
    """
    Fit a statsmodels ARIMA(1,0,1) model to training data and forecast over the test set.

    Parameters
    ----------
    vix_train : pd.Series
        Training portion of the VIX time series.
    vix_test : pd.Series
        Test portion of the VIX time series.

    Returns
    -------
    vix_pred : pd.Series
        Forecasted values aligned to the index of vix_test.
    fittedvalues : pd.Series
        In-sample fitted values from the ARIMA model (same index as vix_train).
    """
    model = ARIMA(vix_train, order=(1, 0, 1))
    results = model.fit()
    print(results.summary().as_text())

    # Forecast over the test period
    steps = len(vix_test)
    forecast_res = results.get_forecast(steps=steps)
    vix_pred = forecast_res.predicted_mean
    vix_pred.index = vix_test.index

    return vix_pred, results.fittedvalues


def plot_arima(vix_train, vix_test, vix_pred, vix_fitted):
    """
    Plot ARIMA model results, including:
    - Actual vs fitted values for the training period
    - Actual vs forecasted values for the test period

    Parameters
    ----------
    vix_train : pd.Series
        Training portion of the VIX series.
    vix_test : pd.Series
        Test portion of the VIX series.
    vix_pred : pd.Series
        Forecasted values from the ARIMA model corresponding to vix_test.
    vix_fitted : pd.Series
        In-sample fitted values returned from the ARIMA model.

    Produces
    --------
    Two vertically stacked subplots:
        (1) Train actual vs fitted
        (2) Test actual vs predicted
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    # Plot Train vs Fitted
    axes[0].plot(vix_train.index, vix_train, label='Actual (Train)', linewidth=2)
    axes[0].plot(vix_fitted.index, vix_fitted, label='Fitted (ARIMA)', linestyle='--')
    axes[0].set_title("VIX: Actual vs Fitted (Train Portion)")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("VIX Level")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Test vs Forecast
    axes[1].plot(vix_test.index, vix_test, label='Actual (Test)', linewidth=2)
    axes[1].plot(vix_pred.index, vix_pred, label='Predicted (ARIMA)', linestyle='--')
    axes[1].set_title("VIX Forecast vs Actual (Test Portion)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("VIX Level")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def arima_eval(vix_test, vix_pred):
    """
    Evaluate ARIMA forecast performance using MAE and RMSE.

    Parameters
    ----------
    vix_test : pd.Series
        Actual observed VIX values for the test set.
    vix_pred : pd.Series
        Predicted VIX values from the ARIMA model.

    Prints
    ------
    MAE : float
        Mean Absolute Error of the forecast.
    RMSE : float
        Root Mean Squared Error of the forecast.
    """
    manual_mae = mean_absolute_error(vix_test, vix_pred)
    manual_rmse = np.sqrt(mean_squared_error(vix_test, vix_pred))

    print(f"\nARIMA MAE:  {manual_mae:.3f}")
    print(f"ARIMA RMSE: {manual_rmse:.3f}")



# #=======ARIMA model (auto_arima)=========

def decompose_series(series: pd.Series, period: int = 252, model: str = "additive", show: bool = True):
    """
    Perform seasonal decomposition on a time series.

    Parameters
    ----------
    series : pd.Series
        Time series with a DatetimeIndex.
    period : int
        Seasonal period (e.g., 252 ~ trading days in a year).
    model : {"additive", "multiplicative"}
        Type of decomposition.
    show : bool
        If True, plot the decomposition.

    Returns
    -------
    DecomposeResult
        statsmodels decomposition result.
    """
    decomposition = seasonal_decompose(series, model=model, period=period)

    if show:
        decomposition.plot()
        plt.suptitle("VIX Decomposition", y=1.02)
        plt.tight_layout()
        plt.show()

    return decomposition

def fit_auto_arima_model(series: pd.Series,
                         start_p: int = 1,
                         start_q: int = 1,
                         max_p: int = 3,
                         max_q: int = 3,
                         seasonal: bool = False,
                         m: int = 1,
                         trace: bool = False):
    """
    Fit a non-seasonal ARIMA model using pmdarima's auto_arima.

    Parameters
    ----------
    series : pd.Series
        Time series to model.
    start_p, start_q, max_p, max_q : int
        AR and MA order search bounds.
    seasonal : bool
        Whether to fit a seasonal ARIMA.
    m : int
        Seasonal period; set to 1 for non-seasonal.
    trace : bool
        If True, print model search progress.

    Returns
    -------
    pmdarima.arima.arima.ARIMA
        Fitted auto_arima model.
    """
    model = auto_arima(
        series,
        start_p=start_p,
        start_q=start_q,
        max_p=max_p,
        max_q=max_q,
        m=m,
        d=None,               # let auto_arima decide differencing via ADF test
        seasonal=seasonal,
        test="adf",
        start_P=0,
        D=0,
        trace=trace,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    return model


def get_in_sample_fitted(model, index: pd.DatetimeIndex):
    """
    Get in-sample fitted values from a pmdarima model.

    Parameters
    ----------
    model : pmdarima model
        Fitted auto_arima model.
    index : pd.DatetimeIndex
        Index to align fitted values with.

    Returns
    -------
    pd.Series
        In-sample fitted series indexed by `index`.
    """
    fitted_values = model.predict_in_sample()
    return pd.Series(fitted_values, index=index)


def plot_fitted(series: pd.Series, fitted: pd.Series, title: str = "VIX: Original vs Fitted (auto_arima)"):
    """
    Plot original series vs in-sample fitted values.

    Parameters
    ----------
    series : pd.Series
        Original series.
    fitted : pd.Series
        In-sample fitted values aligned with `series.index`.
    title : str
        Plot title.
    """
    plot_df = pd.DataFrame({
        "actual": series,
        "fitted": fitted
    })

    plot_df.plot(figsize=(12, 5), title=title)
    plt.ylabel("VIX Level")
    plt.tight_layout()
    plt.show()


def forecast_future(model,
                    last_index: pd.Timestamp,
                    n_periods: int = 30,
                    freq: str = "B") -> tuple[pd.Series, pd.DataFrame]:
    """
    Forecast future values from a fitted pmdarima model.

    Parameters
    ----------
    model : pmdarima model
        Fitted auto_arima model.
    last_index : pd.Timestamp
        Last timestamp of the historical series.
    n_periods : int
        Number of future periods to forecast.
    freq : str
        Frequency string for pd.date_range (e.g., "B" for business days).

    Returns
    -------
    forecast_series : pd.Series
        Forecasted values indexed by future dates.
    conf_int_df : pd.DataFrame
        Confidence intervals with columns ["lower", "upper"].
    """
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    future_index = pd.date_range(
        start=last_index + pd.Timedelta(days=1),
        periods=n_periods,
        freq="B"
    )
    print(forecast)
    # Correct: set index directly when creating the Series
    forecast_series = forecast.copy()
    forecast_series.index = future_index
    conf_int_df = pd.DataFrame(conf_int, index=future_index, columns=["lower", "upper"])

    return forecast_series, conf_int_df


def plot_forecast(history, forecast_series, conf_int_df, 
                  title="VIX Forecast (auto_arima)", history_days=250):
    """
    Plot the most recent portion of a historical time series along with
    ARIMA forecast values and confidence intervals.

    This function trims the historical series to the last `history_days`
    observations so that the forecast is visually interpretable. When plotting
    the full multi-decade history, a short forecast (e.g., 30 business days)
    becomes too compressed to see. Zooming into the tail ensures that both
    the historical data and forecast appear clearly on the same scale.

    Parameters
    ----------
    history : pd.Series
        Full historical time series (e.g., VIX levels), indexed by datetime.
    forecast_series : pd.Series
        Forecasted values from the ARIMA model, indexed by future timestamps.
    conf_int_df : pd.DataFrame
        Confidence interval bounds for the forecast. Must contain
        columns 'lower' and 'upper' aligned to `forecast_series.index`.
    title : str, optional
        Title of the plot. Default is "VIX Forecast (auto_arima)".
    history_days : int, optional
        Number of trailing historical observations to include in the plot.
        Default is 250 (approximately one trading year).

    Returns
    -------
    None
        Displays a plot showing:
            - recent historical VIX values
            - forecasted values
            - 95% confidence interval shading
    """
    # Show only last N days of historical data
    history_tail = history.iloc[-history_days:]

    plt.figure(figsize=(10, 6))
    plt.plot(history_tail.index, history_tail, label="Historical VIX")
    plt.plot(forecast_series.index, forecast_series, label="Forecast", linestyle="--")

    plt.fill_between(
        forecast_series.index,
        conf_int_df["lower"],
        conf_int_df["upper"],
        alpha=0.3,
        label="95% Confidence Interval"
    )

    plt.title(title)
    plt.ylabel("VIX Level")
    plt.legend()
    plt.tight_layout()
    plt.show()