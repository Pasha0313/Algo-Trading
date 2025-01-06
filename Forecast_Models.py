from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,GRU
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def get_model_function(model_name):
    model_functions = {
        "ARIMA": arima_model,
        "Prophet": prophet_model,
        "Exponential_Smoothing": exponential_smoothing_model,
        "XGBoost": xgboost_model,
        "Gaussian": gaussian_process_model,
        "LSTM": lstm_model,
        "GRU": gru_model
    }
    return model_functions.get(model_name, None)

##################################################
# ARIMA Model
##################################################
def arima_model(train_data, **params):
    p, d, q = params.get("p",2), params.get("d",1), params.get("q",2)
    trend = params.get("trend", "n")  
    if d > 0 and trend in ['c', 'ct']:  
        trend = 'n'  
    model = ARIMA(train_data, order=(p, d, q), trend=trend)
    fitted_model = model.fit()
    return fitted_model

##################################################
# XGBoost Model
##################################################
def xgboost_model(X_train, y_train, **params):
    n_estimators = int(params.get("n_estimators", 100))
    max_depth = int(params.get("max_depth", 3))
    learning_rate = float(params.get("learning_rate", 0.1))  
    model = XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
    model.fit(X_train, y_train)
    return model

def forecast_with_xgboost(model, last_known_data, steps, lag_features):
    predictions = []
    current_input = last_known_data[-1].reshape(1, -1)
    for _ in range(steps):
        pred = model.predict(current_input)[0]
        predictions.append(pred)
        current_input = np.roll(current_input, -1)  
        current_input[0, -1] = pred  
    return predictions

############################################################################################
############################################################################################
def prepare_features(data, lags=3):
    df = pd.DataFrame({'t': data})
    for lag in range(1, lags + 1):
        df[f't-{lag}'] = df['t'].shift(lag)
    df = df.dropna()
    X = df.iloc[:, 1:].values  # Lag features
    y = df.iloc[:, 0].values  # Current value
    return np.array(X), np.array(y)

def stacking_regressor_model(X_train, y_train, **params):
    estimators = params.get("estimators", [
        ('xgb', XGBRegressor()),
        ('rf', RandomForestRegressor())
    ])
    final_estimator = params.get("final_estimator", LinearRegression())
    n_jobs = params.get("n_jobs", -1)

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    return model

