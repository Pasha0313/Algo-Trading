from datetime import datetime, timedelta
from tabulate import tabulate
import math

def load_config_from_text(filename):
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip()
    return config

def get_control_settings(config):
    # Extract control settings
    Unsupervised_Learning = config.get("Unsupervised_Learning", "False").lower() == "true"
    Perform_Testing = config.get("Perform_Testing", "False").lower() == "true"
    Print_Data = config.get("Print_Data", "False").lower() == "true"
    Perform_Forecasting = config.get("Perform_Forecasting", "False").lower() == "true"
    Perform_Tuner = config.get("Perform_Tuner", "False").lower() == "true"
    Perform_Trading = config.get("Perform_Trading", "False").lower() == "true"

    return Unsupervised_Learning, Perform_Testing, Print_Data,Perform_Forecasting, Perform_Tuner,Perform_Trading

def print_config_values(symbol, bar_length, leverage, strategy, tc, test_days):
    print("\nIndividual Values:")
    print(f"Symbol: {symbol}")
    print(f"Bar Length: {bar_length}")
    print(f"Leverage: {leverage}")
    print(f"Strategy: {strategy}")
    print(f"Trading Costs: {tc:.5f}")
    print(f"Days: {test_days}")

def prepare_data_from_config(config):
    # Extract and convert values
    symbol = config.get("symbol", "BTCUSDT")
    bar_length = config.get("bar_length", "15m")
    leverage = int(config.get("leverage", 10))
    strategy = config.get("strategy", "RSI")
    tc = float(config.get("tc", -0.00085))
    test_days = int(config.get("test_days", 120))
    metric = config.get("metric", "Sharpe")
    ForecastModelName=config.get("ForecastModelName", "ARIMA") 
    future_forecast_steps=int(config.get("FutureForecastSteps", 50))
    
    # Define time Period
    Today = datetime.utcnow()
    historical_days = timedelta(days=test_days)
    start_date = Today - historical_days
    end_date = Today 

    # Prepare data
    input_data = [
        ["Symbol", symbol],
        ["Bar Length", bar_length],
        ["Leverage", leverage],
        ["Strategy", strategy],
        ["Trading Costs", f"{tc:.5f}"],
        ["Start Date", start_date.strftime('%Y-%m-%d %H:%M')],
        ["End Date", end_date.strftime('%Y-%m-%d %H:%M')],
        ["Metric", metric],
        ["Forecast Model Name", ForecastModelName]
    ]

    # Print table
    print_data_table(input_data)

    return start_date, end_date, symbol, bar_length, leverage, strategy, tc, test_days,metric,ForecastModelName,future_forecast_steps

def prepare_trade_from_config(config):
    # Extract and convert values
    history_days = int(config.get("history_days", 10))  # Default to 120 hours if not set
    trade_hours = int(config.get("trade_hours", 24))  # Default to 120 hours if not set
    minimum_future_trade_value = float(config.get("minimum_future_trade_value", 100))
    trade_value = float(config.get("trade_value", 250.0))
    n_trades = int(config.get("n_trades", 50))
    position = int(config.get("position", 0))
    stop_loss_pct = float(config.get("stop_loss_pct", 0.02))

    # Define time Period
    Today = datetime.utcnow()
    history_days = timedelta(days=history_days)
    loading_from_date = Today - history_days
    trading_period = timedelta(hours=trade_hours)
    stop_trade_date = Today + trading_period
    
    # Prepare trade data
    trade_data = [
        ["Minimum Future Trade Value", f"{minimum_future_trade_value:.2f}"],
        ["Trade Value", f"{trade_value:.2f}"],
        ["Number of Trades", n_trades],
        ["Position", position],
        ["Stop Loss Percentage", f"{stop_loss_pct:.2f}"],
        ["Historical Period", f"{history_days}"],
        ["loading from Date", loading_from_date.strftime('%Y-%m-%d %H:%M')],
        ["Trade Start Date", Today.strftime('%Y-%m-%d %H:%M')],
        ["Trade Stop Date", stop_trade_date.strftime('%Y-%m-%d %H:%M')]        
    ]

    # Print table
    print_data_table(trade_data)

    return  loading_from_date,Today,stop_trade_date, minimum_future_trade_value, trade_value, n_trades, position, stop_loss_pct

def load_api_keys(filename):
    try:
        with open(filename, 'r') as file:
            api_key = file.readline().strip()
            secret_key = file.readline().strip()
            if not api_key or not secret_key:
                raise ValueError("API key or secret key is missing.")
            return api_key, secret_key
    except Exception as e:
        raise ValueError(f"Error loading API keys: {e}")

def print_data_table(data):
    print("\n")
    print(tabulate(data, headers=["Parameter", "Value"], tablefmt="grid"))
    
def round_up(value, decimals=3):
    factor = 10 ** decimals
    return math.ceil(value * factor) / factor