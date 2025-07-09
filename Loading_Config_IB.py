from datetime import datetime, timedelta
from tabulate import tabulate
from ib_insync import *

def get_ib_contract(client, symbol, asset_type="forex", secType='CASH', expiry=None, exchange=None, currency="USD"):

    symbol = symbol.strip().upper()
    asset_type = asset_type.lower()

    # === Default Exchange Routing ===
    if exchange is None:
        if asset_type == "forex":
            exchange = "IDEALPRO"
        elif asset_type == "cfd":
            if symbol in ["DEIDXEUR", "DE40", "UK100", "NAS100", "SPX500", "US30"]:
                exchange = "SMART"  # Index CFDs route via SMART
            else:
                exchange = "SMART"
        elif asset_type == "crypto":
            exchange = "PAXOS"
        else:
            exchange = "SMART"

    print(f"[INFO] Creating IB contract for symbol: {symbol}, type: {asset_type}, exchange: {exchange}, currency {currency}")

    # === Contract Type Routing ===
    if asset_type == "forex":
        assert len(symbol) == 6, "❌ Forex symbol must be like 'EURUSD'"
        contract = Forex(symbol)

    elif asset_type == "stock":
        contract = Stock(symbol, exchange, currency)

    elif asset_type == "future":
        assert expiry, "❌ Future contracts require expiry."
        contract = Future(symbol, expiry, exchange, currency)

    elif asset_type == "crypto":
        contract = Contract(symbol=symbol, secType="CRYPTO", exchange=exchange, currency=currency)

    elif asset_type == "cfd":
        contract = Contract(symbol=symbol, secType="CFD", exchange=exchange, currency=currency)

    else:
        raise ValueError(f"❌ Unsupported asset type: {asset_type}")

    # === Contract Validation ===
    details = client.reqContractDetails(contract)
    if not details:
        raise ValueError(f"❌ No contract found for {symbol} ({asset_type}) with exchange {exchange}")
    
    print(f"\n[✅] Valid contract found: {details[0].contract}\n")
    
    return contract

def load_config_from_text(filename):
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                config[key.strip()] = value.strip()
    return config

def get_control_settings(config):
    Unsupervised_Learning = config.get("Unsupervised_Learning", "False").lower() == "true"
    Perform_BackTesting = config.get("Perform_BackTesting", "False").lower() == "true"
    Print_Data = config.get("Print_Data", "False").lower() == "true"
    Perform_Forecasting = config.get("Perform_Forecasting", "False").lower() == "true"
    Perform_Tuner = config.get("Perform_Tuner", "False").lower() == "true"
    Perform_Trading = config.get("Perform_Trading", "False").lower() == "true"
    return Unsupervised_Learning, Perform_BackTesting, Print_Data, Perform_Forecasting, Perform_Tuner, Perform_Trading

def print_config_values(symbol, bar_length, leverage, strategy, tc, test_days):
    print("\nIndividual Values:")
    print(f"Symbol: {symbol}")
    print(f"Bar Length: {bar_length}")
    print(f"Leverage: {leverage}")
    print(f"Strategy: {strategy}")
    print(f"Trading Costs: {tc:.5f}")
    print(f"Days: {test_days}")

def prepare_data_from_config(config,client):
    # Raw config values
    symbol = config.get("symbol", "BTC")
    asset_type = config.get("asset_type", "Future")
    secType = config.get("secType", "CRYPTO")
    timezone = str(config.get("timezone", "US/Eastern"))

    expiry = config.get("expiry", None)
    exchange = config.get("exchange", None)
    currency = config.get("currency", "USD")

    # IB-specific
    bar_length = config.get("bar_length", "5 mins")
    duration = config.get("duration", "2 D")
    what_to_show = config.get("what_to_show", "MIDPOINT")
    use_rth = config.get("use_rth", "False").lower() == "true"

    leverage = int(config.get("leverage", 1))
    strategy = config.get("strategy", "RSI")
    tc = float(config.get("tc", -0.0002))
    metric = config.get("metric", "Sharpe")
    ForecastModelName = config.get("ForecastModelName", "ARIMA") 
    future_forecast_steps = int(config.get("FutureForecastSteps", 50))

    Today = datetime.utcnow()
    historical_days = parse_duration_str(duration)
    start_date = Today - historical_days
    end_date = Today

    # Build contract
    try:
        contract = get_ib_contract(client, symbol, asset_type, secType, expiry, exchange, currency)
        client.qualifyContracts(contract)
    except Exception as e:
        print(f"\n❌ Error building or qualifying contract for symbol '{symbol}': {e}")
        raise

    # Display info
    input_data = [
        ["Asset Type", asset_type],
        ["Symbol", symbol],
        ['secType',secType],
        ["timezone", timezone],
        ["Expiry", expiry],
        ["Exchange", exchange],
        ["currency", currency],
        ["Bar Length", bar_length],
        ["Duration", duration],
        ["Use RTH", use_rth],
        ["What to Show", what_to_show],
        ["Leverage", leverage],
        ["Strategy", strategy],
        ["Trading Costs", f"{tc:.5f}"],
        ["Start Date", start_date.strftime('%Y-%m-%d %H:%M')],
        ["End Date", end_date.strftime('%Y-%m-%d %H:%M')],
        ["Metric", metric],
        ["Forecast Model Name", ForecastModelName]
    ]
    print_data_table(input_data)
    return start_date, end_date, contract, timezone ,bar_length, leverage, strategy, tc, metric, \
           ForecastModelName, future_forecast_steps, duration, what_to_show, use_rth

def prepare_trade_from_config(config):
    history_days = int(config.get("history_days", 10))
    trade_hours = float(config.get("trade_hours", 24))
    minimum_future_trade_value = float(config.get("minimum_future_trade_value", 100))
    trade_value = float(config.get("trade_value", 250.0))
    TN_trades = int(config.get("TN_trades", 50))
    position = int(config.get("position", 0))
    stop_loss_pct = float(config.get("stop_loss_pct", 0.02))
    Total_stop_loss = float(config.get("Total_stop_loss", 200))
    Total_Take_Profit = float(config.get("Total_Take_Profit", 200))
    Position_Long = config.get("Position_Long", "True").lower() == "true"
    Position_Neutral = config.get("Position_Neutral", "True").lower() == "true"
    Position_Short = config.get("Position_Short", "True").lower() == "true"

    Today = datetime.utcnow()
    loading_from_date = Today - timedelta(days=history_days)
    stop_trade_date = Today + timedelta(hours=trade_hours)
    
    trade_data = [
        ["Minimum Future Trade Value", f"{minimum_future_trade_value:.2f}"],
        ["Trade Value", f"{trade_value:.2f}"],
        ["Number of Trades", TN_trades],
        ["Position", position],
        ["Stop Loss Percentage", f"{stop_loss_pct:.2f}"],
        ["Total Stop Loss", f"{Total_stop_loss:.2f}"],
        ["Total Take Profit", f"{Total_Take_Profit:.2f}"],
        ["Historical Period", f"{history_days}"],
        ["Loading From Date", loading_from_date.strftime('%Y-%m-%d %H:%M')],
        ["Trade Start Date", Today.strftime('%Y-%m-%d %H:%M')],
        ["Trade Stop Date", stop_trade_date.strftime('%Y-%m-%d %H:%M')],
        ["Position_Long", Position_Long],
        ["Position_Neutral", Position_Neutral],
        ["Position_Short", Position_Short],
    ]

    print_data_table(trade_data)

    return loading_from_date, Today, stop_trade_date, minimum_future_trade_value, trade_value, TN_trades, \
           position, stop_loss_pct, Total_stop_loss, Total_Take_Profit, Position_Long, Position_Neutral, Position_Short

def print_data_table(data):
    print("\n")
    print(tabulate(data, headers=["Parameter", "Value"], tablefmt="grid"))

def parse_duration_str(duration_str):
    """
    Converts a string like '2 D' or '5 days' to a timedelta object.
    """
    num, unit = duration_str.strip().split()

    num = int(num)
    unit = unit.lower()

    if "d" in unit:
        return timedelta(days=num)
    elif "h" in unit:
        return timedelta(hours=num)
    elif "min" in unit:
        return timedelta(minutes=num)
    else:
        raise ValueError(f"Unsupported duration unit: {unit}")
