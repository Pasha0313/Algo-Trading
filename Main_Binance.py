import Loading_Config_BN as Loading_Config_BN
import Loading_Config_IB as Loading_Config_IB
import pandas as pd
import os

from Config_Check import make_request_with_retries,Debug_function
from tkinter import TRUE
from Trader_BN import FuturesTrader_BN
from Back_Testing_BN import BackTesting_BN
from Forecast_Testing import ForecastTesting  
from Loading_Strategy import StrategyLoader
from Loading_ForecastModel import LoadingForecastModel
from Unsupervised_learning_trading_strategy import Unsupervised_learning_trading_strategy

from ib_insync import *
#from ib_insync import IB, Forex, Stock, Future, Contract
from binance import Client

import warnings
warnings.filterwarnings("ignore")

Broker = "Binance"
#Broker = "Interactive Broker"

print(f"\nThe Broker is {Broker}\n")

Path_Configs ="Configs"
os.makedirs(Path_Configs, exist_ok=True)

# Load configuration using os.path.join to concatenate the path and file name
def get_BN_price(symbol, client):
    return float(client.futures_symbol_ticker(symbol=symbol)['price'])

def run_binance():
    config = Loading_Config_BN.load_config_from_text(os.path.join(Path_Configs, "Config_BN.txt"))
    strategy_loader = StrategyLoader(os.path.join(Path_Configs,"strategies_config.json"))

    # Get control settings
    Unsupervised_Learning, Perform_Testing, Print_Data, Perform_Forecasting, Perform_Tuner, Perform_Trading\
    = Loading_Config_BN.get_control_settings(config)

    # Prepare data and get individual values
    start_date, end_date, symbol, bar_length, leverage, strategy, tc, test_days,metric , ForecastModelName ,\
    future_forecast_steps = Loading_Config_BN.prepare_data_from_config(config)

    # Load API keys from file
    try:
        api_key, secret_key = Loading_Config_BN.load_api_keys(os.path.join(Path_Configs, "KEY.txt"))
    except ValueError as e:
        print(e)
        exit(1)        

    print(f"\nAttempt : Trying to connect to Binance...")
    client = Client(api_key=api_key, api_secret=secret_key, tld="com", testnet=True)
    print("Successfully connected to Binance!\n")

################################################################################################################  
    if Unsupervised_Learning:
        print("\n Unsupervised Learning Trading Strategy")
        Unsupervised_learning_trading_strategy()

################################################################################################################   
    if Perform_Testing:
        print("\nFutures Back Testing is enabled\n")

        description, parameters_BT, param_ranges_BT = strategy_loader.process_strategy(strategy)

        print(f"\nStrategy: {strategy}")
        print(f"Description: {description}")
        print(f"Parameters Back Testing: {parameters_BT}")
        print(f"Parameter Ranges: {param_ranges_BT}\n")
        
        backtesting = BackTesting_BN(client=client, symbol=symbol, bar_length=bar_length,
                                start=start_date, end=end_date, tc=tc,leverage=leverage, strategy=strategy)
        backtesting.test_strategy(parameters_BT)
        backtesting.add_leverage(leverage=leverage)
        backtesting.plot_strategy_comparison(leverage=True,plot_name=f"{symbol}_{strategy}")
        backtesting.plot_all_indicators(plot_name=f"{symbol}_{strategy}")
        print(backtesting.results.trades.value_counts())
        parameters_BT  = backtesting.optimize_strategy(param_ranges_BT,metric,output_file=f"{strategy}_optimize_results.csv")
        if not parameters_BT == None :      
            backtesting.test_strategy(parameters_BT)
            backtesting.add_leverage(leverage=leverage)
            backtesting.plot_strategy_comparison(leverage=True,plot_name=f"WOpt_{symbol}_{strategy}")
            backtesting.plot_all_indicators(plot_name=f"{symbol}_{strategy}")
        else :
            print("Parameters (BT) is : None")
################################################################################################################   
    if (Perform_Forecasting) :
        print("\nFutures Forecast Testing is enabled\n")

        model_loader = LoadingForecastModel(os.path.join(Path_Configs,"forecast_models_config.json"))
        description, parameters_F, param_ranges_F = model_loader.process_model(ForecastModelName)

        model_loader.print_model_details(ForecastModelName)
        forecast_testing = ForecastTesting(client=client, symbol=symbol, bar_length=bar_length,
                                start=start_date, end=end_date, tc=tc,leverage=leverage, strategy=strategy
                                ,future_forecast_steps=future_forecast_steps)
        
        forecast_testing.Forecast_price(ForecastModelName,parameters_F,param_ranges_F,Perform_Tuner) 

################################################################################################################   
    if Perform_Trading:
        # Prepare trade parameters
        loading_from_date,Today,stop_trade_date, minimum_future_trade_value, trade_value, TN_trades, position,\
        stop_loss_pct ,Total_stop_loss, Total_Take_Profit ,Position_Long,Position_Neutral,Position_Short= \
        Loading_Config_BN.prepare_trade_from_config(config)
        current_price = get_BN_price(symbol, client)
       
        print("\nStart Trading")
        #print(f"\nTrade will continue from: {Today}, until: {stop_trade_date}, Max number of trades is: {TN_trades}")

        print(f"trade_value = {trade_value} , current_price = {current_price} ")
        units = round(trade_value / current_price,3)  
        print(f'units = {units}')

        description, parameters, param_ranges = strategy_loader.process_strategy(strategy)
        print(f"Strategy: {strategy}")
        print(f"Description: {description}")
        strategy_loader.print_strategy_details(strategy)

        if trade_value < minimum_future_trade_value:
            min_required_units = minimum_future_trade_value / current_price
            print(f"\nMinimum trade value {trade_value} is below {minimum_future_trade_value}. Adjust units or strategy.")
            print(f"Current Price: {current_price}")
            print(f"Minimum Required Units for Trade: {min_required_units}\n")
        else:
            trader = FuturesTrader_BN(client=client, symbol=symbol, bar_length=bar_length,parameters=parameters,
                                units=units,stop_trade_date=stop_trade_date, Total_stop_loss=Total_stop_loss,stop_loss_pct=stop_loss_pct,
                                Total_Take_Profit=Total_Take_Profit,Position_Long=Position_Long,Position_Neutral=Position_Neutral,
                                Position_Short=Position_Short, TN_trades=TN_trades, position=position, leverage=leverage,strategy=strategy)
            trader.start_trading(historical_days=loading_from_date)

            if hasattr(trader, 'Rep_Trade') and isinstance(trader.Rep_Trade, pd.DataFrame) and not trader.Rep_Trade.empty:
                print(trader.prepared_data.tail(trader.N_trades))
                print('\n' * 2)
                Report_Trades = trader.Rep_Trade.drop(['id', 'orderId', 'commissionAsset', 'positionSide', 'maker', 'buyer'], axis=1, errors='ignore')
                Report_Trades['time'] = pd.to_datetime(Report_Trades['time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M')
                columns_order = ['time'] + [col for col in Report_Trades.columns if col != 'time']
                Report_Trades = Report_Trades[columns_order]
                print(Report_Trades)

            separator = "-" * 70
            print("\n" * 2 + separator)
            print(f"| {'Final Trade Report'.center(66)} |")
            print(separator)
            print(f"| Trade Number          : {trader.N_trades:<42} |")
            print(f"| Cumulative Profit     : {trader.cum_profits:<42} |")
            print(separator + "\n")        

            # Uncomment to check account info
            # client.get_account()
