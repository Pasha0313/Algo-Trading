
# main.py
import time
import os
from config_loader import load_config_from_text, prepare_data_from_config, get_control_settings,round_up,prepare_trade_from_config,load_api_keys
from config_check import make_request_with_retries,Debug_function
from tkinter import TRUE
from requests.exceptions import ConnectionError, Timeout, RequestException
from binance.client import Client
from futures_trader import FuturesTrader
#from Futures_Backtester_Base import FuturesBacktesterBase
from futures_backtester_PV import Futures_Backtester_PV
from futures_backtester_SMA import Futures_Backtester_SMA
from futures_backtester_RSI import Futures_Backtester_RSI
from futures_backtester_MACD import Futures_Backtester_MACD
from futures_backtester_VWAP import Futures_Backtester_VWAP
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")

# Define the path
PATH = r'C:\DATA\Trading\V1.11\Futures'

# Load configuration using os.path.join to concatenate the path and file name

def new_func(symbol, client):
    current_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
    return current_price

if __name__ == "__main__": 

    # Load configuration
    #config = load_config_from_text('Config.txt')
    config = load_config_from_text(os.path.join(PATH, 'Config.txt'))

    # Get control settings
    Perform_Testing, Print_Data, Perform_Trading = get_control_settings(config)

    # Example of using control settings
    if Perform_Testing:
        print("\nTesting mode is enabled.")

    # Prepare data and get individual values
    start_date, end_date, symbol, bar_length, leverage, strategy, tc, test_days = prepare_data_from_config(config)
   
    # Load API keys from file
    try:
        #api_key, secret_key = load_api_keys('KEY.txt')
        api_key, secret_key = load_api_keys(os.path.join(PATH, 'KEY.txt'))
        #print(f"API Key: {api_key}")
        #print(f"Secret Key: {secret_key}")
    except ValueError as e:
        print(e)
        exit(1)

    print(f"\n Attempt : Trying to connect to Binance...")
    client = Client(api_key=api_key, api_secret=secret_key, tld="com", testnet=True)
    print("Successfully connected to Binance!")

    if Perform_Testing:
        print("\nFutures Backtester\n")

        if strategy == "PV":
            print("\nSIMPLE PRICE & VOLUME STRATEGY")
            
            backtester = Futures_Backtester_PV(client=client, symbol=symbol, bar_length=bar_length,
                                    start=start_date, end=end_date, tc=tc,
                                    leverage=leverage, strategy=strategy)
            
            if Print_Data:
                # Access and print the data
                print("Data:")
                print(backtester.data)
                print("\nData Summary:")
                print(backtester.data.info())
                print("\nFirst few rows of the data:")
                print(backtester.data.head())
                print("\nLast few rows of the data:")
                print(backtester.data.tail())
            
            percentiles=(10, 90, 5, 95)    
            backtester.test_strategy(percentiles)
            backtester.add_leverage(leverage=leverage)
            backtester.plot_strategy_comparison(leverage=True)
            print(backtester.results.trades.value_counts())

            # Optimize PV strategy
            return_low_range=(5, 15, 2)
            return_high_range=(85, 95, 2) 
            vol_low_range=(2, 10, 2) 
            vol_high_range=(85, 95, 2)
            metric="Sharpe"
            return_thresh , volume_thresh = backtester.optimize_strategy(return_low_range,return_high_range,vol_low_range,vol_high_range,metric)

        elif strategy == "SMA":
            print("\nSimple Moving Average Strategy")
            backtester = Futures_Backtester_SMA(client=client, symbol=symbol, bar_length=bar_length,
                                                start=start_date, end=end_date, tc=tc,
                                                leverage=leverage, strategy=strategy)
            SMA = (15, 50, 200)  # (sma_s, sma_m, sma_l)            
            backtester.test_strategy(SMA)
            backtester.add_leverage(leverage=leverage)
            backtester.plot_strategy_comparison(leverage=True)
            print(backtester.results.trades.value_counts())

            # Optimize SMA strategy
            SMA_S_range = (10, 20, 2)
            SMA_M_range = (30, 60, 5)
            SMA_L_range = (100, 300, 10)
            metric = "Sharpe"
            sma_s, sma_m, sma_l = backtester.optimize_strategy(SMA_S_range, SMA_M_range, SMA_L_range, metric)

        elif strategy == "RSI":
            print("\nRelative Strength Index")
            backtester = Futures_Backtester_RSI(client=client, symbol=symbol, bar_length=bar_length,
                                                start=start_date, end=end_date, tc=tc,
                                                leverage=leverage, strategy=strategy)
            rsi_window = 14
            rsi_lower = 40
            rsi_upper = 60
            backtester.test_strategy(rsi_window, rsi_lower, rsi_upper)
            backtester.add_leverage(leverage=leverage)
            backtester.plot_strategy_comparison(leverage=True)
            print(backtester.results.trades.value_counts())
            
            # Optimize RSI strategy
            rsi_window_range = (9, 15, 1)
            rsi_lower_range = (15, 40, 5)
            rsi_upper_range = (60, 85, 5)
            metric = "Sharpe"
            rsi_window,rsi_lower,rsi_upper = backtester.optimize_strategy(rsi_window_range, rsi_lower_range, rsi_upper_range, metric)

        elif strategy == "MACD":
            print("\nMoving Average Convergence Divergence (MACD)")
            # Initialize the MACD backtester
            backtester = Futures_Backtester_MACD(client=client,symbol=symbol,bar_length=bar_length,
                                                start=start_date,end=end_date,tc=tc,
                                                leverage=leverage,strategy=strategy)
            macd_slow = 26   # Slow moving average window
            macd_fast = 12   # Fast moving average window
            macd_signal = 9  # Signal line window    
            backtester.test_strategy(macd_slow, macd_fast, macd_signal)
            backtester.add_leverage(leverage=leverage)
            backtester.plot_strategy_comparison(leverage=True)
            print(backtester.results.trades.value_counts())

            macd_slow_range = (20, 50, 5)     # Range for slow moving average
            macd_fast_range = (5, 20, 3)      # Range for fast moving average
            macd_signal_range = (5, 20, 3)    # Range for signal line
            metric = "Sharpe"                 # Metric to optimize for
            macd_slow, macd_fast, macd_signal = backtester.optimize_strategy(macd_slow_range,macd_fast_range,macd_signal_range,metric)
            
        elif strategy == "VWAP":
            print("\nVolume Weighted Average Price")
            backtester = Futures_Backtester_VWAP(client=client,symbol=symbol,bar_length=bar_length,
                                                start=start_date,end=end_date,tc=tc,
                                                leverage=leverage,strategy=strategy)
            vwap_period = 14
            vwap_threshold = 0.1    
            backtester.test_strategy(vwap_period,vwap_threshold)
            backtester.add_leverage(leverage=leverage)
            backtester.plot_strategy_comparison(leverage=True)
            print(backtester.results.trades.value_counts())
    
            # Optimize VWAP strategy
            vwap_period_range = (20, 50, 5)  # Example range for VWAP period
            vwap_threshold_range = (0.1, 1.0, 0.1)  # Example range for VWAP threshold
            metric = "Sharpe"
            vwap_period, vwap_threshold = backtester.optimize_strategy(vwap_period_range, vwap_threshold_range, metric)

    ##################################################################################################
    ##################################################################################################
    ##################################################################################################   
    if Perform_Trading:
        # Prepare trade parameters
        loading_from_date,Today,stop_trade_date, minimum_future_trade_value, trade_value, n_trades, position, stop_loss_pct = prepare_trade_from_config(config)
        
        print("\nStart Trading")
        #print(f"\nTrade will continue from: {Today}, until: {stop_trade_date}, Max number of trades is: {n_trades}")

        current_price = new_func(symbol, client)
        #units = round_up(trade_value / current_price)  
        units = round(trade_value / current_price,3)  
        print(f'units = {units}')

        if trade_value < minimum_future_trade_value:
            min_required_units = minimum_future_trade_value / current_price
            print(f"\nMinimum trade value {trade_value} is below {minimum_future_trade_value}. Adjust units or strategy.")
            print(f"Current Price: {current_price}")
            print(f"Minimum Required Units for Trade: {min_required_units}\n")
        else:
            if strategy == "PV":
                print("\nSIMPLE PRICE & VOLUME STRATEGY")
                if not Perform_Testing:
                    return_thresh = [-0.008,  0.008]
                    volume_thresh = [-2.0  , 2.0 ]
                print(f"\n return_thresh ={return_thresh}, volume_thresh ={volume_thresh}")
                trader = FuturesTrader(client=client, symbol=symbol, bar_length=bar_length,
                                    return_thresh=return_thresh, volume_thresh=volume_thresh,
                                    units=units, position=position, leverage=leverage,
                                    stop_trade_date=stop_trade_date, strategy=strategy, 
                                    stop_loss_pct=stop_loss_pct, n_trades=n_trades)
            
            elif strategy == "SMA":
                print("\nSimple Moving Average Strategy")
                if not Perform_Testing:
                    sma_s = 15
                    sma_m = 50
                    sma_l = 200
                print(f"sma_s={sma_s}, sma_m={sma_m}, sma_l={sma_l}")
                trader = FuturesTrader(client=client, symbol=symbol, bar_length=bar_length,
                            sma_s=sma_s, sma_m=sma_m, sma_l=sma_l,
                            units=units, position=position, leverage=leverage,
                            stop_trade_date=stop_trade_date, strategy=strategy, 
                            stop_loss_pct=stop_loss_pct, n_trades=n_trades)

            elif strategy == "RSI":
                print("\nRelative Strength Index")
                if not Perform_Testing:
                    rsi_window = 10
                    rsi_lower = 20
                    rsi_upper = 80
                print(f"rsi_window={rsi_window}, rsi_lower={rsi_lower}, rsi_upper={rsi_upper}")
                trader = FuturesTrader(client=client, symbol=symbol, bar_length=bar_length,
                            rsi_window=rsi_window, rsi_lower=rsi_lower, rsi_upper=rsi_upper,
                            units=units, position=position, leverage=leverage,
                            stop_trade_date=stop_trade_date, strategy=strategy, 
                            stop_loss_pct=stop_loss_pct, n_trades=n_trades)
            
            elif strategy == "MACD":
                print("\nMoving Average Convergence Divergence (MACD)")
                if not Perform_Testing:
                    macd_s = 12  # Short-term EMA window
                    macd_l = 26  # Long-term EMA window
                    macd_smooth = 9  # Signal line smoothing
                print(f"macd_s={macd_s}, macd_l={macd_l}, macd_smooth={macd_smooth}")
                trader = FuturesTrader(client=client, symbol=symbol, bar_length=bar_length,
                                       macd_s=macd_s, macd_l=macd_l, macd_smooth=macd_smooth,
                                       units=units, position=position, leverage=leverage,
                                       stop_trade_date=stop_trade_date, strategy=strategy,
                                       stop_loss_pct=stop_loss_pct, n_trades=n_trades)
                
            elif strategy == "VWAP":
                print("\nVolume Weighted Average Price")
                if not Perform_Testing:
                    vwap_period = 14  # Set VWAP period
                    vwap_threshold = 0.01  # Set VWAP threshold
        
                print(f"vwap_period={vwap_period}, vwap_threshold={vwap_threshold}")
    
                trader = FuturesTrader(client=client, symbol=symbol, bar_length=bar_length,
                                       vwap_period=vwap_period, vwap_threshold=vwap_threshold,
                                       units=units, position=position, leverage=leverage,
                                       stop_trade_date=stop_trade_date, strategy=strategy, 
                                       stop_loss_pct=stop_loss_pct, n_trades=n_trades)


            trader.start_trading(historical_days=loading_from_date)
            
            print(trader.prepared_data.tail(20))
            print(trader.cum_profits)

            # Uncomment to check account info
            # client.get_account()
