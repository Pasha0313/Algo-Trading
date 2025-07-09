import Loading_Config_BN as Loading_Config_BN
import Loading_Config_IB as Loading_Config_IB
import pandas as pd
import numpy as np
import os

from Config_Check import make_request_with_retries,Debug_function
from tkinter import TRUE
from Trader_IB import FuturesTrader_IB
from Back_Testing_IB import BackTesting_IB
from Forecast_Testing import ForecastTesting  
from Loading_Strategy import StrategyLoader
from Loading_ForecastModel import LoadingForecastModel
from Unsupervised_learning_trading_strategy import Unsupervised_learning_trading_strategy

from ib_insync import *
#from ib_insync import IB, Forex, Stock, Future, Contract
from binance import Client

import warnings
warnings.filterwarnings("ignore")

Path_Configs ="Configs"
os.makedirs(Path_Configs, exist_ok=True)

# Load configuration using os.path.join to concatenate the path and file name
def get_ibkr_price(client, contract, Print_Data: bool = True):
    client.reqMarketDataType(4)  

    symbol = contract.symbol
    asset_type = contract.asset_type.lower()

    market_data = client.reqMktData(contract, '', False, False)

    for _ in range(10):
        price_fields = [market_data.last, market_data.close, market_data.bid, market_data.ask]
        if any(p is not None and not np.isnan(p) for p in price_fields):
            break
        client.sleep(0.5)

    price_candidates = [market_data.last, market_data.close, market_data.bid, market_data.ask]
    price = next((p for p in price_candidates if p is not None and not np.isnan(p)), None)

    if Print_Data:
        print(f"[DEBUG] Price fields for {symbol} ({asset_type}) - "
              f"last: {market_data.last}, close: {market_data.close}, "
              f"bid: {market_data.bid}, ask: {market_data.ask}")

    if price is None:
        raise RuntimeError(f"No price available for {symbol} ({asset_type})")

    return float(price)

def run_interactive():
    config = Loading_Config_IB.load_config_from_text(os.path.join(Path_Configs, "Config_IB.txt"))

    strategy_loader = StrategyLoader(os.path.join(Path_Configs,"strategies_config.json"))

    # Get control settings
    Unsupervised_Learning, Perform_BackTesting, Print_Data, Perform_Forecasting, Perform_Tuner, Perform_Trading\
    = Loading_Config_IB.get_control_settings(config)

    print("\nAttempt : Connecting to Interactive Brokers...")
    client = IB()
    client.connect('127.0.0.1', 7497, clientId=1)  # default TWS paper trading port
    print("Successfully connected to IBKR!\n")

    # Get account summary and convert to DataFrame
    account_summary = client.accountSummary()
    df_summary = util.df(account_summary)

    # Access Net Liquidation value
    net_liq = df_summary[df_summary['tag'] == 'NetLiquidation']['value'].values[0]
    #print(f"Net Liquidation Value (Equity): {net_liq}")
    print('The Net Liquidation value ')
    print(df_summary[['tag', 'value']])
    
    # Prepare data and get individual values
    start_date, end_date, contract, timezone,bar_length, leverage, strategy, tc, metric, \
    ForecastModelName, future_forecast_steps, duration, what_to_show, use_rth = Loading_Config_IB.prepare_data_from_config(config,client)

################################################################################################################  
    if Unsupervised_Learning:
        print("\n Unsupervised Learning Trading Strategy")
        Unsupervised_learning_trading_strategy()

################################################################################################################   
    if Perform_BackTesting:
        print("\nFutures Back Testing is enabled\n")

        description, parameters_BT, param_ranges_BT = strategy_loader.process_strategy(strategy)

        print(f"\nStrategy: {strategy}")
        print(f"Description: {description}")
        print(f"Parameters Back Testing: {parameters_BT}")
        print(f"Parameter Ranges: {param_ranges_BT}\n")
        
        backtesting = BackTesting_IB(ib=client,contract=contract,timezone=timezone ,bar_length=bar_length,
                                        start=start_date,end=end_date, tc=tc,leverage=leverage,strategy=strategy)
        backtesting.test_strategy(parameters_BT)
        backtesting.add_leverage(leverage=leverage)
        backtesting.plot_strategy_comparison(leverage=True,plot_name=f"{contract.symbol}_{strategy}")
        backtesting.plot_all_indicators(plot_name=f"{contract.symbol}_{strategy}", Print_Data = Print_Data)
        print(backtesting.results.trades.value_counts())
        parameters_BT  = backtesting.optimize_strategy(param_ranges_BT,metric,output_file=f"{strategy}_optimize_results.csv")
        if not parameters_BT == None :      
            backtesting.test_strategy(parameters_BT)
            backtesting.add_leverage(leverage=leverage)
            backtesting.plot_strategy_comparison(leverage=True,plot_name=f"WOpt_{contract.symbol}_{strategy}")
            backtesting.plot_all_indicators(plot_name=f"{contract.symbol}_{strategy}", Print_Data = Print_Data)
        else :
            print("Parameters (BT) is : None")
################################################################################################################   
    if (Perform_Forecasting) :
        print("\nFutures Forecast Testing is enabled\n")

        model_loader = LoadingForecastModel(os.path.join(Path_Configs,"forecast_models_config.json"))
        description, parameters_F, param_ranges_F = model_loader.process_model(ForecastModelName)

        model_loader.print_model_details(ForecastModelName)
        symbol = contract.symbol
        forecast_testing = ForecastTesting(client=client, symbol=symbol, bar_length=bar_length,
                                start=start_date, end=end_date, tc=tc,leverage=leverage, strategy=strategy
                                ,future_forecast_steps=future_forecast_steps)
        
        forecast_testing.Forecast_price(ForecastModelName,parameters_F,param_ranges_F,Perform_Tuner) 

################################################################################################################   
    if Perform_Trading:
        # Prepare trade parameters
        loading_from_date,Today,stop_trade_date, minimum_future_trade_value, trade_value, TN_trades, position,\
        stop_loss_pct ,Total_stop_loss, Total_Take_Profit ,Position_Long,Position_Neutral,Position_Short= \
        Loading_Config_IB.prepare_trade_from_config(config)
        current_price = get_ibkr_price(client, contract)
        
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
            trader = FuturesTrader_IB(ib=client, contract=contract, bar_length=bar_length, parameters=parameters,
                                units=units, position=position, leverage=leverage, stop_trade_date=stop_trade_date,
                                strategy=strategy, Total_stop_loss=Total_stop_loss, stop_loss_pct=stop_loss_pct,
                                Position_Long=Position_Long, Position_Neutral=Position_Neutral, Position_Short=Position_Short,
                                Total_Take_Profit=Total_Take_Profit, TN_trades=TN_trades)
            trader.start_trading(duration)
            
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
