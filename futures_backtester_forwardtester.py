# main.py
from binance.client import Client
from futures_trader import FuturesTrader
from futures_backtester_PV import Futures_Backtester_PV
from futures_backtester_SMA import Futures_Backtester_SMA
from futures_backtester_RSI import Futures_Backtester_RSI
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

########################################
############     Control     ###########
########################################
Perform_Testing = True
PRINT_DATA = False

if __name__ == "__main__": 

    api_key = "bb3927e0cccba6733b4be417ad4557743f1ea37634ac10c0e301cb74a6f1dd8f"
    secret_key = "760b8b3bc096d2baea157775e95dca947b2af2113ad2d6d7d2bece352f09d832"

    client = Client(api_key=api_key, api_secret=secret_key, tld="com", testnet=True)

    symbol = "BTCUSDT"
    bar_length = "3m"

    leverage = 2
    strategy="RSI"  #(PV,SMA,RSI)
    tc = -0.00085   #average/approx Trading Costs in Futures Market

    # Define time Period
    Today = datetime.utcnow()
    #historical_days = timedelta(hours=12)
    historical_days_C = timedelta(days=60)
    historical_days_T = timedelta(days=2)
    period = historical_days_C

    start_date_B = Today-historical_days_T-historical_days_C
    end_date_B   = Today-historical_days_T

    start_date_F = Today-historical_days_T
    end_date_F   = Today

    print(f"\nData is collected from : {start_date_B}, until : {end_date_B}\n")
    if (Perform_Testing) :
        print("\nFutures Backtester \n")
        if (strategy=="PV") :
            print("\nSIMPLE PRICE & VOLUME STRATEGY")
            backtester = Futures_Backtester_PV(client=client, symbol=symbol, bar_length=bar_length,
                                    start = start_date_B, end = end_date_B, tc = tc,
                                    leverage=leverage, strategy=strategy)
            
            if (PRINT_DATA):
                # Access and print the data
                print("Data:")
                print(backtester.data)
                print("\n")

                # If you want to see a summary or specific part of the data, you can use:
                print("Data Summary:")
                print(backtester.data.info())
                print("\n")

                print("First few rows of the data:")
                print(backtester.data.head())
                print("\n")

                print("Last few rows of the data:")
                print(backtester.data.tail())
                print("\n")

            #- Return Threshold: All Returns >= __90th__ Percentile labeled "Very High Return"
            #- Low and High Volume Change Threshold: 
            # All Volume Changes between __5th__ and __20th__ Percentile labeled "Moderate to High Decrease in Volume" 

            backtester.test_strategy(percentiles = (10, 90, 5, 20))
            
            #Plots the cumulative performance of the trading strategy compared to buy-and-hold.
            backtester.plot_strategy_comparison()
            
            #Plots a scatter plot of volume change against returns.
            backtester.plot_results_II()
            
            #Bins returns and volume change, creates a crosstab matrix, and plots a heatmap.
            #_-> Extremely High (positive) returns and Decreases in Volume is a Contrarian (mean-reverting) signal -> prices will fall.__ <br>
            #_-> Extremely Low (negative) returns and Decreases in Volume is a Contrarian (mean-reverting) signal -> prices will rise.__ <br>
            backtester.plot_heatmap()

            backtester.results.trades.value_counts()

            backtester.optimize_strategy(return_low_range = (2, 20, 2),
                                         return_high_range = (80, 98, 2), 
                                         vol_low_range = (0, 18, 2), 
                                         vol_high_range = (18, 40, 2),
                                         metric = "Sharpe") #Multiple,Sharpe
            
            #backtester.plot_strategy_comparison()

            backtester.results.cstrategy.plot(figsize = (12,8))

            backtester.results.position.value_counts()

            backtester.results.trades.value_counts()

            backtester.add_leverage(leverage = leverage) 
            backtester.plot_strategy_comparison(leverage = True)
            #print(backtester.results)

        if (strategy=="SMA") :
            print("\nSimple Moving Average Strategy")
            
            sma_s = 15 #15
            sma_m = 50 #50
            sma_l = 200 #200
            backtester = Futures_Backtester_SMA(client=client, symbol=symbol, bar_length=bar_length,
                                                start = start_date_B, end = end_date_B, tc = tc,
                                                leverage=leverage, strategy=strategy)
            
            backtester.test_strategy(sma_s, sma_m, sma_l)
            #Plots the cumulative performance of the trading strategy compared to buy-and-hold.
            #backtester.plot_strategy_comparison()

            #backtester.results
            backtester.add_leverage(leverage = leverage)
            backtester.plot_strategy_comparison(leverage = True)
            backtester.results

            # Assuming the class instance is created and assigned to strategy_instance
            SMA_S_range = (10, 50, 5)  # start at 10, end at 50, step size of 5
            SMA_M_range = (50, 100, 5)  # start at 50, end at 100, step size of 5
            SMA_L_range = (100, 200, 5)  # start at 100, end at 200, step size of 5
            metric = "Sharpe"  #"Multiple" or "Sharpe" depending on what you want to optimize

            backtester.optimize_strategy(SMA_S_range, SMA_M_range, SMA_L_range, metric)
            backtester.results.trades.value_counts()
            backtester.results.eff_lev.describe()
            backtester.results.eff_lev.plot(figsize = (12, 8))

        if (strategy=="RSI") :
            print("\nRelative Strength Index")
            
            rsi_window=14
            rsi_lower=30
            rsi_upper=70
            backtester = Futures_Backtester_RSI(client=client, symbol=symbol, bar_length=bar_length,
                                                start = start_date_B, end = end_date_B, tc = tc,
                                                leverage=leverage, strategy=strategy)
            
            backtester.test_strategy(rsi_window, rsi_lower, rsi_upper)

            #Plots the cumulative performance of the trading strategy compared to buy-and-hold.
            backtester.plot_strategy_comparison()

            backtester.results
            backtester.add_leverage(leverage = leverage)
            backtester.plot_strategy_comparison(leverage = True)
            backtester.results

            # Define parameter ranges
            rsi_window_range = (1, 20, 2)
            rsi_lower_range = (20, 40, 5)
            rsi_upper_range = (60, 80, 5)
            metric = "Sharpe"  #"Multiple" or "Sharpe" depending on what you want to optimize
            
            backtester.optimize_strategy(rsi_window_range, rsi_lower_range, rsi_upper_range, metric)
            backtester.results.trades.value_counts()
            #acktester.results.eff_lev.describe()
            #backtester.results.eff_lev.plot(figsize = (12, 8))
            print(f"\n Perform Forward testing from : {start_date_F}, until : {end_date_F}\n")
            rsi_window=3
            rsi_lower=35
            rsi_upper=60
            backtester = Futures_Backtester_RSI(client=client, symbol=symbol, bar_length=bar_length,
                                                start = start_date_F, end = end_date_F, tc = tc,
                                                leverage=leverage, strategy=strategy)
            
            backtester.test_strategy(rsi_window, rsi_lower, rsi_upper)

            #Plots the cumulative performance of the trading strategy compared to buy-and-hold.
            backtester.plot_strategy_comparison()

   