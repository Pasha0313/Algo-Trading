# futures_trader.py
from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
import logging
import ta

class FuturesTrader:
    
    def __init__(self, client, symbol, bar_length, units, stop_trade_date, stop_loss_pct,
                 return_thresh= [-0.005, 0.005], volume_thresh=[-2, 2],
                 sma_s = 15,sma_m=50, sma_l=200,
                 rsi_window=14, rsi_lower=30, rsi_upper=70,
                 macd_s = 12,macd_l = 26,macd_smooth = 9,
                 vwap_period=14, vwap_threshold=0.01, 
                 n_trades =100, position=0, leverage=5, strategy="PV"):        
        
        self.client = client  # Store the client object
        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.units = units
        self.position = position
        self.leverage = leverage
        self.cum_profits = 0 
        self.trades = 0 
        self.n_trades = n_trades 
        self.strategy = strategy
        self.stop_date = stop_trade_date
        self.stop_loss_pct = stop_loss_pct  # External stop loss percentage
        
        #*****************add strategy-specific attributes here******************
        self.return_thresh = return_thresh
        self.volume_thresh = volume_thresh
        #************************************************************************
        self.SMA_S = sma_s
        self.SMA_M = sma_m
        self.SMA_L = sma_l
        #************************************************************************
        #RSI stands for Relative Strength Index
        self.rsi_window = rsi_window
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        #************************************************************************ 
        # MACD stands for Moving Average Convergence Divergence
        self.macd_s = macd_s         # Short-term EMA window
        self.macd_l = macd_l         # Long-term EMA window
        self.macd_smooth = macd_smooth  # Signal line smoothing window
        #************************************************************************
        # VWAP stands for Volume Weighted Average Price
        self.vwap_period = vwap_period
        self.vwap_threshold = vwap_threshold       
        # Error

        # Stop loss parameters
        self.stop_loss_price = None
    
    def start_trading(self,historical_days):
        self.client.futures_change_leverage(symbol = self.symbol, leverage = self.leverage) 
        
        self.twm = ThreadedWebsocketManager(testnet = True) # testnet 
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol = self.symbol, interval = self.bar_length,
                                 days =  historical_days) 
            self.twm.start_kline_futures_socket(callback = self.stream_candles,
                              symbol = self.symbol, interval = self.bar_length) # Adj: start_kline_futures_socket
            self.twm.join()
      
    def get_most_recent(self, symbol, interval, days):
        start_str =  str(days)
        end_str = None
        current_start_str = start_str
        all_bars = []  
        previous_candles_count = 0  
        print("\n")
        while True:
            # Fetch a chunk of data (up to 1000 candles)
            print(f"Requesting data from {pd.to_datetime(current_start_str).strftime('%Y-%m-%d %H:%M')}...")
            bars = self.client.futures_historical_klines(symbol = symbol, interval = interval,
                start_str=current_start_str,end_str = end_str,limit=1000)
            if not bars:
               print("No more data available or the API limit has been reached.")
               break
            all_bars.extend(bars)
            last_timestamp = pd.to_datetime(bars[-1][0], unit="ms")
            current_start_str = (last_timestamp + pd.Timedelta(milliseconds=1)).strftime('%Y-%m-%d %H:%M')
            print(f"Collected {len(all_bars)} candles so far...")
            if len(all_bars) == previous_candles_count + 1:
                #print("Only one new candle collected, exiting loop.")
                break
            previous_candles_count = len(all_bars)
            # Add delay to avoid hitting the API rate limit
            time.sleep(1)

        print(f"Total of {len(all_bars)} candles collected.\n")

        df = pd.DataFrame(all_bars)
    
        df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")

        # Get the start and end dates
        start_date = df['Date'].min().strftime('%Y-%m-%d %H:%M')
        end_date = df['Date'].max().strftime('%Y-%m-%d %H:%M')

        # Print the start and end dates
        print(f"Dataset Start from : {start_date}, End at: {end_date} \n")

        df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)
        df.index = df.index.astype(int)  # Ensure index is integer

        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        df["Complete"] = [True for row in range(len(df)-1)] + [False]
        
        self.data = df
    
    def stream_candles(self, msg):
        
        # extract the required items from msg        
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete=       msg["k"]["x"]

        # stop trading session        
        if event_time >= self.stop_date or self.trades >= self.n_trades:
            self.twm.stop()
            if self.position == 1:
                order = self.client.futures_create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 0
            elif self.position == -1:
                order = self.client.futures_create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.report_trade(order, "GOING NEUTRAL AND STOP")
                self.position = 0
            else: 
                if (event_time >= self.stop_date) :
                   print("Streaming Halted Due to Time Limit")
                elif (self.trades >= self.n_trades) :   
                   print("Streaming Halted Due to Trade number Limit")
        else:
            # print out
            print(".", end = "", flush = True) # just print something to get a feedback (everything OK) 
            if ((event_time.minute % 10 == 0) and (event_time.second == 0)): print(f"\nTime : {event_time.strftime('%Y-%m-%d %H:%M')}, Trade Number = {self.trades}")

            self.data.loc[start_time] = [first, high, low, close, volume, complete]

            if complete == True:
                if (self.strategy=="PV"):
                    #print("\nTrade with SIMPLE PRICE & VOLUME STRATEGY \n")
                    self.define_strategy_PV()
                elif (self.strategy=="SMA"):
                    #print("\nTrade with Simple Moving Average Strategy \n")
                    self.define_strategy_SMA()
                elif (self.strategy=="RSI"):
                    #print("\nTrade with Relative Strength Index Strategy \n")
                    self.define_strategy_RSI()  
                elif (self.strategy == "MACD"):
                    #print("\nTrade with Moving Average Convergence Divergence \n")
                    self.define_strategy_MACD()                    
                elif (self.strategy == "VWAP"):
                    #print("\nTrade with Moving Average Convergence Divergence \n")
                    self.define_strategy_VWAP()                        
                self.execute_trades()
    
    def define_strategy_PV(self):
        try:
            if not hasattr(self, 'data') or self.data is None:
                raise ValueError("Data is not initialized")        
            data = self.data.copy()

            #******************** define your strategy here ************************
            data = data[["Close", "Volume"]].copy()
            data["returns"] = np.log(data.Close / data.Close.shift())
            data["vol_ch"] = np.log(data.Volume.div(data.Volume.shift(1)))
            data.loc[data.vol_ch > 3, "vol_ch"] = np.nan
            data.loc[data.vol_ch < -3, "vol_ch"] = np.nan  
            
            cond1 = data.returns <= self.return_thresh[0]
            cond2 = data.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])
            cond3 = data.returns >= self.return_thresh[1]
            
            data["position"] = 0
            data.loc[cond1 & cond2, "position"] = 1
            data.loc[cond3 & cond2, "position"] = -1
            #***********************************************************************

            self.prepared_data = data.copy()
        except ValueError as e:
            warning_message = str(e)
            warnings.warn(warning_message, UserWarning)
            logging.warning(warning_message)
            
    def define_strategy_SMA(self):
        try:
            if not hasattr(self, 'data') or self.data is None:
                raise ValueError("Data is not initialized")
            data = self.data.copy()
        
            #******************** define your strategy here ************************
            data = data[["Close"]].copy()
            
            data["SMA_S"] = data.Close.rolling(window = self.SMA_S).mean()
            data["SMA_M"] = data.Close.rolling(window = self.SMA_M).mean()
            data["SMA_L"] = data.Close.rolling(window = self.SMA_L).mean()
            
            data.dropna(inplace = True)
                    
            cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
            cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)
            
            data["position"] = 0
            data.loc[cond1, "position"] = 1
            data.loc[cond2, "position"] = -1
            #***********************************************************************
            
            self.prepared_data = data.copy()
        except ValueError as e:
            warning_message = str(e)
            warnings.warn(warning_message, UserWarning)
            logging.warning(warning_message)

    def define_strategy_RSI(self):
        try:
            if not hasattr(self, 'data') or self.data is None:
                raise ValueError("Data is not initialized")
                    
            # Calculate RSI
            data = self.data[["Close"]].copy()
            data['RSI'] = ta.momentum.rsi(data['Close'], window=self.rsi_window)

            # Initialize position column
            data['position'] = 0
            
            cond1 = (data['RSI'] < self.rsi_lower)
            cond2 = (data['RSI'] > self.rsi_upper)

            data.loc[cond1, "position"] = 1
            data.loc[cond2, "position"] = -1

            self.prepared_data = data.copy()
        except ValueError as e:
            warning_message = str(e)
            warnings.warn(warning_message, UserWarning)
            logging.warning(warning_message)
            
    def define_strategy_MACD(self):
        try:
            # Ensure that data is initialized
            if not hasattr(self, 'data') or self.data is None:
                raise ValueError("Data is not initialized")

            # Calculate MACD and Signal line using TA-lib or pandas
            data = self.data[["Close"]].copy()

            data['MACD'] = ta.trend.macd(data['Close'], window_slow=self.macd_s, window_fast=self.macd_l)
            data['MACD_Signal'] = ta.trend.macd_signal(data['Close'], window_slow=self.macd_s, window_fast=self.macd_l, window_sign=macd_smooth)

            cond1 = (data['MACD'] > data['MACD_Signal'])
            cond2 = (data['MACD'] < data['MACD_Signal'])

            data['position'] = 0  
            data.loc[cond1, 'position'] = 1  
            data.loc[cond2, 'position'] = -1  
            self.prepared_data = data.copy()  # Save prepared data
            
        except ValueError as e:
            warning_message = str(e)
            warnings.warn(warning_message, UserWarning)
            logging.warning(warning_message)

    def define_strategy_VWAP(self):
        try:
            if not hasattr(self, 'data') or self.data is None:
                raise ValueError("Data is not initialized")

            data = self.data[["High","Low","Close", "Volume"]].copy()
            # Calculate the VWAP using the correct keyword arguments
            data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'],data['Close'], data['Volume'],int(self.vwap_period))  
    
            cond1 = (data['Close'] > data['VWAP'] + self.vwap_threshold)
            cond2 = (data['Close'] < data['VWAP'] - self.vwap_threshold)
    
            data['position'] = 0
            data.loc[cond1, "position"] = 1
            data.loc[cond2, "position"] = -1

            self.prepared_data = data.copy()
        except ValueError as e:
            warning_message = str(e)
            warnings.warn(warning_message, UserWarning)
            logging.warning(warning_message)

    def execute_trades(self):  
        if self.prepared_data["position"].iloc[-1] == 1: # if position is long -> go/stay long
            if self.position == 0:
                order = self.client.futures_create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING LONG")  
                self.stop_loss_price = self.prepared_data["Close"].iloc[-1] * (1 - self.stop_loss_pct)
            elif self.position == -1:
                order = self.client.futures_create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=2 * self.units)
                self.report_trade(order, "GOING LONG")
                self.stop_loss_price = self.prepared_data["Close"].iloc[-1] * (1 - self.stop_loss_pct)
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0: # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = self.client.futures_create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=2 * self.units)
                self.report_trade(order, "GOING NEUTRAL") 
                self.stop_loss_price = self.prepared_data["Close"].iloc[-1] * (1 + self.stop_loss_pct)
            elif self.position == -1:
                order = self.client.futures_create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=2 * self.units)
                self.report_trade(order, "GOING NEUTRAL") 
                self.stop_loss_price = self.prepared_data["Close"].iloc[-1] * (1 - self.stop_loss_pct)   
            self.position = 0
        if self.prepared_data["position"].iloc[-1] == -1: # if position is short -> go/stay short
            if self.position == 0:
                order = self.client.futures_create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING SHORT")
                self.stop_loss_price = self.prepared_data["Close"].iloc[-1] * (1 + self.stop_loss_pct)
            elif self.position == 1:
                order = self.client.futures_create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=2 * self.units)
                self.report_trade(order, "GOING SHORT")
                self.stop_loss_price = self.prepared_data["Close"].iloc[-1] * (1 + self.stop_loss_pct)
            self.position = -1
       
        if self.position == 1 and self.prepared_data["Close"].iloc[-1] <= self.stop_loss_price:
            order = self.client.futures_create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
            self.report_trade(order, "STOP LOSS HIT - CLOSING LONG POSITION")
            self.position = 0
            self.stop_loss_price = None
        elif self.position == -1 and self.prepared_data["Close"].iloc[-1] >= self.stop_loss_price:
            order = self.client.futures_create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
            self.report_trade(order, "STOP LOSS HIT - CLOSING SHORT POSITION")
            self.position = 0
            self.stop_loss_price = None
   
    def report_trade(self, order, going):
        self.trades += 1
        time.sleep(0.1)
        order_time = order["updateTime"]
        trades = self.client.futures_account_trades(symbol = self.symbol, startTime = order_time)
        order_time = pd.to_datetime(order_time, unit = "ms").strftime('%Y-%m-%d %H:%M')

        # extract data from trades object
        df = pd.DataFrame(trades)
        columns = ["qty", "quoteQty", "commission","realizedPnl"]
        for column in columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")
        base_units = round(df.qty.sum(), 5)
        quote_units = round(df.quoteQty.sum(), 5)
        commission = -round(df.commission.sum(), 5)
        real_profit = round(df.realizedPnl.sum(), 5)
        price = round(quote_units / base_units, 5)
        
        # calculate cumulative trading profits
        self.cum_profits += round((commission + real_profit), 5)
        
        # print trade report
        print(2 * "\n" + 100* "-")
        print("{} | {}".format(order_time, going)) 
        print("{} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(order_time, base_units, quote_units, price))
        print("{} | Trade Number = {} | Profit = {} | CumProfits = {} ".format(self.trades,order_time, real_profit, self.cum_profits))
        print(100 * "-" + "\n")
