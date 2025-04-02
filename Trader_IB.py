from ib_insync import *
import Strategy as strategy
import pandas as pd
import numpy as np
import time
from datetime import datetime

class FuturesTrader_IB:

    def __init__(self, ib, contract, bar_length, parameters, units, stop_trade_date, Total_stop_loss, stop_loss_pct,
                 Total_Take_Profit, Position_Long, Position_Neutral, Position_Short,
                 TN_trades=100, position=0, leverage=5, strategy="PV"):

        self.ib = ib
        self.contract = contract
        self.symbol = f"{contract.symbol}/{contract.currency}"
        self.bar_length = bar_length
        self.available_intervals = ["1 min", "2 mins", "3 mins", "5 mins", "15 mins", "30 mins", "1 hour", "1 day"]
        self.units = round(units, 3)
        self.position = position
        self.leverage = leverage
        self.cum_profits = 0
        self.eff_cum_profits = 0
        self.N_trades = 0
        self.TN_trades = TN_trades
        self.strategy = strategy
        self.stop_date = stop_trade_date
        self.Total_stop_loss = Total_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.Total_Take_Profit = Total_Take_Profit
        self.K = 0
        self.K_Threshold = 120
        self.Rep_Trade = pd.DataFrame()

        self.Position_Long = Position_Long
        self.Position_Neutral = Position_Neutral
        self.Position_Short = Position_Short
        self.parameters = parameters

        self.stop_loss_price = None
        self.current_price = None
        self.order = None
        self.filled_quantity = None

        self.data = pd.DataFrame()
        self.prepared_data = pd.DataFrame()

    def start_trading(self, duration):
        print("\nConnecting to IBKR and starting trading session...")
        
        # Step 1: Fetch historical data
        self.get_most_recent(duration)        
        self.stream_candles()

    def get_most_recent(self, duration):
        print("\nFetching historical data from IBKR...")
        bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=self.bar_length,
            whatToShow='MIDPOINT',
            useRTH=False,
            formatDate=1)

        if not bars:
            print("No data returned from IBKR.")
            return

        df = util.df(bars)
        df.set_index('date', inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df["Complete"] = [True for _ in range(len(df) - 1)] + [False]
        df.index = df.index.astype('int64')
        self.data = df
        self.prepared_data = df.copy()

        print(f"Collected {len(df)} bars from {pd.to_datetime(df.index.min())} to {pd.to_datetime(df.index.max())}.")
  
    def get_duration_from_bar_length(self, bar_length):
        bar_length = bar_length.strip().lower()
        mapping = {
            '1 min': '1 D', '2 mins': '1 D', '3 mins': '1 D', '5 mins': '1 D',
            '15 mins': '2 D', '30 mins': '3 D', '1 hour': '5 D', '1 day': '1 W'}
        return mapping.get(bar_length, '1 D')  

    def get_bar_timedelta(self):
        mapping = {
            "1 min": "1min",
            "2 mins": "2min",
            "3 mins": "3min",
            "5 mins": "5min",
            "15 mins": "15min",
            "30 mins": "30min",
            "1 hour": "1H",
            "1 day": "1D",
        }
        return pd.Timedelta(mapping.get(self.bar_length.lower(), "1min"))


    def stream_candles(self):
       
        print("Starting live market data stream (simulated with polling)...")
        while True:
            now = datetime.now()
            if now >= self.stop_date or self.N_trades >= self.TN_trades:
                self.close_all_positions()
                print("Trading stopped (time or trade limit reached).")
                break
            
            duration_str = self.get_duration_from_bar_length(self.bar_length)
            #print (f"duration_str ={duration_str},self.bar_length={self.bar_length}")
            bars = self.ib.reqHistoricalData(
                self.contract,
                endDateTime='',
                durationStr=duration_str,
                barSizeSetting=self.bar_length,
                whatToShow='MIDPOINT',
                useRTH=False,
                formatDate=1)

            if not bars:
                print("No new candle data.")
                time.sleep(10)
                continue

            df = util.df(bars)
            df.set_index('date', inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.columns = ["Open", "High", "Low", "Close", "Volume"]

            #df["Complete"] = [True for _ in range(len(df) - 1)] + [False]
            bar_duration = self.get_bar_timedelta()
            now = pd.Timestamp.utcnow()

            df["Complete"] = True  # default to complete
            if not df.empty:
                last_bar_time = pd.to_datetime(df.index[-1], unit='ns', utc=True)
                if now < last_bar_time + bar_duration:
                    df.loc[df.index[-1], "Complete"] = False

            latest = df.iloc[-1:]
            latest.index = latest.index.astype('int64')
            self.data = pd.concat([self.data, latest])
            self.data = self.data[~self.data.index.duplicated(keep='last')]
            self.data.sort_index(inplace=True)

            data = self.data.copy()
            print(".", end="", flush=True)
            self.K += 1
            if self.K == self.K_Threshold:
                self.K = 0
                self.report_strategy()

            if latest['Complete'].iloc[0]:
                if self.strategy == "PV":  # 0. **Simple Price and Volume**
                    self.prepared_data = strategy.define_strategy_PV(data, self.parameters)
                elif self.strategy == "SMA":  # 1. **Simple Moving Average**
                    self.prepared_data = strategy.define_strategy_SMA(data, self.parameters)
                elif self.strategy == "MACD":  # 2. **Moving Average Convergence Divergence (MACD) Histogram**
                    self.prepared_data = strategy.define_strategy_MACD(data, self.parameters)
                elif self.strategy == "RSI_MA":  # 3. **RSI with Moving Average**
                    self.prepared_data = strategy.define_strategy_RSI_MA(data, self.parameters)
                elif self.strategy == "RSI":  # 4. **Relative Strength Index with Divergence (RSI Divergence)**
                    self.prepared_data = strategy.define_strategy_RSI(data, self.parameters)
                elif self.strategy == "Stochastic_RSI":  # 5. **Stochastic RSI**
                    self.prepared_data = strategy.define_strategy_Stochastic_RSI(data, self.parameters)
                elif self.strategy == "Bollinger_Bands_ADX":  # 6. **Bollinger Bands with ADX Trend Filter**
                    self.prepared_data = strategy.define_strategy_Bollinger_ADX(data, self.parameters)
                elif self.strategy == "TEMA_momentum":  # 7. **MTriple Exponential Moving Average (TEMA) with Momentum Filter**
                    self.prepared_data = strategy.define_strategy_TEMA_momentum(data, self.parameters)
                elif self.strategy == "VWAP":  # 8. **Volume Weighted Average Price (VWAP)**
                    self.prepared_data = strategy.define_strategy_VWAP(data, self.parameters)
                elif self.strategy == "VWAP_momentum":  # 9. **Volume Weighted Average Price (VWAP) with Momentum**
                    self.prepared_data = strategy.define_strategy_VWAP_momentum(data, self.parameters)
                elif self.strategy == "Bollinger_breakout":  # 10. **Bollinger Bands Breakout**
                    self.prepared_data = strategy.define_strategy_Bollinger_breakout(data, self.parameters)
                elif self.strategy == "Bollinger_squeeze":  # 11. **Bollinger Bands Squeeze**
                    self.prepared_data = strategy.define_strategy_Bollinger_squeeze(data, self.parameters)
                elif self.strategy == "EMA_cross":  # 12. **Exponential Moving Average Cross Strategy**
                    self.prepared_data = strategy.define_strategy_EMA_cross(data, self.parameters)
                elif self.strategy == "EMA_envelope":  # 13. **Exponential Moving Average Envelope**
                    self.prepared_data = strategy.define_strategy_EMA_envelope(data, self.parameters)
                elif self.strategy == "TEMA":  # 14. **Triple Exponential Moving Average (TEMA)**
                    self.prepared_data = strategy.define_strategy_TEMA(data, self.parameters)
                elif self.strategy == "Donchian":  # 15. **Donchian Channel**
                    self.prepared_data = strategy.define_strategy_Donchian(data, self.parameters)
                elif self.strategy == "Aroon":  # 16. **Aroon Indicator**
                    self.prepared_data = strategy.define_strategy_Aroon(data, self.parameters)
                elif self.strategy == "WilliamsR":  # 17. **Williams %R**
                    self.prepared_data = strategy.define_strategy_WilliamsR(data, self.parameters)
                elif self.strategy == "Elder_Ray":  # 18. **Elder Ray Index**
                    self.prepared_data = strategy.define_strategy_Elder_Ray(data, self.parameters)
                elif self.strategy == "Klinger":  # 19. **Klinger Oscillator**
                    self.prepared_data = strategy.define_strategy_Klinger(data, self.parameters)
                elif self.strategy == "CMO":  # 20. **Chande Momentum Oscillator (CMO)**
                    self.prepared_data = strategy.define_strategy_CMO(data, self.parameters)
                elif self.strategy == "Price_Oscillator":  # 21. **Price Oscillator**
                    self.prepared_data = strategy.define_strategy_Price_Oscillator(data, self.parameters)
                elif self.strategy == "Ultimate_Oscillator":  # 22. **Ultimate Oscillator**
                    self.prepared_data = strategy.define_strategy_Ultimate_Oscillator(data, self.parameters)
                elif self.strategy == "Chaikin":  # 23. **Chaikin Oscillator**
                    self.prepared_data = strategy.define_strategy_Chaikin(data, self.parameters)
                elif self.strategy == "CMF":  # 24. **Chaikin Money Flow (CMF)**
                    self.prepared_data = strategy.define_strategy_CMF(data, self.parameters)
                elif self.strategy == "Fractal_Chaos":  # 25. **Fractal Chaos Bands**
                    self.prepared_data = strategy.define_strategy_Fractal_Chaos(data, self.parameters)
                elif self.strategy == "SuperTrend":  # 26. **SuperTrend**
                    self.prepared_data = strategy.define_strategy_SuperTrend(data, self.parameters)
                elif self.strategy == "ZigZag":  # 27. **ZigZag Indicator**
                    self.prepared_data = strategy.define_strategy_ZigZag(data, self.parameters)
                elif self.strategy == "Hull_MA":  # 28. **Hull Moving Average**
                    self.prepared_data = strategy.define_strategy_Hull_MA(data, self.parameters)
                elif self.strategy == "Gann_Fan":  # 29. **Gann Fan**
                    self.prepared_data = strategy.define_strategy_Gann_Fan(data, self.parameters)
                elif self.strategy == "ROC":  # 30. **Price Rate of Change (ROC)**
                    self.prepared_data = strategy.define_strategy_ROC(data, self.parameters)
                elif self.strategy == "MFI_divergence":  # 31. **MFI (Money Flow Index) Divergence**
                    self.prepared_data = strategy.define_strategy_MFI_divergence(data, self.parameters)
                elif self.strategy == "PSAR_simple":  # 32. **Parabolic SAR (PSAR)**
                    self.prepared_data = strategy.define_strategy_PSAR_simple(data, self.parameters)
                elif self.strategy == "CMF_ADX":  # 33. **Chaikin Money Flow (CMF) with ADX**
                    self.prepared_data = strategy.define_strategy_CMF_ADX(data, self.parameters)
                elif self.strategy == "PSAR_momentum":  # 34. **Parabolic SAR (PSAR) with Momentum**
                    self.prepared_data = strategy.define_strategy_PSAR_momentum(data, self.parameters)
                elif self.strategy == "Trix":  # 35. **Trix Indicator**
                    self.prepared_data = strategy.define_strategy_Trix(data, self.parameters)
                elif self.strategy == "Keltner_channel":  # 36. **Keltner Channel Breakout**
                    self.prepared_data = strategy.define_strategy_Keltner_channel(data, self.parameters)
                elif self.strategy == "Momentum":  # 37. **Momentum Strategy**
                    self.prepared_data = strategy.define_strategy_Momentum(data, self.parameters)
                elif self.strategy == "Ichimoku":  # 38. **Ichimoku Cloud**
                    self.prepared_data = strategy.define_strategy_Ichimoku(data, self.parameters)
                elif self.strategy == "Zscore":  # 39. **Z-Score Mean Reversion**
                    self.prepared_data = strategy.define_strategy_Zscore(data, self.parameters)
                elif self.strategy == "MA_envelope":  # 40. **Moving Average Envelope**
                    self.prepared_data = strategy.define_strategy_MA_envelope(data, self.parameters)
                elif self.strategy == "ATR":  # 41. **Average True Range (ATR) Breakout**
                    self.prepared_data = strategy.define_strategy_ATR(data, self.parameters)
                elif self.strategy == "ADX":  # 42. **Average Directional Index (ADX)**
                    self.prepared_data = strategy.define_strategy_ADX(data, self.parameters)
                elif self.strategy == "CCI":  # 43. **Commodity Channel Index (CCI)**
                    self.prepared_data = strategy.define_strategy_CCI(data, self.parameters)
                elif self.strategy == "Linear_Regression":  # 44. **Linear Regression Channel**
                    self.prepared_data = strategy.define_strategy_Linear_Regression(data, self.parameters)
                elif self.strategy == "VWMA_Price_Oscillator":  # 45. **Volume Weighted Moving Average (VWMA) with Price Oscillator**
                    self.prepared_data = strategy.define_strategy_VWMA_Price_Oscillator(data, self.parameters)
                elif self.strategy == "Dynamic_Pivot_Points":  # 46. **Dynamic Pivot Points Classic Strategy**
                    self.prepared_data = strategy.define_strategy_Dynamic_Pivot_Points_Classic(data, self.parameters)
                elif self.strategy == "Force_Index":  # 47. **Force Index**
                    self.prepared_data = strategy.define_strategy_Force_Index(data, self.parameters)
                elif self.strategy == "Chandelier_Exit":  # 48. **Chandelier Exit**
                    self.prepared_data = strategy.define_strategy_Chandelier_Exit(data, self.parameters)
                elif self.strategy == "Fibonacci":  # 49. **Fibonacci Retracement Levels**
                    self.prepared_data = strategy.define_strategy_Fibonacci(data, self.parameters)
                elif self.strategy == "ADL":  # 50. **Accumulation/Distribution Line (A/D Line)**
                    self.prepared_data = strategy.define_strategy_ADL(data, self.parameters)
                elif self.strategy == "RSI_Bollinger":  # 51. **RSI with Bollinger Bands**
                    self.prepared_data = strategy.define_strategy_RSI_Bollinger(data, self.parameters)
                elif self.strategy == "Turtle_Trading":  # 52. **Turtle Trading**
                    self.prepared_data = strategy.define_strategy_Turtle_Trading(data, self.parameters)
                elif self.strategy == "Mean_Reversion":  # 53. **Mean Reversion Strategy**
                    self.prepared_data = strategy.define_strategy_Mean_Reversion(data, self.parameters)
                elif self.strategy == "Breakout":  # 54. **Breakout Strategy**
                    self.prepared_data = strategy.define_strategy_Breakout(data, self.parameters)
                elif self.strategy == "RSI_Divergence":  # 55. **RSI Divergence Strategy**
                    self.prepared_data = strategy.define_strategy_RSI_Divergence(data, self.parameters)
                elif self.strategy == "MA_Cross_RSI":  # 56. **Moving Average Cross with RSI Filter**
                    self.prepared_data = strategy.define_strategy_MA_Cross_RSI(data, self.parameters)
                elif self.strategy == "ADX_MA":  # 57. **ADX with Moving Averages**
                    self.prepared_data = strategy.define_strategy_ADX_MA(data, self.parameters)
                elif self.strategy == "Bollinger_Breakout_Momentum":  # 58. **Bollinger Bands Breakout with Momentum Oscillator**
                    self.prepared_data = strategy.define_strategy_Bollinger_Breakout_Momentum_Oscillator(data, self.parameters)
                elif self.strategy == "Fibonacci_MA":  # 59. **Fibonacci Retracement with Moving Average Filter**
                    self.prepared_data = strategy.define_strategy_Fibonacci_MA(data, self.parameters)
                elif self.strategy == "Mean_Variance_Optimization":  # 60. **Mean-Variance Optimization Strategy**
                    self.prepared_data = strategy.define_strategy_Mean_Variance_Optimization(data, self.parameters)
                elif self.strategy == "MA_ribbon":  # 61. **Moving Average Ribbon**
                    self.prepared_data = strategy.define_strategy_MA_ribbon(data, self.parameters)
                elif self.strategy == "ADX_DI":  # 62. **ADX + DI (Directional Indicators)**
                    self.prepared_data = strategy.define_strategy_ADX_DI(data, self.parameters)
                elif self.strategy == "MACD_RSI":  # 63. **MACD Histogram with RSI**
                    self.prepared_data = strategy.define_strategy_MACD_RSI(data, self.parameters)
                elif self.strategy == "Fibonacci_retracement":  # 64. **Fibonacci Retracement Strategy**
                    self.prepared_data = strategy.define_strategy_Fibonacci_retracement(data, self.parameters)
                elif self.strategy == "RSI_trend_reversal":  # 65. **Relative Strength Index (RSI) Trend Reversal Strategy**
                    self.prepared_data = strategy.define_strategy_RSI_trend_reversal(data, self.parameters)
                elif self.strategy == "CMO_EMA":  # 66. **Chande Momentum Oscillator with EMA**
                    self.prepared_data = strategy.define_strategy_CMO_EMA(data, self.parameters)
                elif self.strategy == "MA_momentum":  # 67. **Moving Average Cross with Momentum**
                    self.prepared_data = strategy.define_strategy_MA_momentum(data, self.parameters)
                elif self.strategy == "RSI_Stochastic":  # 68. **RSI and Stochastic Oscillator**
                    self.prepared_data = strategy.define_strategy_RSI_Stochastic(data, self.parameters)
                elif self.strategy == "Garman_Klass":  # 69. **Garman-Klass Volatility Strategy**
                    self.prepared_data = strategy.define_strategy_Garman_Klass_Volatility(data, self.parameters)
                elif self.strategy == "Momentum_MACD":  # 70. **Momentum Oscillator with MACD**
                    self.prepared_data = strategy.define_strategy_Momentum_MACD(data, self.parameters)
                elif self.strategy == "Bollinger_Stochastic":  # 71. **Bollinger Bands with Stochastic Oscillator**
                    self.prepared_data = strategy.define_strategy_Bollinger_Stochastic(data, self.parameters)
                elif self.strategy == "Momentum_Breakout":  # 72. **Momentum Breakout Strategy**
                    self.prepared_data = strategy.define_strategy_Momentum_Breakout(data, self.parameters)
                elif self.strategy == "EMA_MACD":  # 73. **Exponential Moving Average Convergence Divergence (EMA MACD)**
                    self.prepared_data = strategy.define_strategy_EMA_MACD(data, self.parameters)
                elif self.strategy == "Bollinger_EMA":  # 74. **Bollinger Bands + EMA Cross**
                    self.prepared_data = strategy.define_strategy_Bollinger_EMA(data, self.parameters)
                elif self.strategy == "MA_Momentum":  # 75. **Moving Average Cross with Momentum Filter**
                    self.prepared_data = strategy.define_strategy_MA_Momentum_F(data, self.parameters)
                elif self.strategy == "Pivot_Stochastic":  # 76. **Pivot Points with Stochastic Oscillator**
                    self.prepared_data = strategy.define_strategy_Pivot_Stochastic(data, self.parameters)
                elif self.strategy == "VWMA":  # 77. **Volume Weighted Moving Average (VWMA)**
                    self.prepared_data = strategy.define_strategy_VWMA(data, self.parameters)
                elif self.strategy == "EMA_Momentum":  # 78. **Exponential Moving Average with Momentum**
                    self.prepared_data = strategy.define_strategy_EMA_Momentum(data, self.parameters)
                elif self.strategy == "RSI_A_MA":  # 79. **RSI and Moving Average**
                    self.prepared_data = strategy.define_strategy_RSI_A_MA(data, self.parameters)
                elif self.strategy == "EMA_Ribbon":  # 80. **Exponential Moving Average Ribbon**
                    self.prepared_data = strategy.define_strategy_EMA_Ribbon(data, self.parameters)
                elif self.strategy == "RSI_MA_Envelope":  # 81. **RSI and Moving Average Envelope**
                    self.prepared_data = strategy.define_strategy_RSI_MA_Envelope(data, self.parameters)
                elif self.strategy == "OBV_RSI":  # 82. **On-Balance Volume (OBV) with RSI**
                    self.prepared_data = strategy.define_strategy_OBV_RSI(data, self.parameters)
                elif self.strategy == "SuperTrend_RSI":  # 83. **SuperTrend with RSI**
                    self.prepared_data = strategy.define_strategy_ATR_RSI(data, self.parameters)
                elif self.strategy == "EMA_Bollinger":  # 84. **Exponential Moving Average with Bollinger Bands**
                    self.prepared_data = strategy.define_strategy_EMA_Bollinger(data, self.parameters)
                elif self.strategy == "RSI_MA_Ribbon":  # 85. **RSI and Moving Average Ribbon**
                    self.prepared_data = strategy.define_strategy_RSI_MA_Ribbon(data, self.parameters)
                elif self.strategy == "EMA_ADX":  # 86. **EMA Crossover with ADX Filter**
                    self.prepared_data = strategy.define_strategy_EMA_ADX(data, self.parameters)
                elif self.strategy == "RSI_Bollinger_Momentum":  # 87. **RSI with Bollinger Bands and Momentum Filter**
                    self.prepared_data = strategy.define_strategy_RSI_Bollinger_Momentum(data, self.parameters)
                elif self.strategy == "Renko_Box":  # 88. **Renko Box Trading Strategy**
                    self.prepared_data = strategy.define_strategy_Renko_Box_Trading(data, self.parameters)
                elif self.strategy == "ADX_Stochastic":  # 89. **ADX with Stochastic Oscillator**
                    self.prepared_data = strategy.define_strategy_ADX_Stochastic(data, self.parameters)
                elif self.strategy == "MA_Ribbon_ADX":  # 90. **Moving Average Ribbon with ADX Filter**
                    self.prepared_data = strategy.define_strategy_MA_Ribbon_ADX(data, self.parameters)
                elif self.strategy == "EMA_Stochastic":  # 91. **EMA with Stochastic Oscillator**
                    self.prepared_data = strategy.define_strategy_EMA_Stochastic(data, self.parameters)
                elif self.strategy == "RSI_ADX":  # 92. **RSI with ADX Filter**
                    self.prepared_data = strategy.define_strategy_RSI_ADX(data, self.parameters)
                elif self.strategy == "MACD_Stochastic":  # 93. **MACD with Stochastic Oscillator**
                    self.prepared_data = strategy.define_strategy_MACD_Stochastic(data, self.parameters)
                elif self.strategy == "MACD_Bollinger":  # 94. **MACD with Bollinger Bands Filter**
                    self.prepared_data = strategy.define_strategy_MACD_Bollinger(data, self.parameters)
                elif self.strategy == "EMA_Stochastic_Filter":  # 95. **EMA Cross with Stochastic Filter**
                    self.prepared_data = strategy.define_strategy_EMA_Stochastic_Filter(data, self.parameters)
                elif self.strategy == "MACD_MA_Ribbon":  # 96. **MACD with Moving Average Ribbon**
                    self.prepared_data = strategy.define_strategy_MACD_MA_Ribbon(data, self.parameters)
                elif self.strategy == "RSI_MACD_Combo":  # 97. **RSI and MACD Combo Strategy**
                    self.prepared_data = strategy.define_strategy_RSI_MACD_Combo(data, self.parameters)
                elif self.strategy == "Heikin_Ashi_Trend":  # 98. **Heikin Ashi Trend Continuation Strategy**
                    self.prepared_data = strategy.define_strategy_Heikin_Ashi_Trend_Continuation(data, self.parameters)
                elif self.strategy == "Bollinger_Stochastic_RSI":  # 99. **Bollinger Bands with Stochastic RSI**
                    self.prepared_data = strategy.define_strategy_Bollinger_Stochastic_RSI(data, self.parameters)
                elif self.strategy == "Trend_Reversal_RSI":  # 100. **Trend Reversal with RSI**
                    self.prepared_data = strategy.define_strategy_Trend_Reversal_RSI(data, self.parameters)
                elif self.strategy == "Volume_Profile":  # 101. **Volume Profile Strategy**
                    self.prepared_data = strategy.define_strategy_Volume_Profile(data, self.parameters) 
                elif self.strategy == "Grid_Trading":  # 102. **Grid Trading Strategy**
                    self.prepared_data = strategy.define_strategy_Grid_Trading(data, self.parameters)                                          
                elif self.strategy == "EMA_MACD_ADX":  # 103. **EMA + MACD + ADX Hybrid Strategy**
                    self.prepared_data = strategy.define_strategy_EMA_MACD_ADX(data, self.parameters) 
                elif self.strategy == "Trend_Momentum_Volatility":  # 104. **EMA + MACD + ADX + ATR Stochastic + RSI Hybrid Strategy **
                    self.prepared_data = strategy.define_strategy_Trend_Momentum_Volatility(data, self.parameters)                                          
                elif self.strategy == "Stochastic_RSI_Bollinger_VWAP":  # 105. ** Stochastic RSI Bollinger VWAP Hybrid Strategy **
                    self.prepared_data = strategy.define_strategy_Stochastic_RSI_Bollinger_VWAP(data, self.parameters)       
                elif self.strategy == "Stochastic_RSI_FULL":  #106. ** Stochastic RSI Strategy with %K and %D smoothing **
                    self.prepared_data = strategy.define_strategy_Stochastic_RSI_FULL(data, self.parameters) 
                elif self.strategy == "Gaussian_Channel_FULL":  #107. **  Gaussian Channel strategy **
                    self.prepared_data = strategy.define_strategy_Gaussian_Channel_FULL(data, self.parameters) 
                elif self.strategy == "Combined_Gaussian_Stochastic_RSI_FULL":  #108. ** combined the Gaussian Channel and Stochastic RSI strategies **
                    self.prepared_data = strategy.define_strategy_Combined_Gaussian_Stochastic_RSI_FULL(data, self.parameters) 
                elif self.strategy == "OBV": # 109. **On-Balance Volume (OBV) **
                    self.prepared_data = strategy.define_strategy_OBV(data)
                elif self.strategy == "Volume_delta": # 110. ** Volum Delta **
                    self.prepared_data = strategy.define_strategy_volume_delta(data)
                elif self.strategy == "Ease_of_Movement": # 111. ** Ease of Movement **
                    self.prepared_data = strategy.define_strategy_ease_of_movement(data, self.parameters)
                elif self.strategy == "WMA": # 112. ** Weight Moving Average **
                    self.prepared_data = strategy.define_strategy_wma(data,self.parameters)
                elif self.strategy == "EMA": # 113. ** Exponential Moving Average **
                    self.prepared_data = strategy.define_strategy_ema(data,self.parameters)
                elif self.strategy == "DEMA": # 114. ** Double Exponential Moving Average **
                    self.prepared_data = strategy.define_strategy_dema(data, self.parameters)
                elif self.strategy == "AMA": # 115. ** Adaptive Moving Average **
                    self.prepared_data = strategy.define_strategy_ama(data, self.parameters)
                elif self.strategy == "VIDYA": # 116. ** Variable Index Dynamic Average (VIDYA) **
                    self.prepared_data = strategy.define_strategy_vidya(data, self.parameters)
                elif self.strategy == "SMA_cross":  # 117. ** Simple Moving Average Cross **
                    self.prepared_data = strategy.define_strategy_SMA_cross(data, self.parameters)
                elif self.strategy == "Stochastic": # 118. **Stochastic Oscillator Strategy**
                    self.prepared_data = strategy.define_strategy_Stochastic(data, self.parameters)
                elif self.strategy == "AO": # 119. **Awesome Oscillator (AO) Strategy**
                    self.prepared_data = strategy.define_strategy_AO(data, self.parameters)
                elif self.strategy == "KST": # 120. **Know Sure Thing (KST) Strategy**
                    self.prepared_data = strategy.define_strategy_KST(data, self.parameters)
                elif self.strategy == "Bollinger_SMA": # 121. **Bollinger Bands with SMA Strategy**
                    self.prepared_data = strategy.define_strategy_Bollinger(data, self.parameters)
                elif self.strategy == "Bollinger_Keltner_Squeeze": # 122. **Bollinger Bands & Keltner Channel Squeeze Strategy**
                    self.prepared_data = strategy.define_strategy_Squeeze(data, self.parameters)
                elif self.strategy == "StdDev_Channel": # 123. **Standard Deviation Channel Strategy**
                    self.prepared_data = strategy.define_strategy_StdDev_Channel(data, self.parameters)
                elif self.strategy == "HV": # 124. **Historical Volatility (HV) Strategy**
                    self.prepared_data = strategy.define_strategy_HV(data, self.parameters)
                elif self.strategy == "VR": # 125. ** Volatility Ratio (VR) Strategy **
                    self.prepared_data = strategy.define_strategy_VR(data, self.parameters)
                elif self.strategy == "Simple_Pivot_Points": # 125. ** Simple_Pivot_Points Strategy **
                    self.prepared_data = strategy.define_strategy_Simple_Pivot_Points(data)
                elif self.strategy == "DI": # 125. ** Directional Indicator (DI-Only) Strategy **
                    self.prepared_data = strategy.define_strategy_DI(data, self.parameters)
                elif self.strategy == "Stochastic_RSI_StdDev_Channel": # 128. ** Stochastic RSI with Standard Deviation Channel Strategy **
                    self.prepared_data = strategy.define_strategy_Stochastic_RSI_StdDev_Channel(data, self.parameters)  
                elif self.strategy == "Bollinger_Stochastic_RSI_Modified": #129. **Bollinger Bands with Stochastic RSI modified**
                    self.prepared_data = strategy.define_strategy_Bollinger_Stochastic_RSI_modified(data, self.parameters)                                                
                elif self.strategy == "Keltner_Stochastic_RSI": #130. **Keltner Channel Calculation Bands with Stochastic RSI**
                    self.prepared_data = strategy.define_strategy_Keltner_Stochastic_RSI(data, self.parameters)  
                elif self.strategy == "HMA_Stochastic_RSI": # 131. **Hull Moving Average Channel with Stochastic RSI**
                    self.prepared_data = strategy.define_strategy_HMA_StochRSI(data, self.parameters)   
                elif self.strategy == "ADX_ATR_Bollinger_Stochastic_RSI":  # 132. **ADX ATR Bollinger Bands with Stochastic RSI**
                    self.prepared_data = strategy.define_strategy_ADX_ATR_Bollinger_Stochastic_RSI(data, self.parameters)   
                elif self.strategy == "Supertrend_Stochastic_RSI":  # 133. **SuperTrend with Stochastic RSI**
                    self.prepared_data = strategy.define_strategy_Supertrend_Stochastic_RSI(data, self.parameters)   
                self.execute_trades()

            time.sleep(1)

    def execute_trades(self):
        self.current_price = self.prepared_data["Close"].iloc[-1]
        self.Total_stop_loss = self.current_price * self.stop_loss_pct
        signal = self.prepared_data["position"].iloc[-1]

        if self.Position_Long and signal == 1:
            self.place_order('BUY')

        elif self.Position_Short and signal == -1:
            self.place_order('SELL')

        elif self.Position_Neutral and signal == 0:
            self.place_order('NEUTRAL')

        if self.order is not None:
            pnl = self.report_pnl()
            if pnl > self.Total_Take_Profit:
                if self.position == 1:
                    self.place_order('NEUTRAL')
                    self.report_trade("PNL HIT - CLOSING LONG POSITION")
                elif self.position == -1:
                    self.place_order('NEUTRAL')
                    self.report_trade("PNL HIT - CLOSING SHORT POSITION")

    def place_order(self, direction):
        qty = self.units
        if direction == 'BUY':
            if self.position <= 0:
                order = MarketOrder('BUY', qty * (2 if self.position == -1 else 1))
                self.ib.placeOrder(self.contract, order)
                self.order = order
                self.report_trade("GOING LONG")
                self.position = 1

        elif direction == 'SELL':
            if self.position >= 0:
                order = MarketOrder('SELL', qty * (2 if self.position == 1 else 1))
                self.ib.placeOrder(self.contract, order)
                self.order = order
                self.report_trade("GOING SHORT")
                self.position = -1

        elif direction == 'NEUTRAL':
            if self.position != 0:
                side = 'SELL' if self.position == 1 else 'BUY'
                order = MarketOrder(side, qty)
                self.ib.placeOrder(self.contract, order)
                self.order = order
                self.report_trade("GOING NEUTRAL")
                self.position = 0
                self.stop_loss_price = None

    def close_all_positions(self):
        if self.position != 0:
            self.place_order('NEUTRAL')

    def report_strategy(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = self.prepared_data
        excluded_columns = ['Open', 'High', 'Low', 'Complete']
        filtered_data = data.drop(columns=excluded_columns, errors='ignore')
        remaining_columns = filtered_data.columns
        separator = "-" * 50
        print("\n" + separator)
        print(f"| {'Strategy Report'.center(46)} |")
        print(separator)
        print(f"| Trade Number        : {self.N_trades:<24} |")
        print(f"| Time                : {now:<24} |")
        print(separator)
        for col in remaining_columns:
            last_value = filtered_data[col].iloc[-1]
            formatted_value = f"{last_value:.5f}" if isinstance(last_value, float) else str(last_value)
            print(f"| {col:<20}: {formatted_value:<24} |")
        print(separator + "\n")

    def report_trade(self, action):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'%'*100}\n[{now}] {action} | Current Price: {self.current_price:.2f} | Trades: {self.N_trades + 1}")
        self.N_trades += 1
        print("%"*100 + "\n")

    def report_pnl(self):
        fills = self.ib.fills()
        pnl = 0.0
        for fill in fills:
            if fill.contract.symbol == self.contract.symbol:
                pnl += fill.execution.cumQty * fill.execution.avgPrice  
        return pnl
