import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import pandas as pd
import warnings
import ta
import time

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


class Futures_Backtester_MACD():

    def __init__(self, client, symbol, bar_length, start, end, tc, leverage=5, strategy="MACD"):
        self.client = client
        self.symbol = str(symbol)
        self.bar_length = str(bar_length)
        self.start = str(start)
        self.end = str(end) if end else None
        self.tc = tc
        self.leverage = leverage
        self.results = None
        self.get_data()
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

    def __repr__(self):
        return f"Futures Backtester MACD(symbol = {self.symbol}, start = {self.start}, end = {self.end})"

    def get_data(self):
        start_str = self.start
        end_str = self.end if self.end else None
        all_bars = []  
        current_start_str = start_str

        previous_candles_count = 0  
        print("\n")
        while True:
            # Fetch a chunk of data (up to 1000 candles)
            print(f"Requesting data from {pd.to_datetime(current_start_str).strftime('%Y-%m-%d %H:%M')}...")

            bars = self.client.futures_historical_klines(symbol=self.symbol,interval=self.bar_length,
                start_str=current_start_str,end_str=end_str,limit=1000)

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

        data = pd.DataFrame(all_bars)
        data["Date"] = pd.to_datetime(data.iloc[:, 0], unit="ms")
        data.columns = [
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"
        ]
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        data.set_index("Date", inplace=True)
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Complete"] = [True for _ in range(len(data) - 1)] + [False]
        self.data = data

    def test_strategy(self, macd_slow, macd_fast, macd_signal):
        ''' Tests the MACD strategy with given parameters. '''
        self.macd_slow = macd_slow
        self.macd_fast = macd_fast
        self.macd_signal = macd_signal

        self.prepare_data(macd_slow=macd_slow, macd_fast=macd_fast, macd_signal=macd_signal)
        self.run_backtest()
        
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

        self.results = data        

        self.plot_results()
        self.print_performance()

    def prepare_data(self, macd_slow, macd_fast, macd_signal):
        ''' Prepares data for backtesting based on MACD strategy. '''
        
        data = self.data.copy()
        data['MACD'] = ta.trend.macd(data['Close'], window_slow=macd_slow, window_fast=macd_fast)
        data['MACD_Signal'] = ta.trend.macd_signal(data['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)

        # Trading Signals
        data['position'] = 0  # Default to neutral/no position
        data.loc[data['MACD'] > data['MACD_Signal'], 'position'] = 1  # Long when MACD > MACD Signal
        data.loc[data['MACD'] < data['MACD_Signal'], 'position'] = -1  # Short when MACD < MACD Signal

        self.results = data

    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''

        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc
       
        self.results = data

    def plot_strategy_comparison(self, leverage=False):
        ''' 
        Plots the cumulative performance of the trading strategy compared to buy-and-hold.
    
        Parameters
        ==========
        leverage: bool, default False
            If True, plot the performance with leverage.
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = f"{self.symbol} | TC = {self.tc}"
            if leverage:
                title += f" | Leverage = {self.leverage}"

            plt.figure(figsize=(6, 4))

            # Convert index and column data to numpy arrays for better performance
            index = self.results.index.to_numpy()
            creturns = self.results["creturns"].to_numpy()
            cstrategy = self.results["cstrategy"].to_numpy()

            # Plot Buy and Hold vs. Strategy
            plt.plot(index, creturns, label="Buy and Hold")
            plt.plot(index, cstrategy, label="Strategy")

            # Optionally plot strategy with leverage
            if leverage:
                if "cstrategy_levered" in self.results.columns:
                    cstrategy_levered = self.results["cstrategy_levered"].to_numpy()
                    plt.plot(index, cstrategy_levered, label="Strategy with Leverage")
                else:
                    print("Leverage data is not available. Run 'add_leverage()' first.")

            plt.title(title)
            plt.legend()
            save_path = "strategy_comparison.png"
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            plt.show()

            plt.close()  # Close the figure to free up resources

    def plot_results(self):
        ''' 
        Plots the backtest results for the chosen trading strategy (RSI or MACD).
        '''
        if self.results is None:
            print("Run test_strategy() first.")
            return
    
        # Calculate cumulative returns for strategy
        self.results['Total'] = (self.results['strategy'] + 1).cumprod()

        # Set up the main plot
        fig, ax1 = plt.subplots(figsize=(6, 4))

        # Plot close price
        ax1.plot(self.results.index, self.results['Close'], label='Close Price', color='b')
        ax1.set_ylabel('Price')
        ax1.set_xlabel('Date')
        ax1.legend(loc='upper left')

        # Plot strategy returns on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(self.results.index, self.results['Total'], label='Strategy Returns', color='r')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend(loc='upper right')

        # Set plot title based on strategy type
        plt.title('MACD Strategy Backtest')
        # Show plot
        plt.show()
        plt.close(fig)  # Close the figure to free up resources


    def optimize_strategy(self, macd_slow_range, macd_fast_range, macd_signal_range):
        ''' Optimizes the MACD strategy parameters. '''
        
        macd_slow_range = range(*macd_slow_range)
        macd_fast_range = range(*macd_fast_range)
        macd_signal_range = range(*macd_signal_range)
        
        combinations = list(product(macd_slow_range, macd_fast_range, macd_signal_range))
        
        performance = []
        for comb in combinations:
            self.prepare_data(macd_slow=comb[0], macd_fast=comb[1], macd_signal=comb[2])
            self.run_backtest()
            performance.append(self.results['strategy'].sum())
        
        self.results_overview = pd.DataFrame(data=np.array(combinations), columns=["macd_slow", "macd_fast", "macd_signal"])
        self.results_overview["performance"] = performance

        # Check performance values
        print(f"Performance values:\n{self.results_overview}")
                
        macd_slow,macd_fast,macd_signal = self.find_best_strategy()
        return macd_slow,macd_fast,macd_signal

    def find_best_strategy(self):
        ''' Finds the optimal strategy (global maximum). '''
        best = self.results_overview.nlargest(1, "performance")
        macd_s = best.macd_s.iloc[0]
        macd_l = best.macd_l.iloc[0]
        macd_smooth = best.macd_smooth.iloc[0]
        perf = best.performance.iloc[0]
        
        print("MACD_S: {} | MACD_L: {} | MACD_Smooth: {} | {}: {}\n".format(
            macd_s, macd_l, macd_smooth, self.metric, round(perf, 5)))
        
        # Test the strategy with the best parameters
        self.test_strategy(macd_s=macd_s, macd_l=macd_l, macd_smooth=macd_smooth)
        return macd_s, macd_l, macd_smooth

#################################### Add Sessions and Leverage ####################################
    def add_sessions(self, visualize=True):

        if self.results is None:
            print("Run test_strategy() first.")
            return
     
        data = self.results.copy()
    
        # Identify trading sessions based on trade signals
        data["session"] = np.sign(data.trades).cumsum().shift().fillna(0)
    
        # Calculate compound returns for each session
        data["session_compound"] = data.groupby("session").strategy.cumsum().apply(np.exp) - 1
        self.results = data
    
        if visualize:
            fig, ax = plt.subplots(figsize=(6, 4))
            data["session_compound"].plot(ax=ax)
            ax.set_xlabel("Time")
            ax.set_ylabel("Compound Returns")
            ax.set_title("Compound Returns Over Time")
            plt.tight_layout()
            save_path = "MACD_strategy_with_leverage.png"
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            plt.show()
        
            plt.close(fig)  # Close the figure to free up resources

    def add_leverage(self, leverage, report = True): # NEW!!!
        ''' 
        Adds Leverage to the Strategy.
        
        Parameter
        ============
        leverage: float (positive)
            degree of leverage.
        
        report: bool, default True
            if True, print Performance Report incl. Leverage.
        '''
        self.add_sessions()
        self.leverage = leverage
        
        data = self.results.copy()
        data["simple_ret"] = np.exp(data.strategy) - 1
        data["eff_lev"] = leverage * (1 + data.session_compound) / (1 + data.session_compound * leverage)
        data.eff_lev.fillna(leverage, inplace = True)
        data.loc[data.trades !=0, "eff_lev"] = leverage
        levered_returns = data.eff_lev.shift() * data.simple_ret
        levered_returns = np.where(levered_returns < -1, -1, levered_returns)
        data["strategy_levered"] = levered_returns
        data["cstrategy_levered"] = data.strategy_levered.add(1).cumprod()
        
        self.results = data
            
        if report:
            self.print_performance(leverage = True)

    #################################### Performance ####################################

    def print_performance(self, leverage=False):
        ''' Calculates and prints various Performance Metrics for the MACD Strategy. '''
    
        data = self.results.copy()

        if leverage:  # If leverage is applied, calculate based on leveraged returns
            to_analyze = np.log(data.strategy_levered.add(1))
        else: 
            to_analyze = data.strategy

        # Calculate performance metrics
        strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple = round(self.calculate_multiple(data.returns), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(to_analyze), 6)
        ann_mean = round(self.calculate_annualized_mean(to_analyze), 6)
        ann_std = round(self.calculate_annualized_std(to_analyze), 6)
        sharpe = round(self.calculate_sharpe(to_analyze), 6)
   
        # Print performance summary
        print(100 * "=")
        print("MACD STRATEGY | INSTRUMENT = {}".format(self.symbol))
        print("PARAMETERS = MACD_Slow {}, MACD_Fast {}, MACD_Signal {}".format(self.macd_slow, self.macd_fast, self.macd_signal))
        print(100 * "-")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(outperf))
        print("\n")
        print("CAGR:                        {}".format(cagr))
        print("Annualized Mean:             {}".format(ann_mean))
        print("Annualized Std:              {}".format(ann_std))
        print("Sharpe Ratio:                {}".format(sharpe))
        print(100 * "=")
    
    def calculate_multiple(self, series):
        ''' Calculates the performance multiple of the given series. '''
        return np.exp(series.sum())

    def calculate_cagr(self, series):
        ''' Calculates the Compound Annual Growth Rate (CAGR) of the given series. '''
        if ((series.index[-1] - series.index[0]).days == 0) :
            return np.exp(series.sum())**(1/(1 / 365.25)) - 1
        else :
            return np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1
   
    def calculate_annualized_mean(self, series):
        ''' Calculates the annualized mean of the given series. '''
        return series.mean() * self.tp_year

    def calculate_annualized_std(self, series):
        ''' Calculates the annualized standard deviation of the given series. '''
        return series.std() * np.sqrt(self.tp_year)

    def calculate_sharpe(self, series):
        ''' Calculates the Sharpe Ratio of the given series. '''
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)

