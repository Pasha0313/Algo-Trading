import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import pandas as pd
import seaborn as sns
import time
import warnings
import ta
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

class Futures_Backtester_RSI():
 
    def __init__(self, client, symbol, bar_length, start, end, tc, leverage=5, strategy="RSI"):
        self.client = client  # Store the client object
        self.symbol = str(symbol)
        self.bar_length = str(bar_length)
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.start = str(start)
        self.end = str(end) if end else None
        self.tc = tc
        self.leverage = leverage
        self.results = None
        self.get_data()
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

        self.strategy = strategy
        #************************************************************************

        # Stop loss parameters
        self.stop_loss_price = None

    def __repr__(self):
        return "\nFutures Backtester RSI(symbol = {}, start = {}, end = {})\n".format(self.symbol, self.start, self.end)
        
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

    def test_strategy(self, rsi_window, rsi_lower, rsi_upper):

        self.rsi_window = rsi_window
        self.rsi_lower  = rsi_lower
        self.rsi_upper = rsi_upper

        ''' Tests the strategy with given RSI parameters. '''
        self.prepare_data(rsi_window=rsi_window, rsi_lower=rsi_lower, rsi_upper=rsi_upper)
        self.run_backtest()

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        self.plot_results()        
        self.print_performance()

    def prepare_data(self, rsi_window, rsi_lower, rsi_upper):
        
        ########################## Strategy-Specific #############################
        if not hasattr(self, 'data') or self.data is None:
            warnings.warn("Data is not initialized. Define 'data' before calling define_strategy_RSI", UserWarning)
            return  # Exit the method to avoid further errors
        
        # Calculate RSI
        data = self.data[["Close", "returns"]].copy()
        data['RSI'] = ta.momentum.rsi(data['Close'], window=rsi_window)
        
        # Initialize position column
        data['position'] = 0
        
        cond1 = (data['RSI'] < self.rsi_lower)
        cond2 = (data['RSI'] > self.rsi_upper)

        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1

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
        ''' Plots the cumulative performance of the trading strategy compared to buy-and-hold.
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = f"{self.symbol} | TC = {self.tc}"
            if leverage:
                title += f" | Leverage = {self.leverage}"

            plt.figure(figsize=(8, 6))

            # Convert index and column data to numpy arrays
            index = self.results.index.to_numpy()
            creturns = self.results["creturns"].to_numpy()
            cstrategy = self.results["cstrategy"].to_numpy()

            plt.plot(index, creturns, label="Buy and Hold")
            plt.plot(index, cstrategy, label="Strategy")

            if leverage:
                cstrategy_levered = self.results["cstrategy_levered"].to_numpy()
                plt.plot(index, cstrategy_levered, label="Strategy leverage")
                
            plt.title(title)
            plt.legend()
            plt.xticks(rotation=45)  
            save_path = "strategy_comparison.png"
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            plt.show()    
            
            plt.close()  # Close the figure to free up resources

    def plot_results(self):
        ''' Plots the backtest results. '''
        
        self.results['Total'] = (self.results['strategy'] + 1).cumprod()
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        ax1.plot(self.results.index, self.results['Close'], label='Close Price', color='b')
        ax1.set_ylabel('Price')
        ax1.set_xlabel('Date')
        ax1.legend(loc='best')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = ax1.twinx()
        ax2.plot(self.results.index, self.results['Total'], label='Strategy Returns', color='r')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend(loc='best')
        plt.xticks(rotation=45)  
        plt.title('RSI Strategy Backtest')
        plt.show()


    def optimize_strategy(self, rsi_window_range, rsi_lower_range, rsi_upper_range, metric="Multiple"):
        print("\nOptimize Strategy is running.\n")

        self.metric = metric
        
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe

        # Ensure correct ranges
        rsi_window_range = range(*rsi_window_range)
        rsi_lower_range = range(*rsi_lower_range)
        rsi_upper_range = range(*rsi_upper_range)
        
        combinations = list(product(rsi_window_range, rsi_lower_range, rsi_upper_range))

        performance = []
        for comb in combinations:
            self.prepare_data(rsi_window=comb[0], rsi_lower=comb[1], rsi_upper=comb[2])
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))

        self.results_overview = pd.DataFrame(data=np.array(combinations), columns=["rsi_window", "rsi_lower", "rsi_upper"])
        self.results_overview["performance"] = performance

        # Check performance values
        print(f"Performance values:\n{self.results_overview}")

        rsi_window, rsi_lower, rsi_upper = self.find_best_strategy()
        return rsi_window, rsi_lower, rsi_upper

    def find_best_strategy(self):
        ''' Finds the optimal strategy (global maximum). '''
        # Sorting to ensure the correct best strategy is selected
        best = self.results_overview.sort_values(by="performance", ascending=False).iloc[0]
        rsi_window = best.rsi_window
        rsi_lower = best.rsi_lower
        rsi_upper = best.rsi_upper
        perf = best.performance
        
        print(f"Best combination: RSI_Window: {rsi_window} | RSI_Lower: {rsi_lower} | RSI_Upper: {rsi_upper} | {self.metric}: {round(perf, 5)}\n")

        self.test_strategy(rsi_window=rsi_window, rsi_lower=rsi_lower, rsi_upper=rsi_upper)
        return rsi_window, rsi_lower, rsi_upper
 
    def add_sessions(self, visualize = True): # NEW!!!
       
        if self.results is None:
            print("Run test_strategy() first.")
            
        data = self.results.copy()
        data["session"] = np.sign(data.trades).cumsum().shift().fillna(0)
        data["session_compound"] = data.groupby("session").strategy.cumsum().apply(np.exp) - 1
        self.results = data
        if visualize:
            fig, ax = plt.subplots(figsize=(6, 4))
            data["session_compound"].plot(ax=ax)
            ax.set_xlabel("Time")
            ax.set_ylabel("Compound Returns")
            ax.set_title("Compound Returns Over Time")
            plt.tight_layout()
            plt.xticks(rotation=45)  
            save_path = "With_leverage.png"
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

    ############################## Performance ######################################
    def print_performance(self, leverage = False): # Adj
        ''' Calculates and prints various Performance Metrics. '''
        
        data = self.results.copy()

        if leverage:  # NEW!
            to_analyze = np.log(data.strategy_levered.add(1))
        else: 
            to_analyze = data.strategy

        strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple =       round(self.calculate_multiple(data.returns), 6)
        outperf =           round(strategy_multiple - bh_multiple, 6)
        cagr =              round(self.calculate_cagr(to_analyze), 6)
        ann_mean =          round(self.calculate_annualized_mean(to_analyze), 6)
        ann_std =           round(self.calculate_annualized_std(to_analyze), 6)
        sharpe =            round(self.calculate_sharpe(to_analyze), 6)
       
        print(100 * "=")
        print("RSI STRATEGY | INSTRUMENT = {}".format(self.symbol))
        print("PARAMETERS = RSI_Window {}, RSI_Lower {}, RSI_Upper {}".format(self.rsi_window,self.rsi_lower,self.rsi_upper ))
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
        return np.exp(series.sum())
    
    def calculate_cagr(self, series):
        if ((series.index[-1] - series.index[0]).days == 0) :
            return np.exp(series.sum())**(1/(1 / 365.25)) - 1
        else :
            return np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1
   
    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year
    
    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)
    
    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return self.calculate_cagr(series) / self.calculate_annualized_std(series)
