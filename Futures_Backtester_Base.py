import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

class FuturesBacktesterBase:
    def __init__(self, client, symbol, bar_length, start, end=None, tc=0.0, leverage=5, strategy="PV"):
        self.client = client
        self.symbol = str(symbol)
        self.bar_length = str(bar_length)
        self.start = str(start)
        self.end = str(end) if end else None
        self.tc = tc
        self.leverage = leverage
        self.strategy = strategy
        self.data = None
        self.results = None

        # Validate input intervals
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        if self.bar_length not in self.available_intervals:
            raise ValueError(f"Invalid bar length: {self.bar_length}. Choose from {self.available_intervals}.")

        # Fetch data
        try:
            self.get_data()
            self.tp_year = self.calculate_trading_periodicity()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    def get_data(self):
        start_str = self.start
        end_str = self.end if self.end else None
        all_bars = []  
        current_start_str = start_str

        previous_candles_count = 0  # To track the number of candles collected
        print("\n")
        while True:
            print(f"Requesting data from {pd.to_datetime(current_start_str).strftime('%Y-%m-%d %H:%M')}...")
            bars = self.client.futures_historical_klines(
                symbol=self.symbol,interval=self.bar_length,
                start_str=current_start_str,end_str=end_str,limit=1000)

            if not bars:
                print("No more data available or the API limit has been reached.")
                break

            all_bars.extend(bars)
            last_timestamp = pd.to_datetime(bars[-1][0], unit="ms")
            current_start_str = (last_timestamp + pd.Timedelta(milliseconds=1)).strftime('%Y-%m-%d %H:%M')
            print(f"Collected {len(all_bars)} candles so far...")
            
            if len(all_bars) == previous_candles_count + 1:
                print("Only one new candle collected, exiting loop.")
                break
            previous_candles_count = len(all_bars)
            time.sleep(1)

        print(f"Total of {len(all_bars)} candles collected.\n")

        data = pd.DataFrame(all_bars)
        data["Date"] = pd.to_datetime(data.iloc[:, 0], unit="ms")
        data.columns = [
            "Open Time", "Open", "High", "Low", "Close", "Volume",
            "Close Time", "Quote Asset Volume", "Number of Trades",
            "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        data.set_index("Date", inplace=True)
        for column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data["returns"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Complete"] = [True for _ in range(len(data) - 1)] + [False]
        self.data = data

    def run_backtest(self):
        if self.results is None:
            raise ValueError("No strategy results available. Please generate results first.")
        
        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy += data.trades * self.tc
        self.results = data

    def add_leverage(self, leverage, report=True):
        self.leverage = leverage
        self.add_sessions()
        data = self.results.copy()
        data["simple_ret"] = np.exp(data.strategy) - 1
        data["eff_lev"] = leverage * (1 + data.session_compound) / (1 + data.session_compound * leverage)
        data.eff_lev.fillna(leverage, inplace=True)
        data.loc[data.trades != 0, "eff_lev"] = leverage
        levered_returns = data.eff_lev.shift() * data.simple_ret
        levered_returns = np.where(levered_returns < -1, -1, levered_returns)
        data["strategy_levered"] = levered_returns
        data["cstrategy_levered"] = data.strategy_levered.add(1).cumprod()
        self.results = data

        if report:
            self.print_performance(leverage=True)

    def add_sessions(self):
 
        if self.results is None:
            print("Run test_strategy() first.")
            
        data = self.results.copy()
        data["session"] = np.sign(data.trades).cumsum().shift().fillna(0)
        data["session_compound"] = data.groupby("session").strategy.cumsum().apply(np.exp) - 1
        self.results = data            

    def print_performance(self, leverage=False):
        if self.results is None:
            raise ValueError("No strategy results to analyze.")

        data = self.results.copy()
        to_analyze = np.log(data["strategy_levered"].add(1)) if leverage else data.strategy

        strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple = round(self.calculate_multiple(data["returns"]), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(to_analyze), 6)
        ann_mean = round(self.calculate_annualized_mean(to_analyze), 6)
        ann_std = round(self.calculate_annualized_std(to_analyze), 6)
        sharpe = round(self.calculate_sharpe(to_analyze), 6)

        print("=" * 100)
        print(f"STRATEGY PERFORMANCE | INSTRUMENT = {self.symbol}")
        print("-" * 100)
        print(f"Strategy Multiple:           {strategy_multiple}")
        print(f"Buy-and-Hold Multiple:       {bh_multiple}")
        print(f"Out-/Underperformance:       {outperf}")
        print(f"CAGR:                        {cagr}")
        print(f"Annualized Mean Return:      {ann_mean}")
        print(f"Annualized Standard Deviation: {ann_std}")
        print(f"Sharpe Ratio:                {sharpe}")
        print("=" * 100)

    def calculate_multiple(self, series):
        return np.exp(series.sum())

    def calculate_cagr(self, series):
        return np.exp(series.sum()) ** (1 / ((series.index[-1] - series.index[0]).days / 365.25)) - 1

    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year

    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)

    def calculate_sharpe(self, series):
        return self.calculate_cagr(series) / self.calculate_annualized_std(series)  
    
    def calculate_trading_periodicity(self):
        if self.data is None:
            raise ValueError("No data available to calculate periodicity.")
        return self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25)    

#######################################################################################################
#                                       plot_results
#######################################################################################################

    def plot_strategy_comparison(self, leverage=False):
        if self.results is None:
            logger.warning("Run test_strategy() first.")
        else:
            title = f"{self.symbol} | TC = {self.tc}"
            if leverage:
                title += f" | Leverage = {self.leverage}"

            plt.figure(figsize=(10, 6))
            plt.plot(self.results.index, self.results["creturns"], label="Buy and Hold")
            plt.plot(self.results.index, self.results["cstrategy"], label="Strategy")
            if leverage and "cstrategy_levered" in self.results.columns:
                plt.plot(self.results.index, self.results["cstrategy_levered"], label="Strategy Leverage")
            plt.xticks(rotation=45)
            plt.title(title)
            plt.legend()
            plt.show()

    def plot_results_II(self):
        ''' Plots a scatter plot of volume change against returns.
        '''
        if self.results is None:
            print("No data to plot. Please provide data.")
        else:
            plt.scatter(x=self.results['vol_ch'], y=self.results['returns'])
            plt.xlabel("Volume Change")
            plt.ylabel("Returns")
            plt.title(f"{self.symbol} | TC = {self.tc}")
            plt.show()

    def plot_heatmap(self):
        ''' Bins returns and volume change, creates a crosstab matrix, and plots a heatmap.
        '''
        if self.results is None:
            print("No data to process. Please provide data.")
        else:
            # Binning returns and volume change
            self.results["ret_cat"] = pd.qcut(self.results['returns'], q=10, labels=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            self.results["vol_cat"] = pd.qcut(self.results['vol_ch'], q=10, labels=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            
            # Creating crosstab matrix
            matrix_I = pd.crosstab(self.results['vol_cat'], self.results['ret_cat'])
            
            # Plotting the first heatmap
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1)
            sns.heatmap(matrix_I, cmap="RdYlBu_r", annot=True, robust=True, fmt=".0f")
            plt.title(f"Heatmap of Volume Change vs Returns | {self.symbol} | TC = {self.tc}")
            plt.xlabel("Return cat")
            plt.ylabel("Volume cat")
            plt.show()

            #matrix_II = pd.crosstab(self.results['vol_cat'].shift(), self.results['ret_cat'].shift(),values = self.results, aggfunc =np.mean)

            # Shifting categories and calculating the mean of the desired values
            shifted_results = self.results.shift()

            # Creating crosstab matrix for shifted data
            # need to be checked
            matrix_II = pd.crosstab(shifted_results['vol_cat'], shifted_results['ret_cat'], values=shifted_results['returns'], aggfunc=np.mean)
        

            # Plotting the second heatmap
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=0.75)
            sns.heatmap(matrix_II, cmap="RdYlBu", annot=True, robust=True, fmt=".3f")
            plt.title(f"Heatmap of Volume Change vs Returns | {self.symbol} | TC = {self.tc}")
            plt.xlabel("Return cat")
            plt.ylabel("Volume cat")
            plt.show()    
