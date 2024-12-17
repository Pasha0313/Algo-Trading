import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import time
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

class Futures_Backtester_PV():
    ''' Class for the vectorized backtesting of (levered) Futures trading strategies.
    filepath
    Attributes
    ============
    symbol: str
        ticker symbol (instrument) to be backtested
    start: str
        start date for data import
    end: str
        end date for data import
    tc: float
        proportional trading costs per trade
    '''    
    
    def __init__(self, client, symbol, bar_length, start, end, tc, leverage=5, strategy="PV"):
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
        return "\nFutures Backtester PV(symbol = {}, start = {}, end = {})\n".format(self.symbol, self.start, self.end)

##################################################################################
#    get_data:
#        imports the data.
##################################################################################
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

#########################################################################################################
#    test_strategy:
#        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).
#        Parameters
#        ============
#        percentiles: tuple (return_low_perc, return_high_perc, vol_low_perc, vol_high_perc)
#            return and volume percentiles to be considered for the strategy.
#        thresh: tuple (return_low_thresh, return_high_thresh, vol_low_thresh, vol_high_thesh)
#            return and volume thresholds to be considered for the strategy.
#########################################################################################################
    def test_strategy(self, percentiles = None, thresh = None):
        self.prepare_data(percentiles = percentiles, thresh = thresh)
        self.run_backtest()
        
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        self.print_performance()

##################################################################################        
#    prepare_data:
#        prepares the data for backtesting.
###################################################################################        
    def prepare_data(self, percentiles, thresh):
        ########################## Strategy-Specific #############################
        if not hasattr(self, 'data') or self.data is None:
            warnings.warn("Data is not initialized. Define 'data' before calling define_strategy_PV", UserWarning)
            return  # Exit the method to avoid further errors

        data = self.data[["Close", "Volume", "returns"]].copy()
        data["vol_ch"] = np.log(data.Volume.div(data.Volume.shift(1)))
        data.loc[data.vol_ch > 3, "vol_ch"] = np.nan
        data.loc[data.vol_ch < -3, "vol_ch"] = np.nan        
        
        if percentiles:
            self.return_thresh = np.percentile(data.returns.dropna(), [percentiles[0], percentiles[1]])
            self.volume_thresh = np.percentile(data.vol_ch.dropna(), [percentiles[2], percentiles[3]])
        elif thresh:
            self.return_thresh = [thresh[0], thresh[1]]
            self.volume_thresh = [thresh[2], thresh[3]]
                
        cond1 = data.returns <= self.return_thresh[0]
        cond2 = data.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])
        cond3 = data.returns >= self.return_thresh[1]
        
        data["position"] = 0
        data.loc[cond1 & cond2, "position"] = 1
        data.loc[cond3 & cond2, "position"] = -1
     
        self.results = data

##########################################################################
#   run_backtest:
#   runs the strategy backtest.
##########################################################################
    def run_backtest(self):
        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data["trades"] = data.position.diff().fillna(0).abs()
        data.strategy = data.strategy + data.trades * self.tc
        
        self.results = data

#######################################################################################################
#   plot_results:
#   plots the cumulative performance of the trading strategy compared to buy-and-hold.
#######################################################################################################
    def plot_strategy_comparison(self, leverage=False):
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

###############################################################################################################
#  optimize_strategy:
#  backtests strategy for different parameter values incl. optimization and reporting (wrapper).
#  Parameters
#  ============
#  return_low_range: tuple
#  tuples of the form (start, end, step size).
#  return_high_range: tuple
#            tuples of the form (start, end, step size).
#        vol_low_range: tuple
#            tuples of the form (start, end, step size).
#        vol_high_range: tuple
#            tuples of the form (start, end, step size).
#        metric: str
#            performance metric to be optimized (can be "Multiple" or "Sharpe")
################################################################################################################
    def optimize_strategy(self, return_low_range, return_high_range, vol_low_range, vol_high_range, metric = "Multiple"):
        print("\nOptimize Strategy is running. \n")
        self.metric = metric
        
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
        
        return_low_range = range(*return_low_range)
        return_high_range = range(*return_high_range)
        vol_low_range = range(*vol_low_range)
        vol_high_range = range(*vol_high_range)
        
        combinations = list(product(return_low_range, return_high_range, vol_low_range, vol_high_range))
         
        performance = []
        for comb in combinations:
            self.prepare_data(percentiles = comb, thresh = None)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))
    
        self.results_overview =  pd.DataFrame(data = np.array(combinations), columns = ["return_low", "return_high", "vol_low", "vol_high"])
        self.results_overview["performance"] = performance

        # Check performance values
        print(f"Performance values:\n{self.results_overview}")
        
        return_perc,vol_perc = self.find_best_strategy()
        return return_perc,vol_perc

#####################################################################################################
#    find_best_strategy:
#    finds the optimal strategy (global maximum).
##################################################################################################### 
    def find_best_strategy(self):
        best = self.results_overview.nlargest(1, "performance")
    
        # Convert numpy.int64 to native Python int
        return_perc = [int(best.return_low.iloc[0]), int(best.return_high.iloc[0])]
        vol_perc = [int(best.vol_low.iloc[0]), int(best.vol_high.iloc[0])]
        perf = best.performance.iloc[0]
    
        # Print formatted output
        print("Return_Perc: {} | Volume_Perc: {} | {}: {}\n".format(return_perc, vol_perc, self.metric, round(perf, 5)))
    
        # Call the method with Python int values
        self.test_strategy(percentiles=(return_perc[0], return_perc[1], vol_perc[0], vol_perc[1]))
        return return_perc,vol_perc
 
 ###############################################################################################
 #    add_sessions:
 #    adds/labels trading sessions and their compound returns.
 #    Parameter
 #    ============
 #    visualize: bool, default False
 #    if True, visualize compound session returns over time
 ###############################################################################################
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
            #save_path = "/home/saeed/Desktop/Personal/Trading/Analysis/With_leverage.png"
            save_path = "With_leverage.png"
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            plt.show()            
            plt.close(fig)  # Close the figure to free up resources

##################################################################################
#   add_leverage:
#   adds leverage to the strategy.]
#   Parameter
#        ============
#        leverage: float (positive)
#            degree of leverage.
#        
#        report: bool, default True
#            if True, print Performance Report incl. Leverage.
##################################################################################
    def add_leverage(self, leverage, report=True): 
        self.add_sessions()
        self.leverage = leverage
        
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


############################## Performance ######################################
#   print_performance:
#   calculates and prints various performance metrics.
############################## Performance ######################################
    def print_performance(self, leverage = False): # Adj
        data = self.results.copy()

        if leverage: # NEW!
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
        print("SIMPLE PRICE & VOLUME STRATEGY | INSTRUMENT = {}".format(self.symbol))
        print("THRESHOLDS = Return {}, Volume {}".format(np.round(self.return_thresh, 5), np.round(self.volume_thresh, 5)))
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
