from Futures_Backtester_Base import FuturesBacktesterBase
from itertools import product
import numpy as np
import pandas as pd
import warnings
import ta
warnings.filterwarnings("ignore")

class Futures_Backtester_VWAP(FuturesBacktesterBase):
     
    def __init__(self, client, symbol, bar_length, start, end, tc, leverage=5, strategy="VWAP"):
        super().__init__(client, symbol, bar_length, start, end, tc, leverage, strategy)

    def __repr__(self):
        return "\nFutures Backtester VWAP(symbol = {}, start = {}, end = {})\n".format(self.symbol, self.start, self.end)
        
    def test_strategy(self, vwap_period, vwap_threshold):
        self.prepare_data(vwap_period=vwap_period, vwap_threshold=vwap_threshold)
        self.run_backtest()
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        self.print_performance()

    def prepare_data(self, vwap_period, vwap_threshold):

        data = self.data[["High", "Low", "Close" , "Volume" , "returns"]].copy()

        if not hasattr(self, 'data') or self.data is None:
            warnings.warn("Data is not initialized. Define 'data' before calling prepare_data_vwap", UserWarning)
            return  # Exit the method to avoid further errors

        # Calculate the VWAP using the correct keyword arguments
        data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'],data['Close'], data['Volume'],int(vwap_period))  
 
        cond1 = (data['Close'] > data['VWAP'] + vwap_threshold)
        cond2 = (data['Close'] < data['VWAP'] - vwap_threshold)
   
        data['position'] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1

        self.results = data

    def optimize_strategy(self, vwap_period_range, vwap_threshold_range, metric="Multiple"):
        print("\nOptimize Strategy is running. \n")
    
        self.metric = metric
    
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
    
        vwap_period_range = range(*vwap_period_range)
        vwap_threshold_range = np.arange(vwap_threshold_range[0], vwap_threshold_range[1], vwap_threshold_range[2])  # For float ranges
    
        combinations = list(product(vwap_period_range, vwap_threshold_range))
    
        performance = []
        for comb in combinations:
            self.prepare_data(vwap_period=comb[0], vwap_threshold=comb[1])
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))
    
        self.results_overview = pd.DataFrame(data=np.array(combinations), columns=["vwap_period", "vwap_threshold"])
        self.results_overview["performance"] = performance

        # Check performance values
        print(f"Performance values:\n{self.results_overview}")
    
        vwap_period, vwap_threshold = self.find_best_strategy()
        return vwap_period, vwap_threshold

    def find_best_strategy(self):
        best = self.results_overview.nlargest(1, "performance")

        vwap_period = best.vwap_period.iloc[0]
        vwap_threshold = best.vwap_threshold.iloc[0]
        perf = best.performance.iloc[0]

        print("VWAP_Period: {} | VWAP_Threshold: {} | {}: {}\n".format(vwap_period, vwap_threshold, self.metric, round(perf, 5)))
        self.test_strategy(vwap_period=vwap_period, vwap_threshold=vwap_threshold)
        return vwap_period, vwap_threshold