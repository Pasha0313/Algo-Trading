from Futures_Backtester_Base import FuturesBacktesterBase
from itertools import product
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class Futures_Backtester_PV(FuturesBacktesterBase):
    def __init__(self, client, symbol, bar_length, start, end=None, tc=0.0, leverage=5, strategy="PV"):
        # Call the parent class constructor to initialize the base class
        super().__init__(client, symbol, bar_length, start, end, tc, leverage, strategy)

    def __repr__(self):
        return "\nFutures Backtester PV(symbol = {}, start = {}, end = {})\n".format(self.symbol, self.start, self.end)

    def test_strategy(self, percentiles = None, thresh = None):
        self.prepare_data(percentiles = percentiles, thresh = thresh)
        self.run_backtest()
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        self.print_performance()
      
    def prepare_data(self, percentiles, thresh):
        if not hasattr(self, 'data') or self.data is None:
            warnings.warn("Data is not initialized. Define 'data' before calling define_strategy_PV", UserWarning)
            return  # Exit the method to avoid further errors

        data = self.data[[ "Close" , "Volume" , "returns"]].copy()
        data["vol_ch"] = np.log(data.Volume.div(data.Volume.shift(1)))
        data.loc[data.vol_ch > 3, "vol_ch"] = np.nan
        data.loc[data.vol_ch < -3, "vol_ch"] = np.nan        
        
        #print(percentiles)

        if percentiles:
            self.return_thresh = np.percentile(data.returns.dropna(), [percentiles[0], percentiles[1]])
            self.volume_thresh = np.percentile(data.vol_ch.dropna(), [percentiles[2], percentiles[3]])
        elif thresh:
            self.return_thresh = [thresh[0], thresh[1]]
            self.volume_thresh = [thresh[2], thresh[3]]

        #print (self.return_thresh,self.volume_thresh,"\n")  

        cond1 = data.returns <= self.return_thresh[0]
        cond2 = data.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])
        cond3 = data.returns >= self.return_thresh[1]
        
        data["position"] = 0
        data.loc[cond1 & cond2, "position"] = 1
        data.loc[cond3 & cond2, "position"] = -1
     
        self.results = data

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
        print(f"Performance values:{self.results_overview} \n")
        
        return_thresh,volume_thresh = self.find_best_strategy()
        return  return_thresh,volume_thresh

    def find_best_strategy(self):
        best = self.results_overview.nlargest(1, "performance")

        return_perc = [int(best.return_low.iloc[0]), int(best.return_high.iloc[0])]
        vol_perc = [int(best.vol_low.iloc[0]), int(best.vol_high.iloc[0])]
        perf = best.performance.iloc[0]
        data = self.results.copy()
        return_thresh = np.percentile(data.returns.dropna(), [return_perc[0], return_perc[1]])
        volume_thresh = np.percentile(data.vol_ch.dropna(), [vol_perc[0], vol_perc[1]])

        print("\n Return_Perc: {} | Volume_Perc: {} | {}: {}\n".format(return_thresh,volume_thresh, self.metric, round(perf, 5)))
        self.test_strategy(percentiles=(return_perc[0], return_perc[1], vol_perc[0], vol_perc[1]))
        return return_thresh,volume_thresh
 
 