from Futures_Backtester_Base import FuturesBacktesterBase
from itertools import product
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class Futures_Backtester_SMA(FuturesBacktesterBase):
    def __init__(self, client, symbol, bar_length, start, end, tc, leverage=5, strategy="SMA"):
        super().__init__(client, symbol, bar_length, start, end, tc, leverage, strategy)

    def __repr__(self):
        return "\nFutures Backtester SMA(symbol = {}, start = {}, end = {})\n".format(self.symbol, self.start, self.end)
    
    def test_strategy(self, smas):
        self.prepare_data(smas = smas)
        self.run_backtest()
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        self.print_performance()
    
    def prepare_data(self, smas):
        ########################## Strategy-Specific #############################
        if not hasattr(self, 'data') or self.data is None:
            warnings.warn("Data is not initialized. Define 'data' before calling define_strategy_SMA", UserWarning)
            return  # Exit the method to avoid further errors

        data = self.data[[ "Close" , "Volume" , "returns"]].copy()
        data["SMA_S"] = data.Close.rolling(window = smas[0]).mean()
        data["SMA_M"] = data.Close.rolling(window = smas[1]).mean()
        data["SMA_L"] = data.Close.rolling(window = smas[2]).mean()
        
        data.dropna(inplace = True)
                
        cond1 = (data.SMA_S > data.SMA_M) & (data.SMA_M > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_M) & (data.SMA_M < data.SMA_L)
        
        data["position"] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1

        ##########################################################################
        
        self.results = data
             
    def optimize_strategy(self, SMA_S_range, SMA_M_range, SMA_L_range, metric = "Multiple"):
        print("\n Optimize Strategy is running. \n")

        self.metric = metric
        
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe
        
        SMA_S_range = range(*SMA_S_range)
        SMA_M_range = range(*SMA_M_range)
        SMA_L_range = range(*SMA_L_range)
        
        combinations = list(product(SMA_S_range, SMA_M_range, SMA_L_range))
         
        performance = []
        for comb in combinations:
            self.prepare_data(smas = comb)
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))
    
        self.results_overview =  pd.DataFrame(data = np.array(combinations), columns = ["SMA_S", "SMA_M", "SMA_L"])
        self.results_overview["performance"] = performance

        # Check performance values
        print(f"Performance values:\n{self.results_overview}")

        SMA_S, SMA_M, SMA_L = self.find_best_strategy()
        return SMA_S, SMA_M, SMA_L
        
        
    def find_best_strategy(self):
        best = self.results_overview.nlargest(1, "performance")
        
        SMA_S = best.SMA_S.iloc[0]
        SMA_M = best.SMA_M.iloc[0]
        SMA_L = best.SMA_L.iloc[0]
        perf = best.performance.iloc[0]
        
        print("SMA_S: {} | SMA_M: {} | SMA_L : {} | {}: {}".format(SMA_S, SMA_M, SMA_L, self.metric, round(perf, 5)))  
        self.test_strategy(smas = (SMA_S, SMA_M, SMA_L))
        return SMA_S, SMA_M, SMA_L  
   