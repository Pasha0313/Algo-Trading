from Futures_Backtester_Base import FuturesBacktesterBase
from itertools import product
import numpy as np
import pandas as pd
import warnings
import ta
warnings.filterwarnings("ignore")


class Futures_Backtester_MACD(FuturesBacktesterBase):

    def __init__(self, client, symbol, bar_length, start, end, tc, leverage=5, strategy="MACD"):
        super().__init__(client, symbol, bar_length, start, end, tc, leverage, strategy)

    def test_strategy(self, macd_slow, macd_fast, macd_signal):
        self.prepare_data(macd_slow=macd_slow, macd_fast=macd_fast, macd_signal=macd_signal)
        self.run_backtest()
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data        
        self.print_performance()

    def prepare_data(self, macd_slow, macd_fast, macd_signal):
        data = self.data[[ "Close" , "Volume" , "returns"]].copy()
        data['MACD'] = ta.trend.macd(data['Close'], window_slow=macd_slow, window_fast=macd_fast)
        data['MACD_Signal'] = ta.trend.macd_signal(data['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)

        cond1 = (data['MACD'] > data['MACD_Signal'])
        cond2 = (data['MACD'] < data['MACD_Signal'])

        data['position'] = 0  
        data.loc[cond1, 'position'] = 1  
        data.loc[cond2, 'position'] = -1  

        self.results = data

    def optimize_strategy(self, macd_slow_range, macd_fast_range, macd_signal_range, metric="Multiple"):
        print("\nOptimize Strategy is running. \n")
        
        self.metric = metric
        
        if metric == "Multiple":
            performance_function = self.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.calculate_sharpe

        macd_slow_range = range(*macd_slow_range)
        macd_fast_range = range(*macd_fast_range)
        macd_signal_range = range(*macd_signal_range)
        
        combinations = list(product(macd_slow_range, macd_fast_range, macd_signal_range))
        
        performance = []
        for comb in combinations:
            self.prepare_data(macd_slow=comb[0], macd_fast=comb[1], macd_signal=comb[2])
            self.run_backtest()
            performance.append(performance_function(self.results.strategy))
        
        self.results_overview = pd.DataFrame(data=np.array(combinations), columns=["macd_slow", "macd_fast", "macd_signal"])
        self.results_overview["performance"] = performance

        # Check performance values
        print(f"Performance values:\n{self.results_overview}")
                
        macd_slow,macd_fast,macd_signal = self.find_best_strategy()
        return macd_slow,macd_fast,macd_signal

    def find_best_strategy(self):
        best = self.results_overview.nlargest(1, "performance")

        macd_s = best.macd_slow.iloc[0]
        macd_l = best.macd_fast.iloc[0]
        macd_smooth = best.macd_signal.iloc[0]
        perf = best.performance.iloc[0]
        
        print("MACD_S: {} | MACD_L: {} | MACD_Smooth: {} | {}: {}\n".format(macd_s, macd_l, macd_smooth, self.metric, round(perf, 5)))
        self.test_strategy(macd_s=macd_s, macd_l=macd_l, macd_smooth=macd_smooth)
        return macd_s, macd_l, macd_smooth