from Futures_Backtester_Base import FuturesBacktesterBase
from itertools import product
import numpy as np
import pandas as pd
import warnings
import ta
warnings.filterwarnings("ignore")

class Futures_Backtester_RSI(FuturesBacktesterBase):
 
    def __init__(self, client, symbol, bar_length, start, end, tc, leverage=5, strategy="RSI"):
        super().__init__(client, symbol, bar_length, start, end, tc, leverage, strategy)

    def __repr__(self):
        return "\nFutures Backtester RSI(symbol = {}, start = {}, end = {})\n".format(self.symbol, self.start, self.end)
        
    def test_strategy(self, rsi_window, rsi_lower, rsi_upper):
        self.prepare_data(rsi_window=rsi_window, rsi_lower=rsi_lower, rsi_upper=rsi_upper)
        self.run_backtest()
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        self.print_performance()

    def prepare_data(self, rsi_window, rsi_lower, rsi_upper):
        if not hasattr(self, 'data') or self.data is None:
            warnings.warn("Data is not initialized. Define 'data' before calling define_strategy_RSI", UserWarning)
            return  # Exit the method to avoid further errors
        
        # Calculate RSI using the RSIIndicator class
        data = self.data[["Close", "Volume", "returns"]].copy()
        data['RSI'] = ta.momentum.rsi(data['Close'], window=rsi_window)

        cond1 = (data['RSI'] < rsi_lower)
        cond2 = (data['RSI'] > rsi_upper)

        data['position'] = 0
        data.loc[cond1, "position"] = 1
        data.loc[cond2, "position"] = -1

        self.results = data
 
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
        best = self.results_overview.sort_values(by="performance", ascending=False).iloc[0]

        rsi_window = best.rsi_window
        rsi_lower = best.rsi_lower
        rsi_upper = best.rsi_upper
        perf = best.performance
        
        print(f"Best combination: RSI_Window: {rsi_window} | RSI_Lower: {rsi_lower} | RSI_Upper: {rsi_upper} | {self.metric}: {round(perf, 5)}\n")
        self.test_strategy(rsi_window=rsi_window, rsi_lower=rsi_lower, rsi_upper=rsi_upper)
        return rsi_window, rsi_lower, rsi_upper