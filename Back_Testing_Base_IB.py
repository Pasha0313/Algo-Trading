import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import logging
from Loading_Data_IB import fetch_historical_data
import os

# Create plots folder
Plot_folder = "Plots"
os.makedirs(Plot_folder, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

class BackTestingBase_IB:
    def __init__(self, ib, contract,timezone , bar_length, start, end=None, tc=0.0, leverage=5, strategy="PV"):
        self.ib = ib
        self.contract = contract
        self.symbol = f"{contract.symbol}/{contract.currency}"
        self.bar_length = str(bar_length)
        self.start = str(start)
        self.end = str(end) if end else None
        self.tc = tc
        self.leverage = leverage
        self.strategy = strategy
        self.data = None
        self.results = None
        self.timezone = timezone

        # IBKR uses human-readable strings like '1 min', '5 mins', etc.
        self.available_intervals = [
            "1 min", "2 mins", "3 mins", "5 mins", "10 mins", "15 mins", "30 mins", 
            "1 hour", "2 hours", "3 hours", "4 hours", "8 hours", "1 day", "1 week", "1 month"
        ]
        if self.bar_length not in self.available_intervals:
            raise ValueError(f"Invalid bar length: {self.bar_length}. Choose from {self.available_intervals}.")

        # Fetch data
        try:
            self.data = self.get_data()
            self.tp_year = self.calculate_trading_periodicity()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    def get_data(self):
        try:
            data = fetch_historical_data(
                ib=self.ib,
                contract=self.contract,
                bar_length=self.bar_length,
                start=self.start,
                end=self.end, 
                timezone = self.timezone 
            )
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise

    def test_strategy(self, parameters):
        self.prepare_data(parameters)
        self.run_backtest()
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        self.print_performance()

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

    def plot_strategy_comparison(self, leverage=False, plot_name=None):
        if self.results is None:
            logger.warning("Run test_strategy() first.")
        else:
            title = f"{self.strategy} | {self.symbol} | TC = {self.tc}"
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
            plt.savefig(os.path.join(Plot_folder, f'Comparison_{plot_name}.png'))            
            plt.show()

    def plot_results_II(self):
        if self.results is None:
            print("No data to plot. Please provide data.")
        else:
            plt.scatter(x=self.results['vol_ch'], y=self.results['returns'])
            plt.xlabel("Volume Change")
            plt.ylabel("Returns")
            plt.title(f"{self.symbol} | TC = {self.tc}")
            plt.show()

    def plot_heatmap(self):
        if self.results is None:
            print("No data to process. Please provide data.")
        else:
            self.results["ret_cat"] = pd.qcut(self.results['returns'], q=10, labels=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            self.results["vol_cat"] = pd.qcut(self.results['vol_ch'], q=10, labels=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])

            matrix_I = pd.crosstab(self.results['vol_cat'], self.results['ret_cat'])

            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1)
            sns.heatmap(matrix_I, cmap="RdYlBu_r", annot=True, robust=True, fmt=".0f")
            plt.title(f"Heatmap of Volume Change vs Returns | {self.symbol} | TC = {self.tc}")
            plt.xlabel("Return cat")
            plt.ylabel("Volume cat")
            plt.show()

            shifted_results = self.results.shift()
            matrix_II = pd.crosstab(
                shifted_results['vol_cat'],
                shifted_results['ret_cat'],
                values=shifted_results['returns'],
                aggfunc=np.mean
            )

            plt.figure(figsize=(8, 6))
            sns.set(font_scale=0.75)
            sns.heatmap(matrix_II, cmap="RdYlBu", annot=True, robust=True, fmt=".3f")
            plt.title(f"Heatmap of Volume Change vs Returns | {self.symbol} | TC = {self.tc}")
            plt.xlabel("Return cat")
            plt.ylabel("Volume cat")
            plt.show()

    def plot_all_indicators(self, plot_name=None, Print_Data = False):
        if self.results is None:
            logger.warning("Run test_strategy() first.")
            return

        data = self.results.copy()
        title = f"{self.strategy} | {self.symbol} | TC = {self.tc}"
        available_columns = data.columns
        if Print_Data : print(f"Available columns in results: {available_columns}")

        if "Stoch_RSI" in available_columns:
            data["Stoch_RSI_n"] = data["Stoch_RSI"] * 100
            data.drop(columns=["Stoch_RSI"], inplace=True)

        if "std_dev" in available_columns:
            data["std_dev_n"] = data["std_dev"] / data["SMA"]
            data.drop(columns=["std_dev"], inplace=True)

        price_indicators = ["Close", "SMA", "Upper_Band", "Lower_Band"]
        momentum_indicators = ["ADX", "RSI", "Stoch_RSI_n"]
        volatility_indicators = ["ATR", "BB_Width", "std_dev_n"]

        price_columns = [col for col in price_indicators if col in available_columns]
        momentum_columns = [col for col in momentum_indicators if col in available_columns]
        volatility_columns = [col for col in volatility_indicators if col in available_columns]
        position_available = "position" in available_columns

        num_plots = 3 + int(position_available)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 8), sharex=True)

        plot_index = 0

        if price_columns:
            for col in price_columns:
                axes[plot_index].plot(data.index, data[col], label=col)
            axes[plot_index].set_title("Price Indicators")
            axes[plot_index].legend()
            plot_index += 1

        if momentum_columns:
            for col in momentum_columns:
                axes[plot_index].plot(data.index, data[col], label=col)
            axes[plot_index].set_title("Momentum Indicators")
            axes[plot_index].legend()
            plot_index += 1

        if volatility_columns:
            for col in volatility_columns:
                axes[plot_index].plot(data.index, data[col], label=col)
            axes[plot_index].set_title("Volatility Indicators")
            axes[plot_index].legend()
            plot_index += 1

        if position_available:
            axes[plot_index].plot(data.index, data["position"], label="Position", linestyle="dotted", color="black")
            axes[plot_index].set_title("Trading Positions")
            axes[plot_index].axhline(1, linestyle="--", color="green", alpha=0.5, label="Buy Signal (+1)")
            axes[plot_index].axhline(-1, linestyle="--", color="red", alpha=0.5, label="Sell Signal (-1)")
            axes[plot_index].legend()
            plot_index += 1

        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(Plot_folder, f'Indicators_{plot_name}.png') if plot_name else f'Indicators_plot.png'
        plt.savefig(plot_path)
        plt.show()
