import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import logging
from Loading_Data_BN import fetch_historical_data
import os
Plot_folder = "Plots"
os.makedirs(Plot_folder, exist_ok=True)
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

class BackTestingBase_BN:
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
            self.data = self.get_data()
            self.tp_year = self.calculate_trading_periodicity()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    def get_data(self):
        try:
            data = fetch_historical_data(client=self.client,symbol=self.symbol,bar_length=self.bar_length,
                                        start=self.start,end=self.end)
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
        #self.print_performance()

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
        leverage_returns = data.eff_lev.shift() * data.simple_ret
        leverage_returns = np.where(leverage_returns < -1, -1, leverage_returns)
        data["strategy_leverage"] = leverage_returns
        data["cstrategy_leverage"] = data.strategy_leverage.add(1).cumprod()
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
        to_analyze = np.log(data["strategy_leverage"].add(1)) if leverage else data.strategy

        strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple = round(self.calculate_multiple(data["returns"]), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(self.calculate_cagr(to_analyze), 6)
        ann_mean = round(self.calculate_annualized_mean(to_analyze), 6)
        ann_std = round(self.calculate_annualized_std(to_analyze), 6)
        sharpe = round(self.calculate_sharpe(to_analyze), 6)

        #print("=" * 50)
        #print(f"STRATEGY PERFORMANCE | INSTRUMENT = {self.symbol}")
        #print("-" * 50)
        #print(f"Strategy Multiple:           {round(strategy_multiple,2)}")
        #print(f"Buy-and-Hold Multiple:       {round(bh_multiple,2)}")
        #print(f"Out-/Underperformance:       {round(outperf,2)}")
        #print(f"CAGR:                        {round(cagr,2)}")
        #print(f"Annualized Mean Return:      {round(ann_mean,2)}")
        #print(f"Annualized Standard Deviation: {round(ann_std,2)}")
        #print(f"Sharpe Ratio:                {round(sharpe,2)}")
        #print("=" * 50)
        
        print("=" * 50)
        print(f"📊 STRATEGY PERFORMANCE | Symbol = {self.symbol}")
        print("-" * 50)

        metrics = {
            "Strategy Multiple": strategy_multiple,
            "Buy-and-Hold Multiple": bh_multiple,
            "Out-/Underperformance": outperf,
            "CAGR": cagr,
            "Annualized Mean Return": ann_mean,
            "Annualized Std. Dev.": ann_std,
            "Sharpe Ratio": sharpe
        }

        for name, value in metrics.items():
            print(f"{name:<35}: {value:>10.2f}")

        print("=" * 50)


        return strategy_multiple,bh_multiple,outperf,cagr,ann_mean,ann_std,sharpe
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

    def plot_strategy_comparison(self, leverage=False,plot_name=None,plot_show=True):
        if self.results is None:
            logger.warning("Run test_strategy() first.")
        else:
            title = f"{self.strategy} | {self.symbol} | TC = {self.tc}"
            if leverage:
                title += f" | Leverage = {self.leverage}"

            plt.figure(figsize=(10, 6))
            plt.plot(self.results.index, self.results["creturns"], label="Buy and Hold")
            plt.plot(self.results.index, self.results["cstrategy"], label="Strategy")
            if leverage and "cstrategy_leverage" in self.results.columns:
                plt.plot(self.results.index, self.results["cstrategy_leverage"], label="Strategy Leverage")
            plt.xticks(rotation=45)
            plt.title(title)
            plt.legend()
            plt.savefig(os.path.join(Plot_folder, f'Comparison_{plot_name}.png'))            
            if plot_show:
                plt.show()
            else:
                plt.close()

    def plot_results_II(self,plot_show=True):
        ''' Plots a scatter plot of volume change against returns.
        '''
        if self.results is None:
            print("No data to plot. Please provide data.")
        else:
            plt.scatter(x=self.results['vol_ch'], y=self.results['returns'])
            plt.xlabel("Volume Change")
            plt.ylabel("Returns")
            if plot_show:
                plt.show()
            else:
                plt.close()

    def plot_heatmap(self,plot_show=True):
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
            if plot_show:
                plt.show()
            else:
                plt.close()

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
            if plot_show:
                plt.show()
            else:
                plt.close()   

    def plot_all_indicators(self, plot_name=None,plot_show=True, Print_Data= False):
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
            available_columns = data.columns 

        if "std_dev" in available_columns:
            data["std_dev_n"] = data["std_dev"] / data["SMA"]
            data.drop(columns=["std_dev"], inplace=True) 
            available_columns = data.columns              

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
        
        # **Plot 1: Price-Related Indicators**
        if price_columns:
            for col in price_columns:
                axes[plot_index].plot(data.index, data[col], label=col)
            axes[plot_index].set_title("Price Indicators")
            axes[plot_index].legend()
            plot_index += 1

        # **Plot 2: Momentum Indicators (ADX, RSI, Stoch_RSI_n)**
        if momentum_columns:
            for col in momentum_columns:
                axes[plot_index].plot(data.index, data[col], label=col)
            axes[plot_index].set_title("Momentum Indicators")
            axes[plot_index].legend()
            plot_index += 1

        # **Plot 3: Volatility Indicators (ATR, BB Width, std_dev)**
        if volatility_columns:
            for col in volatility_columns:
                axes[plot_index].plot(data.index, data[col], label=col)
            axes[plot_index].set_title("Volatility Indicators")
            axes[plot_index].legend()
            plot_index += 1

        # **Plot 4: Trading Positions (+1 Buy, -1 Sell)**
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
        if plot_show:
            plt.show()
        else:
            plt.close()