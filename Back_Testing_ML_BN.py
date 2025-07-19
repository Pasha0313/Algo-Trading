import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Loading_Data_BN import fetch_historical_data
import logging
import warnings
import os

plots_folder = "Plots"
os.makedirs(plots_folder, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class ML_Strategies:
    def __init__(self, client, symbol, bar_length, start, end=None, tc=0.0):
        self.client = client
        self.symbol = str(symbol)
        self.start = str(start)
        self.end = str(end) if end else None
        self.tc = tc
        self.data = None
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h","6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.MLName = None
        self.ML = None
        self.already_bought = False
        try:
            self.data = self.get_data()
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise

    def get_data(self):
        try:
            data = fetch_historical_data(client=self.client, symbol=self.symbol,
                bar_length=self.bar_length, start=self.start, end=self.end)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise

    def ML_Strategy(self, CFModel, parameters, Perform_Tuner=False):
        data = self.data.copy()
        #print('\n',data.columns)
        #print(data.head())
        data['change_tomorrow'] = data.Close.pct_change(-1) * 100 * -1
        data = data.dropna().copy()
        data['change_tomorrow_direction'] = np.where(data.change_tomorrow > 0, 1, 0) # ('UP', 'DOWN')
        data.change_tomorrow_direction.value_counts()
        self.create_advanced_features()
        data.Close.plot()
        self.data = data

    def create_advanced_features(self):
        from Loading_Strategy import StrategyLoader
        import Strategy as STRATEGY  

        Path_Configs = "Configs"
        strategy_loader = StrategyLoader(os.path.join(Path_Configs, "strategies_config.json"))

        data = self.data.copy()

        strategy = "Stochastic_RSI"
        description, parameters, _ = strategy_loader.process_strategy(strategy,Print_Data=False)
        #print(f"\nStrategy: {strategy}, Description: {description}")
        data = STRATEGY.define_strategy_Stochastic_RSI(data, parameters)

        if 'position' in data.columns:
            data = data.drop(columns='position')
        
        #print("\n📋 Feature columns:", data.columns.tolist())
        
        strategy = "Bollinger_EMA"
        description, parameters, _ = strategy_loader.process_strategy(strategy,Print_Data=False)
        #print(f"\nStrategy: {strategy}, Description: {description}")

        data = STRATEGY.define_strategy_Bollinger_EMA(data, parameters)

        if 'position' in data.columns:
            data = data.drop(columns='position')            

        print("\n📋 Final feature columns:", data.columns.tolist())

        self.data = data  

    def run_model(self, model_type='rf'):
        y = self.data["change_tomorrow_direction"].copy()
        X = self.data.drop(columns=['change_tomorrow_direction'])
        self.feature_names = X.columns.tolist()

        # 🧠 Time series split (no shuffling)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        if model_type == 'rf':
            self.train_random_forest(X, X_train, X_val, y_train, y_val, Tuning=True)
        elif model_type == 'xgb':
            self.train_xgboost(X, X_train, X_val, y_train, y_val, Tuning=True)
        else:
            raise ValueError("Invalid model_type. Choose 'rf', 'xgb', or 'both'.")

    def train_random_forest(self,X ,X_train, X_val, y_train, y_val, Tuning = False):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report

        if Tuning :
            rf = self.fine_tune_random_forest(X_train, y_train)
        else :
            rf = RandomForestClassifier(
                bootstrap=True, criterion='entropy', max_depth=10,
                max_features='sqrt', min_samples_leaf=5, min_samples_split=10,
                n_estimators=100, random_state=2024, n_jobs=-1)
            rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)
        print(classification_report(y_val, y_pred, digits=4))        
        self.data["prediction"] = rf.predict(X)  # Use full data for later backtesting
        self.ML = rf

    def fine_tune_random_forest(self, X_train, y_train, cv=3, n_iter=10):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV

        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [10, 20, 30, None],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 5, 10],
            'bootstrap': [True, False]
        }

        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        search.fit(X_train, y_train)
        print("🔧 Best RF Params:", search.best_params_)
        return search.best_estimator_

    def train_xgboost(self,X, X_train, X_val, y_train, y_val, Tuning = False):
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        
        if Tuning :
            xgb_model = self.fine_tune_xgboost(X_train, y_train)
        else :
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=2024
            )
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = xgb_model.predict(X_val)
        print(classification_report(y_val, y_pred, digits=4))        
        self.data["prediction"] = xgb_model.predict(X)
        self.ML = xgb_model

    def fine_tune_xgboost(self, X_train, y_train, cv=3, n_iter=20):
        import xgboost as xgb
        from sklearn.model_selection import RandomizedSearchCV

        param_distributions = {
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5, 10],
            'gamma': [0, 0.2, 0.5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [1, 3, 5, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200]
        }

        search = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        search.fit(X_train, y_train)
        print("🔧 Best XGB Params:", search.best_params_)
        return search.best_estimator_

    def run_backtesting_strategy(self):
        from backtesting import Backtest
        from Back_Testing_MLClass_BN import SimpleClassificationUD
        if self.data is None or self.ML is None or not hasattr(self, "feature_names"):
            raise ValueError("Ensure data, model, and feature list are initialized.")

        X = self.data.copy()
        model = self.ML
        features = self.feature_names

        # ✅ Dynamically inject model and features into a subclass
        class StrategyWrapper(SimpleClassificationUD):
            def init(inner_self):
                super().init()
                inner_self.model = model
                inner_self.features = features

        bt = Backtest(
            X,
            StrategyWrapper,
            cash=1000,
            commission=self.tc,
            exclusive_orders=True
        )

        results = bt.run()
        print("\n=== Backtest Summary ===")
        print(results)
        return results

    def plot_performance(self, leverage=1.0, save=False, filename=None):
        if "prediction" not in self.data.columns:
            raise ValueError("Prediction column not found. Run DecisionTreeML() first.")
        
        if "change_tomorrow" not in self.data.columns:
            raise ValueError("change_tomorrow column missing. Run ML_Strategy() first.")

        df = self.data.copy()

        # Shift prediction to avoid look-ahead bias
        df["position"] = df["prediction"].shift()

        # Calculate daily returns
        df["strategy_return"] = df["position"] * df["change_tomorrow"] / 100
        df["buy_and_hold"] = df["change_tomorrow"] / 100

        # Apply leverage
        df["strategy_return_leveraged"] = df["strategy_return"] * leverage

        # Cumulative performance
        df["cumulative_strategy"] = (1 + df["strategy_return"]).cumprod()
        df["cumulative_leverage"] = (1 + df["strategy_return_leveraged"]).cumprod()
        df["cumulative_bh"] = (1 + df["buy_and_hold"]).cumprod()

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["cumulative_bh"], label="Buy & Hold", linestyle="--")
        plt.plot(df.index, df["cumulative_strategy"], label="ML Strategy", linewidth=2)
        if leverage != 1.0:
            plt.plot(df.index, df["cumulative_leverage"], label=f"ML Strategy ×{leverage:.1f}", linewidth=2, alpha=0.8)

        plt.title(f"📈 Strategy Performance | {self.symbol} | Leverage = {leverage:.1f}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save:
            fname = filename or f"ML_Strategy_Performance_{self.symbol}.png"
            plt.savefig(os.path.join("Plots", fname))
        plt.show()
