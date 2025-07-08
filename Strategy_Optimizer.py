import os
import json
import pandas as pd
from Back_Testing_BN import BackTesting_BN
from Loading_Strategy import StrategyLoader

def run_optimizer(client, start_date, end_date, symbol, tc, leverage, metric):
    Path_Configs = "Configs"
    strategy_loader = StrategyLoader(os.path.join(Path_Configs, "strategies_config.json"))
    bar_lengths = ["5m", "15m", "30m", "1h"]

    # Format dates
    fmt_start = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M')
    fmt_end = pd.to_datetime(end_date).strftime('%Y-%m-%d %H:%M')
    print(f"\nSymbol = {symbol}, Start = {fmt_start}, End = {fmt_end}, Leverage = {leverage}\n")

    best_results = []

    for strategy_name in strategy_loader.strategies:
        for bar in bar_lengths:
            try:
                description, parameters_BT, param_ranges_BT = strategy_loader.process_strategy(strategy_name)

                print(f"\n>> {strategy_name} | {bar} | Init: {parameters_BT} | Ranges: {param_ranges_BT}")

                bt = BackTesting_BN(client=client, symbol=symbol, bar_length=bar,
                                    start=start_date, end=end_date, tc=tc, leverage=leverage, strategy=strategy_name)

                bt.test_strategy(parameters_BT)
                bt.add_leverage(leverage=leverage)
                #bt.plot_strategy_comparison(leverage=True, plot_name=f"{symbol}_{strategy_name}_{bar}")
                #bt.plot_all_indicators(plot_name=f"{symbol}_{strategy_name}_{bar}")

                print(bt.results.trades.value_counts())

                best_params = bt.optimize_strategy(param_ranges_BT, metric, output_file=f"{strategy_name}_{bar}_optimize_results.csv")

                if best_params is None:
                    print(f"No valid parameters found for {strategy_name} @ {bar}. Skipping.")
                    continue

                sharpe = bt.calculate_sharpe(bt.results["strategy"]) if "strategy" in bt.results else None
                if sharpe is None or pd.isna(sharpe):
                    continue

                best_results.append({
                    "strategy": strategy_name,
                    "bar_length": bar,
                    "params": dict(zip(param_ranges_BT.keys(), best_params)),
                    "sharpe": sharpe
                })

            except Exception as e:
                print(f"Error optimizing {strategy_name} @ {bar}: {e}")

    # Save top strategies
    best_results.sort(key=lambda x: x["sharpe"], reverse=True)
    with open("best_strategies.json", "w") as f:
        json.dump(best_results[:20], f, indent=2)

    print("\nTop 10 strategies saved to best_strategies.json")
