import json

class StrategyLoader:
    def __init__(self, config_file):
        # Load strategy configuration from a JSON file
        with open(config_file, "r") as file:
            self.strategies = json.load(file)

    def get_strategy_config(self, strategy_name):
        """Fetch the strategy config by its name"""
        return self.strategies.get(strategy_name, None)

    def process_strategy(self, strategy_name):
        """Process and return the strategy config in a suitable format"""
        strategy_config = self.get_strategy_config(strategy_name)
        
        if strategy_config is None:
            raise ValueError(f"Strategy '{strategy_name}' not found.")

        # Get the description, parameters, and ranges
        description = strategy_config["description"]
        parameters = tuple(strategy_config["parameters"])
        print('CheckUP : ',parameters)
        param_ranges = {}

        # Convert parameter ranges to appropriate range/list objects dynamically
        for key, value in strategy_config["param_ranges"].items():
            start, stop, step = value  
            if all(isinstance(v, int) for v in value):  
                param_ranges[key] = range(start, stop, step)
            elif all(isinstance(v, (int, float)) for v in value):  
                param_ranges[key] = [start + i * step for i in range(int((stop - start) / step))]
            else:
                raise TypeError(f"Invalid range values for {key}: {value}")

        return description, parameters, param_ranges
    
    def print_strategy_details(self, strategy_name):
        """Print the strategy details, including parameters"""
        strategy_config = self.get_strategy_config(strategy_name)
       
        # Get the description, parameters, and ranges
        parameters = tuple(strategy_config["parameters"])

        # Print parameters in the desired format: Key1: Value1, Key2: Value2, ...
        param_names = list(self.strategies[strategy_name]["param_ranges"].keys())
        parameter_string = ", ".join([f"{param_names[i]}: {parameters[i]}" for i in range(len(parameters))])
        
        print(f"Parameters: {parameter_string}")
