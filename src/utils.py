# Module: utils.py

# Content from cell_11.py
# Content from cell 11

def find_best_metric(pipeline_dict, metric_name, combination_criteria=()):
    """
    Find the best metric (R^2, MAE, or MAPE) from the pipeline_dict based on a specific combination criteria.

    :param pipeline_dict: Dictionary of combinations and their metrics.
    :param metric_name: The name of the metric to optimize ('r2', 'mae', or 'mape').
    :param combination_criteria: A tuple of values to filter combinations (e.g., ('agg1', 'smooth1')).
    :return: A tuple of the best combination and its metrics.
    """
    # Mapping the metric name to its index in the metrics tuple.
    metric_index = {'r2': 0, 'mae': 1, 'mape': 2}

    if metric_name not in metric_index:
        raise ValueError("Invalid metric name. Choose from 'r2', 'mae', or 'mape'.")

    best_combination = None
    best_metric = float('-inf') if metric_name == 'r2' else float('inf')

    for combination, metrics in pipeline_dict.items():
        # Check if all elements in combination_criteria are in the current combination.
        if not all(elem in combination for elem in combination_criteria):
            continue

        metric_value = metrics[metric_index[metric_name]]
        # Update best metric and combination based on the metric type.
        if (metric_name == 'r2' and metric_value > best_metric) or \
           (metric_name in ['mae', 'mape'] and metric_value < best_metric):
            best_metric = metric_value
            best_combination = combination

    return print(f'The best combination with your criteria is {best_combination} with metric {best_metric}.')

# Content from cell_12.py
# Content from cell 12

def get_class_methods(cls):
    '''
    Function to get a list of class methods. 
    '''
    methods = [name for name, method in inspect.getmembers(cls, predicate=inspect.isfunction) if name != '__init__']
    return methods

# Content from cell_13.py
# Content from cell 13



ex_agg = Agg(['agg1'])
ex_filt = Filter(['filter1'])
ex_smooth = Smooth(['smooth1'])
ex_test = Test(['lasso']) # 'linear_reg','lasso', 'stepwise_reg', 'xgboosting'
ex_pipeline = Pipeline(ex_agg, ex_filt, ex_smooth, ex_test)

ex_pipeline.read_and_format_data()
# Smoothing the aggregate survey data
rolling_window_size = 20
df_agg = ex_pipeline.data_filter(window_size=rolling_window_size)


x=ex_pipeline.run()

# Utility functions for general-purpose tasks.

def helper_function_1(*args, **kwargs):
    """Placeholder for helper_function_1"""
    pass

def helper_function_2(*args, **kwargs):
    """Placeholder for helper_function_2"""
    pass

