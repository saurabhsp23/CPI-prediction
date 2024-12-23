import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from os.path import join
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error,mean_absolute_percentage_error
# from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from scipy.stats import f
import statsmodels.api as sm

# Constants
DATA_BASE_PATH = '../Data'
LOOKBACK_PERIOD = 12

def read_and_format_data(base_path=DATA_BASE_PATH):
    '''
    Read in time series of question responses from the daily_aggregate.parquet and
    initial_indicator_dataset.csv files.

    Args:
        base_path (str): The base path for data files.

    Returns:
        pd.DataFrame, pd.DataFrame: A tuple of DataFrames containing daily aggregate data and indicator data.
    '''
    # Read daily aggregate file (predictor library)
    df_agg = pd.read_parquet(join(base_path,'daily_aggregate.pq'))

    # Read indicator data file (target library)
    df_ind = pd.read_csv(join(base_path,'initial_indicator_dataset.csv'))

    # Process indicator data
    df_ind['Date'] = pd.to_datetime(df_ind['Date']) # convert dates to datetimes
    df_ind = df_ind.set_index('Date') # set the date column as the index

    return df_agg, df_ind

def data_filter(df_agg,**kwargs):
    '''
        Filter data using a rolling window.

        Args:
            df (pd.DataFrame): The input DataFrame to be filtered.
            kwargs: will be filter specific arguments; we can expand the filter
            list/method later

        Returns:
            pd.DataFrame: The filtered DataFrame.
    '''
    window_size = kwargs['window_size']     # set window size for moving average filter
    df_smooth = df_agg.rolling(window_size).mean()   # apply moving average filter
    df_smooth = df_smooth.tail(-window_size+1)

    return df_smooth

# TODO Consider other aggregators
def myagg(x):
    '''
    Example aggregator function that just takes the mean.

    Args:
        x (pd.DataFrame): Data to be aggregated.

    Returns:
        np.ndarray: Aggregated data.
    '''
    return np.mean(x, axis=0)

def get_combined_data_with_given_indicator(df_indicator_all, df_daily_agg, indicator_col ='CONSSENT Index'):
    '''
        Combine indicator data with daily aggregate data. Also as indicator data is monthly change the
        daily aggregate data to monthly.

        Args:
            df_indicator_all (pd.DataFrame): The indicator data.
            df_daily_agg (pd.DataFrame): The daily aggregate data.
            indicator_col (str): The name of the indicator column to consider.

        Returns:
            pd.DataFrame: Combined DataFrame with monthly aggregated data.
    '''
    df_indicator = df_indicator_all[indicator_col]  # restrict the indicators df on a single indicator
    df_indicator = df_indicator[df_indicator.diff() != 0].tail(-1)  # get the dates when new values of the
    # indicator arrived (e.g. release dates)
    # and drop the initial date since we do
    # not know precisely when it was released

    # Define the start and end_dates of the aggregation periods
    start_dates = df_indicator.index[:-1]  # list of start dates
    end_dates = (df_indicator.index - datetime.timedelta(days=1))[1:]  # note that we do not
    # include release date here
    range_pairs = zip(start_dates, end_dates)

    ## Slow Loop Aggregator (see if we can find a way to update with pd.groupby()
    # explore ideas like: df_smooth.groupby(pd.cut(start_dates,end_dates)).agg(myagg)

    monthly_agg_values = []
    for stday, endday in range_pairs:
        df_month = df_daily_agg[stday:endday]
        # TODO explore variations here, e.g. last value, exponential moving average, etc.
        agg_month_val = myagg(df_month)
        monthly_agg_values.append(agg_month_val)
    df_monthly_agg = pd.concat(monthly_agg_values, axis=1)
    df_monthly_agg.columns = end_dates
    df_monthly_agg = df_monthly_agg.T
    df_monthly_agg = df_monthly_agg.dropna(axis=0, how='all')  # drop rows with all nans
    df_monthly_agg.index = df_monthly_agg.index + datetime.timedelta(days=1)

    # combined aligned df
    df_monthly_combined = pd.concat([df_monthly_agg, df_indicator], axis=1)
    df_monthly_combined = df_monthly_combined[df_monthly_agg.index.min():]  # filter prior to the first question date
    return df_monthly_combined

# TODO Can you expand this to other performance metric ideas?
def performance_metrics(y_pred, y_act):
    '''
    Print performance metrics such as R-squared, MAE, and MAPE.

    Args:
        y_pred (list): Predicted values.
        y_act (list): Actual values.
    '''
    rsquared = r2_score(y_pred,y_act)
    mae = mean_absolute_error(y_pred,y_act)
    mape = mean_absolute_percentage_error(y_pred,y_act)

    # calculating f stat
    var_oos = np.var(y_act)
    var_pred = np.var(y_pred)

    # Perform F-test
    F_statistic = var_pred / var_oos

    # Calculate degrees of freedom
    dof_pred = len(y_pred) - 1
    dof_oos = len(y_act) - 1

    # Calculate p-value
    p_value = f.sf(F_statistic, dof_pred, dof_oos)

    print(f'R-squared: {rsquared}, MAE: {mae}, MAPE: {mape}, p_value: {p_value}')
    return (rsquared, mae, mape, p_value)

def run_model(df, lookback_period = LOOKBACK_PERIOD, model = LinearRegression(), label = 'LinearRegression'):
    '''
    Walk forward train/validate split using a machine learning model.

    Args:
        df (pd.DataFrame): Combined DataFrame with monthly aggregated data and indicator data.
        lookback_period (int): The number of periods to look back for training.
        model: The machine learning model to use for predictions.
    '''
    df = df.ffill()#.dropna()
    df = df.dropna(axis=1, how='all')  # drop cols with all Nans
    df=df.replace(np.nan, 0) # now dropping any intial rows with nans
    total_periods = len(df)
    y_pred = []  # predicted values
    y_act = []  # actual values
    indicator = df.columns[-1]
    reg_betas = pd.DataFrame(index=df.columns[:-1])

    for i in range(total_periods - lookback_period + 1):
        df_lookback = df.iloc[i:i + lookback_period + 1]

        # TODO IMPORTANT NOTE: think more carefully how we fill nans in the below instead of a
        # back-fill, forward-fill
        # df_lookback = df_lookback.ffill() # backfill nan's currently ... how to make better?
        # and forward fill the final row
        if len(df_lookback.columns) <= 1:
            continue

        df_train = df_lookback.head(-1)
        df_test = df_lookback.tail(1)
        y_act.append(float(df_test.iloc[:, -1].values))
        # train linear regression model and get predicted value
        xtrain = df_train.iloc[:, 0:-1]
        ytrain = df_train.iloc[:, -1]
        xtest = df_test.iloc[:, :-1]  # predictors

        if label == 'XGBoosting':
            # Use XGBoost specific code
            xg_train = xgb.DMatrix(xtrain.values, label=ytrain.values)
            xg_test = xgb.DMatrix(xtest.values)
            params = {
                'tree_method': 'auto',  # Ensure tree_method is set to "auto"
                'objective': 'reg:squarederror',  # Use regression objective
            }
            model = xgb.train(params, xg_train)
            # Update y_pred using XGBoost predictions
            y_pred.append(float(model.predict(xg_test)[0]))
        else:
            model.fit(xtrain.values, ytrain.values)
            # sm_lasso = sm.OLS(ytrain.values, xtrain.values)
            ypred = model.predict(xtest.values)
            y_pred.append(float(ypred))

            model_params = pd.DataFrame(model.coef_, index=list(xtrain.columns))
            reg_betas.loc[model_params.index, xtest.index] = model_params.values
    if label == 'XGBoosting':
            xgb.plot_importance(model)
            plt.show()
            importance = model.get_score(importance_type='weight')
            print(importance)
    return performance_metrics(y_pred, y_act), reg_betas


def main():
    # Survey Aggregate data and indicator data
    df_agg, df_ind = read_and_format_data()
    # Smoothing the aggregate survey data
    rolling_window_size = 20
    df_smooth = data_filter(df_agg, window_size=rolling_window_size)
    # Get the combined monthly survey data with indicator data
    df_comb = get_combined_data_with_given_indicator(df_ind, df_smooth, indicator_col='AUTMUSAG Index')
    print("Linear Regression Model:")
    run_model(df_comb, lookback_period=12, model=LinearRegression())

    print("\nLasso Model (alpha=0.01):")
    run_model(df_comb, lookback_period=12, model=Lasso(alpha=0.01))

if __name__ == "__main__":
    main()
