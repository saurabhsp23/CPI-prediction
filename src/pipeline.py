from LR_implementation import *
from combined_aggregation import *
from filter import *
from smooth import *
from tests import *

class Pipeline():
    """
    Run through a bunch of different processes based on combinations of agg, filt, etc. agg, filt, and smooth are
    instances of the Agg, Filter, and Smooth classes. Each class is initialized with a set of combinations of which
    functions we want to test together. The combination attribute will be used to test different combinations of these
    operatios.
    """
    
    
    def __init__(self, agg, filt, smooth, test, lookback_period=12, base_path='../data'):
        
        self.agg = agg
        self.filt = filt
        self.smooth = smooth
        self.test = test
        self.lookback_period = lookback_period
        self.base_path = base_path
        

    def read_and_format_data(self):
        """
        Read in time series of question responses from the daily_aggregate.parquet and
        initial_indicator_dataset.csv files.

        Args:
            base_path (str): The base path for data files.

        """
        
        # Read daily aggregate file (predictor library)
        df_agg = pd.read_parquet(join(self.base_path,'daily_aggregate.pq'))

        # Read indicator data file (target library)
        df_ind = pd.read_csv(join(self.base_path,'initial_indicator_dataset.csv'))

        # Process indicator data
        df_ind['Date'] = pd.to_datetime(df_ind['Date']) # convert dates to datetimes
        df_ind = df_ind.set_index('Date') # set the date column as the index
        
        self.df_agg = df_agg
        self.df_ind = df_ind
        
        return
    

        
    def data_filter(self, **kwargs):
        """
            Filter data using a rolling window.

            Args:
                df (pd.DataFrame): The input DataFrame to be filtered.
                kwargs: will be filter specific arguments; we can expand the filter
                list/method later

            Returns:
                pd.DataFrame: The filtered DataFrame.
        """
        window_size = kwargs['window_size']     # set window size for moving average filter
        df_smooth = self.df_agg.rolling(window_size).mean()   # apply moving average filter
        df_smooth = df_smooth.tail(-window_size+1)
        
        return df_smooth

    
    def myagg(self, x):
        """
        Example aggregator function that just takes the mean.

        Args:
            x (pd.DataFrame): Data to be aggregated.

        Returns:
            np.ndarray: Aggregated data.
        """
        return np.mean(x, axis=0)

    
    def get_combined_data_with_given_indicator(self, df_indicator_all, df_daily_agg, indicator_col ='CONSSENT Index'):
        """
            Combine indicator data with daily aggregate data. Also as indicator data is monthly change the
            daily aggregate data to monthly.

            Args:
                df_indicator_all (pd.DataFrame): The indicator data.
                df_daily_agg (pd.DataFrame): The daily aggregate data.
                indicator_col (str): The name of the indicator column to consider.

            Returns:
                pd.DataFrame: Combined DataFrame with monthly aggregated data.
        """
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
            agg_month_val = self.myagg(df_month)
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

    
    
    def run(self):
        pipeline_dict = {}
        combo_list = [self.agg.combinations, self.filt.combinations, self.smooth.combinations, self.test.combinations]
        for combination in itertools.product(*combo_list):
            if hasattr(Agg, combination[0]):
                agg_func = getattr(Agg, combination[0])
            else:
                raise InvalidOperation
            
            if hasattr(Filter, combination[1]):
                filt_func = getattr(Filter, combination[1])
            else:
                raise InvalidOperation
                
            if hasattr(Smooth, combination[2]):
                smooth_func = getattr(Smooth, combination[2])
            else:
                raise InvalidOperation
                
            if hasattr(Test, combination[3]):
                tester = getattr(Test, combination[3])
            else:
                raise InvalidOperation
            
            df_ind = self.df_ind.copy()
            df_agg = self.df_agg.copy()

            reg_final_stats_df = pd.DataFrame(index=df_ind.columns, columns=['rsquared', 'mae', 'mape', 'p_value'])
            reg_final_stats_df_pct_change = pd.DataFrame(index=df_ind.columns, columns=['rsquared', 'mae', 'mape', 'p_value'])

            for indicator in df_ind.columns[:5]:
                # Get the combined monthly survey data with indicator data
                
                print(f"Indicator: {indicator}")
                if pd.api.types.is_numeric_dtype(df_ind[indicator]):

                    df_comb = self.get_combined_data_with_given_indicator(df_ind, df_agg, indicator_col=indicator)
                    # df_comb_processed = df_comb[df_comb.iloc[:,:-1] <= 1].replace(np.inf, 0)

                    df_comb_processed_change = df_comb.pct_change().replace(np.inf, 0)
                    
                    print(f'Running {tester} model')

                    combo = Combo(df = df_comb_processed_change, agg_func = agg_func, filt_func = filt_func,
                                  smooth_func = smooth_func, tester=tester, lookback_period = self.lookback_period)
                    
                    combo.process_df()

                    reg_final_stats_df_pct_change.loc[indicator], pct_change_reg_coefs_df = combo.run_model()
                    pipeline_dict[indicator] = reg_final_stats_df_pct_change.loc[indicator]
                    df_comb_processed_change.to_csv(f'../results/{indicator}_combined_monthly_processed_data.csv')
                    pct_change_reg_coefs_df.to_csv(f'../results/{indicator}_reg_pct_change_coefs.csv')
                    
            reg_final_stats_df_pct_change.to_csv('../results/reg_pct_change_final_stats.csv')


        return pipeline_dict

