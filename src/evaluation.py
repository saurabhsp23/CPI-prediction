# Module: evaluation.py


class Filter():
    def __init__(self, combinations):
        self.combinations = combinations
        
    '''
    Filter1: Fill the NaNs with the Prophet model, which is a procedure for forecasting time series based on 
             an additive model where non-linear trends are fit with yearly, weekly seasonality.
             Here we train the model with the data up to the NaN date, and fill the NaN interval with forecasting data.
    '''
    @staticmethod
    def filter1(df):
        for id in df.columns:
            available_date = df.index[0]

            # Get the NaNs interval
            nan_mask = df[id].isna()
            nan_index = [[*g.index] for _, g in df[nan_mask].groupby((~nan_mask).cumsum())]
            
            # Filling the NaNs interval
            for idx_lst in nan_index:
                start_date = idx_lst[0]
                end_date = idx_lst[-1]

                # If the NaN interval is from the beginning, skip it
                if start_date == available_date:
                    available_date = end_date

                # If the NaN interval is after the first data available date, try to fill it
                else:
                    # Select the data up to the date of the first NaN as traing set
                    df_train = df[[id]].loc[available_date: start_date].reset_index().rename(columns={'index': 'ds', id: 'y'})
                    
                    # If the traiding set has less than two Non-NaN rows, just forward fill it
                    if df_train.dropna().shape[0] < 2:
                        df.loc[idx_lst, id] = df.loc[idx_lst, id].ffill()
                        continue
                    
                    # Otherwise, train a prophet model to forecast the NaN-period
                    model = Prophet()
                    model.fit(df_train)
                    future = pd.DataFrame(data=idx_lst, columns=['ds'])
                    forecast = model.predict(future)

                    # Fill the NaNs with forecasting data
                    df.loc[idx_lst, id] = forecast['yhat'].values
        return df
    
    @staticmethod
    def filter2(df):
        return df
    
    @staticmethod
    def filter3(df):
        return df

# Content from cell_08.py
# Content from cell 8

class Smooth():
    def __init__(self, combinations):
        self.combinations = combinations
        
    '''
    Smooth1: Moving Average
    Takes in a dataframe and returns the moving average based on the window size.

    Input: 

    Output:
    '''
    @staticmethod
    def smooth1(df, window=12):
        moving_avg = df.rolling(window=window).mean()
        return moving_avg
    

    '''
    Smooth2: Exponential Smoothing
    Takes in a dataframe column and returns an exponentially smoothed column. Can choose to add trend
    or seasonality fit based on parameters "is_seasonal" and "is_trend". For seasonality, can input a 
    seasonality period (with default being 12). Can choose the start and end point for the prediction values.
    
    Input: 

    Output:
    '''    
    ### RUNS INTO ISSUES WITH NAN VALUES -> DOES NOT RETURN FULL DF UNLESS NO NANs
    ### ALSO RUNS INTO ISSUE WHEN NO DATAFRAME FREQUENCY
    @staticmethod
    def smooth2(df, seasonal_periods=12, is_seasonal=None, is_trend=None):
        def exponential_smooth(col, seasonal_periods=12, is_seasonal=None, is_trend=None):
            if is_seasonal:
                is_seasonal = 'add'
            if is_trend:
                is_trend = 'add'

            model = ExponentialSmoothing(col, seasonal_periods=seasonal_periods, trend=is_trend, seasonal=is_seasonal).fit()

            return model.predict(start=0, end=len(col)-1)
        return df.apply(exponential_smooth)    

    '''
    Smooth3: Lowess Smoothing
    Takes in a dataframe column and returns an lowess smoothed column. Fraction parameter set to a
    default value of 0.1.

    Input: 

    Output:
    '''
    ### RUNS INTO ISSUES WITH NAN VALUES -> WILL PRODUCE ERROR IN RUN_MODEL METHOD
    @staticmethod
    def smooth3(df):
        def sm_lowess(col, frac=0.1):
            # smoothed = lowess(col, range(len(col)), frac)
            smoothed = pd.Series(lowess(col, range(len(col)), 0.01)[:, 1], index=col.dropna().index).reindex(col.index)
            # print(smoothed)
            return smoothed

        return df.apply(sm_lowess)
    
    '''
    Smooth4: Fourier Smoothing
    Takes in a dataframe column and returns an fourier smoothed column. Fraction parameter set to a
    default value of 0.02.

    Input: 

    Output:
    '''
    def fourier_smooth(data: pd.DataFrame, cutoff_freq=0.02):

        new_data = pd.DataFrame()
        for col in data.columns:
            fourier_transform = fft(data[col].dropna().values)
            
            n = len(data[col])
            freqs = np.fft.fftfreq(n)
            cutoff_index = int(cutoff_freq * n)
            
            fourier_transform[cutoff_index:-cutoff_index] = 0
            
            smoothed_data = np.real(ifft(fourier_transform))
            smoothed_data = pd.Series(smoothed_data, index=data[col].dropna().index)
            new_data[col] = smoothed_data
        new_data.index = data.index
        return new_data

# Performs model evaluation and generates metrics reports.

def evaluate_model(*args, **kwargs):
    """Placeholder for evaluate_model"""
    pass

def generate_metrics_report(*args, **kwargs):
    """Placeholder for generate_metrics_report"""
    pass

