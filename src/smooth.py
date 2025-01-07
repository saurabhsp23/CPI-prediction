from imports import *
class Smooth():
    def __init__(self, combinations):
        self.combinations = combinations
        
    """
    Smooth1: Moving Average
    Takes in a dataframe and returns the moving average based on the window size.

    Input: 

    Output:
    """
    @staticmethod
    def smooth1(df, window=12):
        moving_avg = df.rolling(window=window).mean()
        return moving_avg
    

    """
    Smooth2: Exponential Smoothing
    Takes in a dataframe column and returns an exponentially smoothed column. Can choose to add trend
    or seasonality fit based on parameters "is_seasonal" and "is_trend". For seasonality, can input a 
    seasonality period (with default being 12). Can choose the start and end point for the prediction values.
    
    """    

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

    """
    Smooth3: Lowess Smoothing
    Takes in a dataframe column and returns an lowess smoothed column. Fraction parameter set to a
    default value of 0.1.

    """

    @staticmethod
    def smooth3(df):
        def sm_lowess(col, frac=0.1):
            # smoothed = lowess(col, range(len(col)), frac)
            smoothed = pd.Series(lowess(col, range(len(col)), 0.01)[:, 1], index=col.dropna().index).reindex(col.index)
            # print(smoothed)
            return smoothed

        return df.apply(sm_lowess)
    
    """
    Smooth4: Fourier Smoothing
    Takes in a dataframe column and returns an fourier smoothed column. Fraction parameter set to a
    default value of 0.02.

    """
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



