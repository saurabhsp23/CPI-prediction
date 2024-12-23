
from constants import TRAINING_DEFAULT_LOOKBACK_WINDOW, LOWESS_FRAC_CONST 

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.nonparametric.smoothers_lowess import lowess

class Smoother: 
    ''' 
    Smoother class to work on pre-processing data 
    ''' 
    
    def __init__(self,data,smoother_type = 'moving_average',lookback_window=21,**kwargs): 
        ''' 
        Initial smoothing methods that we 

        Here kwargs are other specific smoother techniques that 
        are required 
        '''
        
        self.data = data 
        self.smoother_type = smoother_type 
        self.lookback_window = lookback_window
        self.params_dict = kwargs 

    def exponential_smoothing(self, col, seasonal_periods=TRAINING_DEFAULT_LOOKBACK_WINDOW, is_seasonal=None, is_trend=None):
        '''
        Takes in a dataframe column and returns an exponentially smoothed column. Can choose to add trend
        or seasonality fit based on parameters "is_seasonal" and "is_trend". For seasonality, can input a 
        seasonality period (with default being 12). Can choose the start and end point for the prediction values.
        '''
        
        if is_seasonal:
            is_seasonal = 'add'
        if is_trend:
            is_trend = 'add'

        model = ExponentialSmoothing(col, seasonal_periods=seasonal_periods, trend=is_trend, seasonal=is_seasonal).fit()

        return model.predict(start=0, end= len(col)-1)

    def lowess_smoothing(self, col, frac = LOWESS_FRAC_CONST):
        '''
        Takes in a dataframe column and returns an lowess smoothed column. Fraction parameter set to a
        default value of 0.1.
        '''
        
        smoothed = lowess(col, range(len(col)), frac)

        return smoothed[:, 1]
   
    def moving_average(self, lookback_window = TRAINING_DEFAULT_LOOKBACK_WINDOW):
        '''
        Takes in a dataframe and returns the moving average based on the window size.
        '''
        
        moving_avg = data.rolling(window=window).mean().dropna()

        return moving_avg
