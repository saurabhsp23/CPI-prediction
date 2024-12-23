## This file is meant to be a global constants file 
from os.path import join  

# ML/regression model imports 
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR 
import xgboost as xgb

WALK_FORWARD_TESTING_DEFAULT_LOOKBACK_WINDOW = 12
TRAINING_DEFAULT_LOOKBACK_WINDOW = 12 
LOWESS_FRAC_CONST = 0.33  # lowess regression constant 

ERROR_EPSILON = 1e-10 # epsilon to use in error metrics 

# Global base path to read data files: set this to the Data/
# folder in the local checkout/google drive instance of your code
DATA_BASE_PATH = join('..','..','Data') 

DAILY_AGG_PATH = join(DATA_BASE_PATH,'daily_aggregate.pq')
INDICATOR_DATA_PATH = join(DATA_BASE_PATH,'initial_indicator_dataset.csv') 

# Default automl models 
AUTOML_DEFAULT_MODELS = {'Linear Regressoin': LinearRegression(fit_intercept=True, copy_X=True), 
                         'Lasso': Lasso(fit_intercept=True), 
                         'Ridge': Ridge(fit_intercept=True),
                         'SVR': SVR(kernel= 'rbf'),                  

                         # Note that the API is different for xgb ... need to pass in data directly
                         # Come back and fix this 
                         #'XGBoosting': xb),   
        }
