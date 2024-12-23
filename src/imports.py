# Standard library imports

import datetime 
import itertools
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from os.path import join
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import r2_score, mean_absolute_error,mean_absolute_percentage_error
from sklearn.linear_model import Lasso
import xgboost as xgb
import inspect
from scipy.fft import fft, ifft
import statsmodels.api as sm
from scipy.stats import f
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")