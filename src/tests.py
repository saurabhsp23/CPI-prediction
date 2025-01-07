from imports import *
from LR_implementation import *

class Test():
    def __init__(self, combinations):
        self.combinations = combinations

    @staticmethod
    def linear_reg():
        return LinearRegression()

    # stepwise_reg = StepwiseRegression(criterion='bic', verbose=False)
    # lasso = Lasso(alpha=0.01)
    # xgboosting = xgb.XGBRegressor(objective ='reg:squarederror',
    #                      colsample_bytree = 0.3,
    #                      learning_rate = 0.1,
    #                      max_depth = 5,
    #                      alpha = 10,
    #                      n_estimators = 10)
