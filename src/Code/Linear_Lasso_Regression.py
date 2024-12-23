#!/usr/bin/env python
# coding: utf-8

# In[13]:


import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
import statsmodels.api as sm
import pipeline

df_agg, df_ind = pipeline.read_and_format_data()
# Smoothing the aggregate survey data
rolling_window_size = 20
df_agg = pipeline.data_filter(df_agg, window_size=rolling_window_size)

# In[15]:


df_ind.columns

# In[16]:


df_ind.describe()

# In[17]:

reg_final_stats_df = pd.DataFrame(index=df_ind.columns, columns=['rsquared', 'mae', 'mape', 'p_value'])
reg_final_stats_df_pct_change = pd.DataFrame(index=df_ind.columns, columns=['rsquared', 'mae', 'mape', 'p_value'])

for indicator in df_ind.columns:
    # Get the combined monthly survey data with indicator data
    print(f"Indicator: {indicator}")
    if pd.api.types.is_numeric_dtype(df_ind[indicator]):
        df_comb = pipeline.get_combined_data_with_given_indicator(df_ind, df_agg, indicator_col=indicator)
        # df_comb_processed = df_comb[df_comb.iloc[:,:-1] <= 1].replace(np.inf, 0)
        df_comb_processed_change = df_comb.pct_change().replace(np.inf, 0)
        print("Lasso Model (alpha=0.01):")


        # reg_final_stats_df.loc[indicator], reg_coefs_df = pipeline.run_model(df_comb_processed,
        #                                                             lookback_period=12, model=Lasso(alpha=0.1))
        reg_final_stats_df_pct_change.loc[indicator], pct_change_reg_coefs_df = pipeline.run_model(
            df_comb_processed_change,  lookback_period=12, model=Lasso(alpha=0.01))

        df_comb_processed_change.to_csv(f'../results/{indicator}_combined_monthly_processed_data.csv')
        # reg_coefs_df.to_csv(f'../results/{indicator}_reg_coefs.csv')
        pct_change_reg_coefs_df.to_csv(f'../results/{indicator}_reg_pct_change_coefs.csv')
    break
# print(results_df)
# reg_final_stats_df.to_csv('../results/reg_final_stats.csv')
reg_final_stats_df_pct_change.to_csv('../results/reg_pct_change_final_stats.csv')

#
# # In[18]:
#
#
# df_comb
#
#
# # # Running regression models on the data as is
#
# # In[19]:
#
#
# def get_linear_regression_summary(X, y):
#     # adding the constant term
#     X = sm.add_constant(X)
#     # performing the regression
#     # and fitting the model
#     result = sm.OLS(y, X).fit()
#     # printing the summary table
#     print(result.summary())
#
# def get_lasso_regression_summary(X, y, alp = 0.001):
#     lasso = Lasso(alpha = alp)
#     lasso.fit(X, y)
#
#     coef = pd.Series(lasso.coef_, index = X.columns)
# #     print(lasso.coef_ != 0)
#     pred = lasso.predict(X)
#     train_score = lasso.score(X, y)
#     important_features = pd.concat([coef.sort_values().head(10),\
#                      coef.sort_values().tail(10)])
#     important_features.plot(kind = "barh")
#     plt.title(f"Coefficients in the Lasso Model with alpha = {alp}")
#     plt.show()
#     print(f'alpha = {alp}')
#     print(f'Lasso coef: {lasso.coef_}')
#     print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " \
#           +  str(sum(coef == 0)) + " variables")
#     print(f'Train score of trained model: {train_score*100}')
#
#
# # In[20]:
#
#
# for indicator in df_ind.columns:
#     # Get the combined monthly survey data with indicator data
#     print(f"Indicator: {indicator}")
#     if pd.api.types.is_numeric_dtype(df_ind[indicator]):
#         df_comb = pipeline.get_combined_data_with_given_indicator(df_ind, df_agg, indicator_col=indicator)
#         df_comb = df_comb.dropna(axis=1,how='all').bfill().ffill()
#         print(df_comb.shape)
#         X = df_comb.iloc[:,:-1]
#         y = df_comb.iloc[:,-1]
#         y = (y - np.mean(y))/np.std(y)
#         get_linear_regression_summary(X,y)
#         print('\n')
#
#
# # In[21]:
#
#
# for indicator in df_ind.columns:
#     # Get the combined monthly survey data with indicator data
#     print(f"Indicator: {indicator}")
#     if pd.api.types.is_numeric_dtype(df_ind[indicator]):
#         df_comb = pipeline.get_combined_data_with_given_indicator(df_ind, df_agg, indicator_col=indicator)
#         df_comb = df_comb.dropna(axis=1,how='all').bfill().ffill()
#         print(df_comb.shape)
#         X = df_comb.iloc[:,:-1]
#         y = df_comb.iloc[:,-1]
#         y = (y - np.mean(y))/np.std(y)
#         get_lasso_regression_summary(X,y)
#         print('\n')
#
