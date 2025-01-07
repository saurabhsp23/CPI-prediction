from imports import *
class Filter():
    def __init__(self, combinations):
        self.combinations = combinations
        
    """
    Filter1: Fill the NaNs with the Prophet model, which is a procedure for forecasting time series based on 
             an additive model where non-linear trends are fit with yearly, weekly seasonality.
             Here we train the model with the data up to the NaN date, and fill the NaN interval with forecasting data.
    """
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

