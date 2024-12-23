import pandas as pd 
from os.path import join 
from constants import DAILY_AGG_PATH, INDICATOR_DATA_PATH 

def read_and_format_data():  
    '''
    Read in time series of question responses from the daily_aggregate.pq and 
    initial_indicator_dataset.csv files 
    '''
    # Read daily aggregate file (predictor library)   
    df_agg = pd.read_parquet(DAILY_AGG_PATH) 

    # Read indicator data file (target library) 
    df_ind = pd.read_csv(INDICATOR_DATA_PATH)

    # Process indicator data  
    df_ind['Date'] = pd.to_datetime(df_ind['Date']) # convert dates to datetimes
    df_ind = df_ind.set_index('Date') # set the date column as the index 

    return df_agg, df_ind 

if __name__ == "__main__": 
    # Test the reader for the file  

    df_agg, df_ind = read_and_format_data() 

