from imports import *
from pipeline import Pipeline
from combined_aggregation import Combo, Agg
from utils import get_class_methods
from smooth import Smooth
from filter import Filter
from tests import Test

def run_pipeline():
    print('Pipeline Execution Started...')

    agg = Agg(['multi_and_gradient'])
    filt = Filter(['filter1'])
    smooth = Smooth(['smooth1'])
    test = Test(['linear_reg'])

    # Initialize Pipeline
    pipeline = Pipeline(agg, filt, smooth, test, lookback_period=12, base_path='../data')

    # Step 1: Read and format data
    print('Reading and formatting data...')
    pipeline.read_and_format_data()

    print('Applying data filter...')
    pipeline.data_filter(window_size=20)

    print('Running the pipeline...')
    pipeline.run()

    print('Pipeline Execution Completed!')


if __name__ == '__main__':
    run_pipeline()

