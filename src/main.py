from imports import *
from pipeline import Pipeline
from combined_aggregation import Combo, Agg
from utils import get_class_methods
from smooth import Smooth
from filter import Filter

def run_pipeline():
    print('Pipeline Execution Started...')

    # Define combinations for pipeline steps
    agg = Agg(['multi_and_gradient', 'subcat_agg'])
    filt = Filter(['filter1'])
    smooth = Smooth(['smooth1', 'smooth2'])
    test = Combo()

    # Initialize Pipeline
    pipeline = Pipeline(agg, filt, smooth, test, lookback_period=12, base_path='../Data')

    # Step 1: Read and format data
    print('Reading and formatting data...')
    pipeline.read_and_format_data()

    # Step 2: Apply data filtering
    print('Applying data filter...')
    filtered_data = pipeline.data_filter(window_size=20)

    # Step 3: Run the pipeline logic
    print('Running the pipeline...')
    results = pipeline.run()

    # Step 4: Process results (Optional visualization)
    print('Pipeline Execution Completed!')
    
    # Placeholder for visualization or saving results
    # plot_results()
    # create_summary_charts()

if __name__ == '__main__':
    run_pipeline()

