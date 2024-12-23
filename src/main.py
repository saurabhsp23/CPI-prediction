# Main script to run the modular pipeline

from imports import *
import data_processing
import model_training
import evaluation
import visualization
import utils

def run_pipeline():
    print('Pipeline Execution Started...')

    # Step 1: Data Processing
    data_processing.load_data()
    data_processing.preprocess_data()
    data_processing.feature_engineering()

    # Step 2: Model Training
    model_training.train_model()
    model_training.optimize_model()
    model_training.save_model()

    # Step 3: Evaluation
    evaluation.evaluate_model()
    evaluation.generate_metrics_report()

    # Step 4: Visualization
    visualization.plot_results()
    visualization.create_summary_charts()

    print('Pipeline Execution Completed!')

if __name__ == '__main__':
    run_pipeline()
