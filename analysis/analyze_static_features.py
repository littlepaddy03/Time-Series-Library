
import numpy as np
import pandas as pd
import argparse
import os

def analyze_features(data_path, region):
    """
    Analyzes the static features for a given region and prints summary statistics.

    Args:
        data_path (str): The root path to the dataset.
        region (str): The specific region to analyze.
    """
    print(f"--- Analyzing Static Features for Region: {region} ---")

    static_features_path = os.path.join(data_path, region, 'static_features.npy')

    if not os.path.exists(static_features_path):
        print(f"Error: Static features file not found at {static_features_path}")
        return

    try:
        static_features = np.load(static_features_path, mmap_mode='r')
    except Exception as e:
        print(f"Error loading numpy file: {e}")
        return

    num_samples, num_features = static_features.shape
    print(f"Loaded data with {num_samples} samples and {num_features} features.\n")

    # Create a DataFrame for easier analysis
    df = pd.DataFrame(static_features)

    # Calculate summary statistics for each feature
    summary = df.describe().T
    summary['unique_values'] = df.nunique()

    # Reorder columns for better readability
    summary = summary[['mean', 'std', 'min', 'max', 'unique_values']]

    print("--- Feature Statistics ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(summary)
    print("\n--- End of Analysis ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Static Features of the Yield Prediction Dataset')
    parser.add_argument('--data_path', type=str, default='./dataset/global_yield_dataset/',
                        help='Root path of the dataset')
    parser.add_argument('--regions', type=str, default='1,2,3,4,5,6,7,8',
                        help='Comma-separated list of regions to process. Only the first region will be analyzed.')

    args = parser.parse_args()

    # We only analyze the first region provided in the list for simplicity.
    first_region = args.regions.split(',')[0]

    analyze_features(args.data_path, first_region)
