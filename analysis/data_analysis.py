import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_provider.data_loader_yield import ShardedYieldDataset

def analyze_data(args):
    """
    Loads the yield dataset, performs per-crop analysis, and generates
    summary statistics and visualizations.
    """
    print("Loading dataset for analysis...")
    # Initialize the dataset to get access to metadata and file handles
    # We use 'train' flag to trigger the index and split creation
    dataset = ShardedYieldDataset(
        data_path=args.data_path,
        regions=args.regions,
        flag='train'
    )

    # --- Prepare data for analysis ---
    print("Preparing data for analysis...")

    # Use the metadata DataFrame which is already available
    analysis_df = dataset.metadata.copy()

    # Extract target values (yield) for all samples
    targets = []
    for g_idx in range(len(dataset.global_index)):
        region, local_idx = dataset.global_index[g_idx]
        # Assuming target is a single value, take the first element
        target_value = dataset.data_files[region]['targets'][local_idx][0]
        targets.append(target_value)

    analysis_df['yield'] = targets

    # Map crop IDs to names for readability
    crop_map = {1.0: 'Maize', 2.0: 'Rice', 3.0: 'Soybean', 4.0: 'Wheat'}
    analysis_df['crop_name'] = analysis_df['crop_id'].map(crop_map)

    # --- Perform Statistical Analysis ---
    print("\n--- Per-Crop Yield Statistics ---")
    yield_stats = analysis_df.groupby('crop_name')['yield'].describe()
    print(yield_stats)

    # --- Generate Visualizations ---
    print("\nGenerating visualizations...")
    output_dir = os.path.dirname(__file__)

    # Yield Distribution Plot
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='crop_name', y='yield', data=analysis_df)
    plt.title('Yield Distribution by Crop Type')
    plt.xlabel('Crop Type')
    plt.ylabel('Yield')
    yield_plot_path = os.path.join(output_dir, 'yield_distribution_by_crop.png')
    plt.savefig(yield_plot_path)
    print(f"Saved yield distribution plot to: {yield_plot_path}")
    plt.close()

    print("\nData analysis complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yield Data Analysis Script')
    parser.add_argument('--data_path', type=str, default='dataset/global_yield_dataset/', help='path to the dataset')
    parser.add_argument('--regions', type=str, default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16', help='list of regions to process')

    args = parser.parse_args()

    analyze_data(args)
