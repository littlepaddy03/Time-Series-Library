
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Constants ---

# Map crop IDs to names for labeling outputs
CROP_MAP = {
    1.0: 'Maize',
    2.0: 'Rice',
    3.0: 'Soybean',
    4.0: 'Wheat',
}

# Define the full list of static feature columns based on etl.py
# This is crucial for creating the pandas DataFrame with correct headers.
CONTEXT_FEATURES = ['longitude', 'latitude', 'year', 'crop_id']
SOIL_FEATURES = [
    'bdod_bdod_0-5cm_mean', 'bdod_bdod_100-200cm_mean', 'bdod_bdod_15-30cm_mean',
    'bdod_bdod_30-60cm_mean', 'bdod_bdod_5-15cm_mean', 'bdod_bdod_60-100cm_mean',
    'cec_cec_0-5cm_mean', 'cec_cec_100-200cm_mean', 'cec_cec_15-30cm_mean',
    'cec_cec_30-60cm_mean', 'cec_cec_5-15cm_mean', 'cec_cec_60-100cm_mean',
    'cfvo_cfvo_0-5cm_mean', 'cfvo_cfvo_100-200cm_mean', 'cfvo_cfvo_15-30cm_mean',
    'cfvo_cfvo_30-60cm_mean', 'cfvo_cfvo_5-15cm_mean', 'cfvo_cfvo_60-100cm_mean',
    'clay_clay_0-5cm_mean', 'clay_clay_100-200cm_mean', 'clay_clay_15-30cm_mean',
    'clay_clay_30-60cm_mean', 'clay_clay_5-15cm_mean', 'clay_clay_60-100cm_mean',
    'nitrogen_nitrogen_0-5cm_mean', 'nitrogen_nitrogen_100-200cm_mean',
    'nitrogen_nitrogen_15-30cm_mean', 'nitrogen_nitrogen_30-60cm_mean',
    'nitrogen_nitrogen_5-15cm_mean', 'nitrogen_nitrogen_60-100cm_mean',
    'ocd_ocd_0-5cm_mean', 'ocd_ocd_100-200cm_mean', 'ocd_ocd_15-30cm_mean',
    'ocd_ocd_30-60cm_mean', 'ocd_ocd_5-15cm_mean', 'ocd_ocd_60-100cm_mean',
    'ocs_ocs_0-30cm_mean',
    'phh2o_phh2o_0-5cm_mean', 'phh2o_phh2o_100-200cm_mean',
    'phh2o_phh2o_15-30cm_mean', 'phh2o_phh2o_30-60cm_mean',
    'phh2o_phh2o_5-15cm_mean', 'phh2o_phh2o_60-100cm_mean',
    'sand_sand_0-5cm_mean', 'sand_sand_100-200cm_mean', 'sand_sand_15-30cm_mean',
    'sand_sand_30-60cm_mean', 'sand_sand_5-15cm_mean', 'sand_sand_60-100cm_mean',
    'silt_silt_0-5cm_mean', 'silt_silt_100-200cm_mean', 'silt_silt_15-30cm_mean',
    'silt_silt_30-60cm_mean', 'silt_silt_5-15cm_mean', 'silt_silt_60-100cm_mean',
    'soc_soc_0-5cm_mean', 'soc_soc_100-200cm_mean', 'soc_soc_15-30cm_mean',
    'soc_soc_30-60cm_mean', 'soc_soc_5-15cm_mean', 'soc_soc_60-100cm_mean'
]
CLIMATE_FEATURE = ['koppen_geiger_zone']
STATIC_FEATURE_COLUMNS = CONTEXT_FEATURES + SOIL_FEATURES + CLIMATE_FEATURE

# A smaller, representative subset of features to analyze in detail
FEATURES_TO_ANALYZE = [
    'latitude',
    'bdod_bdod_0-5cm_mean',    # Soil bulk density
    'phh2o_phh2o_0-5cm_mean',  # Soil pH
    'soc_soc_0-5cm_mean',      # Soil organic carbon
    'nitrogen_nitrogen_0-5cm_mean' # Soil nitrogen
]

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Data Analysis Script for Crop Yield Prediction')
    parser.add_argument('--data_path', type=str, required=True, help='Root directory of the sharded dataset (e.g., ./dataset/global_yield_dataset/)')
    parser.add_argument('--regions', type=str, required=True, help='Comma-separated list of regions to process (e.g., ar,br,cn,eu,in,us)')
    parser.add_argument('--output_dir', type=str, default='./analysis_output', help='Directory to save analysis results')
    return parser.parse_args()

def load_data(data_path, regions):
    """
    Loads static features and targets from sharded .npy files into a single pandas DataFrame.
    """
    all_static_data = []
    all_target_data = []

    print("Loading data from sharded .npy files...")
    for region in tqdm(regions, desc="Processing Regions"):
        region_path = os.path.join(data_path, region)
        if not os.path.isdir(region_path):
            print(f"Warning: Region directory not found: {region_path}. Skipping.")
            continue

        static_file = os.path.join(region_path, 'static_features.npy')
        target_file = os.path.join(region_path, 'targets.npy')

        if os.path.exists(static_file) and os.path.exists(target_file):
            static_data = np.load(static_file, mmap_mode='r')
            target_data = np.load(target_file, mmap_mode='r')

            # Ensure the number of static features matches the expected column count
            if static_data.shape[1] != len(STATIC_FEATURE_COLUMNS):
                 raise ValueError(f"Mismatch in static features for region '{region}'. "
                                 f"Expected {len(STATIC_FEATURE_COLUMNS)} columns, but found {static_data.shape[1]}. "
                                 "Please ensure etl.py has been run correctly.")

            all_static_data.append(static_data)
            all_target_data.append(target_data)
        else:
            print(f"Warning: Data files not found for region {region}. Skipping.")

    if not all_static_data:
        raise FileNotFoundError("No data was loaded. Please check --data_path and --regions.")

    # Concatenate data from all regions
    combined_static = np.vstack(all_static_data)
    combined_targets = np.vstack(all_target_data)

    # Create DataFrame
    df = pd.DataFrame(combined_static, columns=STATIC_FEATURE_COLUMNS)
    df['yield'] = combined_targets
    print(f"Successfully loaded {len(df)} samples.")
    return df

def analyze_data(df, output_dir):
    """
    Performs analysis on the loaded data, generating plots and summary statistics.
    """
    print(f"Starting data analysis. Results will be saved to '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)

    # --- Analysis Part 1: Yield Distribution ---
    yield_stats_list = []
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")

    unique_crops = sorted(df['crop_id'].unique())

    for crop_id in unique_crops:
        crop_name = CROP_MAP.get(crop_id, f"Unknown_{crop_id}")
        crop_df = df[df['crop_id'] == crop_id]
        yield_series = crop_df['yield']

        # Collect summary statistics
        stats = yield_series.describe()
        stats['crop_name'] = crop_name
        yield_stats_list.append(stats)

        # Plot distribution for each crop on the same axes for comparison
        sns.kdeplot(yield_series, label=crop_name, fill=True, alpha=0.3)

    plt.title('Comparative Yield Distribution by Crop', fontsize=16)
    plt.xlabel('Yield', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    yield_dist_path = os.path.join(output_dir, 'yield_distribution_comparison.png')
    plt.savefig(yield_dist_path)
    plt.close()
    print(f"Saved yield distribution plot to: {yield_dist_path}")

    # Save yield statistics to CSV
    yield_stats_df = pd.DataFrame(yield_stats_list).set_index('crop_name')
    yield_csv_path = os.path.join(output_dir, 'yield_summary_stats.csv')
    yield_stats_df.to_csv(yield_csv_path)
    print(f"Saved yield summary stats to: {yield_csv_path}")
    print("\nYield Summary Statistics:\n", yield_stats_df)

    # --- Analysis Part 2: Feature Distribution ---
    feature_stats_list = []
    for feature_name in FEATURES_TO_ANALYZE:
        stats_by_crop = df.groupby('crop_id')[feature_name].describe()
        stats_by_crop['crop_name'] = stats_by_crop.index.map(CROP_MAP)
        stats_by_crop = stats_by_crop.set_index('crop_name')
        feature_stats_list.append(stats_by_crop)

        # Plot feature distribution
        plt.figure(figsize=(15, 10))
        for crop_id in unique_crops:
            crop_name = CROP_MAP.get(crop_id)
            sns.kdeplot(df[df['crop_id'] == crop_id][feature_name], label=crop_name, fill=True, alpha=0.3)
        plt.title(f'Comparative Distribution of "{feature_name}" by Crop', fontsize=16)
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        feature_dist_path = os.path.join(output_dir, f'feature_dist_{feature_name}.png')
        plt.savefig(feature_dist_path)
        plt.close()
        print(f"Saved {feature_name} distribution plot to: {feature_dist_path}")


    # Save feature statistics to CSV
    feature_summary_df = pd.concat(feature_stats_list, keys=FEATURES_TO_ANALYZE, axis=0)
    feature_csv_path = os.path.join(output_dir, 'feature_summary_stats.csv')
    feature_summary_df.to_csv(feature_csv_path)
    print(f"Saved feature summary stats to: {feature_csv_path}")
    print("\nFeature Summary Statistics:\n", feature_summary_df)


def main():
    """Main execution function."""
    args = parse_args()
    try:
        df = load_data(args.data_path, args.regions.split(','))
        analyze_data(df, args.output_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Please ensure the provided paths are correct and the data has been processed.")

if __name__ == '__main__':
    main()
