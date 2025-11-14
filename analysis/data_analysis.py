
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

DYNAMIC_FEATURE_COLUMNS = [
    'NDVI', 'Wind_Speed_10m_Mean', 'Temperature_Air_2m_Min_24h',
    'Temperature_Air_2m_Max_24h', 'Temperature_Air_2m_Mean_24h',
    'Temperature_Air_2m_Max_Day_Time', 'Temperature_Air_2m_Mean_Day_Time',
    'Temperature_Air_2m_Min_Night_Time', 'Temperature_Air_2m_Mean_Night_Time',
    'Dew_Point_Temperature_2m_Mean', 'Precipitation_Flux',
    'Precipitation_Rain_Duration_Fraction', 'Precipitation_Solid_Duration_Fraction',
    'Snow_Thickness_Mean', 'Snow_Thickness_LWE_Mean', 'Vapour_Pressure_Mean',
    'Solar_Radiation_Flux', 'Cloud_Cover_Mean', 'Relative_Humidity_2m_06h',
    'Relative_Humidity_2m_15h'
]

# Automatically select all top-layer (0-5cm) soil features for analysis
SOIL_FEATURES_TO_ANALYZE = [col for col in SOIL_FEATURES if '0-5cm' in col]

# Select key dynamic features for trend analysis
DYNAMIC_FEATURES_TO_ANALYZE = [
    'NDVI',
    'Temperature_Air_2m_Mean_24h',
    'Precipitation_Flux',
    'Solar_Radiation_Flux'
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
    Loads static, dynamic, and target data from sharded .npy files.
    Returns a pandas DataFrame for static/targets and a NumPy array for dynamic features.
    """
    all_static_data, all_target_data, all_dynamic_data = [], [], []

    print("Loading data from sharded .npy files...")
    for region in tqdm(regions, desc="Processing Regions"):
        region_path = os.path.join(data_path, region)
        if not os.path.isdir(region_path):
            print(f"Warning: Region directory not found: {region_path}. Skipping.")
            continue

        static_file = os.path.join(region_path, 'static_features.npy')
        target_file = os.path.join(region_path, 'targets.npy')
        dynamic_file = os.path.join(region_path, 'dynamic_features.npy')

        if os.path.exists(static_file) and os.path.exists(target_file) and os.path.exists(dynamic_file):
            static_data = np.load(static_file, mmap_mode='r')
            target_data = np.load(target_file, mmap_mode='r')
            dynamic_data = np.load(dynamic_file, mmap_mode='r')

            if static_data.shape[1] != len(STATIC_FEATURE_COLUMNS):
                 raise ValueError(f"Mismatch in static features for region '{region}'. "
                                 f"Expected {len(STATIC_FEATURE_COLUMNS)}, but found {static_data.shape[1]}.")

            all_static_data.append(static_data)
            all_target_data.append(target_data)
            all_dynamic_data.append(dynamic_data)
        else:
            print(f"Warning: Data files not found for region {region}. Skipping.")

    if not all_static_data:
        raise FileNotFoundError("No data was loaded. Please check --data_path and --regions.")

    # Concatenate data from all regions
    df = pd.DataFrame(np.vstack(all_static_data), columns=STATIC_FEATURE_COLUMNS)
    df['yield'] = np.vstack(all_target_data)
    dynamic_features = np.vstack(all_dynamic_data)

    print(f"Successfully loaded {len(df)} samples.")
    return df, dynamic_features

def analyze_static_features(df, output_dir):
    """Analyzes yield and static feature distributions."""
    print("--- Starting Static Feature Analysis ---")
    os.makedirs(output_dir, exist_ok=True)
    unique_crops = sorted(df['crop_id'].unique())

    # 1. Yield Distribution
    plt.figure(figsize=(15, 10))
    for crop_id in unique_crops:
        crop_name = CROP_MAP.get(crop_id, f"ID_{crop_id}")
        sns.kdeplot(df[df['crop_id'] == crop_id]['yield'], label=crop_name, fill=True, alpha=0.3)
    plt.title('Comparative Yield Distribution by Crop')
    plt.xlabel('Yield'); plt.ylabel('Density')
    plt.legend(); plt.savefig(os.path.join(output_dir, 'yield_distribution.png')); plt.close()
    print("Saved yield distribution plot.")

    yield_stats_df = df.groupby('crop_id')['yield'].describe()
    yield_stats_df['crop_name'] = yield_stats_df.index.map(CROP_MAP)
    yield_stats_df.to_csv(os.path.join(output_dir, 'yield_summary_stats.csv'))
    print("Saved yield summary stats.\n", yield_stats_df)

    # 2. Soil Feature Distributions
    print(f"\nAnalyzing {len(SOIL_FEATURES_TO_ANALYZE)} top-layer soil features...")
    for feature in tqdm(SOIL_FEATURES_TO_ANALYZE, desc="Soil Features"):
        plt.figure(figsize=(15, 10))
        for crop_id in unique_crops:
            crop_name = CROP_MAP.get(crop_id)
            sns.kdeplot(df[df['crop_id'] == crop_id][feature], label=crop_name, fill=True, alpha=0.3)
        plt.title(f'Distribution of "{feature}" by Crop'); plt.xlabel(feature); plt.ylabel('Density')
        plt.legend(); plt.savefig(os.path.join(output_dir, f'soil_dist_{feature}.png')); plt.close()

    soil_stats_df = df.groupby('crop_id')[SOIL_FEATURES_TO_ANALYZE].describe()
    soil_stats_df.to_csv(os.path.join(output_dir, 'soil_features_summary_stats.csv'))
    print("Saved all soil feature plots and summary stats.")

def analyze_dynamic_features(df, dynamic_features, output_dir):
    """Analyzes dynamic feature trends across the year."""
    print("\n--- Starting Dynamic Feature Analysis ---")
    os.makedirs(output_dir, exist_ok=True)
    unique_crops = sorted(df['crop_id'].unique())
    days = np.arange(1, 366)

    all_trends_data = []

    for feature_name in tqdm(DYNAMIC_FEATURES_TO_ANALYZE, desc="Dynamic Features"):
        feature_idx = DYNAMIC_FEATURE_COLUMNS.index(feature_name)
        plt.figure(figsize=(15, 10))

        for crop_id in unique_crops:
            crop_name = CROP_MAP.get(crop_id, f"ID_{crop_id}")

            # Get all dynamic data for the current crop
            crop_mask = (df['crop_id'] == crop_id).values
            crop_dynamic_data = dynamic_features[crop_mask, :, feature_idx] # (n_samples, 365)

            # Calculate mean for each day, ignoring zeros from padding
            daily_means = []
            for day in range(365):
                daily_values = crop_dynamic_data[:, day]
                non_zero_values = daily_values[daily_values != 0]
                daily_means.append(np.mean(non_zero_values) if non_zero_values.size > 0 else 0)

            plt.plot(days, daily_means, label=crop_name)

            # Store data for CSV export
            for day, value in enumerate(daily_means, 1):
                all_trends_data.append([feature_name, crop_name, day, value])

        plt.title(f'Average Annual Trend of "{feature_name}" by Crop'); plt.xlabel('Day of Year'); plt.ylabel(f'Mean {feature_name}')
        plt.grid(True); plt.legend(); plt.savefig(os.path.join(output_dir, f'dynamic_trend_{feature_name}.png')); plt.close()

    # Save the collected trend data to a single CSV
    trends_df = pd.DataFrame(all_trends_data, columns=['feature', 'crop_name', 'day_of_year', 'mean_value'])
    trends_df.to_csv(os.path.join(output_dir, 'dynamic_features_annual_trends.csv'), index=False)
    print("Saved all dynamic feature trend plots and summary CSV.")


def main():
    """Main execution function."""
    args = parse_args()
    try:
        df, dynamic_features = load_data(args.data_path, args.regions.split(','))

        # Create a dedicated sub-directory for static and dynamic analysis outputs
        static_output_dir = os.path.join(args.output_dir, 'static_features')
        dynamic_output_dir = os.path.join(args.output_dir, 'dynamic_features')

        analyze_static_features(df, static_output_dir)
        analyze_dynamic_features(df, dynamic_features, dynamic_output_dir)

        print(f"\nAnalysis complete. All results saved in '{args.output_dir}'")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Please ensure paths are correct and data has been processed via etl.py.")

if __name__ == '__main__':
    main()
