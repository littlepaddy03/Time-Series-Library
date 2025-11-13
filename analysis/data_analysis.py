
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import warnings

# --- Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description='Data Analysis Script for Crop Yield')
    parser.add_argument('--data_path', type=str, default='./dataset/global_yield_dataset/', help='Path to the processed data')
    parser.add_argument('--output_path', type=str, default='./output/data_analysis_results/', help='Path to save analysis results')
    parser.add_argument('--regions', type=str, default='ar,br,cn,eu,in,us', help='Comma-separated list of regions to analyze')
    return parser.parse_args()

CROP_MAP = {1.0: 'Maize', 2.0: 'Rice', 3.0: 'Soybean', 4.0: 'Wheat'}
STATIC_FEATURE_NAMES = [
    'longitude', 'latitude', 'year', 'crop_id',
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
    'soc_soc_30-60cm_mean', 'soc_soc_5-15cm_mean', 'soc_soc_60-100cm_mean',
    'kg_zone' # Added Köppen-Geiger zone
]

DYNAMIC_FEATURE_NAMES = [
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
# --- Data Loading ---
def load_all_data(data_path, regions):
    """Loads and concatenates data from all specified regions."""
    all_static_dfs, all_dynamic_dfs, all_targets_dfs = [], [], []

    print("Loading data...")
    for region in tqdm(regions, desc="Loading regions"):
        region_path = os.path.join(data_path, region)
        if not os.path.exists(region_path):
            print(f"Warning: Region path not found: {region_path}")
            continue

        static_features = np.load(os.path.join(region_path, 'static_features.npy'))
        dynamic_features = np.load(os.path.join(region_path, 'dynamic_features.npy'))
        targets = np.load(os.path.join(region_path, 'targets.npy'))

        num_samples = targets.shape[0]
        region_ids = np.array([f"{region}_{i}" for i in range(num_samples)])

        # Static Data
        static_df = pd.DataFrame(static_features, columns=STATIC_FEATURE_NAMES)
        static_df['sample_id'], static_df['region'] = region_ids, region
        all_static_dfs.append(static_df)

        # Dynamic Data (aggregated)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dynamic_means = np.nan_to_num(np.mean(dynamic_features, axis=1))
            dynamic_std = np.nan_to_num(np.std(dynamic_features, axis=1))

        agg_dynamic_df = pd.DataFrame({f"{col}_mean": dynamic_means[:, i] for i, col in enumerate(DYNAMIC_FEATURE_NAMES)})
        agg_dynamic_df.update({f"{col}_std": dynamic_std[:, i] for i, col in enumerate(DYNAMIC_FEATURE_NAMES)})
        agg_dynamic_df['sample_id'] = region_ids
        all_dynamic_dfs.append(agg_dynamic_df)

        # Target Data
        targets_df = pd.DataFrame(targets, columns=['yield'])
        targets_df['sample_id'] = region_ids
        all_targets_dfs.append(targets_df)

    if not all_static_dfs:
        print("Error: No data loaded."); return None

    static_df_full = pd.concat(all_static_dfs, ignore_index=True)
    dynamic_df_full = pd.concat(all_dynamic_dfs, ignore_index=True)
    targets_df_full = pd.concat(all_targets_dfs, ignore_index=True)

    df = pd.merge(static_df_full, dynamic_df_full, on='sample_id')
    df = pd.merge(df, targets_df_full, on='sample_id')
    df['crop_name'] = df['crop_id'].map(CROP_MAP)
    return df

# --- Analysis and Plotting ---
def generate_summary_report(df, output_path):
    """Generates a text report with descriptive statistics for each crop."""
    print("Generating summary report...")
    with open(os.path.join(output_path, 'summary_report.txt'), 'w') as f:
        f.write("--- Overall Dataset Statistics ---\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write("Crop distribution:\n")
        f.write(str(df['crop_name'].value_counts()))
        f.write("\n\n")

        for crop_name, group in df.groupby('crop_name'):
            f.write(f"--- Statistics for {crop_name} ---\n")
            f.write(group.describe().to_string())
            f.write("\n\n")
    print("Summary report saved.")

def plot_distributions(df, features, output_path):
    """Plots and saves distributions (box plot and histogram) for given features, grouped by crop."""
    print(f"Plotting distributions for {len(features)} features...")
    for feature in tqdm(features, desc="Plotting features"):
        plt.figure(figsize=(15, 7))

        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(x='crop_name', y=feature, data=df)
        plt.title(f'Box Plot of {feature} by Crop')

        # Histogram
        plt.subplot(1, 2, 2)
        sns.histplot(data=df, x=feature, hue='crop_name', kde=True, multiple="stack")
        plt.title(f'Histogram of {feature} by Crop')

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'dist_{feature}.png'))
        plt.close()
    print("Distribution plots saved.")

def plot_yield_per_region(df, output_path):
    """Plots the average yield per region for each crop."""
    print("Plotting yield per region...")
    plt.figure(figsize=(15, 8))
    sns.barplot(x='region', y='yield', hue='crop_name', data=df.groupby(['region', 'crop_name'])['yield'].mean().reset_index())
    plt.title('Average Yield by Region and Crop')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'yield_by_region.png'))
    plt.close()
    print("Yield plot saved.")

# --- Main Execution ---
def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    regions = args.regions.split(',')
    df = load_all_data(args.data_path, regions)

    if df is None: return

    generate_summary_report(df, args.output_path)

    # Select some key features for visualization to avoid generating too many plots
    KEY_STATIC_FEATURES = ['latitude', 'phh2o_phh2o_0-5cm_mean', 'nitrogen_nitrogen_0-5cm_mean', 'yield']
    KEY_DYNAMIC_FEATURES = ['NDVI_mean', 'Temperature_Air_2m_Mean_24h_mean', 'Precipitation_Flux_mean']

    plot_distributions(df, KEY_STATIC_FEATURES + KEY_DYNAMIC_FEATURES, args.output_path)
    plot_yield_per_region(df, args.output_path)

    print("\nData analysis complete. Results are saved in:", args.output_path)

if __name__ == '__main__':
    main()
