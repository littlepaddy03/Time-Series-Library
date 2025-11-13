import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
# Crop ID to Name Mapping, based on etl.py
CROP_MAP = {
    1.0: 'Maize',
    2.0: 'Rice',
    3.0: 'Soybean',
    4.0: 'Wheat',
}

# Selected features for distribution analysis to avoid clutter
SELECTED_DYNAMIC_FEATURES = {
    'Temperature_Air_2m_Mean_24h': 4,
    'Precipitation_Flux': 10,
    'Solar_Radiation_Flux': 16,
    'NDVI': 0
}
SELECTED_STATIC_FEATURES = {
    'bdod_bdod_0-5cm_mean': 5, # Context features are 0-3, Köppen is last
    'cec_cec_0-5cm_mean': 8,
    'clay_clay_0-5cm_mean': 20,
    'nitrogen_nitrogen_0-5cm_mean': 26,
    'phh2o_phh2o_0-5cm_mean': 39,
    'soc_soc_0-5cm_mean': 57
}

class DataAnalyzer:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.regions = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        os.makedirs(self.output_path, exist_ok=True)
        print(f"Data source: {self.data_path}")
        print(f"Output path: {self.output_path}")
        print(f"Found {len(self.regions)} regions: {self.regions}")

    def load_and_preprocess_data(self):
        """Loads all sharded data and returns a unified metadata DataFrame with yields."""
        print("Loading and preprocessing data...")
        all_metadata = []

        for region in tqdm(self.regions, desc="Loading regions"):
            region_path = os.path.join(self.data_path, region)
            try:
                static_features = np.load(os.path.join(region_path, 'static_features.npy'), mmap_mode='r')
                targets = np.load(os.path.join(region_path, 'targets.npy'), mmap_mode='r')

                # Extract context from static features: lon, lat, year, crop_id
                region_df = pd.DataFrame(static_features[:, [0, 1, 2, 3]], columns=['lon', 'lat', 'year', 'crop_id'])
                region_df['yield'] = targets
                region_df['region'] = region
                all_metadata.append(region_df)
            except FileNotFoundError:
                print(f"Warning: Data files not found for region '{region}', skipping.")
                continue

        if not all_metadata:
            raise ValueError("No data loaded. Check the data_path.")

        self.metadata_df = pd.concat(all_metadata, ignore_index=True)
        self.metadata_df['crop_name'] = self.metadata_df['crop_id'].map(CROP_MAP)
        print("Data loading complete.")
        print("Basic DataFrame info:")
        self.metadata_df.info()

    def analyze_yield_distribution(self):
        """Analyzes and visualizes the distribution of crop yields."""
        print("\n--- 1. Analyzing Yield Distribution ---")
        df = self.metadata_df

        # Print summary statistics
        print("Yield statistics by crop:")
        print(df.groupby('crop_name')['yield'].describe())

        # Plotting
        plt.style.use('seaborn-v0_8-whitegrid')

        # Boxplot
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(x='crop_name', y='yield', data=df, ax=ax)
        ax.set_title('Yield Distribution by Crop', fontsize=16)
        ax.set_xlabel('Crop', fontsize=12)
        ax.set_ylabel('Yield', fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(self.output_path, 'yield_distribution_boxplot.png')
        plt.savefig(save_path)
        print(f"Saved yield distribution boxplot to {save_path}")
        plt.close()

        # Histogram / KDE Plot
        g = sns.FacetGrid(df, col='crop_name', col_wrap=2, height=5, aspect=1.5, sharex=False, sharey=False)
        g.map(sns.histplot, 'yield', kde=True, bins=30)
        g.set_titles("Yield Distribution for {col_name}", size=14)
        g.set_axis_labels("Yield", "Count")
        plt.tight_layout()
        save_path = os.path.join(self.output_path, 'yield_distribution_histograms.png')
        plt.savefig(save_path)
        print(f"Saved yield distribution histograms to {save_path}")
        plt.close()

    def check_missing_values(self):
        """Checks for missing values in dynamic and static features."""
        print("\n--- 2. Checking for Missing & Anomaly Values ---")

        nan_counts = {'dynamic': 0, 'static': 0, 'targets': 0}
        total_counts = {'dynamic': 0, 'static': 0, 'targets': 0}

        for region in tqdm(self.regions, desc="Checking for NaNs"):
            region_path = os.path.join(self.data_path, region)
            try:
                dynamic = np.load(os.path.join(region_path, 'dynamic_features.npy'))
                static = np.load(os.path.join(region_path, 'static_features.npy'))
                targets = np.load(os.path.join(region_path, 'targets.npy'))

                nan_counts['dynamic'] += np.isnan(dynamic).sum()
                nan_counts['static'] += np.isnan(static).sum()
                nan_counts['targets'] += np.isnan(targets).sum()

                total_counts['dynamic'] += np.size(dynamic)
                total_counts['static'] += np.size(static)
                total_counts['targets'] += np.size(targets)

            except FileNotFoundError:
                continue

        print("NaN Value Check Results:")
        for key in nan_counts:
            print(f"  - {key.capitalize()}: {nan_counts[key]} NaNs out of {total_counts[key]} total values.")

        print("\nYield Anomaly Check (values <= 0):")
        non_positive_yields = self.metadata_df[self.metadata_df['yield'] <= 0]
        if not non_positive_yields.empty:
            print(f"Found {len(non_positive_yields)} samples with yield <= 0.")
            print(non_positive_yields.head())
        else:
            print("No samples with yield <= 0 found.")

    def analyze_feature_distribution(self):
        """Analyzes and compares distributions of selected features across crops."""
        print("\n--- 3. Analyzing Feature Distribution ---")

        # Inefficient to load all data. We'll sample a subset of indices for this analysis.
        sample_indices, _ = train_test_split(
            self.metadata_df.index,
            train_size=min(20000, len(self.metadata_df)), # Sample up to 20k points
            stratify=self.metadata_df['crop_id'],
            random_state=42
        )
        sampled_df = self.metadata_df.loc[sample_indices].copy()

        # --- Dynamic Feature Analysis ---
        print("Analyzing dynamic features (using mean across sequence)...")
        dynamic_means = []
        # Re-construct global index for quick lookup
        self.metadata_df['global_idx'] = self.metadata_df.index
        region_groups = self.metadata_df.groupby('region')
        region_indices = {name: group['global_idx'].tolist() for name, group in region_groups}

        for idx in tqdm(sampled_df.index, desc="Calculating dynamic means"):
            row = self.metadata_df.loc[idx]
            region = row['region']
            # Find the local index within the region's original file
            local_idx = region_indices[region].index(idx)
            dynamic_file = np.load(os.path.join(self.data_path, region, 'dynamic_features.npy'), mmap_mode='r')
            # Calculate mean of non-zero values for each feature
            sample_dynamic = dynamic_file[local_idx]
            means = [np.mean(sample_dynamic[:, col_idx][sample_dynamic[:, col_idx] != 0]) if (sample_dynamic[:, col_idx] != 0).any() else 0
                     for col_name, col_idx in SELECTED_DYNAMIC_FEATURES.items()]
            dynamic_means.append(means)

        dynamic_df = pd.DataFrame(dynamic_means, index=sampled_df.index, columns=SELECTED_DYNAMIC_FEATURES.keys())
        sampled_df = pd.concat([sampled_df, dynamic_df], axis=1)

        for col_name in SELECTED_DYNAMIC_FEATURES:
            plt.figure(figsize=(12, 7))
            sns.violinplot(x='crop_name', y=col_name, data=sampled_df)
            plt.title(f'Mean {col_name} Distribution by Crop', fontsize=16)
            save_path = os.path.join(self.output_path, f'dynamic_feature_{col_name}.png')
            plt.savefig(save_path)
            plt.close()
        print(f"Saved dynamic feature plots to {self.output_path}")

        # --- Static Feature Analysis ---
        print("Analyzing static features...")
        # We need to load the full static feature matrix for the sampled indices
        static_features_list = []
        for idx in tqdm(sampled_df.index, desc="Loading static features"):
            row = self.metadata_df.loc[idx]
            region = row['region']
            local_idx = region_indices[region].index(idx)
            static_file = np.load(os.path.join(self.data_path, region, 'static_features.npy'), mmap_mode='r')
            static_features_list.append(static_file[local_idx])

        full_static_df = pd.DataFrame(static_features_list, index=sampled_df.index)

        for col_name, col_idx in SELECTED_STATIC_FEATURES.items():
            sampled_df[col_name] = full_static_df[col_idx]
            plt.figure(figsize=(12, 7))
            sns.violinplot(x='crop_name', y=col_name, data=sampled_df)
            plt.title(f'{col_name} Distribution by Crop', fontsize=16)
            save_path = os.path.join(self.output_path, f'static_feature_{col_name}.png')
            plt.savefig(save_path)
            plt.close()
        print(f"Saved static feature plots to {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Data Analysis Script for Global Yield Dataset")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the root of the sharded dataset (e.g., 'dataset/global_yield_dataset/')")
    parser.add_argument('--output_path', type=str, default='./data_analysis_output', help="Directory to save analysis plots and reports")
    args = parser.parse_args()

    analyzer = DataAnalyzer(data_path=args.data_path, output_path=args.output_path)
    analyzer.load_and_preprocess_data()
    analyzer.analyze_yield_distribution()
    analyzer.check_missing_values()
    analyzer.analyze_feature_distribution()

    print("\n--- Analysis Complete ---")

if __name__ == '__main__':
    main()
