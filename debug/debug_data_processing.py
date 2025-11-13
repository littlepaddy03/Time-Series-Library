
import os
import numpy as np
import pandas as pd
import torch
import argparse

# This script mimics the data loading, splitting, and scaling process
# from data_provider/data_loader_yield.py for debugging purposes.

CROP_MAP_INV = {'Maize': 1.0, 'Rice': 2.0, 'Soybean': 3.0, 'Wheat': 4.0}
STATIC_FEATURE_NAMES = [ # Assuming 66 static features, names are for clarity
    'longitude', 'latitude', 'year', 'crop_id'] + [f'static_{i}' for i in range(62)]

def parse_args():
    parser = argparse.ArgumentParser(description='Debug Data Processing for Crop Yield')
    parser.add_argument('--data_path', type=str, default='./dataset/global_yield_dataset/', help='Path to the processed data')
    parser.add_argument('--regions', type=str, default='us', help='Comma-separated list of regions')
    parser.add_argument('--crop_name', type=str, default='Rice', help='Specific crop to debug')
    return parser.parse_args()

def main():
    args = parse_args()

    print("--- 1. Data Loading and Indexing ---")

    # Mimic the initial scan from ShardedYieldDataset
    global_index = []
    metadata_list = []
    regions = args.regions.split(',')

    for region in regions:
        region_path = os.path.join(args.data_path, region)
        if not os.path.isdir(region_path):
            print(f"Warning: Region path not found, skipping: {region_path}")
            continue

        static_features_mmap = np.load(os.path.join(region_path, 'static_features.npy'), mmap_mode='r')
        num_samples = static_features_mmap.shape[0]

        for i in range(num_samples):
            global_index.append((region, i))

        metadata_list.append(pd.DataFrame(static_features_mmap[:, [0, 1, 2, 3]], columns=['lon', 'lat', 'year', 'crop_id']))

    if not metadata_list:
        print("Error: No data loaded. Exiting.")
        return

    metadata = pd.concat(metadata_list, ignore_index=True)
    metadata['global_idx'] = np.arange(len(global_index))

    print(f"Total samples found across all regions: {len(metadata)}")

    # --- Filtering for the specific crop ---
    crop_id_to_filter = CROP_MAP_INV[args.crop_name]
    metadata = metadata[metadata['crop_id'] == crop_id_to_filter].reset_index(drop=True)

    filtered_indices = metadata['global_idx'].values
    global_index_filtered = [global_index[i] for i in filtered_indices]
    metadata['global_idx'] = np.arange(len(global_index_filtered)) # Reset index after filtering

    print(f"Total samples after filtering for '{args.crop_name}': {len(metadata)}")
    if len(metadata) == 0:
        print("No data to process. Exiting.")
        return

    print("\n--- 2. Data Splitting Verification ---")

    train_indices, val_indices, test_indices = [], [], []
    metadata['group'] = metadata['lon'].astype(str) + '_' + metadata['lat'].astype(str) # Group by location

    for group in metadata['group'].unique():
        group_df = metadata[metadata['group'] == group].sort_values('year')
        years = group_df['year'].unique()
        n_years = len(years)

        if n_years < 5:
            train_indices.extend(group_df['global_idx'].values)
            continue

        n_test = max(1, int(n_years * 0.1))
        n_val = max(1, int(n_years * 0.2))

        if n_years - n_test - n_val < 1: # Not enough for all splits
            train_indices.extend(group_df['global_idx'].values)
            continue

        test_years = years[-n_test:]
        val_years = years[-(n_test + n_val):-n_test]
        train_years = years[:-(n_test + n_val)]

        train_indices.extend(group_df[group_df['year'].isin(train_years)]['global_idx'].values)
        val_indices.extend(group_df[group_df['year'].isin(val_years)]['global_idx'].values)
        test_indices.extend(group_df[group_df['year'].isin(test_years)]['global_idx'].values)

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")

    # --- Open mmap files to get actual target values ---
    data_files = {region: {
        'targets': np.load(os.path.join(args.data_path, region, 'targets.npy'), mmap_mode='r'),
        'static': np.load(os.path.join(args.data_path, region, 'static_features.npy'), mmap_mode='r'),
        'dynamic': np.load(os.path.join(args.data_path, region, 'dynamic_features.npy'), mmap_mode='r')
    } for region in regions if os.path.isdir(os.path.join(args.data_path, region))}

    def get_yield_stats(indices):
        yield_values = []
        for g_idx in indices:
            region, local_idx = global_index_filtered[g_idx]
            yield_values.append(data_files[region]['targets'][local_idx])
        yield_values = np.array(yield_values)
        return {
            'mean': np.mean(yield_values),
            'std': np.std(yield_values),
            'min': np.min(yield_values),
            'max': np.max(yield_values)
        }

    print("\nYield statistics per dataset split:")
    print("Train Set:", get_yield_stats(train_indices))
    print("Val Set:  ", get_yield_stats(val_indices))
    print("Test Set: ", get_yield_stats(test_indices))

    print("\n--- 3. Scaler Calculation Verification ---")

    static_sum = np.zeros(66, dtype=np.float64)
    dynamic_sum = np.zeros(20, dtype=np.float64)
    static_sq_sum = np.zeros(66, dtype=np.float64)
    dynamic_sq_sum = np.zeros(20, dtype=np.float64)

    for g_idx in train_indices:
        region, local_idx = global_index_filtered[g_idx]
        static_features = data_files[region]['static'][local_idx]
        dynamic_features = data_files[region]['dynamic'][local_idx]

        static_sum += static_features
        static_sq_sum += np.square(static_features)
        dynamic_sum += dynamic_features.sum(axis=0)
        dynamic_sq_sum += np.square(dynamic_features).sum(axis=0)

    num_train_samples = len(train_indices)
    static_mean = static_sum / num_train_samples
    static_var = static_sq_sum / num_train_samples - np.square(static_mean)
    static_std = np.sqrt(np.maximum(static_var, 1e-8))

    # Simplified dynamic scaler for verification
    num_dynamic_points = num_train_samples * 365
    dynamic_mean = dynamic_sum / num_dynamic_points
    dynamic_var = dynamic_sq_sum / num_dynamic_points - np.square(dynamic_mean)
    dynamic_std = np.sqrt(np.maximum(dynamic_var, 1e-8))

    print("\nCalculated Scaler Statistics (from Train set):")
    print(f"Static Mean (first 5 features): {static_mean[:5]}")
    print(f"Static Std (first 5 features): {static_std[:5]}")
    print(f"Dynamic Mean (first 5 features): {dynamic_mean[:5]}")
    print(f"Dynamic Std (first 5 features): {dynamic_std[:5]}")

    print("\n--- 4. Single Sample Transformation ---")
    if val_indices:
        sample_g_idx = val_indices[0] # Pick the first validation sample
        region, local_idx = global_index_filtered[sample_g_idx]

        static_orig = data_files[region]['static'][local_idx]
        dynamic_orig = data_files[region]['dynamic'][local_idx]
        yield_orig = data_files[region]['targets'][local_idx]

        print(f"\nOriginal Validation Sample (Index: {sample_g_idx})")
        print(f"Yield: {yield_orig}")
        print(f"Original Static (first 5): {static_orig[:5]}")
        print(f"Original Dynamic (mean of first 5 features): {[np.mean(dynamic_orig[:,i]) for i in range(5)]}")

        # Manual Scaling
        static_scaled = (static_orig - static_mean) / static_std
        dynamic_scaled = (dynamic_orig - dynamic_mean) / dynamic_std

        print(f"\nScaled Validation Sample (Index: {sample_g_idx})")
        print(f"Scaled Static (first 5): {static_scaled[:5]}")
        print(f"Scaled Dynamic (mean of first 5 features): {[np.mean(dynamic_scaled[:,i]) for i in range(5)]}")
    else:
        print("\nNo validation samples to trace.")

if __name__ == '__main__':
    main()
