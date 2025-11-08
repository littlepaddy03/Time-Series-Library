import argparse
import numpy as np
import sys
import os

# Add project root to sys.path to allow imports from other modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_provider.data_loader_yield import ShardedYieldDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Data Split Verification Script')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the processed data directory (e.g., dataset/global_yield_dataset/)')
    parser.add_argument('--regions', type=str, default='us,ar,br,cn,in,eu', help='Comma-separated list of regions to check')
    return parser.parse_args()

def verify_split(args):
    print("--- Starting Data Split Verification ---")

    # 1. Load the dataset to get all indices and metadata
    try:
        initial_dataset = ShardedYieldDataset(
            data_path=args.data_path,
            regions=args.regions,
            flag='train' # This flag initializes all splits
        )
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset. Please check your --data_path.")
        print(f"  Details: {e}")
        return

    train_indices = set(initial_dataset.train_indices)
    val_indices = set(initial_dataset.val_indices)
    test_indices = set(initial_dataset.test_indices)
    all_indices = set(range(len(initial_dataset.global_index)))

    metadata = initial_dataset.metadata

    # --- 1.1. Integrity Check ---
    print("\n[1.1] Running Integrity Check...")

    # Check for empty sets
    if not train_indices:
        print("❌ FAILED: Training set is empty.")
        return

    # Check for overlap
    train_val_overlap = train_indices.intersection(val_indices)
    train_test_overlap = train_indices.intersection(test_indices)
    val_test_overlap = val_indices.intersection(test_indices)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("❌ FAILED: Overlap detected between dataset splits.")
        print(f"  Train-Val Overlap: {len(train_val_overlap)} samples")
        print(f"  Train-Test Overlap: {len(train_test_overlap)} samples")
        print(f"  Val-Test Overlap: {len(val_test_overlap)} samples")
        return
    else:
        print("  ✅ No overlap between train, val, and test sets.")

    # Check for completeness
    union_indices = train_indices.union(val_indices).union(test_indices)
    if union_indices != all_indices:
        print("❌ FAILED: The union of splits does not equal the full dataset.")
        print(f"  Missing samples: {len(all_indices - union_indices)}")
        print(f"  Extra samples in splits: {len(union_indices - all_indices)}")
        return
    else:
        print("  ✅ Union of splits correctly forms the full dataset.")

    print("--- Integrity Check PASSED ---")

    # --- 1.2, 1.3, 1.4: Group-level Checks ---
    print("\n[1.2, 1.3, 1.4] Running Group-level Checks...")

    all_groups = metadata['group'].unique()
    failures = []

    for group in all_groups:
        group_df = metadata[metadata['group'] == group]

        group_train_indices = set(group_df[group_df['global_idx'].isin(train_indices)]['global_idx'])
        group_val_indices = set(group_df[group_df['global_idx'].isin(val_indices)]['global_idx'])
        group_test_indices = set(group_df[group_df['global_idx'].isin(test_indices)]['global_idx'])

        group_years = np.sort(group_df['year'].unique())

        # 1.4. Small data group check
        if len(group_years) < 5:
            if group_val_indices or group_test_indices:
                failures.append(f"Group '{group}': FAILED small data check. Has < 5 years but found in val/test sets.")
            continue # Skip further checks for this group

        # 1.3. Chronological Correctness Check
        if not group_train_indices or (not group_val_indices and not group_test_indices):
            continue # Can't check order if a set is empty (e.g., test set might be empty for some groups)

        train_years = group_df[group_df['global_idx'].isin(group_train_indices)]['year']
        max_train_year = train_years.max()

        if group_val_indices:
            val_years = group_df[group_df['global_idx'].isin(group_val_indices)]['year']
            min_val_year = val_years.min()
            if max_train_year >= min_val_year:
                failures.append(f"Group '{group}': FAILED chronological check. max_train_year ({max_train_year}) >= min_val_year ({min_val_year}).")

            if group_test_indices:
                max_val_year = val_years.max()
                test_years = group_df[group_df['global_idx'].isin(group_test_indices)]['year']
                min_test_year = test_years.min()
                if max_val_year >= min_test_year:
                    failures.append(f"Group '{group}': FAILED chronological check. max_val_year ({max_val_year}) >= min_test_year ({min_test_year}).")

    if failures:
        print("❌ FAILED: One or more groups failed checks.")
        for f in failures:
            print(f"  - {f}")
    else:
        print("  ✅ All groups passed chronological and small data checks.")
        print("--- Group-level Checks PASSED ---")

    print("\n🎉 --- All Data Split Verification Checks PASSED! --- 🎉")

    # --- Print Sample Example ---
    print("\n[INFO] All checks passed. Generating a detailed sample report in `data_verification_sample.log`...")

    # Get the first sample from the training set
    sample_index = 0
    dynamic_features, static_features, target = initial_dataset[sample_index]

    # Get metadata for this sample
    global_idx = initial_dataset.indices[sample_index]
    sample_metadata = metadata[metadata['global_idx'] == global_idx].iloc[0]

    # Dynamic feature analysis
    # Note: We are analyzing the NORMALIZED data here.
    active_days_mask = dynamic_features.abs().sum(dim=1) > 0
    num_active_days = active_days_mask.sum().item()
    active_dynamic_features = dynamic_features[active_days_mask]

    # Key dynamic feature columns (indices)
    key_features = {
        'NDVI': 0,
        'Temperature_Air_2m_Mean_24h': 4,
        'Precipitation_Flux': 10,
    }

    dynamic_stats = {}
    for name, idx in key_features.items():
        feature_slice = active_dynamic_features[:, idx]
        dynamic_stats[name] = {
            'mean': feature_slice.mean().item(),
            'std': feature_slice.std().item(),
            'min': feature_slice.min().item(),
            'max': feature_slice.max().item(),
        }

    with open("data_verification_sample.log", "w") as f:
        f.write("--- Detailed Sample Analysis Report ---\n\n")
        f.write("This report provides a complete profile of the first sample from the training set.\n")
        f.write("Note: All feature values shown below are NORMALIZED.\n\n")

        f.write("== 1. Sample Metadata ==\n")
        f.write(f"  - Global Index: {global_idx}\n")
        f.write(f"  - Region: {sample_metadata['region']}\n")
        f.write(f"  - Crop ID: {sample_metadata['crop_id']}\n")
        f.write(f"  - Year: {sample_metadata['year']}\n")
        f.write(f"  - Longitude: {sample_metadata['lon']}\n")
        f.write(f"  - Latitude: {sample_metadata['lat']}\n\n")

        f.write("== 2. Static Features ==\n")
        f.write(f"  - Shape: {static_features.shape}\n")
        f.write(f"  - Full Vector (all 65 features):\n")
        f.write(np.array2string(static_features.numpy(), precision=4, separator=', ') + "\n\n")

        f.write("== 3. Dynamic Features Summary ==\n")
        f.write(f"  - Shape: {dynamic_features.shape}\n")
        f.write(f"  - Number of Active (non-zero) Days: {num_active_days}\n")
        f.write("  - Statistics for key features (over active days):\n")
        for name, stats in dynamic_stats.items():
            f.write(f"    - {name}:\n")
            f.write(f"        Mean: {stats['mean']:.4f}\n")
            f.write(f"        Std:  {stats['std']:.4f}\n")
            f.write(f"        Min:  {stats['min']:.4f}\n")
            f.write(f"        Max:  {stats['max']:.4f}\n")
        f.write("\n")

        f.write("== 4. Target Value ==\n")
        f.write(f"  - Shape: {target.shape}\n")
        f.write(f"  - Yield Value: {target.item()}\n")

    print("  ✅ Detailed sample report written to `data_verification_sample.log`.")


if __name__ == "__main__":
    args = parse_args()
    verify_split(args)
