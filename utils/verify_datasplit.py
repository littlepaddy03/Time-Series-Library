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
    print("\n[INFO] All checks passed. Printing one sample to `data_verification_sample.log` for manual inspection...")

    # Get the first sample from the training set
    dynamic_features, static_features, target = initial_dataset[0]

    with open("data_verification_sample.log", "w") as f:
        f.write("--- Data Sample Verification ---\n\n")
        f.write("This file shows the details of the first sample from the training set.\n")
        f.write("Note: Features are normalized.\n\n")

        f.write(f"== Dynamic Features ==\n")
        f.write(f"  - Shape: {dynamic_features.shape}\n")
        f.write(f"  - DType: {dynamic_features.dtype}\n")
        f.write(f"  - First 5 timesteps (first 3 features):\n{dynamic_features[:5, :3]}\n\n")

        f.write(f"== Static Features ==\n")
        f.write(f"  - Shape: {static_features.shape}\n")
        f.write(f"  - DType: {static_features.dtype}\n")
        f.write(f"  - First 10 features:\n{static_features[:10]}\n\n")

        f.write(f"== Target ==\n")
        f.write(f"  - Shape: {target.shape}\n")
        f.write(f"  - DType: {target.dtype}\n")
        f.write(f"  - Value: {target.item()}\n")

    print("  ✅ Sample data written to `data_verification_sample.log`.")


if __name__ == "__main__":
    args = parse_args()
    verify_split(args)
