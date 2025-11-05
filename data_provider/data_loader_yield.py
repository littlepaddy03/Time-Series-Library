import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

class ShardedYieldDataset(Dataset):
    def __init__(self, data_path, regions, flag, scaler=None, indices=None, global_index=None, metadata=None):
        self.data_path = data_path
        self.regions = regions.split(',')
        self.flag = flag
        self.scaler = scaler

        if indices is None:
            self._scan_and_build_index()
        else:
            self.indices = indices
            self.global_index = global_index
            self.metadata = metadata
        
        self._open_mmap_files()
        print(f"Initialized dataset for flag: {self.flag}, Number of samples: {len(self.indices)}")

    def _open_mmap_files(self):
        self.data_files = {region: {
            'dynamic': np.load(os.path.join(self.data_path, region, 'dynamic_features.npy'), mmap_mode='r'),
            'static': np.load(os.path.join(self.data_path, region, 'static_features.npy'), mmap_mode='r'),
            'targets': np.load(os.path.join(self.data_path, region, 'targets.npy'), mmap_mode='r'),
        } for region in self.regions}

    def _scan_and_build_index(self):
        print("Building global index and splitting data...")
        self.global_index = []
        metadata_list = []

        for region in self.regions:
            region_path = os.path.join(self.data_path, region)
            static_features_mmap = np.load(os.path.join(region_path, 'static_features.npy'), mmap_mode='r')
            num_samples = static_features_mmap.shape[0]

            for i in range(num_samples):
                self.global_index.append((region, i))

            # Append metadata in a vectorized way
            metadata_list.append(pd.DataFrame(static_features_mmap[:, [0, 1, 2, 3]], columns=['lon', 'lat', 'year', 'crop_id']))

        self.metadata = pd.concat(metadata_list, ignore_index=True)
        self.metadata['global_idx'] = np.arange(len(self.global_index))
        self.metadata['region'] = [item[0] for item in self.global_index]

        # --- Data Splitting Logic (Chronological per group) ---
        train_indices, val_indices, test_indices = [], [], []
        self.metadata['group'] = self.metadata['region'] + '_' + self.metadata['crop_id'].astype(str)

        for group in self.metadata['group'].unique():
            group_df = self.metadata[self.metadata['group'] == group]
            years = np.sort(group_df['year'].unique())
            n_years = len(years)

            if n_years < 5:
                train_indices.extend(group_df['global_idx'].values)
                continue

            n_test = max(1, int(n_years * 0.1))
            n_val = max(1, int(n_years * 0.2))

            # Ensure train set is not empty
            if n_years - n_test - n_val < 1:
                train_indices.extend(group_df['global_idx'].values)
                continue

            test_years = years[-n_test:]
            val_years = years[-(n_test + n_val):-n_test]
            train_years = years[:-(n_test + n_val)]

            train_indices.extend(group_df[group_df['year'].isin(train_years)]['global_idx'].values)
            val_indices.extend(group_df[group_df['year'].isin(val_years)]['global_idx'].values)
            test_indices.extend(group_df[group_df['year'].isin(test_years)]['global_idx'].values)

        self.train_indices, self.val_indices, self.test_indices = train_indices, val_indices, test_indices

        if self.flag == 'train': self.indices = self.train_indices
        elif self.flag == 'val': self.indices = self.val_indices
        else: self.indices = self.test_indices

    @staticmethod
    def _calculate_scaler(train_dataset, data_files):
        print("Calculating scaler on training data...")

        static_sum = np.zeros(65, dtype=np.float64)
        static_sq_sum = np.zeros(65, dtype=np.float64)
        dynamic_sum = np.zeros(20, dtype=np.float64)
        dynamic_sq_sum = np.zeros(20, dtype=np.float64)
        non_zero_count = np.zeros(20, dtype=np.int64)

        chunk_size = 10000
        for i in range(0, len(train_dataset.indices), chunk_size):
            chunk_indices = train_dataset.indices[i:i+chunk_size]

            static_list, dynamic_list = [], []
            for g_idx in chunk_indices:
                region, local_idx = train_dataset.global_index[g_idx]
                static_list.append(data_files[region]['static'][local_idx])
                dynamic_list.append(data_files[region]['dynamic'][local_idx])

            static_chunk = np.array(static_list)
            dynamic_chunk = np.array(dynamic_list)

            static_sum += static_chunk.sum(axis=0)
            static_sq_sum += np.square(static_chunk).sum(axis=0)
            non_zero_mask = dynamic_chunk != 0
            dynamic_sum += dynamic_chunk.sum(axis=(0, 1))
            dynamic_sq_sum += np.square(dynamic_chunk).sum(axis=(0, 1))
            non_zero_count += non_zero_mask.sum(axis=(0, 1))

        num_train_samples = len(train_dataset.indices)
        static_mean = static_sum / num_train_samples
        static_var = static_sq_sum / num_train_samples - np.square(static_mean)
        static_std = np.sqrt(np.maximum(static_var, 1e-8))

        non_zero_count[non_zero_count == 0] = 1
        dynamic_mean = dynamic_sum / non_zero_count
        dynamic_var = dynamic_sq_sum / non_zero_count - np.square(dynamic_mean)
        dynamic_std = np.sqrt(np.maximum(dynamic_var, 1e-8))

        scaler_dict = {
            'dynamic_mean': torch.FloatTensor(dynamic_mean), 'dynamic_std': torch.FloatTensor(dynamic_std),
            'static_mean': torch.FloatTensor(static_mean), 'static_std': torch.FloatTensor(static_std),
        }
        return scaler_dict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        global_idx = self.indices[index]
        region, local_idx = self.global_index[global_idx]
        
        dynamic_features = self.data_files[region]['dynamic'][local_idx]
        static_features = self.data_files[region]['static'][local_idx]
        target = self.data_files[region]['targets'][local_idx]

        dynamic_features = torch.FloatTensor(dynamic_features)
        static_features = torch.FloatTensor(static_features)
        target = torch.FloatTensor(target)

        if self.scaler:
            dynamic_features = (dynamic_features - self.scaler['dynamic_mean']) / (self.scaler['dynamic_std'] + 1e-8)
            static_features = (static_features - self.scaler['static_mean']) / (self.scaler['static_std'] + 1e-8)

        return dynamic_features, static_features, target

def data_provider_yield(args, flag):
    # This provider function now returns all datasets and dataloaders at once when flag is 'train'
    if flag != 'train':
        # In this setup, validation and test sets are created alongside the training set.
        # This function should ideally be called only once with flag='train'.
        return None, None

    # 1. Create the initial training dataset to perform splitting
    initial_train_dataset = ShardedYieldDataset(
        data_path=args.data_path,
        regions=args.regions,
        flag='train'
    )

    # 2. Calculate scaler ONLY on the training set
    scaler = ShardedYieldDataset._calculate_scaler(initial_train_dataset, initial_train_dataset.data_files)
    initial_train_dataset.scaler = scaler

    # 3. Create validation and test datasets, passing the scaler and pre-computed indices
    val_dataset = ShardedYieldDataset(
        data_path=args.data_path, regions=args.regions, flag='val', scaler=scaler,
        indices=initial_train_dataset.val_indices,
        global_index=initial_train_dataset.global_index,
        metadata=initial_train_dataset.metadata
    )
    test_dataset = ShardedYieldDataset(
        data_path=args.data_path, regions=args.regions, flag='test', scaler=scaler,
        indices=initial_train_dataset.test_indices,
        global_index=initial_train_dataset.global_index,
        metadata=initial_train_dataset.metadata
    )
    
    # 4. Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        initial_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False
    )

    return initial_train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader
