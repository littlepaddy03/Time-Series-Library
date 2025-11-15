import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch

def custom_collate_fn(batch):
    """
    Collate function to handle variable length sequences and create attention masks.
    """
    # Separate the components of the batch
    dynamic_features_list = [item[0] for item in batch]
    static_features_list = [item[1] for item in batch]
    targets_list = [item[2] for item in batch]
    unnormalized_statics_list = [item[3] for item in batch]

    # Determine the maximum sequence length in the batch
    max_len = max(seq.shape[0] for seq in dynamic_features_list)

    # Pad dynamic features and create attention masks
    padded_dynamics = torch.zeros(len(batch), max_len, dynamic_features_list[0].shape[1])
    attention_masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, seq in enumerate(dynamic_features_list):
        length = seq.shape[0]
        padded_dynamics[i, :length, :] = seq
        attention_masks[i, :length] = True

    # Stack static features and targets
    statics = torch.stack(static_features_list)
    targets = torch.stack(targets_list)
    unnormalized_statics = torch.stack(unnormalized_statics_list)

    return padded_dynamics, statics, targets, unnormalized_statics, attention_masks

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
            'dynamic': np.load(os.path.join(self.data_path, region, 'dynamic_features.npy'), allow_pickle=True), # mmap_mode removed for object array
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
    def _calculate_scaler(train_dataset, data_files, args):
        print("--- Starting multi-strategy scaler calculation ---")
        dynamic_feat_dim = args.enc_in

        # Define indices for different feature types
        lon_lat_year_indices = [0, 1, 2]
        crop_id_index = 3
        soil_indices = list(range(4, 65))
        climate_index = 65

        # --- Initialize accumulators and data collectors ---
        soil_sum = np.zeros(len(soil_indices), dtype=np.float64)
        soil_sq_sum = np.zeros(len(soil_indices), dtype=np.float64)
        dynamic_sum = np.zeros(dynamic_feat_dim, dtype=np.float64)
        dynamic_sq_sum = np.zeros(dynamic_feat_dim, dtype=np.float64)
        non_zero_count = np.zeros(dynamic_feat_dim, dtype=np.int64)

        # Collect all lon/lat/year features to fit the MinMaxScaler
        lon_lat_year_data = np.zeros((len(train_dataset.indices), len(lon_lat_year_indices)), dtype=np.float64)

        print(f"Total training samples to process for scaler: {len(train_dataset.indices)}")
        for i, g_idx in enumerate(train_dataset.indices):
            region, local_idx = train_dataset.global_index[g_idx]
            static_sample = data_files[region]['static'][local_idx]
            dynamic_sample = data_files[region]['dynamic'][local_idx]

            # Accumulate for StandardScaler (Soil)
            soil_features = static_sample[soil_indices]
            soil_sum += soil_features
            soil_sq_sum += np.square(soil_features)

            # Collect for MinMaxScaler (Lon, Lat, Year)
            lon_lat_year_data[i] = static_sample[lon_lat_year_indices]

            # Accumulate for StandardScaler (Dynamic)
            non_zero_mask = dynamic_sample != 0
            dynamic_sum += dynamic_sample.sum(axis=0)
            dynamic_sq_sum += np.square(dynamic_sample).sum(axis=0)
            non_zero_count += non_zero_mask.sum(axis=0)

        print("--- Finished data accumulation. Calculating scalers. ---")
        num_train_samples = len(train_dataset.indices)

        # --- Calculate StandardScaler for Soil Features ---
        soil_mean = soil_sum / num_train_samples
        soil_var = soil_sq_sum / num_train_samples - np.square(soil_mean)
        soil_std = np.sqrt(np.maximum(soil_var, 1e-8))

        # --- Fit MinMaxScaler for Lon, Lat, Year ---
        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        min_max_scaler.fit(lon_lat_year_data)

        print(f"  [DEBUG] Soil mean (first 3): {soil_mean[:3]}")
        print(f"  [DEBUG] Soil std (first 3): {soil_std[:3]}")
        print(f"  [DEBUG] MinMaxScaler min_ (lon,lat,year): {min_max_scaler.min_}")
        print(f"  [DEBUG] MinMaxScaler scale_ (lon,lat,year): {min_max_scaler.scale_}")

        # --- Calculate StandardScaler for Dynamic Features ---
        non_zero_count[non_zero_count == 0] = 1
        dynamic_mean = dynamic_sum / non_zero_count
        dynamic_var = dynamic_sq_sum / non_zero_count - np.square(dynamic_mean)
        dynamic_std = np.sqrt(np.maximum(dynamic_var, 1e-8))

        scaler_dict = {
            'dynamic_mean': torch.FloatTensor(dynamic_mean), 'dynamic_std': torch.FloatTensor(dynamic_std),
            'soil_mean': torch.FloatTensor(soil_mean), 'soil_std': torch.FloatTensor(soil_std),
            'min_max_scaler': min_max_scaler
        }
        return scaler_dict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        global_idx = self.indices[index]
        region, local_idx = self.global_index[global_idx]
        
        dynamic_features = self.data_files[region]['dynamic'][local_idx]
        static_features_orig = self.data_files[region]['static'][local_idx]
        target = self.data_files[region]['targets'][local_idx]

        # Keep a copy of the original static features for analysis
        unnormalized_static_features = torch.FloatTensor(static_features_orig.copy())

        # --- Apply Dynamic Feature Scaling ---
        dynamic_features_tensor = torch.FloatTensor(dynamic_features)
        if self.scaler and 'dynamic_mean' in self.scaler:
             dynamic_features_tensor = (dynamic_features_tensor - self.scaler['dynamic_mean']) / (self.scaler['dynamic_std'] + 1e-8)

        # --- Apply Multi-Strategy Static Feature Scaling ---
        if self.scaler:
            # 1. Separate features
            lon_lat_year = static_features_orig[[0, 1, 2]]
            crop_id = static_features_orig[[3]]
            soil = static_features_orig[4:65]
            climate = static_features_orig[[65]]

            # 2. Apply transformations
            # Min-Max scale lon, lat, year
            lon_lat_year_scaled = self.scaler['min_max_scaler'].transform(lon_lat_year.reshape(1, -1)).flatten()

            # Standard scale soil features
            soil_scaled = (soil - self.scaler['soil_mean'].numpy()) / (self.scaler['soil_std'].numpy() + 1e-8)

            # 3. Recombine into a single tensor
            static_features_tensor = torch.cat([
                torch.FloatTensor(lon_lat_year_scaled),
                torch.FloatTensor(crop_id),
                torch.FloatTensor(soil_scaled),
                torch.FloatTensor(climate)
            ], dim=0)
        else:
            # If no scaler, just convert to tensor
            static_features_tensor = torch.FloatTensor(static_features_orig)

        target_tensor = torch.FloatTensor(target)

        return dynamic_features_tensor, static_features_tensor, target_tensor, unnormalized_static_features

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
    scaler = ShardedYieldDataset._calculate_scaler(initial_train_dataset, initial_train_dataset.data_files, args)
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
        num_workers=args.num_workers, drop_last=True, collate_fn=custom_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, collate_fn=custom_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, collate_fn=custom_collate_fn
    )

    return initial_train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader
