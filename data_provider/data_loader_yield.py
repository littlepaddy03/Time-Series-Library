import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

class ShardedYieldDataset(Dataset):
    def __init__(self, data_path, regions, flag='train', test_ratio=0.1, val_ratio=0.2):
        self.data_path = data_path
        self.regions = regions.split(',')
        self.flag = flag
        
        self.scaler = None
        self._scan_and_build_index()
        self._calculate_scaler()

    def _scan_and_build_index(self):
        print("Building global index...")
        self.global_index = []
        self.metadata = []

        for region in self.regions:
            region_path = os.path.join(self.data_path, region)
            static_features = np.load(os.path.join(region_path, 'static_features.npy'), mmap_mode='r')

            num_samples = static_features.shape[0]
            for i in range(num_samples):
                # Store (region, local_index)
                self.global_index.append((region, i))
                # Store metadata for splitting: (lon, lat, year, crop_id)
                self.metadata.append(static_features[i, [0, 1, 2, 3]])

        self.metadata = pd.DataFrame(self.metadata, columns=['lon', 'lat', 'year', 'crop_id'])
        self.metadata['global_idx'] = np.arange(len(self.global_index))

        # --- Data Splitting Logic ---
        train_indices, val_indices, test_indices = [], [], []

        # Group by crop and region (implicitly handled by crop_id uniqueness for now)
        for crop_id in self.metadata['crop_id'].unique():
            crop_df = self.metadata[self.metadata['crop_id'] == crop_id]

            # Split by years for this crop
            years = crop_df['year'].unique()

            # First, split off a test set
            train_val_years, test_years = train_test_split(years, test_size=0.1, random_state=42)

            # Then, split the remainder into train and validation
            train_years, val_years = train_test_split(train_val_years, test_size=0.22, random_state=42) # 0.22 * 0.9 = ~0.2

            # Collect indices based on year splits
            train_indices.extend(crop_df[crop_df['year'].isin(train_years)]['global_idx'].values)
            val_indices.extend(crop_df[crop_df['year'].isin(val_years)]['global_idx'].values)
            test_indices.extend(crop_df[crop_df['year'].isin(test_years)]['global_idx'].values)
        
        if self.flag == 'train':
            self.indices = train_indices
        elif self.flag == 'val':
            self.indices = val_indices
        else:
            self.indices = test_indices
        
        print(f"Flag: {self.flag}, Number of samples: {len(self.indices)}")


    def _calculate_scaler(self):
        print("Calculating scaler on training data...")
        # Use only training indices to calculate mean and std
        train_idxs = self.indices if self.flag == 'train' else \
                     [i for i, meta in self.metadata.iterrows() if meta['global_idx'] in self.indices]


        # More memory-efficient calculation of scaler
        # Process in chunks to avoid loading all data into memory

        # Static features
        static_sum = np.zeros(65, dtype=np.float64)
        static_sq_sum = np.zeros(65, dtype=np.float64)
        
        # Dynamic features
        dynamic_sum = np.zeros(20, dtype=np.float64)
        dynamic_sq_sum = np.zeros(20, dtype=np.float64)
        non_zero_count = np.zeros(20, dtype=np.int64)

        temp_data_cache = {region: {
            'dynamic': np.load(os.path.join(self.data_path, region, 'dynamic_features.npy'), mmap_mode='r'),
            'static': np.load(os.path.join(self.data_path, region, 'static_features.npy'), mmap_mode='r')
        } for region in self.regions}

        chunk_size = 10000
        for i in range(0, len(train_idxs), chunk_size):
            chunk_indices = train_idxs[i:i+chunk_size]
            
            static_chunk = np.array([temp_data_cache[self.global_index[g_idx][0]]['static'][self.global_index[g_idx][1]] for g_idx in chunk_indices])
            dynamic_chunk = np.array([temp_data_cache[self.global_index[g_idx][0]]['dynamic'][self.global_index[g_idx][1]] for g_idx in chunk_indices])

            static_sum += static_chunk.sum(axis=0)
            static_sq_sum += np.square(static_chunk).sum(axis=0)

            non_zero_mask = dynamic_chunk != 0
            dynamic_sum += dynamic_chunk.sum(axis=(0, 1))
            dynamic_sq_sum += np.square(dynamic_chunk).sum(axis=(0, 1))
            non_zero_count += non_zero_mask.sum(axis=(0, 1))

        num_train_samples = len(train_idxs)
        static_mean = static_sum / num_train_samples
        static_std = np.sqrt(static_sq_sum / num_train_samples - np.square(static_mean))

        dynamic_mean = dynamic_sum / non_zero_count
        dynamic_std = np.sqrt(dynamic_sq_sum / non_zero_count - np.square(dynamic_mean))

        self.scaler = {
            'dynamic_mean': torch.FloatTensor(dynamic_mean),
            'dynamic_std': torch.FloatTensor(dynamic_std),
            'static_mean': torch.FloatTensor(static_mean),
            'static_std': torch.FloatTensor(static_std),
        }
        print("Scaler calculation finished.")


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        global_idx = self.indices[index]
        region, local_idx = self.global_index[global_idx]

        region_path = os.path.join(self.data_path, region)
        
        # Load data for the specific item
        dynamic_features = np.load(os.path.join(region_path, 'dynamic_features.npy'), mmap_mode='r')[local_idx]
        static_features = np.load(os.path.join(region_path, 'static_features.npy'), mmap_mode='r')[local_idx]
        target = np.load(os.path.join(region_path, 'targets.npy'), mmap_mode='r')[local_idx]

        # Convert to tensor
        dynamic_features = torch.FloatTensor(dynamic_features)
        static_features = torch.FloatTensor(static_features)
        target = torch.FloatTensor(target)

        # Apply normalization
        if self.scaler:
            dynamic_features = (dynamic_features - self.scaler['dynamic_mean']) / (self.scaler['dynamic_std'] + 1e-8)
            static_features = (static_features - self.scaler['static_mean']) / (self.scaler['static_std'] + 1e-8)

        return dynamic_features, static_features, target

def data_provider_yield(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else: # train or val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    dataset = ShardedYieldDataset(
        data_path=args.data_path,
        regions=args.regions,
        flag=flag
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return dataset, data_loader
