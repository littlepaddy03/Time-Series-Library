import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

# --- 静态特征的列索引 (基于 Step 1 的 ETL 脚本) ---
# [county_id, year, crop_id, lat, lon, ...soil...]
YEAR_COLUMN_INDEX = 1 # 'year' 在 static_features.npy 中的第 1 列 (0-indexed)

class ShardedYieldDataset(Dataset):
    """
    一个自定义的 PyTorch Dataset 类, 用于处理按区域分片的大型Numpy数据集。
    
    它会:
    1.  加载所有指定区域(分片)的 .npy 文件。
    2.  使用内存映射 (mmap_mode) 来避免RAM溢出。
    3.  根据 'year' 列将数据动态分割为 'train', 'val', 'test' 集。
    4.  创建一个全局索引, 映射到 (shard_index, local_index)。
    """
    def __init__(self, 
                 root_path: str, 
                 regions: List[str], 
                 mode: str = 'train',
                 val_years: List[int] = [2018, 2019],
                 test_years: List[int] = [2020, 2021, 2022]):
        """
        初始化 Dataset.
        """
        super().__init__()
        
        self.root_path = root_path
        self.regions = regions
        self.mode = mode
        
        self.dynamic_data = []
        self.static_data = []
        self.targets = []
        
        # self.index_map 是核心: (shard_index, local_index)
        self.index_map: List[Tuple[int, int]] = []

        print(f"[{self.mode} Mode] Loading regions: {self.regions}...")
        
        for shard_index, region_name in enumerate(self.regions):
            region_dir = os.path.join(self.root_path, region_name)
            
            if not os.path.isdir(region_dir):
                print(f"  -> 警告: 区域目录 {region_dir} 未找到, 跳过.")
                continue

            try:
                # 1. 使用 mmap_mode='r' 加载数据, 'r' = 只读
                dyn_path = os.path.join(region_dir, 'dynamic_features.npy')
                stat_path = os.path.join(region_dir, 'static_features.npy')
                tgt_path = os.path.join(region_dir, 'targets.npy')
                
                mmap_dyn = np.load(dyn_path, mmap_mode='r')
                mmap_stat = np.load(stat_path, mmap_mode='r')
                mmap_tgt = np.load(tgt_path, mmap_mode='r')
                
                # 2. 将 mmap 对象存入列表
                self.dynamic_data.append(mmap_dyn)
                self.static_data.append(mmap_stat)
                self.targets.append(mmap_tgt)
                
                # 3. 根据年份构建索引
                years = mmap_stat[:, YEAR_COLUMN_INDEX] 
                
                if self.mode == 'train':
                    mask = ~np.isin(years, val_years + test_years)
                elif self.mode == 'val':
                    mask = np.isin(years, val_years)
                elif self.mode == 'test':
                    mask = np.isin(years, test_years)
                
                local_indices = np.where(mask)[0]
                
                print(f"  -> 区域 '{region_name}' 加载成功. "
                      f"总样本: {len(years)}, "
                      f"'{self.mode}' 样本: {len(local_indices)}")

                # 4. 创建全局索引映射
                for local_idx in local_indices:
                    self.index_map.append((shard_index, local_idx))

            except Exception as e:
                print(f"  -> 错误: 加载区域 {region_name} 失败: {e}. 跳过.")

        self.total_length = len(self.index_map)
        print(f"[{self.mode} Mode] Dataset 初始化完毕. 总样本数 = {self.total_length}")


    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.total_length:
            raise IndexError(f"索引 {index} 超出范围 (0 to {self.total_length-1})")
            
        # 1. 映射全局索引
        shard_index, local_index = self.index_map[index]
        
        # 2. 从 mmap 对象中获取数据
        x_dynamic = self.dynamic_data[shard_index][local_index].astype(np.float32)
        x_static = self.static_data[shard_index][local_index].astype(np.float32)
        y_target = self.targets[shard_index][local_index].astype(np.float32)
        
        # 3. 转换为 PyTorch Tensors
        return (
            torch.tensor(x_dynamic, dtype=torch.float32),
            torch.tensor(x_static, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32)
        )

def data_provider_yield(args, flag):
    """
    TSLib 实验文件 (exp_yield.py) 将调用的主函数
    """
    
    # 从 args 中解析年份和区域
    # 我们假设 args.val_years = "2018,2019"
    # 我们假设 args.test_years = "2020,2021,2022"
    # 我们假设 args.regions = "ar,br,us,cn,in,eu"
    
    try:
        val_years = [int(y) for y in args.val_years.split(',')]
        test_years = [int(y) for y in args.test_years.split(',')]
        regions = args.regions.split(',')
    except Exception as e:
        print(f"错误: 解析 val_years, test_years, 或 regions 失败. 请检查参数。")
        raise e

    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        mode = 'train'
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        mode = 'val'
    elif flag == 'test':
        shuffle_flag = False
        drop_last = False
        mode = 'test'
    else:
        raise ValueError(f"无效的 flag: {flag}")

    dataset = ShardedYieldDataset(
        root_path=args.root_path,
        regions=regions,
        mode=mode,
        val_years=val_years,
        test_years=test_years
    )
    
    print(f"Data loader for {flag} created with {len(dataset)} samples.")
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        pin_memory=True, # 如果使用GPU, 建议开启
        drop_last=drop_last,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    return dataset, data_loader
