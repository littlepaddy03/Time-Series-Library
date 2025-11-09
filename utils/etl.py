#!/usr/bin/env python3

"""
ETL (Extract, Transform, Load) 脚本

功能:
1.  读取多个 "长格式" 的区域作物CSV文件 (ar_maize.csv, us_soybean.csv, ...)。
2.  将每个唯一的样本 (按 lon, lat, year 分组) 转换为3个独立的Numpy数组：
    a. dynamic_features: (L, C) 形状, L=365, 包含所有时序数据 (天气, NDVI),
       非生长期数据用0填充。
    b. static_features: (M,) 形状, 包含所有静态数据 (土壤, 纬度, 经度, 作物ID等)。
    c. target: (1,) 形状, 包含该样本的最终产量。
3.  按区域 ('ar', 'br', 'us', ...) 分片, 将所有样本数据堆叠并保存为3个大型 .npy 文件。

最终输出结构:
/path/to/global_yield_dataset/
    ├── ar/
    │   ├── dynamic_features.npy  (N_ar, 365, 20)
    │   ├── static_features.npy   (N_ar, 65)
    │   └── targets.npy           (N_ar, 1)
    ├── br/
    │   └── ...
    └── ...
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
import rasterio

import argparse

# --- 常量定义 (CONFIGURATION) ---

def parse_args():
    parser = argparse.ArgumentParser(description='ETL Script for Crop Yield Data')
    parser.add_argument('--source_dir', type=str, default='./data/all_origin', help='Source directory containing CSV files')
    parser.add_argument('--target_dir', type=str, default='./data/tsl', help='Target directory to save processed .npy files')
    return parser.parse_args()

# 2. 序列长度
L_SEQUENCE_LENGTH = 365

# 3. 动态特征 (C) - 20个
C_DYNAMIC_COLS = [
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
# C = len(C_DYNAMIC_COLS)

# 4. 静态土壤特征 (M) - 61个
M_STATIC_SOIL_COLS = [
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

# 5. 上下文特征 (我们将把它们添加到M中)
# (longitude, latitude, year 在加载时获取)
# (crop_id 在文件映射中定义)
# 总静态特征 M = 61 (土壤) + 4 (上下文) = 65

# 6. 目标列 (Y)
Y_COL = 'yield'


SOURCE_FILE_MAP = {
    # Argentina
    'ar_maize.csv':   ('ar', 1.0),
    'ar_soybean.csv': ('ar', 3.0),
    'ar_rice.csv':    ('ar', 2.0),
    'ar_wheat.csv':   ('ar', 4.0),
    # Brazil
    'br_maize.csv':   ('br', 1.0),
    'br_soybean.csv': ('br', 3.0),
    'br_rice.csv':    ('br', 2.0),
    'br_wheat.csv':   ('br', 4.0),
    # China
    'cn_maize.csv':   ('cn', 1.0),
    'cn_soybean.csv': ('cn', 3.0),
    'cn_rice.csv':    ('cn', 2.0),
    'cn_wheat.csv':   ('cn', 4.0),
    # Europe
    'eu_maize.csv':   ('eu', 1.0),
    'eu_wheat.csv':   ('eu', 4.0),
    # India
    'in_maize.csv':   ('in', 1.0),
    'in_soybean.csv': ('in', 3.0),
    'in_rice.csv':    ('in', 2.0),
    'in_wheat.csv':   ('in', 4.0),
    # United States
    'us_maize.csv':   ('us', 1.0),
    'us_soybean.csv': ('us', 3.0),
    'us_rice.csv':    ('us', 2.0),
    'us_wheat.csv':   ('us', 4.0),
}

# --- 脚本主函数 ---
def main(args):
    """
    执行ETL过程的主函数
    """
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

    SOURCE_DATA_DIR = args.source_dir
    TARGET_DATA_DIR = args.target_dir

    # --- 1. 初始化数据收集器和气候数据 ---
    data_collectors = {}
    all_regions = sorted(list(set(val[0] for val in SOURCE_FILE_MAP.values())))

    for region in all_regions:
        data_collectors[region] = {'dynamic_list': [], 'static_list': [], 'target_list': []}
        os.makedirs(os.path.join(TARGET_DATA_DIR, region), exist_ok=True)

    # 加载 Köppen-Geiger TIF 文件
    KG_TIF_PATH = 'koppen_geiger_tif/koppen_geiger_0p00833333.tif'
    kg_dataset = None
    if os.path.exists(KG_TIF_PATH):
        try:
            kg_dataset = rasterio.open(KG_TIF_PATH)
            print(f"成功加载 Köppen-Geiger TIF 文件: {KG_TIF_PATH}")
        except Exception as e:
            print(f"警告: 无法加载 Köppen-Geiger TIF 文件. 错误: {e}. 将不会添加气候分区特征.")
            kg_dataset = None
    else:
        print(f"警告: 未找到 Köppen-Geiger TIF 文件 at '{KG_TIF_PATH}'. 将不会添加气候分区特征.")

    print(f"ETL开始. 将处理 {len(SOURCE_FILE_MAP)} 个文件, 存入 {len(all_regions)} 个区域目录.")

    # --- 2. 遍历并处理所有源文件 ---
    for source_file, (region_name, crop_id) in tqdm(SOURCE_FILE_MAP.items(), desc="Processing Source Files"):

        file_path = os.path.join(SOURCE_DATA_DIR, source_file)
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 未找到, 跳过.")
            continue

        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
        except Exception as e:
            print(f"错误: 无法读取 {file_path}. 错误信息: {e}. 跳过.")
            continue

        # 定义样本的唯一键
        # (我们使用 year, lon, lat 来分组。'year' 列已在CSV中提供, 代表"生长季")
        try:
            sample_groups = df.groupby(['longitude', 'latitude', 'year'])
        except KeyError as e:
            print(f"错误: 文件 {file_path} 缺少关键列: {e}. 跳过.")
            continue

        # 迭代处理该文件中的每一个独立样本
        for (lon, lat, year), sample_df in tqdm(sample_groups, desc=f"  -> {source_file}", leave=False):

            # --- a. 创建动态特征数组 (L, C) ---
            # (L, C) 的全零数组
            dynamic_sample_array = np.zeros((L_SEQUENCE_LENGTH, len(C_DYNAMIC_COLS)), dtype=np.float32)

            # 获取该样本的动态数据
            try:
                growing_season_data = sample_df[C_DYNAMIC_COLS].values
            except KeyError as e:
                print(f"错误: {source_file} 缺少动态列 {e}. 跳过此样本.")
                continue # 跳过这个损坏的样本

            # 获取对应的日历天 (1-365)
            day_of_year = sample_df['date'].dt.dayofyear.values

            # 检查越界 (如果数据日期错误)
            if (day_of_year > 365).any() or (day_of_year < 1).any():
                print(f"警告: {source_file} 样本 (lon:{lon}, lat:{lat}, year:{year}) 包含无效日期. 跳过此样本.")
                continue

            # 将生长期数据填充到 365 天数组的正确位置 (doy-1 转换为 0-indexed)
            dynamic_sample_array[day_of_year - 1] = growing_season_data

            # --- b. 创建静态特征数组 (M,) ---
            # 从该样本的第一行获取所有静态数据
            first_row = sample_df.iloc[0]

            # 1. 上下文特征 (4个)
            context_features = [lon, lat, year, crop_id]

            # 2. 土壤特征 (61个)
            try:
                soil_features = first_row[M_STATIC_SOIL_COLS].values.tolist()
            except KeyError as e:
                print(f"错误: {source_file} 缺少静态土壤列 {e}. 跳过此样本.")
                continue # 跳过这个损坏的样本

            # 组合成一个静态特征向量 (总计 4 + 61 = 65 个特征)
            static_sample_list = context_features + soil_features

            # 3. 添加 Köppen-Geiger 气候分区特征 (如果可用)
            if kg_dataset:
                try:
                    # 使用 rasterio.sample 来查询. 它需要一个坐标列表.
                    coord_iter = [(lon, lat)]
                    # sample 方法返回一个生成器, 我们获取第一个结果
                    kg_zone = next(kg_dataset.sample(coord_iter))[0]
                    static_sample_list.append(kg_zone)
                except Exception as e:
                    # 如果查询失败 (例如, 坐标超出范围), 添加一个占位符 (例如, 0)
                    static_sample_list.append(0)
                    print(f"警告: 无法为坐标 ({lon}, {lat}) 查询气候分区. 错误: {e}. 使用 0 代替.")

            static_sample_array = np.array(static_sample_list, dtype=np.float32)

            # --- c. 创建目标数组 (1,) ---
            try:
                target_sample_array = np.array([first_row[Y_COL]], dtype=np.float32)
            except KeyError as e:
                print(f"错误: {source_file} 缺少目标列 {e}. 跳过此样本.")
                continue # 跳过这个损坏的样本

            # --- d. 将处理好的数组添加到收集器 ---
            data_collectors[region_name]['dynamic_list'].append(dynamic_sample_array)
            data_collectors[region_name]['static_list'].append(static_sample_array)
            data_collectors[region_name]['target_list'].append(target_sample_array)

    print("\n" + "="*30)
    print("所有CSV文件处理完毕.")
    print("开始将数据保存到 .npy 文件...")
    print("="*30)

    # --- 3. 保存所有区域分片 ---
    total_samples = 0
    for region_name, data in data_collectors.items():

        N_r = len(data['target_list']) # 该区域的样本总数
        if N_r == 0:
            print(f"区域 {region_name}: 0 个样本, 跳过.")
            continue

        print(f"正在保存区域: {region_name} (共 {N_r} 个样本)")

        target_region_path = os.path.join(TARGET_DATA_DIR, region_name)

        # 将列表转换为大型Numpy数组
        try:
            dynamic_arr = np.array(data['dynamic_list'], dtype=np.float32)
            static_arr = np.array(data['static_list'], dtype=np.float32)
            target_arr = np.array(data['target_list'], dtype=np.float32)
        except Exception as e:
            print(f"错误: 转换区域 {region_name} 的数据为Numpy数组时失败. {e}")
            print(f"Dynamic shape: {len(data['dynamic_list'])}, Static shape: {len(data['static_list'])}, Target shape: {len(data['target_list'])}")
            continue

        # 校验形状
        M_expected = len(M_STATIC_SOIL_COLS) + 4  # 61 (土壤) + 4 (上下文) = 65
        if kg_dataset:
            M_expected += 1 # 如果气候数据可用, 则为66
        C_expected = len(C_DYNAMIC_COLS) # 20

        if dynamic_arr.shape != (N_r, L_SEQUENCE_LENGTH, C_expected):
            print(f"  -> 错误: Dynamic array 形状不匹配! 预期: {(N_r, L_SEQUENCE_LENGTH, C_expected)}, 得到: {dynamic_arr.shape}")
        if static_arr.shape[0] != N_r or (static_arr.shape[1] != M_expected and static_arr.shape[1] != M_expected -1):
             print(f"  -> 错误: Static array 形状不匹配! 预期样本数: {N_r}, 预期特征数: {M_expected}, 得到: {static_arr.shape}")
        if target_arr.shape != (N_r, 1):
            print(f"  -> 错误: Target array 形状不匹配! 预期: {(N_r, 1)}, 得到: {target_arr.shape}")

        # 保存 .npy 文件
        np.save(os.path.join(target_region_path, 'dynamic_features.npy'), dynamic_arr)
        np.save(os.path.join(target_region_path, 'static_features.npy'), static_arr)
        np.save(os.path.join(target_region_path, 'targets.npy'), target_arr)

        print(f"  -> {region_name} 保存成功.")
        total_samples += N_r

    print("\n" + "="*30)
    print(f"ETL 过程完毕.")
    print(f"总计 {total_samples} 个样本被处理并保存到 {len(all_regions)} 个区域目录中.")
    print(f"输出目录: {TARGET_DATA_DIR}")
    print("="*30)

if __name__ == "__main__":
    args = parse_args()
    main(args)
