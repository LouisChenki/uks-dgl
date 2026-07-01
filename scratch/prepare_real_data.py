# -*- coding: utf-8 -*-
"""
真实数据集 Meuse 和 California Temperature 预处理与划分脚本
将原始 CSV 清洗并划分为 70% 训练集和 30% 测试集，归一化空间坐标和协变量，
并将结果保存至 data/real/meuse_processed.npz 和 data/real/california_processed.npz。
"""

import os
import pandas as pd
import numpy as np

def prepare_meuse():
    csv_path = "data/real/meuse.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. 坐标归一化
    x = df['x'].values
    y = df['y'].values
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    coords = np.stack([
        (x - x_min) / (x_max - x_min),
        (y - y_min) / (y_max - y_min)
    ], axis=-1)  # [N, 2]
    
    # 2. 外部协变量归一化
    dist = df['dist'].values
    elev = df['elev'].values
    dist_norm = (dist - dist.min()) / (dist.max() - dist.min())
    elev_norm = (elev - elev.min()) / (elev.max() - elev.min())
    covariates = np.stack([dist_norm, elev_norm], axis=-1)  # [N, 2]
    
    # 3. 提取目标变量（双通道：主通道 zinc，辅助通道 cadmium）
    zinc = df['zinc'].values
    cadmium = df['cadmium'].values
    Z = np.stack([zinc, cadmium], axis=-1)  # [N, 2]
    
    # 4. 随机划分 Train/Test (70% vs 30%, seed=42)
    np.random.seed(42)
    N = len(df)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    split_idx = int(0.7 * N)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    # 5. 计算训练集的均值和标准差用于标准化（地统计学中均值回归与体积流拉正要求标准化）
    Z_train_raw = Z[train_idx]
    mean_Z = np.mean(Z_train_raw, axis=0)
    std_Z = np.std(Z_train_raw, axis=0)
    std_Z = np.where(std_Z == 0, 1.0, std_Z)
    
    # 6. 保存 processed 数据
    npz_path = "data/real/meuse_processed.npz"
    np.savez(
        npz_path,
        coords_train=coords[train_idx],
        Z_train=Z[train_idx],
        cov_train=covariates[train_idx],
        coords_test=coords[test_idx],
        Z_test=Z[test_idx],
        cov_test=covariates[test_idx],
        mean_Z=mean_Z,
        std_Z=std_Z,
        x_min_max=np.array([x_min, x_max]),
        y_min_max=np.array([y_min, y_max]),
        raw_x=x,
        raw_y=y,
        raw_zinc=zinc,
        raw_cadmium=cadmium
    )
    print(f"--> [Meuse 预处理完成] 样本总数: {N}, 训练集: {len(train_idx)}, 测试集: {len(test_idx)}")
    print(f"    zinc 训练集原始均值: {mean_Z[0]:.2f}, 标准差: {std_Z[0]:.2f}")

def prepare_california():
    csv_path = "data/real/california_temperature.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # 1. 经纬度归一化 (将 longitude, latitude 映射到 [0, 1])
    lon = df['longitude'].values
    lat = df['latitude'].values
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    
    coords = np.stack([
        (lon - lon_min) / (lon_max - lon_min),
        (lat - lat_min) / (lat_max - lat_min)
    ], axis=-1)  # [N, 2]
    
    # 2. 海拔高度外部协变量归一化
    elev = df['elevation'].values
    elev_min, elev_max = elev.min(), elev.max()
    elev_norm = (elev - elev_min) / (elev_max - elev_min)
    covariates = elev_norm[:, np.newaxis]  # [N, 1]
    
    # 3. 提取主目标变量（单通道：temp 温度）
    temp = df['temp'].values
    Z = temp[:, np.newaxis]  # [N, 1]
    
    # 4. 随机划分 Train/Test (70% vs 30%, seed=42)
    np.random.seed(42)
    N = len(df)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    split_idx = int(0.7 * N)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    # 5. 标准化参数
    Z_train_raw = Z[train_idx]
    mean_Z = np.mean(Z_train_raw, axis=0)
    std_Z = np.std(Z_train_raw, axis=0)
    std_Z = np.where(std_Z == 0, 1.0, std_Z)
    
    # 6. 保存 processed 数据
    npz_path = "data/real/california_processed.npz"
    np.savez(
        npz_path,
        coords_train=coords[train_idx],
        Z_train=Z[train_idx],
        cov_train=covariates[train_idx],
        coords_test=coords[test_idx],
        Z_test=Z[test_idx],
        cov_test=covariates[test_idx],
        mean_Z=mean_Z,
        std_Z=std_Z,
        lon_min_max=np.array([lon_min, lon_max]),
        lat_min_max=np.array([lat_min, lat_max]),
        elev_min_max=np.array([elev_min, elev_max]),
        raw_lon=lon,
        raw_lat=lat,
        raw_elev=elev,
        raw_temp=temp
    )
    print(f"--> [California Temp 预处理完成] 样本总数: {N}, 训练集: {len(train_idx)}, 测试集: {len(test_idx)}")
    print(f"    temp 训练集原始均值: {mean_Z[0]:.2f}, 标准差: {std_Z[0]:.2f}")

if __name__ == "__main__":
    prepare_meuse()
    prepare_california()
