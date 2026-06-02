# -*- coding: utf-8 -*-
"""
空间模拟数据生成器 (Spatial Simulation Data Generator)
该脚本用于生成满足地统计一致性的三套具有不同复杂度（D1、D2、D3）的模拟数据集。
支持生成具有空间自相关的物理协变量，支持非平稳趋势面、各向异性协方差核以及强非高斯物理逆映射。
"""

import os
import numpy as np
from scipy.spatial.distance import cdist

def generate_spatial_data(
    n_samples: int = 300,
    n_train: int = 200,
    sigma_sq: float = 0.5,
    noise_var: float = 0.05,
    seed: int = 42,
    dataset_type: str = "D2",
    save_path: str = "../data/synthetic_data.npz"
):
    """
    生成地统计空间模拟数据并保存。
    
    参数:
        n_samples (int): 总样本点数 N
        n_train (int): 训练样本（已知采样点）数
        sigma_sq (float): 空间自相关残差场的方差 (Variance)
        noise_var (float): 独立同分布测量噪声的方差 (Noise Variance)
        seed (int): 随机数种子 (Random Seed)
        dataset_type (str): 数据集类型 ("D1", "D2", "D3")
        save_path (str): 数据保存路径 (Save Path)
    """
    # 1. 设置随机数种子以确保结果可重复性 (Reproducibility)
    np.random.seed(seed)
    
    print(f"\n--- 开始生成空间模拟数据 [类型: {dataset_type}] ---")
    
    # 2. 坐标定义：在 [0, 1] x [0, 1] 二维物理坐标系 (2D Physical Coordinate System) 中采样
    coords = np.random.uniform(0, 1, size=(n_samples, 2))  # [N, 2]
    
    # 3. 随机划分已知观测采样点（训练集）与未观测预测点（测试集）
    indices = np.arange(n_samples)  # [N]
    np.random.shuffle(indices)      # 随机打乱索引
    
    train_idx = indices[:n_train]    # [N_train]
    test_idx = indices[n_train:]     # [N_test]
    
    u_x = coords[:, 0]  # [N]
    u_y = coords[:, 1]  # [N]
    
    # 4. 生成 2 个具有空间自相关的辅助协变量 (Spatial Covariates) X1 和 X2
    # 协变量对于所有数据集都生成，但在 D1, D2 中只作为背景变量保留，在 D3 中会作为外部漂移驱动全局趋势面
    X1 = 1.5 * np.sin(2.0 * np.pi * u_x) * np.cos(2.0 * np.pi * u_y) + 0.3 * u_y  # [N]
    X2 = 2.0 * u_x - 1.2 * u_y + 0.5 * np.cos(2.0 * np.pi * (u_x + u_y))          # [N]
    X = np.stack([X1, X2], axis=-1)  # [N, 2]
    
    # 5. 构建大尺度全局趋势面 (Global Trend Surface) M(u)
    if dataset_type == "D1":
        # D1: 平稳常数趋势 (Stationary Constant Trend)
        M = np.full(n_samples, 7.5)  # [N]
    elif dataset_type == "D2":
        # D2: 非平稳一阶斜面趋势 (Non-stationary First-order Linear Drift)
        M = 1.5 * u_x + 1.0 * u_y    # [N]
    elif dataset_type == "D3":
        # D3: 强非平稳多变量外部漂移趋势 (Universal Drift with External Physical Covariates, KED)
        M = 1.5 * u_x - 1.2 * u_y + 1.0 * X1 + 0.8 * X2  # [N]
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
    # 6. 计算空间自相关残差场 R(u) 的各向异性协方差矩阵 (Anisotropic Covariance Matrix)
    # 根据数据集类型配置各向异性与核函数参数
    if dataset_type == "D1":
        # D1: 各向同性指数核 (Isotropic Exponential Kernel)
        theta_0 = 0.0
        l1 = 0.20
        l2 = 0.20
    elif dataset_type == "D2":
        # D2: 中等旋转各向异性指数核 (Anisotropic Exponential Kernel)
        theta_0 = np.pi / 6.0  # 30 度
        l1 = 0.35              # 长轴相关长度
        l2 = 0.10              # 短轴相关长度
    else:  # D3
        # D3: 强旋转各向异性马特恩核 (Anisotropic Matern Kernel, nu = 1.5)
        theta_0 = np.pi / 4.0  # 45 度
        l1 = 0.40              # 长轴相关长度
        l2 = 0.08              # 短轴相关长度

    # 旋转矩阵 R_rot [2, 2]
    R_rot = np.array([
        [np.cos(theta_0), -np.sin(theta_0)],
        [np.sin(theta_0),  np.cos(theta_0)]
    ])  # [2, 2]
    
    # 两两点对的物理坐标差向量 [N, N, 2]
    coords_diff = coords[:, None, :] - coords[None, :, :]  # [N, N, 2]
    
    # 将坐标差投影到各向异性主次轴方向上 [N, N, 2]
    coords_diff_rot = np.einsum('ijk,lk->ijl', coords_diff, R_rot)  # [N, N, 2]
    
    # 各向异性马氏距离 D_aniso [N, N]
    D_aniso = np.sqrt((coords_diff_rot[:, :, 0] / l1)**2 + (coords_diff_rot[:, :, 1] / l2)**2)  # [N, N]
    
    # 协方差核计算 (Covariance Kernel Calculation)
    if dataset_type in ["D1", "D2"]:
        # 指数核 (Exponential Kernel)
        C = sigma_sq * np.exp(-D_aniso)  # [N, N]
    else:
        # Matern 核 (nu = 1.5)
        C = sigma_sq * (1.0 + np.sqrt(3.0) * D_aniso) * np.exp(-np.sqrt(3.0) * D_aniso)  # [N, N]
        
    # 为确保协方差矩阵正定性，加入微小 nugget 扰动以避免数值 Cholesky 分解失败
    nugget = 1e-6 if dataset_type == "D3" else 1e-8
    C += nugget * np.eye(n_samples)  # [N, N]
    
    # 通过多元高斯采样 (Multivariate Gaussian Sampling) 得到残差场 R(u) ~ N(0, C)
    try:
        L = np.linalg.cholesky(C)  # [N, N]
        z = np.random.normal(size=n_samples)  # [N]
        R = L @ z  # [N]
    except np.linalg.LinAlgError:
        print("[警告] Cholesky 分解失败，改用传统多元高斯分布采样。")
        R = np.random.multivariate_normal(np.zeros(n_samples), C)  # [N]
        
    # 7. 计算隐空间高斯场 (Latent Gaussian Field) Y(u) = M(u) + R(u)
    Y = M + R  # [N]
    
    # 8. 非高斯物理观测场 (Non-Gaussian Physical Observation Field) Z(u)
    # 依据数据集难度，配置不同偏态的物理逆映射流层
    eta = np.random.normal(0, np.sqrt(noise_var), size=n_samples)  # 测量误差 (Observation Error) [N]
    
    if dataset_type == "D1":
        # D1: 线性变换 (物理 Z 场即为高斯隐空间场加上微小噪声)
        Z = Y + eta  # [N]
    elif dataset_type == "D2":
        # D2: 温和的双曲正弦变换 (Mildly Skewed sinh Transformation)
        Z = np.sinh(0.8 * Y) + eta  # [N]
    else:
        # D3: 高度非对称偏态与重尾非线性变换 (Highly Skewed Heavy-tailed sinh Transformation)
        # 通过复合非线性拉大色条对比度，制造空间极值的聚集特征
        Z = Y + 0.8 * np.sign(Y) * (np.abs(Y) ** 1.6) + 0.15 * np.sinh(1.5 * Y) + eta  # [N]
        
    # 9. 划分数据集 (Dataset Splitting)
    # 训练集 (Training Set - 已采样观测点)
    coords_train = coords[train_idx]  # [N_train, 2]
    Z_train = Z[train_idx]            # [N_train]
    X_train = X[train_idx]            # [N_train, 2]
    Y_train = Y[train_idx]            # [N_train]
    M_train = M[train_idx]            # [N_train]
    R_train = R[train_idx]            # [N_train]
    
    # 测试集 (Testing Set - 未观测预测点)
    coords_test = coords[test_idx]    # [N_test, 2]
    Z_test = Z[test_idx]              # [N_test]
    X_test = X[test_idx]              # [N_test, 2]
    Y_test = Y[test_idx]              # [N_test]
    M_test = M[test_idx]              # [N_test]
    R_test = R[test_idx]              # [N_test]
    
    # 10. 防御性检查 (Defensive Check)：实时监测确保数据中没有 NaN 或 Inf
    check_dict = {
        "coords_train": coords_train, "Z_train": Z_train, "X_train": X_train, "Y_train": Y_train, "M_train": M_train, "R_train": R_train,
        "coords_test": coords_test, "Z_test": Z_test, "X_test": X_test, "Y_test": Y_test, "M_test": M_test, "R_test": R_test
    }
    
    for name, arr in check_dict.items():
        assert not np.any(np.isnan(arr)), f"地统计防御异常: 字段 {name} 中存在 NaN 值！"
        assert not np.any(np.isinf(arr)), f"地统计防御异常: 字段 {name} 中存在 Inf 值！"
    
    print("--- 防御性检查通过：数据中无 NaN 或 Inf 值 ---")
    
    # 11. 打印生成的训练集和测试集数据的各字段均值与方差
    print("\n--- 空间数据统计特征 (Statistical Metrics) ---")
    print(f"{'字段 (Field)':<15} | {'训练集均值 (Train Mean)':<22} | {'训练集方差 (Train Var)':<22} | {'测试集均值 (Test Mean)':<22} | {'测试集方差 (Test Var)':<22}")
    print("-" * 115)
    
    fields = ["Z", "Y", "M", "R"]
    for f in fields:
        train_val = check_dict[f"{f}_train"]
        test_val = check_dict[f"{f}_test"]
        print(f"{f:<15} | {np.mean(train_val):<22.6f} | {np.var(train_val):<22.6f} | {np.mean(test_val):<22.6f} | {np.var(test_val):<22.6f}")
    
    print(f"{'X1':<15} | {np.mean(X_train[:, 0]):<22.6f} | {np.var(X_train[:, 0]):<22.6f} | {np.mean(X_test[:, 0]):<22.6f} | {np.var(X_test[:, 0]):<22.6f}")
    print(f"{'X2':<15} | {np.mean(X_train[:, 1]):<22.6f} | {np.var(X_train[:, 1]):<22.6f} | {np.mean(X_test[:, 1]):<22.6f} | {np.var(X_test[:, 1]):<22.6f}")
    
    # 12. 保存数据 (Data Saving)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        coords_train=coords_train,
        Z_train=Z_train,
        X_train=X_train,
        Y_train=Y_train,
        M_train=M_train,
        R_train=R_train,
        coords_test=coords_test,
        Z_test=Z_test,
        X_test=X_test,
        Y_test=Y_test,
        M_test=M_test,
        R_test=R_test
    )
    print(f"数据已打包成功并保存至: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    # 一键启动生成 D1、D2、D3 三套数据集
    target_dir = "/Users/chenkaiqi/Documents/Papers/Learning Mechanics of Predictive GeoAI/learning_mechanics_test/data"
    
    # 数据集 D1 生成并保存
    generate_spatial_data(
        n_samples=300,
        n_train=200,
        sigma_sq=0.5,
        noise_var=0.05,
        seed=42,
        dataset_type="D1",
        save_path=os.path.join(target_dir, "synthetic_data_d1.npz")
    )
    
    # 数据集 D2 生成并保存
    generate_spatial_data(
        n_samples=300,
        n_train=200,
        sigma_sq=0.5,
        noise_var=0.05,
        seed=42,
        dataset_type="D2",
        save_path=os.path.join(target_dir, "synthetic_data_d2.npz")
    )
    
    # 数据集 D3 生成并保存
    generate_spatial_data(
        n_samples=300,
        n_train=200,
        sigma_sq=0.5,
        noise_var=0.05,
        seed=42,
        dataset_type="D3",
        save_path=os.path.join(target_dir, "synthetic_data_d3.npz")
    )
