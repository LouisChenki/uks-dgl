# -*- coding: utf-8 -*-
"""
空间多维度矩阵化模拟数据生成器 (Spatial Multivariable Simulation Data Generator)
支持 5 个核心地理统计维度控制的参数化引擎，正交生成 Scenario A - E 的 5 套 Benchmark 空间数据集。
支持多通道物理主/协变量联合输出、非平稳旋转各向异性距离计算、非线性物理响应调制以及大面积空间块状缺失(Gap)。
"""

import os
import numpy as np
from scipy.spatial.distance import cdist

def generate_nonstationary_covariance(coords, l1_func, l2_func, theta_func, sigma_sq=0.5, nugget=1e-8):
    """
    基于 Higdon 对称化各向异性核，计算非平稳旋转各向异性空间协方差矩阵。
    """
    n = coords.shape[0]
    C = np.zeros((n, n))
    
    # 预先计算每个点处的旋转矩阵和特征值尺度
    rot_matrices = []
    l1_vals = []
    l2_vals = []
    for i in range(n):
        u = coords[i]
        theta = theta_func(u)
        l1 = l1_func(u)
        l2 = l2_func(u)
        l1_vals.append(l1)
        l2_vals.append(l2)
        
        # 旋转矩阵 R
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rot_matrices.append(R)
        
    # 计算对称化的各向异性协方差
    for i in range(n):
        for j in range(i, n):
            h = coords[i] - coords[j] # [2]
            
            # 在 i 处度量下的马氏距离
            h_rot_i = rot_matrices[i] @ h
            d_i_sq = (h_rot_i[0] / l1_vals[i])**2 + (h_rot_i[1] / l2_vals[i])**2
            
            # 在 j 处度量下的马氏距离
            h_rot_j = rot_matrices[j] @ h
            d_j_sq = (h_rot_j[0] / l1_vals[j])**2 + (h_rot_j[1] / l2_vals[j])**2
            
            # 对称化距离平方
            d_sym_sq = 0.5 * (d_i_sq + d_j_sq)
            d_sym = np.sqrt(d_sym_sq)
            
            val = sigma_sq * np.exp(-d_sym)
            C[i, j] = val
            C[j, i] = val
            
    C += nugget * np.eye(n)
    return C

def generate_spatial_data(
    n_samples: int = 300,
    seed: int = 42,
    dataset_type: str = "E",
    save_path: str = "../data/synthetic_data.npz"
):
    """
    根据五大控制维度参数生成 Co-Kriging 协同空间模拟数据。
    
    参数:
        n_samples (int): 总样本数 N
        seed (int): 随机数种子
        dataset_type (str): "A", "B", "C", "D", "E"
        save_path (str): 保存路径
    """
    np.random.seed(seed)
    print(f"\n--- 开始生成矩阵化空间模拟数据 [类型: Scenario {dataset_type}] ---")
    
    # 1. 物理空间定义：二维正方形坐标网格
    coords = np.random.uniform(0, 1, size=(n_samples, 2))  # [N, 2]
    u_x = coords[:, 0]
    u_y = coords[:, 1]
    
    # 2. 独立背景高斯场生成器
    # 构建基础平稳空间指数核 [N, N]
    D_phys = cdist(coords, coords, metric='euclidean')
    K_spatial = np.exp(-D_phys / 0.25) + 1e-8 * np.eye(n_samples)
    L_spatial = np.linalg.cholesky(K_spatial)
    
    # 采样两个独立的标准空间高斯噪声场，维度均为 [N]
    e1 = L_spatial @ np.random.normal(size=n_samples)
    e2 = L_spatial @ np.random.normal(size=n_samples)
    
    # 3. 数据集正交物理生成流程
    noise_var = 0.05
    eta1 = np.random.normal(0, np.sqrt(noise_var), size=n_samples) # 观测噪声 1 [N]
    eta2 = np.random.normal(0, np.sqrt(noise_var), size=n_samples) # 观测噪声 2 [N]
    
    if dataset_type == "A":
        # === Scenario A: 平稳高斯基准 ===
        # 3.1a 趋势面 (常数趋势)
        M1 = np.full(n_samples, 0.5)
        M2 = np.full(n_samples, 0.0)
        
        # 3.1b 潜空间 Y_1, Y_2 共区域化 LMC 相关 (强线性空间自相关, rho=0.85)
        Y1_res = e1
        Y2_res = 0.85 * e1 + np.sqrt(1 - 0.85**2) * e2
        
        Y1 = M1 + Y1_res
        Y2 = M2 + Y2_res
        
        # 3.1c 恒等映射 (纯高斯场)
        Z1 = Y1 + eta1
        Z2 = Y2 + eta2
        
        # 3.1d 采样率与缺失模式：50% 均匀随机缺失
        n_train = 150
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
    elif dataset_type == "B":
        # === Scenario B: 非平稳旋转各向异性核 ===
        # 3.2a 趋势面 (二次多项式趋势)
        M1 = 0.2 + 0.5 * u_x - 0.3 * u_y + 0.8 * (u_x**2) - 0.4 * (u_y**2)
        M2 = np.full(n_samples, 0.0)
        
        # 3.2b 潜空间非平稳各向异性协方差核生成
        l1_func = lambda u: 0.1 + 0.2 * u[0]
        l2_func = lambda u: 0.05 + 0.05 * u[1]
        theta_func = lambda u: (np.pi / 2.0) * u[0]
        
        C_nonstat = generate_nonstationary_covariance(coords, l1_func, l2_func, theta_func, sigma_sq=0.8)
        L_nonstat = np.linalg.cholesky(C_nonstat)
        
        e1_nonstat = L_nonstat @ np.random.normal(size=n_samples)
        e2_nonstat = L_nonstat @ np.random.normal(size=n_samples)
        
        # 潜空间 LMC 中度相关 (rho=0.6)
        Y1_res = e1_nonstat
        Y2_res = 0.6 * e1_nonstat + np.sqrt(1 - 0.6**2) * e2_nonstat
        
        Y1 = M1 + Y1_res
        Y2 = M2 + Y2_res
        
        # 3.2c 恒等高斯映射
        Z1 = Y1 + eta1
        Z2 = Y2 + eta2
        
        # 3.2d 采样率与缺失模式：30% 局部群聚式缺失
        n_train = 90
        # 三中心群聚传感器网络划定
        centers = np.array([[0.2, 0.3], [0.7, 0.4], [0.5, 0.8]])
        prob_weights = np.zeros(n_samples)
        for i in range(n_samples):
            dists = np.sum((coords[i] - centers)**2, axis=1) # 三中心平方距离
            prob_weights[i] = np.sum(np.exp(-dists / 0.02)) + 0.05
            
        prob_weights /= np.sum(prob_weights)
        train_idx = np.random.choice(n_samples, size=n_train, replace=False, p=prob_weights)
        test_idx = np.array([idx for idx in range(n_samples) if idx not in train_idx])
        
    elif dataset_type == "C":
        # === Scenario C: 大尺度非线性趋势 ===
        # 预先确定协变量 Y2 (作为物理驱动场)
        M2 = np.full(n_samples, 0.0)
        Y2_res = e2
        Y2 = M2 + Y2_res
        Z2 = Y2 + eta2  # 协变量物理观测值 [N]
        
        # 3.3a 主变量大尺度趋势非线性受 Z2 物理驱动
        M1 = 1.2 * np.cos(2.0 * np.pi * Z2) + 0.8 * np.log(np.abs(Z2) + 2.0)
        
        # 3.3b 潜空间残差 LMC 弱相关 (rho=0.4)
        Y1_res = 0.4 * e2 + np.sqrt(1 - 0.4**2) * e1
        Y1 = M1 + Y1_res
        Z1 = Y1 + eta1
        
        # 3.3c 缺失模式：30% 均匀随机缺失
        n_train = 90
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
    elif dataset_type == "D":
        # === Scenario D: 极度非高斯极值重尾 ===
        # 3.4a 无趋势漂移
        M1 = np.full(n_samples, 0.0)
        M2 = np.full(n_samples, 0.0)
        
        # 3.4b 潜空间强相关 (rho=0.7)
        Y1_res = e1
        Y2_res = 0.7 * e1 + np.sqrt(1 - 0.7**2) * e2
        Y1 = M1 + Y1_res
        Y2 = M2 + Y2_res
        
        # 3.4c 强非高斯映射 (复合偏重尾极值映射)
        Z1 = Y1 + 0.7 * np.sign(Y1) * (np.abs(Y1) ** 1.8) + 0.2 * np.sinh(1.6 * Y1) + eta1
        Z2 = Y2 + eta2
        
        # 3.4d 缺失模式：30% 均匀随机缺失
        n_train = 90
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
    elif dataset_type == "E":
        # === Scenario E: 物理非线性协变量调制 + 旋转各向异性核 + 双曲正弦映射 + 中心大空缺 (Gap) ===
        # 协变量场生成 (平稳高斯自相关)
        M2 = np.full(n_samples, 0.0)
        Y2_res = e2
        Y2 = M2 + Y2_res
        Z2 = Y2 + eta2
        
        # 3.5a 指数/二次复合非平稳协变量驱动均值
        M1 = 0.5 * np.exp(Z2) - 0.3 * (Z2**2)
        
        # 3.5b 非平稳旋转各向异性核残差 e1_nonstat
        l1_func = lambda u: 0.08 + 0.15 * u[0] * u[1]
        l2_func = lambda u: 0.04
        theta_func = lambda u: np.pi * (u[0] - u[1])
        
        C_nonstat = generate_nonstationary_covariance(coords, l1_func, l2_func, theta_func, sigma_sq=0.6)
        L_nonstat = np.linalg.cholesky(C_nonstat)
        e1_nonstat = L_nonstat @ np.random.normal(size=n_samples)
        
        # 非线性物理响应相关调制 (残差受到协变量高频空间信号调幅)
        Y1_res = e1_nonstat * (1.0 + 0.5 * np.sin(4.0 * np.pi * Z2))
        
        # 潜空间 Y_1
        Y1 = M1 + Y1_res
        
        # 3.5c 双曲正弦非高斯映射
        Z1 = np.sinh(1.2 * Y1) + eta1
        
        # 3.5d 采样率与缺失模式：10% 极度稀疏采样 (N_train=30) 且块状大面积空缺 (Gap Area)
        n_train = 30
        
        # 中心 Gap 挖空：[0.35, 0.65] x [0.35, 0.65]
        gap_mask = (u_x >= 0.35) & (u_x <= 0.65) & (u_y >= 0.35) & (u_y <= 0.65)
        
        # 所有 Gap 区域内的点全部强行分配给测试集
        non_gap_indices = np.where(~gap_mask)[0]
        
        # 在非 Gap 区域内随机抽取 n_train 个观测已知点
        train_idx = np.random.choice(non_gap_indices, size=n_train, replace=False)
        test_idx = np.array([idx for idx in range(n_samples) if idx not in train_idx])
        
    else:
        raise ValueError(f"不支持的场景类型: {dataset_type}")
        
    # 4. 构建 Co-Kriging 多通道堆叠格式 [N, q=2]
    Z = np.stack([Z1, Z2], axis=-1)  # [N, 2]
    Y = np.stack([Y1, Y2], axis=-1)  # [N, 2]
    M = np.stack([M1, M2], axis=-1)  # [N, 2]
    R = np.stack([Y1 - M1, Y2 - M2], axis=-1) # [N, 2]
    
    # 5. 划分训练/测试集
    coords_train = coords[train_idx]  # [N_train, 2]
    Z_train = Z[train_idx]            # [N_train, 2]
    Y_train = Y[train_idx]            # [N_train, 2]
    M_train = M[train_idx]            # [N_train, 2]
    R_train = R[train_idx]            # [N_train, 2]
    
    coords_test = coords[test_idx]    # [N_test, 2]
    Z_test = Z[test_idx]              # [N_test, 2]
    Y_test = Y[test_idx]              # [N_test, 2]
    M_test = M[test_idx]              # [N_test, 2]
    R_test = R[test_idx]              # [N_test, 2]
    
    # 本地没有物理协变量 X 的冗余，用 dummy 填充以兼容原 baselines 的趋势项求逆接口
    X_train = coords_train
    X_test = coords_test
    
    # 6. 防御性检测
    check_dict = {
        "coords_train": coords_train, "Z_train": Z_train, "Y_train": Y_train, "M_train": M_train, "R_train": R_train,
        "coords_test": coords_test, "Z_test": Z_test, "Y_test": Y_test, "M_test": M_test, "R_test": R_test
    }
    for name, arr in check_dict.items():
        assert not np.any(np.isnan(arr)), f"地统计防御异常: 字段 {name} 中存在 NaN 值！"
        assert not np.any(np.isinf(arr)), f"地统计防御异常: 字段 {name} 中存在 Inf 值！"
    print("--> [数据生成防御] 防御性检查通过，未发现 NaN 或 Inf。")
    
    # 7. 保存文件
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
    print(f"--> [打包完成] 数据集已成功写入: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    target_dir = "/Users/chenkaiqi/Documents/Papers/Learning Mechanics of Predictive GeoAI/learning_mechanics_test/data"
    
    # 自动生成 Scenario A - E 五套数据
    for s_type in ["A", "B", "C", "D", "E"]:
        generate_spatial_data(
            n_samples=300,
            seed=42,
            dataset_type=s_type,
            save_path=os.path.join(target_dir, f"synthetic_data_{s_type.lower()}.npz")
        )
