# -*- coding: utf-8 -*-
"""
UKS-DGL 扩展数据集生成脚本 (Dataset Generator for Scenario E & F)
1. Scenario E: 双变量空间二次非线性联合相依随机场 (Multivariate Non-linear Bivariate Field)
2. Scenario F: 空间平滑旋转各向异性强偏态非高斯随机场 (Spatially Rotated Anisotropic Skewed Field)
"""

import os
import numpy as np

def generate_scenario_e():
    np.random.seed(42)
    # 1. 生成 150 个训练点和 150 个测试点
    n_train = 150
    n_test = 150
    n_total = n_train + n_test
    
    # 随机均匀采样空间位置 [0, 1] x [0, 1]
    coords = np.random.rand(n_total, 2)
    coords_train = coords[:n_train]
    coords_test = coords[n_train:]
    
    # 2. 构造辅助协变量 Z2 (包含二次均值与高斯空间波动)
    def T2(u):
        return u[:, 0]**2 + u[:, 1]**2
        
    l_corr = 0.07
    dist_matrix = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1))
    cov_matrix = np.exp(-dist_matrix / l_corr)
    
    # 采样协变量空间中等尺度扰动 (提供足够自相关让克里金捕获，协同辅助插值)
    xi2 = np.random.multivariate_normal(np.zeros(n_total), cov_matrix)
    Z2 = T2(coords) + 0.30 * xi2
    
    # 3. 构造主变量 Z1，协同二次非线性抛物线映射
    xi1 = np.random.multivariate_normal(np.zeros(n_total), cov_matrix)
    Z1 = Z2**2 - 1.0 + 0.10 * xi1
    
    Z_total = np.zeros((n_total, 2))
    Z_total[:, 0] = Z1
    Z_total[:, 1] = Z2
    
    # 物理数学期望的趋势面
    # E[Z1] = E[Z2^2] - 1.0 = T2^2 + 0.16 * E[xi2^2] - 1.0 = T2^2 - 0.84
    # E[Z2] = T2
    M_total = np.zeros((n_total, 2))
    M_total[:, 0] = T2(coords)**2 - 0.84
    M_total[:, 1] = T2(coords)
    
    M_train = M_total[:n_train]
    M_test = M_total[n_train:]
    
    R_total = Z_total - M_total
    R_train = R_total[:n_train]
    R_test = R_total[n_train:]
    
    Z_train = Z_total[:n_train]
    Z_test = Z_total[n_train:]
    
    # 生成空占位符以兼容旧的评测格式
    X_dummy_train = np.zeros_like(coords_train)
    Y_dummy_train = np.zeros_like(coords_train)
    X_dummy_test = np.zeros_like(coords_test)
    Y_dummy_test = np.zeros_like(coords_test)
    
    # 保存为 E 场景的 npz
    output_path = "data/synthetic_data_e.npz"
    np.savez(
        output_path,
        coords_train=coords_train,
        Z_train=Z_train,
        X_train=X_dummy_train,
        Y_train=Y_dummy_train,
        M_train=M_train,
        R_train=R_train,
        coords_test=coords_test,
        Z_test=Z_test,
        X_test=X_dummy_test,
        Y_test=Y_dummy_test,
        M_test=M_test,
        R_test=R_test
    )
    print(f"--> [数据生成] 场景 E 双变量非线性相依数据集已保存至: {output_path}")

def generate_scenario_f():
    np.random.seed(42)
    n_train = 150
    n_test = 150
    n_total = n_train + n_test
    
    coords = np.random.rand(n_total, 2)
    coords_train = coords[:n_train]
    coords_test = coords[n_train:]
    
    # 1. 均值趋势包含两个高斯模态中心 (源 A 与源 B)
    def T1(u):
        center_a = np.array([0.25, 0.25])
        center_b = np.array([0.75, 0.75])
        term_a = 1.0 * np.exp(-8.0 * np.sum((u - center_a)**2, axis=-1))
        term_b = 1.5 * np.exp(-10.0 * np.sum((u - center_b)**2, axis=-1))
        return term_a + term_b
    
    def T2(u):
        return 0.8 * T1(u)
        
    # 2. 构建空间自适应局部旋转各向异性协方差矩阵 (Spatially Rotated Anisotropy)
    # 旋转角随坐标线性平滑旋转 theta(u) = pi * (u_x + u_y)
    thetas = np.pi * (coords[:, 0] + coords[:, 1])
    
    l_major = 0.20
    l_minor = 0.045
    
    # 计算每一个采样点处的局部各向异性度量矩阵 G_i
    G = []
    for theta in thetas:
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        diag = np.array([
            [1.0 / (l_major ** 2), 0.0],
            [0.0, 1.0 / (l_minor ** 2)]
        ])
        G_i = rot @ diag @ rot.T
        G.append(G_i)
    G = np.array(G) # [n_total, 2, 2]
    
    # 计算对称的局部自适应马氏距离平方矩阵 D_ij^2 = 0.5 * (du^T G_i du + du^T G_j du)
    dist_sq = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            du = coords[i] - coords[j]
            d2_i = du @ G[i] @ du
            d2_j = du @ G[j] @ du
            dist_sq[i, j] = 0.5 * (d2_i + d2_j)
            
    cov_matrix = np.exp(-np.sqrt(dist_sq + 1e-12))
    
    # 采样高斯残差
    e1 = np.random.multivariate_normal(np.zeros(n_total), cov_matrix)
    e2 = np.random.multivariate_normal(np.zeros(n_total), cov_matrix)
    
    # 3. 通过非线性对数拉伸生成高偏态各向异性物理场 (利用 RealNVP 对数体积流拉直偏态)
    Z_total = np.zeros((n_total, 2))
    Z_total[:, 0] = np.exp(T1(coords) + 0.35 * e1)
    Z_total[:, 1] = np.exp(T2(coords) + 0.35 * e2)
    
    M_total = np.zeros((n_total, 2))
    M_total[:, 0] = np.exp(T1(coords))
    M_total[:, 1] = np.exp(T2(coords))
    
    R_total = Z_total - M_total
    
    Z_train = Z_total[:n_train]
    M_train = M_total[:n_train]
    R_train = R_total[:n_train]
    
    Z_test = Z_total[n_train:]
    M_test = M_total[n_train:]
    R_test = R_total[n_train:]
    
    X_dummy_train = np.zeros_like(coords_train)
    Y_dummy_train = np.zeros_like(coords_train)
    X_dummy_test = np.zeros_like(coords_test)
    Y_dummy_test = np.zeros_like(coords_test)
    
    output_path = "data/synthetic_data_f.npz"
    np.savez(
        output_path,
        coords_train=coords_train,
        Z_train=Z_train,
        X_train=X_dummy_train,
        Y_train=Y_dummy_train,
        M_train=M_train,
        R_train=R_train,
        coords_test=coords_test,
        Z_test=Z_test,
        X_test=X_dummy_test,
        Y_test=Y_dummy_test,
        M_test=M_test,
        R_test=R_test
    )
    print(f"--> [数据生成] 场景 F 局部旋转各向异性非高斯数据集已保存至: {output_path}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_scenario_e()
    generate_scenario_f()
