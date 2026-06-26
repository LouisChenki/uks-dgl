# -*- coding: utf-8 -*-
"""
UKS-DGL 扩展数据集生成脚本 (Dataset Generator for Scenario E & F)
1. Scenario E: 双变量空间异质交叉相关非平稳随机场 (Multivariate Non-stationary Cross-correlation Field)
2. Scenario F: 偏态多中心非高斯随机场 (Highly Skewed Multimodal Non-Gaussian Field)
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
    
    # 2. 构造大尺度均值趋势 (双通道各自不同且非线性)
    def T1(u):
        return 1.5 * np.sin(np.pi * u[:, 0]) * np.cos(np.pi * u[:, 1])
    
    def T2(u):
        return u[:, 0]**2 + u[:, 1]**2
        
    M_train = np.zeros((n_train, 2))
    M_train[:, 0] = T1(coords_train)
    M_train[:, 1] = T2(coords_train)
    
    M_test = np.zeros((n_test, 2))
    M_test[:, 0] = T1(coords_test)
    M_test[:, 1] = T2(coords_test)
    
    # 3. 构造非平稳交叉空间残差
    # 产生两个独立的空间平稳高斯过程 ξ1, ξ2，其变程为 0.08，方差为 1.0
    l_corr = 0.08
    dist_matrix = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1))
    cov_matrix = np.exp(-dist_matrix / l_corr)
    
    # 采样
    xi1 = np.random.multivariate_normal(np.zeros(n_total), cov_matrix)
    xi2 = np.random.multivariate_normal(np.zeros(n_total), cov_matrix)
    
    # 计算空间位置自适应互相关系数 rho(u)
    # 随着坐标之和变化而正弦波动
    rho = 0.85 * np.sin(np.pi * (coords[:, 0] + coords[:, 1]))
    
    # 耦合生成空间互协变随机残差场
    R_total = np.zeros((n_total, 2))
    R_total[:, 0] = xi1
    R_total[:, 1] = rho * xi1 + np.sqrt(1.0 - rho**2) * xi2
    
    # 4. 叠加得到最终双通道观测值
    # 空间残差尺度设为 0.4
    sigma_res = 0.4
    
    T_total = np.zeros((n_total, 2))
    T_total[:, 0] = T1(coords)
    T_total[:, 1] = T2(coords)
    
    Z_total = T_total + sigma_res * R_total
    
    # 物理量分块
    Z_train = Z_total[:n_train]
    R_train = R_total[:n_train]
    
    Z_test = Z_total[n_train:]
    R_test = R_total[n_train:]
    
    # 生成空占位符以兼容 Run 11 对照 npz 的旧格式
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
    print(f"--> [数据生成] 场景 E 双变量非平稳互相关数据集已保存至: {output_path}")

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
        # 辅助协变量也具有相应的多中心物理规律
        return 0.8 * T1(u)
        
    # 2. 引入旋转 45° 的马氏空间各向异性残差场
    theta = np.pi / 4.0
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # 旋转变换坐标
    coords_rot = coords @ rot.T
    
    # 各向异性核变程
    l_major = 0.12
    l_minor = 0.05
    
    # 计算旋转后的各向异性马氏距离矩阵
    dx = coords_rot[:, None, 0] - coords_rot[None, :, 0]
    dy = coords_rot[:, None, 1] - coords_rot[None, :, 1]
    dist_aniso = np.sqrt((dx / l_major)**2 + (dy / l_minor)**2)
    
    # 指数变差各向异性协方差
    cov_matrix = np.exp(-dist_aniso)
    
    # 采样高斯残差
    e1 = np.random.multivariate_normal(np.zeros(n_total), cov_matrix)
    e2 = np.random.multivariate_normal(np.zeros(n_total), cov_matrix)
    
    # 3. 通过非线性对数拉伸生成高偏态非高斯物理场
    # Z = exp(T + e)
    sigma_noise = 0.35
    
    Z_total = np.zeros((n_total, 2))
    # 主变量
    Z_total[:, 0] = np.exp(T1(coords) + sigma_noise * e1)
    # 辅助变量
    Z_total[:, 1] = np.exp(T2(coords) + sigma_noise * e2)
    
    # 大尺度均值趋势的偏态值 (真实偏态趋势面)
    M_total = np.zeros((n_total, 2))
    M_total[:, 0] = np.exp(T1(coords))
    M_total[:, 1] = np.exp(T2(coords))
    
    # 偏态残差场
    R_total = Z_total - M_total
    
    # 分割
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
    print(f"--> [数据生成] 场景 F 偏态多中心非高斯数据集已保存至: {output_path}")

if __name__ == "__main__":
    # 创建 data 目录 (防御性)
    os.makedirs("data", exist_ok=True)
    generate_scenario_e()
    generate_scenario_f()
