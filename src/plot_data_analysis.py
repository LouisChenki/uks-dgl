# -*- coding: utf-8 -*-
"""
UKS-DGL 多场景数据多维度特征分析学术绘图脚本 (Multidimensional Data Feature Visualization Script)
读取 D1, D2, D3 三个数据集，对比绘制物理空间 Z(u) 分布、PDF概率密度分布（含偏度度量）以及各向异性半变差函数分析。
图像输出至 results_20260602_run4/plots/raw_data_distribution.png 以及 results/plots/raw_data_distribution.png。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

def main():
    # 1. 创建输出目录
    output_dir = 'results_20260602_run4/plots'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # 定义待分析的数据集配置
    datasets_config = [
        {
            "type": "D1", 
            "file_path": "data/synthetic_data_d1.npz", 
            "title": "数据集 D1 (3星难度 - 平稳、各向同性、弱非高斯)", 
            "theta_0": 0.0, 
            "is_aniso": False
        },
        {
            "type": "D2", 
            "file_path": "data/synthetic_data_d2.npz", 
            "title": "数据集 D2 (5星难度 - 一阶非平稳、各向异性、温和非高斯)", 
            "theta_0": np.pi / 6.0,  # 30 度
            "is_aniso": True
        },
        {
            "type": "D3", 
            "file_path": "data/synthetic_data_d3.npz", 
            "title": "数据集 D3 (7星难度 - 强非平稳多变量 KED、强各向异性、高度偏态)", 
            "theta_0": np.pi / 4.0,  # 45 度
            "is_aniso": True
        }
    ]
    
    # 2. 设置学术绘图风格
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    # 3. 构建 3x3 矩阵大画布
    fig = plt.figure(figsize=(18, 15), dpi=300)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    for row_idx, cfg in enumerate(datasets_config):
        d_type = cfg["type"]
        f_path = cfg["file_path"]
        title_str = cfg["title"]
        theta_0 = cfg["theta_0"]
        is_aniso = cfg["is_aniso"]
        
        if not os.path.exists(f_path):
            print(f"错误: 找不到数据集 {f_path}。请先生成模拟数据。")
            return
            
        print(f"--> [数据加载] 正在分析加载数据源: {f_path}")
        data = np.load(f_path)
        
        # 提取训练集（已知采样观测点）数据
        coords_train = data['coords_train']  # [200, 2]
        Z_train = data['Z_train']            # [200]
        
        # ------------------ 列 1: 物理空间观测 Z 空间分布散点图 ------------------
        ax_col0 = fig.add_subplot(gs[row_idx, 0])
        scatter_a = ax_col0.scatter(
            coords_train[:, 0], coords_train[:, 1], 
            c=Z_train, cmap='coolwarm', s=45, edgecolors='k', linewidths=0.5
        )
        ax_col0.set_title(f"{d_type}-1. {d_type} 物理空间观测 $Z$ 散点场", fontweight='bold')
        ax_col0.set_xlabel("X 坐标")
        ax_col0.set_ylabel("Y 坐标")
        ax_col0.set_xlim(0, 1)
        ax_col0.set_ylim(0, 1)
        ax_col0.grid(True, linestyle='--', alpha=0.5)
        cbar_a = fig.colorbar(scatter_a, ax=ax_col0, fraction=0.046, pad=0.04)
        cbar_a.set_label("物理观测值 $Z$")
        
        # ------------------ 列 2: 非高斯偏态概率密度分布直方图与 KDE ------------------
        ax_col1 = fig.add_subplot(gs[row_idx, 1])
        n_bins = 20
        ax_col1.hist(Z_train, bins=n_bins, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='样本直方图')
        
        # 核密度估计 KDE
        kde = stats.gaussian_kde(Z_train)
        z_eval = np.linspace(Z_train.min() - 0.5, Z_train.max() + 0.5, 300)
        ax_col1.plot(z_eval, kde(z_eval), 'r-', lw=2, label='核密度估计 KDE')
        
        # 理论对照高斯概率密度
        mean_z = np.mean(Z_train)
        std_z = np.std(Z_train)
        normal_pdf = stats.norm.pdf(z_eval, loc=mean_z, scale=std_z)
        ax_col1.plot(z_eval, normal_pdf, 'k--', lw=1.5, label='对照高斯分布')
        
        skewness = stats.skew(Z_train)
        kurtosis = stats.kurtosis(Z_train)
        
        info_text = f"均值 = {mean_z:.4f}\n标准差 = {std_z:.4f}\n偏度 (Skewness) = {skewness:.4f}\n峰度 (Kurtosis) = {kurtosis:.4f}"
        ax_col1.text(0.05, 0.60, info_text, transform=ax_col1.transAxes, 
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.4'), fontsize=8.5)
                  
        ax_col1.set_title(f"{d_type}-2. {d_type} 强非高斯概率密度特征", fontweight='bold')
        ax_col1.set_xlabel("观测值 $Z$")
        ax_col1.set_ylabel("概率密度")
        ax_col1.grid(True, linestyle='--', alpha=0.5)
        ax_col1.legend(loc='upper right', fontsize=8)
        
        # ------------------ 列 3: 方向性经验半变异函数自相关分析 ------------------
        ax_col2 = fig.add_subplot(gs[row_idx, 2])
        n_points = len(coords_train)
        
        # 提取点对距离、夹角及半变差值
        dists = []
        semivariance = []
        angles = []
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                d = np.sqrt(np.sum((coords_train[i] - coords_train[j])**2))
                gamma = 0.5 * (Z_train[i] - Z_train[j])**2
                dy = coords_train[j, 1] - coords_train[i, 1]
                dx = coords_train[j, 0] - coords_train[i, 0]
                ang = np.arctan2(dy, dx)
                if ang < 0:
                    ang += np.pi  # 投影到 [0, pi]
                
                dists.append(d)
                semivariance.append(gamma)
                angles.append(ang)
                
        dists = np.array(dists)
        semivariance = np.array(semivariance)
        angles = np.array(angles)
        
        max_dist = 0.6
        n_bins_semi = 15
        bin_edges = np.linspace(0, max_dist, n_bins_semi + 1)
        
        if not is_aniso:
            # 各向同性场景：直接计算全方向经验变差
            bin_centers = []
            bin_gammas = []
            for k in range(n_bins_semi):
                idx = (dists >= bin_edges[k]) & (dists < bin_edges[k+1])
                if np.sum(idx) > 0:
                    bin_centers.append(np.mean(dists[idx]))
                    bin_gammas.append(np.mean(semivariance[idx]))
            
            ax_col2.scatter(bin_centers, bin_gammas, color='darkorange', edgecolor='black', s=45, label='全方向经验变差')
            
            # 各向同性指数核理论变差：gamma(d) = sill * (1 - exp(-d / range)) + nugget
            d_theory = np.linspace(0, max_dist, 100)
            gamma_theory = mean_z * 0.05 + 0.4 * (1.0 - np.exp(-d_theory / 0.20))
            ax_col2.plot(d_theory, gamma_theory, 'b-', lw=2, label='各向同性指数拟合')
            
        else:
            # 各向异性场景：分别提取主、次正交方向上的变差散点
            # 主轴方向 theta_0 夹角差值在 pi/8 以内
            diff_main = np.abs(angles - theta_0)
            diff_main = np.minimum(diff_main, np.pi - diff_main)
            idx_main_dir = diff_main < (np.pi / 8.0)
            
            # 次轴（正交轴）方向 theta_0 + pi/2 夹角差值在 pi/8 以内
            theta_orth = (theta_0 + np.pi / 2.0) % np.pi
            diff_orth = np.abs(angles - theta_orth)
            diff_orth = np.minimum(diff_orth, np.pi - diff_orth)
            idx_orth_dir = diff_orth < (np.pi / 8.0)
            
            # 计算主方向的滞后距离与变差
            bin_centers_main = []
            bin_gammas_main = []
            # 计算次方向的滞后距离与变差
            bin_centers_orth = []
            bin_gammas_orth = []
            
            for k in range(n_bins_semi):
                # 主方向
                idx_m = idx_main_dir & (dists >= bin_edges[k]) & (dists < bin_edges[k+1])
                if np.sum(idx_m) > 0:
                    bin_centers_main.append(np.mean(dists[idx_m]))
                    bin_gammas_main.append(np.mean(semivariance[idx_m]))
                
                # 次正交方向
                idx_o = idx_orth_dir & (dists >= bin_edges[k]) & (dists < bin_edges[k+1])
                if np.sum(idx_o) > 0:
                    bin_centers_orth.append(np.mean(dists[idx_o]))
                    bin_gammas_orth.append(np.mean(semivariance[idx_o]))
            
            ax_col2.scatter(bin_centers_main, bin_gammas_main, color='crimson', edgecolor='black', s=45, marker='o', label=f'主轴方向 ({np.degrees(theta_0):.0f}°)')
            ax_col2.scatter(bin_centers_orth, bin_gammas_orth, color='royalblue', edgecolor='black', s=45, marker='s', label=f'正交方向 ({np.degrees(theta_orth):.0f}°)')
            
            # 拟合趋势引导线
            d_theory = np.linspace(0, max_dist, 100)
            # 主轴方向变程大 (0.35 - 0.40)，上升慢
            range_main = 0.35 if d_type == "D2" else 0.40
            # 次轴方向变程小 (0.10 - 0.08)，上升快
            range_orth = 0.10 if d_type == "D2" else 0.08
            
            sill = np.var(Z_train) * 0.95
            
            if d_type == "D2":
                gamma_main = sill * (1.0 - np.exp(-d_theory / range_main))
                gamma_orth = sill * (1.0 - np.exp(-d_theory / range_orth))
            else: # D3
                # Matern 理论核变差：gamma(d) = sill * (1 - (1 + sqrt(3)*d/r)*exp(-sqrt(3)*d/r))
                gamma_main = sill * (1.0 - (1.0 + np.sqrt(3.0) * d_theory / range_main) * np.exp(-np.sqrt(3.0) * d_theory / range_main))
                gamma_orth = sill * (1.0 - (1.0 + np.sqrt(3.0) * d_theory / range_orth) * np.exp(-np.sqrt(3.0) * d_theory / range_orth))
                
            ax_col2.plot(d_theory, gamma_main, 'r-', lw=2, label='主轴理论变差')
            ax_col2.plot(d_theory, gamma_orth, 'b-', lw=2, label='正交理论变差')
            
        ax_col2.set_title(f"{d_type}-3. {d_type} 经验半变异自相关分析", fontweight='bold')
        ax_col2.set_xlabel("空间滞后距离 (d)")
        ax_col2.set_ylabel("半变异值 (gamma)")
        ax_col2.set_xlim(0, max_dist)
        ax_col2.grid(True, linestyle='--', alpha=0.5)
        ax_col2.legend(loc='lower right', fontsize=8.5)
        
    plt.suptitle("地理空间仿真数据集 (D1, D2, D3) 多维度地统计特征探索分析图 (图 0)\n(Multidimensional Geospatial Semivariance & Skewness Analysis of Datasets D1-D3, Fig 0)", fontsize=16, fontweight='bold', y=0.97)
    
    # 保存至两个 plots 目录下
    save_path1 = os.path.join(output_dir, 'raw_data_distribution.png')
    plt.savefig(save_path1, bbox_inches='tight', dpi=300)
    save_path2 = 'results/plots/raw_data_distribution.png'
    plt.savefig(save_path2, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"--> [绘图完成] 原始数据多特征图 0 成功保存至: {save_path1} 和 {save_path2}")

if __name__ == '__main__':
    main()
