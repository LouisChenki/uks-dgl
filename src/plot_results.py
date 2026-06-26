# -*- coding: utf-8 -*-
"""
UKS-DGL 实验结果高维学术绘图脚本 (Academic Results Visualization Plotter)
读取 A, B, C, D 四个数据集的评估指标和中间变量，生成高清晰度 (DPI=300) 的学术图表 1-8。
所有图表均输出至 results_20260608_run11/plots/ 目录下，并同步保存至 results/plots/。
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata, Rbf
import scipy.stats as stats

def main():
    # 1. 确保绘图输出目录存在
    output_dir = 'results_20260608_run11/plots'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # 2. 检查并载入六个场景数据
    d_names = ["A", "B", "C", "D", "E", "F"]
    data_dict = {}
    res_dict = {}
    metrics_dict = {}
    
    for d in d_names:
        data_path = f"data/synthetic_data_{d.lower()}.npz"
        res_path = f"results_20260608_run11/{d}/experiment_results.npz"
        metrics_path = f"results_20260608_run11/{d}/metrics.json"
        
        if not (os.path.exists(data_path) and os.path.exists(res_path) and os.path.exists(metrics_path)):
            print(f"错误: 找不到数据集或实验成果文件: {d}")
            return
            
        data_dict[d] = np.load(data_path)
        res_dict[d] = np.load(res_path)
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_dict[d] = json.load(f)
            
    print(f"--> [绘图启动] 成功重载第十一轮实验多场景成果数据。")
    
    # 设置学术绘图风格 (Academic Plotting Style)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9.0
    
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    
    # ------------------ 图 1: kriging_vs_mlp.png (4行8列空间插值矩阵) ------------------
    print("--> 1. 正在绘制: kriging_vs_mlp.png (图 1)")
    fig = plt.figure(figsize=(26, 22.5), dpi=300)
    gs = gridspec.GridSpec(6, 8, figure=fig, hspace=0.22, wspace=0.16)

    
    for r_idx, d in enumerate(d_names):
        data = data_dict[d]
        res = res_dict[d]
        coords_train = data['coords_train']
        Z_train = data['Z_train'][:, 0]
        coords_test = res['coords_test']
        
        points = np.vstack([coords_train, coords_test])
        
        # 1) 真值场
        val_true = np.concatenate([Z_train, res['Z_test']])
        grid_true = griddata(points, val_true, (grid_x, grid_y), method='cubic')
        # 2) OK 基线
        val_ok = np.concatenate([Z_train, res['Z_pred_ok']])
        grid_ok = griddata(points, val_ok, (grid_x, grid_y), method='cubic')
        # 3) UK 基线
        val_uk = np.concatenate([Z_train, res['Z_pred_uk']])
        grid_uk = griddata(points, val_uk, (grid_x, grid_y), method='cubic')
        # 4) CK 基线
        val_ck = np.concatenate([Z_train, res['Z_pred_ck']])
        grid_ck = griddata(points, val_ck, (grid_x, grid_y), method='cubic')
        # 5) MLP 基线
        val_mlp = np.concatenate([Z_train, res['Z_pred_mlp']])
        grid_mlp = griddata(points, val_mlp, (grid_x, grid_y), method='cubic')
        # 6) DKNN 基线
        val_dknn = np.concatenate([Z_train, res['Z_pred_dknn']])
        grid_dknn = griddata(points, val_dknn, (grid_x, grid_y), method='cubic')
        # 7) DeepKriging 基线
        val_dk = np.concatenate([Z_train, res['Z_pred_dk']])
        grid_dk = griddata(points, val_dk, (grid_x, grid_y), method='cubic')
        # 8) Ours (UKS-DGL)
        val_uks = np.concatenate([Z_train, res['Z_pred_uks']])
        grid_uks = griddata(points, val_uks, (grid_x, grid_y), method='cubic')
        
        vmin = min(val_true.min(), val_uks.min())
        vmax = max(val_true.max(), val_uks.max())
        
        titles = [
            f"Scenario {d}: 真实值", 
            f"Scenario {d}: OK", 
            f"Scenario {d}: UK", 
            f"Scenario {d}: CK", 
            f"Scenario {d}: MLP Net", 
            f"Scenario {d}: DKNN", 
            f"Scenario {d}: DeepKriging", 
            f"Scenario {d}: UKS-DGL (Ours)"
        ]
        grids = [grid_true, grid_ok, grid_uk, grid_ck, grid_mlp, grid_dknn, grid_dk, grid_uks]
        
        for c_idx in range(8):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            im = ax.imshow(grids[c_idx].T, extent=(0, 1, 0, 1), origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax)
            
            # 在真值场上绘制已知的采样观测点
            if c_idx == 0:
                ax.scatter(coords_train[:, 0], coords_train[:, 1], c='black', s=4, alpha=0.5, label='采样观测点')
                ax.legend(loc='upper right', fontsize=7.0)
                
            ax.set_title(titles[c_idx], fontweight='bold', fontsize=8.5)
            ax.set_xlabel("X 坐标", fontsize=8.0)
            ax.set_ylabel("Y 坐标", fontsize=8.0)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    plt.suptitle("不同场景下各模型空间插值预测对比矩阵 (图 1)\n(Spatial Prediction Fields Comparison across Multi-scenarios A-D, Fig 1)", fontsize=14, fontweight='bold', y=0.96)
    plt.savefig(f'{output_dir}/kriging_vs_mlp.png', bbox_inches='tight', dpi=300)
    plt.savefig('results/plots/kriging_vs_mlp.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # ------------------ 图 2: uncertainty_variance.png (1行6列条件估计方差场) ------------------
    print("--> 2. 正在绘制: uncertainty_variance.png (图 2)")
    fig, axes = plt.subplots(1, 6, figsize=(33, 4.8), dpi=300)
    
    for c_idx, d in enumerate(d_names):
        res = res_dict[d]
        coords_test = res['coords_test']
        Z_var_uks = res['Z_var_uks']
        
        grid_var = griddata(coords_test, Z_var_uks, (grid_x, grid_y), method='cubic')
        
        ax = axes[c_idx]
        im = ax.contourf(grid_x, grid_y, grid_var.T, levels=25, cmap='plasma')
        contours = ax.contour(grid_x, grid_y, grid_var.T, levels=8, colors='white', linewidths=0.5, alpha=0.6)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.3f')
        
        data = data_dict[d]
        coords_train = data['coords_train']
        ax.scatter(coords_train[:, 0], coords_train[:, 1], c='green', s=8, alpha=0.4, label='已观测采样点')
        
        ax.set_title(f"Scenario {d} 条件方差场 ($\\sigma_Z^2$)", fontweight='bold')
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")
        ax.legend(loc='upper right', fontsize=8)
        fig.colorbar(im, ax=ax, label='物理空间条件方差 $\sigma_Z^2$')
        
    plt.suptitle("物理空间估计不确定性条件方差场等值线图 (图 2)\n(Physical Conditional Variance Anisotropy Contour Maps, Fig 2)", fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(f'{output_dir}/uncertainty_variance.png', bbox_inches='tight', dpi=300)
    plt.savefig('results/plots/uncertainty_variance.png', bbox_inches='tight', dpi=300)
    plt.close()

    
    # ------------------ 图 3: trend_surface_fit.png (6行3列趋势面解耦与绝对偏差场) ------------------
    print("--> 3. 正在绘制: trend_surface_fit.png (图 3)")
    fig = plt.figure(figsize=(15, 25.5), dpi=300)
    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.25, wspace=0.20)
    
    for r_idx, d in enumerate(d_names):
        data = data_dict[d]
        res = res_dict[d]
        coords_train = data['coords_train']
        coords_test = res['coords_test']
        points = np.vstack([coords_train, coords_test])
        
        M_train = data['M_train']
        M_test = data['M_test']
        M_true_all = np.concatenate([M_train, M_test])
        if M_true_all.ndim > 1:
            M_true_all = M_true_all[:, 0]
            
        M_hat_train = res['M_hat_train']
        M_hat_test = res['M_hat_test']
        M_hat_all = np.concatenate([M_hat_train, M_hat_test])
        if M_hat_all.ndim > 1:
            M_hat_all = M_hat_all[:, 0]
        
        # 计算绝对差场
        abs_diff_all = np.abs(M_true_all - M_hat_all)

        
        grid_mt = griddata(points, M_true_all, (grid_x, grid_y), method='cubic')
        grid_mh = griddata(points, M_hat_all, (grid_x, grid_y), method='cubic')
        grid_diff = griddata(points, abs_diff_all, (grid_x, grid_y), method='cubic')
        
        titles_trend = [
            f"Scenario {d}: 真实趋势面 $T(u)$", 
            f"Scenario {d}: 解耦趋势面 $\\hat{{T}}(u)$", 
            f"Scenario {d}: 趋势拟合绝对偏差 |$T - \\hat{{T}}$|"
        ]
        
        # vmin/vmax for Trend
        vmin_m = min(M_true_all.min(), M_hat_all.min())
        vmax_m = max(M_true_all.max(), M_hat_all.max())
        
        # col 0: 真实趋势面
        ax_col0 = fig.add_subplot(gs[r_idx, 0])
        im0 = ax_col0.imshow(grid_mt.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis', vmin=vmin_m, vmax=vmax_m)
        ax_col0.set_title(titles_trend[0], fontweight='bold')
        ax_col0.set_xlabel("X 坐标")
        ax_col0.set_ylabel("Y 坐标")
        fig.colorbar(im0, ax=ax_col0, fraction=0.046, pad=0.04)
        
        # col 1: 网络解耦趋势面
        ax_col1 = fig.add_subplot(gs[r_idx, 1])
        im1 = ax_col1.imshow(grid_mh.T, extent=(0, 1, 0, 1), origin='lower', cmap='viridis', vmin=vmin_m, vmax=vmax_m)
        ax_col1.set_title(titles_trend[1], fontweight='bold')
        ax_col1.set_xlabel("X 坐标")
        ax_col1.set_ylabel("Y 坐标")
        fig.colorbar(im1, ax=ax_col1, fraction=0.046, pad=0.04)
        
        # col 2: 绝对偏差场
        ax_col2 = fig.add_subplot(gs[r_idx, 2])
        im2 = ax_col2.imshow(grid_diff.T, extent=(0, 1, 0, 1), origin='lower', cmap='plasma')
        ax_col2.set_title(titles_trend[2], fontweight='bold')
        ax_col2.set_xlabel("X 坐标")
        ax_col2.set_ylabel("Y 坐标")
        fig.colorbar(im2, ax=ax_col2, fraction=0.046, pad=0.04)
        
    plt.suptitle("大尺度趋势面拟合与神经网络解耦绝对偏差对比场 (图 3)\n(Decoupled Trend Surface vs. True Trend & Absolute Error Comparison, Fig 3)", fontsize=15, fontweight='bold', y=0.97)
    plt.savefig(f'{output_dir}/trend_surface_fit.png', bbox_inches='tight', dpi=300)
    plt.savefig('results/plots/trend_surface_fit.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # ------------------ 图 4: loss_weighting_history.png (超参敏感响应面与训练Loss权重历程) ------------------
    print("--> 4. 正在绘制: loss_weighting_history.png (图 4)")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)
    
    # 子图 A: Scenario B HPO 敏感度响应面分析 (Log-scale Scatter)
    hpo_path = "results_20260608_run11/B/hpo_history.npz"
    if os.path.exists(hpo_path):
        hpo_data = np.load(hpo_path)
        lambda_flow = hpo_data["lambda_flow"]
        lambda_geo = hpo_data["lambda_geo"]
        r2s = hpo_data["r2"]
        
        # 归一化 R2 大小，以便显示不同的点大小
        size_norm = (r2s - r2s.min()) / (r2s.max() - r2s.min() + 1e-5) * 120 + 40
        
        sc0 = axes[0].scatter(lambda_flow, lambda_geo, s=size_norm, c=r2s, cmap='viridis', edgecolors='black', alpha=0.85, linewidths=0.8)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel("可逆流正则权重 $\\lambda_{flow}$ (Log Scale)", fontweight='bold')
        axes[0].set_ylabel("测地几何正则权重 $\\lambda_{geo}$ (Log Scale)", fontweight='bold')
        axes[0].set_title("A. 场景 B 物理超参空间 HPO 寻优响应面 (散点大小/颜色代表 R2)", fontweight='bold', fontsize=10)
        axes[0].grid(True, which="both", ls="--", color="gray", alpha=0.3)
        
        # 标注最优参数组合
        best_idx = np.argmax(r2s)
        axes[0].scatter(lambda_flow[best_idx], lambda_geo[best_idx], s=250, facecolors='none', edgecolors='red', linewidths=1.8, marker='o', label='最优 HPO 组合')
        axes[0].legend(loc='lower left', fontsize=8.5)
        
        cbar0 = fig.colorbar(sc0, ax=axes[0])
        cbar0.set_label("拟合优度 $R^2$")
    else:
        axes[0].text(0.5, 0.5, "Scenario B HPO 数据未找到", ha="center", va="center")
        
    # 子图 B: Scenario B 的训练 Loss 曲线与损失权重自适应变迁 (双轴对齐)
    best_hist_path = "results_20260608_run11/B/best_training_history.npz"
    if os.path.exists(best_hist_path):
        hist_data = np.load(best_hist_path)
        epochs = hist_data["epoch"]
        loss_total = hist_data["loss_total"]
        loss_pred = hist_data["loss_pred"]
        loss_flow = hist_data["loss_flow"]
        loss_geo = hist_data["loss_geo"]
        loss_uks = hist_data["loss_uks"]
        weights = hist_data["weights"]  # [num_epochs, 4]
        
        # 绘制 Loss 曲线 (使用左轴)
        ax_left = axes[1]
        line1 = ax_left.plot(epochs, loss_total, 'k-', lw=1.8, label='总损失 (Loss Total)')
        line2 = ax_left.plot(epochs, loss_pred, 'r--', lw=1.2, label='预测损失 (Loss Pred)')
        line3 = ax_left.plot(epochs, loss_flow, 'g:', lw=1.2, label='流体积损失 (Loss Flow)')
        ax_left.set_xlabel("训练轮次 (Epoch)", fontweight='bold')
        ax_left.set_ylabel("物理损失值 (Loss Value)", fontweight='bold')
        ax_left.set_title("B. 场景 B 最优模型的多任务 Loss 收敛与权重自适应调整历程", fontweight='bold', fontsize=10)
        ax_left.grid(True, linestyle='--', alpha=0.5)
        
        # 绘制 Weights 曲线 (使用右轴)
        ax_right = ax_left.twinx()
        line4 = ax_right.plot(epochs, weights[:, 0], 'r-', alpha=0.6, lw=1.5, label='预测权重 (w_Pred)')
        line5 = ax_right.plot(epochs, weights[:, 1], 'g-', alpha=0.6, lw=1.5, label='流体积权重 (w_Flow)')
        line6 = ax_right.plot(epochs, weights[:, 2], 'b-', alpha=0.6, lw=1.5, label='几何权重 (w_Geo)')
        line7 = ax_right.plot(epochs, weights[:, 3], 'y-', alpha=0.6, lw=1.5, label='UKS权重 (w_UKS)')
        ax_right.set_ylabel("自适应归一化损失权重 (Loss Weights)", color='blue', fontweight='bold')
        ax_right.tick_params(axis='y', labelcolor='blue')
        
        # 合并图例
        lines = line1 + line2 + line3 + line4 + line5 + line6 + line7
        labels = [l.get_label() for l in lines]
        ax_left.legend(lines, labels, loc='upper right', fontsize=8.0, framealpha=0.8)
    else:
        axes[1].text(0.5, 0.5, "Scenario B 最优模型训练历史未找到", ha="center", va="center")
        
    plt.suptitle("自适应多场景超参寻优及多任务损失同方差加权历史收敛图 (图 4)\n(Multi-scenario Hyperparameter Search & Homoscedastic Weighting History, Fig 4)", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(f'{output_dir}/loss_weighting_history.png', bbox_inches='tight', dpi=300)
    plt.savefig('results/plots/loss_weighting_history.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # ------------------ 图 5: adaptive_covariance.png (4行3列协方差马氏椭圆对比) ------------------
    print("--> 5. 正在绘制: adaptive_covariance.png (图 5)")
    fig = plt.figure(figsize=(15, 24), dpi=300)
    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.25, wspace=0.20)
    
    ref_pts = [[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]]
    
    for r_idx, d in enumerate(d_names):
        res = res_dict[d]
        size1 = res['cov_field_1'].size
        n = int(np.sqrt(size1))
        
        g_x_cov = np.linspace(0, 1, n)
        g_y_cov = np.linspace(0, 1, n)
        g_xx, g_yy = np.meshgrid(g_x_cov, g_y_cov)
        
        cov1 = res['cov_field_1'].reshape(n, n)
        cov2 = res['cov_field_2'].reshape(n, n)
        cov3 = res['cov_field_3'].reshape(n, n)
        covs = [cov1, cov2, cov3]

        
        labels_cov = [
            f"Scenario {d}: u1=(0.2, 0.2)", 
            f"Scenario {d}: u2=(0.5, 0.5)", 
            f"Scenario {d}: u3=(0.8, 0.8)"
        ]
        
        for c_idx in range(3):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            im = ax.contourf(g_xx, g_yy, covs[c_idx].T, levels=20, cmap='plasma')
            contours = ax.contour(g_xx, g_yy, covs[c_idx].T, levels=6, colors='white', linewidths=0.4, alpha=0.5)
            ax.clabel(contours, inline=True, fontsize=7.5, fmt='%.2f')
            ax.scatter(ref_pts[c_idx][0], ref_pts[c_idx][1], color='cyan', marker='*', s=120, edgecolors='black', linewidths=0.8, zorder=5)
            
            ax.set_title(labels_cov[c_idx], fontweight='bold')
            ax.set_xlabel("X 坐标")
            ax.set_ylabel("Y 坐标")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
    plt.suptitle("自适应局部协方差空间各向异性与非平稳马氏椭圆拟合图 (图 5)\n(Local Anisotropic Covariance Mahalanobis Ellipses comparison, Fig 5)", fontsize=14, fontweight='bold', y=0.96)
    plt.savefig(f'{output_dir}/adaptive_covariance.png', bbox_inches='tight', dpi=300)
    plt.savefig('results/plots/adaptive_covariance.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # ------------------ 图 6: gradient_profile.png (Scenario B 在常数趋势最优点的前反向伴随) ------------------
    print("--> 6. 正在绘制: gradient_profile.png (图 6)")
    res_b = res_dict["B"]
    data_b = data_dict["B"]
    
    if 'Lambda_u0' in res_b.files and 'lambda_C_u0' in res_b.files:
        Lambda_u0 = res_b['Lambda_u0']
        lambda_C_u0 = res_b['lambda_C_u0']
        coords_train = data_b['coords_train']
        
        N_train = len(coords_train)
        Lambda_u0_sub = Lambda_u0[:N_train]
        lambda_C_u0_sub = lambda_C_u0[:N_train]
        
        # 采用 griddata cubic 方法对散点权重进行精确的 2D 空间插值投影，保证局部各向异性的细节不被大尺度 RBF 抹平
        grid_x_cov = np.linspace(0, 1, 150)
        grid_y_cov = np.linspace(0, 1, 150)
        g_xx, g_yy = np.meshgrid(grid_x_cov, grid_y_cov)
        
        grid_Lambda = griddata(coords_train, Lambda_u0_sub, (g_xx, g_yy), method='cubic')
        grid_Lambda = np.nan_to_num(grid_Lambda, nan=0.0)
        
        grid_lambda_C = griddata(coords_train, lambda_C_u0_sub, (g_xx, g_yy), method='cubic')
        grid_lambda_C = np.nan_to_num(grid_lambda_C, nan=0.0)
        
        corr_val = np.corrcoef(Lambda_u0_sub, lambda_C_u0_sub)[0, 1]
        
        fig = plt.figure(figsize=(18, 5.5), dpi=300)
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.25)
        
        u0_coords = res_b['u0_coords'] if 'u0_coords' in res_b.files else coords_test[0]
        
        # (a) 前向插值权重扩散场
        ax_a = fig.add_subplot(gs[0, 0])
        vlim_a = max(np.max(np.abs(Lambda_u0_sub)), 1e-4)
        im_a = ax_a.contourf(g_xx, g_yy, grid_Lambda, levels=25, cmap='RdBu_r', vmin=-vlim_a, vmax=vlim_a)
        contours_a = ax_a.contour(g_xx, g_yy, grid_Lambda, levels=8, colors='black', linewidths=0.3, alpha=0.3)
        ax_a.clabel(contours_a, inline=True, fontsize=7.5, fmt='%.3f')
        ax_a.scatter(u0_coords[0], u0_coords[1], color='yellow', marker='*', s=250, edgecolors='black', linewidths=1.2, label='预测点 $u_0$', zorder=10)
        
        # 叠加离散观测点实际前向克里金权重
        sizes_a = np.abs(Lambda_u0_sub) / vlim_a * 150 + 10
        sc_a = ax_a.scatter(coords_train[:, 0], coords_train[:, 1], c=Lambda_u0_sub, s=sizes_a, cmap='RdBu_r', edgecolors='black', linewidths=0.4, vmin=-vlim_a, vmax=vlim_a, alpha=0.75, zorder=5)
        
        ax_a.set_title("A. 2D 前向估计权重场 $\\Lambda$ (Scenario B)\n(等值线结合离散观测点气泡贡献大小)", fontweight='bold', fontsize=9.5)
        ax_a.set_xlabel("X 坐标")
        ax_a.set_ylabel("Y 坐标")
        ax_a.legend(loc='upper right', fontsize=8)
        fig.colorbar(sc_a, ax=ax_a, label='前向权重 $\\lambda_i$')
        
        # (b) 反向伴随敏感场
        ax_b = fig.add_subplot(gs[0, 1])
        vlim_b = max(np.max(np.abs(lambda_C_u0_sub)), 1e-4)
        im_b = ax_b.contourf(g_xx, g_yy, grid_lambda_C, levels=25, cmap='RdBu_r', vmin=-vlim_b, vmax=vlim_b)
        contours_b = ax_b.contour(g_xx, g_yy, grid_lambda_C, levels=8, colors='black', linewidths=0.3, alpha=0.3)
        ax_b.clabel(contours_b, inline=True, fontsize=7.5, fmt='%.3f')
        ax_b.scatter(u0_coords[0], u0_coords[1], color='yellow', marker='*', s=250, edgecolors='black', linewidths=1.2, label='预测点 $u_0$', zorder=10)
        
        # 叠加离散观测点实际反向伴随误差敏感度
        sizes_b = np.abs(lambda_C_u0_sub) / vlim_b * 150 + 10
        sc_b = ax_b.scatter(coords_train[:, 0], coords_train[:, 1], c=lambda_C_u0_sub, s=sizes_b, cmap='RdBu_r', edgecolors='black', linewidths=0.4, vmin=-vlim_b, vmax=vlim_b, alpha=0.75, zorder=5)
        
        ax_b.set_title("B. 2D 反向误差敏感场 $\\lambda_C$ (Scenario B)\n(等值线结合离散观测点气泡贡献大小)", fontweight='bold', fontsize=9.5)
        ax_b.set_xlabel("X 坐标")
        ax_b.set_ylabel("Y 坐标")
        ax_b.legend(loc='upper right', fontsize=8)
        fig.colorbar(sc_b, ax=ax_b, label='伴随状态变量 $\\lambda_{C,i}$')
        
        # (c) 相关对照散点
        ax_c = fig.add_subplot(gs[0, 2])
        ax_c.scatter(Lambda_u0_sub, lambda_C_u0_sub, color='purple', alpha=0.6, edgecolors='k', s=45)
        # 绘制主对角斜线参考
        lims = [
            np.min([ax_c.get_xlim(), ax_c.get_ylim()]),  # min of both axes
            np.max([ax_c.get_xlim(), ax_c.get_ylim()]),  # max of both axes
        ]
        ax_c.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        ax_c.set_title("C. 前向权重 vs. 反向伴随对照散点 (Scenario B)", fontweight='bold', fontsize=9.5)
        ax_c.set_xlabel("前向权重 $\\lambda_i$")
        ax_c.set_ylabel("反向伴随变量 $\\lambda_{C,i}$")
        ax_c.grid(True, linestyle='--', alpha=0.5)
        ax_c.text(0.05, 0.90, f"Pearson 相关 = {corr_val:.4f}", transform=ax_c.transAxes,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'), fontsize=9.5, fontweight='bold')
    else:
        fig = plt.figure(figsize=(18, 5.2), dpi=300)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Scenario B 伴随点梯度数据尚未计算", ha="center", va="center")
        
    plt.suptitle("前反向传播物理伴随同构双扩散场分析图 (图 6)\n(Forward-Backward Spatial Adjoint Isomorphism Dual Diffusion Fields, Fig 6)", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(f'{output_dir}/gradient_profile.png', bbox_inches='tight', dpi=300)
    plt.savefig('results/plots/gradient_profile.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # ------------------ 图 7: cross_section_profile.png (Scenario D 切线剖面估计对照图) ------------------
    print("--> 7. 正在绘制: cross_section_profile.png (图 7)")
    res_d = res_dict["D"]
    data_d = data_dict["D"]
    coords_train_d = data_d['coords_train']
    Z_train_d = data_d['Z_train'][:, 0]
    coords_test_d = res_d['coords_test']
    
    points_d = np.vstack([coords_train_d, coords_test_d])
    
    val_true_d = np.concatenate([Z_train_d, res_d['Z_test']])
    val_uks_d = np.concatenate([Z_train_d, res_d['Z_pred_uks']])
    val_ok_d = np.concatenate([Z_train_d, res_d['Z_pred_ok']])
    val_mlp_d = np.concatenate([Z_train_d, res_d['Z_pred_mlp']])
    
    grid_true_d = griddata(points_d, val_true_d, (grid_x, grid_y), method='cubic')
    grid_uks_d = griddata(points_d, val_uks_d, (grid_x, grid_y), method='cubic')
    grid_ok_d = griddata(points_d, val_ok_d, (grid_x, grid_y), method='cubic')
    grid_mlp_d = griddata(points_d, val_mlp_d, (grid_x, grid_y), method='cubic')
    
    # 提取 Y=0.5 剖面线
    x_profile = grid_x[:, 50]
    profile_true = grid_true_d[:, 50]
    profile_uks = grid_uks_d[:, 50]
    profile_ok = grid_ok_d[:, 50]
    profile_mlp = grid_mlp_d[:, 50]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    
    ax.plot(x_profile, profile_true, label='真实值 (True Field)', color='black', linestyle='-', linewidth=2.0)
    ax.plot(x_profile, profile_uks, label='UKS-DGL (Ours)', color='red', linestyle='--', linewidth=1.8)
    ax.plot(x_profile, profile_ok, label='普通克里金 (Ordinary Kriging)', color='blue', linestyle='-.', linewidth=1.2)
    ax.plot(x_profile, profile_mlp, label='纯 MLP 预测 (MLP Baseline)', color='green', linestyle=':', linewidth=1.2)
    
    # 标出在该切线附近的已知观测点 (Y 坐标在 0.45 到 0.55 之间)
    near_mask = (coords_train_d[:, 1] >= 0.45) & (coords_train_d[:, 1] <= 0.55)
    ax.scatter(coords_train_d[near_mask, 0], Z_train_d[near_mask], color='black', marker='o', s=45, zorder=5, label='临近已知采样点')
    
    ax.set_title("场景 D 大尺度随机场中心剖面线一维估计对照图 (Y = 0.5)\n(Scenario D Cross-Section Profile Interpolation comparison at Y = 0.5, Fig 7)", fontsize=11, fontweight='bold')
    ax.set_xlabel("X 空间坐标")
    ax.set_ylabel("主变量估计值 $Z_1$")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.savefig(f'{output_dir}/cross_section_profile.png', bbox_inches='tight', dpi=300)
    plt.savefig('results/plots/cross_section_profile.png', bbox_inches='tight', dpi=300)
    plt.close()
 
    # ------------------ 图 8: residual_distribution.png (预测残差空间高斯性诊断概率密度图) ------------------
    print("--> 8. 正在绘制: residual_distribution.png (图 8)")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
    
    # 子图 A: 场景 D 的残差 KDE 曲线
    res_d = res_dict["D"]
    Z_test_raw_d = data_dict["D"]['Z_test'][:, 0]
    
    err_uks_d = Z_test_raw_d - res_d['Z_pred_uks']
    err_ok_d = Z_test_raw_d - res_d['Z_pred_ok']
    err_mlp_d = Z_test_raw_d - res_d['Z_pred_mlp']
    
    kde_uks_d = stats.gaussian_kde(err_uks_d)
    kde_ok_d = stats.gaussian_kde(err_ok_d)
    kde_mlp_d = stats.gaussian_kde(err_mlp_d)
    
    x_range_d = np.linspace(-2.5, 2.5, 300)
    axes[0].plot(x_range_d, kde_uks_d(x_range_d), label='UKS-DGL (Ours)', color='red', linewidth=2.0)
    axes[0].plot(x_range_d, kde_ok_d(x_range_d), label='Ordinary Kriging (OK)', color='blue', linestyle='-.', linewidth=1.5)
    axes[0].plot(x_range_d, kde_mlp_d(x_range_d), label='纯 MLP 基线', color='green', linestyle=':', linewidth=1.5)
    
    norm_pdf_d = stats.norm.pdf(x_range_d, loc=0, scale=1.0)
    axes[0].plot(x_range_d, norm_pdf_d, label='标准正态参考 $N(0, 1)$', color='gray', linestyle='--', alpha=0.7)
    
    axes[0].set_title("A. 场景 D（典型非平稳随机场）预测残差 KDE 密度分布", fontweight='bold')
    axes[0].set_xlabel("预测残差 ($Z_{true} - Z_{pred}$)")
    axes[0].set_ylabel("概率密度")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    
    # 子图 B: 全场景 (A-F) 汇总残差 KDE 曲线
    all_err_uks = []
    all_err_ok = []
    all_err_mlp = []
    for d in d_names:
        res_temp = res_dict[d]
        Z_test_raw_temp = data_dict[d]['Z_test'][:, 0]
        all_err_uks.extend(Z_test_raw_temp - res_temp['Z_pred_uks'])
        all_err_ok.extend(Z_test_raw_temp - res_temp['Z_pred_ok'])
        all_err_mlp.extend(Z_test_raw_temp - res_temp['Z_pred_mlp'])
        
    all_err_uks = np.array(all_err_uks)
    all_err_ok = np.array(all_err_ok)
    all_err_mlp = np.array(all_err_mlp)
    
    kde_uks_all = stats.gaussian_kde(all_err_uks)
    kde_ok_all = stats.gaussian_kde(all_err_ok)
    kde_mlp_all = stats.gaussian_kde(all_err_mlp)
    
    x_range_all = np.linspace(-2.5, 2.5, 300)
    axes[1].plot(x_range_all, kde_uks_all(x_range_all), label='UKS-DGL (Ours)', color='red', linewidth=2.0)
    axes[1].plot(x_range_all, kde_ok_all(x_range_all), label='Ordinary Kriging (OK)', color='blue', linestyle='-.', linewidth=1.5)
    axes[1].plot(x_range_all, kde_mlp_all(x_range_all), label='纯 MLP 基线', color='green', linestyle=':', linewidth=1.5)
    
    norm_pdf_all = stats.norm.pdf(x_range_all, loc=0, scale=1.0)
    axes[1].plot(x_range_all, norm_pdf_all, label='标准正态参考 $N(0, 1)$', color='gray', linestyle='--', alpha=0.7)
    
    axes[1].set_title("B. 全局六场景 (A-F) 汇总测试集预测残差 KDE 密度分布", fontweight='bold')
    axes[1].set_xlabel("预测残差 ($Z_{true} - Z_{pred}$)")
    axes[1].set_ylabel("概率密度")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')
    
    plt.suptitle("各模型插值残差概率密度空间高斯性诊断对比图 (图 8)\n(Interpolation Residual Probability Density (KDE) and Gaussianity Diagnostics, Fig 8)", fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(f'{output_dir}/residual_distribution.png', bbox_inches='tight', dpi=300)
    plt.savefig('results/plots/residual_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("--> [绘图完成] 8个高清晰度学术图表已经全部生成并保存。")

    
if __name__ == '__main__':
    main()

