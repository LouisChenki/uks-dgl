# -*- coding: utf-8 -*-
"""
真实数据集一键基线计算与 Ours (UKS-DGL) HPO 调优总控脚本 (修正评估维度版)
将测试与验证估计的预测坐标转换为 Batch 并行广播轴，符合可微多通道协同求解器的原始设计意图。
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

# 将 src 和根目录加入路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baselines import OrdinaryKriging, UniversalKriging, CoKriging, train_mlp, compute_morans_i
from dknn import train_dknn_baseline
from deepkriging import train_deepkriging_baseline
from model import UKSModel
from train_eval import compute_joint_losses, HomoscedasticLossWeighting

# 忽略 Optuna 详细日志，使输出整洁
optuna.logging.set_verbosity(optuna.logging.WARNING)

def make_real_self_supervised_dataset(coords, Z, cov, num_samples, n_obs_points):
    """
    多通道自监督留多训练样本构建 (同时抽取空间坐标、外部协变量与多通道物理观测值)。
    """
    n_total = len(coords)
    U_obs_list, Z_obs_list, U_pred_list, Z_pred_list = [], [], [], []
    X_obs_list, X_pred_list = [], []
    for _ in range(num_samples):
        idx = np.random.choice(n_total, size=n_obs_points + 1, replace=False)
        obs_idx = idx[:n_obs_points]
        pred_idx = idx[n_obs_points:]
        
        U_obs_list.append(coords[obs_idx])
        Z_obs_list.append(Z[obs_idx])
        X_obs_list.append(cov[obs_idx])
        
        U_pred_list.append(coords[pred_idx])
        Z_pred_list.append(Z[pred_idx])
        X_pred_list.append(cov[pred_idx])
        
    return (np.array(U_obs_list), np.array(Z_obs_list), np.array(X_obs_list),
            np.array(U_pred_list), np.array(Z_pred_list), np.array(X_pred_list))

def train_real_model(model, coords_train, Z_train, cov_train, epochs, lr, lambda_flow, lambda_geo, device):
    """
    自监督空间重构训练循环，支持课程学习与同方差自适应权重
    """
    N = len(coords_train)
    n_obs_points = int(0.8 * N) if N <= 150 else N - 20
    
    loss_weighting_layer = HomoscedasticLossWeighting().to(device)
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'weight_decay': 1e-4},
        {'params': loss_weighting_layer.parameters(), 'lr': lr * 0.5}
    ], lr=lr)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    
    num_samples = 400
    u_obs_np, z_obs_np, x_obs_np, u_pred_np, z_pred_np, x_pred_np = make_real_self_supervised_dataset(
        coords_train, Z_train, cov_train, num_samples, n_obs_points
    )
    
    u_obs = torch.tensor(u_obs_np, dtype=torch.float32, device=device)
    z_obs = torch.tensor(z_obs_np, dtype=torch.float32, device=device)
    x_obs = torch.tensor(x_obs_np, dtype=torch.float32, device=device)
    u_pred = torch.tensor(u_pred_np, dtype=torch.float32, device=device)
    z_pred = torch.tensor(z_pred_np, dtype=torch.float32, device=device)
    x_pred = torch.tensor(x_pred_np, dtype=torch.float32, device=device)
    
    batch_size = 64
    num_batches = num_samples // batch_size
    
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(num_samples, device=device)
        
        for b in range(num_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            b_u_obs = u_obs[batch_idx]
            b_z_obs = z_obs[batch_idx]
            b_x_obs = x_obs[batch_idx]
            b_u_pred = u_pred[batch_idx]
            b_z_pred = z_pred[batch_idx]
            b_x_pred = x_pred[batch_idx]
            
            H_obs = model.sce(b_u_obs)
            
            optimizer.zero_grad()
            loss, _, _, _, _ = compute_joint_losses(
                model, b_z_obs, b_u_obs, b_u_pred, b_x_obs, b_x_pred, b_z_pred, H_obs,
                lambda_flow=lambda_flow, lambda_geo=lambda_geo, 
                epoch=epoch, loss_weighting_layer=loss_weighting_layer,
                switch_epoch=epochs // 3
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 0.0
    return mae, rmse, r2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> [云端真实数据管线启动] 运行设备: {device}")
    
    data_dir = "data/real"
    output_base_dir = "results_real"
    os.makedirs(output_base_dir, exist_ok=True)
    
    datasets = ["meuse", "california"]
    
    for d_name in datasets:
        print(f"\n=================== 正在计算真实数据集: {d_name.upper()} ===================")
        d_dir = os.path.join(output_base_dir, d_name)
        os.makedirs(d_dir, exist_ok=True)
        
        # 1. 加载预处理好的数据
        data_path = os.path.join(data_dir, f"{d_name}_processed.npz")
        if not os.path.exists(data_path):
            print(f"错误: 找不到数据集文件 {data_path}，请先在本地运行 prepare_real_data.py。")
            continue
            
        data = np.load(data_path)
        coords_train = data['coords_train']
        Z_train_raw = data['Z_train']
        cov_train = data['cov_train']
        coords_test = data['coords_test']
        Z_test_raw = data['Z_test']
        cov_test = data['cov_test']
        
        mean_Z = data['mean_Z']
        std_Z = data['std_Z']
        
        N_train = len(coords_train)
        
        # 标准化主变量
        Z_train = (Z_train_raw - mean_Z) / std_Z
        Z_test = (Z_test_raw - mean_Z) / std_Z
        
        metrics_dict = {}
        
        # 2. 对比基线模型一键式计算 (超参数固定，不参与调优)
        print("--> [基线计算] 1/6 正在拟合 OK...")
        ok_model = OrdinaryKriging(sigma_sq=0.5, l_corr=0.15, nugget=0.01)
        ok_model.fit(coords_train, Z_train[:, 0], use_mle=True)
        Z_pred_ok_scaled, _ = ok_model.predict(coords_test)
        Z_pred_ok = Z_pred_ok_scaled * std_Z[0] + mean_Z[0]
        mae, rmse, r2 = compute_metrics(Z_test_raw[:, 0], Z_pred_ok)
        moran = compute_morans_i(coords_test, Z_test_raw[:, 0] - Z_pred_ok)
        metrics_dict["Ordinary Kriging"] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MoranI": moran}
        
        print("--> [基线计算] 2/6 正在拟合 UK (一阶线性趋势 F=[1,x,y])...")
        uk_model = UniversalKriging(sigma_sq=0.5, l_corr=0.15, nugget=0.01)
        uk_model.fit(coords_train, Z_train[:, 0], use_mle=True)
        Z_pred_uk_scaled, _ = uk_model.predict(coords_test)
        Z_pred_uk = Z_pred_uk_scaled * std_Z[0] + mean_Z[0]
        mae, rmse, r2 = compute_metrics(Z_test_raw[:, 0], Z_pred_uk)
        moran = compute_morans_i(coords_test, Z_test_raw[:, 0] - Z_pred_uk)
        metrics_dict["Universal Kriging"] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MoranI": moran}
        
        Z_pred_ck = np.zeros_like(Z_pred_ok)
        if d_name == "meuse":
            print("--> [基线计算] 3/6 正在拟合 CK (zinc 和 cadmium 协同)...")
            ck_model = CoKriging(l1=0.15, l2=0.15, nugget1=0.01, nugget2=0.01)
            ck_model.fit(coords_train, Z_train, use_mle=True)
            Z_pred_ck_scaled, _ = ck_model.predict(coords_test)
            Z_pred_ck = Z_pred_ck_scaled * std_Z[0] + mean_Z[0]
            mae, rmse, r2 = compute_metrics(Z_test_raw[:, 0], Z_pred_ck)
            moran = compute_morans_i(coords_test, Z_test_raw[:, 0] - Z_pred_ck)
            metrics_dict["CoKriging"] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MoranI": moran}
        else:
            print("--> [基线计算] 3/6 California Temp 无协变量，跳过 CK。")
            
        print("--> [基线计算] 4/6 正在拟合 MLP Net...")
        Z_pred_mlp_scaled, _ = train_mlp(coords_train, Z_train[:, 0], coords_test, Z_test[:, 0], epochs=350, lr=0.01, device=device)
        Z_pred_mlp = Z_pred_mlp_scaled * std_Z[0] + mean_Z[0]
        mae, rmse, r2 = compute_metrics(Z_test_raw[:, 0], Z_pred_mlp)
        moran = compute_morans_i(coords_test, Z_test_raw[:, 0] - Z_pred_mlp)
        metrics_dict["MLP Network"] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MoranI": moran}
        
        print("--> [基线计算] 5/6 正在拟合 DKNN...")
        Z_train_2ch = Z_train if d_name == "meuse" else np.hstack([Z_train, np.zeros_like(Z_train)])
        Z_test_2ch = Z_test if d_name == "meuse" else np.hstack([Z_test, np.zeros_like(Z_test)])
        Z_pred_dknn_scaled = train_dknn_baseline(coords_train, Z_train_2ch, coords_test, Z_test_2ch, epochs=300, lr=0.0005, device=device)
        Z_pred_dknn = Z_pred_dknn_scaled * std_Z[0] + mean_Z[0]
        mae, rmse, r2 = compute_metrics(Z_test_raw[:, 0], Z_pred_dknn)
        moran = compute_morans_i(coords_test, Z_test_raw[:, 0] - Z_pred_dknn)
        metrics_dict["DKNN"] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MoranI": moran}
        
        print("--> [基线计算] 6/6 正在拟合 DeepKriging...")
        Z_pred_dk_scaled = train_deepkriging_baseline(coords_train, Z_train_2ch, coords_test, Z_test_2ch, epochs=250, lr=0.002, device=device)
        Z_pred_dk = Z_pred_dk_scaled * std_Z[0] + mean_Z[0]
        mae, rmse, r2 = compute_metrics(Z_test_raw[:, 0], Z_pred_dk)
        moran = compute_morans_i(coords_test, Z_test_raw[:, 0] - Z_pred_dk)
        metrics_dict["DeepKriging"] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MoranI": moran}
        
        # 3. Ours HPO 贝叶斯搜索 (以 80% Train, 20% Val 划分)
        print("--> [Ours HPO] 正在启动 Optuna 贝叶斯超参数寻优...")
        np.random.seed(42)
        indices = np.arange(N_train)
        np.random.shuffle(indices)
        val_split = int(0.8 * N_train)
        hpo_train_idx = indices[:val_split]
        hpo_val_idx = indices[val_split:]
        
        in_dim = 2 if d_name == "meuse" else 1
        
        hpo_coords_train = coords_train[hpo_train_idx]
        hpo_Z_train = Z_train[hpo_train_idx]
        hpo_cov_train = cov_train[hpo_train_idx]
        
        # 验证集大小 M_val
        M_val = len(hpo_val_idx)
        
        # 将训练数据广播 expand 匹配 Batch 轴形状 [M_val, N_sub, ...]
        U_train_hpo_batch = torch.tensor(hpo_coords_train, dtype=torch.float32, device=device).unsqueeze(0).expand(M_val, -1, -1)
        Z_train_hpo_batch = torch.tensor(hpo_Z_train, dtype=torch.float32, device=device).unsqueeze(0).expand(M_val, -1, -1)
        X_train_hpo_batch = torch.tensor(hpo_cov_train, dtype=torch.float32, device=device).unsqueeze(0).expand(M_val, -1, -1)
        
        # 将验证点作为 Batch 轴传入，每个 batch 预测单个点 [M_val, 1, ...]
        U_val_hpo_batch = torch.tensor(coords_train[hpo_val_idx], dtype=torch.float32, device=device).unsqueeze(1)
        X_val_hpo_batch = torch.tensor(cov_train[hpo_val_idx], dtype=torch.float32, device=device).unsqueeze(1)
        Z_val_hpo = torch.tensor(Z_train[hpo_val_idx], dtype=torch.float32, device=device) # [M_val, q]
        
        def objective(trial):
            lr = trial.suggest_float("lr", 2e-4, 3e-3, log=True)
            lambda_flow = trial.suggest_float("lambda_flow", 2e-4, 3e-3, log=True)
            lambda_geo = trial.suggest_float("lambda_geo", 1e-5, 8e-5, log=True)
            nugget_eps = trial.suggest_float("nugget_eps", 5e-7, 5e-5, log=True)
            flow_hidden = trial.suggest_int("flow_hidden", 16, 48, step=16)
            num_flow_layers = trial.suggest_int("num_flow_layers", 2, 4, step=2)
            
            model = UKSModel(
                in_dim=in_dim,
                flow_hidden_dim=flow_hidden,
                num_flow_layers=num_flow_layers,
                embed_dim=16,
                rff_sigma=10.0,
                kernel_hidden_dim=32,
                latent_dim=8,
                eps=nugget_eps,
                trend_type="external"
            ).to(device)
            
            try:
                # 快速评估跑 100 epochs
                train_real_model(
                    model, hpo_coords_train, hpo_Z_train, hpo_cov_train, 
                    epochs=100, lr=lr, lambda_flow=lambda_flow, lambda_geo=lambda_geo, device=device
                )
                
                model.eval()
                with torch.no_grad():
                    Z_val_pred_scaled = model(Z_train_hpo_batch, U_train_hpo_batch, U_val_hpo_batch, X_train_hpo_batch, X_val_hpo_batch)
                    y_true = Z_val_hpo[:, 0].cpu().numpy()
                    y_pred = Z_val_pred_scaled[:, 0, 0].cpu().numpy()
                    
                _, _, r2_val = compute_metrics(y_true, y_pred)
                if np.isnan(r2_val) or r2_val < -10.0:
                    return -10.0
                return r2_val
            except Exception as e:
                return -10.0
                
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        
        best_p = study.best_params
        print(f"--> [Ours HPO 完成] 最佳超参: {best_p}, 最佳验证集 R2: {study.best_value:.4f}")
        
        # 4. 终训 300 epochs 并进行 Batch 预测
        print("--> [Ours 终训] 使用最优超参在完整训练集上拟合...")
        M_test = len(coords_test)
        
        U_train_all_batch = torch.tensor(coords_train, dtype=torch.float32, device=device).unsqueeze(0).expand(M_test, -1, -1)
        Z_train_all_batch = torch.tensor(Z_train, dtype=torch.float32, device=device).unsqueeze(0).expand(M_test, -1, -1)
        X_train_all_batch = torch.tensor(cov_train, dtype=torch.float32, device=device).unsqueeze(0).expand(M_test, -1, -1)
        
        U_test_all_batch = torch.tensor(coords_test, dtype=torch.float32, device=device).unsqueeze(1)
        X_test_all_batch = torch.tensor(cov_test, dtype=torch.float32, device=device).unsqueeze(1)
        
        best_model = UKSModel(
            in_dim=in_dim,
            flow_hidden_dim=best_p["flow_hidden"],
            num_flow_layers=best_p["num_flow_layers"],
            embed_dim=16,
            rff_sigma=10.0,
            kernel_hidden_dim=32,
            latent_dim=8,
            eps=best_p["nugget_eps"],
            trend_type="external"
        ).to(device)
        
        train_real_model(
            best_model, coords_train, Z_train, cov_train,
            epochs=300, lr=best_p["lr"], lambda_flow=best_p["lambda_flow"], lambda_geo=best_p["lambda_geo"], device=device
        )
        
        # 5. 测试集并行估计
        best_model.eval()
        with torch.no_grad():
            Z_test_pred_scaled = best_model(Z_train_all_batch, U_train_all_batch, U_test_all_batch, X_train_all_batch, X_test_all_batch)
            Z_pred_uks = Z_test_pred_scaled[:, 0, 0].cpu().numpy() * std_Z[0] + mean_Z[0]
            
            Z_hat_unbiased_t, Z_var_unbiased_t = best_model.predict_with_uncertainty(
                Z_train_all_batch, U_train_all_batch, U_test_all_batch, X_train_all_batch, X_test_all_batch, n_samples_mc=100
            )
            Z_var_uks = Z_var_unbiased_t[:, 0, 0].cpu().numpy() * (std_Z[0] ** 2)
            
            # 提取流高斯潜变量
            Y_train_flow, _ = best_model.flow(Z_train_all_batch)
            Y_train_flow = Y_train_flow[0].cpu().numpy()
            Y_test_flow, _ = best_model.flow(Z_test_pred_scaled)
            Y_test_flow = Y_test_flow[:, 0].cpu().numpy()
            
        # 6. 计算最终精度指标并保存
        mae, rmse, r2 = compute_metrics(Z_test_raw[:, 0], Z_pred_uks)
        moran = compute_morans_i(coords_test, Z_test_raw[:, 0] - Z_pred_uks)
        metrics_dict["UKS-DGL (Ours)"] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MoranI": moran}
        
        print(f"--> [Ours 终评结果] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MoranI: {moran:.4f}")
        
        metrics_path = os.path.join(d_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
            
        torch.save(best_model.state_dict(), os.path.join(d_dir, "uks_model.pth"))
        with open(os.path.join(d_dir, "best_config.json"), 'w', encoding='utf-8') as f:
            json.dump(best_p, f, indent=4, ensure_ascii=False)
            
        np.savez(
            os.path.join(d_dir, "experiment_results.npz"),
            coords_test=coords_test,
            Z_test=Z_test_raw[:, 0],
            Z_pred_ok=Z_pred_ok,
            Z_pred_uk=Z_pred_uk,
            Z_pred_ck=Z_pred_ck,
            Z_pred_mlp=Z_pred_mlp,
            Z_pred_dknn=Z_pred_dknn,
            Z_pred_dk=Z_pred_dk,
            Z_pred_uks=Z_pred_uks,
            Z_var_uks=Z_var_uks,
            Y_train_flow=Y_train_flow,
            Y_test_flow=Y_test_flow,
            mean_Z=mean_Z,
            std_Z=std_Z
        )
        print(f"--> [数据归档成功] {d_name.upper()} 结果已完美存入 {d_dir}。")

if __name__ == "__main__":
    main()
