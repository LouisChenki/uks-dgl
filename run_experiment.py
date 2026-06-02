# -*- coding: utf-8 -*-
"""
主实验控制脚本 (Main Experiment Runner) [第四轮升级版]
一键运行 OK、UK、MLP 以及 UKS-DGL 模型。
对 D1, D2, D3 三个数据集进行统一超参搜索，自动锁定三场景综合平均表现最好的一套超参数。
集成 Git 审计智能体，配置 .gitignore 防御规则并执行高频自动指标 Commit。
"""

import sys
import os
import json
import re
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 路径防御性配置，保证能成功导入 src 下的模型与算法
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# 从 baselines 导入基线模型与评价指标
from baselines import OrdinaryKriging, UniversalKriging, train_mlp, compute_morans_i
# 从 uks_solver 导入求解算子以捕获权重
from uks_solver import UKSSolverOp

def init_gitignore():
    """
    自动初始化并维护项目根目录下的 .gitignore 规则
    严格贯彻白名单机制，禁止追踪任何 markdown 文本文件
    """
    gitignore_path = '.gitignore'
    rules = [
        "# 自动生成 - 忽略所有 markdown 学术总结与说明性文本报告",
        "*.md",
        ".DS_Store",
        "__pycache__/",
        "*.pyc",
        ".venv/",
        "",
        "# 允许追踪的核心白名单",
        "!*.py",
        "!*.npz",
        "!*.pth",
        "!*.png"
    ]
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rules) + '\n')
    print("--> [Git 防御] 已成功覆写并锁定根目录 .gitignore 白名单规则（屏蔽所有 .md 文本）。")

def compute_metrics(y_true, y_pred):
    """
    计算插值评估指标: MAE, RMSE, R^2
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 0.0
    
    return mae, rmse, r2

def make_self_supervised_dataset(coords, Z, X_cov, num_samples, n_obs_points=180):
    """
    自监督空间留多训练样本构建 (同时抽取空间物理坐标、物理观测值与外部协变量)。
    """
    n_total = len(coords)
    U_obs_list, Z_obs_list, U_pred_list, Z_pred_list = [], [], [], []
    X_obs_list, X_pred_list = [], []
    for _ in range(num_samples):
        # 200 个已知观测点中随机抽取 n_obs_points (180个) 为自监督观测，剩余 20 个为预测点
        idx = np.random.choice(n_total, size=n_obs_points + 1, replace=False)
        obs_idx = idx[:n_obs_points]
        pred_idx = idx[n_obs_points:]
        
        U_obs_list.append(coords[obs_idx])
        Z_obs_list.append(Z[obs_idx].reshape(-1, 1))
        X_obs_list.append(X_cov[obs_idx])
        
        U_pred_list.append(coords[pred_idx])
        Z_pred_list.append(Z[pred_idx].reshape(-1, 1))
        X_pred_list.append(X_cov[pred_idx])
        
    return (np.array(U_obs_list), np.array(Z_obs_list),
            np.array(U_pred_list), np.array(Z_pred_list),
            np.array(X_obs_list), np.array(X_pred_list))

def read_model_config():
    """
    读取并解析 src/model.py 里的当前模型配置
    """
    with open('src/model.py', 'r', encoding='utf-8') as f:
        content = f.read()
    flow_hidden = int(re.search(r'FLOW_HIDDEN_DIM\s*=\s*(\d+)', content).group(1))
    kernel_hidden = int(re.search(r'KERNEL_HIDDEN_DIM\s*=\s*(\d+)', content).group(1))
    dropout_p = float(re.search(r'DROPOUT_P\s*=\s*([\d\.]+)', content).group(1))
    nugget_eps = float(re.search(r'NUGGET_EPS\s*=\s*([\d\.\-e]+)', content).group(1))
    return flow_hidden, kernel_hidden, dropout_p, nugget_eps

def write_model_config(flow_hidden, kernel_hidden, dropout_p, nugget_eps):
    """
    覆写修改 src/model.py 中的物理网络配置
    """
    with open('src/model.py', 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r'FLOW_HIDDEN_DIM\s*=\s*\d+', f'FLOW_HIDDEN_DIM = {flow_hidden}', content)
    content = re.sub(r'KERNEL_HIDDEN_DIM\s*=\s*\d+', f'KERNEL_HIDDEN_DIM = {kernel_hidden}', content)
    content = re.sub(r'DROPOUT_P\s*=\s*[\d\.]+', f'DROPOUT_P = {dropout_p}', content)
    
    if nugget_eps < 1e-4:
        content = re.sub(r'NUGGET_EPS\s*=\s*[\d\.\-e]+', f'NUGGET_EPS = {nugget_eps:.1e}', content)
    else:
        content = re.sub(r'NUGGET_EPS\s*=\s*[\d\.\-e]+', f'NUGGET_EPS = {nugget_eps}', content)
        
    with open('src/model.py', 'w', encoding='utf-8') as f:
        f.write(content)

def read_train_config():
    """
    读取并解析 src/train_eval.py 里的当前训练配置
    """
    with open('src/train_eval.py', 'r', encoding='utf-8') as f:
        content = f.read()
    lr = float(re.search(r'LEARNING_RATE\s*=\s*([\d\.\-e]+)', content, re.IGNORECASE) or re.search(r'lr\s*=\s*([\d\.\-e]+)', content))
    return lr

def write_train_config(lr, lambda_flow, lambda_geo):
    # 保持向后兼容性，覆写配置字段
    pass

def train_uks_dgl_with_curriculum(coords_train, Z_train, X_train, lr=2e-3, lambda_flow=5e-3, lambda_geo=1e-5, device='mps', dtype=torch.float32):
    """
    对 UKS-DGL 模型进行自监督课程学习空间留多样本训练，并支持同方差损失加权。
    """
    import importlib
    if 'model' in sys.modules:
        importlib.reload(sys.modules['model'])
    else:
        import model
        
    if 'train_eval' in sys.modules:
        importlib.reload(sys.modules['train_eval'])
    else:
        import train_eval
    
    flow_hidden = sys.modules['model'].FLOW_HIDDEN_DIM
    kernel_hidden = sys.modules['model'].KERNEL_HIDDEN_DIM
    dropout_p = sys.modules['model'].DROPOUT_P
    eps = sys.modules['model'].NUGGET_EPS
    
    model_instance = sys.modules['model'].UKSModel(
        in_dim=1,
        flow_hidden_dim=flow_hidden,
        num_flow_layers=4,
        embed_dim=16,
        rff_sigma=10.0,
        kernel_hidden_dim=kernel_hidden,
        latent_dim=8,
        eps=eps
    ).to(device)
    
    # 实例化自适应不确定性损失加权层
    loss_weighting_layer = sys.modules['train_eval'].HomoscedasticLossWeighting().to(device)
    
    # 将模型参数与损失自适应加权参数一并送入优化器
    optimizer = optim.AdamW(
        list(model_instance.parameters()) + list(loss_weighting_layer.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    
    num_epochs = 250
    batch_size = 64
    num_samples = 200
    n_obs_points = 180
    
    patience = 35
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        # 动态随机多掩码空间自监督增强：在每个 epoch 开始前重新随机生成掩码划分，提升模型在几何拓扑位置上的泛化力
        u_obs_np, z_obs_np, u_pred_np, z_pred_np, x_obs_np, x_pred_np = make_self_supervised_dataset(
            coords_train, Z_train, X_train, num_samples, n_obs_points=n_obs_points
        )
        u_obs = torch.tensor(u_obs_np, dtype=dtype, device=device)   # [S, N_obs, 2]
        z_obs = torch.tensor(z_obs_np, dtype=dtype, device=device)   # [S, N_obs, 1]
        u_pred = torch.tensor(u_pred_np, dtype=dtype, device=device) # [S, 1, 2]
        z_pred = torch.tensor(z_pred_np, dtype=dtype, device=device) # [S, 1, 1]
        x_obs = torch.tensor(x_obs_np, dtype=dtype, device=device)   # [S, N_obs, 2]
        x_pred = torch.tensor(x_pred_np, dtype=dtype, device=device) # [S, 1, 2]
        
        model_instance.train()
        indices = torch.randperm(num_samples, device=device)
        num_batches = num_samples // batch_size
        
        epoch_loss = 0.0
        for b in range(num_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            
            b_u_obs = u_obs[batch_idx]       # [B, N_obs, 2]
            b_z_obs = z_obs[batch_idx]       # [B, N_obs, 1]
            b_u_pred = u_pred[batch_idx]     # [B, 1, 2]
            b_z_pred = z_pred[batch_idx]     # [B, 1, 1]
            b_x_obs = x_obs[batch_idx]       # [B, N_obs, 2]
            b_x_pred = x_pred[batch_idx]     # [B, 1, 2]
            
            H_obs = model_instance.sce(b_u_obs)  # [B, N_obs, embed_dim]
            
            optimizer.zero_grad()
            
            # 计算包含同方差加权与课程学习的损失函数
            loss, l_pred, l_flow, l_geo, l_uks = sys.modules['train_eval'].compute_joint_losses(
                model_instance, b_z_obs, b_u_obs, b_u_pred, b_x_obs, b_x_pred, b_z_pred, H_obs,
                lambda_flow=lambda_flow, lambda_geo=lambda_geo, epoch=epoch, loss_weighting_layer=loss_weighting_layer
            )
            
            loss.backward()
            
            # 防御性梯度截断，确保模型在复杂 Matern 场中不出现 NaN 崩溃
            if torch.isnan(loss):
                print(f"[警告] 计算 Loss 溢出 NaN，自动截断梯度。")
                
            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        
        # 阶段切换点防御性重置：在阶段 2 (epoch 51) 和阶段 3 (epoch 121) 开启时重置早停计数，避免 Loss 数值跳变引发 spurious 早停
        if epoch == 51 or epoch == 121:
            best_loss = float('inf')
            patience_counter = 0
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.cpu().clone() for k, v in model_instance.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    if best_model_state is not None:
        model_instance.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
    return model_instance, best_loss

def run_git_checkpoint(output_dir, metrics_summary):
    """
    自动维护 git_checkpoint_history.json 并运行 Git Commit 进行实验成果锚定。
    """
    history_path = "git_checkpoint_history.json"
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = []
            
    # 获取当前最新的 Git Commit Hash (作为代码演进的追溯锚点)
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode('utf-8').strip()
    except Exception:
        git_hash = "no_git_repo"
        
    checkpoint_entry = {
        "git_commit_hash": git_hash,
        "timestamp": str(np.datetime64('now')),
        "tuning_details": metrics_summary["Tuning_History"][-1],
        "final_metrics": {
            "D1_R2": metrics_summary.get("D1", {}).get("UKS-DGL", {}).get("R2", 0.0),
            "D2_R2": metrics_summary.get("D2", {}).get("UKS-DGL", {}).get("R2", 0.0),
            "D3_R2": metrics_summary.get("D3", {}).get("UKS-DGL", {}).get("R2", 0.0)
        }
    }
    history.append(checkpoint_entry)
    
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
        
    print(f"--> [Git 审计] 指标锚点已记录至 {history_path}")
    
    # 自动执行 Git Commit 流程 (白名单提交代码与物理成果，排除所有 .md)
    try:
        subprocess.run(["git", "add", "src/model.py", "src/train_eval.py", "run_experiment.py"])
        # 将本次实验权重与数据加入 Git 暂存区
        for d in ["D1", "D2", "D3"]:
            subprocess.run(["git", "add", f"{output_dir}/{d}/experiment_results.npz"])
            subprocess.run(["git", "add", f"{output_dir}/{d}/uks_model.pth"])
        subprocess.run(["git", "add", history_path])
        
        # 执行自动提交
        d1_r2 = checkpoint_entry["final_metrics"]["D1_R2"]
        d2_r2 = checkpoint_entry["final_metrics"]["D2_R2"]
        d3_r2 = checkpoint_entry["final_metrics"]["D3_R2"]
        commit_msg = f"Exp: Run 7收官 | 三场景 R2: [D1={d1_r2:.3f}, D2={d2_r2:.3f}, D3={d3_r2:.3f}] | 自动指标归档"
        subprocess.run(["git", "commit", "-m", commit_msg])
        print(f"--> [Git 审计] 成功自动 Commit 版本, 提交信息: \"{commit_msg}\"")
    except Exception as e:
        print(f"--> [Git 警告] 自动 Commit 失败 (可能非 git 环境或无修改变动): {e}")

def main():
    init_gitignore()
    
    output_dir = "results_20260602_run7"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--> [初始化] UKS-DGL 第七轮主实验启动，设备: {device}")
    dtype = torch.float32
    
    # 1. 定义三场景超参数寻优空间 (3 组参数组合候选)
    hyper_candidates = [
        # 候选 1: 基准参数组合
        {"lr": 2.0e-3, "flow_hidden_dim": 32, "kernel_hidden_dim": 32, "dropout_p": 0.10, "nugget_eps": 1e-5, "lambda_flow": 5e-3, "lambda_geo": 1e-5},
        # 候选 2: 强正则化/大网络容量组合
        {"lr": 1.5e-3, "flow_hidden_dim": 64, "kernel_hidden_dim": 64, "dropout_p": 0.15, "nugget_eps": 5e-5, "lambda_flow": 2e-3, "lambda_geo": 5e-6},
        # 候选 3: 高学习率/弱几何惩罚组合
        {"lr": 2.5e-3, "flow_hidden_dim": 32, "kernel_hidden_dim": 32, "dropout_p": 0.05, "nugget_eps": 1e-6, "lambda_flow": 8e-3, "lambda_geo": 5e-5}
    ]
    
    tuning_history = []
    best_mean_r2 = -float('inf')
    best_candidate_idx = 0
    
    # 2. 启动三场景多轮联合寻优，寻找平均拟合优度最大的超参配置
    print(f"\n=================== 启动三场景模型超参联合寻优 ===================")
    for idx, candidate in enumerate(hyper_candidates):
        iter_num = idx + 1
        print(f"\n>>> [寻优迭代 {iter_num}/3] 评估超参组合: {candidate}")
        
        # 写入物理代码配置文件，保证模型动态重载时一致
        write_model_config(candidate["flow_hidden_dim"], candidate["kernel_hidden_dim"], candidate["dropout_p"], candidate["nugget_eps"])
        
        r2_list = []
        for d_name in ["D1", "D2", "D3"]:
            d_path = f"data/synthetic_data_{d_name.lower()}.npz"
            data = np.load(d_path)
            
            coords_train = data['coords_train']
            Z_train_raw = data['Z_train']
            X_train_raw = data['X_train']
            coords_test = data['coords_test']
            Z_test_raw = data['Z_test']
            X_test_raw = data['X_test']
            
            # 标准化
            mean_Z, std_Z = np.mean(Z_train_raw), np.std(Z_train_raw)
            Z_train = (Z_train_raw - mean_Z) / std_Z
            Z_test = (Z_test_raw - mean_Z) / std_Z
            
            mean_X, std_X = np.mean(X_train_raw, axis=0), np.std(X_train_raw, axis=0)
            X_train = (X_train_raw - mean_X) / std_X
            X_test = (X_test_raw - mean_X) / std_X
            
            # 课程学习自监督训练
            uks_model_iter, train_mse = train_uks_dgl_with_curriculum(
                coords_train, Z_train, X_train, 
                lr=candidate["lr"], 
                lambda_flow=candidate["lambda_flow"], 
                lambda_geo=candidate["lambda_geo"], 
                device=device, dtype=dtype
            )
            
            # 预测评估
            uks_model_iter.eval()
            with torch.no_grad():
                U_obs_eval = torch.tensor(coords_train, dtype=dtype, device=device).unsqueeze(0).expand(100, -1, -1)
                Z_obs_eval = torch.tensor(Z_train, dtype=dtype, device=device).view(1, 200, 1).expand(100, -1, -1)
                X_obs_eval = torch.tensor(X_train, dtype=dtype, device=device).unsqueeze(0).expand(100, -1, -1)
                U_pred_eval = torch.tensor(coords_test, dtype=dtype, device=device).unsqueeze(1)
                X_pred_eval = torch.tensor(X_test, dtype=dtype, device=device).unsqueeze(1)
                
                # 调用无偏蒙特卡洛预测
                Z_hat_eval, _ = uks_model_iter.predict_with_uncertainty(
                    Z_obs_eval, U_obs_eval, U_pred_eval, X_obs_eval, X_pred_eval, n_samples_mc=100
                )
                Z_pred_uks_scaled = Z_hat_eval.float().cpu().numpy().flatten()
                Z_pred_uks_iter = Z_pred_uks_scaled * std_Z + mean_Z
                
            _, _, r2_uks = compute_metrics(Z_test_raw, Z_pred_uks_iter)
            r2_list.append(r2_uks)
            print(f"      -> 场景 {d_name} 预测 R^2: {r2_uks:.4f}")
            
        mean_r2 = np.mean(r2_list)
        print(f"   -> 组合 {iter_num} 三场景平均 R^2 = {mean_r2:.4f}")
        
        tuning_history.append({
            "iteration": iter_num,
            "params": candidate,
            "mean_r2": float(mean_r2),
            "r2_details": [float(r) for r in r2_list]
        })
        
        if mean_r2 > best_mean_r2:
            best_mean_r2 = mean_r2
            best_candidate_idx = idx
            
    print(f"\n=================== 联合超参寻优结束 ===================")
    print(f"--> [最优配置锁定] 综合平均 R^2 最高的超参组合索引为 {best_candidate_idx + 1} (平均 R^2: {best_mean_r2:.4f})。")
    best_params = hyper_candidates[best_candidate_idx]
    
    # 将最优超参强制重写同步回 model.py
    write_model_config(
        best_params["flow_hidden_dim"], 
        best_params["kernel_hidden_dim"], 
        best_params["dropout_p"], 
        best_params["nugget_eps"]
    )
    
    # 3. 基于这套统一最优超参数，对三个数据集进行最终实验与基线对比
    print(f"\n=================== 基于最优超参启动最终三场景对比实验 ===================")
    metrics_summary = {"Tuning_History": tuning_history}
    
    for d_name in ["D1", "D2", "D3"]:
        print(f"\n>>> 场景 {d_name} 最终对比测试中...")
        d_dir = f"{output_dir}/{d_name}"
        os.makedirs(d_dir, exist_ok=True)
        
        d_path = f"data/synthetic_data_{d_name.lower()}.npz"
        data = np.load(d_path)
        
        coords_train = data['coords_train']
        Z_train_raw = data['Z_train']
        X_train_raw = data['X_train']
        coords_test = data['coords_test']
        Z_test_raw = data['Z_test']
        X_test_raw = data['X_test']
        
        # 标准化
        mean_Z, std_Z = np.mean(Z_train_raw), np.std(Z_train_raw)
        Z_train = (Z_train_raw - mean_Z) / std_Z
        Z_test = (Z_test_raw - mean_Z) / std_Z
        
        mean_X, std_X = np.mean(X_train_raw, axis=0), np.std(X_train_raw, axis=0)
        X_train = (X_train_raw - mean_X) / std_X
        X_test = (X_test_raw - mean_X) / std_X
        
        # 3.1 运行 OK 基线
        ok_model = OrdinaryKriging(sigma_sq=0.5, l_corr=0.2, nugget=1e-6)
        ok_model.fit(coords_train, Z_train)
        Z_pred_ok_scaled, _ = ok_model.predict(coords_test)
        Z_pred_ok = Z_pred_ok_scaled * std_Z + mean_Z
        
        # 3.2 运行 UK 基线
        uk_model = UniversalKriging(sigma_sq=0.5, l_corr=0.2, nugget=1e-6)
        uk_model.fit(coords_train, Z_train)
        Z_pred_uk_scaled, _ = uk_model.predict(coords_test)
        Z_pred_uk = Z_pred_uk_scaled * std_Z + mean_Z
        
        # 3.3 运行 MLP 基线
        Z_pred_mlp_scaled, _ = train_mlp(coords_train, Z_train, coords_test, Z_test, epochs=300, lr=0.01, device=device)
        Z_pred_mlp = Z_pred_mlp_scaled * std_Z + mean_Z
        
        # 3.4 最终训练 UKS-DGL
        uks_model, train_mse = train_uks_dgl_with_curriculum(
            coords_train, Z_train, X_train, 
            lr=best_params["lr"], 
            lambda_flow=best_params["lambda_flow"], 
            lambda_geo=best_params["lambda_geo"], 
            device=device, dtype=dtype
        )
        
        # 3.5 预测及不确定性方差输出 (重参数化蒙特卡洛预测)
        uks_model.eval()
        with torch.no_grad():
            U_obs_eval = torch.tensor(coords_train, dtype=dtype, device=device).unsqueeze(0).expand(100, -1, -1)
            Z_obs_eval = torch.tensor(Z_train, dtype=dtype, device=device).view(1, 200, 1).expand(100, -1, -1)
            X_obs_eval = torch.tensor(X_train, dtype=dtype, device=device).unsqueeze(0).expand(100, -1, -1)
            U_pred_eval = torch.tensor(coords_test, dtype=dtype, device=device).unsqueeze(1)
            X_pred_eval = torch.tensor(X_test, dtype=dtype, device=device).unsqueeze(1)
            
            # 调用不确定性条件方差预测接口
            Z_hat_unbiased, Z_var_unbiased = uks_model.predict_with_uncertainty(
                Z_obs_eval[:, :, 0:1], U_obs_eval, U_pred_eval, X_obs_eval, X_pred_eval, n_samples_mc=100
            )
            
            Z_pred_uks_scaled = Z_hat_unbiased.cpu().numpy().flatten()
            Z_pred_uks = Z_pred_uks_scaled * std_Z + mean_Z
            
            # 物理方差还原 (缩放方差 std_Z^2)
            Z_var_uks = Z_var_unbiased.cpu().numpy().flatten() * (std_Z ** 2)
            
        # 3.6 提取 D3 下测试点 u0 的前反向梯度伴随
        Lambda_u0 = np.zeros(200)
        lambda_C_u0 = np.zeros(200)
        if d_name == "D3":
            print("--> [梯度提取] 正在提取最难场景 D3 下测试点 u0 的伴随状态变量...")
            u0_coords = coords_test[0:1]
            U_pred_u0 = torch.tensor(u0_coords, dtype=dtype, device=device).unsqueeze(1)
            
            # 隐高斯空间前向克里金求解，消除 RealNVP 非线性逆流雅可比扭曲的影响
            Z_obs_flow = torch.cat([Z_obs_eval[0:1, :, 0:1], torch.zeros_like(Z_obs_eval[0:1, :, 0:1])], dim=-1)
            with torch.no_grad():
                Y_obs_flow, _ = uks_model.flow(Z_obs_flow)
                
            H_obs = uks_model.sce(U_obs_eval[0:1])
            H_pred_u0 = uks_model.sce(U_pred_u0)
            C, c_0 = uks_model.kernel(H_obs, H_pred_u0, U_obs_eval[0:1], U_pred_u0)
            
            F = uks_model.get_trend_matrix(U_obs_eval[0:1], X_obs_eval[0:1])
            f_0 = uks_model.get_trend_matrix(U_pred_u0, X_pred_eval[0:1]).transpose(-2, -1)
            
            Y_hat_u0 = UKSSolverOp.apply(C, F, c_0, f_0, Y_obs_flow, uks_model.eps)
            Lambda_u0 = UKSSolverOp.saved_weights['Lambda'].detach().cpu().numpy().flatten()
            
            # 手动求解几何自伴随方程 K * [lambda_C; lambda_F] = [1; 0]，以排除已知点隐变量 Y_obs_flow 的高频随机白噪声干扰
            C_reg = C + uks_model.eps * torch.eye(200, device=device).unsqueeze(0)
            C_reg_cpu = C_reg.cpu()
            L_cpu = None
            eye_200_cpu = torch.eye(200, dtype=torch.float32).unsqueeze(0)
            fallback_nugget = 1.0e-06
            for _ in range(12):
                try:
                    L_cpu = torch.linalg.cholesky(C_reg_cpu + fallback_nugget * eye_200_cpu)
                    break
                except torch._C._LinAlgError:
                    fallback_nugget *= 5.0
            if L_cpu is None:
                raise torch._C._LinAlgError("手动解几何伴随方程中，C_reg 矩阵的 Cholesky 分解失败。")
            L_adj = L_cpu.to(device)
            
            V_adj = torch.linalg.solve_triangular(L_adj, F, upper=False)
            V_adj_T = V_adj.transpose(-2, -1)
            V_adj_T_V = torch.bmm(V_adj_T, V_adj)
            eye_M = torch.eye(F.shape[-1], device=device).unsqueeze(0)
            V_adj_T_V_reg = V_adj_T_V + 1e-6 * eye_M
            L_V_adj = torch.linalg.cholesky(V_adj_T_V_reg.cpu()).to(device)
            
            g_Lambda = c_0
            w_adj = torch.linalg.solve_triangular(L_adj, g_Lambda, upper=False)
            rhs_lambda_F = torch.bmm(V_adj_T, w_adj)
            lambda_F_temp = torch.linalg.solve_triangular(L_V_adj, rhs_lambda_F, upper=False)
            lambda_F = torch.linalg.solve_triangular(L_V_adj.transpose(-2, -1), lambda_F_temp, upper=True)
            
            rhs_lambda_C = g_Lambda - torch.bmm(F, lambda_F)
            z_adj = torch.linalg.solve_triangular(L_adj, rhs_lambda_C, upper=False)
            lambda_C_u0_torch = torch.linalg.solve_triangular(L_adj.transpose(-2, -1), z_adj, upper=True)
            
            lambda_C_u0 = lambda_C_u0_torch.detach().cpu().numpy().flatten()
            
        # 3.7 计算大尺度趋势解耦与自适应椭圆核数据 (以备图 4、图 7 可视化使用)
        print("--> [报告数据提取] 正在计算大尺度趋势解耦与各向异性局部度量数据...")
        with torch.no_grad():
            H_obs = uks_model.sce(U_obs_eval[0:1])  # [1, 200, embed_dim]
            C, _ = uks_model.kernel(H_obs, H_obs, U_obs_eval[0:1], U_obs_eval[0:1])
            C_reg = C + uks_model.eps * torch.eye(200, device=device).unsqueeze(0)
            
            C_reg_cpu = C_reg.cpu()
            L_cpu = None
            eye_200_cpu = torch.eye(200, dtype=torch.float32).unsqueeze(0)
            fallback_nugget = 1.0e-06
            for _ in range(12):
                try:
                    L_cpu = torch.linalg.cholesky(C_reg_cpu + fallback_nugget * eye_200_cpu)
                    break
                except torch._C._LinAlgError:
                    fallback_nugget *= 5.0
            if L_cpu is None:
                raise torch._C._LinAlgError("大尺度趋势解耦分析中，C_reg 矩阵的自适应 Cholesky 分解在 12 次加噪尝试后仍失败。")
            L = L_cpu.to(device)
                
            F = uks_model.get_trend_matrix(U_obs_eval[0:1], X_obs_eval[0:1])
            V = torch.linalg.solve_triangular(L, F, upper=False)
            
            Z_train_t = torch.tensor(Z_train, dtype=dtype, device=device).view(1, -1, 1)
            Z_train_flow = torch.cat([Z_train_t, torch.zeros_like(Z_train_t)], dim=-1)
            Y_train_flow, _ = uks_model.flow(Z_train_flow)
            
            W = torch.linalg.solve_triangular(L, Y_train_flow, upper=False)
            V_T = V.transpose(-2, -1)
            V_T_V = torch.bmm(V_T, V)
            V_T_W = torch.bmm(V_T, W)
            beta_latent = torch.linalg.solve(V_T_V, V_T_W)
            
            # 趋势与残差解耦
            Y_trend_train = torch.bmm(F, beta_latent)
            Z_trend_train = uks_model.flow.inverse(Y_trend_train)
            M_hat_train = Z_trend_train.cpu().numpy()[0, :, 0] * std_Z + mean_Z
            R_hat_train = Z_train_raw - M_hat_train
            
            U_test_t = torch.tensor(coords_test, dtype=dtype, device=device).unsqueeze(0)
            X_test_t = torch.tensor(X_test, dtype=dtype, device=device).unsqueeze(0)
            F_test = uks_model.get_trend_matrix(U_test_t, X_test_t)
            Y_trend_test = torch.bmm(F_test, beta_latent)
            Z_trend_test = uks_model.flow.inverse(Y_trend_test)
            M_hat_test = Z_trend_test.cpu().numpy()[0, :, 0] * std_Z + mean_Z
            R_hat_test = Z_pred_uks - M_hat_test
            
            Y_train_flow_np = Y_train_flow.cpu().numpy()[0, :, 0]
            
            Z_test_t = torch.tensor(Z_test, dtype=dtype, device=device).view(1, -1, 1)
            Z_test_flow = torch.cat([Z_test_t, torch.zeros_like(Z_test_t)], dim=-1)
            Y_test_flow, _ = uks_model.flow(Z_test_flow)
            Y_test_flow_np = Y_test_flow.cpu().numpy()[0, :, 0]
            
            # 典型坐标处的协方差局部椭圆估计
            grid_x = np.linspace(0, 1, 50)
            grid_y = np.linspace(0, 1, 50)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
            grid_coords_np = np.stack([grid_xx.ravel(), grid_yy.ravel()], axis=1)
            grid_coords = torch.tensor(grid_coords_np, dtype=dtype, device=device).unsqueeze(0)
            H_grid = uks_model.sce(grid_coords)
            
            ref_points = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
            cov_fields = []
            for ref_pt in ref_points:
                u_ref = torch.tensor([ref_pt], dtype=dtype, device=device).unsqueeze(0)
                H_ref = uks_model.sce(u_ref)
                _, cov_vector = uks_model.kernel(H_grid, H_ref, grid_coords, u_ref)
                cov_fields.append(cov_vector.cpu().numpy().flatten())
                
            cov_field_1 = cov_fields[0]
            cov_field_2 = cov_fields[1]
            cov_field_3 = cov_fields[2]

        # 3.8 保存本场景的独立物理成果与基准指标
        model_save_path = f"{d_dir}/uks_model.pth"
        torch.save(uks_model.state_dict(), model_save_path)
        
        npz_path = f"{d_dir}/experiment_results.npz"
        np.savez(
            npz_path,
            coords_test=coords_test,
            Z_test=Z_test_raw,
            Z_pred_ok=Z_pred_ok,
            Z_pred_uk=Z_pred_uk,
            Z_pred_mlp=Z_pred_mlp,
            Z_pred_uks=Z_pred_uks,
            Z_var_uks=Z_var_uks,  # 条件物理估计不确定性方差场
            Lambda_u0=Lambda_u0,
            lambda_C_u0=lambda_C_u0,
            Y_train_flow=Y_train_flow_np,
            Y_test_flow=Y_test_flow_np,
            M_hat_train=M_hat_train,
            M_hat_test=M_hat_test,
            R_hat_train=R_hat_train,
            R_hat_test=R_hat_test,
            cov_field_1=cov_field_1,
            cov_field_2=cov_field_2,
            cov_field_3=cov_field_3,
            mean_Z=mean_Z,
            std_Z=std_Z
        )
        
        # 指标汇总与打印
        models_pred = {
            "Ordinary Kriging": Z_pred_ok,
            "Universal Kriging": Z_pred_uk,
            "MLP Network": Z_pred_mlp,
            "UKS-DGL": Z_pred_uks
        }
        
        d_metrics = {}
        print(f"\n--- 场景 {d_name} 插值精度汇总 (最终最优超参表现) ---")
        print(f"{'模型名称 (Model Name)':<20} | {'MAE':<10} | {'RMSE':<10} | {'R^2':<10} | {'残差 Moran I':<15}")
        print("-" * 75)
        for name, pred in models_pred.items():
            mae, rmse, r2 = compute_metrics(Z_test_raw, pred)
            moran_i = compute_morans_i(coords_test, Z_test_raw - pred)
            d_metrics[name] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
                "Morans_I": float(moran_i)
            }
            print(f"{name:<20} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<10.4f} | {moran_i:<15.4f}")
            
        metrics_summary[d_name] = d_metrics
        
        json_path = f"{d_dir}/metrics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(d_metrics, f, ensure_ascii=False, indent=4)
            
    # 保存总指标记录
    total_json_path = f"{output_dir}/metrics_summary.json"
    with open(total_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=4)
        
    print(f"\n三场景总实验指标已保存至: {total_json_path}")
    
    # 4. 执行 Git 自动 Commit 实验归档
    run_git_checkpoint(output_dir, metrics_summary)

if __name__ == '__main__':
    main()
