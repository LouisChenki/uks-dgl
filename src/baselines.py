# -*- coding: utf-8 -*-
"""
对比基线模型实现，包括普通克里金 (Ordinary Kriging, OK)、漂移克里金 (Universal Kriging, UK)、
纯 MLP 神经网络以及 Moran's I 空间自相关指标计算。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist

class OrdinaryKriging:
    """
    经典普通克里金 (Ordinary Kriging, OK) 空间插值基线模型。
    """
    def __init__(self, sigma_sq=0.5, l_corr=0.2, nugget=1e-6):
        self.sigma_sq = sigma_sq
        self.l_corr = l_corr
        self.nugget = nugget
        self.coords_train = None
        self.Z_train = None

    def fit(self, coords_train, Z_train):
        """
        拟合训练集数据 (Fit Training Data)
        """
        self.coords_train = np.array(coords_train)
        self.Z_train = np.array(Z_train).reshape(-1, 1)

    def predict(self, coords_test):
        """
        测试集插值预测 (Interpolation Prediction)
        """
        N = self.coords_train.shape[0]
        M = coords_test.shape[0]
        
        # 1. 计算已知点之间的距离与协方差矩阵 (Covariance Matrix) C
        D_train = cdist(self.coords_train, self.coords_train, metric='euclidean') # [N, N]
        C = self.sigma_sq * np.exp(-D_train / self.l_corr) # [N, N]
        C += self.nugget * np.eye(N) # [N, N]
        
        # 2. 构建 OK 系统控制矩阵 A_OK (Ordinary Kriging Matrix)
        # 形状为 [N+1, N+1]
        A_OK = np.zeros((N + 1, N + 1))
        A_OK[:N, :N] = C
        A_OK[:N, N] = 1.0
        A_OK[N, :N] = 1.0
        A_OK[N, N] = 0.0
        
        # 3. 计算预测点与已知点之间的互协方差向量 (Cross-covariance Vector) c0
        D_test = cdist(self.coords_train, coords_test, metric='euclidean') # [N, M]
        c0 = self.sigma_sq * np.exp(-D_test / self.l_corr) # [N, M]
        
        # 4. 构建 OK 系统右端矩阵 B_OK
        # 形状为 [N+1, M]
        B_OK = np.zeros((N + 1, M))
        B_OK[:N, :] = c0
        B_OK[N, :] = 1.0
        
        # 5. 矩阵化求解方程组 (Solving System Matrix)
        X_OK = np.linalg.solve(A_OK, B_OK) # [N+1, M]
        
        # 6. 提取插值权重 Lambda
        Lambda = X_OK[:N, :] # [N, M]
        
        # 7. 物理值插值计算 Z_pred = Lambda^T @ Z_train
        Z_pred = Lambda.T @ self.Z_train # [M, 1]
        
        return Z_pred.flatten(), Lambda


class UniversalKriging:
    """
    漂移克里金 (Universal Kriging, UK) 空间插值基线模型。
    引入一阶线性趋势面 (First-order Linear Trend Surface) F = [1, u_x, u_y]。
    """
    def __init__(self, sigma_sq=0.5, l_corr=0.2, nugget=1e-6):
        self.sigma_sq = sigma_sq
        self.l_corr = l_corr
        self.nugget = nugget
        self.coords_train = None
        self.Z_train = None

    def fit(self, coords_train, Z_train):
        """
        拟合训练集数据 (Fit Training Data)
        """
        self.coords_train = np.array(coords_train)
        self.Z_train = np.array(Z_train).reshape(-1, 1)

    def _get_trend_matrix(self, coords):
        """
        获取一阶趋势面基矩阵 (Trend Base Matrix) F
        """
        N = coords.shape[0]
        F = np.ones((N, 3))
        F[:, 1:3] = coords # [N, 3]，第 0 列为全 1，第 1、2 列分别为坐标 x 和 y
        return F

    def predict(self, coords_test):
        """
        测试集插值预测 (Interpolation Prediction)
        """
        N = self.coords_train.shape[0]
        M = coords_test.shape[0]
        
        # 1. 计算已知点之间的距离与协方差矩阵 (Covariance Matrix) C
        D_train = cdist(self.coords_train, self.coords_train, metric='euclidean') # [N, N]
        C = self.sigma_sq * np.exp(-D_train / self.l_corr) # [N, N]
        C += self.nugget * np.eye(N) # [N, N]
        
        # 2. 构造已知点处的趋势矩阵 (Trend Matrix) F
        F = self._get_trend_matrix(self.coords_train) # [N, 3]
        
        # 3. 构建 UK 系统控制矩阵 A_UK (Universal Kriging Matrix)
        # 形状为 [N+3, N+3]
        A_UK = np.zeros((N + 3, N + 3))
        A_UK[:N, :N] = C
        A_UK[:N, N:] = F
        A_UK[N:, :N] = F.T
        A_UK[N:, N:] = 0.0
        
        # 4. 计算预测点与已知点之间的互协方差向量 (Cross-covariance Vector) c0
        D_test = cdist(self.coords_train, coords_test, metric='euclidean') # [N, M]
        c0 = self.sigma_sq * np.exp(-D_test / self.l_corr) # [N, M]
        
        # 5. 计算预测点处的趋势向量 (Trend Vector) f0
        f0 = self._get_trend_matrix(coords_test) # [M, 3]
        
        # 6. 构建 UK 系统右端矩阵 B_UK
        # 形状为 [N+3, M]
        B_UK = np.zeros((N + 3, M))
        B_UK[:N, :] = c0
        B_UK[N:, :] = f0.T
        
        # 7. 矩阵化求解方程组 (Solving System Matrix)
        X_UK = np.linalg.solve(A_UK, B_UK) # [N+3, M]
        
        # 8. 提取插值权重 Lambda
        Lambda = X_UK[:N, :] # [N, M]
        
        # 9. 物理值插值计算 Z_pred = Lambda^T @ Z_train
        Z_pred = Lambda.T @ self.Z_train # [M, 1]
        
        return Z_pred.flatten(), Lambda


class MLPRegressor(nn.Module):
    """
    纯 MLP 神经网络模型 (Pure MLP Model)，直接将 2D 坐标拟合物理观测值 Z。
    结构采用 3 层全连接层: 输入 2 维坐标 -> 64 -> 64 -> 1 维物理输出。
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: 空间 2D 坐标 [B, 2] -> [B, 1]
        """
        out = self.net(x) # [B, 1]
        return out


def train_mlp(coords_train, Z_train, coords_test, Z_test, epochs=300, lr=0.01, device='mps'):
    """
    纯 MLP 的模型训练与评估。
    根据 Apple Silicon M4 Max 硬件特点，优先支持 MPS 加速以及 bfloat16 混合精度计算。
    """
    # 优先采用 bfloat16 精度进行训练
    dtype = torch.bfloat16
    
    # 数据转换，确保与设备和精度匹配
    X_tr = torch.tensor(coords_train, dtype=dtype, device=device) # [N_train, 2]
    Y_tr = torch.tensor(Z_train, dtype=dtype, device=device).unsqueeze(1) # [N_train, 1]
    
    X_te = torch.tensor(coords_test, dtype=dtype, device=device) # [N_test, 2]
    Y_te = torch.tensor(Z_test, dtype=dtype, device=device).unsqueeze(1) # [N_test, 1]
    
    # 实例化模型
    model = MLPRegressor(hidden_dim=64).to(device).to(dtype)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print("--> [训练基线 MLP] 开始训练纯 MLP 神经网络...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # 自动混合精度 (Automatic Mixed Precision, AMP)
        with torch.amp.autocast('mps'):
            pred = model(X_tr) # [N_train, 1]
            # 损失函数 (Loss Function) 显式转回 float32 精度以防下溢或精度损失
            loss = torch.mean((pred.float() - Y_tr.float()) ** 2)
            
        loss.backward()
        # 梯度裁剪防梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch == 1 or epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast('mps'):
                    test_pred = model(X_te)
                    test_loss = torch.mean((test_pred.float() - Y_te.float()) ** 2)
            print(f"    Epoch {epoch:03d}/{epochs:03d} | Train MSE: {loss.item():.6f} | Test MSE: {test_loss.item():.6f}")
            
    # 推理阶段，必须显式调用 model.eval() 并在 torch.no_grad() 下执行
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('mps'):
            pred_test = model(X_te).float().cpu().numpy().flatten()
            
    return pred_test, model


def compute_morans_i(coords, residuals):
    """
    计算空间残差的空间自相关 Moran's I 指数 (Moran's I Index of Spatial Autocorrelation)。
    残差: e = Z_true - Z_pred
    空间权重矩阵: 倒数反距离权重 w_ij = 1.0 / (d_ij + eps)，对角线设为 0
    """
    coords = np.array(coords)
    residuals = np.array(residuals).flatten()
    M = len(residuals)
    
    # 1. 计算点对之间的欧氏距离矩阵
    D = cdist(coords, coords, metric='euclidean') # [M, M]
    
    # 2. 构建反距离权重矩阵 (Inverse Distance Weight Matrix)
    eps = 1e-6
    W = 1.0 / (D + eps)
    np.fill_diagonal(W, 0.0) # 对角线 w_ii = 0.0，避免自己对自己自相关
    
    # 3. 计算 Moran's I 公式中的各分量
    S0 = np.sum(W)
    mean_res = np.mean(residuals)
    res_diff = residuals - mean_res
    
    # 双重求和项: sum_i sum_j w_ij * (e_i - e_bar) * (e_j - e_bar)
    # 利用矩阵乘法加速: res_diff^T @ W @ res_diff
    numerator = res_diff.T @ W @ res_diff
    
    # 分母部分: sum_i (e_i - e_bar)^2
    denominator = np.sum(res_diff ** 2)
    
    if denominator == 0:
        return 0.0
        
    moran_i = (M / S0) * (numerator / denominator)
    return moran_i
