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
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

def estimate_kriging_parameters(coords, Z, trend_type='constant', X_cov=None, nugget_fixed=None):
    """
    通过极大似然估计 (Maximum Likelihood Estimation, MLE) 自动估计空间自相关核参数。
    """
    N = coords.shape[0]
    Z = Z.reshape(-1, 1)
    
    # 1. 计算两两点对的欧式距离矩阵
    D = cdist(coords, coords, metric='euclidean') # [N, N]
    
    # 2. 根据趋势类型构造大尺度均值趋势的设计矩阵 F
    if trend_type == 'constant':
        F = np.ones((N, 1))
    elif trend_type == 'linear':
        F = np.ones((N, 3))
        F[:, 1:3] = coords
    elif trend_type == 'external':
        F = np.ones((N, 1 + X_cov.shape[1]))
        F[:, 1:] = X_cov
    else:
        F = None # 零均值残差系统
        
    def negative_log_likelihood(params):
        """
        负对数似然损失函数
        """
        if nugget_fixed is not None:
            sigma_sq, l_corr = params
            tau_sq = nugget_fixed
        else:
            sigma_sq, l_corr, tau_sq = params
            
        # 指数相关结构矩阵
        R = np.exp(-D / l_corr)
        C = sigma_sq * R + tau_sq * np.eye(N)
        
        try:
            L = np.linalg.cholesky(C)  # C = L L^T
        except np.linalg.LinAlgError:
            return 1e10  # 发生奇异时返回极大惩罚值
            
        if F is not None:
            # 广义最小二乘 (GLS) 估计均值系数 beta
            V = solve_triangular(L, F, lower=True)
            w = solve_triangular(L, Z, lower=True)
            V_T_V = V.T @ V + 1e-6 * np.eye(F.shape[1])
            beta = np.linalg.solve(V_T_V, V.T @ w)
            res = Z - F @ beta
        else:
            res = Z
            
        w_res = solve_triangular(L, res, lower=True)
        quadratic_form = np.sum(w_res ** 2)
        log_det_C = 2.0 * np.sum(np.log(np.maximum(np.diag(L), 1e-6)))
        
        nll = 0.5 * quadratic_form + 0.5 * log_det_C + 0.5 * N * np.log(2.0 * np.pi)
        return nll

    # 设置超参数寻优边界与初始猜测值
    if nugget_fixed is not None:
        init_guess = [0.5, 0.2]          # [sigma_sq, l_corr]
        bounds = [(0.05, 3.0), (0.05, 1.2)]
    else:
        init_guess = [0.5, 0.2, 0.01]    # [sigma_sq, l_corr, nugget]
        bounds = [(0.05, 3.0), (0.05, 1.2), (1e-5, 0.3)]
        
    res_opt = minimize(negative_log_likelihood, init_guess, bounds=bounds, method='L-BFGS-B')
    
    if res_opt.success:
        if nugget_fixed is not None:
            return float(res_opt.x[0]), float(res_opt.x[1]), nugget_fixed
        else:
            return float(res_opt.x[0]), float(res_opt.x[1]), float(res_opt.x[2])
    else:
        if nugget_fixed is not None:
            return 0.5, 0.2, nugget_fixed
        else:
            return 0.5, 0.2, 1e-5


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

    def fit(self, coords_train, Z_train, use_mle=True):
        """
        拟合训练集数据，支持通过极大似然估计 (MLE) 自适应求解空间结构超参数。
        """
        self.coords_train = np.array(coords_train)
        self.Z_train = np.array(Z_train).reshape(-1, 1)
        
        if use_mle:
            # MLE 自动寻参
            self.sigma_sq, self.l_corr, self.nugget = estimate_kriging_parameters(
                self.coords_train, self.Z_train, trend_type='constant'
            )

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

    def fit(self, coords_train, Z_train, use_mle=True):
        """
        一阶漂移趋势下，支持基于极大似然估计 (MLE) 自适应优化变差函数核参数。
        """
        self.coords_train = np.array(coords_train)
        self.Z_train = np.array(Z_train).reshape(-1, 1)
        
        if use_mle:
            self.sigma_sq, self.l_corr, self.nugget = estimate_kriging_parameters(
                self.coords_train, self.Z_train, trend_type='linear'
            )
            print(f"--> [UK MLE 拟合成功] σ^2 (基台): {self.sigma_sq:.4f}, l (变程): {self.l_corr:.4f}, nugget (块金): {self.nugget:.6f}")


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


class CoKriging:
    """
    经典多变量普通协同克里金 (Co-Kriging, CK) 基线模型。
    支持基于双核线性共区域化模型 (LMC) 的极大似然估计 (MLE) 参数拟合。
    """
    def __init__(self, l1=0.2, l2=0.4, nugget1=1e-5, nugget2=1e-5):
        self.l1 = l1
        self.l2 = l2
        self.nugget1 = nugget1
        self.nugget2 = nugget2
        self.W1 = np.array([[0.5, 0.0], [0.1, 0.5]])
        self.W2 = np.array([[0.3, 0.0], [0.05, 0.3]])
        self.coords_train = None
        self.Z_train = None
        
    def fit(self, coords_train, Z_train, use_mle=True):
        """
        拟合协同克里金模型参数。
        Z_train 维度为 [N, 2]，其中通道 0 为主变量，通道 1 为协变量。
        """
        self.coords_train = np.array(coords_train)
        self.Z_train = np.array(Z_train) # [N, 2]
        N = self.coords_train.shape[0]
        
        # 展平观测 Z 为 [2N, 1]，前 N 个为主变量，后 N 个为协变量
        Z_flat = np.concatenate([self.Z_train[:, 0], self.Z_train[:, 1]]).reshape(-1, 1)
        
        D_train = cdist(self.coords_train, self.coords_train, metric='euclidean')
        
        # 定义趋势设计矩阵 F [2N, 2]
        F_design = np.zeros((2 * N, 2))
        F_design[:N, 0] = 1.0
        F_design[N:, 1] = 1.0
        
        if use_mle:
            def nll_func(params):
                # 提取参数
                # params: [l1, l2, w11_1, w21_1, w22_1, w11_2, w21_2, w22_2, tau1, tau2]
                l1_val, l2_val = params[0], params[1]
                w11_1, w21_1, w22_1 = params[2], params[3], params[4]
                w11_2, w21_2, w22_2 = params[5], params[6], params[7]
                tau1_sq, tau2_sq = params[8]**2, params[9]**2
                
                # 构建共区域化矩阵 B
                B1 = np.array([[w11_1**2, w11_1 * w21_1], [w11_1 * w21_1, w21_1**2 + w22_1**2]])
                B2 = np.array([[w11_2**2, w11_2 * w21_2], [w11_2 * w21_2, w21_2**2 + w22_2**2]])
                
                # 空间相关核
                R1 = np.exp(-D_train / l1_val)
                R2 = np.exp(-D_train / l2_val)
                
                # 构建 2N x 2N 协方差矩阵
                C = np.zeros((2 * N, 2 * N))
                
                # C11, C12, C21, C22
                C[:N, :N] = B1[0, 0] * R1 + B2[0, 0] * R2 + tau1_sq * np.eye(N)
                C[:N, N:] = B1[0, 1] * R1 + B2[0, 1] * R2
                C[N:, :N] = C[:N, N:].T
                C[N:, N:] = B1[1, 1] * R1 + B2[1, 1] * R2 + tau2_sq * np.eye(N)
                
                # Cholesky 分解
                try:
                    L = np.linalg.cholesky(C + 1e-8 * np.eye(2 * N))
                except np.linalg.LinAlgError:
                    return 1e12
                
                # GLS 均值估计
                V = solve_triangular(L, F_design, lower=True)
                w_vec = solve_triangular(L, Z_flat, lower=True)
                V_T_V = V.T @ V + 1e-6 * np.eye(2)
                beta = np.linalg.solve(V_T_V, V.T @ w_vec)
                res = Z_flat - F_design @ beta
                
                w_res = solve_triangular(L, res, lower=True)
                quad_form = np.sum(w_res ** 2)
                log_det_C = 2.0 * np.sum(np.log(np.maximum(np.diag(L), 1e-6)))
                
                nll = 0.5 * quad_form + 0.5 * log_det_C + 0.5 * (2 * N) * np.log(2.0 * np.pi)
                return nll
            
            # 初始猜测
            init_guess = [0.2, 0.4, 0.5, 0.1, 0.5, 0.3, 0.05, 0.3, 0.05, 0.05]
            bounds = [
                (0.05, 1.2), (0.05, 1.2), # l1, l2
                (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), # W1
                (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), # W2
                (1e-3, 0.3), (1e-3, 0.3) # tau1, tau2
            ]
            
            res_opt = minimize(nll_func, init_guess, bounds=bounds, method='L-BFGS-B')
            if res_opt.success:
                p = res_opt.x
                self.l1, self.l2 = p[0], p[1]
                self.W1 = np.array([[p[2], 0.0], [p[3], p[4]]])
                self.W2 = np.array([[p[5], 0.0], [p[6], p[7]]])
                self.nugget1, self.nugget2 = p[8]**2, p[9]**2
                print(f"--> [Co-Kriging MLE 拟合成功] l1={self.l1:.4f}, l2={self.l2:.4f}, nugget1={self.nugget1:.6f}, nugget2={self.nugget2:.6f}")
            else:
                print("--> [Co-Kriging MLE 拟合失败] 使用默认参数")

    def predict(self, coords_test):
        """
        预测测试点的主变量。
        """
        N = self.coords_train.shape[0]
        M = coords_test.shape[0]
        
        # 1. 拼装训练协方差矩阵 C [2N, 2N]
        D_train = cdist(self.coords_train, self.coords_train, metric='euclidean')
        R1_train = np.exp(-D_train / self.l1)
        R2_train = np.exp(-D_train / self.l2)
        
        B1 = self.W1 @ self.W1.T
        B2 = self.W2 @ self.W2.T
        
        C = np.zeros((2 * N, 2 * N))
        C[:N, :N] = B1[0, 0] * R1_train + B2[0, 0] * R2_train + self.nugget1 * np.eye(N)
        C[:N, N:] = B1[0, 1] * R1_train + B2[0, 1] * R2_train
        C[N:, :N] = C[:N, N:].T
        C[N:, N:] = B1[1, 1] * R1_train + B2[1, 1] * R2_train + self.nugget2 * np.eye(N)
        
        # 2. 趋势设计矩阵 F [2N, 2]
        F = np.zeros((2 * N, 2))
        F[:N, 0] = 1.0
        F[N:, 1] = 1.0
        
        # 3. 构造协同克里金系统矩阵 A_CK [2N+2, 2N+2]
        A_CK = np.zeros((2 * N + 2, 2 * N + 2))
        A_CK[:2*N, :2*N] = C
        A_CK[:2*N, 2*N:] = F
        A_CK[2*N:, :2*N] = F.T
        
        # 4. 预测点与已知点之间的互协方差向量 c0
        D_test = cdist(self.coords_train, coords_test, metric='euclidean') # [N, M]
        R1_test = np.exp(-D_test / self.l1)
        R2_test = np.exp(-D_test / self.l2)
        
        c0 = np.zeros((2 * N, M))
        c0[:N, :] = B1[0, 0] * R1_test + B2[0, 0] * R2_test
        c0[N:, :] = B1[1, 0] * R1_test + B2[1, 0] * R2_test
        
        # 5. 预测点处的趋势向量 f0 [2, M]
        f0 = np.zeros((2, M))
        f0[0, :] = 1.0
        
        # 6. 右端项 B_CK [2N+2, M]
        B_CK = np.zeros((2 * N + 2, M))
        B_CK[:2*N, :] = c0
        B_CK[2*N:, :] = f0
        
        # 7. 求解方程组
        try:
            X_CK = np.linalg.solve(A_CK, B_CK) # [2N+2, M]
        except np.linalg.LinAlgError:
            A_CK[:2*N, :2*N] += 1e-4 * np.eye(2 * N)
            X_CK = np.linalg.solve(A_CK, B_CK)
            
        Lambda = X_CK[:2*N, :] # [2N, M]
        
        # 8. 计算插值
        Z_train_flat = np.concatenate([self.Z_train[:, 0], self.Z_train[:, 1]]).reshape(-1, 1) # [2N, 1]
        Z_pred = Lambda.T @ Z_train_flat # [M, 1]
        
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
