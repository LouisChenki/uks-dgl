# -*- coding: utf-8 -*-
"""
统一克里金系统 (Unified Kriging System, UKS) 神经网络模型。
集成了 Normalizing Flow (RealNVP)、谱坐标嵌入网络 (SCE) 以及显式马氏各向异性自适应核网络 (AKN)。
支持在趋势面解耦中引入外部协变量。
"""

import math
import torch
import torch.nn as nn
from uks_solver import UKSSolverOp

# ==========================================
# MODEL CONFIGURATION (模型物理结构参数配置区)
# ==========================================
FLOW_HIDDEN_DIM = 32
KERNEL_HIDDEN_DIM = 32
DROPOUT_P = 0.1
NUGGET_EPS = 1.0e-05
# ==========================================

class CouplingLayer(nn.Module):
    """
    RealNVP 可逆流的耦合层 (Coupling Layer)。
    将输入通道拆分为两部分，通过 MLP 计算尺度 (scale) 与平移 (translation)。
    """
    def __init__(self, dim, hidden_dim, mask_type):
        super().__init__()
        self.dim = dim
        self.mask_type = mask_type  # 'even' 偶数通道不发生变换，'odd' 奇数通道不发生变换

        # 拆分通道数
        in_dim = dim // 2
        out_dim = dim - in_dim

        # 2层 MLP 用于计算缩放与平移参数，应用配置区定义的 DROPOUT_P
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(hidden_dim, out_dim * 2)  # 同时输出缩放 s 与平移 t
        )

    def forward(self, x):
        """
        正向变换 (Forward Translation): 物理空间 -> 隐空间
        x: [B_total, D]，其中 B_total = B * N
        """
        # 沿特征维度分割
        if self.mask_type == 'even':
            x1, x2 = x[:, :self.dim//2], x[:, self.dim//2:]  # [B_total, d], [B_total, D-d]
        else:
            x2, x1 = x[:, :self.dim//2], x[:, self.dim//2:]  # [B_total, D-d], [B_total, d]

        # x1 作为输入计算缩放项与平移项
        h = self.net(x1)                  # [B_total, (D-d)*2]
        s, t = h.chunk(2, dim=-1)         # [B_total, D-d], [B_total, D-d]

        # 限制缩放幅度，防止数值溢出
        s = torch.tanh(s) * 5.0           # [B_total, D-d]

        y1 = x1                           # [B_total, d]
        y2 = x2 * torch.exp(s) + t        # [B_total, D-d]

        # 雅可比行列式对数 (Log-Determinant of Jacobian, LDJ)
        log_det = torch.sum(s, dim=-1)     # [B_total]

        if self.mask_type == 'even':
            y = torch.cat([y1, y2], dim=-1) # [B_total, D]
        else:
            y = torch.cat([y2, y1], dim=-1) # [B_total, D]

        return y, log_det

    def inverse(self, y):
        """
        逆向变换 (Inverse Translation): 隐空间 -> 物理空间
        y: [B_total, D]
        """
        if self.mask_type == 'even':
            y1, y2 = y[:, :self.dim//2], y[:, self.dim//2:]  # [B_total, d], [B_total, D-d]
        else:
            y2, y1 = y[:, :self.dim//2], y[:, self.dim//2:]  # [B_total, D-d], [B_total, d]

        h = self.net(y1)                  # [B_total, (D-d)*2]
        s, t = h.chunk(2, dim=-1)         # [B_total, D-d], [B_total, D-d]
        s = torch.tanh(s) * 5.0           # [B_total, D-d]

        # 逆变换公式
        x1 = y1                           # [B_total, d]
        x2 = (y2 - t) * torch.exp(-s)     # [B_total, D-d]

        if self.mask_type == 'even':
            x = torch.cat([x1, x2], dim=-1) # [B_total, D]
        else:
            x = torch.cat([x2, x1], dim=-1) # [B_total, D]

        return x


class RealNVP(nn.Module):
    """
    可逆映射层 Φ (Normalizing Flow - RealNVP)。
    通过交替掩码的耦合层，将物理观测数据高斯化转换到隐空间。
    默认耦合层数为 2 层。
    """
    def __init__(self, dim, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mask_type = 'even' if i % 2 == 0 else 'odd'
            self.layers.append(CouplingLayer(dim, hidden_dim, mask_type))

    def forward(self, x):
        """
        高斯化前向转换
        x: [B, N, D]
        """
        B, N, D = x.shape                 # 维度解析
        x_flat = x.view(-1, D)            # [B*N, D]
        log_det_sum = torch.zeros(B * N, device=x.device, dtype=torch.float32) # [B*N]

        for layer in self.layers:
            x_flat, log_det = layer(x_flat)  # [B*N, D], [B*N]
            log_det_sum = log_det_sum + log_det  # [B*N]

        y = x_flat.view(B, N, D)          # [B, N, D]
        log_det_sum = log_det_sum.view(B, N) # [B, N]
        return y, log_det_sum

    def inverse(self, y):
        """
        反高斯化逆向还原
        y: [B, N, D]
        """
        B, N, D = y.shape                 # 维度解析
        y_flat = y.view(-1, D)            # [B*N, D]

        for layer in reversed(self.layers):
            y_flat = layer.inverse(y_flat)  # [B*N, D]

        x = y_flat.view(B, N, D)          # [B, N, D]
        return x


class SpectralCoordinateEmbedding(nn.Module):
    """
    谱坐标嵌入网络 (Spectral Coordinate Embedding, SCE)。
    使用随机傅里叶特征 (Random Fourier Features, RFF) 对 2D 坐标进行高频嵌入。
    """
    def __init__(self, in_features=2, out_features=16, sigma=10.0):
        super().__init__()
        assert out_features % 2 == 0, "输出嵌入特征数 out_features 必须为偶数"
        self.in_features = in_features
        self.out_features = out_features

        # 随机傅里叶投影权重投影矩阵 B_F
        B_F = torch.randn(out_features // 2, in_features) * sigma
        self.register_buffer('B_F', B_F)

    def forward(self, coords):
        """
        coords: [B, N, 2]
        """
        proj = torch.matmul(coords, self.B_F.t()) # [B, N, out_features // 2]
        
        cos_proj = torch.cos(proj)        # [B, N, out_features // 2]
        sin_proj = torch.sin(proj)        # [B, N, out_features // 2]

        rff = torch.cat([cos_proj, sin_proj], dim=-1) # [B, N, out_features]
        return rff


class AdaptiveKernelNetwork(nn.Module):
    """
    显式马氏自适应核网络 (Local Adaptive Mahalanobis Anisotropy Kernel Network)。
    通过谱特征，使用 MLP 预测每个点的局部各向异性几何参数：旋转角 theta, 主轴长度 l_1, 次轴长度 l_2。
    利用马氏距离公式对称化计算两点间的自适应非平稳协方差，并在长度尺度上采用 Sigmoid 平滑截断以规避对角奇异化。
    """
    def __init__(self, embed_dim=16, hidden_dim=32):
        super().__init__()
        # 输出 3 个标量参数：t(旋转相关), p1(主轴相关), p2(次轴相关)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(hidden_dim, 3)
        )
        # 可学习的协方差边际尺度 log_sigma_f (Log-scale)
        self.log_sigma_f = nn.Parameter(torch.tensor(0.0))

    def get_local_metric_matrix(self, H):
        """
        根据谱嵌入特征计算局部 Riemannian 度量矩阵 G(u) = R^T S^-2 R
        H: [B, N, embed_dim]
        返回 G: [B, N, 2, 2]
        """
        params = self.mlp(H)  # [B, N, 3]
        t = params[:, :, 0]   # [B, N]
        p1 = params[:, :, 1]  # [B, N]
        p2 = params[:, :, 2]  # [B, N]

        # 1. 各向异性主方向旋转角估计 theta = pi * tanh(t)
        theta = math.pi * torch.tanh(t)  # [B, N]

        # 2. 长度尺度 l_1 和 l_2 平滑有界映射，放宽限制以支持自适应退化为各向同性
        # 长轴与短轴均允许在 [0.05, 0.60] 范围内，对称解耦
        l1 = 0.05 + (0.60 - 0.05) * torch.sigmoid(p1)  # [B, N]
        l2 = 0.05 + (0.60 - 0.05) * torch.sigmoid(p2)  # [B, N]

        # 3. 构造局部旋转与拉伸度量矩阵
        cos_t = torch.cos(theta)  # [B, N]
        sin_t = torch.sin(theta)  # [B, N]

        inv_l1_sq = 1.0 / (l1 ** 2)  # [B, N]
        inv_l2_sq = 1.0 / (l2 ** 2)  # [B, N]

        # 构造度量矩阵分量
        g11 = cos_t**2 * inv_l1_sq + sin_t**2 * inv_l2_sq        # [B, N]
        g12 = cos_t * sin_t * (inv_l1_sq - inv_l2_sq)            # [B, N]
        g22 = sin_t**2 * inv_l1_sq + cos_t**2 * inv_l2_sq        # [B, N]

        # 堆叠为 [B, N, 2, 2] 度量矩阵
        G_row1 = torch.stack([g11, g12], dim=-1)  # [B, N, 2]
        G_row2 = torch.stack([g12, g22], dim=-1)  # [B, N, 2]
        G = torch.stack([G_row1, G_row2], dim=-2)  # [B, N, 2, 2]

        return G

    def forward(self, H_obs, H_pred, U_obs, U_pred):
        """
        H_obs: 已知点谱嵌入特征 [B, N, embed_dim]
        H_pred: 预测点谱嵌入特征 [B, 1, embed_dim]
        U_obs: 已知点物理坐标 [B, N, 2]
        U_pred: 预测点物理坐标 [B, 1, 2]
        """
        B, N, _ = H_obs.shape

        # 提取观测点与测试点的局部几何度量矩阵
        G_obs = self.get_local_metric_matrix(H_obs)    # [B, N, 2, 2]
        G_pred = self.get_local_metric_matrix(H_pred)  # [B, 1, 2, 2]

        # 1. 计算已知点两两之间的对称化非平稳马氏距离平方 dist_sq_C
        # 差值向量 coords_diff_C: [B, N, N, 2]
        coords_diff_C = U_obs.unsqueeze(2) - U_obs.unsqueeze(1)  # [B, N, 1, 2] - [B, 1, N, 2] -> [B, N, N, 2]
        
        # 观测点 i 处的单侧变换：temp = G_i * diff_ij
        temp_diff_C = torch.einsum('bijk,bikm->bijm', coords_diff_C, G_obs)  # [B, N, N, 2]
        dist_sq_i = torch.sum(coords_diff_C * temp_diff_C, dim=-1)  # [B, N, N]
        
        # 对称化处理：d^2 = 0.5 * (d_i^2 + d_j^2)
        dist_sq_C = 0.5 * (dist_sq_i + dist_sq_i.transpose(-2, -1))  # [B, N, N]

        # 2. 计算已知点与预测点之间的对称化非平稳马氏距离平方 dist_sq_c0
        # 差值向量 coords_diff_c0: [B, N, 1, 2]
        coords_diff_c0 = U_obs.unsqueeze(2) - U_pred.unsqueeze(1)  # [B, N, 1, 2]
        
        # 观测点处的单侧变换
        temp_diff_c0_obs = torch.einsum('bijk,bikm->bijm', coords_diff_c0, G_obs)  # [B, N, 1, 2]
        dist_sq_c0_obs = torch.sum(coords_diff_c0 * temp_diff_c0_obs, dim=-1)      # [B, N, 1]
        
        # 预测点处的单侧变换
        temp_diff_c0_pred = torch.einsum('bijk,bikm->bijm', coords_diff_c0, G_pred.expand(-1, N, -1, -1))  # [B, N, 1, 2]
        dist_sq_c0_pred = torch.sum(coords_diff_c0 * temp_diff_c0_pred, dim=-1)    # [B, N, 1]
        
        # 对称化
        dist_sq_c0 = 0.5 * (dist_sq_c0_obs + dist_sq_c0_pred)  # [B, N, 1]

        sigma_f_sq = torch.exp(2.0 * self.log_sigma_f)

        # 3. 构造自适应协方差矩阵与互协方差向量
        C = sigma_f_sq * torch.exp(-0.5 * dist_sq_C)    # [B, N, N]
        c_0 = sigma_f_sq * torch.exp(-0.5 * dist_sq_c0)  # [B, N, 1]

        # 显式对称化以完全消除浮点误差
        C = 0.5 * (C + C.transpose(-2, -1))  # [B, N, N]

        # 显式修正对角线元素，使其精确为 sigma_f_sq，剔除微小浮点误差
        eye_N = torch.eye(N, device=C.device, dtype=torch.float32).unsqueeze(0)  # [1, N, N]
        C = C * (1.0 - eye_N) + sigma_f_sq * eye_N  # [B, N, N]

        return C, c_0


class UKSModel(nn.Module):
    """
    统一克里金系统 (Unified Kriging System, UKS) 完整神经网络。
    集成了物理可逆高斯化流 (RealNVP)、谱嵌入 (SCE)、自适应马氏核 (AKN) 以及克里金求解算子。
    支持接收外部协变量作为大尺度全局趋势解耦的物理基底 (KED)。
    """
    def __init__(self, in_dim=1, flow_hidden_dim=32, num_flow_layers=2,
                 embed_dim=16, rff_sigma=10.0,
                 kernel_hidden_dim=32, latent_dim=8, eps=None, cov_dim=2):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps if eps is not None else NUGGET_EPS

        self.flow_dim = max(2, in_dim * 2) if in_dim == 1 else in_dim
        self.flow = RealNVP(dim=self.flow_dim, hidden_dim=flow_hidden_dim, num_layers=num_flow_layers)

        self.sce = SpectralCoordinateEmbedding(in_features=2, out_features=embed_dim, sigma=rff_sigma)

        # 谱特征映射为局部协方差椭圆参数，由 AKN 接收计算
        self.kernel = AdaptiveKernelNetwork(embed_dim=embed_dim, hidden_dim=kernel_hidden_dim)

    def get_trend_matrix(self, coords, X_cov):
        """
        提取外部协变量辅助趋势基矩阵，采用光滑的多变量二次多项式展开
        F = [1, u_x, u_y, X_1, X_2, X_1^2, X_2^2, X_1*X_2]
        coords: [B, N, 2]
        X_cov: [B, N, D_x]
        返回 F: [B, N, 8]
        """
        u_x = coords[:, :, 0:1]           # [B, N, 1]
        u_y = coords[:, :, 1:2]           # [B, N, 1]
        ones = torch.ones_like(u_x)       # [B, N, 1]
        
        X1 = X_cov[:, :, 0:1]             # [B, N, 1]
        X2 = X_cov[:, :, 1:2]             # [B, N, 1]
        X1_sq = X1 ** 2                   # [B, N, 1]
        X2_sq = X2 ** 2                   # [B, N, 1]
        X12 = X1 * X2                     # [B, N, 1]
        
        F = torch.cat([ones, u_x, u_y, X1, X2, X1_sq, X2_sq, X12], dim=-1)  # [B, N, 8]
        return F

    def forward(self, Z_obs, U_obs, U_pred, X_obs, X_pred, Z_pred=None):
        """
        前向计算与预测。
        Z_obs: [B, N, 1]
        U_obs: [B, N, 2]
        U_pred: [B, 1, 2]
        X_obs: [B, N, D_x]
        X_pred: [B, 1, D_x]
        Z_pred: [B, 1, 1]
        """
        B, N, D_in = Z_obs.shape

        # 1. 物理层处理：若输入为 1 维，则拼接 1 维 dummy 零向量，保证流耦合运算在至少 2 维上运行
        if D_in == 1:
            Z_obs_flow = torch.cat([Z_obs, torch.zeros_like(Z_obs)], dim=-1)  # [B, N, 2]
            if Z_pred is not None:
                Z_pred_flow = torch.cat([Z_pred, torch.zeros_like(Z_pred)], dim=-1)  # [B, 1, 2]
                Z_all_flow = torch.cat([Z_obs_flow, Z_pred_flow], dim=1)  # [B, N+1, 2]
            else:
                Z_all_flow = Z_obs_flow   # [B, N, 2]
        else:
            if Z_pred is not None:
                Z_all_flow = torch.cat([Z_obs, Z_pred], dim=1)  # [B, N+1, D]
            else:
                Z_all_flow = Z_obs         # [B, N, D]

        # 2. 运行 RealNVP 前向变换将物理值投影至高斯隐空间
        Y_all_flow, log_det_all = self.flow(Z_all_flow)  # Y_all_flow: [B, N_all, D_flow], log_det_all: [B, N_all]

        Y_obs_flow = Y_all_flow[:, :N, :]  # [B, N, D_flow]

        # 3. 提取空间坐标的谱特征，计算自适应马氏协方差结构
        H_obs = self.sce(U_obs)           # [B, N, embed_dim]
        H_pred = self.sce(U_pred)         # [B, 1, embed_dim]

        C, c_0 = self.kernel(H_obs, H_pred, U_obs, U_pred)  # C: [B, N, N], c_0: [B, N, 1]

        # 4. 构建外部协变量趋势项基矩阵
        F = self.get_trend_matrix(U_obs, X_obs)            # [B, N, 3 + D_x]
        f_0 = self.get_trend_matrix(U_pred, X_pred).transpose(-2, -1)  # [B, 3 + D_x, 1]

        # 5. 调用自定义可微克里金系统求解器，预测隐空间隐变量
        Y_hat_pred_flow = UKSSolverOp.apply(C, F, c_0, f_0, Y_obs_flow, self.eps)  # [B, 1, D_flow]

        # 6. 利用 RealNVP 逆变换将隐变量预测还原回物理空间
        Z_hat_pred_flow = self.flow.inverse(Y_hat_pred_flow)  # [B, 1, D_flow]

        # 提取物理观测维度的预测
        Z_hat = Z_hat_pred_flow[:, :, :D_in]  # [B, 1, D_in]

        if Z_pred is not None:
            return Z_hat, Y_all_flow, log_det_all
        else:
            return Z_hat

    def predict_with_uncertainty(self, Z_obs, U_obs, U_pred, X_obs, X_pred, n_samples_mc=100):
        """
        使用重参数化蒙特卡洛采样 (Reparameterization MC) 进行无偏物理预测与物理估计方差计算。
        通过积分矫正，消除非高斯可逆流映射带来的 Jensen 不等式系统偏差。
        
        参数:
            Z_obs: 已知点物理观测 [B, N, 1]
            U_obs: 已知点物理坐标 [B, N, 2]
            U_pred: 预测点物理坐标 [B, 1, 2]
            X_obs: 已知点协变量 [B, N, D_x]
            X_pred: 预测点协变量 [B, 1, D_x]
            n_samples_mc (int): 蒙特卡洛采样次数 M
            
        返回:
            Z_hat_unbiased: 物理空间无偏估计预测值 [B, 1, 1]
            Z_var_unbiased: 物理估计条件方差 (不确定性场) [B, 1, 1]
        """
        B, N, D_in = Z_obs.shape
        assert D_in == 1, "不确定性方差估计暂只支持 1 维物理观测场"

        # 1. 物理层高斯化预处理：拼接 dummy 零维度
        Z_obs_flow = torch.cat([Z_obs, torch.zeros_like(Z_obs)], dim=-1)  # [B, N, 1] -> [B, N, 2]
        
        # 2. 运行 RealNVP 前向变换投影至高斯潜空间
        Y_obs_flow, _ = self.flow(Z_obs_flow)  # [B, N, 2]
        
        # 3. 提取空间物理坐标的谱特征，计算自适应非平稳马氏协方差结构
        H_obs = self.sce(U_obs)           # [B, N, 2] -> [B, N, embed_dim]
        H_pred = self.sce(U_pred)         # [B, 1, 2] -> [B, 1, embed_dim]
        C, c_0 = self.kernel(H_obs, H_pred, U_obs, U_pred)  # C: [B, N, N], c_0: [B, N, 1]
        
        # 4. 构建外部漂移趋势基底
        F = self.get_trend_matrix(U_obs, X_obs)            # [B, N, 3 + D_x]
        f_0 = self.get_trend_matrix(U_pred, X_pred).transpose(-2, -1)  # [B, 3 + D_x, 1]
        
        # 5. 调用克里金求解算子计算隐空间预测均值
        Y_hat_pred_flow = UKSSolverOp.apply(C, F, c_0, f_0, Y_obs_flow, self.eps)  # [B, 1, 2]
        
        # 6. 计算潜高斯空间的条件插值估计方差 (UKS 估计方差)
        Lambda = UKSSolverOp.saved_weights['Lambda']  # [B, N, 1]
        mu = UKSSolverOp.saved_weights['mu']          # [B, M, 1]
        sigma_f_sq = torch.exp(2.0 * self.kernel.log_sigma_f)  # 标量/张量
        
        term1 = torch.bmm(Lambda.transpose(-2, -1), c_0)  # [B, 1, N] x [B, N, 1] -> [B, 1, 1]
        term2 = torch.bmm(mu.transpose(-2, -1), f_0)      # [B, 1, M] x [B, M, 1] -> [B, 1, 1]
        Y_var = torch.clamp(sigma_f_sq - term1 - term2, min=1e-6)  # [B, 1, 1]
        
        # 7. 重参数化蒙特卡洛采样 (Reparameterization Monte Carlo Sampling)
        # 高斯空间物理维度估计均值与标准差
        Y_mean = Y_hat_pred_flow[:, :, 0:1]  # [B, 1, 1]
        Y_std = torch.sqrt(Y_var)            # [B, 1, 1]
        
        # 采样标准高斯噪声
        eps_rand = torch.randn(B, n_samples_mc, 1, device=Z_obs.device, dtype=torch.float32)  # [B, M_mc, 1]
        
        # 通过重参数化公式生成隐变量样本
        # 广播计算: [B, 1, 1] + [B, M_mc, 1] * [B, 1, 1] -> [B, M_mc, 1]
        Y_samples = Y_mean + eps_rand * Y_std  # [B, M_mc, 1]
        
        # 拼接 dummy 零维度，为投影回物理空间做准备
        Y_samples_flow = torch.cat([Y_samples, torch.zeros_like(Y_samples)], dim=-1)  # [B, M_mc, 1] -> [B, M_mc, 2]
        
        # 8. 通过可逆流逆向投影回物理观测空间
        Z_samples_flow = self.flow.inverse(Y_samples_flow)  # [B, M_mc, 2]
        Z_samples = Z_samples_flow[:, :, 0:1]  # [B, M_mc, 1]
        
        # 9. 计算样本均值与条件物理方差 (通过 keepdim=True 保持三维输出张量格式)
        Z_hat_unbiased = torch.mean(Z_samples, dim=1, keepdim=True)  # [B, M_mc, 1] -> [B, 1, 1]
        Z_var_unbiased = torch.var(Z_samples, dim=1, keepdim=True)    # [B, M_mc, 1] -> [B, 1, 1]
        
        return Z_hat_unbiased, Z_var_unbiased
