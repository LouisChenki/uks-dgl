# -*- coding: utf-8 -*-
"""
统一克里金系统 (Unified Kriging System, UKS) 神经网络模型。
集成了多通道联合可逆流高斯化 (Joint RealNVP)、谱变形坐标嵌入 (SCE)、
以及基于线性共区域化模型 (LMC) 理论同构的多头自适应空间互协方差网络。
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
DROPOUT_P = 0.05
NUGGET_EPS = 1.0e-06
L2_MAX_LIMIT = 0.08
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

        # 2层 MLP 用于计算缩放与平移参数，使用 GELU 激活与配置区定义的 DROPOUT_P
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(hidden_dim, out_dim * 2)
        )

    def forward(self, x):
        # x 形状为 [B, N, dim] 或 [B, dim]
        # 获取后备通道划分
        d = self.dim // 2
        
        if self.mask_type == 'even':
            x1, x2 = x[..., :d], x[..., d:]
        else:
            x2, x1 = x[..., :d], x[..., d:]

        # 前向流：计算对 x1 的映射参数，并作用到 x2
        params = self.net(x1)  # [..., out_dim * 2]
        s, t = torch.chunk(params, 2, dim=-1)
        s = torch.tanh(s)  # 约束缩放防止梯度爆炸

        y1 = x1
        y2 = x2 * torch.exp(s) + t

        if self.mask_type == 'even':
            y = torch.cat([y1, y2], dim=-1)
        else:
            y = torch.cat([y2, y1], dim=-1)

        # 雅可比行列式对数项 (Log-Jacobian Determinant) = sum(s)
        log_det = torch.sum(s, dim=-1)
        return y, log_det

    def inverse(self, y):
        d = self.dim // 2
        if self.mask_type == 'even':
            y1, y2 = y[..., :d], y[..., d:]
        else:
            y2, y1 = y[..., :d], y[..., d:]

        params = self.net(y1)
        s, t = torch.chunk(params, 2, dim=-1)
        s = torch.tanh(s)

        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        if self.mask_type == 'even':
            x = torch.cat([x1, x2], dim=-1)
        else:
            x = torch.cat([x2, x1], dim=-1)
        return x


class RealNVP(nn.Module):
    """
    可逆实值非体积保持流网络 (RealNVP)。
    通过堆叠多个耦合层，实现任意复杂概率分布到标准高斯分布之间的双射映射。
    """
    def __init__(self, dim=2, hidden_dim=32, num_layers=4):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        # 交替使用奇偶遮罩 (Alternating Mask Types)
        for i in range(num_layers):
            mask_type = 'even' if i % 2 == 0 else 'odd'
            self.layers.append(CouplingLayer(dim, hidden_dim, mask_type))

    def forward(self, x):
        # x 形状为 [B, N, dim]
        log_det_total = torch.zeros(x.shape[:-1], device=x.device, dtype=torch.float32)
        out = x
        for layer in self.layers:
            out, log_det = layer(out)
            log_det_total += log_det
        return out, log_det_total

    def inverse(self, y):
        # y 形状为 [B, N, dim] 或 [B, M_mc, dim]
        out = y
        for layer in reversed(self.layers):
            out = layer.inverse(out)
        return out


class SpectralCoordinateEmbedding(nn.Module):
    """
    流形变形谱坐标嵌入层 (Manifold-Warping Spectral Coordinate Embedding, MW-SCE)。
    包含空间变形 (Spatial Warping) 与 Random Fourier Features (RFF) 谱映射，以感知复杂的非平稳旋转边界与局部高频突变。
    """
    def __init__(self, in_features=2, out_features=16, sigma=10.0):
        super().__init__()
        self.out_features = out_features
        # 1. 空间变形层 (Spatial Warping MLP)
        self.warping_net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.GELU(),
            nn.Linear(16, in_features)
        )
        # 2. 谱编码层：固定频率矩阵 B ~ N(0, sigma^2)
        # 预留 2 维给原始物理坐标残差直连
        half_dim = (out_features - 2) // 2
        B_freq = torch.randn(in_features, half_dim) * sigma
        self.register_buffer("B_freq", B_freq)
        
        # 3. 特征投影层 (Feature Projection Layer) 对齐最终 16 维表示
        self.projection = nn.Sequential(
            nn.Linear(out_features, 32),
            nn.GELU(),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(32, out_features)
        )

    def forward(self, coords):
        # coords: [B, N, 2]
        # 计算 2D 空间位置变形位移并保存，以便计算 warping 几何正则损失
        warp = self.warping_net(coords) # [B, N, 2] -> [B, N, 2] 维度追踪 (Dimension Tracking)
        self.last_warp = warp
        
        # 2D 空间位置变形：u_warped = u + MLP(u)
        warped_coords = coords + warp # [B, N, 2] -> [B, N, 2] 维度追踪 (Dimension Tracking)
        
        # 计算 2D 投影，投影矩阵乘积 [B, N, 2] x [2, half_dim] -> [B, N, half_dim]
        proj = torch.matmul(warped_coords, self.B_freq)
        cos_proj = torch.cos(2.0 * math.pi * proj)
        sin_proj = torch.sin(2.0 * math.pi * proj)

        rff = torch.cat([cos_proj, sin_proj], dim=-1) # [B, N, out_features - 2] -> [B, N, 14] 维度追踪 (Dimension Tracking)
        
        # 拼接 RFF 特征与 2 维原始物理坐标 0.1 * coords 的残差特征
        res_features = torch.cat([rff, 0.1 * coords], dim=-1) # [B, N, out_features] -> [B, N, 16] 维度追踪 (Dimension Tracking)
        
        # 应用特征投影层做非线性特征投影
        out_features = self.projection(res_features) # [B, N, out_features] -> [B, N, 16] 维度追踪 (Dimension Tracking)
        return out_features


class AdaptiveKernelNetwork(nn.Module):
    """
    双头 LMC 协同克里金自适应核网络 (Adaptive Kernel Network)。
    对两个 Attention Head 独立拟合局部马氏几何矩阵，并分别建立半正定的共区域化系数矩阵 B_h，
    最终累加生成多通道空间互协方差矩阵 C (维度 Nq x Nq)。
    """
    def __init__(self, embed_dim=16, hidden_dim=32, num_heads=2, q=2, force_isotropic=False):
        super().__init__()
        self.num_heads = num_heads
        self.q = q
        self.force_isotropic = force_isotropic
        
        # Sequential MLP 为每个 Head 预测几何参数：t(旋转相关), p1(主轴相关), p2(次轴相关)
        # 输出维度为 3 * num_heads
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(hidden_dim, 3 * num_heads)
        )
        
        # 每个 Head 独立的可学习边际尺度参数
        self.log_sigma_f = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        
        # 每个 Head 独立的线性共区域化参数矩阵 W_h 维度 [q, q]，用于构建半正定 B_h = W_h W_h^T
        self.W = nn.Parameter(torch.randn(num_heads, q, q, dtype=torch.float32) * 0.1 + torch.eye(q).unsqueeze(0))
        
        # 可学习度量混合调节阀 log_gamma，初始化为 -2.5 (各向同性约占 7.5%)
        self.log_gamma = nn.Parameter(torch.tensor([-2.5], dtype=torch.float32))

    def get_local_metric_matrix(self, H, head_idx):
        """
        提取特定 Head 对应的局部各向异性 Riemannian 度量矩阵，并进行各向同性混合
        """
        if getattr(self, 'force_isotropic', False):
            # 若开启强制各向同性退化，返回自适应对角各向同性度量矩阵
            params = self.mlp(H)
            p1 = params[:, :, head_idx * 3 + 1]
            l_iso = 0.08 + (L2_MAX_LIMIT - 0.08) * torch.sigmoid(p1)
            inv_l_sq = 1.0 / (l_iso ** 2)
            
            g11 = inv_l_sq
            g12 = torch.zeros_like(g11)
            g22 = inv_l_sq
            
            G_row1 = torch.stack([g11, g12], dim=-1)
            G_row2 = torch.stack([g12, g22], dim=-1)
            G = torch.stack([G_row1, G_row2], dim=-2)
            
            gamma = torch.sigmoid(self.log_gamma)
            eye = torch.eye(2, device=G.device).view(1, 1, 2, 2)
            G_mixed = (1.0 - gamma) * G + gamma * 16.0 * eye
            return G_mixed
            
            
        params = self.mlp(H)  # [B, N, 3 * num_heads] -> [B, N, 6] 维度追踪 (Dimension Tracking)
        t = params[:, :, head_idx * 3]
        p1 = params[:, :, head_idx * 3 + 1]
        p2 = params[:, :, head_idx * 3 + 2]

        theta = math.pi * torch.tanh(t)
        l1 = 0.08 + (0.50 - 0.08) * torch.sigmoid(p1)
        l2 = 0.03 + (L2_MAX_LIMIT - 0.03) * torch.sigmoid(p2)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        inv_l1_sq = 1.0 / (l1 ** 2)
        inv_l2_sq = 1.0 / (l2 ** 2)

        g11 = cos_t**2 * inv_l1_sq + sin_t**2 * inv_l2_sq
        g12 = cos_t * sin_t * (inv_l1_sq - inv_l2_sq)
        g22 = sin_t**2 * inv_l1_sq + cos_t**2 * inv_l2_sq

        G_row1 = torch.stack([g11, g12], dim=-1)  # [B, N, 2] -> [B, N, 2] 维度追踪 (Dimension Tracking)
        G_row2 = torch.stack([g12, g22], dim=-1)  # [B, N, 2] -> [B, N, 2] 维度追踪 (Dimension Tracking)
        G = torch.stack([G_row1, G_row2], dim=-2)  # [B, N, 2, 2] -> [B, N, 2, 2] 维度追踪 (Dimension Tracking)
        
        # 使用 log_gamma 对 Riemannian 局部各向异性度量矩阵与各向同性对角矩阵 16.0 * I 线性混合
        gamma = torch.sigmoid(self.log_gamma)
        eye = torch.eye(2, device=G.device).view(1, 1, 2, 2)
        G_mixed = (1.0 - gamma) * G + gamma * 16.0 * eye # [B, N, 2, 2] -> [B, N, 2, 2] 维度追踪 (Dimension Tracking)
        return G_mixed

    def forward(self, H_obs, H_pred, U_obs, U_pred):
        """
        计算多变量分块互协方差矩阵 C [B, Nq, Nq] 与预测点互协方差向量 c_0 [B, Nq, q]。
        """
        B, N, _ = H_obs.shape
        q = self.q
        
        # 1. 初始化各分块
        # 对于 q=2，分块元素 C_uv 为 [B, N, N]
        C_blocks = [[torch.zeros(B, N, N, device=U_obs.device) for _ in range(q)] for _ in range(q)]
        c0_blocks = [[torch.zeros(B, N, 1, device=U_obs.device) for _ in range(q)] for _ in range(q)]
        
        # 2. 遍历 Head 累加自相关与互相关特征
        for h in range(self.num_heads):
            G_obs = self.get_local_metric_matrix(H_obs, h)    # [B, N, 2, 2]
            G_pred = self.get_local_metric_matrix(H_pred, h)  # [B, 1, 2, 2]
            
            # 2.1 计算已知点间对称距离平方 dist_sq_C
            coords_diff_C = U_obs.unsqueeze(2) - U_obs.unsqueeze(1)  # [B, N, N, 2]
            temp_diff_C = torch.einsum('bijk,bikm->bijm', coords_diff_C, G_obs)
            dist_sq_i = torch.sum(coords_diff_C * temp_diff_C, dim=-1)
            dist_sq_C = 0.5 * (dist_sq_i + dist_sq_i.transpose(-2, -1))  # [B, N, N]
            K_C = torch.exp(-0.5 * dist_sq_C)  # 空间自相关矩阵 [B, N, N]
            
            # 2.2 计算已知与预测点间对称距离平方 dist_sq_c0
            coords_diff_c0 = U_obs.unsqueeze(2) - U_pred.unsqueeze(1)  # [B, N, 1, 2]
            temp_diff_c0_obs = torch.einsum('bijk,bikm->bijm', coords_diff_c0, G_obs)
            dist_sq_c0_obs = torch.sum(coords_diff_c0 * temp_diff_c0_obs, dim=-1)
            temp_diff_c0_pred = torch.einsum('bijk,bikm->bijm', coords_diff_c0, G_pred.expand(-1, N, -1, -1))
            dist_sq_c0_pred = torch.sum(coords_diff_c0 * temp_diff_c0_pred, dim=-1)
            dist_sq_c0 = 0.5 * (dist_sq_c0_obs + dist_sq_c0_pred)  # [B, N, 1]
            k_c0 = torch.exp(-0.5 * dist_sq_c0) # 空间互相关向量 [B, N, 1]
            
            # 2.3 提取半正定共区域化系数 B_h = W_h W_h^T
            W_h = self.W[h]  # [q, q]
            B_h = torch.matmul(W_h, W_h.transpose(-2, -1))  # [q, q]
            
            sigma_f_sq = torch.exp(2.0 * self.log_sigma_f[h])
            
            # 2.4 将本 Head 分量累加至 LMC 分块矩阵中
            for u in range(q):
                for v in range(q):
                    coef = B_h[u, v] * sigma_f_sq
                    C_blocks[u][v] = C_blocks[u][v] + coef * K_C
                    c0_blocks[u][v] = c0_blocks[u][v] + coef * k_c0
                    
        # 3. 拼接分块矩阵得到完整的互协方差系统
        # 拼装 C [B, Nq, Nq]
        C_rows = []
        for u in range(q):
            C_rows.append(torch.cat(C_blocks[u], dim=-1)) # [B, N, Nq]
        C = torch.cat(C_rows, dim=-2) # [B, Nq, Nq]
        
        # 拼装 c_0 [B, Nq, q]
        # 通道 v 对应的列 [B, Nq, 1] 为观测各通道与预测点 v 的互协方差堆叠
        c0_cols = []
        for v in range(q):
            c0_col_v = torch.cat([c0_blocks[u][v] for u in range(q)], dim=-2)  # [B, Nq, 1]
            c0_cols.append(c0_col_v)
        c_0 = torch.cat(c0_cols, dim=-1) # [B, Nq, q]
        
        # 显式对称化以完全消除浮点误差
        C = 0.5 * (C + C.transpose(-2, -1))  # [B, Nq, Nq]
        return C, c_0


class UKSModel(nn.Module):
    """
    统一克里金系统 (Unified Kriging System, UKS) 多变量协同插值神经网络。
    集成了多通道联合 Flow、谱变形 SCE 编码、自适应 LMC 共区域化各向异性核以及可微多通道协同求解器。
    """
    def __init__(self, in_dim=2, flow_hidden_dim=32, num_flow_layers=2,
                 embed_dim=16, rff_sigma=10.0,
                 kernel_hidden_dim=32, latent_dim=8, eps=None, cov_dim=2, trend_type='quadratic',
                 force_isotropic=False):
        super().__init__()
        self.in_dim = in_dim # 通道数 q = 2
        self.eps = eps if eps is not None else NUGGET_EPS
        self.q = in_dim
        self.trend_type = trend_type
        
        # 1. 联合流模型通道 q
        self.flow = RealNVP(dim=in_dim, hidden_dim=flow_hidden_dim, num_layers=num_flow_layers)
        # 2. 谱特征提取层
        self.sce = SpectralCoordinateEmbedding(in_features=2, out_features=embed_dim, sigma=rff_sigma)
        # 3. 双头 LMC 各向异性核
        self.kernel = AdaptiveKernelNetwork(embed_dim=embed_dim, hidden_dim=kernel_hidden_dim, num_heads=2, q=in_dim, force_isotropic=force_isotropic)

    def get_single_trend_matrix(self, coords):
        """
        计算单通道坐标设计基底：支持常数、线性或二阶平面多项式趋势面 (Dimension dynamic)
        """
        u_x = coords[:, :, 0:1]      # [B, N, 1] -> [B, N, 1] 维度追踪 (Dimension Tracking)
        u_y = coords[:, :, 1:2]      # [B, N, 1] -> [B, N, 1] 维度追踪 (Dimension Tracking)
        ones = torch.ones_like(u_x)  # [B, N, 1] -> [B, N, 1] 维度追踪 (Dimension Tracking)
        
        trend_t = getattr(self, 'trend_type', 'quadratic')
        if trend_t == 'constant':
            return ones
        elif trend_t == 'linear':
            return torch.cat([ones, u_x, u_y], dim=-1)  # [B, N, 3] -> [B, N, 3] 维度追踪 (Dimension Tracking)
        else:
            # 引入二阶项以应对非线性大尺度起伏趋势，解耦更平滑
            u_xx = coords[:, :, 0:1] ** 2  # [B, N, 1] -> [B, N, 1] 维度追踪 (Dimension Tracking)
            u_yy = coords[:, :, 1:2] ** 2  # [B, N, 1] -> [B, N, 1] 维度追踪 (Dimension Tracking)
            u_xy = coords[:, :, 0:1] * coords[:, :, 1:2]  # [B, N, 1] -> [B, N, 1] 维度追踪 (Dimension Tracking)
            
            F_0 = torch.cat([ones, u_x, u_y, u_xx, u_yy, u_xy], dim=-1)  # [B, N, 6] -> [B, N, 6] 维度追踪 (Dimension Tracking)
            return F_0

    def get_block_trend_matrix(self, F_0):
        """
        将单通道设计矩阵 F_0 [B, N, 6] 拼装为多通道分块对角矩阵 F [B, Nq, q(L+1)] (即 B x 2N x 12)
        """
        B, N, L = F_0.shape
        zeros = torch.zeros_like(F_0)  # [B, N, 6] -> [B, N, 6] 维度追踪 (Dimension Tracking)
        
        # 对于 q=2 分块对角拼接，二阶多项式特征维度 L = 6，拼接后为 12 维
        F_row1 = torch.cat([F_0, zeros], dim=-1) # [B, N, 12] -> [B, N, 12] 维度追踪 (Dimension Tracking)
        F_row2 = torch.cat([zeros, F_0], dim=-1) # [B, N, 12] -> [B, N, 12] 维度追踪 (Dimension Tracking)
        F = torch.cat([F_row1, F_row2], dim=-2)  # [B, 2N, 12] -> [B, 2N, 12] 维度追踪 (Dimension Tracking)
        return F

    def forward(self, Z_obs, U_obs, U_pred, X_obs, X_pred, Z_pred=None):
        """
        前向计算联合高斯化和克里金协同预测。
        Z_obs: [B, N, q] (联合物理观测)
        U_obs: [B, N, 2]
        U_pred: [B, 1, 2]
        X_obs: [B, N, D_x] (废弃， dummy 对齐)
        X_pred: [B, 1, D_x] (废弃， dummy 对齐)
        Z_pred: [B, 1, q]
        """
        B, N, q = Z_obs.shape
        
        # 1. 拼接全观测数据进行联合 Flow 高斯变换
        if Z_pred is not None:
            Z_all_flow = torch.cat([Z_obs, Z_pred], dim=1)  # [B, N+1, q]
        else:
            Z_all_flow = Z_obs  # [B, N, q]
            
        Y_all_flow, log_det_all = self.flow(Z_all_flow)  # [B, N_all, q], [B, N_all]
        Y_obs_flow = Y_all_flow[:, :N, :]  # [B, N, q]
        
        # 2. 将高斯潜变量在空间与通道维度一并堆叠为列向量 𝕐 [B, Nq, 1]
        # 转置为 [B, q, N] 后 reshape 确保各通道内点坐标的连续排列
        Y_stacked = Y_obs_flow.transpose(-2, -1).reshape(B, N * q, 1)  # [B, Nq, 1]
        
        # 3. 自适应互协方差 C [B, Nq, Nq] 与 c_0 [B, Nq, q] 计算
        H_obs = self.sce(U_obs)    # [B, N, embed_dim]
        H_pred = self.sce(U_pred)  # [B, 1, embed_dim]
        C, c_0 = self.kernel(H_obs, H_pred, U_obs, U_pred)
        
        # 4. 拼装分块对角趋势矩阵 F [B, Nq, q(L+1)] (即 B x 2N x 12) 与 f_0 [B, q(L+1), q] (即 B x 12 x 2)
        F_0 = self.get_single_trend_matrix(U_obs)  # [B, N, 6] -> [B, N, 6] 维度追踪 (Dimension Tracking)
        F = self.get_block_trend_matrix(F_0)      # [B, 2*N, 12] -> [B, 2*N, 12] 维度追踪 (Dimension Tracking)
        
        f0_pred = self.get_single_trend_matrix(U_pred).transpose(-2, -1)  # [B, 6, 1] -> [B, 6, 1] 维度追踪 (Dimension Tracking)
        zeros_pred = torch.zeros_like(f0_pred)  # [B, 6, 1] -> [B, 6, 1] 维度追踪 (Dimension Tracking)
        f_row1 = torch.cat([f0_pred, zeros_pred], dim=-1) # [B, 6, 2] -> [B, 6, 2] 维度追踪 (Dimension Tracking)
        f_row2 = torch.cat([zeros_pred, f0_pred], dim=-1) # [B, 6, 2] -> [B, 6, 2] 维度追踪 (Dimension Tracking)
        f_0 = torch.cat([f_row1, f_row2], dim=-2)         # [B, 12, 2] -> [B, 12, 2] 维度追踪 (Dimension Tracking)
        
        # 5. 调用可微求解器，求解多通道插值权重与估计值
        # 返回多通道联合预测值 Y_hat_stacked: [B, q, 1]
        Y_hat_stacked = UKSSolverOp.apply(C, F, c_0, f_0, Y_stacked, self.eps)
        
        # 6. 转置回 RealNVP 输入维度 [B, 1, q]
        Y_hat_pred_flow = Y_hat_stacked.transpose(-2, -1)  # [B, 1, q]
        
        # 7. 逆映射还原回物理值 Z_hat [B, 1, q]
        Z_hat_pred_flow = self.flow.inverse(Y_hat_pred_flow) # [B, 1, q]
        
        if Z_pred is not None:
            return Z_hat_pred_flow, Y_all_flow, log_det_all
        else:
            return Z_hat_pred_flow

    def predict_with_uncertainty(self, Z_obs, U_obs, U_pred, X_obs, X_pred, n_samples_mc=100):
        """
        多通道协同克里金 MC 积分无偏估计与条件方差计算。
        """
        B, N, q = Z_obs.shape
        
        # 1. 流变换高斯化得到潜变量
        Y_obs_flow, _ = self.flow(Z_obs)  # [B, N, q]
        Y_stacked = Y_obs_flow.transpose(-2, -1).reshape(B, N * q, 1)  # [B, Nq, 1]
        
        # 2. 互协方差与设计矩阵构建
        H_obs = self.sce(U_obs)
        H_pred = self.sce(U_pred)
        C, c_0 = self.kernel(H_obs, H_pred, U_obs, U_pred)  # C: [B, 2N, 2N], c_0: [B, 2N, 2]
        
        F_0 = self.get_single_trend_matrix(U_obs)  # [B, N, 6] -> [B, N, 6] 维度追踪 (Dimension Tracking)
        F = self.get_block_trend_matrix(F_0)  # [B, 2*N, 12] -> [B, 2*N, 12] 维度追踪 (Dimension Tracking)
        
        f0_pred = self.get_single_trend_matrix(U_pred).transpose(-2, -1)  # [B, 6, 1] -> [B, 6, 1] 维度追踪 (Dimension Tracking)
        zeros_pred = torch.zeros_like(f0_pred)  # [B, 6, 1] -> [B, 6, 1] 维度追踪 (Dimension Tracking)
        f_row1 = torch.cat([f0_pred, zeros_pred], dim=-1)  # [B, 6, 2] -> [B, 6, 2] 维度追踪 (Dimension Tracking)
        f_row2 = torch.cat([zeros_pred, f0_pred], dim=-1)  # [B, 6, 2] -> [B, 6, 2] 维度追踪 (Dimension Tracking)
        f_0 = torch.cat([f_row1, f_row2], dim=-2)  # [B, 12, 2] -> [B, 12, 2] 维度追踪 (Dimension Tracking)
        
        # 3. 求解潜空间均值预测
        Y_hat_stacked = UKSSolverOp.apply(C, F, c_0, f_0, Y_stacked, self.eps) # [B, 2, 1]
        
        # 4. 计算潜高斯空间的条件插值协方差矩阵 Y_var [B, q, q]
        Lambda = UKSSolverOp.saved_weights['Lambda']  # [B, Nq, q]
        mu = UKSSolverOp.saved_weights['mu']          # [B, 12, q]
        
        # 预测点自身的潜协方差 C_0 [B, q, q] = sum(B_h)
        C_0 = torch.zeros(B, q, q, device=Z_obs.device)
        for h in range(self.kernel.num_heads):
            W_h = self.kernel.W[h]
            B_h = torch.matmul(W_h, W_h.transpose(-2, -1))
            sigma_f_sq = torch.exp(2.0 * self.kernel.log_sigma_f[h])
            C_0 += B_h * sigma_f_sq
            
        term1 = torch.bmm(Lambda.transpose(-2, -1), c_0)  # [B, q, Nq] x [B, Nq, q] -> [B, q, q]
        term2 = torch.bmm(mu.transpose(-2, -1), f_0)      # [B, q, 12] x [B, 12, q] -> [B, q, q]
        
        Y_var = C_0 - term1 - term2  # [B, q, q]
        
        # 加上微小扰动确保严格正定并 Cholesky 分解，引入自适应防灾与对角矩阵兜底
        eye_q = torch.eye(q, device=Z_obs.device).unsqueeze(0)
        L_var = None
        fallback_nugget = 1e-5
        for _ in range(10):
            try:
                L_var = torch.linalg.cholesky(Y_var + fallback_nugget * eye_q)
                break
            except torch._C._LinAlgError:
                fallback_nugget *= 5.0
                
        if L_var is None:
            # 极端奇异或非正定兜底：退化为独立物理信道的经验方差对角矩阵
            diag_vars = torch.clamp(torch.diagonal(Y_var, dim1=-2, dim2=-1), min=1e-6)
            L_var = torch.diag_embed(torch.sqrt(diag_vars))  # [B, q, q]
        
        # 5. 重参数化蒙特卡洛采样
        # 采样标准噪声 eps: [B, M_mc, q, 1]
        eps_rand = torch.randn(B, n_samples_mc, q, 1, device=Z_obs.device, dtype=torch.float32)
        
        # 计算采样点: Y_samples = Y_mean + L_var * eps
        Y_mean = Y_hat_stacked.unsqueeze(1)  # [B, 1, q, 1]
        L_var_expanded = L_var.unsqueeze(1)  # [B, 1, q, q]
        
        # 广播相乘: [B, 1, q, q] x [B, M_mc, q, 1] -> [B, M_mc, q, 1]
        Y_samples_stacked = Y_mean + torch.matmul(L_var_expanded, eps_rand)  # [B, M_mc, q, 1]
        Y_samples_flow = Y_samples_stacked.squeeze(-1) # [B, M_mc, q]
        
        # 6. 通过流逆变换还原物理观测样本
        Z_samples_flow = self.flow.inverse(Y_samples_flow) # [B, M_mc, q]
        
        # 7. 统计计算主变量（第 0 通道）的无偏估计和方差
        Z_hat_unbiased = torch.mean(Z_samples_flow[:, :, 0:1], dim=1, keepdim=True) # [B, 1, 1]
        Z_var_unbiased = torch.var(Z_samples_flow[:, :, 0:1], dim=1, keepdim=True)   # [B, 1, 1]
        
        return Z_hat_unbiased, Z_var_unbiased
