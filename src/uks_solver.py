# -*- coding: utf-8 -*-
"""
统一克里金系统 (Unified Kriging System, UKS) 求解器算子。
实现了自定义 PyTorch Autograd 函数，以高效、可微地求解克里金方程。
使用 CPU 临时执行 Cholesky 分解，以规避 Apple Silicon MPS 平台的底层 linalg bug，并提供迭代自适应加噪备用机制与诊断。
统一使用 torch.float32 精度，不含任何混合精度转换。
"""

import torch

class UKSSolverOp(torch.autograd.Function):
    """
    统一克里金系统求解器算子 (Unified Kriging System Solver Operator)。
    
    前向传播 (Forward): 求解对称不定的克里金方程组。
    反向传播 (Backward): 求解伴随状态方程并计算梯度。
    """
    saved_weights = {}

    @staticmethod
    def forward(ctx, C, F, c_0, f_0, Y, eps=1e-5):
        """
        前向传播 (Forward)
        
        参数 (Arguments):
            C: 已知观测点之间的协方差矩阵 (Covariance Matrix), 形状为 [B, N, N], 统一使用 float32
            F: 已知观测点处的趋势基矩阵 (Trend Base Matrix), 形状为 [B, N, M], 统一使用 float32
            c_0: 预测点与已知点之间的协方差向量 (Cross-covariance Vector), 形状为 [B, N, 1], 统一使用 float32
            f_0: 预测点处的趋势基向量 (Trend Base Vector), 形状为 [B, M, 1], 统一使用 float32
            Y: 已知观测点处的隐变量 (Latent Variables), 形状为 [B, N, D], 统一使用 float32
            eps: 协方差矩阵对角线扰动 nugget 扰动 (Regularization eps), 标量
            
        返回 (Returns):
            Y_hat: 预测点处的隐变量估计值 (Predicted Latent Variables), 形状为 [B, 1, D], 统一使用 float32
        """
        ctx.eps = eps

        B, N, M = F.shape

        # 1. 协方差矩阵添加扰动项 (Nugget Effect) 保证严格正定
        # [1, N, N]
        eye_N = torch.eye(N, device=C.device, dtype=torch.float32).unsqueeze(0)
        C_reg = C + eps * eye_N  # [B, N, N]

        # 2. 对对称正定矩阵 C_reg 进行 Cholesky 分解：C_reg = L L^T
        # CUDA 设备下直接在 GPU 显存内极速求解，避免主机-设备拷贝开销
        device = C_reg.device
        if 'cuda' in str(device):
            try:
                L = torch.linalg.cholesky(C_reg)
            except torch._C._LinAlgError:
                fallback_nugget = 1e-4
                L = None
                eye_N_cuda = torch.eye(N, device=device, dtype=torch.float32).unsqueeze(0)
                for _ in range(10):
                    try:
                        L = torch.linalg.cholesky(C_reg + fallback_nugget * eye_N_cuda)
                        break
                    except torch._C._LinAlgError:
                        fallback_nugget *= 5.0
                if L is None:
                    raise torch._C._LinAlgError("Cholesky decomposition of C_reg failed on CUDA even with 10 adaptive fallbacks.")
        else:
            # MPS/CPU 下临时转移至 CPU 规避 Apple Silicon MPS 底层 linalg bug
            C_reg_cpu = C_reg.cpu()
            try:
                L_cpu = torch.linalg.cholesky(C_reg_cpu)
            except torch._C._LinAlgError:
                fallback_nugget = 1e-4
                L_cpu = None
                eye_N_cpu = torch.eye(N, device='cpu', dtype=torch.float32).unsqueeze(0)
                for _ in range(10):
                    try:
                        L_cpu = torch.linalg.cholesky(C_reg_cpu + fallback_nugget * eye_N_cpu)
                        break
                    except torch._C._LinAlgError:
                        fallback_nugget *= 5.0
                if L_cpu is None:
                    raise torch._C._LinAlgError("Cholesky decomposition of C_reg failed on CPU even with 10 adaptive fallbacks.")
            L = L_cpu.to(device)  # [B, N, N]

        # 3. 求解 L V = F (得到 V) 以及 L v_0 = c_0 (得到 v_0)
        V = torch.linalg.solve_triangular(L, F, upper=False)  # [B, N, M]
        v_0 = torch.linalg.solve_triangular(L, c_0, upper=False)  # [B, N, 1]

        # 4. 求解 (V^T V) mu = V^T v_0 - f_0
        V_T = V.transpose(-2, -1)  # [B, N, M] -> [B, M, N]
        V_T_V = torch.bmm(V_T, V)  # [B, M, N] x [B, N, M] -> [B, M, M]
        
        # 为 V_T_V 添加微小的对角线扰动，确保其 Cholesky 分解的数值稳定性
        # [1, M, M]
        eye_M = torch.eye(M, device=C.device, dtype=torch.float32).unsqueeze(0)
        V_T_V_reg = V_T_V + 1e-6 * eye_M  # [B, M, M]
        
        # CUDA/MPS 设备自适应分解
        if 'cuda' in str(device):
            try:
                L_V = torch.linalg.cholesky(V_T_V_reg)
            except torch._C._LinAlgError:
                fallback_nugget = 1e-4
                L_V = None
                eye_M_cuda = torch.eye(M, device=device, dtype=torch.float32).unsqueeze(0)
                while fallback_nugget <= 0.5:
                    try:
                        L_V = torch.linalg.cholesky(V_T_V_reg + fallback_nugget * eye_M_cuda)
                        break
                    except torch._C._LinAlgError:
                        fallback_nugget *= 5
                if L_V is None:
                    raise torch._C._LinAlgError("Cholesky decomposition of V^T V failed on CUDA even with adaptive fallbacks.")
        else:
            # 同样在 CPU 上对 V^T V 进行 Cholesky，带迭代加噪防护
            V_T_V_reg_cpu = V_T_V_reg.cpu()
            try:
                L_V_cpu = torch.linalg.cholesky(V_T_V_reg_cpu)
            except torch._C._LinAlgError:
                fallback_nugget = 1e-4
                L_V_cpu = None
                eye_M_cpu = torch.eye(M, device='cpu', dtype=torch.float32).unsqueeze(0) # [1, M, M]
                while fallback_nugget <= 0.5:
                    try:
                        L_V_cpu = torch.linalg.cholesky(V_T_V_reg_cpu + fallback_nugget * eye_M_cpu)
                        break
                    except torch._C._LinAlgError:
                        fallback_nugget *= 5
                if L_V_cpu is None:
                    raise torch._C._LinAlgError("Cholesky decomposition of V^T V failed even with adaptive fallbacks.")
            L_V = L_V_cpu.to(device)  # [B, M, M]
        
        # 右端项 rhs_mu = V^T v_0 - f_0
        # [B, M, N] x [B, N, 1] -> [B, M, 1]
        rhs_mu = torch.bmm(V_T, v_0) - f_0  # [B, M, 1]
        
        # 双步解求出拉格朗日乘子 mu
        mu_temp = torch.linalg.solve_triangular(L_V, rhs_mu, upper=False)  # [B, M, 1]
        mu = torch.linalg.solve_triangular(L_V.transpose(-2, -1), mu_temp, upper=True)  # [B, M, 1]

        # 5. 求解 L L^T Lambda = c_0 - F mu
        # [B, N, M] x [B, M, 1] -> [B, N, 1]
        rhs_Lambda = c_0 - torch.bmm(F, mu)  # [B, N, 1]
        
        # 分步解前代和回代
        z = torch.linalg.solve_triangular(L, rhs_Lambda, upper=False)  # [B, N, 1]
        Lambda = torch.linalg.solve_triangular(L.transpose(-2, -1), z, upper=True)  # [B, N, 1]

        # 保存插值权重 Lambda 和拉格朗日乘子 mu 到类属性，以备伴随梯度与方差计算使用
        UKSSolverOp.saved_weights['Lambda'] = Lambda.detach()
        UKSSolverOp.saved_weights['mu'] = mu.detach()

        # 6. 计算预测值 Y_hat = Lambda^T Y
        # [B, 1, N] x [B, N, D] -> [B, 1, D]
        Y_hat = torch.bmm(Lambda.transpose(-2, -1), Y)  # [B, 1, D]

        # 7. 保存前向计算因子以备反向传播使用
        ctx.save_for_backward(L, V, L_V, Lambda, mu, Y, F)

        return Y_hat

    @staticmethod
    def backward(ctx, g_Y_hat):
        """
        反向传播 (Backward)
        """
        eps = ctx.eps

        # 恢复前向计算得到的变量
        L, V, L_V, Lambda, mu, Y, F = ctx.saved_tensors

        # 1. 计算对权重的梯度 g_Lambda 以及对已知观测的梯度 g_Y
        # [B, N, D] x [B, D, 1] -> [B, N, 1]
        g_Lambda = torch.bmm(Y, g_Y_hat.transpose(-2, -1))  # [B, N, 1]
        # [B, N, 1] x [B, 1, D] -> [B, N, D]
        g_Y = torch.bmm(Lambda, g_Y_hat)  # [B, N, D]

        # 2. 求解伴随状态方程: K lambda_adj = [g_Lambda; 0]
        # L w = g_Lambda
        w = torch.linalg.solve_triangular(L, g_Lambda, upper=False)  # [B, N, 1]

        # (V^T V) lambda_F = V^T w
        # [B, M, N] x [B, N, 1] -> [B, M, 1]
        rhs_lambda_F = torch.bmm(V.transpose(-2, -1), w)  # [B, M, 1]
        
        # 复用前向的 L_V 分解因子
        lambda_F_temp = torch.linalg.solve_triangular(L_V, rhs_lambda_F, upper=False)  # [B, M, 1]
        lambda_F = torch.linalg.solve_triangular(L_V.transpose(-2, -1), lambda_F_temp, upper=True)  # [B, M, 1]

        # L L^T lambda_C = g_Lambda - F lambda_F
        # [B, N, M] x [B, M, 1] -> [B, N, 1]
        rhs_lambda_C = g_Lambda - torch.bmm(F, lambda_F)  # [B, N, 1]
        z_adj = torch.linalg.solve_triangular(L, rhs_lambda_C, upper=False)  # [B, N, 1]
        lambda_C = torch.linalg.solve_triangular(L.transpose(-2, -1), z_adj, upper=True)  # [B, N, 1]

        # 保存伴随变量 lambda_C 到类属性，以备伴随梯度提取使用
        UKSSolverOp.saved_weights['lambda_C'] = lambda_C.detach()

        # 3. 显式计算关于输入参数的梯度
        # C 的梯度: g_C = -0.5 * (lambda_C @ Lambda^T + Lambda @ lambda_C^T)
        # [B, N, 1] x [B, 1, N] -> [B, N, N]
        g_C = -0.5 * (torch.bmm(lambda_C, Lambda.transpose(-2, -1)) + torch.bmm(Lambda, lambda_C.transpose(-2, -1)))  # [B, N, N]

        # F 的梯度: g_F = -(lambda_C @ mu^T + Lambda @ lambda_F^T)
        # [B, N, 1] x [B, 1, M] -> [B, N, M]
        g_F = -(torch.bmm(lambda_C, mu.transpose(-2, -1)) + torch.bmm(Lambda, lambda_F.transpose(-2, -1)))  # [B, N, M]

        # c_0 的梯度: g_c0 = lambda_C
        g_c0 = lambda_C  # [B, N, 1]

        # f_0 的梯度: g_f0 = lambda_F
        g_f0 = lambda_F  # [B, M, 1]

        # 4. 防御性检查，确保计算中没有 NaN/Inf
        if torch.isnan(g_C).any() or torch.isinf(g_C).any():
            g_C = torch.nan_to_num(g_C, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(g_F).any() or torch.isinf(g_F).any():
            g_F = torch.nan_to_num(g_F, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(g_c0).any() or torch.isinf(g_c0).any():
            g_c0 = torch.nan_to_num(g_c0, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(g_f0).any() or torch.isinf(g_f0).any():
            g_f0 = torch.nan_to_num(g_f0, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(g_Y).any() or torch.isinf(g_Y).any():
            g_Y = torch.nan_to_num(g_Y, nan=0.0, posinf=1e6, neginf=-1e6)

        return (
            g_C,
            g_F,
            g_c0,
            g_f0,
            g_Y,
            None  # eps 无梯度
        )
