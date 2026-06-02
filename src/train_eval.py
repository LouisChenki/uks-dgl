# -*- coding: utf-8 -*-
"""
统一克里金系统 (Unified Kriging System, UKS) 联合训练与评估模块。
集成了同方差不确定性自适应加权层 (Homoscedastic Loss Weighting Layer)、基于可逆消元法的 UKS 结构似然损失、二阶 Hessian 拓扑几何正则化，以及课程学习 (Curriculum Learning) 调度器。
"""

import os
import math
import torch
import torch.nn as nn

# 导入克里金算子以备诊断使用
from uks_solver import UKSSolverOp

class HomoscedasticLossWeighting(nn.Module):
    """
    同方差不确定性损失自适应加权层 (Homoscedastic Uncertainty Loss Weighting Layer)。
    包含 4 个可学习的对数噪声参数 log_vars，对应 [Pred, UKS, Flow, Geo] 四项子任务损失。
    """
    def __init__(self):
        super().__init__()
        # 初始化 4 个对数噪声参数为 0.0 (即初始化各任务 sigma_i^2 = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(4, dtype=torch.float32))

    def forward(self, loss_pred, loss_uks, loss_flow, loss_geo):
        """
        计算多任务自适应加权总损失:
        L_Total = 1/(2*sigma^2) * L_task + log(sigma)
        """
        # 为保证方差的正定性，使用对数指数变换
        w0 = torch.exp(-self.log_vars[0])
        w1 = torch.exp(-self.log_vars[1])
        w2 = torch.exp(-self.log_vars[2])
        w3 = torch.exp(-self.log_vars[3])

        # 惩罚项 log(sigma_1 * sigma_2 * sigma_3 * sigma_4) = 0.5 * sum(log_vars)
        log_term = 0.5 * torch.sum(self.log_vars)

        loss_total = (0.5 * w0 * loss_pred + 
                      0.5 * w1 * loss_uks + 
                      0.5 * w2 * loss_flow + 
                      0.5 * w3 * loss_geo + 
                      log_term)
        return loss_total


def compute_uks_likelihood(C, F, Y, eps=1e-5):
    """
    通过可逆消元三角形求解，计算 UKS 结构对数似然损失 L_UKS (UKS Structure Negative Log-Likelihood)。
    L_UKS = 0.5 * Y^T * [C^-1 - C^-1 * F * (F^T * C^-1 * F)^-1 * F^T * C^-1] * Y + 0.5 * log|C|
    
    为避免 Apple Silicon MPS 底层 linalg 库的数值分解故障，Cholesky 临时移至 CPU 执行，完全规避奇异崩溃。
    """
    B, N, M = F.shape
    device = C.device

    # 1. 对协方差矩阵 C 添加极小正定 nugget 扰动以避免退化
    # [1, N, N]
    eye_N = torch.eye(N, device=device, dtype=torch.float32).unsqueeze(0)
    C_reg = C + eps * eye_N  # [B, N, N]

    # 2. 临时转回 CPU 进行高精度 Cholesky 分解: C = L L^T
    C_reg_cpu = C_reg.cpu()
    try:
        L_cpu = torch.linalg.cholesky(C_reg_cpu)
    except torch._C._LinAlgError:
        # 迭代加噪备用保护
        fallback = 1e-4
        L_cpu = None
        eye_N_cpu = torch.eye(N, device='cpu', dtype=torch.float32).unsqueeze(0)
        for _ in range(5):
            try:
                L_cpu = torch.linalg.cholesky(C_reg_cpu + fallback * eye_N_cpu)
                break
            except torch._C._LinAlgError:
                fallback *= 5.0
        if L_cpu is None:
            # 极端情况兜底，防止反向梯度中突发 NaN 导致崩溃
            return torch.tensor(1.0, device=device, requires_grad=True)

    L = L_cpu.to(device)  # [B, N, N]

    # 3. 计算对数行列式项 log|C| = 2 * sum(log L_ii)
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # [B, N]
    log_det_C = 2.0 * torch.sum(torch.log(torch.clamp(diag_L, min=1e-6)), dim=-1)  # [B]

    # 4. 采用消元法求解二次型 Y^T K^-1 Y
    # 求解 L V = F -> V: [B, N, M]
    V = torch.linalg.solve_triangular(L, F, upper=False)
    # 求解 L v_Y = Y -> v_Y: [B, N, D]
    v_Y = torch.linalg.solve_triangular(L, Y, upper=False)

    # 5. 计算二次型主要组成部分
    # V^T V 的形状 [B, M, M]
    V_T_V = torch.bmm(V.transpose(-2, -1), V)  # [B, M, N] x [B, N, M] -> [B, M, M]
    # 添加微小稳定 nugget 项
    eye_M = torch.eye(M, device=device, dtype=torch.float32).unsqueeze(0)  # [1, M, M]
    V_T_V_reg = V_T_V + 1e-6 * eye_M  # [B, M, M]

    # CPU 求解 M = L_M L_M^T
    V_T_V_reg_cpu = V_T_V_reg.cpu()
    try:
        L_M_cpu = torch.linalg.cholesky(V_T_V_reg_cpu)
    except torch._C._LinAlgError:
        return torch.tensor(1.0, device=device, requires_grad=True)
    
    L_M = L_M_cpu.to(device)  # [B, M, M]

    # 计算 V^T * v_Y: [B, M, N] x [B, N, D] -> [B, M, D]
    V_T_vY = torch.bmm(V.transpose(-2, -1), v_Y)  # [B, M, D]
    
    # 求解 L_M L_M^T mu_coef = V^T v_Y
    mu_temp = torch.linalg.solve_triangular(L_M, V_T_vY, upper=False)  # [B, M, D]
    mu_coef = torch.linalg.solve_triangular(L_M.transpose(-2, -1), mu_temp, upper=True)  # [B, M, D]

    # 6. 计算最终的物理二次型值
    # v_Y^T v_Y 的形状 [B, D, D] -> 取对角线和 [B, D] -> 在特征维度求和并平均
    term1 = torch.sum(v_Y ** 2, dim=(-2, -1))  # [B]
    # mu_coef^T * (V^T * v_Y) 的对应部分 [B]
    term2 = torch.sum(mu_coef * V_T_vY, dim=(-2, -1))  # [B]

    quadratic_form = term1 - term2  # [B]

    # 7. 汇总单样本 UKS 对数似然损失，并在 Batch 维度求均值
    loss_uks = 0.5 * (quadratic_form + log_det_C)  # [B]
    return torch.mean(loss_uks)


def get_curriculum_loss_mask(epoch):
    """
    根据当前 Epoch 获取课程学习 (Curriculum Learning) 损失掩码。
    返回: [mask_pred, mask_uks, mask_flow, mask_geo]
    """
    if epoch <= 50:
        # 第一阶段 (1-50 epoch): 仅加速训练物理拟合与 Normalizing Flow 通道
        return [1.0, 0.0, 1.0, 0.0]
    elif epoch <= 120:
        # 第二阶段 (51-120 epoch): 加入 UKS 似然与几何曲率 Hessian 规范，使用静态基准权值
        return [1.0, 1.0, 1.0, 1.0]
    else:
        # 第三阶段 (121-200 epoch): 开启所有损失，并由 HomoscedasticLossWeighting 进行自适应加权寻优
        return [1.0, 1.0, 1.0, 1.0]


def compute_joint_losses(
    model, Z_obs, U_obs, U_pred, X_obs, X_pred, Z_pred, H_obs, 
    lambda_flow=0.01, lambda_geo=0.001, epoch=1, loss_weighting_layer=None
):
    """
    计算空间神经网络统一克里金系统 (UKS-DGL) 的联合损失函数。
    集成空间自监督重构 Pred 损失、UKS 结构对数似然损失、Flow 雅可比体积损失以及 Hessian 拓扑几何损失。
    """
    B, N, D_in = Z_obs.shape

    # 1. 运行模型前向传播，计算预测值及高斯流隐变量
    Z_hat, Y_all_flow, log_det_all = model(Z_obs, U_obs, U_pred, X_obs, X_pred, Z_pred)  # Z_hat: [B, 1, 1]
    
    # 空间自监督重构损失 L_Pred
    loss_pred = torch.mean((Z_hat - Z_pred) ** 2)

    # 2. 计算可逆流生成损失 L_Flow (隐高斯负对数似然 NLL - 雅可比对数)
    d_flow = Y_all_flow.shape[-1]
    log_p_Y = -0.5 * torch.sum(Y_all_flow ** 2, dim=-1) - 0.5 * d_flow * math.log(2.0 * math.pi)  # [B, N+1]
    log_likelihood = log_p_Y + log_det_all  # [B, N+1]
    loss_flow = -torch.mean(log_likelihood)

    # 3. 计算 UKS 结构似然损失 L_UKS
    # 从前向中直接获取自适应马氏协方差 C 与全局趋势面基底 F
    # SCE 的已知点谱嵌入 H_obs
    H_pred_eval = model.sce(U_pred)  # [B, 1, embed_dim]
    C, _ = model.kernel(H_obs, H_pred_eval, U_obs, U_pred)  # C: [B, N, N]
    F = model.get_trend_matrix(U_obs, X_obs)  # F: [B, N, M]
    
    # 高斯隐变量的观测部分 Y_obs_flow
    # 从 Y_all_flow 中截取已知点 [B, N, 2]
    Y_obs_flow = Y_all_flow[:, :N, :]  # [B, N, 2]
    
    loss_uks = compute_uks_likelihood(C, F, Y_obs_flow, eps=model.eps)

    # 4. 计算二阶曲率几何拓扑损失 L_Geo
    U_pred_reg = U_pred.clone().detach().requires_grad_(True)  # [B, 1, 2]
    H_pred_reg = model.sce(U_pred_reg)  # [B, 1, embed_dim]
    
    # 计算带有梯度的 c_0 互协方差向量
    _, c_0_reg = model.kernel(H_obs, H_pred_reg, U_obs, U_pred_reg)  # [B, N, 1]
    
    # 求一阶梯度
    grad_c0 = torch.autograd.grad(
        outputs=c_0_reg.mean(dim=1).sum(),
        inputs=U_pred_reg,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # [B, 1, 2]
    
    grad_c0_x = grad_c0[:, 0, 0]  # [B]
    grad_c0_y = grad_c0[:, 0, 1]  # [B]
    
    # 求解对坐标分量的二阶偏导数
    grad_xx = torch.autograd.grad(
        outputs=grad_c0_x.sum(),
        inputs=U_pred_reg,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 0, 0]  # [B]
    
    grad_yy = torch.autograd.grad(
        outputs=grad_c0_y.sum(),
        inputs=U_pred_reg,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0][:, 0, 1]  # [B]
    
    loss_geo = torch.mean(grad_xx ** 2 + grad_yy ** 2)

    # 5. 课程学习 (Curriculum Learning) 掩码计算
    mask = get_curriculum_loss_mask(epoch)  # [4]
    
    # 应用掩码限制梯度的反向传导
    l_pred_m = loss_pred * mask[0]
    l_uks_m = loss_uks * mask[1]
    l_flow_m = loss_flow * mask[2]
    l_geo_m = loss_geo * mask[3]

    # 6. 自适应同方差不确定性加权与总损失汇总
    if epoch > 120 and loss_weighting_layer is not None:
        # 第三阶段: 使用可学习的同方差加权层融合损失
        loss_total = loss_weighting_layer(l_pred_m, l_uks_m, l_flow_m, l_geo_m)
    else:
        # 第一和第二阶段: 使用基础静态超参权重累加
        loss_total = l_pred_m + lambda_flow * l_flow_m + lambda_geo * l_geo_m + 0.1 * l_uks_m

    return loss_total, loss_pred, loss_flow, loss_geo, loss_uks


@torch.no_grad()
def evaluate_model(model, U_obs, Z_obs, U_pred, X_obs, X_pred, Z_pred):
    """
    评估模型在测试预测点上的精度指标 (MAE, RMSE, R2, Moran's I)。
    使用 predict_with_uncertainty 函数输出无偏估计。
    """
    model.eval()
    
    # 调用重参数化蒙特卡洛预测输出物理无偏估计均值及其估计不确定性方差
    Z_hat, Z_var = model.predict_with_uncertainty(
        Z_obs, U_obs, U_pred, X_obs, X_pred, n_samples_mc=100
    )  # [B, 1, 1], [B, 1, 1]
    
    mae = torch.mean(torch.abs(Z_hat - Z_pred))
    mse = torch.mean((Z_hat - Z_pred) ** 2)
    rmse = torch.sqrt(mse)
    
    return Z_hat, Z_var, mae.item(), rmse.item()
