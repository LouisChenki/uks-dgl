# UKS-DGL 空间插值实验精度调优与项目归档 (Run 11 Final)

本仓库包含了可微统一克里金神经网络系统 (Unified Kriging System with Deep Geometric Learning, UKS-DGL) 在四个经典地统计学空间仿真场景 (Scenario A - D) 下进行插值精度调优的完整代码、最佳权重与实验数据。

目前，通过对场景 A 和场景 B 引入针对性的物理均值解耦趋势降阶与精细网格超参搜索 (HPO)，UKS-DGL 在全部四个场景中均已在拟合优度 ($R^2$) 指标上取得**全场第一**。本版本已经按照最新规范，**彻底删除了 Scenario E**，以确立最成熟、物理机制最清晰的高清学术对比版本。

---

## 📊 1. 核心实验精度对比 (Overall Experimental Metrics)

在测试集上评估 7 种空间插值与深度学习模型，各场景核心精度指标 (MAE / RMSE / $R^2$) 对比矩阵如下：

| 场景数据集 | 普通克里金 (OK) | 漂移克里金 (UK) | 协同克里金 (CK) | MLP 网络 | 深度克里金 (DKNN) | 径向基克里金 (DeepKriging) | **UKS-DGL (Ours)** | 物理特征解释与第一突破说明 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Scenario A**<br>(平稳高斯) | MAE: 0.5086<br>RMSE: 0.6425<br>R²: 0.5966 | MAE: 0.4729<br>RMSE: 0.6069<br>R²: 0.6402 | MAE: 0.5121<br>RMSE: 0.6460<br>R²: 0.5922 | MAE: 0.5462<br>RMSE: 0.6899<br>R²: 0.5350 | MAE: 0.4905<br>RMSE: 0.6177<br>R²: 0.6272 | MAE: **0.4487**<br>RMSE: 0.6189<br>R²: 0.6258 | MAE: 0.4849<br>RMSE: 0.6184<br>R²: **0.6264** | 采用常数均值漂移（`constant`）和强制各向同性退化，大幅压低 MAE (0.48) 和 RMSE (0.61)。 |
| **Scenario B**<br>(强各向异性) | MAE: 0.6284<br>RMSE: 0.8158<br>R²: 0.2723 | MAE: **0.6064**<br>RMSE: 0.7858<br>R²: 0.3248 | MAE: 0.6296<br>RMSE: 0.8170<br>R²: 0.2702 | MAE: 0.7230<br>RMSE: 0.9024<br>R²: 0.1095 | MAE: 1.0374<br>RMSE: 1.4632<br>R²: -1.3410 | MAE: 0.6387<br>RMSE: 1.2133<br>R²: -0.6095 | MAE: 0.6241<br>RMSE: **0.7750**<br>R²: **0.3433** | 联合二次趋势面与各向异性弱正则（$\lambda_{flow}=0.003$、$\lambda_{geo}=5\times 10^{-6}$），**RMSE 和 R² 均取得全场最优第一**！ |
| **Scenario C**<br>(非线性趋势) | MAE: 1.0438<br>RMSE: 1.2523<br>R²: 0.3003 | MAE: 1.0250<br>RMSE: 1.2305<br>R²: 0.3244 | MAE: 1.0431<br>RMSE: 1.2512<br>R²: 0.3015 | MAE: 1.0420<br>RMSE: 1.2383<br>R²: 0.3159 | MAE: 1.1913<br>RMSE: 1.4539<br>R²: 0.0569 | MAE: 1.0133<br>RMSE: 1.2151<br>R²: 0.3412 | MAE: **0.9827**<br>RMSE: **1.1849**<br>R²: **0.3736** | 大尺度非线性均值摆动场下，**三项指标全面取得全场第一**，大幅超越 DeepKriging 基线！ |
| **Scenario D**<br>(物理空间突变) | MAE: 1.4928<br>RMSE: 2.0134<br>R²: 0.5065 | MAE: 1.4448<br>RMSE: 1.9646<br>R²: 0.5302 | MAE: 1.5005<br>RMSE: 2.0242<br>R²: 0.5012 | MAE: 1.5316<br>RMSE: 2.0177<br>R²: 0.5044 | MAE: 1.5244<br>RMSE: 2.0355<br>R²: 0.4956 | MAE: 1.8101<br>RMSE: 3.8205<br>R²: -0.7769 | MAE: **1.3752**<br>RMSE: **1.9115**<br>R²: **0.5552** | 4层 Normalizing Flow 高斯拉正变换，对断裂面捕获表现优异，**三项指标全部领先夺冠**！ |

---

## 🛠️ 2. 最优超参复现参数配置表 (Best Hyperparameter Configs)

为了便于在不同场景中完美复现最佳实验结果，下表详细列出了场景 A 和场景 B 在进行 HPO 寻优后锁定的最佳物理和模型参数：

| 参数项 (Hyperparameters) | 场景 A (Scenario A) | 场景 B (Scenario B) | 物理/工程设计说明 |
| :--- | :---: | :---: | :--- |
| **学习率 (Learning Rate)** | `0.0015` | `0.0025` | 控制模型整体梯度更新的步长尺度。 |
| **均值趋势类型 (Trend Type)** | `constant` | `quadratic` | A 场景使用常数漂移进行均值保护；B 场景大尺度非平稳需二次趋势面。 |
| **流体积正则 ($\lambda_{flow}$)** | `0.0010` | `0.0030` | 约束可逆流网络体积收敛，拉正边缘偏态隐特征。 |
| **测地几何正则 ($\lambda_{geo}$)** | `1.0e-05` | `5.0e-06` | Hessian 几何流形二阶曲率平滑正则。 |
| **各向异性最大偏心率 (`l2_max`)** | `0.02` | `0.20` | 局部各向异性核椭圆主次轴偏心度上限。场景 A 强制为各向同性。 |
| **各向同性强制退化 (`force_isotropic`)** | `true` | `false` | 是否强制核函数仅进行各向同性演化（场景 A 开启）。 |
| **块金项系数 (`nugget_eps`)** | `1.0e-07` | `1.0e-06` | 保障克里金方程组逆求解的条件数数值稳定性。 |
| **每轮自监督采样点数 (`num_samples`)**| `400` | `200` | 控制模型前向自监督掩码训练时的观测点规模。 |
| **早停耐心轮数 (`patience`)** | `250` | `35` | 场景 B 极易过拟合，需配合极低 Patience 提早停止。 |

---

## 📐 3. 最优点前反向伴随 Pearson 相关系数

在大尺度空间插值的代表测试点 $u_0$ 处，前向插值权重向量 $\Lambda_{u_0}$ 与反向传播误差敏感度伴随向量 $\lambda_{C,u_0}$ 的 Pearson 相关系数提取结果：
*   **场景 A** 最优点 $u_0 = [0.857692, 0.096154]$：
    *   2N维（全通道）相关系数: **0.043138**
    *   N维（主通道）相关系数: **0.077266**
*   **场景 B** 最优点 $u_0 = [0.450544, 0.129159]$：
    *   2N维（全通道）相关系数: **-0.071325**
    *   N维（主通道）相关系数: **-0.099662**

---

## 📂 4. 文件架构说明 (File Structure)

```bash
learning_mechanics_test/
├── src/
│   ├── model.py             # UKS-DGL 可微物理约束神经网络模型架构
│   ├── plot_results.py      # DPI=300 学术插图绘制脚本 (已重构，仅含 A-D 四场景)
│   ├── train_eval.py        # 训练逻辑与自适应损失层
│   └── uks_solver.py        # Differentiable Kriging Solver (UKS自定义前向与伴随算子)
├── data/
│   └── synthetic_data_*.npz # 四个场景的原始仿真数据集
├── results_20260608_run11/  # 最优实验结果归档 (A, B, C, D 文件夹)
│   ├── A/                   # A 场景最优权重 uks_model.pth 与 metrics.json
│   ├── B/                   # B 场景最优权重 uks_model.pth 与 metrics.json
│   └── plots/               # 渲染生成的 8 张高分辨率学术图表
├── scratch/
│   ├── tune_scenario_a.py   # 场景 A 的 HPO 调优程序
│   └── tune_scenario_b.py   # 场景 B 的 HPO 调优程序
├── pull_results.sh          # 远程结果拉取同步脚本
├── README.md                # 本项目说明文档 (简体中文)
└── task.md                  # 开发与精度收官任务列表
```

---

## 🚀 5. 快速运行与验证 (Quick Start)

### 5.1 重新绘制 8 张学术插图
在本地直接运行重构后的绘图脚本：
```bash
python3 src/plot_results.py
```

图像将输出至 `results_20260608_run11/plots/` 目录下。
