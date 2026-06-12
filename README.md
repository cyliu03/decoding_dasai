# 触觉 fMRI 脑解码系统

基于 LSS (Least-Squares-Separate) GLM 和 MVPA (多体素模式分析) 的触觉 fMRI 脑解码系统，能够从大脑活动模式中识别 13 类触觉刺激的几何形状类别。

## 项目结构

```
sygdasai/
├── code/
│   ├── run_lss_glm.py           # LSS GLM 提取 trial-level beta 图像
│   ├── run_mvpa_rsa.py          # MVPA/RSA 全脑/ROI 分析与 Top-k 评估
│   ├── run_subject_tests.py     # 单被试和跨被试测试（使用真实 LSS beta 数据）
│   └── server.py                # 网页演示后端（HTTP API + 静态文件服务）
├── data/
│   └── lss/                     # LSS beta 数据目录
│       ├── sub-001/func/        # 被试 001 的 beta NIfTI 和事件文件
│       ├── sub-002/func/        # 被试 002
│       └── sub-003/func/        # 被试 003
├── demo_web_app/                # 前端演示页面
│   └── public/
│       ├── assets/stimuli/      # 刺激图形 SVG
│       ├── data/
│       │   └── model_profile.json  # 模型配置与指标数据
│       ├── app.js               # 前端逻辑
│       ├── index.html           # 主页面
│       └── styles.css           # 样式
├── models/                      # 训练好的模型
│   ├── mvpa_optimized.joblib    # CalibratedClassifierCV(LinearSVC) 模型
│   ├── mvpa_optimized.json      # 模型元信息
│   └── test_set.json            # 演示用测试集
├── results/                     # 测试结果
│   ├── accuracy_comparison.png  # 准确率对比图
│   └── test_results.json        # 测试结果数据
├── requirements.txt             # Python 依赖
└── README.md
```

## 环境要求

- Python >= 3.10
- 操作系统：Windows / Linux / macOS

## 安装步骤

```bash
cd sygdasai
pip install -r requirements.txt
```

依赖包：
- numpy >= 1.24
- pandas >= 2.0
- scipy >= 1.10
- nibabel >= 5.0
- scikit-learn >= 1.3
- joblib >= 1.3
- matplotlib >= 3.7

## 使用方法

### 1. 生成 LSS beta 图像

如果 `data/lss/` 中没有 beta NIfTI 文件（只有 `_events.tsv`），需要先运行：

```bash
python code/run_lss_glm.py
```

### 2. 训练与测试

运行单被试（Leave-One-Run-Out）和跨被试（Leave-One-Subject-Out）测试：

```bash
python code/run_subject_tests.py
```

可选参数：
- `--data-dir`：LSS beta 数据目录（默认 `data/lss`）
- `--output`：结果输出目录（默认 `results`）
- `--model-dir`：模型保存目录（默认 `models`）
- `--feature-k`：SelectKBest 保留的特征数（默认 5000）
- `--max-voxels`：按方差保留的最大体素数（默认 30000）
- `--svm-c`：LinearSVC 的 C 参数（默认 1.0）

该脚本会：
- 对每个被试做 Leave-One-Run-Out 交叉验证
- 做 Leave-One-Subject-Out 跨被试测试
- 训练最终模型并保存到 `models/`
- 保存测试集到 `models/test_set.json`
- 生成准确率对比图到 `results/accuracy_comparison.png`
- 保存结果到 `results/test_results.json`

### 3. MVPA/RSA 详细分析

```bash
python code/run_mvpa_rsa.py
```

运行全脑和 ROI 的 MVPA 与 RSA 分析，输出 Top-k 排序结果。

### 4. 网页演示

```bash
python code/server.py
```

打开浏览器访问 http://127.0.0.1:8765

可选参数：端口号（默认 8765）

```bash
python code/server.py 9000
```

## 实验结果

### 13 类触觉刺激分类

| 指标 | 机会水平 | 说明 |
|------|---------|------|
| Top-1 | 7.7% | 预测排名第一的正确率 |
| Top-3 | 23.1% | 预测排名前三包含正确类别的比率 |
| Top-5 | 38.5% | 预测排名前五包含正确类别的比率 |

### 最佳模型结果（参考 model_profile.json）

| 被试 | Top-1 | Top-3 | Top-5 |
|------|-------|-------|-------|
| sub-001 | 15.0% | 43.8% | 43.8% |
| sub-002 | 10.0% | 39.4% | 47.5% |
| sub-003 | 11.9% | 47.5% | 48.8% |

模型：CalibratedClassifierCV(LinearSVC)，使用 LSS trial-level beta 作为输入特征。

## 数据说明

- **被试数**：3 名
- **每个被试 run 数**：4 个
- **每个 run 的 trial 数**：约 40 个
- **总 trial 数**：约 480 个
- **分类类别**：13 个字母类别（E, I, J, L, M, N, O, P, R, S, T, V, X）
- **刺激材料**：触觉几何形状物体，按首字母分为 13 类
- **数据格式**：BIDS 衍生格式，包含 LSS beta NIfTI 图像和事件 TSV 文件

### 数据文件命名

- Beta 图像：`sub-XXX_task-tactile_run-XX_space-T1w_desc-lssTrialBetas_beta.nii.gz`
- 事件文件：`sub-XXX_task-tactile_run-XX_space-T1w_desc-lssTrialBetas_events.tsv`
- 设计摘要：`sub-XXX_task-tactile_run-XX_space-T1w_desc-lssDesignSummary.json`
