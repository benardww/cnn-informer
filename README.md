# CNN-Informer：基于脑电信号的自动癫痫检测

CNN-Informer 是一个针对 CHB-MIT 头皮脑电数据集的端到端癫痫发作自动检测系统，融合了一维卷积神经网络（CNN）的局部特征提取能力与 Informer 编码器的长程依赖建模能力。

---

## 目录

- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [快速开始](#快速开始)
- [运行参数](#运行参数)
- [输出结果](#输出结果)
- [方法细节](#方法细节)

---

## 模型架构

```
原始 EEG [18ch × N样本]
        │
        ▼  Db4 五层 DWT（保留 0.5–29 Hz）
        │
        ▼  滑动窗口分割（4秒，发作段50%重叠）
        │
[B, 18, 1024]
        │
        ▼  CNN 模块（三层一维卷积）
[B, 63, 64]
        │
        ▼  Informer 编码器（3层 ProbSparse 注意力 + 2层蒸馏）
           63 → 32 → 16
[B, 16, 64]
        │
        ▼  全连接分类头（1024 → 512 → 128 → 2）
[B, 2] logits
        │
        ▼  后处理（MAF平滑 → 阈值 → Collar事件匹配）
        │
  ┌─────┴─────┐
  ▼           ▼
表格1        表格2
分段指标     事件指标
```

### CNN 模块维度

| 层 | 操作 | 输出形状 |
|:---:|---|:---:|
| 1 | Conv1d(18,18,k=4,s=2) → BN → ELU → MaxPool1d(2,2) | [B, 18, 255] |
| 2 | Conv1d(18,18,k=4,s=2) → BN → ELU → MaxPool1d(2,2) | [B, 18, 63] |
| 3 | Conv1d(18,64,k=3,s=1,p=1) → BN → ELU → Dropout | [B, 64, 63] |
| T | 转置 | [B, 63, 64] |

### Informer 编码器

- **ProbSparse 注意力**：采样因子 c=3，仅对 top-u 查询（u = c·⌈ln L⌉）计算全注意力，复杂度 O(L log L)
- **蒸馏层**：Conv1d → ELU → MaxPool1d(k=3,s=2,p=1)，序列长度 63→32→16
- **d\_model**=64，**n\_heads**=8，**d\_ff**=256，3 层编码器

---

## 项目结构

```
cnn-informer/
├── config.py                     # 全局超参数
├── requirements.txt
├── main.py                       # CLI 入口
├── data/
│   ├── summary_parser.py         # 解析 chb##-summary.txt
│   ├── edf_loader.py             # MNE EDF 读取 + 18 通道选择
│   └── dataset.py                # CHBMITPatientDataset（在线 DWT，不写磁盘）
├── preprocessing/
│   └── dwt_filter.py             # Db4 五层 DWT 带通重建
├── models/
│   ├── cnn_module.py
│   ├── prob_attention.py         # ProbSparse 多头自注意力
│   ├── informer_encoder.py       # 编码层 + 蒸馏层
│   └── cnn_informer.py           # 完整模型
├── training/
│   ├── loss.py                   # 逆频率加权交叉熵
│   └── trainer.py                # 按患者训练 + 早停
├── postprocessing/
│   └── postprocess.py            # MAF / 阈值 / Collar 匹配
└── evaluation/
    └── metrics.py                # 分段 & 事件指标 + DataFrame 输出
```

---

## 环境配置

**Python ≥ 3.10**

```bash
pip install -r requirements.txt
```

主要依赖：

| 包 | 版本要求 | 用途 |
|---|---|---|
| torch | ≥ 2.1.0 | 模型训练推理 |
| mne | ≥ 1.6.0 | EDF 文件读取 |
| PyWavelets | ≥ 1.5.0 | DWT 滤波 |
| numpy | ≥ 1.26.0 | 数值计算 |
| pandas | ≥ 2.1.0 | 表格输出 |
| tqdm | ≥ 4.66.0 | 训练进度条 |

---

## 数据集准备

使用 [CHB-MIT 头皮脑电数据集](https://physionet.org/content/chbmit/1.0.0/)（PhysioNet）。

将数据集放置于以下路径（与 `config.py` 中的 `DATA_ROOT` 一致）：

```
autodl-tmp/
└── chb-mit-scalp-eeg-database-1.0.0/
    ├── chb01/
    │   ├── chb01-summary.txt
    │   ├── chb01_01.edf
    │   ├── chb01_02.edf
    │   └── ...
    ├── chb02/
    └── ...
```

> **注意**：chb06 和 chb16 已从评估中排除，共使用 22 名患者。

如需修改数据集路径，编辑 [config.py](config.py) 中的 `DATA_ROOT`：

```python
DATA_ROOT = "autodl-tmp/chb-mit-scalp-eeg-database-1.0.0"
```

---

## 快速开始

### 运行所有患者

```bash
python main.py
```

### 仅测试指定患者

```bash
python main.py --patients chb01 chb02 chb03
```

### 快速验证（减少轮次）

```bash
python main.py --patients chb01 --epochs 5
```

程序将为每个患者依次完成训练和评估，最终打印两张汇总表格。

---

## 运行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_root` | `autodl-tmp/chb-mit-scalp-eeg-database-1.0.0` | 数据集根路径 |
| `--patients` | 全部 22 名 | 指定患者列表 |
| `--epochs` | 50 | 最大训练轮次 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 1e-4 | Adam 学习率 |
| `--maf_window` | 10 | 移动平均窗口大小（单位：窗口数） |
| `--threshold` | 0.5 | 二值化阈值 |
| `--collar_k` | 5.0 | Collar 容忍秒数 |

---

## 输出结果

程序结束后输出两张评估表格：

### 表格 1 — 分段级别指标

| Patient | Sensitivity | Specificity | Accuracy |
|---|---|---|---|
| chb01 | 0.xxxx | 0.xxxx | 0.xxxx |
| ... | ... | ... | ... |
| **Mean** | **0.xxxx** | **0.xxxx** | **0.xxxx** |

- **Sensitivity**（灵敏度）= TP / (TP + FN)
- **Specificity**（特异度）= TN / (TN + FP)
- **Accuracy**（准确率）= (TP + TN) / 总样本数

### 表格 2 — 事件级别指标

| Patient | Sensitivity | FDR/hr | Latency(s) |
|---|---|---|---|
| chb01 | 0.xxxx | 0.xxxx | xx.xx |
| ... | ... | ... | ... |
| **Mean** | **0.xxxx** | **0.xxxx** | **xx.xx** |

- **Sensitivity**（事件灵敏度）= 检测到的发作事件数 / 真实发作事件总数
- **FDR/hr**（每小时误检率）= 误报事件数 / 总记录小时数
- **Latency**（检测延迟，秒）= 检测事件起始时间 − 真实发作起始时间的均值

---

## 方法细节

### 数据预处理

1. **通道选择**：从每个 EDF 文件中提取 18 个标准双极导联（FP1-F7、F7-T7、T7-P7、P7-O1、FP1-F3、F3-C3、C3-P3、P3-O1、FP2-F4、F4-C4、C4-P4、P4-O2、FP2-F8、F8-T8、T8-P8、P8-O2、FZ-CZ、CZ-PZ），通道名自动归一化（去除 `"EEG "` 前缀及 `"-REF"`、`"-LE"` 后缀）。

2. **DWT 滤波**：对每个 4 秒窗口逐通道进行 Daubechies-4 五层离散小波变换，保留 cA5（0–4 Hz）、cD5（4–8 Hz）、cD4（8–16 Hz）、cD3（16–32 Hz），置零 cD2 和 cD1，重建得到覆盖 0.5–29 Hz 的带通信号，滤除眼动伪影与工频噪声。

3. **滑动窗口**：窗口长度 4 秒（1024 采样点），发作段采用 50% 重叠（步长 512），非发作段无重叠（步长 1024）。**预处理结果不保存到磁盘**，DWT 在 DataLoader 的 `__getitem__` 中即时计算。

### 训练策略

- **时序切分**：按 summary 文件中 EDF 文件的时间顺序，前 80% 用于训练，后 20% 用于验证，防止数据泄漏。
- **类别不平衡**：使用逆频率加权交叉熵损失，同时在 DataLoader 中使用 `WeightedRandomSampler` 对发作样本进行过采样。
- **优化器**：Adam（lr=1e-4），梯度裁剪（max_norm=1.0），`ReduceLROnPlateau` 调度（factor=0.5，patience=5）。
- **早停**：验证集损失连续 10 轮不降则停止训练，保存最优模型权重。

### 后处理

1. **移动平均滤波（MAF）**：对发作类概率序列进行因果移动平均（窗口 10 个，即 40 秒），平滑短暂误报。
2. **阈值判定**：平滑概率 ≥ 0.5 标记为发作。
3. **Collar 技术**：事件级评估时，检测事件起始时间落在真实发作区间 [onset − K, offset + K]（K=5s）内视为正确检测，抑制孤立误报并补偿检测延迟。
