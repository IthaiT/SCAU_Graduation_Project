# V5-beta: LSTM-mTrans-MLP 模型架构设计

> 严格按照 Kabir, Bhadra, Ridoy & Milanova, "LSTM–Transformer-Based Robust Hybrid Deep Learning
> Model for Financial Time Series Forecasting" (Sci 2025, 7, 7) 实现。

## 1. 系统架构总览

本项目实现论文提出的 **LSTM-mTrans-MLP** 三段式混合架构，并以纯 LSTM 和纯 Transformer 作为对比 Baseline。

```
输入 (B, 60, 1)              ← 仅收盘价, 60 日窗口
    │
    ▼
┌───────────────────────┐
│  Block 1: LSTM ×2     │    LSTM(60) → LSTM(60) → Reshape → Dropout(0.1)
│  输出 (B, 60, 1)      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Block 2: mTransformer │    Pre-Norm → MHA(heads=5, head_dim=120) → Residual
│  (单层 encoder)       │    Pre-Norm → FFN(1→5→1) → Residual
│  输出 (B, 60, 1)      │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  Block 3: MLP         │    GAP → Dropout(0.1) → Dense(30, ReLU) → Dense(1)
│  输出 (B, 1)          │
└───────────────────────┘
```

### 1.1 论文核心改进 (Modified Transformer)

| 原始 Transformer    | mTransformer (本文)                         |
| ------------------- | ------------------------------------------- |
| Post-normalization  | **Pre-normalization** (LayerNorm 在 MHA 前) |
| Positional Encoding | **无位置编码**                              |
| Input embedding     | **无 input embedding**                      |
| Encoder + Decoder   | **仅 Encoder**, Decoder 被 MLP 替代         |
| 多层 Encoder        | **单层 Encoder**                            |

---

## 2. 逐层架构详解

### 2.1 Block 1: LSTM — 局部时序特征提取

双层 LSTM 网络对收盘价序列进行时序建模。

**LSTM 单元计算:**

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \qquad \text{(遗忘门)}
$$
$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \qquad \text{(输入门)}
$$
$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$
$$
h_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \odot \tanh(C_t) \qquad \text{(输出门)}
$$

**配置:** `input_dim=1, hidden_dim=60, num_layers=2, dropout=0.1`

- 第一层 LSTM: `return_sequences=True` (论文)，输出 (B, 60, 60)
- 第二层 LSTM: `return_sequences=False` (论文)，输出 (B, 60)
- Reshape: (B, 60) → (B, 60, 1)，即 d_model=1
- Dropout(0.1)

### 2.2 Block 2: Modified Transformer (mTrans) — 全局依赖建模

单层 Pre-Norm Encoder，处理 (B, 60, 1) 张量。

#### 2.2.1 自定义多头注意力 (CustomMHA)

由于 d_model=1 无法被 num_heads=5 整除，实现自定义 MHA:
- 将 d_model=1 投射到 total_dim = num_heads × head_dim = 5 × 120 = 600
- 计算缩放点积注意力后投射回 d_model=1

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \quad d_k = 120
$$

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

其中 $W^Q, W^K, W^V \in \mathbb{R}^{1 \times 600}$, $W^O \in \mathbb{R}^{600 \times 1}$。

#### 2.2.2 Pre-Norm Encoder 完整流程

$$
x' = x + \text{Dropout}_{0.15}(\text{MHA}(\text{LayerNorm}(x)))
$$
$$
\text{output} = x' + \text{FFN}(\text{LayerNorm}(x'))
$$

**FFN:** $\text{Linear}(1 \to 5) \to \text{ReLU} \to \text{Dropout}(0.15) \to \text{Linear}(5 \to 1)$

### 2.3 Block 3: MLP — 预测输出

$$
z = \text{GlobalAvgPool1D}(\text{mTrans output}) \quad (B, 60, 1) \to (B, 60)
$$
$$
\hat{y} = W_2 \cdot \text{ReLU}(W_1 \cdot \text{Dropout}_{0.1}(z) + b_1) + b_2
$$

$W_1 \in \mathbb{R}^{60 \times 30}$, $W_2 \in \mathbb{R}^{30 \times 1}$

---

## 3. 张量维度追踪

以 `Batch_Size=1, Seq_Len=60, input_dim=1` 为例:

```
输入层
  Input:                      (1, 60, 1)       — 60天收盘价

Block 1: LSTM
  LSTM Layer 1 (hidden=60):   (1, 60, 60)      — return_sequences=True
  LSTM Layer 2 (hidden=60):   取最后时间步 → (1, 60)  — return_sequences=False
  Reshape:                    (1, 60, 1)       — d_model=1
  Dropout(0.1):               (1, 60, 1)

Block 2: Modified Transformer
  LayerNorm(1):               (1, 60, 1)
  ├─ Q projection (1→600):   (1, 60, 600)     → reshape (1, 5, 60, 120)
  ├─ K projection (1→600):   (1, 60, 600)     → reshape (1, 5, 60, 120)
  ├─ V projection (1→600):   (1, 60, 600)     → reshape (1, 5, 60, 120)
  ├─ Attention weights:      (1, 5, 60, 60)   — 5 头 × 60×60
  ├─ Attention output:        (1, 5, 60, 120)  → concat → (1, 60, 600)
  ├─ Output projection (600→1): (1, 60, 1)
  ├─ Dropout(0.15):           (1, 60, 1)
  └─ Residual + input:        (1, 60, 1)
  LayerNorm(1):               (1, 60, 1)
  ├─ FFN Dense(5):            (1, 60, 5)       — ReLU + Dropout
  ├─ FFN Dense(1):            (1, 60, 1)
  └─ Residual:                (1, 60, 1)

Block 3: MLP
  GAP (channels_first):       (1, 60)          — mean over last dim
  Dropout(0.1):               (1, 60)
  Dense(30, ReLU):            (1, 30)
  Dense(1, linear):           (1, 1)           — 最终预测值
```

---

## 4. 参数量统计

### LSTM-mTrans-MLP (论文主模型)

| 模块         | 论文 (Keras) | PyTorch 实现 | 差异说明                    |
| ------------ | ------------ | ------------ | --------------------------- |
| LSTM Layer 1 | 14,880       | 15,120       | PyTorch 双 bias (+240)      |
| LSTM Layer 2 | 29,040       | 29,280       | PyTorch 双 bias (+240)      |
| mTransformer | 4,221        | 4,221        | 完全一致                    |
| MLP          | 1,861        | 1,861        | 完全一致                    |
| **总计**     | **50,002**   | **50,482**   | 仅 LSTM bias 差异, 功能一致 |

**mTransformer 参数分解:**
- Wq (1→600) + bias: 1,200
- Wk (1→600) + bias: 1,200
- Wv (1→600) + bias: 1,200
- Wo (600→1) + bias:   601
- LayerNorm ×2:           4
- Dense(5) + bias:       10
- Dense(1) + bias:        6
- **合计: 4,221**

---

## 5. 训练配方 (严格匹配论文)

| 超参数          | 值           | 来源             |
| --------------- | ------------ | ---------------- |
| Input           | 仅收盘价     | 论文 Table 2     |
| Sequence Length | 60           | 论文 Table 4     |
| Batch Size      | 1            | 论文 Table 4     |
| Optimizer       | Adam         | 论文 Table 4     |
| Learning Rate   | 0.001        | 论文 Table 4     |
| Loss Function   | MSE          | 论文 Eq. (4)     |
| Epochs          | 30           | 论文 12-30 range |
| Data Split      | 82:18        | 论文 Table 5     |
| Normalization   | MinMaxScaler | 论文 Eq. (3)     |
| Early Stopping  | patience=10  | 工程保障         |

---

## 6. Baseline 模型

### 6.1 LSTM Baseline

纯 LSTM + FC，与论文主模型的 Block 1 共享相同 LSTM 配置:

```
Input (B, 60, 1) → LSTM×2(hidden=60) → 取last step → Dropout(0.1) → FC(60→1)
```

### 6.2 Transformer Baseline

标准 Transformer Encoder (Post-Norm) + 位置编码:

```
Input (B, 60, 1) → Proj(1→60) → PosEnc → Encoder×2(d_model=60, heads=5) → FC(60→1)
```

---

## 7. 评估指标

| 指标 | 公式                                                                                      | 含义                       |
| ---- | ----------------------------------------------------------------------------------------- | -------------------------- |
| MSE  | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$                                                      | 均方误差                   |
| RMSE | $\sqrt{MSE}$                                                                              | 均方根误差，与原始量纲一致 |
| MAE  | $\frac{1}{N}\sum\|y_i - \hat{y}_i\|$                                                      | 平均绝对误差               |
| MAPE | $\frac{100}{N}\sum\left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$                             | 平均绝对百分比误差         |
| R²   | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$                               | 决定系数                   |
| DA   | $\frac{100}{N-1}\sum \mathbb{1}[\text{sign}(\Delta y_i) = \text{sign}(\Delta \hat{y}_i)]$ | 方向准确率                 |
