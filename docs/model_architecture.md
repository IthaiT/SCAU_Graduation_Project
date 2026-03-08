# Hybrid LSTM-Transformer 模型架构设计

## 1. 系统架构总览

本文提出的混合 LSTM-Transformer 模型采用**串行级联架构**，核心思想是利用 LSTM 提取输入序列的**局部时序特征**，再通过 Transformer Encoder 捕获序列中的**全局依赖关系**，最终通过全连接层输出预测值。

```
Input Tensor ──→ LSTM Layer ──→ Positional Encoding ──→ Transformer Encoder (×2) ──→ FC Layer ──→ Output
 (B, S, F)      (B, S, H)        (B, S, H)             (B, S, H)                   (B, 1)
```

### 1.1 设计动机

| 组件                   | 捕获的特征类型                 | 局限性                         |
| ---------------------- | ------------------------------ | ------------------------------ |
| LSTM                   | 局部时序依赖（短期趋势、动量） | 长距离依赖捕获能力有限         |
| Transformer            | 全局位置间依赖（跨时段关联）   | 缺乏归纳偏置，对序列顺序不敏感 |
| **LSTM + Transformer** | **局部 + 全局特征的互补融合**  | —                              |

LSTM 为 Transformer 提供了经过时序建模的高质量隐状态表示，使 Transformer 的自注意力机制能在更丰富的特征空间上学习全局依赖。

---

## 2. 逐层架构详解

### 2.1 LSTM 层 — 局部时序特征提取

双层 LSTM 网络对原始量价特征进行时序建模，逐步从低阶特征提取高阶时序表示。

**数学表达**（单层 LSTM 单元）:

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$
$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$
$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$
$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$
$$
h_t = o_t \odot \tanh(C_t)
$$

其中 $f_t, i_t, o_t$ 分别为遗忘门、输入门、输出门，$C_t$ 为细胞状态，$h_t$ 为隐状态输出。

**配置**: `input_dim=F, hidden_dim=64, num_layers=2, dropout=0.2`

### 2.2 正弦位置编码 — 注入序列位置信息

由于 Transformer 本身不具备对输入顺序的感知能力，需要显式注入位置信息:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

位置编码与 LSTM 输出逐元素相加后，经 Dropout ($p=0.2$) 正则化。

### 2.3 Transformer Encoder（×2 层）— 全局依赖建模

每层 Transformer Encoder 由**多头自注意力 (MHA)** 和**前馈网络 (FFN)** 组成，均配有残差连接与层归一化。

**多头自注意力机制:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

其中 $Q = K = V = X$（自注意力），$h=4$ 个注意力头，$d_k = d_{model}/h = 16$。

**前馈网络 (FFN):**

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

FFN 维度为 256，经 Dropout ($p=0.2$) 正则化。

**每层完整计算流程（Pre-LN 变体）:**

$$
x' = \text{LayerNorm}(x + \text{Dropout}(\text{MHA}(x, x, x)))
$$
$$
\text{output} = \text{LayerNorm}(x' + \text{Dropout}(\text{FFN}(x')))
$$

堆叠 **2 层** 相同结构的 Encoder，最后一层的注意力权重矩阵用于可视化分析模型的可解释性。

### 2.4 全连接输出层

取 Transformer 输出序列中**最后一个时间步**的隐状态，通过线性映射得到预测值:

$$
\hat{y} = W_{fc} \cdot h_S + b_{fc}
$$

---

## 3. 张量维度追踪 (Tensor Shape Tracking)

以 `Batch_Size=32, Seq_Len=30, Features=F` 为例，追踪数据在网络中的完整流转:

```
输入层
  Input:                    (32, 30, F)      — 30天窗口 × F个量价/技术指标特征

LSTM 层 (2层, hidden_dim=64, dropout=0.2)
  LSTM output:              (32, 30, 64)     — 每个时间步产生64维隐状态
  LSTM hidden state:        (2, 32, 64)      — 2层各保留最终隐状态 (丢弃)

位置编码 (d_model=64)
  + Positional Encoding:    (32, 30, 64)     — 逐元素加法，维度不变
  Dropout(0.2):             (32, 30, 64)

Transformer Encoder Layer 1 (num_heads=4, ffn_dim=256)
  ├─ Q, K, V:              (32, 30, 64)     — 自注意力: Q=K=V=input
  ├─ MHA output:           (32, 30, 64)     — 4头注意力, 每头 d_k=16
  ├─ Attention weights:    (32, 30, 30)     — softmax(QK^T/√16)
  ├─ Add & LayerNorm:      (32, 30, 64)     — 残差 + 归一化
  ├─ FFN:                  (32, 30, 256)    — 线性升维
  ├─ FFN → ReLU → Drop:   (32, 30, 64)     — 线性降维
  └─ Add & LayerNorm:      (32, 30, 64)     — 残差 + 归一化

Transformer Encoder Layer 2 (结构同 Layer 1)
  └─ output:               (32, 30, 64)     — 全局依赖编码后的序列表示
     attention weights:    (32, 30, 30)     — 最后一层权重用于可视化

取最后时间步
  Select [:, -1, :]:        (32, 64)         — 仅保留 t=30 的隐状态

全连接输出层
  Linear(64, 1):            (32, 1)          — 预测下一交易日收盘价
```

---

## 4. 超参数总结

| 超参数             | 值                | 说明                                        |
| ------------------ | ----------------- | ------------------------------------------- |
| Batch Size         | 32                | 小批量梯度下降                              |
| Sequence Length    | 30                | 输入窗口 = 30 个交易日                      |
| Prediction Length  | 1                 | 单步预测                                    |
| LSTM Hidden Dim    | 64                | LSTM 隐状态维度                             |
| LSTM Layers        | 2                 | 堆叠两层 LSTM                               |
| Transformer Layers | 2                 | 堆叠两层 Encoder                            |
| Attention Heads    | 4                 | 多头注意力头数                              |
| FFN Dimension      | 256               | 前馈网络中间维度                            |
| Dropout            | 0.2               | 全局 Dropout 率                             |
| Optimizer          | Adam              | 自适应学习率优化器                          |
| Learning Rate      | 0.001             | 初始学习率                                  |
| LR Scheduler       | ReduceLROnPlateau | 验证损失停滞时衰减 (factor=0.5, patience=5) |
| Loss Function      | MSE               | 均方误差                                    |
| Epochs             | 50                | 最大训练轮数                                |
| Early Stopping     | patience=10       | 连续10轮无改善则停止                        |

## 5. 评估指标

| 指标 | 公式                                                                                      | 含义                       |
| ---- | ----------------------------------------------------------------------------------------- | -------------------------- |
| MSE  | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$                                                      | 均方误差                   |
| RMSE | $\sqrt{MSE}$                                                                              | 均方根误差，与原始量纲一致 |
| MAPE | $\frac{100}{N}\sum\left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$                             | 平均绝对百分比误差         |
| DA   | $\frac{100}{N-1}\sum \mathbb{1}[\text{sign}(\Delta y_i) = \text{sign}(\Delta \hat{y}_i)]$ | 方向准确率                 |
