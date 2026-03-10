# 三代融合模型架构对比：Serial → Parallel → LSTM-mTrans-MLP

> **背景**：本项目在沪深300指数预测任务上，先后探索了三种 LSTM-Transformer 融合架构。  
> 本文档详细对比三者在**网络结构、数据流、超参数、训练策略、实验结果**上的差异。

---

## 0. 模型一览

| 维度     | ① Serial LSTM-Transformer (V1-V4)                       | ② Parallel LSTM-Transformer (V5)            | ③ LSTM-mTrans-MLP (V5-beta, 论文复现)         |
| -------- | ------------------------------------------------------- | ------------------------------------------- | --------------------------------------------- |
| 来源     | 自主设计，经 4 轮迭代优化                               | 自主设计，V4 基础上改并行                   | 复现 Kabir et al. (Sci 2025, 7, 7)            |
| Git 标签 | `737a875` (V4)                                          | `f9ba0fe` (并行)                            | `a1b7781` (LSTM-mTrans-MLP)                   |
| 总参数量 | **162,945**                                             | **174,529**                                 | **50,482**                                    |
| 核心思路 | LSTM 提取局部特征 → 送入标准 Transformer → 特征拼接融合 | 双塔并行处理原始输入 → Sigmoid 动态门控融合 | 论文三段式：LSTM → Modified Transformer → MLP |

---

## 1. 架构对比

### 1.1 ① Serial LSTM-Transformer（V4 最终版）

```
输入 (B, 30, 17)                  ← 17 维技术指标特征
    │
    ▼
┌────────────────────────┐
│  LSTM (2层, hidden=64)  │
│  dropout=0.2            │
└──────────┬─────────────┘
           │ (B, 30, 64)
           ▼
┌────────────────────────┐
│  LayerNorm(64)          │    ← V2 新增：消除方差偏移
└──────────┬─────────────┘
           │ (B, 30, 64)
           │
     ┌─────┴──────┐
     │             │
     ▼             ▼
 取 last step   Positional Encoding
     │             │ (B, 30, 64)
     │             ▼
     │    ┌────────────────────┐
     │    │ Transformer Encoder │
     │    │ × 2 层 (Post-Norm)  │
     │    │ MHA: 4 heads        │
     │    │ FFN: 64→256→64      │
     │    └────────┬───────────┘
     │             │ (B, 30, 64)
     │             ▼
     │         取 last step
     │             │ (B, 64)
     ▼             ▼
┌────────────────────────────┐
│  Concat (dim=-1)            │    ← V3 新增：跳跃连接
│  lstm_last ⊕ trans_last    │
└──────────┬─────────────────┘
           │ (B, 128)
           ▼
┌────────────────────────────┐
│  Gated Fusion Bottleneck    │    ← V4 升级
│  Linear(128→64) → ReLU     │
│  → Dropout(0.2)            │
│  → Linear(64→1)            │
└──────────┬─────────────────┘
           │ (B, 1)
           ▼
        输出预测值
```

**关键特点**：
- LSTM 输出**全部时间步**送入 Transformer（串行流水线）
- Transformer 处理的是 LSTM 特征，而非原始输入
- V3 的跳跃连接保留 LSTM 短期特征，防止 Transformer 过度平滑
- V4 的瓶颈网络实现对双分支特征的非线性过滤

### 1.2 ② Parallel LSTM-Transformer（V5）

```
                   输入 (B, 30, 17)              ← 17 维技术指标特征
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────────┐     ┌───────────────────────────┐
│  ★ 左塔: LSTM ★      │     │  ★ 右塔: Transformer ★     │
│  LSTM(17→64, 2层)    │     │  Linear(17→64) 投影        │
│  dropout=0.2         │     │  Positional Encoding(64)   │
│  取 last step        │     │  TransformerEncoder × 2层   │
│  LayerNorm(64)       │     │  (4头 MHA + FFN(256))      │
│                      │     │  LayerNorm(64)             │
│  → feat_lstm (B,64)  │     │  Global Average Pooling    │
└──────────┬───────────┘     │  → feat_trans (B,64)       │
           │                 └────────────┬──────────────┘
           │                              │
           ▼                              ▼
┌───────────────────────────────────────────────────────┐
│  Concat → (B, 128)                                     │
│                                                        │
│  ★ Dynamic Gating ★                                    │
│  gate = Sigmoid(Linear(128→128))                       │
│  gate_lstm, gate_trans = gate.chunk(2)                 │
│  fused = gate_lstm ⊙ feat_lstm + gate_trans ⊙ feat_trans │
│  → (B, 64)                                             │
├────────────────────────────────────────────────────────┤
│  Predictor: Linear(64→32) → ReLU → Dropout → Linear(32→1) │
└──────────────────────┬─────────────────────────────────┘
                       │ (B, 1)
                       ▼
                    输出预测值
```

**关键特点**：
- LSTM 和 Transformer **各自独立处理原始输入**（并行双塔）
- Transformer 右塔使用 GAP（全局平均池化）而非 last step
- Sigmoid 门控：逐维度决定信任 LSTM 还是 Transformer（不是 Softmax 非此即彼）
- 门控权重可暴露用于白盒可解释性分析

### 1.3 ③ LSTM-mTrans-MLP（V5-beta，论文复现）

```
输入 (B, 60, 1)                   ← 仅收盘价, 单特征
    │
    ▼
┌─────────────────────────────┐
│  Block 1: LSTM               │
│  LSTM(1→60, 2层, drop=0.1)  │
│  取 last step → (B, 60)      │
│  Reshape → (B, 60, 1)        │    ← d_model = 1
│  Dropout(0.1)                │
└───────────┬─────────────────┘
            │ (B, 60, 1)
            ▼
┌─────────────────────────────┐
│  Block 2: Modified Transformer │
│  ★ Pre-Norm (非 Post-Norm) ★   │
│  ★ 无位置编码 ★                 │
│  ★ 无 Input Embedding ★        │
│  ★ 仅单层 Encoder ★            │
│                                │
│  LayerNorm(1) → Custom MHA     │
│  (5 heads, head_dim=120)       │
│  → Dropout → Residual          │
│                                │
│  LayerNorm(1) → FFN(1→5→1)    │
│  → Residual                    │
│                                │
│  ★ ReZero 初始化 ★             │    ← wo, ffn 输出层零初始化
└───────────┬─────────────────┘
            │ (B, 60, 1)
            ▼
┌─────────────────────────────┐
│  Block 3: MLP                │
│  GAP: mean(dim=-1) → (B,60) │
│  Dropout(0.1)                │
│  Dense(60→30, ReLU)          │
│  Dense(30→1, linear)         │
└───────────┬─────────────────┘
            │ (B, 1)
            ▼
         输出预测值
```

**关键特点**：
- 输入仅为**收盘价单特征**（d_model=1），与①②的 17 维特征截然不同
- LSTM 输出经 reshape 形成 (B, 60, 1)，即"60 个时间步 × 1 维"的序列新生成
- Modified Transformer 与标准 Transformer 有 5 处核心区别（详见下表）
- 自定义 MHA 解决 d_model=1 无法被 num_heads=5 整除的问题（投射到 5×120=600 维）
- ReZero 初始化使 mTransformer 初始为恒等映射，避免 LayerNorm(1) 退化

---

## 2. Transformer 组件对比

这是三个模型最核心的结构差异所在：

| 特性                   | ① Serial / ② Parallel                       | ③ LSTM-mTrans-MLP (论文)                       |
| ---------------------- | ------------------------------------------- | ---------------------------------------------- |
| **Normalization 位置** | Post-Norm（先 attention/FFN，后 LayerNorm） | **Pre-Norm**（先 LayerNorm，后 attention/FFN） |
| **位置编码**           | 正弦位置编码 (Sinusoidal PE)                | **无位置编码**                                 |
| **Input Embedding**    | Linear 投影 (input_dim→d_model)             | **无**（LSTM reshape 直接输入）                |
| **Encoder 层数**       | 2 层                                        | **1 层**                                       |
| **d_model**            | 64                                          | **1**                                          |
| **num_heads**          | 4                                           | **5**                                          |
| **head_dim**           | 16 (= 64/4)                                 | **120**（自定义 MHA，投射到 600 维）           |
| **FFN 中间维度**       | 256                                         | **5**                                          |
| **MHA 实现**           | `nn.MultiheadAttention` (PyTorch 内置)      | **自定义** `_CustomMultiHeadAttention`         |
| **Decoder**            | 无（取 last step 或 GAP）                   | **无**（被 MLP Block 替代）                    |
| **ReZero 初始化**      | 无                                          | **有**（输出投影零初始化）                     |

**为什么论文不需要位置编码？**  
LSTM 的隐藏状态本身已编码了时序顺序信息。经 reshape 后的 (B, 60, 1) 序列中，每个位置的值已经蕴含了 LSTM 对该时间步之前所有信息的累积编码。再加正弦位置编码反而是冗余的。

**为什么 d_model=1 却使用 5 个注意力头？**  
论文的设计意图是让注意力机制在**高维投射空间**中工作（5×120=600 维），而非在原始 1 维空间中。每个头有 120 维的查询/键空间，提供了充分的表达能力。这与标准 Transformer 中 d_model 必须被 num_heads 整除的常规做法完全不同。

---

## 3. 数据流对比

### LSTM 与 Transformer 的连接方式

|                           | ① Serial                        | ② Parallel                       | ③ LSTM-mTrans-MLP                  |
| ------------------------- | ------------------------------- | -------------------------------- | ---------------------------------- |
| **Transformer 输入来源**  | LSTM 全部时间步输出 (B,30,64)   | 原始输入经 Linear 投影 (B,30,64) | LSTM last step reshape (B,60,1)    |
| **LSTM→Transformer 中介** | LayerNorm → PosEnc              | 无直接连接（并行）               | 无中介（reshape 直连）             |
| **Transformer 看到什么**  | LSTM 二阶特征                   | 原始一阶特征                     | LSTM 隐藏状态的**逐维**序列        |
| **维度含义**              | 时间步=30, 特征=64(LSTM hidden) | 时间步=30, 特征=64(投影)         | 时间步=60(LSTM hidden dim), 特征=1 |

### 特别说明：③ 的 reshape 操作

这是论文最独特的设计。LSTM 输出 (B, 60) 被 reshape 为 (B, 60, 1)。此时**序列的每一步对应 LSTM hidden state 的一个维度**，而非原始时间步。mTransformer 的注意力机制在这 60 个维度之间建立全局依赖关系，本质上是在做**特征维度间的注意力**而非时间维度间的注意力。

---

## 4. 超参数对比

### 4.1 网络结构参数

| 参数                  | ① Serial (V4)        | ② Parallel (V5)             | ③ LSTM-mTrans-MLP |
| --------------------- | -------------------- | --------------------------- | ----------------- |
| seq_len               | 30                   | 30                          | **60**            |
| input_dim             | 17 (技术指标)        | 17 (技术指标)               | **1** (仅收盘价)  |
| LSTM hidden_dim       | 64                   | 64                          | **60**            |
| LSTM layers           | 2                    | 2                           | 2                 |
| LSTM dropout          | 0.2                  | 0.2                         | **0.1**           |
| d_model (Transformer) | 64                   | 64                          | **1**             |
| num_heads             | 4                    | 4                           | **5**             |
| head_dim              | 16                   | 16                          | **120**           |
| Transformer 层数      | 2                    | 2                           | **1**             |
| FFN 中间维度          | 256                  | 256                         | **5**             |
| Transformer dropout   | 0.2                  | 0.2                         | **0.15**          |
| 融合层                | Bottleneck(128→64→1) | Gate(128→128)+Pred(64→32→1) | **MLP(60→30→1)**  |
| **总参数**            | **162,945**          | **174,529**                 | **50,482**        |

### 4.2 训练策略

| 参数          | ① Serial (V4)                       | ② Parallel (V5)      | ③ LSTM-mTrans-MLP                |
| ------------- | ----------------------------------- | -------------------- | -------------------------------- |
| batch_size    | 32                                  | 32                   | **32** (论文原文=1)              |
| Loss          | **HuberLoss(δ=1.0)**                | **HuberLoss(δ=1.0)** | **MSELoss**                      |
| LR (融合模型) | **5e-4**                            | **5e-4**             | **1e-3**                         |
| LR (基线)     | 1e-3                                | 1e-3                 | 1e-3                             |
| LR Scheduler  | CosineAnnealing / ReduceLROnPlateau | 同左                 | **无**                           |
| weight_decay  | **1e-3** (融合)                     | **1e-3** (融合)      | **0**                            |
| Gradient Clip | max_norm=1.0                        | max_norm=1.0         | max_norm=1.0                     |
| Epochs        | 50                                  | 50                   | **30** (基线) / **100** (mTrans) |
| Patience      | 10                                  | 10                   | 10 / **20** (mTrans)             |
| Data Split    | 80/2/18                             | 80/2/18              | **72/10/18**                     |
| 输入数据      | csi300_features.csv (17特征)        | 同左                 | **csi300_raw.csv (仅close)**     |

---

## 5. 融合机制对比

这是三个模型的核心设计差异：

### ① Serial：Gated Fusion Bottleneck（特征级融合）

```python
# V3: 跳跃连接保留 LSTM 短期特征
lstm_last = lstm_normalized[:, -1, :]     # (B, 64)
trans_last = transformer_out[:, -1, :]    # (B, 64)
fused = torch.cat([lstm_last, trans_last], dim=-1)  # (B, 128)
# V4: 非线性瓶颈过滤
output = Linear(128→64) → ReLU → Dropout → Linear(64→1)
```

**特点**：两个分支的特征拼接后通过瓶颈网络压缩，ReLU 实现选择性门控。但权重是固定的（训练后不再变化），无法对不同样本动态调整两个分支的贡献。

### ② Parallel：Sigmoid Dynamic Gating（维度级融合）

```python
concat_feats = torch.cat([feat_lstm, feat_trans], dim=-1)  # (B, 128)
gate_weights = Sigmoid(Linear(128→128))(concat_feats)      # (B, 128)
gate_lstm, gate_trans = gate_weights.chunk(2, dim=-1)       # 各 (B, 64)
fused = gate_lstm * feat_lstm + gate_trans * feat_trans      # (B, 64)
output = Linear(64→32) → ReLU → Dropout → Linear(32→1)
```

**特点**：Sigmoid 门控是**输入依赖的**——不同样本产生不同的门控权重，实现对 LSTM 和 Transformer 特征的**动态逐维度加权**。Sigmoid（而非 Softmax）允许某些维度同时保留或同时抑制两塔的特征。

### ③ LSTM-mTrans-MLP：Serial Pipeline（无显式融合）

```python
lstm_out = LSTM(x)[:, -1, :]         # (B, 60)
lstm_reshaped = Dropout(lstm_out).unsqueeze(-1)  # (B, 60, 1)
trans_out = mTransformer(lstm_reshaped)           # (B, 60, 1)
gap = trans_out.mean(dim=-1)                      # (B, 60)
output = Dropout → Dense(60→30, ReLU) → Dense(30→1)
```

**特点**：没有显式的融合机制。LSTM 和 mTransformer 完全串行，mTransformer 直接在 LSTM 特征上做注意力增强。最终 MLP 替代了传统 Transformer Decoder 的角色。这是最简洁的设计——50,482 参数仅为①②的约 1/3。

---

## 6. 设计哲学对比

| 维度                 | ① Serial (V4)                                           | ② Parallel (V5)                  | ③ LSTM-mTrans-MLP                        |
| -------------------- | ------------------------------------------------------- | -------------------------------- | ---------------------------------------- |
| **信息流**           | LSTM 特征 → Transformer 深加工                          | 两塔独立提取 → 智能融合          | LSTM 特征 → Transformer 增强 → MLP 输出  |
| **Transformer 角色** | 对 LSTM 特征做全局注意力建模                            | 对原始输入做独立全局建模         | 对 LSTM 隐藏维度做特征间注意力           |
| **核心假设**         | Transformer 可以增强 LSTM 特征的长程依赖                | 两个编码器捕获互补特征，门控选优 | LSTM 隐藏维度间存在可学习的注意力模式    |
| **问题与解法**       | LSTM→Transformer 方差偏移 → LayerNorm + 梯度裁剪 + 低LR | 无串行耦合问题，但参数量最大     | d_model=1 LayerNorm 退化 → ReZero 初始化 |
| **可解释性**         | Attention heatmap                                       | Attention + 门控权重分布         | Attention heatmap                        |
| **参数效率**         | 162,945 (中)                                            | 174,529 (最大)                   | **50,482** (最小，仅 ~30%)               |
| **输入复杂度**       | 17 维技术指标                                           | 17 维技术指标                    | **1 维收盘价**                           |

---

## 7. 实验结果对比

### ① Serial (V4) — 基于 17 维特征，seq_len=30

| 模型                 | MSE        | DA (%)     |
| -------------------- | ---------- | ---------- |
| LSTM                 | 7,773      | 49.30%     |
| Transformer          | 77,050     | ~50%       |
| **LSTM-Transformer** | **19,510** | **50.70%** |

### ② Parallel (V5) — 基于 17 维特征，seq_len=30（10 轮基准测试）

| 模型                    | MSE             | RMSE      | MAPE          | DA (%)         |
| ----------------------- | --------------- | --------- | ------------- | -------------- |
| LSTM                    | 10,578±4,112    | 101±19    | 2.03±0.48     | 49.45±0.54     |
| Transformer             | 21,416±6,156    | 145±21    | 2.99±0.67     | 49.59±0.26     |
| LSTM-Transformer (串行) | 9,164±5,876     | 92±27     | 1.81±0.71     | 50.42±0.31     |
| **Parallel-LSTM-Trans** | **7,433±2,117** | **85±12** | **1.60±0.27** | **50.42±0.34** |

### ③ LSTM-mTrans-MLP (V5-beta) — 仅收盘价，seq_len=60

| 模型                | R²        | MAE      | RMSE      |
| ------------------- | --------- | -------- | --------- |
| LSTM                | 0.951     | 126.0    | 175.2     |
| Transformer         | 0.492     | 379.1    | 567.2     |
| **LSTM-mTrans-MLP** | **0.984** | **77.3** | **101.3** |

> **注意**：①② 与 ③ 的结果**不可直接比较**，因为输入数据（17 维 vs 1 维）、序列长度（30 vs 60）、评估指标（DA/MSE vs R²/MAE）均不同。

---

## 8. 各架构的优劣分析

### ① Serial LSTM-Transformer

| 优势                                | 劣势                                                    |
| ----------------------------------- | ------------------------------------------------------- |
| 跳跃连接保留 LSTM 短期特征          | LSTM→Transformer 串行耦合导致方差偏移，需额外 LayerNorm |
| 经 4 轮迭代验证了多项训练稳定性技术 | Transformer 过度平滑 LSTM 特征（V2 中 MSE 反超基线）    |
| 瓶颈融合层有信息压缩效果            | 融合权重固定，无法逐样本动态调整                        |
|                                     | 参数量偏大（162k），17 维输入 + 复杂网络但 DA 仅 50.70% |

### ② Parallel LSTM-Transformer

| 优势                                         | 劣势                                      |
| -------------------------------------------- | ----------------------------------------- |
| 并行架构消除串行耦合问题                     | 参数量最大（174k），两套完整的编码器      |
| Sigmoid 动态门控可逐样本、逐维度调整两塔权重 | 右塔 Transformer 在小数据集上仍不稳定     |
| 门控权重可暴露用于白盒可解释性               | MSE 方差较大（7,433±2,117），稳定性不如 ③ |
| V5 基准测试在 MSE 和 DA 上全面最优           | 依赖 17 维手工特征工程                    |

### ③ LSTM-mTrans-MLP（论文复现）

| 优势                                   | 劣势                                                                           |
| -------------------------------------- | ------------------------------------------------------------------------------ |
| 仅需收盘价单特征，无需特征工程         | d_model=1 导致 LayerNorm 退化，必须 ReZero                                     |
| 参数量最少（50k），参数效率极高        | 论文原始 batch_size=1 在长周期数据上无法收敛                                   |
| R²=0.984，模型拟合能力极强             | Modified Transformer 与标准 Transformer 差异大，可解释性依赖 attention heatmap |
| Pre-Norm + 无位置编码 = 更简洁的信息流 | 单层 Encoder 的表达能力上限可能低于多层                                        |
| 论文已发表，有学术权威性背书           | mTransformer 注意力作用于特征维度（非时间维度），物理含义不直观                |

---

## 9. 关键设计差异总结

### 9.1 LSTM 输出的处理方式（最核心差异）

- **① Serial**：LSTM 全程序列输出 (B,30,64) → Transformer 在**时间维度**做注意力
- **② Parallel**：LSTM 仅取 last step (B,64) → 不送入 Transformer
- **③ 论文**：LSTM 取 last step (B,60) → reshape (B,60,1) → mTransformer 在**特征维度**做注意力

### 9.2 Transformer 的 Normalization

- **① ② Post-Norm**：`x = LayerNorm(x + Attention(x))`
- **③ Pre-Norm**：`x = x + Attention(LayerNorm(x))`

Pre-Norm 的梯度流更稳定（残差连接的梯度不经过 LayerNorm），这是论文选择 Pre-Norm 的理论依据。

### 9.3 为什么 ③ 不需要 ①② 的那些训练稳定性技巧？

| ①② 需要的技巧                   | ③ 为什么不需要                            |
| ------------------------------- | ----------------------------------------- |
| LSTM→Transformer 间的 LayerNorm | d_model=1，且 Pre-Norm 已包含 LayerNorm   |
| 低学习率 5e-4                   | 参数量仅 50k，优化面更简单                |
| CosineAnnealingLR 调度器        | 标准 Adam(lr=1e-3) 即可收敛               |
| weight_decay=1e-3 (L2 正则化)   | 模型够小，过拟合风险低                    |
| HuberLoss (鲁棒损失)            | MSELoss 即可（输入仅 1 维，异常值影响小） |

但 ③ 有自己独特的问题——LayerNorm(d_model=1) 退化，需要 ReZero 初始化来解决。

---

## 10. 结论

三个模型代表了三种不同的 LSTM-Transformer 融合设计哲学：

1. **① Serial**（V1-V4）是**工程驱动**的迭代：从论文出发，遇到训练崩溃，逐层诊断（方差偏移→梯度爆炸→优化震荡），逐层修复（LayerNorm→GradClip→低LR→特征融合→门控瓶颈）。价值在于**系统化的调试方法论**。

2. **② Parallel**（V5）是**架构创新**的尝试：消除串行耦合，引入动态门控机制，提供可解释性。价值在于**并行双塔 + Sigmoid 门控的原创设计**。

3. **③ LSTM-mTrans-MLP**（V5-beta）是**学术复现**：严格按论文架构实现，用最少的参数（50k）和最简单的输入（仅收盘价）达到最高的拟合精度（R²=0.984）。价值在于**简洁高效的论文设计**和 ReZero 适配技巧。

这三条路线并非互相替代，而是从不同角度验证了 LSTM-Transformer 融合在金融时序预测中的有效性。
