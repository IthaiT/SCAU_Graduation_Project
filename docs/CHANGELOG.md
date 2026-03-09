# 变更日志

## 2026-03-09 回归任务重构为三分类任务

### 变更范围
- 任务目标从价格回归切换为趋势分类：`0=Drop`、`1=Flat`、`2=Rise`。
- 保留四个模型对比：`LSTM`、`Transformer`、`LSTM_Transformer`、`Parallel_LSTM_Transformer`。
- 评估主入口统一为 `scripts/evaluator.py`。

### 模型结构变更

#### `src/models/networks.py`
- 所有模型输出头由 `pred_len` 改为 `num_classes=3`。
- 所有模型前向输出统一为分类 logits，形状为 `(batch, 3)`。
- 保留可解释性输出：
  - Transformer 注意力权重。
  - 并行混合模型门控权重 `gate_lstm` / `gate_trans`。
- 在注释中明确输入可来自 CEEMDAN 分解后的离线特征。

### 训练流程变更

#### `scripts/train.py`
- 损失函数从 `HuberLoss` 切换为 `nn.CrossEntropyLoss()`。
- 删除回归相关参数语义，新增 `NUM_CLASSES = 3`。
- 四个模型实例化统一使用分类头。
- 保留并强调 CEEMDAN 预处理叙述（作为上游离线流程）。

#### `src/engine/trainer.py`
- `_run_epoch` 返回从单一 loss 改为 `(loss, accuracy)`。
- 标签在训练/验证阶段统一转换为 `LongTensor` 并展平：`y.long().view(-1)`。
- `history` 扩展为：
  - `train_loss`、`val_loss`
  - `train_acc`、`val_acc`
- 训练日志改为输出 loss + accuracy + learning rate。

### 评估流程变更

#### `scripts/evaluator.py`
- 删除全部回归指标与图表：
  - `MSE`、`RMSE`、`MAE`、`MAPE`、`R2`、`DA`
  - 真实值-预测值折线图、误差分布图
- 新增分类指标：
  - `accuracy_score`
  - `f1_score`（`macro` 与 `weighted`）
  - `classification_report`
- 推理阶段基于 `argmax(logits)` 生成预测类别。
- 输出图表调整为：
  - `confusion_matrices.png`
  - `acc_f1_comparison.png`
  - `loss_acc_curves.png`
  - `attention_heatmap.png`
  - `gate_weights.png`
- 文本报告输出到：`results/classification_reports.txt`。
- 权重加载兼容不同 PyTorch 版本（`weights_only` 回退处理）。

### 基准测试流程变更

#### `scripts/benchmark.py`
- 训练目标切换为分类：`CrossEntropyLoss`。
- 评估逻辑切换为分类：`argmax` + `Accuracy/F1`。
- 输出表格与统计项由回归指标改为：
  - `Accuracy`
  - `F1_macro`
  - `F1_weighted`
- 删除回归遗留逻辑：
  - `scaler_target` 反归一化
  - 回归指标计算与相关日志字段

### 清理与兼容
- 删除冗余入口 `scripts/evaluate.py`，避免双评估脚本并存导致歧义。
- 评估入口以 `scripts/evaluator.py` 为准。

### 兼容性说明
- 旧版回归权重与当前分类头不兼容，必须重新训练。
- 训练完成后，`models/*_history.json` 需包含 `train_acc` 与 `val_acc` 字段，评估绘图才能完整生成。

## 2026-03-09 数据链路修正（补齐分类标签处理）

### 问题
- 原数据管道仍输出连续 `close` 数值，训练与评估虽然能运行，但并非真实三分类任务。
- `src.data.__init__` 强依赖 `pandas_ta/numba`，导致仅使用 `dataset` 也可能因环境版本冲突而崩溃。

### 修复

#### `src/data/dataset.py`
- 重构标签构造逻辑：
  - 以窗口末端价格和未来 `pred_len` 步价格计算收益率。
  - 按阈值离散为三类：`0=Drop`、`1=Flat`、`2=Rise`。
- `TimeSeriesDataset` 输出改为：
  - `X: (N, seq_len, F)`
  - `y: (N,)` 且 `dtype=torch.long`
- `get_dataloaders` 返回值改为三元组：
  - `(train_loader, val_loader, test_loader)`
- 新增每个 split 的类别分布日志，便于检查类别塌缩与数据偏斜。

#### 调用侧同步
- `scripts/train.py`、`scripts/evaluator.py`、`scripts/benchmark.py` 已同步新接口。
- `tests/test_dataloader.py` 改为校验分类标签形状与类型。
- `tests/test_models.py` 改为校验分类输出 `(B, num_classes)`，并覆盖并行混合模型。

#### 依赖隔离
- `src/data/__init__.py` 删除 `build_features` 的顶层导入，避免无关脚本触发 `pandas_ta/numba` 依赖错误。
- `scripts/download_data.py` 改为直接从 `src.data.fetch` 导入，降低包初始化耦合。
