# pip install torch scikit-learn
"""时序 Dataset / DataLoader 工厂模块 (三分类版)。

核心约束:
    - 切分严格按时间顺序 (train → val → test)，禁止 shuffle test/val。
    - Scaler 仅在 train 上 fit，防止未来数据泄漏。
    - 标签: 基于原始 close 涨跌幅分为 Drop(0) / Flat(1) / Rise(2) 三类。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


# ── 标签生成 ──────────────────────────────────────────────────────
def _compute_labels(raw_close: NDArray, threshold: float = 0.005) -> NDArray[np.int64]:
    """根据收盘价日收益率划分三分类标签。

    返回 (T-1,) int64 数组: 0=跌, 1=平, 2=涨。
    """
    returns = np.diff(raw_close) / (np.abs(raw_close[:-1]) + 1e-8)
    labels = np.ones(len(returns), dtype=np.int64)  # 默认 Flat
    labels[returns < -threshold] = 0   # Drop
    labels[returns > threshold] = 2    # Rise
    return labels


# ── Dataset ───────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    """滑动窗口时序分类 Dataset。

    X: (N, seq_len, num_features)
    y: (N,) — 三分类标签 {0, 1, 2}
    """

    def __init__(self, data: NDArray[np.float32], labels: NDArray[np.int64], seq_len: int) -> None:
        # 窗口 i 使用 data[i : i+seq_len]，标签 = labels[i+seq_len-1]
        # labels[j] = close[j] → close[j+1] 的方向，长度 = len(data)-1
        total = len(data) - seq_len
        self.X = torch.as_tensor(
            np.array([data[i : i + seq_len] for i in range(total)]),
            dtype=torch.float32,
        )
        self.y = torch.as_tensor(
            labels[seq_len - 1 : seq_len - 1 + total],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── 核心工厂函数 ──────────────────────────────────────────────────
def get_dataloaders(
    df_values: NDArray[np.float64],
    columns: list[str],
    seq_len: int = 30,
    batch_size: int = 64,
    target_col: str = "close",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    flat_threshold: float = 0.005,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """一站式构建 train/val/test DataLoader (三分类)。

    Returns:
        (train_loader, val_loader, test_loader)
    """
    target_idx = columns.index(target_col)
    logger.info("目标列: {} (idx={})", target_col, target_idx)

    n = len(df_values)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    train_raw, val_raw, test_raw = df_values[:t1], df_values[t1:t2], df_values[t2:]
    logger.info("切分: train={} val={} test={}", len(train_raw), len(val_raw), len(test_raw))

    # 特征归一化 (fit on train only)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
    val_scaled = scaler.transform(val_raw).astype(np.float32)
    test_scaled = scaler.transform(test_raw).astype(np.float32)

    # 各 split 内独立计算标签 (避免跨 split 泄漏)
    labels_train = _compute_labels(train_raw[:, target_idx], flat_threshold)
    labels_val = _compute_labels(val_raw[:, target_idx], flat_threshold)
    labels_test = _compute_labels(test_raw[:, target_idx], flat_threshold)

    for name, lbl in [("train", labels_train), ("val", labels_val), ("test", labels_test)]:
        counts = np.bincount(lbl, minlength=3)
        logger.info("{} 标签分布: Drop={} Flat={} Rise={}", name, *counts)

    loaders: list[DataLoader] = []
    for name, arr, lbl in [("train", train_scaled, labels_train),
                           ("val", val_scaled, labels_val),
                           ("test", test_scaled, labels_test)]:
        ds = TimeSeriesDataset(arr, lbl, seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"), drop_last=False)
        logger.info("{}: {} samples → {} batches", name, len(ds), len(loader))
        loaders.append(loader)

    return loaders[0], loaders[1], loaders[2]
