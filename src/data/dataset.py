# pip install torch scikit-learn
"""时序 Dataset / DataLoader 工厂模块。

核心约束:
    - 切分严格按时间顺序 (train → val → test)，禁止 shuffle。
    - Scaler 仅在 train 上 fit，防止未来数据泄漏。
    - close 列使用独立 Scaler，便于 inverse_transform 还原预测值。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


# ── 数据结构 ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class SplitArrays:
    """归一化后的 train/val/test numpy 数组。"""
    train: NDArray[np.float32]
    val: NDArray[np.float32]
    test: NDArray[np.float32]
    scaler_all: MinMaxScaler
    scaler_target: MinMaxScaler
    target_col_idx: int


# ── Dataset ───────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    """滑动窗口时序 Dataset。

    X: (N, seq_len, num_features)
    y: (N, pred_len)  — 仅 target 列
    """

    def __init__(
        self,
        data: NDArray[np.float32],
        seq_len: int,
        pred_len: int,
        target_col_idx: int,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_idx = target_col_idx

        # 预先切好所有窗口，避免 __getitem__ 里做任何计算
        total = len(data) - seq_len - pred_len + 1
        self.X = torch.as_tensor(
            np.array([data[i : i + seq_len] for i in range(total)]),
            dtype=torch.float32,
        )
        self.y = torch.as_tensor(
            np.array([
                data[i + seq_len : i + seq_len + pred_len, self.target_idx]
                for i in range(total)
            ]),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── 核心工厂函数 ──────────────────────────────────────────────────
def _split_and_scale(
    values: NDArray[np.float64],
    target_col_idx: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> SplitArrays:
    """按时间顺序切分 + 仅 train fit 归一化。"""
    n = len(values)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))

    train_raw, val_raw, test_raw = values[:t1], values[t1:t2], values[t2:]
    logger.info("切分: train={} val={} test={}", len(train_raw), len(val_raw), len(test_raw))

    # 全特征 Scaler — 仅 fit on train
    scaler_all = MinMaxScaler()
    train_scaled = scaler_all.fit_transform(train_raw).astype(np.float32)
    val_scaled = scaler_all.transform(val_raw).astype(np.float32)
    test_scaled = scaler_all.transform(test_raw).astype(np.float32)

    # 目标列独立 Scaler — 用于预测后 inverse_transform
    scaler_target = MinMaxScaler()
    scaler_target.fit(train_raw[:, target_col_idx : target_col_idx + 1])

    return SplitArrays(
        train=train_scaled,
        val=val_scaled,
        test=test_scaled,
        scaler_all=scaler_all,
        scaler_target=scaler_target,
        target_col_idx=target_col_idx,
    )


def get_dataloaders(
    df_values: NDArray[np.float64],
    columns: list[str],
    seq_len: int = 30,
    pred_len: int = 1,
    batch_size: int = 64,
    target_col: str = "close",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler]:
    """一站式构建 train/val/test DataLoader。

    Args:
        df_values: shape (T, F) 的原始数值矩阵 (未归一化)。
        columns: 列名列表，用于定位 target_col。
        seq_len: 输入窗口长度。
        pred_len: 预测窗口长度。
        batch_size: DataLoader batch size。
        target_col: 预测目标列名。
        train_ratio: 训练集占比。
        val_ratio: 验证集占比。

    Returns:
        (train_loader, val_loader, test_loader, scaler_target)
    """
    target_col_idx = columns.index(target_col)
    logger.info("目标列: {}  (idx={})", target_col, target_col_idx)

    split = _split_and_scale(df_values, target_col_idx, train_ratio, val_ratio)

    loaders: list[DataLoader] = []
    for name, arr in [("train", split.train), ("val", split.val), ("test", split.test)]:
        ds = TimeSeriesDataset(arr, seq_len, pred_len, split.target_col_idx)
        # train shuffle 可选，但时序预测通常不 shuffle 以保持局部时序结构
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
        logger.info("{}: {} samples → {} batches", name, len(ds), len(loader))
        loaders.append(loader)

    return loaders[0], loaders[1], loaders[2], split.scaler_target
