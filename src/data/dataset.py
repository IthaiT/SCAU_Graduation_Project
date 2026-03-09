"""Time-series Dataset / DataLoader factory (binary classification).

Design invariants:
    - Chronological split (train → val → test), NO shuffle on val/test.
    - Rolling Z-Score standardization: strictly causal, no global scaler.
    - Binary label: 5-day forward cumulative return > 0 → 1, else 0.
    - num_classes = 2.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from src.data.features import rolling_zscore


# ── Label generation ──────────────────────────────────────────────
def _compute_labels(close: NDArray, horizon: int = 5) -> NDArray[np.int64]:
    """Binary labels from forward-looking cumulative return.

    ret_t = close[t + horizon] / close[t] - 1
    label = 1 if ret_t > 0 else 0

    Returns (T - horizon,) int64 array aligned to indices [0, T-horizon).
    """
    future = close[horizon:]
    current = close[:len(close) - horizon]
    returns = (future - current) / (np.abs(current) + 1e-9)
    labels = (returns > 0).astype(np.int64)
    return labels


# ── Dataset ───────────────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    """Sliding-window time-series binary classification Dataset.

    X: (N, seq_len, num_features)
    y: (N,) — binary {0, 1}
    """

    def __init__(self, features: NDArray[np.float32], labels: NDArray[np.int64], seq_len: int) -> None:
        n_samples = len(labels) - seq_len + 1
        if n_samples <= 0:
            raise ValueError(f"Not enough data: {len(labels)} rows for seq_len={seq_len}")

        self.X = torch.as_tensor(
            np.array([features[i: i + seq_len] for i in range(n_samples)]),
            dtype=torch.float32,
        )
        self.y = torch.as_tensor(
            labels[seq_len - 1: seq_len - 1 + n_samples],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ── Core factory ──────────────────────────────────────────────────
def get_dataloaders(
    df: pd.DataFrame,
    seq_len: int = 30,
    batch_size: int = 64,
    target_col: str = "close",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    horizon: int = 5,
    zscore_window: int = 60,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders with zero data leakage.

    Pipeline:
        1. Compute binary labels from raw close prices (forward-looking target).
        2. Apply causal rolling z-score to features (backward-looking only).
        3. Align features & labels, drop NaN rows.
        4. Chronological split.
        5. Build sliding-window datasets.
    """
    df = df.copy()

    # 1. Binary labels from raw close
    raw_close = df[target_col].values
    labels_full = _compute_labels(raw_close, horizon=horizon)

    # 2. Causal rolling z-score on all features
    df_z = rolling_zscore(df, window=zscore_window)

    # 3. Align: labels are for indices [0, len-horizon), z-scored features lost first rows
    # Find the overlapping valid range
    z_start_idx = len(df) - len(df_z)  # how many rows dropped from start by zscore
    label_end_idx = len(df) - horizon  # labels valid up to this original index

    # Valid range in original DataFrame coordinates
    valid_start = z_start_idx
    valid_end = label_end_idx
    if valid_start >= valid_end:
        raise ValueError(f"No valid data after alignment: zscore drops {z_start_idx}, horizon={horizon}")

    # Slice aligned features and labels
    features = df_z.iloc[:valid_end - valid_start].values.astype(np.float32)
    labels = labels_full[valid_start:valid_end].astype(np.int64)

    assert len(features) == len(labels), f"Alignment error: {len(features)} != {len(labels)}"

    n = len(features)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (features[:t1], labels[:t1]),
        "val": (features[t1:t2], labels[t1:t2]),
        "test": (features[t2:], labels[t2:]),
    }

    for name, (feat, lbl) in splits.items():
        counts = np.bincount(lbl, minlength=2)
        logger.info("{} → {} samples | Drop={} Rise={}", name, len(lbl), counts[0], counts[1])

    loaders: list[DataLoader] = []
    for name in ("train", "val", "test"):
        feat, lbl = splits[name]
        ds = TimeSeriesDataset(feat, lbl, seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"), drop_last=False)
        logger.info("{}: {} sequences → {} batches", name, len(ds), len(loader))
        loaders.append(loader)

    return loaders[0], loaders[1], loaders[2]
