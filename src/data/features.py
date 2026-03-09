# pip install pandas_ta EMD-signal
"""技术指标特征工程 + CEEMDAN 信号分解模块。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """在 OHLCV DataFrame 上追加技术指标并清除 NaN 行。"""
    shape_before = df.shape
    logger.info("特征扩充前: {}", shape_before)

    df = df.copy()
    df.ta.sma(length=5, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)

    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()

    logger.info("特征扩充后: {} (丢弃 {} 行 NaN)", df.shape, shape_before[0] - len(df))
    return df


def apply_ceemdan(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """对目标列施加 CEEMDAN 分解，将各 IMF 分量及残差作为新特征列追加。

    CEEMDAN 将原始非平稳金融序列分解为多个本征模态函数 (IMF):
      - 高频 IMF  → 短期市场噪声
      - 低频 IMF + 残差 → 中长期趋势信号
    为模型提供多尺度时频信息，显著降低输入信噪比。
    """
    from PyEMD import CEEMDAN as _CEEMDAN

    df = df.copy()
    signal = df[column].values.astype(np.float64)

    decomposer = _CEEMDAN(trials=50, epsilon=0.005, ext_EMD=None)
    imfs = decomposer(signal)  # shape: (n_components, T)

    for i in range(len(imfs) - 1):
        df[f"imf_{i + 1}"] = imfs[i]
    df["imf_residue"] = imfs[-1]

    logger.info("CEEMDAN 分解: {} 个 IMF + 残差 → 新增 {} 列", len(imfs) - 1, len(imfs))
    return df
